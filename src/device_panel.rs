//! The always-visible bottom Device Panel (Bitwig/Ableton-style docked device rack): the CLAP/
//! built-in effect-chain editor (`TrackFxUi`, `fx_chain_ui` and its slot-drawing helpers) shared
//! by the Channel Rack, Mixer, and this panel's own FX chain section; the synth engine picker and
//! preset bar; and a lane's own synth-override editor. `EffectEditorTarget`/`FxChainKind`/
//! `DeviceChainFocus` are this module's own addressing types, reused by the Channel Rack/Mixer/
//! Beats window wherever they construct a `TrackFxUi` or set which track/lane the panel focuses.

use std::path::Path;

use crate::builtin_fx::BuiltInEffect;
use crate::factory_presets::factory_presets;
use crate::file_ops::{browse_for_file, load_effect};
use crate::model::{
    EqBandType, FilterMode, Lane, ProjectPlugin, RegionContent, SessionClipContent, Song, SynthEngine, SynthParams,
    SynthPreset, Track, TrackEffectConfig, TrackKind,
};
use crate::plugin_host::{
    self, DawHost, EffectInstance, LoadedEffect, PluginGuiHandle, PluginParamInfo, TrackEffectSlots,
};
use crate::synth_simple_panel::synth_params_ui;
use crate::synth_trine_panel::trine_params_ui;
use crate::synth_wave_panel::wave_params_ui;
use clack_host::prelude::PluginInstance;
use raw_window_handle::RawWindowHandle;

/// Which effect's parameter-editor window (if any) is currently open. There's only ever one such
/// window at a time, shared by the master bus, every track, and every send bus. `Master(slot_index)`/
/// `Track(track_index, slot_index)`/`Send(send_index, slot_index)` identify one slot within the
/// master chain/that track's chain/that send bus's chain respectively.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum EffectEditorTarget {
    Master(usize),
    Track(usize, usize),
    Send(usize, usize),
    Submix(usize, usize),
}

/// Which chain `TrackFxUi` is editing — only changes which `EffectEditorTarget` variant its
/// "Params" button opens (see `fx_chain_ui`), so each location's editor state doesn't collide with
/// the others'. The chain's own index within `slots` (a real track's index, or a send bus's index;
/// meaningless — always 0 — for `Master`) still comes from `TrackFxUi::track_index`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum FxChainKind {
    Track,
    Master,
    Send,
    Submix,
}

/// Which track's or step-grid lane's instrument + effect chain the always-visible bottom Device
/// Panel (see `device_panel_contents_ui`) is currently showing — the Bitwig/Ableton-style docked
/// device rack, replacing the old per-track/per-lane synth-settings windows this app used to open
/// on a "🎹" click. `Lane`/`SessionSlotLane` are one level deeper than `Track` since a lane's
/// synth belongs to a specific region's (or session slot's) own lane rather than the whole track
/// (see `Lane::synth_override`) — the two are separate variants, not `RegionEditTarget`-addressed,
/// since a lane's device focus can outlive whichever editor window opened it.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeviceChainFocus {
    Track(usize),
    Lane { track_index: usize, region_index: usize, lane_index: usize },
    SessionSlotLane { track_index: usize, slot_index: usize, lane_index: usize },
}

/// What kind of row `track_ui` should draw for one FX chain slot — determined by peeking at the
/// live `TrackEffectSlots` entry rather than tracking a parallel "kind" array, since that entry is
/// already the source of truth for what's actually running. `Clap` covers both an already-loaded
/// CLAP plugin and a slot still awaiting its path/Load click, since both need the same path-field
/// UI; `BuiltIn` carries the effect's display label (e.g. "Delay") for slots that are always live
/// immediately once added.
#[derive(Clone, Copy, PartialEq, Eq)]
enum FxSlotKind {
    Clap,
    BuiltIn(&'static str),
}

/// Per-track (or master-bus) CLAP/built-in effect-chain UI state, bundled to keep `track_ui`'s
/// parameter list manageable. `paths`/`instances`/`messages` are this one chain, indexed the same
/// as `Track::effects`/`Song::master_effects` (slot 0 first, feeding into slot 1, and so on) — one
/// entry per effect slot, whether or not that slot has successfully loaded a plugin yet.
pub(crate) struct TrackFxUi<'a> {
    /// Row into `slots` this chain lives at — always 0 when `chain_kind` is `Master` (see
    /// `plugin_host::MasterEffectSlots`'s doc comment on why the master chain still uses the
    /// per-track `TrackEffectSlots` shape, just pinned to one row).
    pub(crate) track_index: usize,
    /// Which location this chain belongs to — see `FxChainKind`.
    pub(crate) chain_kind: FxChainKind,
    pub(crate) paths: &'a mut Vec<String>,
    pub(crate) messages: &'a mut Vec<Option<(bool, String)>>,
    pub(crate) slots: TrackEffectSlots,
    pub(crate) instances: &'a mut Vec<Option<PluginInstance<DawHost>>>,
    pub(crate) guis: &'a mut Vec<Option<PluginGuiHandle>>,
    /// (sample_rate, min_frames, max_frames) from the running audio engine, or `None` if it
    /// failed to start — plugins can't be activated without a device to size buffers for.
    pub(crate) engine_config: Option<(f64, u32, u32)>,
    /// The project's imported CLAP plugin library (`Song::plugins`), offered by mnemonic name —
    /// both as a picker next to each CLAP slot's path field, and as one-click entries in
    /// "+ Add Effect" — so paths don't need retyping.
    pub(crate) known_plugins: &'a [ProjectPlugin],
    /// Every track's name, in `Song::tracks` order — offered as sidechain key-source choices for
    /// a CLAP/Compressor/NoiseGate slot's sidechain picker (see `fx_chain_ui`), regardless of
    /// which chain (a track's, a send's, a submix's, or the master's) is being edited.
    pub(crate) track_names: &'a [String],
    pub(crate) editor: &'a mut Option<EffectEditorTarget>,
    /// Set by channel_rack_row_ui's "🎹" button to focus the bottom Device Panel on that track.
    /// Unused (and meaningless) for the master bus, which has no synth.
    pub(crate) device_chain_focus: &'a mut Option<DeviceChainFocus>,
    /// Set by channel_rack_row_ui's "🗑" button; applied by the caller after the track loop ends
    /// (can't remove from `song.tracks` mid-iteration since it's borrowed via `iter_mut`). Unused
    /// for the master bus, which can't be deleted.
    pub(crate) remove_requested: &'a mut Option<usize>,
    /// Whether `fx_chain_ui` should draw a built-in effect's own knobs right under its slot row
    /// (only set by the bottom Device Panel — see `device_panel_track_fx_chain_ui`) rather than
    /// requiring the separate "FX Params" window every other caller of `fx_chain_ui` still uses.
    /// A CLAP plugin's params still only open in that window even here — inlining those would need
    /// the embedded/floating-GUI machinery `effect_editor`'s window already carries (window handle,
    /// `active_embedded_gui`), which the Device Panel doesn't have access to.
    pub(crate) inline_params: bool,
}

/// This chain's `fx.editor` target for `slot_index` — `Master`/`Track`/`Send` per `fx.chain_kind`,
/// so none of the three ever collide even though a send bus's own `TrackEffectSlots` row index can
/// coincide with a real track's (and the master chain's row index is always 0, same as a real
/// track 0 would use).
fn fx_editor_target(fx: &TrackFxUi, slot_index: usize) -> EffectEditorTarget {
    match fx.chain_kind {
        FxChainKind::Master => EffectEditorTarget::Master(slot_index),
        FxChainKind::Track => EffectEditorTarget::Track(fx.track_index, slot_index),
        FxChainKind::Send => EffectEditorTarget::Send(fx.track_index, slot_index),
        FxChainKind::Submix => EffectEditorTarget::Submix(fx.track_index, slot_index),
    }
}

/// Hover-text label for a `Track::pan` value: "C" at dead center, otherwise a percentage toward
/// hard left/right (e.g. "35% L").
pub(crate) fn pan_label(pan: f32) -> String {
    if pan.abs() < 0.01 {
        "C".to_string()
    } else if pan < 0.0 {
        format!("{:.0}% L", -pan * 100.0)
    } else {
        format!("{:.0}% R", pan * 100.0)
    }
}

/// Bundles the bottom Device Panel's mutable app-state borrows (see `device_panel_contents_ui`),
/// the panel counterpart of `ChannelRackUi`/`MixerUi` — a plain struct rather than a dozen
/// positional parameters, for the same reason those have one.
pub(crate) struct DevicePanelUi<'a> {
    pub(crate) detached: &'a mut bool,
    pub(crate) focus: &'a mut Option<DeviceChainFocus>,
    pub(crate) track_effect_slots: &'a TrackEffectSlots,
    pub(crate) track_effect_instances: &'a mut Vec<Vec<Option<PluginInstance<DawHost>>>>,
    pub(crate) track_effect_guis: &'a mut Vec<Vec<Option<PluginGuiHandle>>>,
    pub(crate) track_effect_paths: &'a mut Vec<Vec<String>>,
    pub(crate) track_effect_messages: &'a mut Vec<Vec<Option<(bool, String)>>>,
    pub(crate) effect_editor: &'a mut Option<EffectEditorTarget>,
    pub(crate) new_preset_name: &'a mut String,
    pub(crate) preset_message: &'a mut Option<(bool, String)>,
}

/// The always-visible bottom Device Panel's contents — whichever track's or step-grid lane's
/// instrument + effect chain `panel.focus` (see `DeviceChainFocus`) currently points at, laid out
/// inline rather than behind a separate window/menu (Bitwig/Ableton's docked device-rack pattern).
/// Shows a placeholder until a track/lane's "🎹" button is clicked at least once.
pub(crate) fn device_panel_contents_ui(
    ui: &mut egui::Ui,
    song: &mut Song,
    engine_config: Option<(f64, u32, u32)>,
    panel: &mut DevicePanelUi,
) {
    ui.horizontal(|ui| {
        ui.heading("Device Panel");
        if ui
            .small_button(if *panel.detached { "⏷ Dock" } else { "⧉ Detach" })
            .clicked()
        {
            *panel.detached = !*panel.detached;
        }
    });
    ui.separator();

    let Some(focus) = *panel.focus else {
        ui.weak("Click a track's or step-grid lane's 🎹 button to show its instrument and effects here.");
        return;
    };
    match focus {
        DeviceChainFocus::Track(index) => {
            let Some(track) = song.tracks.get(index) else {
                *panel.focus = None;
                return;
            };
            if track.kind == TrackKind::Audio {
                ui.weak("Audio track — no instrument.");
            } else {
                egui::CollapsingHeader::new("Synth")
                    .default_open(true)
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("Engine:");
                            let track = &mut song.tracks[index];
                            if ui
                                .selectable_label(
                                    track.synth_engine == SynthEngine::Simple,
                                    "Simple Synth",
                                )
                                .clicked()
                            {
                                track.synth_engine = SynthEngine::Simple;
                            }
                            if ui
                                .selectable_label(track.synth_engine == SynthEngine::Trine, "Trine")
                                .clicked()
                            {
                                track.synth_engine = SynthEngine::Trine;
                            }
                            if ui
                                .selectable_label(track.synth_engine == SynthEngine::Wave, "Wave")
                                .clicked()
                            {
                                track.synth_engine = SynthEngine::Wave;
                            }
                        });
                        match song.tracks[index].synth_engine {
                            SynthEngine::Simple => {
                                synth_preset_bar_ui(
                                    ui,
                                    song,
                                    index,
                                    panel.new_preset_name,
                                    panel.preset_message,
                                );
                                synth_params_ui(ui, &mut song.tracks[index].synth);
                            }
                            SynthEngine::Trine => {
                                synth_preset_bar_ui(
                                    ui,
                                    song,
                                    index,
                                    panel.new_preset_name,
                                    panel.preset_message,
                                );
                                trine_params_ui(ui, &mut song.tracks[index].trine);
                            }
                            SynthEngine::Wave => {
                                synth_preset_bar_ui(
                                    ui,
                                    song,
                                    index,
                                    panel.new_preset_name,
                                    panel.preset_message,
                                );
                                wave_params_ui(ui, &mut song.tracks[index].wave);
                            }
                        }
                    });
            }
            ui.separator();
            device_panel_track_fx_chain_ui(ui, song, index, engine_config, panel);
        }
        DeviceChainFocus::Lane { track_index, region_index, lane_index } => {
            let lane = song
                .tracks
                .get_mut(track_index)
                .and_then(|t| t.regions.get_mut(region_index))
                .and_then(|r| match &mut r.content {
                    RegionContent::StepGrid(lanes) => lanes.get_mut(lane_index),
                    _ => None,
                });
            let Some(lane) = lane else {
                *panel.focus = None;
                ui.weak("Lane no longer exists.");
                return;
            };
            lane_synth_ui(ui, lane);
            ui.separator();
            // A lane has no effect chain of its own — effects are track-scoped — so the panel
            // shows the owning track's chain too, since that's still part of the sound being heard.
            device_panel_track_fx_chain_ui(ui, song, track_index, engine_config, panel);
        }
        DeviceChainFocus::SessionSlotLane { track_index, slot_index, lane_index } => {
            let lane = song
                .tracks
                .get_mut(track_index)
                .and_then(|t| t.session_clips.get_mut(slot_index))
                .and_then(|slot| slot.as_mut())
                .and_then(|clip| match &mut clip.content {
                    SessionClipContent::Region { content: RegionContent::StepGrid(lanes), .. } => {
                        lanes.get_mut(lane_index)
                    }
                    _ => None,
                });
            let Some(lane) = lane else {
                *panel.focus = None;
                ui.weak("Lane no longer exists.");
                return;
            };
            lane_synth_ui(ui, lane);
            ui.separator();
            device_panel_track_fx_chain_ui(ui, song, track_index, engine_config, panel);
        }
    }
}

/// A lane's own synth-override editor — the "Override the track synth for this lane" checkbox
/// plus, when checked, the engine picker and that engine's params. Shared by `DeviceChainFocus::
/// Lane`/`SessionSlotLane`, which differ only in how they resolve `lane` (a Playlist region's vs.
/// a session slot's own lane list).
fn lane_synth_ui(ui: &mut egui::Ui, lane: &mut Lane) {
    ui.checkbox(&mut lane.synth_override, "Override the track synth for this lane");
    if !lane.sample_path.trim().is_empty() {
        ui.weak(
            "This lane has a sample loaded — the sample takes priority and plays instead of any \
             synth until it's cleared.",
        );
    }
    if lane.synth_override {
        egui::CollapsingHeader::new("Synth").default_open(true).show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("Engine:");
                if ui.selectable_label(lane.synth_engine == SynthEngine::Simple, "Simple Synth").clicked() {
                    lane.synth_engine = SynthEngine::Simple;
                }
                if ui.selectable_label(lane.synth_engine == SynthEngine::Trine, "Trine").clicked() {
                    lane.synth_engine = SynthEngine::Trine;
                }
                if ui.selectable_label(lane.synth_engine == SynthEngine::Wave, "Wave").clicked() {
                    lane.synth_engine = SynthEngine::Wave;
                }
            });
            match lane.synth_engine {
                SynthEngine::Simple => synth_params_ui(ui, &mut lane.synth),
                SynthEngine::Trine => trine_params_ui(ui, &mut lane.trine),
                SynthEngine::Wave => wave_params_ui(ui, &mut lane.wave),
            }
        });
    } else {
        ui.weak("Unchecked: this lane plays the track's own synth.");
    }
}

/// The `fx_chain_ui` slice of the Device Panel, shared by both a `Track` focus and a `Lane`
/// focus (a lane's sound also passes through its track's own chain — see the caller).
fn device_panel_track_fx_chain_ui(
    ui: &mut egui::Ui,
    song: &mut Song,
    track_index: usize,
    engine_config: Option<(f64, u32, u32)>,
    panel: &mut DevicePanelUi,
) {
    if track_index >= song.tracks.len() || track_index >= panel.track_effect_paths.len() {
        return;
    }
    let mut unused_remove_requested: Option<usize> = None;
    let track_names: Vec<String> = song.tracks.iter().map(|t| t.name.clone()).collect();
    let mut fx = TrackFxUi {
        track_index,
        chain_kind: FxChainKind::Track,
        paths: &mut panel.track_effect_paths[track_index],
        messages: &mut panel.track_effect_messages[track_index],
        slots: panel.track_effect_slots.clone(),
        instances: &mut panel.track_effect_instances[track_index],
        guis: &mut panel.track_effect_guis[track_index],
        engine_config,
        known_plugins: &song.plugins,
        track_names: &track_names,
        editor: &mut *panel.effect_editor,
        device_chain_focus: &mut *panel.focus,
        remove_requested: &mut unused_remove_requested,
        inline_params: true,
    };
    fx_chain_ui(ui, &mut fx);
}

/// One slot's row (CLAP path/Load or built-in label, sidechain picker, Params button, status
/// message, remove button) plus — only when `fx.inline_params` is set (the Device Panel) — that
/// built-in effect's own knobs directly beneath it. Shared by `fx_chain_ui`'s two container
/// shapes (a plain vertical list for the Channel Rack/Mixer "FX" menus, a row of boxed device
/// columns for the Device Panel) so the slot-editing logic itself only exists once.
fn fx_chain_slot_ui(
    ui: &mut egui::Ui,
    fx: &mut TrackFxUi,
    slot_index: usize,
    fx_slot_to_remove: &mut Option<usize>,
) {
    let slot_kind = fx
        .slots
        .lock()
        .ok()
        .and_then(|guard| {
            let slot = guard.get(fx.track_index)?.get(slot_index)?.as_ref();
            Some(match slot {
                Some(EffectInstance::BuiltIn(effect)) => FxSlotKind::BuiltIn(effect.label()),
                Some(EffectInstance::Clap(_)) | None => FxSlotKind::Clap,
            })
        })
        .unwrap_or(FxSlotKind::Clap);
    ui.horizontal(|ui| {
        // Drag handle for reordering — see `fx_chain_ui`'s `dnd_drop_zone` wrapping each slot,
        // which is what actually applies the reorder once this is dropped onto another slot.
        let drag_id =
            egui::Id::new(("fx_chain_device_drag", fx.chain_kind, fx.track_index, slot_index));
        ui.dnd_drag_source(drag_id, slot_index, |ui| {
            ui.label("☰");
        })
        .response
        .on_hover_text("Drag to reorder");
        ui.weak(format!("{}.", slot_index + 1));
        match slot_kind {
            FxSlotKind::Clap => {
                ui.add_sized(
                    [150.0, 20.0],
                    egui::TextEdit::singleline(&mut fx.paths[slot_index]).hint_text("effect.clap"),
                );
                if !fx.known_plugins.is_empty() {
                    ui.menu_button("📁", |ui| {
                        for plugin in fx.known_plugins {
                            if ui.button(&plugin.name).clicked() {
                                fx.paths[slot_index] = plugin.path.clone();
                                ui.close();
                            }
                        }
                    })
                    .response
                    .on_hover_text("Pick from the project's imported plugins");
                }
                let can_load =
                    fx.engine_config.is_some() && !fx.paths[slot_index].trim().is_empty();
                if ui
                    .add_enabled(can_load, egui::Button::new("Load"))
                    .clicked()
                {
                    if let Some((sample_rate, min_frames, max_frames)) = fx.engine_config {
                        let path = std::path::Path::new(fx.paths[slot_index].trim());
                        let result = plugin_host::load_and_activate(
                            path,
                            sample_rate,
                            min_frames,
                            max_frames,
                        );
                        fx.messages[slot_index] = Some(match result {
                            Ok((instance, effect, gui)) => {
                                fx.instances[slot_index] = Some(instance);
                                fx.guis[slot_index] = Some(gui);
                                if let Ok(mut slots) = fx.slots.lock() {
                                    if let Some(chain) = slots.get_mut(fx.track_index) {
                                        if let Some(entry) = chain.get_mut(slot_index) {
                                            *entry = Some(EffectInstance::Clap(effect));
                                        }
                                    }
                                }
                                (true, format!("Loaded {}", path.display()))
                            }
                            Err(err) => (false, format!("{err:#}")),
                        });
                    }
                }
            }
            FxSlotKind::BuiltIn(label) => {
                ui.label(label);
            }
        }
    });
    // Second row — sidechain picker, Params/remove buttons, status message — kept off the name
    // row above so a narrow Device Panel column (see `fx_chain_ui`'s boxed-device layout) doesn't
    // force that row wider than the column itself; costs an extra row in the plain vertical list
    // (Channel Rack/Mixer "FX" menus) too, but that has room to spare.
    ui.horizontal(|ui| {
        // Sidechain key-source picker — only rendered for a slot kind that actually carries a
        // `sidechain_source` (a loaded CLAP plugin, Compressor, or NoiseGate; see
        // `plugin_host::EffectInstance::sidechain_source_mut`). Reads/writes the live runtime
        // effect directly, same as every other per-effect parameter in this UI.
        if let Ok(mut slots) = fx.slots.lock()
            && let Some(chain) = slots.get_mut(fx.track_index)
            && let Some(Some(instance)) = chain.get_mut(slot_index)
            && let Some(source) = instance.sidechain_source_mut()
        {
            let selected_label = source
                .and_then(|index| fx.track_names.get(index))
                .map(String::as_str)
                .unwrap_or("None");
            egui::ComboBox::from_id_salt(("sidechain_source", fx.track_index, slot_index))
                .selected_text(format!("SC: {selected_label}"))
                .width(90.0)
                .show_ui(ui, |ui| {
                    ui.selectable_value(source, None, "None");
                    for (index, name) in fx.track_names.iter().enumerate() {
                        ui.selectable_value(source, Some(index), name);
                    }
                });
        }
        if ui.small_button("Params").clicked() {
            *fx.editor = Some(fx_editor_target(fx, slot_index));
        }
        if ui
            .small_button("🗑")
            .on_hover_text("Remove this effect from the chain")
            .clicked()
        {
            *fx_slot_to_remove = Some(slot_index);
        }
        if let Some((ok, message)) = fx.messages[slot_index].as_ref() {
            let color = if *ok {
                egui::Color32::from_rgb(120, 220, 140)
            } else {
                egui::Color32::RED
            };
            ui.colored_label(color, message);
        }
    });
    if fx.inline_params
        && let FxSlotKind::BuiltIn(_) = slot_kind
        && let Ok(mut slots) = fx.slots.lock()
        && let Some(chain) = slots.get_mut(fx.track_index)
        && let Some(Some(EffectInstance::BuiltIn(effect))) = chain.get_mut(slot_index)
    {
        built_in_effect_params_ui(ui, effect);
    }
}

/// Renders the "+ Add Effect" menu and the ordered list of the track's effect-chain slots
/// (CLAP path/Load or built-in label, Params button, status message, remove button). Opened
/// from each Channel Rack row's "FX" popup menu, the Mixer's per-strip "FX" menu, and the bottom
/// Device Panel (see `channel_rack_row_ui`/`device_panel_track_fx_chain_ui`).
pub(crate) fn fx_chain_ui(ui: &mut egui::Ui, fx: &mut TrackFxUi) {
    ui.label("FX chain:");
    ui.menu_button("+ Add Effect", |ui| {
        if ui.button("CLAP Plugin…").clicked() {
            fx.paths.push(String::new());
            fx.instances.push(None);
            fx.guis.push(None);
            fx.messages.push(None);
            if let Ok(mut slots) = fx.slots.lock() {
                if let Some(chain) = slots.get_mut(fx.track_index) {
                    chain.push(None);
                }
            }
            ui.close();
        }
        if !fx.known_plugins.is_empty() {
            ui.separator();
            ui.menu_button("From Project Library", |ui| {
                for plugin in fx.known_plugins {
                    if ui.button(&plugin.name).clicked() {
                        fx.paths.push(plugin.path.clone());
                        let (instance, gui, chain_entry, message) =
                            match load_effect(&plugin.path, &[], fx.engine_config) {
                                Ok((instance, effect, gui)) => (
                                    Some(instance),
                                    Some(gui),
                                    Some(EffectInstance::Clap(effect)),
                                    (true, format!("Loaded {}", plugin.name)),
                                ),
                                Err(err) => (None, None, None, (false, err)),
                            };
                        fx.instances.push(instance);
                        fx.guis.push(gui);
                        fx.messages.push(Some(message));
                        if let Ok(mut slots) = fx.slots.lock() {
                            if let Some(chain) = slots.get_mut(fx.track_index) {
                                chain.push(chain_entry);
                            }
                        }
                        ui.close();
                    }
                }
            });
        }
        ui.separator();
        // Built-in effects need no external file and no separate Load step — they're
        // live in the chain as soon as they're added (unlike CLAP, above).
        for (label, config) in [
            ("Delay / Echo", TrackEffectConfig::default_delay()),
            ("Bitcrusher", TrackEffectConfig::default_bitcrusher()),
            ("Distortion", TrackEffectConfig::default_distortion()),
            ("Reverb", TrackEffectConfig::default_reverb()),
            ("Chorus", TrackEffectConfig::default_chorus()),
            ("Filter", TrackEffectConfig::default_filter()),
            ("Tremolo", TrackEffectConfig::default_tremolo()),
            ("Compressor", TrackEffectConfig::default_compressor()),
            ("Flanger", TrackEffectConfig::default_flanger()),
            ("Phaser", TrackEffectConfig::default_phaser()),
            (
                "Ring Modulator",
                TrackEffectConfig::default_ring_modulator(),
            ),
            ("Noise Gate", TrackEffectConfig::default_noise_gate()),
            ("Phase Invert", TrackEffectConfig::default_phase_invert()),
            ("Channel EQ", TrackEffectConfig::default_channel_eq()),
            ("Limiter", TrackEffectConfig::default_limiter()),
        ] {
            if ui.button(label).clicked() {
                fx.paths.push(String::new());
                fx.instances.push(None);
                fx.guis.push(None);
                let sample_rate = fx.engine_config.map(|(sr, _, _)| sr as f32);
                let (entry, message) =
                    match sample_rate.and_then(|sr| BuiltInEffect::from_config(&config, sr)) {
                        Some(effect) => (Some(EffectInstance::BuiltIn(effect)), None),
                        None => (None, Some((false, "Audio engine not running".to_string()))),
                    };
                fx.messages.push(message);
                if let Ok(mut slots) = fx.slots.lock() {
                    if let Some(chain) = slots.get_mut(fx.track_index) {
                        chain.push(entry);
                    }
                }
                ui.close();
            }
        }
    });
    let mut fx_slot_to_remove: Option<usize> = None;
    // Set when a slot's "☰" drag handle (see `fx_chain_slot_ui`) is dropped onto another slot's
    // `dnd_drop_zone` below — applied once, after the loop, the same "compute during the loop,
    // apply after" shape `fx_slot_to_remove` already uses (mutating `fx.paths` etc. mid-loop would
    // desync the zero-indexed `slot_index` every remaining iteration is still relying on).
    let mut reorder: Option<(usize, usize)> = None;
    if fx.inline_params {
        // Bitwig/Ableton-style device rack: one boxed column per device, side by side, rather
        // than the plain vertical list every other `fx_chain_ui` caller (Channel Rack/Mixer "FX"
        // dropdown menus) still uses — those have no inline knobs to box up (see
        // `TrackFxUi::inline_params`), so a vertical list is still the right shape there.
        egui::ScrollArea::horizontal()
            .id_salt(("device_rack_scroll", fx.track_index))
            .scroll_bar_visibility(egui::containers::scroll_area::ScrollBarVisibility::AlwaysVisible)
            .show(ui, |ui| {
                ui.horizontal_top(|ui| {
                    for slot_index in 0..fx.paths.len() {
                        let (_, dropped) = ui.dnd_drop_zone::<usize, _>(
                            egui::Frame::group(ui.style()).inner_margin(egui::Margin::symmetric(8, 6)),
                            |ui| {
                                ui.set_width(190.0);
                                ui.spacing_mut().item_spacing = egui::vec2(4.0, 3.0);
                                ui.vertical(|ui| {
                                    fx_chain_slot_ui(ui, fx, slot_index, &mut fx_slot_to_remove);
                                });
                            },
                        );
                        if let Some(dragged_from) = dropped {
                            reorder = Some((*dragged_from, slot_index));
                        }
                    }
                });
            });
    } else {
        for slot_index in 0..fx.paths.len() {
            let (_, dropped) = ui.dnd_drop_zone::<usize, _>(egui::Frame::new(), |ui| {
                fx_chain_slot_ui(ui, fx, slot_index, &mut fx_slot_to_remove);
            });
            if let Some(dragged_from) = dropped {
                reorder = Some((*dragged_from, slot_index));
            }
        }
    }
    if let Some((from, to)) = reorder {
        reorder_fx_slot(fx, from, to);
    }
    if let Some(slot_index) = fx_slot_to_remove {
        fx.paths.remove(slot_index);
        let mut removed_instance = fx.instances.remove(slot_index);
        let mut removed_gui = fx.guis.remove(slot_index);
        if let (Some(instance), Some(gui)) = (removed_instance.as_mut(), removed_gui.as_mut()) {
            plugin_host::close_plugin_gui(instance, gui);
        }
        fx.messages.remove(slot_index);
        if let Ok(mut slots) = fx.slots.lock() {
            if let Some(chain) = slots.get_mut(fx.track_index) {
                if slot_index < chain.len() {
                    chain.remove(slot_index);
                }
            }
        }
        if *fx.editor == Some(fx_editor_target(fx, slot_index)) {
            *fx.editor = None;
        }
    }
}

/// Moves the effect at `from` to `to` within one chain — every parallel `Vec` `TrackFxUi` carries
/// (`paths`/`instances`/`guis`/`messages`) plus the live `slots` chain, kept in the same order by
/// construction (see `TrackFxUi`'s fields) so they must all move together. Drops any open "FX
/// Params" window for this chain rather than trying to track which slot it should now follow —
/// same simplification `fx_chain_ui`'s slot-removal already makes.
fn reorder_fx_slot(fx: &mut TrackFxUi, from: usize, to: usize) {
    if from == to || from >= fx.paths.len() || to >= fx.paths.len() {
        return;
    }
    let path = fx.paths.remove(from);
    fx.paths.insert(to, path);
    let instance = fx.instances.remove(from);
    fx.instances.insert(to, instance);
    let gui = fx.guis.remove(from);
    fx.guis.insert(to, gui);
    let message = fx.messages.remove(from);
    fx.messages.insert(to, message);
    if let Ok(mut slots) = fx.slots.lock()
        && let Some(chain) = slots.get_mut(fx.track_index)
        && from < chain.len()
    {
        let entry = chain.remove(from);
        chain.insert(to.min(chain.len()), entry);
    }
    *fx.editor = None;
}

/// Renders sliders for `effect`'s parameters (if any are loaded), writing changes straight into
/// the plugin via `LoadedEffect::set_param`. Shared by both the master-bus and per-track windows.
pub(crate) fn effect_params_ui(ui: &mut egui::Ui, effect: Option<&mut LoadedEffect>) {
    let Some(effect) = effect else {
        ui.weak("No effect loaded.");
        return;
    };
    if effect.params.is_empty() {
        ui.weak("This plugin exposes no parameters.");
        return;
    }
    for index in 0..effect.params.len() {
        let PluginParamInfo {
            name,
            min_value,
            max_value,
            ..
        } = effect.params[index].clone();
        let mut value = effect.param_value(index).unwrap_or(min_value);
        if ui
            .add(egui::Slider::new(&mut value, min_value..=max_value).text(name))
            .changed()
        {
            effect.set_param(index, value);
        }
    }
}

/// Renders an "Open GUI"/"Close GUI" toggle for a loaded CLAP plugin's GUI, next to
/// `effect_params_ui`'s sliders in the FX params window — and, while the currently-open GUI is
/// embedded, reserves a panel of window space matching its negotiated size and keeps its native
/// container view positioned under that panel every frame. Renders nothing if the plugin doesn't
/// implement the `gui` extension. Also polls whether the plugin closed its own floating window
/// since the last frame (e.g. the user hit its close button), so the toggle's label stays in sync.
///
/// `embed_target` is `Some((window_handle, scale_factor))` whenever the app's own window handle
/// is available (always, outside headless/test contexts) — `open_plugin_gui` tries embedding
/// first and only falls back to a floating window if the plugin or platform doesn't support it.
/// `target`/`active_embedded_gui` track which slot currently owns the one embedded panel that can
/// exist at a time (see `SimpleDawApp::active_embedded_gui`'s doc comment); the caller is
/// responsible for having already closed any *other* slot's embedded GUI before this runs.
pub(crate) fn plugin_gui_button_ui(
    ui: &mut egui::Ui,
    instance: &mut PluginInstance<DawHost>,
    gui: &mut PluginGuiHandle,
    title: &str,
    embed_target: Option<(RawWindowHandle, f64)>,
    target: EffectEditorTarget,
    active_embedded_gui: &mut Option<EffectEditorTarget>,
) {
    if !gui.is_supported() {
        return;
    }
    plugin_host::plugin_gui_poll_closed(instance, gui);
    ui.separator();
    if gui.is_open() {
        if ui.button("Close GUI").clicked() {
            plugin_host::close_plugin_gui(instance, gui);
            if *active_embedded_gui == Some(target) {
                *active_embedded_gui = None;
            }
        } else if let Some(size) = gui.embedded_size() {
            let (rect, _response) = ui.allocate_exact_size(
                egui::vec2(size.width as f32, size.height as f32),
                egui::Sense::hover(),
            );
            plugin_host::resize_embedded_plugin_gui(
                gui,
                rect.min.x as f64,
                rect.min.y as f64,
                rect.width() as f64,
                rect.height() as f64,
            );
        }
    } else if ui.button("Open GUI").clicked() {
        match plugin_host::open_plugin_gui(instance, gui, title, embed_target) {
            Ok(_) => {
                if gui.is_embedded() {
                    *active_embedded_gui = Some(target);
                }
            }
            Err(err) => {
                ui.colored_label(egui::Color32::RED, format!("{err:#}"));
            }
        }
    }
}

/// Closes the GUI (floating or embedded) for the effect slot at `target`, if one is open — used
/// to tear down a stale embedded GUI when the "FX Params" window switches to a different slot or
/// closes entirely (see the two call sites in `SimpleDawApp::ui`). A free function taking each
/// collection explicitly, the same shape as `apply_loaded_effects` and friends, since it needs to
/// look a slot up in whichever of the four (master/track/send/submix) it turns out to be in.
#[allow(clippy::too_many_arguments)]
pub(crate) fn close_effect_gui(
    target: EffectEditorTarget,
    master_effect_instances: &mut [Option<PluginInstance<DawHost>>],
    master_effect_guis: &mut [Option<PluginGuiHandle>],
    track_effect_instances: &mut [Vec<Option<PluginInstance<DawHost>>>],
    track_effect_guis: &mut [Vec<Option<PluginGuiHandle>>],
    send_effect_instances: &mut [Vec<Option<PluginInstance<DawHost>>>],
    send_effect_guis: &mut [Vec<Option<PluginGuiHandle>>],
    submix_effect_instances: &mut [Vec<Option<PluginInstance<DawHost>>>],
    submix_effect_guis: &mut [Vec<Option<PluginGuiHandle>>],
) {
    let (instance, gui) = match target {
        EffectEditorTarget::Master(slot_index) => (
            master_effect_instances
                .get_mut(slot_index)
                .and_then(|instance| instance.as_mut()),
            master_effect_guis.get_mut(slot_index).and_then(|gui| gui.as_mut()),
        ),
        EffectEditorTarget::Track(track_index, slot_index) => (
            track_effect_instances
                .get_mut(track_index)
                .and_then(|slots| slots.get_mut(slot_index))
                .and_then(|instance| instance.as_mut()),
            track_effect_guis
                .get_mut(track_index)
                .and_then(|slots| slots.get_mut(slot_index))
                .and_then(|gui| gui.as_mut()),
        ),
        EffectEditorTarget::Send(send_index, slot_index) => (
            send_effect_instances
                .get_mut(send_index)
                .and_then(|slots| slots.get_mut(slot_index))
                .and_then(|instance| instance.as_mut()),
            send_effect_guis
                .get_mut(send_index)
                .and_then(|slots| slots.get_mut(slot_index))
                .and_then(|gui| gui.as_mut()),
        ),
        EffectEditorTarget::Submix(submix_index, slot_index) => (
            submix_effect_instances
                .get_mut(submix_index)
                .and_then(|slots| slots.get_mut(slot_index))
                .and_then(|instance| instance.as_mut()),
            submix_effect_guis
                .get_mut(submix_index)
                .and_then(|slots| slots.get_mut(slot_index))
                .and_then(|gui| gui.as_mut()),
        ),
    };
    if let (Some(instance), Some(gui)) = (instance, gui) {
        plugin_host::close_plugin_gui(instance, gui);
    }
}

/// Renders sliders for a built-in (non-CLAP) effect's parameters, writing straight into its live
/// DSP state. Unlike a CLAP effect there's no separate plugin process to notify of a change — a
/// direct field write here takes effect on the very next processed audio block, since the UI
/// thread and the audio callback share this same `BuiltInEffect` behind `TrackEffectSlots`' mutex.
pub(crate) fn built_in_effect_params_ui(ui: &mut egui::Ui, effect: &mut BuiltInEffect) {
    match effect {
        BuiltInEffect::Delay(e) => {
            ui.add(
                egui::Slider::new(&mut e.time_ms, 1.0..=2000.0)
                    .text("Time")
                    .suffix(" ms"),
            );
            ui.add(egui::Slider::new(&mut e.feedback, 0.0..=0.95).text("Feedback"));
            ui.add(egui::Slider::new(&mut e.mix, 0.0..=1.0).text("Mix"));
        }
        BuiltInEffect::Bitcrusher(e) => {
            ui.add(egui::Slider::new(&mut e.bit_depth, 1.0..=16.0).text("Bit depth"));
            ui.add(egui::Slider::new(&mut e.rate_divisor, 1..=50).text("Rate divisor"));
            ui.add(egui::Slider::new(&mut e.mix, 0.0..=1.0).text("Mix"));
        }
        BuiltInEffect::Distortion(e) => {
            ui.add(egui::Slider::new(&mut e.drive, 1.0..=20.0).text("Drive"));
            ui.add(egui::Slider::new(&mut e.mix, 0.0..=1.0).text("Mix"));
        }
        BuiltInEffect::Reverb(e) => {
            ui.add(egui::Slider::new(&mut e.room_size, 0.0..=1.0).text("Room size"));
            ui.add(egui::Slider::new(&mut e.damping, 0.0..=1.0).text("Damping"));
            ui.add(egui::Slider::new(&mut e.mix, 0.0..=1.0).text("Mix"));
        }
        BuiltInEffect::Chorus(e) => {
            ui.add(
                egui::Slider::new(&mut e.rate_hz, 0.05..=10.0)
                    .text("Rate")
                    .suffix(" Hz"),
            );
            ui.add(
                egui::Slider::new(&mut e.depth_ms, 0.0..=30.0)
                    .text("Depth")
                    .suffix(" ms"),
            );
            ui.add(egui::Slider::new(&mut e.mix, 0.0..=1.0).text("Mix"));
        }
        BuiltInEffect::Filter(e) => {
            ui.horizontal(|ui| {
                ui.label("Mode:");
                ui.selectable_value(&mut e.mode, FilterMode::LowPass, "Low-pass");
                ui.selectable_value(&mut e.mode, FilterMode::HighPass, "High-pass");
            });
            ui.add(
                egui::Slider::new(&mut e.cutoff_hz, 20.0..=18000.0)
                    .text("Cutoff")
                    .suffix(" Hz"),
            );
            ui.add(egui::Slider::new(&mut e.resonance, 0.0..=0.99).text("Resonance"));
            ui.add(egui::Slider::new(&mut e.mix, 0.0..=1.0).text("Mix"));
        }
        BuiltInEffect::Tremolo(e) => {
            ui.add(
                egui::Slider::new(&mut e.rate_hz, 0.1..=20.0)
                    .text("Rate")
                    .suffix(" Hz"),
            );
            ui.add(egui::Slider::new(&mut e.depth, 0.0..=1.0).text("Depth"));
        }
        BuiltInEffect::Compressor(e) => {
            ui.add(
                egui::Slider::new(&mut e.threshold_db, -60.0..=0.0)
                    .text("Threshold")
                    .suffix(" dB"),
            );
            ui.add(egui::Slider::new(&mut e.ratio, 1.0..=20.0).text("Ratio"));
            ui.add(
                egui::Slider::new(&mut e.attack_ms, 0.1..=200.0)
                    .text("Attack")
                    .suffix(" ms"),
            );
            ui.add(
                egui::Slider::new(&mut e.release_ms, 5.0..=1000.0)
                    .text("Release")
                    .suffix(" ms"),
            );
            ui.add(
                egui::Slider::new(&mut e.makeup_db, 0.0..=24.0)
                    .text("Makeup")
                    .suffix(" dB"),
            );
        }
        BuiltInEffect::Flanger(e) => {
            ui.add(
                egui::Slider::new(&mut e.rate_hz, 0.05..=5.0)
                    .text("Rate")
                    .suffix(" Hz"),
            );
            ui.add(
                egui::Slider::new(&mut e.depth_ms, 0.0..=10.0)
                    .text("Depth")
                    .suffix(" ms"),
            );
            ui.add(egui::Slider::new(&mut e.feedback, 0.0..=0.95).text("Feedback"));
            ui.add(egui::Slider::new(&mut e.mix, 0.0..=1.0).text("Mix"));
        }
        BuiltInEffect::Phaser(e) => {
            ui.add(
                egui::Slider::new(&mut e.rate_hz, 0.05..=5.0)
                    .text("Rate")
                    .suffix(" Hz"),
            );
            ui.add(egui::Slider::new(&mut e.depth, 0.0..=1.0).text("Depth"));
            ui.add(egui::Slider::new(&mut e.feedback, 0.0..=0.95).text("Feedback"));
            ui.add(egui::Slider::new(&mut e.mix, 0.0..=1.0).text("Mix"));
        }
        BuiltInEffect::RingModulator(e) => {
            ui.add(
                egui::Slider::new(&mut e.carrier_hz, 1.0..=2000.0)
                    .text("Carrier")
                    .suffix(" Hz"),
            );
            ui.add(egui::Slider::new(&mut e.mix, 0.0..=1.0).text("Mix"));
        }
        BuiltInEffect::NoiseGate(e) => {
            ui.add(
                egui::Slider::new(&mut e.threshold_db, -80.0..=0.0)
                    .text("Threshold")
                    .suffix(" dB"),
            );
            ui.add(
                egui::Slider::new(&mut e.attack_ms, 0.1..=100.0)
                    .text("Attack")
                    .suffix(" ms"),
            );
            ui.add(
                egui::Slider::new(&mut e.release_ms, 5.0..=1000.0)
                    .text("Release")
                    .suffix(" ms"),
            );
            ui.add(
                egui::Slider::new(&mut e.range_db, -96.0..=0.0)
                    .text("Range")
                    .suffix(" dB"),
            );
        }
        BuiltInEffect::PhaseInvert(e) => {
            ui.checkbox(&mut e.invert_left, "Invert L");
            ui.checkbox(&mut e.invert_right, "Invert R");
        }
        BuiltInEffect::Limiter(e) => {
            ui.add(
                egui::Slider::new(&mut e.input_gain_db, -12.0..=24.0)
                    .text("Input Gain")
                    .suffix(" dB"),
            );
            ui.add(
                egui::Slider::new(&mut e.ceiling_db, -12.0..=0.0)
                    .text("Ceiling")
                    .suffix(" dB"),
            );
            ui.add(
                egui::Slider::new(&mut e.release_ms, 5.0..=500.0)
                    .text("Release")
                    .suffix(" ms"),
            );
        }
        BuiltInEffect::ChannelEq(e) => {
            for (index, band) in e.bands.iter_mut().enumerate() {
                ui.horizontal(|ui| {
                    ui.checkbox(&mut band.enabled, "");
                    egui::ComboBox::from_id_salt(("channel-eq-band-type", index))
                        .selected_text(eq_band_type_label(band.band_type))
                        .show_ui(ui, |ui| {
                            for band_type in [
                                EqBandType::HighPass,
                                EqBandType::LowShelf,
                                EqBandType::Peak,
                                EqBandType::HighShelf,
                                EqBandType::LowPass,
                            ] {
                                ui.selectable_value(
                                    &mut band.band_type,
                                    band_type,
                                    eq_band_type_label(band_type),
                                );
                            }
                        });
                    ui.add(
                        egui::Slider::new(&mut band.freq_hz, 20.0..=20_000.0)
                            .logarithmic(true)
                            .text("Freq")
                            .suffix(" Hz"),
                    );
                    let has_gain =
                        !matches!(band.band_type, EqBandType::HighPass | EqBandType::LowPass);
                    ui.add_enabled(
                        has_gain,
                        egui::Slider::new(&mut band.gain_db, -18.0..=18.0)
                            .text("Gain")
                            .suffix(" dB"),
                    );
                    ui.add(egui::Slider::new(&mut band.q, 0.1..=10.0).text("Q"));
                });
            }
        }
    }
}

/// Short label for a Channel EQ band's shape, used by the type combo box in `render_effect_params`.
fn eq_band_type_label(band_type: EqBandType) -> &'static str {
    match band_type {
        EqBandType::HighPass => "High Pass",
        EqBandType::LowShelf => "Low Shelf",
        EqBandType::Peak => "Peak",
        EqBandType::HighShelf => "High Shelf",
        EqBandType::LowPass => "Low Pass",
    }
}

/// Whether `preset` holds exactly the params `track` currently has loaded for `preset.engine` —
/// used to highlight the active selection in the preset picker combo box.
fn preset_matches_track(preset: &SynthPreset, track: &Track) -> bool {
    match preset.engine {
        SynthEngine::Simple => preset.params == track.synth,
        SynthEngine::Trine => preset.trine.as_ref() == Some(&track.trine),
        SynthEngine::Wave => preset.wave.as_ref() == Some(&track.wave),
    }
}

/// Renders the preset bar at the top of a track's synth window: pick a saved preset from
/// `song.synth_presets` to load into this track's synth, save the track's current synth as a new
/// named preset, rename/delete existing presets, or import/export a single preset to its own JSON
/// file (see `SynthPreset`). Scoped to whichever engine the track currently has selected — the
/// picker and "Manage presets" list only show presets saved from that engine, since a `Trine`
/// preset has nothing meaningful to load into a `Simple Synth` track. Loading/saving copies the
/// engine's params struct by value — it's not a live link, so editing the track afterward never
/// mutates the library entry, and vice versa.
fn synth_preset_bar_ui(
    ui: &mut egui::Ui,
    song: &mut Song,
    track_index: usize,
    new_preset_name: &mut String,
    message: &mut Option<(bool, String)>,
) {
    let engine = song.tracks[track_index].synth_engine;
    let factory = factory_presets();
    let selected_name = factory
        .iter()
        .chain(song.synth_presets.iter())
        .find(|p| p.engine == engine && preset_matches_track(p, &song.tracks[track_index]))
        .map(|p| p.name.clone());

    let mut load_preset = None;
    ui.horizontal(|ui| {
        ui.label("Preset:");
        egui::ComboBox::from_id_salt(("synth-preset-picker", track_index))
            .selected_text(selected_name.as_deref().unwrap_or("— none —"))
            .show_ui(ui, |ui| {
                ui.label(egui::RichText::new("Factory").weak());
                for preset in factory.iter().filter(|p| p.engine == engine) {
                    if ui.selectable_label(false, &preset.name).clicked() {
                        load_preset = Some(preset.clone());
                    }
                }
                ui.separator();
                ui.label(egui::RichText::new("Custom").weak());
                for preset in song.synth_presets.iter().filter(|p| p.engine == engine) {
                    if ui.selectable_label(false, &preset.name).clicked() {
                        load_preset = Some(preset.clone());
                    }
                }
            });
        if ui.button("Import…").clicked() {
            if let Some(path) = browse_for_file("", "Synth preset", &["json"], None) {
                match SynthPreset::load_from_file(Path::new(&path)) {
                    Ok(preset) => {
                        *message = Some((true, format!("Imported preset \"{}\"", preset.name)));
                        song.synth_presets.push(preset);
                    }
                    Err(err) => *message = Some((false, format!("{err:#}"))),
                }
            }
        }
    });
    if let Some(preset) = load_preset {
        let track = &mut song.tracks[track_index];
        match engine {
            SynthEngine::Simple => track.synth = preset.params,
            SynthEngine::Trine => {
                if let Some(trine) = preset.trine {
                    track.trine = trine;
                }
            }
            SynthEngine::Wave => {
                if let Some(wave) = preset.wave {
                    track.wave = wave;
                }
            }
        }
    }

    ui.horizontal(|ui| {
        ui.add(
            egui::TextEdit::singleline(new_preset_name)
                .hint_text("Preset name")
                .desired_width(160.0),
        );
        let can_save = !new_preset_name.trim().is_empty();
        if ui
            .add_enabled(can_save, egui::Button::new("Save as preset"))
            .clicked()
        {
            let name = new_preset_name.trim().to_string();
            let track = &song.tracks[track_index];
            let preset = match engine {
                SynthEngine::Simple => SynthPreset {
                    name: name.clone(),
                    engine,
                    params: track.synth,
                    trine: None,
                    wave: None,
                },
                SynthEngine::Trine => SynthPreset {
                    name: name.clone(),
                    engine,
                    params: SynthParams::default(),
                    trine: Some(track.trine.clone()),
                    wave: None,
                },
                SynthEngine::Wave => SynthPreset {
                    name: name.clone(),
                    engine,
                    params: SynthParams::default(),
                    trine: None,
                    wave: Some(track.wave.clone()),
                },
            };
            song.synth_presets.push(preset);
            new_preset_name.clear();
            *message = Some((true, format!("Saved preset \"{name}\"")));
        }
    });

    let has_presets_for_engine = song.synth_presets.iter().any(|p| p.engine == engine);
    if has_presets_for_engine {
        ui.collapsing("Manage presets", |ui| {
            let mut to_delete = None;
            let mut to_export = None;
            for (index, preset) in song.synth_presets.iter_mut().enumerate() {
                if preset.engine != engine {
                    continue;
                }
                ui.horizontal(|ui| {
                    ui.add(egui::TextEdit::singleline(&mut preset.name).desired_width(140.0));
                    if ui.button("Export…").clicked() {
                        to_export = Some(index);
                    }
                    if ui.button("Delete").clicked() {
                        to_delete = Some(index);
                    }
                });
            }
            if let Some(index) = to_export {
                let default_name = format!("{}.json", song.synth_presets[index].name);
                if let Some(path) =
                    browse_for_file("", "Synth preset", &["json"], Some(&default_name))
                {
                    *message = match song.synth_presets[index].save_to_file(Path::new(&path)) {
                        Ok(()) => Some((true, format!("Exported to {path}"))),
                        Err(err) => Some((false, format!("{err:#}"))),
                    };
                }
            }
            if let Some(index) = to_delete {
                song.synth_presets.remove(index);
            }
        });
    }

    if let Some((ok, text)) = message {
        let color = if *ok {
            egui::Color32::from_rgb(120, 220, 140)
        } else {
            egui::Color32::RED
        };
        ui.colored_label(color, text.as_str());
    }

    ui.separator();
}
