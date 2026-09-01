//! The Mixer window: `MixerUi` bundles the mutable app-state borrows shared by the docked and
//! detached-window renderings (see `SimpleDawApp::update`), and `mixer_contents_ui` draws the
//! classic vertical channel-strip row (per track, Master, every send bus, every submix bus) that
//! both renderings share. Peak/RMS bar meters and LUFS formatting live here too since nothing
//! else in the app currently needs them.

use crate::metering::{self, MeterHandles, MeterReadings};
use crate::model::{SendBus, Song, SubmixBus, Track, TrackOutput};
use crate::plugin_host::{
    DawHost, MasterEffectSlots, PluginGuiHandle, SendEffectSlots, SubmixEffectSlots, TrackEffectSlots,
};
use crate::{
    DeviceChainFocus, EffectEditorTarget, FL_ACCENT_GREEN, FL_ACCENT_ORANGE, FL_ACCENT_YELLOW, FxChainKind,
    TrackFxUi, fx_chain_ui, pan_label, remove_track_effects, remove_track_meter, resize_track_effects,
    resize_track_meters, track_color,
};
use clack_host::prelude::PluginInstance;

/// Bundles the Mixer's mutable app-state borrows, for the same reason as `ChannelRackUi` — reused
/// between the docked and detached-window renderings. Also carries the master bus's own
/// effect-chain bookkeeping (see `SimpleDawApp::master_effect_paths` and friends) so the Mixer can
/// show a Master strip alongside the per-track ones, the same chain the "Plugins" window's "Master
/// bus FX chain" section edits.
pub(crate) struct MixerUi<'a> {
    pub(crate) detached: &'a mut bool,
    pub(crate) track_effect_slots: &'a TrackEffectSlots,
    pub(crate) track_effect_instances: &'a mut Vec<Vec<Option<PluginInstance<DawHost>>>>,
    pub(crate) track_effect_guis: &'a mut Vec<Vec<Option<PluginGuiHandle>>>,
    pub(crate) track_effect_paths: &'a mut Vec<Vec<String>>,
    pub(crate) track_effect_messages: &'a mut Vec<Vec<Option<(bool, String)>>>,
    pub(crate) track_meters: &'a MeterHandles,
    pub(crate) effect_editor: &'a mut Option<EffectEditorTarget>,
    pub(crate) master_effect_paths: &'a mut Vec<String>,
    pub(crate) master_effect_slots: MasterEffectSlots,
    pub(crate) master_effect_instances: &'a mut Vec<Option<PluginInstance<DawHost>>>,
    pub(crate) master_effect_guis: &'a mut Vec<Option<PluginGuiHandle>>,
    pub(crate) master_effect_messages: &'a mut Vec<Option<(bool, String)>>,
    pub(crate) master_meter: &'a MeterHandles,
    /// Every send bus's own effect-chain bookkeeping — same shape as `track_effect_*` (one entry
    /// per `Song::sends` row), kept in sync via `resize_track_effects`/`remove_track_effects`, the
    /// same helpers `track_effect_*`/`send_effect_*` both reuse (see `SimpleDawApp::send_effect_slots`).
    pub(crate) send_effect_slots: &'a SendEffectSlots,
    pub(crate) send_effect_instances: &'a mut Vec<Vec<Option<PluginInstance<DawHost>>>>,
    pub(crate) send_effect_guis: &'a mut Vec<Vec<Option<PluginGuiHandle>>>,
    pub(crate) send_effect_paths: &'a mut Vec<Vec<String>>,
    pub(crate) send_effect_messages: &'a mut Vec<Vec<Option<(bool, String)>>>,
    /// Every submix bus's own effect-chain bookkeeping — same shape/sync mechanism as
    /// `send_effect_*` above (one entry per `Song::submixes` row).
    pub(crate) submix_effect_slots: &'a SubmixEffectSlots,
    pub(crate) submix_effect_instances: &'a mut Vec<Vec<Option<PluginInstance<DawHost>>>>,
    pub(crate) submix_effect_guis: &'a mut Vec<Vec<Option<PluginGuiHandle>>>,
    pub(crate) submix_effect_paths: &'a mut Vec<Vec<String>>,
    pub(crate) submix_effect_messages: &'a mut Vec<Vec<Option<(bool, String)>>>,
    pub(crate) submix_meters: &'a MeterHandles,
}

/// The Mixer's heading/Detach toggle plus one classic vertical channel strip per track, ending in
/// a Master strip — shared by the docked and detached-window renderings so the two stay in sync.
/// Unlike the Channel Rack's compact horizontal rows, each strip lays its controls out top to
/// bottom (name, FX, pan, mute/solo, then a tall fader) the way a hardware/DAW mixer console does.
/// These are the same `Track::volume`/`pan`/`muted`/`solo`/`effects` the Channel Rack row already
/// edits inline — the Mixer is an additional view onto the same data, not a separate copy of it.
pub(crate) fn mixer_contents_ui(
    ui: &mut egui::Ui,
    song: &mut Song,
    engine_config: Option<(f64, u32, u32)>,
    mixer: &mut MixerUi,
) {
    ui.horizontal(|ui| {
        ui.heading("Mixer");
        if ui
            .small_button(if *mixer.detached { "⏷ Dock" } else { "⧉ Detach" })
            .clicked()
        {
            *mixer.detached = !*mixer.detached;
        }
    });
    ui.separator();

    let track_names: Vec<String> = song.tracks.iter().map(|t| t.name.clone()).collect();
    egui::ScrollArea::horizontal().show(ui, |ui| {
        ui.horizontal_top(|ui| {
            for (track_index, track) in song.tracks.iter_mut().enumerate() {
                // Unused by `fx_chain_ui` itself (only `channel_rack_row_ui`'s own Synth/Remove
                // buttons touch these) — the Mixer strip has neither button, but `TrackFxUi` needs
                // somewhere to point since it's shared with the Channel Rack. Same pattern as the
                // "Plugins" window's master-chain `unused_device_chain_focus`/`unused_remove_requested`.
                let mut unused_device_chain_focus: Option<DeviceChainFocus> = None;
                let mut unused_remove_requested: Option<usize> = None;
                let mut fx = TrackFxUi {
                    track_index,
                    chain_kind: FxChainKind::Track,
                    paths: &mut mixer.track_effect_paths[track_index],
                    messages: &mut mixer.track_effect_messages[track_index],
                    slots: mixer.track_effect_slots.clone(),
                    instances: &mut mixer.track_effect_instances[track_index],
                    guis: &mut mixer.track_effect_guis[track_index],
                    engine_config,
                    known_plugins: &song.plugins,
                    track_names: &track_names,
                    editor: &mut *mixer.effect_editor,
                    device_chain_focus: &mut unused_device_chain_focus,
                    remove_requested: &mut unused_remove_requested,
                    inline_params: false,
                };
                let meter = mixer
                    .track_meters
                    .lock()
                    .ok()
                    .and_then(|handles| handles.get(track_index).map(|m| m.snapshot()))
                    .unwrap_or_default();
                mixer_channel_strip_ui(
                    ui,
                    track,
                    track_index,
                    &song.sends,
                    &song.submixes,
                    &mut fx,
                    meter,
                );
            }

            let mut unused_device_chain_focus: Option<DeviceChainFocus> = None;
            let mut unused_remove_requested: Option<usize> = None;
            let mut master_fx = TrackFxUi {
                track_index: 0,
                chain_kind: FxChainKind::Master,
                paths: mixer.master_effect_paths,
                messages: mixer.master_effect_messages,
                slots: mixer.master_effect_slots.clone(),
                instances: mixer.master_effect_instances,
                guis: mixer.master_effect_guis,
                engine_config,
                known_plugins: &song.plugins,
                track_names: &track_names,
                editor: &mut *mixer.effect_editor,
                device_chain_focus: &mut unused_device_chain_focus,
                remove_requested: &mut unused_remove_requested,
                inline_params: false,
            };
            let master_meter = mixer
                .master_meter
                .lock()
                .ok()
                .and_then(|handles| handles.first().map(|m| m.snapshot()))
                .unwrap_or_default();
            mixer_master_strip_ui(ui, &mut master_fx, master_meter);

            ui.separator();
            let mut send_to_remove: Option<usize> = None;
            for (send_index, send) in song.sends.iter_mut().enumerate() {
                let mut unused_device_chain_focus: Option<DeviceChainFocus> = None;
                let mut unused_remove_requested: Option<usize> = None;
                let mut send_fx = TrackFxUi {
                    track_index: send_index,
                    chain_kind: FxChainKind::Send,
                    paths: &mut mixer.send_effect_paths[send_index],
                    messages: &mut mixer.send_effect_messages[send_index],
                    slots: mixer.send_effect_slots.clone(),
                    instances: &mut mixer.send_effect_instances[send_index],
                    guis: &mut mixer.send_effect_guis[send_index],
                    engine_config,
                    known_plugins: &song.plugins,
                    track_names: &track_names,
                    editor: &mut *mixer.effect_editor,
                    device_chain_focus: &mut unused_device_chain_focus,
                    remove_requested: &mut unused_remove_requested,
                    inline_params: false,
                };
                mixer_send_strip_ui(ui, send, send_index, &mut send_fx, &mut send_to_remove);
            }
            if let Some(index) = send_to_remove {
                song.remove_send(index);
                remove_track_effects(
                    mixer.send_effect_slots,
                    mixer.send_effect_instances,
                    mixer.send_effect_guis,
                    mixer.send_effect_paths,
                    mixer.send_effect_messages,
                    index,
                );
            }
            ui.vertical(|ui| {
                ui.add_space(4.0);
                if ui.button("+ Add Send").clicked() {
                    song.add_send(format!("Send {}", song.sends.len() + 1));
                    resize_track_effects(
                        mixer.send_effect_slots,
                        mixer.send_effect_instances,
                        mixer.send_effect_guis,
                        mixer.send_effect_paths,
                        mixer.send_effect_messages,
                        song.sends.len(),
                    );
                }
            });

            ui.separator();
            let mut submix_to_remove: Option<usize> = None;
            for (submix_index, submix) in song.submixes.iter_mut().enumerate() {
                let mut unused_device_chain_focus: Option<DeviceChainFocus> = None;
                let mut unused_remove_requested: Option<usize> = None;
                let mut submix_fx = TrackFxUi {
                    track_index: submix_index,
                    chain_kind: FxChainKind::Submix,
                    paths: &mut mixer.submix_effect_paths[submix_index],
                    messages: &mut mixer.submix_effect_messages[submix_index],
                    slots: mixer.submix_effect_slots.clone(),
                    instances: &mut mixer.submix_effect_instances[submix_index],
                    guis: &mut mixer.submix_effect_guis[submix_index],
                    engine_config,
                    known_plugins: &song.plugins,
                    track_names: &track_names,
                    editor: &mut *mixer.effect_editor,
                    device_chain_focus: &mut unused_device_chain_focus,
                    remove_requested: &mut unused_remove_requested,
                    inline_params: false,
                };
                let meter = mixer
                    .submix_meters
                    .lock()
                    .ok()
                    .and_then(|handles| handles.get(submix_index).map(|m| m.snapshot()))
                    .unwrap_or_default();
                mixer_submix_strip_ui(
                    ui,
                    submix,
                    submix_index,
                    &mut submix_fx,
                    meter,
                    &mut submix_to_remove,
                );
            }
            if let Some(index) = submix_to_remove {
                song.remove_submix(index);
                remove_track_effects(
                    mixer.submix_effect_slots,
                    mixer.submix_effect_instances,
                    mixer.submix_effect_guis,
                    mixer.submix_effect_paths,
                    mixer.submix_effect_messages,
                    index,
                );
                remove_track_meter(mixer.submix_meters, index);
            }
            ui.vertical(|ui| {
                ui.add_space(4.0);
                if ui.button("+ Add Submix").clicked() {
                    song.add_submix(format!("Submix {}", song.submixes.len() + 1));
                    resize_track_effects(
                        mixer.submix_effect_slots,
                        mixer.submix_effect_instances,
                        mixer.submix_effect_guis,
                        mixer.submix_effect_paths,
                        mixer.submix_effect_messages,
                        song.submixes.len(),
                    );
                    resize_track_meters(mixer.submix_meters, song.submixes.len());
                }
            });
        });
    });
}

/// Floor of the peak/RMS bar meters' dB scale — anything quieter than this reads as an empty bar.
const METER_MIN_DB: f32 = -60.0;
/// Bar meter width/height for one channel (L or R) — see `peak_rms_bar_meter_ui`.
const METER_BAR_SIZE: egui::Vec2 = egui::vec2(6.0, 140.0);

/// Maps a linear amplitude to a 0.0..1.0 bar-fill fraction on `METER_MIN_DB..0dB`.
fn meter_amplitude_to_unit(amplitude: f32) -> f32 {
    if amplitude <= 0.0 {
        0.0
    } else {
        ((20.0 * amplitude.log10() - METER_MIN_DB) / -METER_MIN_DB).clamp(0.0, 1.0)
    }
}

/// Green under -6dB-ish, yellow approaching 0dB, red once a channel is effectively clipping —
/// the same traffic-light convention as this app's `FL_ACCENT_*` palette elsewhere.
fn meter_zone_color(unit: f32) -> egui::Color32 {
    if unit > 0.95 {
        egui::Color32::RED
    } else if unit > 0.8 {
        FL_ACCENT_YELLOW
    } else {
        FL_ACCENT_GREEN
    }
}

/// A small vertical L/R peak+RMS bar meter: each channel's RMS level fills the bar (colored by
/// how close to 0dB it is), with a bright peak-hold cap line drawn on top — the same
/// `rect_filled`-on-`allocate_exact_size` custom-widget idiom the recording-level meter uses.
fn peak_rms_bar_meter_ui(ui: &mut egui::Ui, readings: MeterReadings) {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 2.0;
        for (peak, rms) in [(readings.peak_l, readings.rms_l), (readings.peak_r, readings.rms_r)] {
            let (rect, _) = ui.allocate_exact_size(METER_BAR_SIZE, egui::Sense::hover());
            ui.painter().rect_filled(rect, 1.0, ui.visuals().extreme_bg_color);

            let rms_unit = meter_amplitude_to_unit(rms);
            if rms_unit > 0.0 {
                let mut fill_rect = rect;
                fill_rect.set_top(rect.bottom() - rect.height() * rms_unit);
                ui.painter().rect_filled(fill_rect, 1.0, meter_zone_color(rms_unit));
            }

            let peak_unit = meter_amplitude_to_unit(peak);
            let peak_y = rect.bottom() - rect.height() * peak_unit;
            let cap = egui::Rect::from_min_max(
                egui::pos2(rect.left(), (peak_y - 1.0).max(rect.top())),
                egui::pos2(rect.right(), peak_y + 1.0),
            );
            ui.painter().rect_filled(cap, 0.0, egui::Color32::WHITE);
        }
    });
}

/// "-14.2" for a normal reading, or an em dash once a channel is at/below the meter's silence
/// floor (no signal yet, or gated out of the integrated-loudness average).
fn format_lufs(value: f32) -> String {
    if value <= metering::SILENCE_LUFS {
        "\u{2014}".to_string()
    } else {
        format!("{value:.1}")
    }
}

/// One track's classic vertical channel strip in the Mixer: name, an "FX" menu (the same
/// `fx_chain_ui` the Channel Rack's "FX" button opens), a "Sends" menu (one level slider per
/// `Song::sends` entry, writing into `track.send_levels`), a pan slider, Mute/Solo buttons, a
/// peak/RMS bar meter beside a tall vertical volume fader, and an integrated-LUFS readout
/// (momentary/short-term available via tooltip) — see `mixer_contents_ui`.
fn mixer_channel_strip_ui(
    ui: &mut egui::Ui,
    track: &mut Track,
    track_index: usize,
    sends: &[SendBus],
    submixes: &[SubmixBus],
    fx: &mut TrackFxUi,
    meter: MeterReadings,
) {
    let color = track_color(track_index);
    egui::Frame::new()
        .fill(egui::Color32::from_rgb(40, 40, 40))
        .corner_radius(3.0)
        .inner_margin(egui::Margin::same(4))
        .show(ui, |ui| {
            ui.set_width(92.0);
            ui.vertical_centered(|ui| {
                let (swatch_rect, _) =
                    ui.allocate_exact_size(egui::vec2(58.0, 4.0), egui::Sense::hover());
                ui.painter().rect_filled(swatch_rect, 1.0, color);

                ui.add(
                    egui::TextEdit::singleline(&mut track.name)
                        .desired_width(64.0)
                        .font(egui::TextStyle::Small),
                );

                ui.menu_button("FX", |ui| {
                    fx_chain_ui(ui, fx);
                });

                if !sends.is_empty() {
                    // Self-heals a `send_levels` shorter than `sends` (e.g. a `Song` mutated
                    // outside `Song::add_send`) rather than panicking on an out-of-range index —
                    // `Song::add_send`/`add_track` already keep the two in lockstep in the normal
                    // path, so this only ever actually runs after unusual external edits.
                    while track.send_levels.len() < sends.len() {
                        track.send_levels.push(0.0);
                    }
                    ui.menu_button("Sends", |ui| {
                        for (send_index, send) in sends.iter().enumerate() {
                            ui.add(
                                egui::Slider::new(
                                    &mut track.send_levels[send_index],
                                    0.0..=1.5,
                                )
                                .text(&send.name),
                            );
                        }
                    });
                }

                // Where this track's output sums into (see `TrackOutput`) — the "Track Stack"/
                // alternate-output-routing mechanism, a plain dropdown rather than a wiring UI to
                // stay consistent with this app's no-patch-cable mixer design.
                let output_label = match track.output {
                    TrackOutput::Master => "Master".to_string(),
                    TrackOutput::Submix(index) => submixes
                        .get(index)
                        .map(|s| s.name.clone())
                        .unwrap_or_else(|| "Master".to_string()),
                };
                egui::ComboBox::from_id_salt(("track-output", track_index))
                    .selected_text(output_label)
                    .width(64.0)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut track.output, TrackOutput::Master, "Master");
                        for (submix_index, submix) in submixes.iter().enumerate() {
                            ui.selectable_value(
                                &mut track.output,
                                TrackOutput::Submix(submix_index),
                                &submix.name,
                            );
                        }
                    });

                ui.add(egui::Slider::new(&mut track.pan, -1.0..=1.0).show_value(false))
                    .on_hover_text(format!("Pan: {}", pan_label(track.pan)));

                ui.horizontal(|ui| {
                    let mute_color = if track.muted {
                        FL_ACCENT_ORANGE
                    } else {
                        egui::Color32::from_gray(150)
                    };
                    if ui
                        .add(
                            egui::Button::new(egui::RichText::new("M").color(mute_color))
                                .small()
                                .min_size(egui::vec2(18.0, 20.0)),
                        )
                        .on_hover_text(if track.muted { "Unmute" } else { "Mute" })
                        .clicked()
                    {
                        track.muted = !track.muted;
                    }

                    let solo_color = if track.solo {
                        FL_ACCENT_YELLOW
                    } else {
                        egui::Color32::from_gray(150)
                    };
                    if ui
                        .add(
                            egui::Button::new(egui::RichText::new("S").color(solo_color))
                                .small()
                                .min_size(egui::vec2(18.0, 20.0)),
                        )
                        .on_hover_text(if track.solo { "Unsolo" } else { "Solo" })
                        .clicked()
                    {
                        track.solo = !track.solo;
                    }
                });

                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    peak_rms_bar_meter_ui(ui, meter);
                    ui.add_sized(
                        [28.0, METER_BAR_SIZE.y],
                        egui::Slider::new(&mut track.volume, 0.0..=1.5)
                            .vertical()
                            .show_value(false),
                    )
                    .on_hover_text(format!("Volume: {:.2}", track.volume));
                });
                ui.label(egui::RichText::new(format!("{:.2}", track.volume)).small());
                ui.label(egui::RichText::new(format!("{} LUFS", format_lufs(meter.lufs_integrated))).small())
                    .on_hover_text(format!(
                        "Momentary: {} LUFS\nShort-term: {} LUFS\nIntegrated: {} LUFS",
                        format_lufs(meter.lufs_momentary),
                        format_lufs(meter.lufs_short_term),
                        format_lufs(meter.lufs_integrated),
                    ));
            });
        });
}

/// The Mixer's Master strip: a label, the master bus's own "FX" menu (the same chain the
/// "Plugins" window's "Master bus FX chain" section edits), a peak/RMS bar meter, and all three
/// LUFS readings stacked (there's no `Song::master_volume`/pan/mute/solo field to put a fader or
/// M/S buttons on, unlike a real track's strip, so there's headroom for the full loudness readout
/// here rather than the per-track strip's tooltip-only momentary/short-term).
fn mixer_master_strip_ui(ui: &mut egui::Ui, fx: &mut TrackFxUi, meter: MeterReadings) {
    egui::Frame::new()
        .fill(egui::Color32::from_rgb(50, 46, 30))
        .corner_radius(3.0)
        .inner_margin(egui::Margin::same(4))
        .show(ui, |ui| {
            ui.set_width(92.0);
            ui.vertical_centered(|ui| {
                ui.strong("Master");
                ui.menu_button("FX", |ui| {
                    fx_chain_ui(ui, fx);
                });
                ui.add_space(4.0);
                peak_rms_bar_meter_ui(ui, meter);
                ui.add_space(2.0);
                ui.label(egui::RichText::new(format!("M {}", format_lufs(meter.lufs_momentary))).small());
                ui.label(egui::RichText::new(format!("S {}", format_lufs(meter.lufs_short_term))).small());
                ui.label(egui::RichText::new(format!("I {}", format_lufs(meter.lufs_integrated))).small());
            });
        });
}

/// One send bus's compact strip in the Mixer: an editable name, its own "FX" menu (the same
/// `fx_chain_ui` a track/master chain uses), and a remove button — no pan/mute/solo/fader/meter,
/// since a send bus has no fader of its own in this minimal model (see `audio.rs`'s mixdown: a
/// send's chain output sums straight into the master mix at whatever level each track sent it).
fn mixer_send_strip_ui(
    ui: &mut egui::Ui,
    send: &mut SendBus,
    send_index: usize,
    fx: &mut TrackFxUi,
    remove_requested: &mut Option<usize>,
) {
    egui::Frame::new()
        .fill(egui::Color32::from_rgb(34, 42, 46))
        .corner_radius(3.0)
        .inner_margin(egui::Margin::same(4))
        .show(ui, |ui| {
            ui.set_width(92.0);
            ui.vertical_centered(|ui| {
                ui.add(
                    egui::TextEdit::singleline(&mut send.name)
                        .desired_width(64.0)
                        .font(egui::TextStyle::Small),
                );
                ui.menu_button("FX", |ui| {
                    fx_chain_ui(ui, fx);
                });
                ui.add_space(4.0);
                if ui.small_button("🗑 Remove").clicked() {
                    *remove_requested = Some(send_index);
                }
            });
        });
}

/// One submix bus's strip in the Mixer: an editable name, its own "FX" menu (same `fx_chain_ui`
/// every other chain uses), Mute/Solo (see `Sequencer::process`'s `track_silent` — silences every
/// member track at the synthesis stage, not just this bus's own summed output), a peak/RMS bar
/// meter, and a `volume` fader — the "one fader" a Track Stack sums its member tracks into. Unlike
/// `mixer_send_strip_ui`, a submix has all of these since it stands in for its member tracks'
/// direct contribution to the mix rather than being a parallel tap (see `SubmixBus`'s doc comment).
fn mixer_submix_strip_ui(
    ui: &mut egui::Ui,
    submix: &mut SubmixBus,
    submix_index: usize,
    fx: &mut TrackFxUi,
    meter: MeterReadings,
    remove_requested: &mut Option<usize>,
) {
    egui::Frame::new()
        .fill(egui::Color32::from_rgb(46, 34, 46))
        .corner_radius(3.0)
        .inner_margin(egui::Margin::same(4))
        .show(ui, |ui| {
            ui.set_width(92.0);
            ui.vertical_centered(|ui| {
                ui.add(
                    egui::TextEdit::singleline(&mut submix.name)
                        .desired_width(64.0)
                        .font(egui::TextStyle::Small),
                );
                ui.menu_button("FX", |ui| {
                    fx_chain_ui(ui, fx);
                });

                ui.add(egui::Slider::new(&mut submix.pan, -1.0..=1.0).show_value(false))
                    .on_hover_text(format!("Pan: {}", pan_label(submix.pan)));

                ui.horizontal(|ui| {
                    let mute_color = if submix.muted {
                        FL_ACCENT_ORANGE
                    } else {
                        egui::Color32::from_gray(150)
                    };
                    if ui
                        .add(
                            egui::Button::new(egui::RichText::new("M").color(mute_color))
                                .small()
                                .min_size(egui::vec2(18.0, 20.0)),
                        )
                        .on_hover_text(if submix.muted { "Unmute" } else { "Mute" })
                        .clicked()
                    {
                        submix.muted = !submix.muted;
                    }

                    let solo_color = if submix.solo {
                        FL_ACCENT_YELLOW
                    } else {
                        egui::Color32::from_gray(150)
                    };
                    if ui
                        .add(
                            egui::Button::new(egui::RichText::new("S").color(solo_color))
                                .small()
                                .min_size(egui::vec2(18.0, 20.0)),
                        )
                        .on_hover_text(if submix.solo { "Unsolo" } else { "Solo" })
                        .clicked()
                    {
                        submix.solo = !submix.solo;
                    }
                });

                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    peak_rms_bar_meter_ui(ui, meter);
                    ui.add_sized(
                        [28.0, METER_BAR_SIZE.y],
                        egui::Slider::new(&mut submix.volume, 0.0..=1.5)
                            .vertical()
                            .show_value(false),
                    )
                    .on_hover_text(format!("Volume: {:.2}", submix.volume));
                });
                ui.label(egui::RichText::new(format!("{:.2}", submix.volume)).small());

                ui.add_space(4.0);
                if ui.small_button("🗑 Remove").clicked() {
                    *remove_requested = Some(submix_index);
                }
            });
        });
}
