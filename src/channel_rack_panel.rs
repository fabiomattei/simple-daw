//! The Channel Rack (left panel): `ChannelRackUi` bundles the mutable app-state borrows shared
//! by the docked and detached-window renderings (see `SimpleDawApp::update`),
//! `channel_rack_contents_ui` draws the heading/"+ Add" menu/track-row list, and
//! `channel_rack_row_ui` draws one compact per-track row (swatch, mute/solo, name, volume/pan,
//! freeze/bounce, arm-for-record, instrument/FX shortcuts).

use crate::audio;
use crate::audio_input;
use crate::file_ops::browse_for_file;
use crate::metering::MeterHandles;
use crate::model::{AudioClip, Song, Track, TrackKind};
use crate::plugin_host::{DawHost, PluginGuiHandle, TrackEffectSlots};
use crate::device_panel::{DeviceChainFocus, EffectEditorTarget, FxChainKind, TrackFxUi, fx_chain_ui, pan_label};
use crate::{
    FL_ACCENT_ORANGE, FL_ACCENT_YELLOW, TrackFreezeAction, audio_clip_length_ticks, resize_track_effects,
    resize_track_meters, track_color,
};
use clack_host::prelude::PluginInstance;

/// Bundles the Channel Rack's mutable app-state borrows so `channel_rack_contents_ui` (shared
/// between the docked `egui::Panel::left` rendering and the detached-window rendering — see
/// `SimpleDawApp::ui`) doesn't need a dozen positional parameters. `song.lock()` is held for the
/// rest of `SimpleDawApp::ui`, so this borrows individual fields rather than `&mut self` — a
/// method taking `&mut self` would conflict with that outstanding lock guard.
pub(crate) struct ChannelRackUi<'a> {
    pub(crate) selected_track: &'a Option<usize>,
    pub(crate) selected_beats_track: &'a Option<usize>,
    pub(crate) detached: &'a mut bool,
    pub(crate) track_effect_slots: &'a TrackEffectSlots,
    pub(crate) track_effect_instances: &'a mut Vec<Vec<Option<PluginInstance<DawHost>>>>,
    pub(crate) track_effect_guis: &'a mut Vec<Vec<Option<PluginGuiHandle>>>,
    pub(crate) track_effect_paths: &'a mut Vec<Vec<String>>,
    pub(crate) track_effect_messages: &'a mut Vec<Vec<Option<(bool, String)>>>,
    pub(crate) track_meters: &'a MeterHandles,
    pub(crate) effect_editor: &'a mut Option<EffectEditorTarget>,
    pub(crate) device_chain_focus: &'a mut Option<DeviceChainFocus>,
    /// The record-armed `Audio`-kind track (if any) and its chosen input device — see
    /// `SimpleDawApp::record_armed_track`/`selected_input_device`.
    pub(crate) record_armed_track: &'a mut Option<usize>,
    pub(crate) selected_input_device: &'a mut Option<String>,
}

/// The Channel Rack's heading/"+ Add" menu/track-row list, including the Detach/Dock toggle
/// button — shared by the docked and detached-window renderings so the two stay in sync.
pub(crate) fn channel_rack_contents_ui(
    ui: &mut egui::Ui,
    song: &mut Song,
    engine_config: Option<(f64, u32, u32)>,
    track_to_remove: &mut Option<usize>,
    freeze_requested: &mut Option<(usize, TrackFreezeAction)>,
    rack: &mut ChannelRackUi,
) {
    let sample_rate = engine_config.map(|(sr, _, _)| sr as u32);
    let bpm = song.bpm;
    ui.horizontal(|ui| {
        ui.heading("Channel Rack");
        ui.menu_button("+ Add", |ui| {
            if ui.button("Piano Roll Track").clicked() {
                let midi_channel = (song.tracks.len() as u8 % 16) + 1;
                let name = format!("Track {}", song.tracks.len() + 1);
                song.add_track(name, midi_channel, TrackKind::PianoRoll);
                resize_track_effects(
                    rack.track_effect_slots,
                    rack.track_effect_instances,
                    rack.track_effect_guis,
                    rack.track_effect_paths,
                    rack.track_effect_messages,
                    song.tracks.len(),
                );
                resize_track_meters(rack.track_meters, song.tracks.len());
                ui.close();
            }
            if ui.button("Step Grid Track").clicked() {
                let midi_channel = (song.tracks.len() as u8 % 16) + 1;
                let name = format!("Track {}", song.tracks.len() + 1);
                song.add_track(name, midi_channel, TrackKind::StepGrid);
                resize_track_effects(
                    rack.track_effect_slots,
                    rack.track_effect_instances,
                    rack.track_effect_guis,
                    rack.track_effect_paths,
                    rack.track_effect_messages,
                    song.tracks.len(),
                );
                resize_track_meters(rack.track_meters, song.tracks.len());
                ui.close();
            }
            if ui.button("Audio Track").clicked() {
                let midi_channel = (song.tracks.len() as u8 % 16) + 1;
                let name = format!("Track {}", song.tracks.len() + 1);
                let new_index = song.add_track(name, midi_channel, TrackKind::Audio);
                *rack.record_armed_track = Some(new_index);
                resize_track_effects(
                    rack.track_effect_slots,
                    rack.track_effect_instances,
                    rack.track_effect_guis,
                    rack.track_effect_paths,
                    rack.track_effect_messages,
                    song.tracks.len(),
                );
                resize_track_meters(rack.track_meters, song.tracks.len());
                ui.close();
            }
        });
        if ui
            .small_button(if *rack.detached {
                "⏷ Dock"
            } else {
                "⧉ Detach"
            })
            .clicked()
        {
            *rack.detached = !*rack.detached;
        }
    });
    ui.separator();

    let track_names: Vec<String> = song.tracks.iter().map(|t| t.name.clone()).collect();
    egui::ScrollArea::vertical().show(ui, |ui| {
        for (track_index, track) in song.tracks.iter_mut().enumerate() {
            let mut fx = TrackFxUi {
                track_index,
                chain_kind: FxChainKind::Track,
                paths: &mut rack.track_effect_paths[track_index],
                messages: &mut rack.track_effect_messages[track_index],
                slots: rack.track_effect_slots.clone(),
                instances: &mut rack.track_effect_instances[track_index],
                guis: &mut rack.track_effect_guis[track_index],
                engine_config,
                known_plugins: &song.plugins,
                track_names: &track_names,
                editor: &mut *rack.effect_editor,
                device_chain_focus: &mut *rack.device_chain_focus,
                remove_requested: &mut *track_to_remove,
                inline_params: false,
            };
            channel_rack_row_ui(
                ui,
                track,
                track_index,
                rack.selected_track,
                rack.selected_beats_track,
                &mut fx,
                rack.record_armed_track,
                rack.selected_input_device,
                sample_rate,
                bpm,
                freeze_requested,
            );
            ui.add_space(4.0);
        }
    });
}
/// One compact row in the Channel Rack (left panel): a colored swatch, mute LED, name field,
/// volume slider, and Synth/FX/Remove buttons. Neither piano-roll nor step-grid tracks show their
/// notes/regions inline, and there's no button here to open either editor — a track's Piano
/// Roll/Beats window only opens by double-clicking one of its regions in the Playlist (see
/// `PlaylistEditorTargets`); this row just dims/highlights (`is_roll_selected`/`is_beats_selected`)
/// to reflect whether that window happens to be open for this track already.
#[allow(clippy::too_many_arguments)]
fn channel_rack_row_ui(
    ui: &mut egui::Ui,
    track: &mut Track,
    track_index: usize,
    selected_track: &Option<usize>,
    selected_beats_track: &Option<usize>,
    fx: &mut TrackFxUi,
    record_armed_track: &mut Option<usize>,
    selected_input_device: &mut Option<String>,
    sample_rate: Option<u32>,
    bpm: f32,
    freeze_requested: &mut Option<(usize, TrackFreezeAction)>,
) {
    let is_piano_roll = track.kind == TrackKind::PianoRoll;
    let is_step_grid = track.kind == TrackKind::StepGrid;
    let is_audio = track.kind == TrackKind::Audio;
    let is_roll_selected = is_piano_roll && *selected_track == Some(track_index);
    let is_beats_selected = is_step_grid && *selected_beats_track == Some(track_index);
    let color = track_color(track_index);

    let is_armed = is_audio && *record_armed_track == Some(track_index);
    let frame = egui::Frame::new()
        .fill(if is_armed {
            egui::Color32::from_rgb(64, 32, 32)
        } else if is_roll_selected || is_beats_selected {
            egui::Color32::from_rgb(61, 61, 40)
        } else {
            egui::Color32::from_rgb(45, 45, 45)
        })
        .corner_radius(3.0)
        .inner_margin(egui::Margin::symmetric(6, 4));

    frame.show(ui, |ui| {
        ui.vertical(|ui| {
            ui.horizontal(|ui| {
                let (swatch_rect, _) =
                    ui.allocate_exact_size(egui::vec2(6.0, 20.0), egui::Sense::hover());
                ui.painter().rect_filled(swatch_rect, 1.0, color);

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

                ui.add(egui::TextEdit::singleline(&mut track.name).desired_width(84.0));

                ui.add(
                    egui::Slider::new(&mut track.volume, 0.0..=1.5)
                        .show_value(false)
                        .trailing_fill(true),
                )
                .on_hover_text(format!("Volume: {:.2}", track.volume));

                ui.add(egui::Slider::new(&mut track.pan, -1.0..=1.0).show_value(false))
                    .on_hover_text(format!("Pan: {}", pan_label(track.pan)));

                let freeze_color =
                    if track.frozen { egui::Color32::from_rgb(120, 200, 240) } else { egui::Color32::from_gray(150) };
                if ui
                    .add(
                        egui::Button::new(egui::RichText::new("❄").color(freeze_color))
                            .small()
                            .min_size(egui::vec2(20.0, 20.0)),
                    )
                    .on_hover_text(if track.frozen {
                        "Unfreeze: resume live synthesis/effects"
                    } else {
                        "Freeze: bake this track's notes/steps/audio + effects to audio, to save CPU"
                    })
                    .clicked()
                {
                    *freeze_requested = Some((
                        track_index,
                        if track.frozen { TrackFreezeAction::Unfreeze } else { TrackFreezeAction::Freeze },
                    ));
                }
                if ui
                    .small_button("💾")
                    .on_hover_text(
                        "Bounce in place: permanently replace this track's content with baked audio",
                    )
                    .clicked()
                {
                    *freeze_requested = Some((track_index, TrackFreezeAction::Bounce));
                }

                if !is_audio {
                    let is_focused =
                        *fx.device_chain_focus == Some(DeviceChainFocus::Track(fx.track_index));
                    if ui
                        .add(egui::Button::new("🎹").small().selected(is_focused))
                        .on_hover_text("Show this track's instrument + FX chain in the Device Panel")
                        .clicked()
                    {
                        *fx.device_chain_focus = Some(DeviceChainFocus::Track(fx.track_index));
                    }
                }
                if is_audio {
                    let armed_glyph = if is_armed { "⏺" } else { "⏵" };
                    let armed_color = if is_armed {
                        FL_ACCENT_ORANGE
                    } else {
                        egui::Color32::from_gray(150)
                    };
                    if ui
                        .add(
                            egui::Button::new(egui::RichText::new(armed_glyph).color(armed_color))
                                .small()
                                .min_size(egui::vec2(20.0, 20.0)),
                        )
                        .on_hover_text(if is_armed {
                            "Recording-armed (click to disarm)"
                        } else {
                            "Arm for recording"
                        })
                        .clicked()
                    {
                        *record_armed_track = if is_armed { None } else { Some(track_index) };
                    }
                    if ui
                        .small_button("📂")
                        .on_hover_text("Import a WAV file onto this track")
                        .clicked()
                    {
                        if let Some(path) = browse_for_file("", "WAV audio", &["wav"], None) {
                            let ticks_per_second = audio::ticks_per_second(bpm);
                            let start_tick = track
                                .audio_clips
                                .iter()
                                .map(|c| {
                                    c.start_tick + audio_clip_length_ticks(c, ticks_per_second)
                                })
                                .max()
                                .unwrap_or(0);
                            let mut clip = AudioClip::new(start_tick, path);
                            if let Some(rate) = sample_rate {
                                clip.load(rate);
                            }
                            track.audio_clips.push(clip);
                        }
                    }
                }
                ui.menu_button("FX", |ui| {
                    fx_chain_ui(ui, fx);
                });
                if ui
                    .small_button("🗑")
                    .on_hover_text("Delete this track")
                    .clicked()
                {
                    *fx.remove_requested = Some(fx.track_index);
                }
            });

            if is_armed {
                ui.horizontal(|ui| {
                    ui.weak("Input:");
                    egui::ComboBox::from_id_salt(("audio_input_device_picker", track_index))
                        .selected_text(selected_input_device.as_deref().unwrap_or("Default"))
                        .show_ui(ui, |ui| {
                            if ui
                                .selectable_label(selected_input_device.is_none(), "Default")
                                .clicked()
                            {
                                *selected_input_device = None;
                            }
                            for name in audio_input::list_input_devices() {
                                let selected =
                                    selected_input_device.as_deref() == Some(name.as_str());
                                if ui.selectable_label(selected, &name).clicked() {
                                    *selected_input_device = Some(name);
                                }
                            }
                        });
                    ui.weak("Trim:");
                    ui.add(egui::Slider::new(&mut track.input_gain, 0.0..=2.0).show_value(false))
                        .on_hover_text(format!("Input gain: {:.2}", track.input_gain));
                });
            }
        });
    });
}
