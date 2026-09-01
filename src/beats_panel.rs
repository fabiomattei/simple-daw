//! The Beats window: the step-grid counterpart of the Piano Roll. `beats_contents_ui` draws the
//! header (selected track, dock/detach) and step grid for whichever Playlist region or Session
//! View slot is currently being edited (see `RegionEditTarget`); `step_grid_lanes_ui` draws one
//! pattern's lanes (name, sample controls, step buttons); `step_grid_lane_groove_menu_ui` is each
//! lane's own humanize/groove-template popup, the step-grid counterpart of the Piano Roll
//! toolbar's quantize/humanize controls.

use crate::audio;
use crate::groove::{self, GROOVE_TEMPLATES};
use crate::model::{
    Lane, MAX_STEP_TIMING_OFFSET_TICKS, RegionContent, SessionClipContent, Song, StepData, TICKS_PER_STEP,
    TrackEffectConfig, TrackKind,
};
use crate::device_panel::DeviceChainFocus;
use crate::plugin_host::{MasterEffectSlots, SendEffectSlots, TrackEffectSlots};
use crate::{
    AutomationDrag, FL_ACCENT_ORANGE, RegionEditTarget, automation_lanes_ui, lane_sample_controls, note_name,
    track_color,
};

/// Bundles the Beats window's shared groove/humanize controls — the same underlying
/// `SimpleDawApp` fields the Piano Roll toolbar uses (see `PianoRollPanelUi`'s `groove_*`
/// fields), reused here since "how much to humanize by" is a general preference, not a
/// per-lane setting.
pub(crate) struct StepGrooveUi<'a> {
    pub(crate) humanize_timing_ticks: &'a mut usize,
    pub(crate) humanize_velocity: &'a mut u8,
    pub(crate) template_index: &'a mut usize,
}

/// One lane's groove menu ("🎲"): humanize timing/velocity sliders and a groove-template picker,
/// each with its own Apply button — the step-grid counterpart of the Piano Roll's Quantize/
/// Humanize/Groove Template toolbar (`piano_roll_quantize_humanize_groove_ui`). Applies to every
/// active step in this lane — the Beats window has no per-step selection to narrow it to.
fn step_grid_lane_groove_menu_ui(
    ui: &mut egui::Ui,
    lane_index: usize,
    lane: &mut Lane,
    groove: &mut StepGrooveUi,
) {
    ui.menu_button("🎲", |ui| {
        ui.label("Humanize");
        ui.add(
            egui::Slider::new(groove.humanize_timing_ticks, 0..=MAX_STEP_TIMING_OFFSET_TICKS as usize)
                .text("Timing"),
        );
        ui.add(egui::Slider::new(groove.humanize_velocity, 0..=40).text("Velocity"));
        if ui
            .button("Apply")
            .on_hover_text("Randomly nudge every active step's timing/velocity in this lane")
            .clicked()
        {
            let seed = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0);
            groove::humanize_steps(
                &mut lane.steps,
                *groove.humanize_timing_ticks as u8,
                *groove.humanize_velocity,
                seed,
            );
        }
        ui.separator();
        ui.label("Groove Template");
        egui::ComboBox::from_id_salt(("step_groove_template", lane_index))
            .selected_text(GROOVE_TEMPLATES[*groove.template_index].name)
            .show_ui(ui, |ui| {
                for (index, template) in GROOVE_TEMPLATES.iter().enumerate() {
                    ui.selectable_value(groove.template_index, index, template.name);
                }
            });
        if ui
            .button("Apply")
            .on_hover_text("Apply this template's swing/accent to every active step in this lane")
            .clicked()
        {
            groove::apply_groove_template_to_steps(
                &mut lane.steps,
                &GROOVE_TEMPLATES[*groove.template_index],
            );
        }
    });
}

/// A step-grid pattern's lanes: each lane's name, sample-load controls, and step buttons — the
/// Beats window's contents (see `beats_contents_ui`), extracted so the row layout is defined in
/// one place.
/// Draws every lane's row and returns the index of a lane the user clicked "🗑" on, if any —
/// the caller applies the removal via `Song::remove_lane` so it stays in sync across patterns.
/// The `DeviceChainFocus` a click on `lane_index`'s "🎹" button should set — `Lane` for a
/// Playlist region, `SessionSlotLane` for a Session View slot, matching `edit_target`.
fn device_chain_focus_for_lane(
    track_index: usize,
    edit_target: RegionEditTarget,
    lane_index: usize,
) -> DeviceChainFocus {
    match edit_target {
        RegionEditTarget::Region(region_index) => {
            DeviceChainFocus::Lane { track_index, region_index, lane_index }
        }
        RegionEditTarget::SessionSlot(slot_index) => {
            DeviceChainFocus::SessionSlotLane { track_index, slot_index, lane_index }
        }
    }
}

fn step_grid_lanes_ui(
    ui: &mut egui::Ui,
    lanes: &mut [Lane],
    current_tick: Option<usize>,
    sample_rate: Option<u32>,
    color: egui::Color32,
    track_index: usize,
    edit_target: RegionEditTarget,
    device_chain_focus: &mut Option<DeviceChainFocus>,
    groove: &mut StepGrooveUi,
) -> Option<usize> {
    let mut remove_lane = None;
    let current_step = current_tick.map(|t| t / TICKS_PER_STEP);
    for (lane_index, lane) in lanes.iter_mut().enumerate() {
        ui.horizontal(|ui| {
            ui.add(
                egui::TextEdit::singleline(&mut lane.name)
                    .desired_width(64.0)
                    .font(egui::TextStyle::Small),
            );
            if ui.small_button("🗑").on_hover_text("Remove lane").clicked() {
                remove_lane = Some(lane_index);
            }
            ui.add(egui::DragValue::new(&mut lane.pitch).range(0..=127))
                .on_hover_text(format!(
                    "Pitch (synth lanes only) — {}",
                    note_name(lane.pitch)
                ));
            let this_lane_focus = device_chain_focus_for_lane(track_index, edit_target, lane_index);
            let is_focused = *device_chain_focus == Some(this_lane_focus);
            let synth_button = egui::Button::new("🎹").selected(lane.synth_override || is_focused);
            if ui
                .add(synth_button)
                .on_hover_text("Show this lane's own synth (overrides the track synth) in the Device Panel")
                .clicked()
            {
                *device_chain_focus = Some(this_lane_focus);
            }
            step_grid_lane_groove_menu_ui(ui, lane_index, lane, groove);
            lane_sample_controls(ui, lane, sample_rate);
            for (i, step) in lane.steps.iter_mut().enumerate() {
                if i > 0 && i % 4 == 0 {
                    ui.add_space(4.0);
                }
                let active = step.is_some();
                let fill = if active {
                    color
                } else if (i / 4) % 2 == 0 {
                    egui::Color32::from_rgb(54, 54, 54)
                } else {
                    egui::Color32::from_rgb(46, 46, 46)
                };
                let mut button = egui::Button::new("")
                    .fill(fill)
                    .min_size(egui::vec2(16.0, 16.0));
                if current_step == Some(i) {
                    button = button.stroke(egui::Stroke::new(2.0, egui::Color32::WHITE));
                }
                if ui.add(button).clicked() {
                    *step = if active {
                        None
                    } else {
                        Some(StepData { velocity: 100, timing_offset_ticks: 0 })
                    };
                }
            }
        });
    }
    remove_lane
}

/// The Beats window's header (selected track name/mute badge, dock/detach toggle) and step grid,
/// rendered either inside its own OS window or docked into the central area (see `beats_detached`/
/// `beats_docked` in `ui`, `impl eframe::App for SimpleDawApp`) — the step-grid counterpart of
/// `piano_roll_contents_ui`, including the "no in-window picker, double-click a region in the
/// Playlist instead" behavior.
#[allow(clippy::too_many_arguments)]
pub(crate) fn beats_contents_ui(
    ui: &mut egui::Ui,
    song: &mut Song,
    current_tick: Option<usize>,
    sample_rate: Option<u32>,
    selected_beats_track: Option<usize>,
    editing_target: &mut Option<RegionEditTarget>,
    device_chain_focus: &mut Option<DeviceChainFocus>,
    track_effect_slots: &TrackEffectSlots,
    send_effect_slots: &SendEffectSlots,
    master_effect_slots: &MasterEffectSlots,
    automation_drag: &mut Option<AutomationDrag>,
    track_automation_drag: &mut Option<AutomationDrag>,
    groove: &mut StepGrooveUi,
    detached: &mut bool,
) {
    let selected = selected_beats_track
        .filter(|&i| i < song.tracks.len())
        .filter(|&i| song.tracks[i].kind == TrackKind::StepGrid);
    // Resolves `editing_target` against `selected`'s track, validating the target still exists —
    // same reasoning as `piano_roll_contents_ui`'s equivalent (see `RegionEditTarget`'s doc
    // comment); for a session slot, its content must still be `StepGrid`-shaped.
    let region = selected.and_then(|index| match (*editing_target)? {
        RegionEditTarget::Region(region_index) => (region_index < song.tracks[index].regions.len())
            .then_some((index, RegionEditTarget::Region(region_index))),
        RegionEditTarget::SessionSlot(slot_index) => song.tracks[index]
            .session_clips
            .get(slot_index)
            .and_then(|slot| slot.as_ref())
            .is_some_and(|clip| {
                matches!(
                    clip.content,
                    SessionClipContent::Region { content: RegionContent::StepGrid(_), .. }
                )
            })
            .then_some((index, RegionEditTarget::SessionSlot(slot_index))),
    });

    ui.horizontal(|ui| match selected {
        Some(index) => {
            let color = track_color(index);
            let (swatch_rect, _) =
                ui.allocate_exact_size(egui::vec2(10.0, 22.0), egui::Sense::hover());
            ui.painter().rect_filled(swatch_rect, 2.0, color);
            ui.heading(song.tracks[index].name.clone());
            if song.tracks[index].muted {
                ui.colored_label(FL_ACCENT_ORANGE, "MUTED");
            }
            if let Some((_, target)) = region {
                ui.separator();
                match target {
                    RegionEditTarget::Region(region_index) => {
                        ui.weak(&song.tracks[index].regions[region_index].name);
                    }
                    RegionEditTarget::SessionSlot(slot_index) => {
                        let name = song.tracks[index].session_clips[slot_index]
                            .as_ref()
                            .map(|clip| clip.name.as_str())
                            .unwrap_or_default();
                        ui.weak(format!("{name} (Session View)"));
                    }
                }
            }
        }
        None => {
            ui.heading("Beats");
        }
    });
    if ui
        .small_button(if *detached { "⏷ Dock" } else { "⧉ Detach" })
        .clicked()
    {
        *detached = !*detached;
    }
    ui.separator();

    if let Some(index) = selected {
        // Same pre-borrow snapshot the region panel below takes for the same reason — see its
        // own comment. Computed separately (a second clone per frame) rather than shared, so this
        // panel doesn't depend on whether a region is also being edited below.
        let track_effects_snapshot = song.tracks[index].effects.clone();
        let other_tracks_snapshot: Vec<(String, Vec<TrackEffectConfig>)> = song
            .tracks
            .iter()
            .map(|t| (t.name.clone(), t.effects.clone()))
            .collect();
        let arrangement_span_ticks = audio::arrangement_length_ticks(song);
        // Beats has no continuous zoom of its own (its step grid is fixed-width cells) — a fixed
        // default keeps this graph at a reasonable density regardless, same as the region panel.
        let zoom = 1.0;
        ui.collapsing("Track Automation", |ui| {
            egui::ScrollArea::vertical().id_salt("track_wide_automation").max_height(90.0).show(
                ui,
                |ui| {
                    automation_lanes_ui(
                        ui,
                        &mut song.tracks[index].automation,
                        arrangement_span_ticks,
                        index,
                        &track_effects_snapshot,
                        track_effect_slots,
                        &other_tracks_snapshot,
                        &song.sends,
                        send_effect_slots,
                        &song.master_effects,
                        master_effect_slots,
                        zoom,
                        track_automation_drag,
                    );
                },
            );
        });
        ui.separator();
    }

    match region {
        None => {
            ui.centered_and_justified(|ui| {
                ui.weak("Double-click a region in the Playlist to edit it here.");
            });
        }
        Some((index, RegionEditTarget::Region(region_index))) => {
            let color = track_color(index);
            if ui.small_button("+ Lane").clicked() {
                let lane_count = match &song.tracks[index].regions[region_index].content {
                    RegionContent::StepGrid(lanes) => lanes.len(),
                    _ => 0,
                };
                song.tracks[index].add_lane(format!("Lane {}", lane_count + 1), 60);
            }
            let track_effects_snapshot = song.tracks[index].effects.clone();
            // Same pre-borrow snapshot as `piano_roll_contents_ui` — see its comment.
            let other_tracks_snapshot: Vec<(String, Vec<TrackEffectConfig>)> = song
                .tracks
                .iter()
                .map(|t| (t.name.clone(), t.effects.clone()))
                .collect();
            let mut lane_to_remove = None;
            let region = &mut song.tracks[index].regions[region_index];
            if let RegionContent::StepGrid(lanes) = &mut region.content {
                lane_to_remove = step_grid_lanes_ui(
                    ui,
                    lanes,
                    current_tick,
                    sample_rate,
                    color,
                    index,
                    RegionEditTarget::Region(region_index),
                    device_chain_focus,
                    groove,
                );
            }
            ui.separator();
            // Beats has no continuous zoom of its own (its step grid is fixed-width cells) —
            // a fixed default keeps the automation graph at a reasonable density regardless.
            let zoom = 1.0;
            let region_span_ticks = region.loop_length_steps * TICKS_PER_STEP;
            egui::ScrollArea::vertical().max_height(110.0).show(ui, |ui| {
                automation_lanes_ui(
                    ui,
                    &mut region.automation,
                    region_span_ticks,
                    index,
                    &track_effects_snapshot,
                    track_effect_slots,
                    &other_tracks_snapshot,
                    &song.sends,
                    send_effect_slots,
                    &song.master_effects,
                    master_effect_slots,
                    zoom,
                    automation_drag,
                );
            });
            if let Some(lane_index) = lane_to_remove {
                song.tracks[index].remove_lane(lane_index);
            }
        }
        Some((index, RegionEditTarget::SessionSlot(slot_index))) => {
            let color = track_color(index);
            // Same pre-borrow snapshot as the `Region` arm above, for the same reason —
            // `automation_lanes_ui`'s "Other Track" targets need to list every track.
            let track_effects_snapshot = song.tracks[index].effects.clone();
            let other_tracks_snapshot: Vec<(String, Vec<TrackEffectConfig>)> = song
                .tracks
                .iter()
                .map(|t| (t.name.clone(), t.effects.clone()))
                .collect();
            // Unlike `Track::add_lane`/`remove_lane` (which apply to every region on the track at
            // once — see their doc comments), a session slot's lanes are its own independent list,
            // so lane add/remove here just mutates this one slot's `Vec<Lane>` directly.
            let Some(Some(clip)) = song.tracks[index].session_clips.get_mut(slot_index) else {
                ui.weak("Clip no longer exists.");
                return;
            };
            let SessionClipContent::Region { content, content_length_steps, loop_length_steps } =
                &mut clip.content
            else {
                ui.weak("Clip no longer exists.");
                return;
            };
            let clip_span_ticks = *loop_length_steps * TICKS_PER_STEP;
            let RegionContent::StepGrid(lanes) = content else {
                ui.weak("Clip no longer exists.");
                return;
            };
            if ui.small_button("+ Lane").clicked() {
                let lane_count = lanes.len();
                lanes.push(Lane::new(format!("Lane {}", lane_count + 1), 60, *content_length_steps));
            }
            let lane_to_remove = step_grid_lanes_ui(
                ui,
                lanes,
                current_tick,
                sample_rate,
                color,
                index,
                RegionEditTarget::SessionSlot(slot_index),
                device_chain_focus,
                groove,
            );
            if let Some(lane_index) = lane_to_remove
                && lane_index < lanes.len()
            {
                lanes.remove(lane_index);
            }
            ui.separator();
            // Beats has no continuous zoom of its own — same fixed default as the `Region` arm.
            let zoom = 1.0;
            egui::ScrollArea::vertical().max_height(110.0).show(ui, |ui| {
                automation_lanes_ui(
                    ui,
                    &mut clip.automation,
                    clip_span_ticks,
                    index,
                    &track_effects_snapshot,
                    track_effect_slots,
                    &other_tracks_snapshot,
                    &song.sends,
                    send_effect_slots,
                    &song.master_effects,
                    master_effect_slots,
                    zoom,
                    automation_drag,
                );
            });
        }
    }
}
