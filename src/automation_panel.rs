//! The automation panel shared by the Piano Roll/Beats region-scoped editors and each track's
//! own track-wide panel: `automation_lanes_ui` is the entry point (an "+ Add Lane" menu plus one
//! `automation_lane_graph_ui` row per lane), generic over which `Vec<AutomationLane>`/`span_ticks`
//! it's editing so a `Region`'s own lanes and a `Track`'s track-wide lanes share the same UI code.

use crate::builtin_fx::automatable_params_for_config;
use crate::model::{
    AutomationLane, AutomationPoint, AutomationTarget, CurveShape, EffectParamKey, SendBus, TrackEffectConfig,
};
use crate::plugin_host::{EffectInstance, MasterEffectSlots, SendEffectSlots, TrackEffectSlots};
use crate::{FL_ACCENT_GREEN, tick_to_x, x_to_tick};

/// At most one automation lane point is being dragged at a time, shared by the Piano Roll's and
/// Beats' automation panels (see `automation_lanes_ui`) the same way `PlaylistDrag` is shared by
/// the region-move/resize/fade gestures. `lane_index`/`point_index` re-check bounds every frame,
/// in case the lane or point was removed (right-click) since the drag began.
pub(crate) struct AutomationDrag {
    lane_index: usize,
    point_index: usize,
}

/// Height of one automation lane's point-graph canvas — see `automation_lane_graph_ui`.
const AUTOMATION_LANE_HEIGHT: f32 = 44.0;
/// Pixel radius of an automation point's drawn dot and its click/drag hit-test — matches
/// `RESIZE_HANDLE_PX`'s role for the Playlist's region-edge/fade handles.
const AUTOMATION_POINT_RADIUS: f32 = 4.0;
/// How many line segments approximate one non-linear `CurveShape` segment's preview — coarse
/// enough to be cheap to redraw every frame, fine enough that `Exponential`/`Logarithmic`'s curve
/// reads as smooth rather than faceted at any zoom level this canvas is drawn at.
const AUTOMATION_CURVE_PREVIEW_SAMPLES: usize = 16;

/// Human label for an effect chain slot's kind, straight from its saved config (no live instance
/// needed) — the automation target picker's counterpart to `BuiltInEffect::label()`, which only
/// works on a live instance.
fn track_effect_config_label(config: &TrackEffectConfig) -> &'static str {
    match config {
        TrackEffectConfig::Clap { .. } => "CLAP",
        TrackEffectConfig::Delay { .. } => "Delay",
        TrackEffectConfig::Bitcrusher { .. } => "Bitcrusher",
        TrackEffectConfig::Distortion { .. } => "Distortion",
        TrackEffectConfig::Reverb { .. } => "Reverb",
        TrackEffectConfig::Chorus { .. } => "Chorus",
        TrackEffectConfig::Filter { .. } => "Filter",
        TrackEffectConfig::Tremolo { .. } => "Tremolo",
        TrackEffectConfig::Compressor { .. } => "Compressor",
        TrackEffectConfig::Flanger { .. } => "Flanger",
        TrackEffectConfig::Phaser { .. } => "Phaser",
        TrackEffectConfig::RingModulator { .. } => "Ring Mod",
        TrackEffectConfig::NoiseGate { .. } => "Noise Gate",
        TrackEffectConfig::PhaseInvert { .. } => "Phase Invert",
        TrackEffectConfig::ChannelEq { .. } => "Channel EQ",
        TrackEffectConfig::Limiter { .. } => "Limiter",
    }
}

/// This chain slot's automatable parameters, as (display name, min, max, target key) — for a
/// built-in effect this is static (from the saved config alone, via `automatable_params_for_config`);
/// for a CLAP plugin it comes from whatever's actually currently loaded there (`PluginParamInfo`,
/// only known once the plugin's loaded and declared its parameters), so a CLAP slot offers nothing
/// here until it's been loaded at least once. Owned `String`s rather than borrowing from the
/// locked chain, so the result outlives the lock.
fn effect_slot_automatable_params(
    config: &TrackEffectConfig,
    track_effect_slots: &TrackEffectSlots,
    track_index: usize,
    slot_index: usize,
) -> Vec<(String, f32, f32, EffectParamKey)> {
    match config {
        TrackEffectConfig::Clap { .. } => {
            let Ok(chains) = track_effect_slots.lock() else {
                return Vec::new();
            };
            let Some(Some(EffectInstance::Clap(effect))) =
                chains.get(track_index).and_then(|chain| chain.get(slot_index))
            else {
                return Vec::new();
            };
            effect
                .params
                .iter()
                .map(|p| {
                    (
                        p.name.clone(),
                        p.min_value as f32,
                        p.max_value as f32,
                        EffectParamKey::Clap { param_id: p.id.get() },
                    )
                })
                .collect()
        }
        _ => automatable_params_for_config(config)
            .iter()
            .map(|&(name, min, max)| {
                (
                    name.to_string(),
                    min,
                    max,
                    EffectParamKey::BuiltIn { param_name: name.to_string() },
                )
            })
            .collect(),
    }
}

/// Human label for one effect-chain automation target: `FX <slot+1> (<kind>): <param>`, falling
/// back to a generic label if the slot's contents changed since the lane was created. Shared body
/// behind `automation_target_label`'s `EffectParam`/`OtherTrackEffectParam`/`SendEffectParam`/
/// `MasterEffectParam` arms, which differ only in which chain (`effects`) they read from.
fn effect_param_label(slot_index: usize, key: &EffectParamKey, effects: &[TrackEffectConfig]) -> String {
    let param_name = match key {
        EffectParamKey::Clap { param_id } => format!("param {param_id}"),
        EffectParamKey::BuiltIn { param_name } => param_name.clone(),
    };
    match effects.get(slot_index) {
        Some(config) => {
            format!("FX {} ({}): {param_name}", slot_index + 1, track_effect_config_label(config))
        }
        None => format!("FX {}: {param_name}", slot_index + 1),
    }
}

/// Human label for an already-existing lane's target, for its row header — `Volume`/`Pan` as-is
/// for this lane's own track, `<track>: Volume`/`<track>: Pan`/`<track>: Send <send>` for a
/// redirected `OtherTrack*` target, `Send: <name>` for this track's own `SendLevel`, and an
/// `effect_param_label` for any `*EffectParam` target — all falling back to a generic label (index
/// instead of name) if the referenced track/send/slot no longer matches what the lane was created
/// for.
#[allow(clippy::too_many_arguments)]
fn automation_target_label(
    target: &AutomationTarget,
    track_effects: &[TrackEffectConfig],
    other_tracks: &[(String, Vec<TrackEffectConfig>)],
    sends: &[SendBus],
    master_effects: &[TrackEffectConfig],
) -> String {
    let track_name = |track_index: usize| {
        other_tracks
            .get(track_index)
            .map_or_else(|| format!("Track {}", track_index + 1), |(name, _)| name.clone())
    };
    let send_name = |send_index: usize| {
        sends.get(send_index).map_or_else(|| format!("Send {}", send_index + 1), |s| s.name.clone())
    };
    match target {
        AutomationTarget::Volume => "Volume".to_string(),
        AutomationTarget::Pan => "Pan".to_string(),
        AutomationTarget::SendLevel { send_index } => format!("Send: {}", send_name(*send_index)),
        AutomationTarget::EffectParam { slot_index, key } => {
            effect_param_label(*slot_index, key, track_effects)
        }
        AutomationTarget::OtherTrackVolume { track_index } => {
            format!("{}: Volume", track_name(*track_index))
        }
        AutomationTarget::OtherTrackPan { track_index } => {
            format!("{}: Pan", track_name(*track_index))
        }
        AutomationTarget::OtherTrackSendLevel { track_index, send_index } => {
            format!("{}: Send {}", track_name(*track_index), send_name(*send_index))
        }
        AutomationTarget::OtherTrackEffectParam { track_index, slot_index, key } => {
            let effects = other_tracks.get(*track_index).map_or(&[][..], |(_, e)| e.as_slice());
            format!("{}: {}", track_name(*track_index), effect_param_label(*slot_index, key, effects))
        }
        AutomationTarget::SendEffectParam { send_index, slot_index, key } => {
            let effects = sends.get(*send_index).map_or(&[][..], |s| s.effects.as_slice());
            format!("{}: {}", send_name(*send_index), effect_param_label(*slot_index, key, effects))
        }
        AutomationTarget::MasterEffectParam { slot_index, key } => {
            format!("Master: {}", effect_param_label(*slot_index, key, master_effects))
        }
    }
}

/// This target's value range, for the lane graph's y-axis and new-point clamping — static ranges
/// for `Volume`/`Pan`/`SendLevel` (matching their sliders elsewhere in the Mixer, whether this
/// lane's own track or a redirected `OtherTrack*` target), or whatever
/// `effect_slot_automatable_params` reports for any `*EffectParam` target (falling back to
/// 0.0..1.0 if the slot's contents no longer match the key the lane was created for, e.g. a
/// different effect was loaded into that slot since).
#[allow(clippy::too_many_arguments)]
fn automation_target_range(
    target: &AutomationTarget,
    track_effects: &[TrackEffectConfig],
    track_effect_slots: &TrackEffectSlots,
    track_index: usize,
    other_tracks: &[(String, Vec<TrackEffectConfig>)],
    sends: &[SendBus],
    send_effect_slots: &SendEffectSlots,
    master_effects: &[TrackEffectConfig],
    master_effect_slots: &MasterEffectSlots,
) -> (f32, f32) {
    let range_for = |effects: &[TrackEffectConfig],
                      slots: &TrackEffectSlots,
                      owner_index: usize,
                      slot_index: usize,
                      key: &EffectParamKey| {
        let Some(config) = effects.get(slot_index) else {
            return (0.0, 1.0);
        };
        let params = effect_slot_automatable_params(config, slots, owner_index, slot_index);
        params
            .iter()
            .find(|(_, _, _, k)| k == key)
            .map(|&(_, min, max, _)| (min, max))
            .unwrap_or((0.0, 1.0))
    };
    match target {
        AutomationTarget::Volume | AutomationTarget::OtherTrackVolume { .. } => (0.0, 1.5),
        AutomationTarget::Pan | AutomationTarget::OtherTrackPan { .. } => (-1.0, 1.0),
        AutomationTarget::SendLevel { .. } | AutomationTarget::OtherTrackSendLevel { .. } => {
            (0.0, 1.5)
        }
        AutomationTarget::EffectParam { slot_index, key } => {
            range_for(track_effects, track_effect_slots, track_index, *slot_index, key)
        }
        AutomationTarget::OtherTrackEffectParam { track_index: other_index, slot_index, key } => {
            let effects = other_tracks.get(*other_index).map_or(&[][..], |(_, e)| e.as_slice());
            range_for(effects, track_effect_slots, *other_index, *slot_index, key)
        }
        AutomationTarget::SendEffectParam { send_index, slot_index, key } => {
            let effects = sends.get(*send_index).map_or(&[][..], |s| s.effects.as_slice());
            range_for(effects, send_effect_slots, *send_index, *slot_index, key)
        }
        AutomationTarget::MasterEffectParam { slot_index, key } => {
            range_for(master_effects, master_effect_slots, 0, *slot_index, key)
        }
    }
}

/// One automation lane's point graph: a connecting line through every point (sorted by tick for
/// display only — see `AutomationLane::value_at_fractional`'s doc comment on why storage order doesn't
/// matter), a dot per point, click-empty-space to add a point, drag a point to move it (both tick
/// and value), right-click a point to delete it. The `Region` fade triangles' visual counterpart
/// for a full multi-point "ride" rather than a single ramp.
fn automation_lane_graph_ui(
    ui: &mut egui::Ui,
    lane: &mut AutomationLane,
    lane_index: usize,
    span_ticks: usize,
    value_range: (f32, f32),
    zoom: f32,
    drag: &mut Option<AutomationDrag>,
) {
    let canvas_width = tick_to_x(span_ticks, zoom).max(40.0);
    let (response, painter) = ui.allocate_painter(
        egui::vec2(canvas_width, AUTOMATION_LANE_HEIGHT),
        egui::Sense::click_and_drag(),
    );
    let response = response.on_hover_text(
        "Click empty space: add a point  ·  Click a point: cycle its curve shape (Linear → \
         Exponential → Logarithmic → Hold)  ·  Drag: move a point  ·  Right-click: delete",
    );
    let rect = response.rect;
    painter.rect_filled(rect, 0u8, ui.visuals().extreme_bg_color);

    let (min, max) = value_range;
    let value_to_y = |value: f32| {
        let frac = if max > min { (value - min) / (max - min) } else { 0.5 };
        rect.bottom() - frac.clamp(0.0, 1.0) * rect.height()
    };
    let y_to_value = |y: f32| {
        let frac = ((rect.bottom() - y) / rect.height()).clamp(0.0, 1.0);
        min + frac * (max - min)
    };
    let point_pos = |point: &AutomationPoint| {
        egui::pos2(rect.left() + tick_to_x(point.tick, zoom), value_to_y(point.value))
    };

    let mut sorted_points: Vec<AutomationPoint> = lane.points.clone();
    sorted_points.sort_by_key(|p| p.tick);
    for pair in sorted_points.windows(2) {
        let (a, b) = (&pair[0], &pair[1]);
        let (pos_a, pos_b) = (point_pos(a), point_pos(b));
        if a.curve == CurveShape::Linear {
            // The common case, and the one every pre-curve-shapes lane already looks like —
            // drawn as a single straight segment rather than a coarse polyline through it.
            painter.add(egui::Shape::line_segment([pos_a, pos_b], egui::Stroke::new(1.5, FL_ACCENT_GREEN)));
            continue;
        }
        // A short polyline sampling `CurveShape::warp` rather than a closed-form curve shape —
        // `value_at_fractional` (the actual playback-time interpolation) evaluates the same
        // `warp` function, so this is a faithful preview, not just decoration. `Hold`'s constant
        // `warp` traces a flat line that stops short of `pos_b` — reading as "held, then jumps"
        // without needing an explicit vertical stroke to the next point's own marker.
        let segment: Vec<egui::Pos2> = (0..=AUTOMATION_CURVE_PREVIEW_SAMPLES)
            .map(|i| {
                let t = i as f32 / AUTOMATION_CURVE_PREVIEW_SAMPLES as f32;
                let value = a.value + (b.value - a.value) * a.curve.warp(t);
                egui::pos2(pos_a.x + (pos_b.x - pos_a.x) * t, value_to_y(value))
            })
            .collect();
        painter.add(egui::Shape::line(segment, egui::Stroke::new(1.5, FL_ACCENT_GREEN)));
    }
    for point in &lane.points {
        painter.circle_filled(point_pos(point), AUTOMATION_POINT_RADIUS, egui::Color32::WHITE);
    }

    let point_near = |lane: &AutomationLane, pos: egui::Pos2| -> Option<usize> {
        lane.points
            .iter()
            .position(|p| (point_pos(p) - pos).length() <= AUTOMATION_POINT_RADIUS + 3.0)
    };

    if drag.is_none() && response.drag_started() {
        if let Some(pos) = response.interact_pointer_pos() {
            if let Some(point_index) = point_near(lane, pos) {
                *drag = Some(AutomationDrag { lane_index, point_index });
            }
        }
    }

    if let Some(state) = drag {
        if state.lane_index == lane_index {
            if state.point_index >= lane.points.len() {
                *drag = None;
            } else {
                if response.dragged() {
                    if let Some(pos) = response.interact_pointer_pos() {
                        let tick = x_to_tick((pos.x - rect.left()).max(0.0), zoom).min(span_ticks);
                        let value = y_to_value(pos.y).clamp(min, max);
                        if let Some(point) = lane.points.get_mut(state.point_index) {
                            point.tick = tick;
                            point.value = value;
                        }
                    }
                }
                if response.drag_stopped() {
                    *drag = None;
                }
            }
        }
    }

    if response.clicked()
        && drag.is_none()
        && let Some(pos) = response.interact_pointer_pos()
    {
        if let Some(point_index) = point_near(lane, pos) {
            // Cycling rather than a menu: the common case (nudging one point a step or two)
            // stays a single click, and the curve preview drawn above gives instant feedback
            // on where in the cycle it landed — see this canvas's hover text.
            if let Some(point) = lane.points.get_mut(point_index) {
                point.curve = point.curve.next();
            }
        } else {
            let tick = x_to_tick((pos.x - rect.left()).max(0.0), zoom).min(span_ticks);
            let value = y_to_value(pos.y).clamp(min, max);
            lane.points.push(AutomationPoint { tick, value, curve: CurveShape::default() });
        }
    }
    if response.secondary_clicked() {
        if let Some(pos) = response.interact_pointer_pos() {
            if let Some(point_index) = point_near(lane, pos) {
                lane.points.remove(point_index);
            }
        }
    }
}

/// One "FX <slot+1> (<kind>)" submenu per slot in `effects` that has at least one automatable
/// param, each listing its params as buttons — `on_pick(slot_index, key)` is called when one's
/// clicked (the caller closes the popup and pushes the new lane; this only builds the menu tree).
/// Shared body behind `automation_lanes_ui`'s four near-identical effect-chain submenus (this
/// track's own chain, a redirected other-track's chain, a send's chain, the master chain), which
/// differ only in which `effects`/`slots`/`owner_index` they read from.
fn effect_chain_automation_menu(
    ui: &mut egui::Ui,
    effects: &[TrackEffectConfig],
    slots: &TrackEffectSlots,
    owner_index: usize,
    mut on_pick: impl FnMut(usize, EffectParamKey),
) {
    for (slot_index, config) in effects.iter().enumerate() {
        let params = effect_slot_automatable_params(config, slots, owner_index, slot_index);
        if params.is_empty() {
            continue;
        }
        let label = format!("FX {} ({})", slot_index + 1, track_effect_config_label(config));
        ui.menu_button(label, |ui| {
            for (name, _min, _max, key) in &params {
                if ui.button(name).clicked() {
                    on_pick(slot_index, key.clone());
                    ui.close();
                }
            }
        });
    }
}

/// The automation panel shown under a region's own content editor (Piano Roll or Beats) *or* under
/// a selected track's header as its track-wide panel — an "+ Add Lane" menu (Volume/Pan/every
/// send/every automatable param on this track's own effect chain, plus "Other Track"/"Send FX"/
/// "Master FX" submenus for redirected targets — see `AutomationTarget`'s doc comment) plus one
/// `automation_lane_graph_ui` row per existing lane, each with a remove button. Generic over which
/// `Vec<AutomationLane>`/`span_ticks` it edits — a `Region`'s own `automation` capped at that
/// region's on-timeline length, or a `Track`'s own track-wide `automation` capped at the whole
/// arrangement's length (`audio::arrangement_length_ticks`) — since the panel itself doesn't care
/// which; only the ticks' meaning (region-local vs. absolute) differs, and that's resolved entirely
/// by `collect_automation` at playback time, not here. `track_effects`/`track_effect_slots` are
/// this track's own; `other_tracks` is every track's (name, effects) snapshot, index-aligned with
/// `Song::tracks`, for the "Other Track" submenu.
#[allow(clippy::too_many_arguments)]
pub(crate) fn automation_lanes_ui(
    ui: &mut egui::Ui,
    automation: &mut Vec<AutomationLane>,
    span_ticks: usize,
    track_index: usize,
    track_effects: &[TrackEffectConfig],
    track_effect_slots: &TrackEffectSlots,
    other_tracks: &[(String, Vec<TrackEffectConfig>)],
    sends: &[SendBus],
    send_effect_slots: &SendEffectSlots,
    master_effects: &[TrackEffectConfig],
    master_effect_slots: &MasterEffectSlots,
    zoom: f32,
    drag: &mut Option<AutomationDrag>,
) {
    ui.horizontal(|ui| {
        ui.strong("Automation");
        ui.menu_button("+ Add Lane", |ui| {
            if ui.button("Volume").clicked() {
                automation.push(AutomationLane { target: AutomationTarget::Volume, points: Vec::new() });
                ui.close();
            }
            if ui.button("Pan").clicked() {
                automation.push(AutomationLane { target: AutomationTarget::Pan, points: Vec::new() });
                ui.close();
            }
            if !sends.is_empty() {
                ui.menu_button("Send Level", |ui| {
                    for (send_index, send) in sends.iter().enumerate() {
                        if ui.button(&send.name).clicked() {
                            automation.push(AutomationLane {
                                target: AutomationTarget::SendLevel { send_index },
                                points: Vec::new(),
                            });
                            ui.close();
                        }
                    }
                });
            }
            effect_chain_automation_menu(ui, track_effects, track_effect_slots, track_index, |slot_index, key| {
                automation.push(AutomationLane {
                    target: AutomationTarget::EffectParam { slot_index, key },
                    points: Vec::new(),
                });
            });
            if other_tracks.len() > 1 {
                ui.menu_button("Other Track", |ui| {
                    for (other_index, (name, effects)) in other_tracks.iter().enumerate() {
                        if other_index == track_index {
                            continue;
                        }
                        ui.menu_button(name, |ui| {
                            if ui.button("Volume").clicked() {
                                automation.push(AutomationLane {
                                    target: AutomationTarget::OtherTrackVolume { track_index: other_index },
                                    points: Vec::new(),
                                });
                                ui.close();
                            }
                            if ui.button("Pan").clicked() {
                                automation.push(AutomationLane {
                                    target: AutomationTarget::OtherTrackPan { track_index: other_index },
                                    points: Vec::new(),
                                });
                                ui.close();
                            }
                            if !sends.is_empty() {
                                ui.menu_button("Send Level", |ui| {
                                    for (send_index, send) in sends.iter().enumerate() {
                                        if ui.button(&send.name).clicked() {
                                            automation.push(AutomationLane {
                                                target: AutomationTarget::OtherTrackSendLevel {
                                                    track_index: other_index,
                                                    send_index,
                                                },
                                                points: Vec::new(),
                                            });
                                            ui.close();
                                        }
                                    }
                                });
                            }
                            effect_chain_automation_menu(ui, effects, track_effect_slots, other_index, |slot_index, key| {
                                automation.push(AutomationLane {
                                    target: AutomationTarget::OtherTrackEffectParam {
                                        track_index: other_index,
                                        slot_index,
                                        key,
                                    },
                                    points: Vec::new(),
                                });
                            });
                        });
                    }
                });
            }
            if !sends.is_empty() {
                ui.menu_button("Send FX", |ui| {
                    for (send_index, send) in sends.iter().enumerate() {
                        ui.menu_button(&send.name, |ui| {
                            effect_chain_automation_menu(
                                ui,
                                &send.effects,
                                send_effect_slots,
                                send_index,
                                |slot_index, key| {
                                    automation.push(AutomationLane {
                                        target: AutomationTarget::SendEffectParam { send_index, slot_index, key },
                                        points: Vec::new(),
                                    });
                                },
                            );
                        });
                    }
                });
            }
            if !master_effects.is_empty() {
                ui.menu_button("Master FX", |ui| {
                    effect_chain_automation_menu(ui, master_effects, master_effect_slots, 0, |slot_index, key| {
                        automation.push(AutomationLane {
                            target: AutomationTarget::MasterEffectParam { slot_index, key },
                            points: Vec::new(),
                        });
                    });
                });
            }
        });
    });

    if automation.is_empty() {
        ui.weak("No automation lanes yet.");
        return;
    }

    let mut lane_to_remove = None;
    for lane_index in 0..automation.len() {
        let label = automation_target_label(
            &automation[lane_index].target,
            track_effects,
            other_tracks,
            sends,
            master_effects,
        );
        let value_range = automation_target_range(
            &automation[lane_index].target,
            track_effects,
            track_effect_slots,
            track_index,
            other_tracks,
            sends,
            send_effect_slots,
            master_effects,
            master_effect_slots,
        );
        ui.horizontal(|ui| {
            ui.label(&label);
            if ui.small_button("🗑").on_hover_text("Remove lane").clicked() {
                lane_to_remove = Some(lane_index);
            }
        });
        automation_lane_graph_ui(
            ui,
            &mut automation[lane_index],
            lane_index,
            span_ticks,
            value_range,
            zoom,
            drag,
        );
    }
    if let Some(index) = lane_to_remove {
        automation.remove(index);
        if drag.as_ref().is_some_and(|d| d.lane_index == index) {
            *drag = None;
        }
    }
}
