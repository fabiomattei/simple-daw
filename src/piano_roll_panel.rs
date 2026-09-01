//! The Piano Roll window: free note editing (draw/move/resize/select notes on a continuous time
//! axis, unlike the Beats window's fixed step grid) plus its own Quantize/Humanize/Groove
//! Template toolbar and velocity lane. `PianoRollPanelUi` bundles the mutable app-state borrows
//! shared by the docked and detached-window renderings (see `SimpleDawApp::update`);
//! `piano_roll_contents_ui` is the entry point. `draw_region_note_preview` (a small note-hit
//! preview drawn inside a Playlist region block) is also here since it's piano-roll/step-grid
//! rendering logic, even though its only caller is the Playlist.

use std::collections::HashSet;

use crate::groove::{self, GROOVE_TEMPLATES};
use crate::model::{
    self, Note, Region, RegionContent, SessionClipContent, Song, TICKS_PER_STEP, TrackEffectConfig, TrackKind,
    add_note, clear_overlaps, find_note_mut, remove_note,
};
use crate::plugin_host::{MasterEffectSlots, SendEffectSlots, TrackEffectSlots};
use crate::{
    AutomationDrag, FL_ACCENT_GREEN, FL_ACCENT_ORANGE, RESIZE_HANDLE_PX, RegionEditTarget, audio,
    automation_lanes_ui, note_name, pitch_class_name, tick_to_x, track_color, x_to_tick,
};

/// Piano-roll pitch range: the full MIDI note range, since a melodic part can
/// use any pitch. The canvas is taller than any screen at this range, so the
/// note grid (not the velocity lane) scrolls vertically — see `piano_roll_ui`.
const PIANO_ROLL_LOW: u8 = 0;
const PIANO_ROLL_HIGH: u8 = 127;
/// Old default range's center (was 28..=48), used to pick a sensible initial
/// vertical scroll position instead of dropping the user at MIDI note 127.
const PIANO_ROLL_DEFAULT_CENTER_PITCH: u8 = 38;

const ROW_HEIGHT: f32 = 15.0;
const KEY_LABEL_WIDTH: f32 = 42.0;
const VELOCITY_LANE_HEIGHT: f32 = 46.0;

/// Piano-roll zoom range: how far the "Zoom" slider can shrink/enlarge the
/// grid, applied uniformly to both axes (see `tick_to_x`/`row_height`).
const PIANO_ROLL_ZOOM_MIN: f32 = 0.25;
const PIANO_ROLL_ZOOM_MAX: f32 = 3.0;
/// Floor for the central panel's piano-roll note-grid viewport height, so a
/// very small window doesn't collapse it to nothing.
const PIANO_ROLL_HEIGHT_MIN: f32 = 120.0;
/// Height of the piano roll's bar/beat ruler row above the note grid, mirroring
/// `PLAYLIST_RULER_HEIGHT`.
const PIANO_ROLL_RULER_HEIGHT: f32 = 20.0;

fn row_height(zoom: f32) -> f32 {
    ROW_HEIGHT * zoom
}

fn y_to_pitch(y: f32, zoom: f32) -> u8 {
    let row = (y / row_height(zoom)).floor() as i32;
    (PIANO_ROLL_HIGH as i32 - row).clamp(PIANO_ROLL_LOW as i32, PIANO_ROLL_HIGH as i32) as u8
}

/// Linearly blends `base` toward `tint` by `t` (0=`base`, 1=`tint`) — used to highlight in-scale
/// piano-roll rows with the region's own color rather than a fixed accent.
fn blend_color(base: egui::Color32, tint: egui::Color32, t: f32) -> egui::Color32 {
    let mix = |a: u8, b: u8| (a as f32 + (b as f32 - a as f32) * t).round() as u8;
    egui::Color32::from_rgb(mix(base.r(), tint.r()), mix(base.g(), tint.g()), mix(base.b(), tint.b()))
}

/// A modal or pentatonic scale the piano roll can highlight in its row background as a visual
/// composing aid (see `piano_roll_ui`'s left panel and `PianoRollScale::contains`). Purely a UI
/// overlay — it never restricts where notes can be placed, so it isn't part of `model::Song`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PianoRollScale {
    Off,
    Major,
    Dorian,
    Phrygian,
    Lydian,
    Mixolydian,
    Minor,
    Locrian,
    MajorPentatonic,
    MinorPentatonic,
}

impl PianoRollScale {
    const ALL: [PianoRollScale; 10] = [
        PianoRollScale::Off,
        PianoRollScale::Major,
        PianoRollScale::Dorian,
        PianoRollScale::Phrygian,
        PianoRollScale::Lydian,
        PianoRollScale::Mixolydian,
        PianoRollScale::Minor,
        PianoRollScale::Locrian,
        PianoRollScale::MajorPentatonic,
        PianoRollScale::MinorPentatonic,
    ];

    fn label(self) -> &'static str {
        match self {
            PianoRollScale::Off => "Off",
            PianoRollScale::Major => "Major (Ionian)",
            PianoRollScale::Dorian => "Dorian",
            PianoRollScale::Phrygian => "Phrygian",
            PianoRollScale::Lydian => "Lydian",
            PianoRollScale::Mixolydian => "Mixolydian",
            PianoRollScale::Minor => "Minor (Aeolian)",
            PianoRollScale::Locrian => "Locrian",
            PianoRollScale::MajorPentatonic => "Major Pentatonic",
            PianoRollScale::MinorPentatonic => "Minor Pentatonic",
        }
    }

    /// Semitone offsets from the root, within one octave.
    fn intervals(self) -> &'static [u8] {
        match self {
            PianoRollScale::Off => &[],
            PianoRollScale::Major => &[0, 2, 4, 5, 7, 9, 11],
            PianoRollScale::Dorian => &[0, 2, 3, 5, 7, 9, 10],
            PianoRollScale::Phrygian => &[0, 1, 3, 5, 7, 8, 10],
            PianoRollScale::Lydian => &[0, 2, 4, 6, 7, 9, 11],
            PianoRollScale::Mixolydian => &[0, 2, 4, 5, 7, 9, 10],
            PianoRollScale::Minor => &[0, 2, 3, 5, 7, 8, 10],
            PianoRollScale::Locrian => &[0, 1, 3, 5, 6, 8, 10],
            PianoRollScale::MajorPentatonic => &[0, 2, 4, 7, 9],
            PianoRollScale::MinorPentatonic => &[0, 3, 5, 7, 10],
        }
    }

    /// Whether `pitch` belongs to this scale rooted at `root` (a pitch class, 0=C..11=B).
    /// `Off` treats every pitch as in-scale, since it means "no highlight" rather than "empty
    /// scale" — callers branch on `Off` separately before using this for coloring anyway.
    fn contains(self, root: u8, pitch: u8) -> bool {
        if self == PianoRollScale::Off {
            return true;
        }
        let offset = (pitch % 12 + 12 - root % 12) % 12;
        self.intervals().contains(&offset)
    }
}

/// What the currently in-progress piano-roll drag (if any) is doing.
enum PianoRollDragMode {
    /// Dragging one note's body (it's unselected, or the only selected note):
    /// changes start_tick and/or pitch.
    Move { note_id: u64, grab_tick_offset: i64 },
    /// Dragging the whole multi-note selection together as a rigid group.
    /// `origin` snapshots every selected note's (id, start_tick, pitch) at
    /// drag start, so each frame's new positions are computed as a delta
    /// from that fixed baseline rather than accumulated incrementally.
    MoveSelection {
        anchor_id: u64,
        grab_tick_offset: i64,
        start_pitch: i32,
        origin: Vec<(u64, usize, u8)>,
    },
    /// Dragging an existing note's right edge: changes length_ticks only.
    Resize { note_id: u64 },
    /// Drawing a brand-new note out from a click on empty space.
    Create {
        note_id: u64,
        start_tick: usize,
        pitch: u8,
    },
    /// Dragging a bar in the velocity lane.
    Velocity { note_id: u64 },
    /// Rubber-band-dragging a selection rectangle out from empty space,
    /// started with Shift held so it doesn't collide with the plain
    /// click-drag gesture that draws a new note. `start_local` is the
    /// canvas-local pixel position the drag began at.
    BoxSelect { start_local: egui::Pos2 },
}

pub(crate) struct PianoRollDrag {
    mode: PianoRollDragMode,
}

/// Bundles the Piano Roll's mutable app-state borrows for the same reason as `ChannelRackUi`.
pub(crate) struct PianoRollPanelUi<'a> {
    /// See `SimpleDawApp::piano_roll_detached`.
    pub(crate) detached: &'a mut bool,
    pub(crate) selected_track: Option<usize>,
    pub(crate) piano_roll_drag: &'a mut Option<PianoRollDrag>,
    pub(crate) selected_notes: &'a mut HashSet<u64>,
    /// See `SimpleDawApp::groove_quantize_grid_ticks` and its sibling `groove_*` fields — bundled
    /// into one borrow the same way the other toolbar controls above are.
    pub(crate) groove_quantize_grid_ticks: &'a mut usize,
    pub(crate) groove_quantize_strength: &'a mut f32,
    pub(crate) groove_humanize_timing_ticks: &'a mut usize,
    pub(crate) groove_humanize_velocity: &'a mut u8,
    pub(crate) groove_template_index: &'a mut usize,
    pub(crate) piano_roll_zoom: &'a mut f32,
    /// See `SimpleDawApp::piano_roll_scale_root`.
    pub(crate) scale_root: &'a mut u8,
    /// See `SimpleDawApp::piano_roll_scale`.
    pub(crate) scale: &'a mut PianoRollScale,
    /// See `SimpleDawApp::piano_roll_region`/`RegionEditTarget`.
    pub(crate) editing_target: &'a mut Option<RegionEditTarget>,
    /// See `SimpleDawApp::piano_roll_scroll_to`.
    pub(crate) scroll_to: &'a mut Option<usize>,
    /// The open region's own track's live effect chain — read by `automation_lanes_ui`'s "+ Add
    /// Lane" menu to offer a currently-loaded CLAP plugin's real parameter names.
    pub(crate) track_effect_slots: &'a TrackEffectSlots,
    /// Every send bus's and the master bus's live effect chains — same reason as
    /// `track_effect_slots`, for `automation_lanes_ui`'s "Send FX"/"Master FX" cross-bus targets.
    pub(crate) send_effect_slots: &'a SendEffectSlots,
    pub(crate) master_effect_slots: &'a MasterEffectSlots,
    /// See `AutomationDrag`.
    pub(crate) automation_drag: &'a mut Option<AutomationDrag>,
    /// See `SimpleDawApp::track_automation_drag`.
    pub(crate) track_automation_drag: &'a mut Option<AutomationDrag>,
}

/// Grid choices for the Piano Roll's Quantize/Groove Template toolbar: (label, ticks-per-cell).
/// `TICKS_PER_STEP` is one 16th note, so these are simple multiples/fractions of it.
const QUANTIZE_GRID_CHOICES: &[(&str, usize)] = &[
    ("1/4", TICKS_PER_STEP * 4),
    ("1/8", TICKS_PER_STEP * 2),
    ("1/16", TICKS_PER_STEP),
    ("1/32", TICKS_PER_STEP / 2),
    ("1/16 triplet", TICKS_PER_STEP * 2 / 3),
];

/// Quantize/Humanize/Groove Template controls for the open piano-roll region, shown just above
/// the note grid. Each action targets `panel.selected_notes` if any are selected, else every note
/// in `notes` — see `groove::quantize_notes`/`humanize_notes`/`apply_groove_template`.
fn piano_roll_quantize_humanize_groove_ui(
    ui: &mut egui::Ui,
    notes: &mut Vec<Note>,
    panel: &mut PianoRollPanelUi,
) {
    let selection = |panel: &PianoRollPanelUi| {
        (!panel.selected_notes.is_empty()).then(|| panel.selected_notes.clone())
    };
    ui.horizontal(|ui| {
        ui.label("Grid");
        let grid_label = QUANTIZE_GRID_CHOICES
            .iter()
            .find(|(_, ticks)| *ticks == *panel.groove_quantize_grid_ticks)
            .map(|(label, _)| *label)
            .unwrap_or("custom");
        egui::ComboBox::from_id_salt("quantize_grid").selected_text(grid_label).show_ui(ui, |ui| {
            for (label, ticks) in QUANTIZE_GRID_CHOICES {
                ui.selectable_value(panel.groove_quantize_grid_ticks, *ticks, *label);
            }
        });
        ui.add(egui::Slider::new(panel.groove_quantize_strength, 0.0..=1.0).text("Strength"));
        if ui
            .button("Quantize")
            .on_hover_text("Snap selected notes (or all, if none selected) to the grid")
            .clicked()
        {
            let selection = selection(panel);
            groove::quantize_notes(
                notes,
                selection.as_ref(),
                *panel.groove_quantize_grid_ticks,
                *panel.groove_quantize_strength,
            );
        }
    });
    ui.horizontal(|ui| {
        ui.label("Humanize");
        ui.add(egui::Slider::new(panel.groove_humanize_timing_ticks, 0..=24).text("Timing"));
        ui.add(egui::Slider::new(panel.groove_humanize_velocity, 0..=40).text("Velocity"));
        if ui
            .button("Apply")
            .on_hover_text("Randomly nudge selected notes' (or all, if none selected) timing/velocity")
            .clicked()
        {
            let selection = selection(panel);
            let seed = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0);
            groove::humanize_notes(
                notes,
                selection.as_ref(),
                *panel.groove_humanize_timing_ticks,
                *panel.groove_humanize_velocity,
                seed,
            );
        }
    });
    ui.horizontal(|ui| {
        ui.label("Groove Template");
        egui::ComboBox::from_id_salt("groove_template")
            .selected_text(GROOVE_TEMPLATES[*panel.groove_template_index].name)
            .show_ui(ui, |ui| {
                for (index, template) in GROOVE_TEMPLATES.iter().enumerate() {
                    ui.selectable_value(panel.groove_template_index, index, template.name);
                }
            });
        if ui
            .button("Apply")
            .on_hover_text("Snap selected notes (or all, if none selected) to the grid, then apply this template's swing/accent")
            .clicked()
        {
            let selection = selection(panel);
            groove::apply_groove_template(
                notes,
                selection.as_ref(),
                *panel.groove_quantize_grid_ticks,
                &GROOVE_TEMPLATES[*panel.groove_template_index],
            );
        }
    });
    ui.separator();
}

/// The Piano Roll's header (selected track name/mute badge, dock/detach toggle) and note grid,
/// rendered either inside its own OS window or docked into the central area (see
/// `piano_roll_detached`/`piano_roll_docked` in `ui`, `impl eframe::App for SimpleDawApp`) — same
/// dock/detach split as the Channel Rack, except it only exists when a piano-roll track is
/// selected, and closing its window clears the selection instead of leaving it docked-but-empty.
/// There's no picker here to switch regions — double-click a different region in the Playlist
/// instead (see `PlaylistEditorTargets`).
pub(crate) fn piano_roll_contents_ui(
    ui: &mut egui::Ui,
    song: &mut Song,
    current_tick: Option<usize>,
    panel: &mut PianoRollPanelUi,
) {
    let selected = panel
        .selected_track
        .filter(|&i| i < song.tracks.len())
        .filter(|&i| song.tracks[i].kind == TrackKind::PianoRoll);
    // Resolves `panel.editing_target` against `selected`'s track, validating the target still
    // exists (a region/slot can be deleted out from under an open editor) — and, for a session
    // slot, that its content is still `PianoRoll`-shaped (see `RegionEditTarget`'s doc comment
    // on why a session slot has to be checked this way rather than just indexed).
    let region = selected.and_then(|index| match (*panel.editing_target)? {
        RegionEditTarget::Region(region_index) => (region_index
            < song.tracks[index].regions.len())
        .then_some((index, RegionEditTarget::Region(region_index))),
        RegionEditTarget::SessionSlot(slot_index) => song.tracks[index]
            .session_clips
            .get(slot_index)
            .and_then(|slot| slot.as_ref())
            .is_some_and(|clip| {
                matches!(
                    clip.content,
                    SessionClipContent::Region { content: RegionContent::PianoRoll(_), .. }
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
            ui.heading("Piano Roll");
        }
    });
    if ui
        .small_button(if *panel.detached { "⏷ Dock" } else { "⧉ Detach" })
        .clicked()
    {
        *panel.detached = !*panel.detached;
    }
    ui.horizontal(|ui| {
        ui.label("Zoom");
        ui.add(
            egui::Slider::new(panel.piano_roll_zoom, PIANO_ROLL_ZOOM_MIN..=PIANO_ROLL_ZOOM_MAX)
                .fixed_decimals(2)
                .suffix("x"),
        );
        if ui.small_button("Reset").clicked() {
            *panel.piano_roll_zoom = 1.0;
        }
    });
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
                        panel.track_effect_slots,
                        &other_tracks_snapshot,
                        &song.sends,
                        panel.send_effect_slots,
                        &song.master_effects,
                        panel.master_effect_slots,
                        *panel.piano_roll_zoom,
                        panel.track_automation_drag,
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
            // Reserve room below the note grid for the automation panel (header + roughly one
            // lane's graph; more lanes than that scroll within their own area instead of pushing
            // the note grid further up).
            let automation_reserved = 110.0;
            let visible_height =
                (ui.available_height() - automation_reserved).max(PIANO_ROLL_HEIGHT_MIN);
            let steps_per_bar = song.steps_per_bar();
            let steps_per_beat = song.steps_per_beat();
            // Snapshot every track's name/effects *before* borrowing `song.tracks[index]` below —
            // `automation_lanes_ui`'s "Other Track" targets need to list every track, which would
            // otherwise alias the same `song.tracks` field the open region's own mutable borrow
            // comes from.
            let other_tracks_snapshot: Vec<(String, Vec<TrackEffectConfig>)> = song
                .tracks
                .iter()
                .map(|t| (t.name.clone(), t.effects.clone()))
                .collect();
            let next_note_id = &mut song.next_note_id;
            let track = &mut song.tracks[index];
            let track_effects_snapshot = track.effects.clone();
            let default_note_length_ticks = &mut track.default_note_length_ticks;
            let region = &mut track.regions[region_index];
            if let RegionContent::PianoRoll(notes) = &mut region.content {
                piano_roll_quantize_humanize_groove_ui(ui, notes, panel);
                piano_roll_ui(
                    ui,
                    notes,
                    next_note_id,
                    default_note_length_ticks,
                    &mut region.content_length_steps,
                    current_tick,
                    panel.piano_roll_drag,
                    panel.selected_notes,
                    *panel.piano_roll_zoom,
                    visible_height,
                    color,
                    panel.scroll_to,
                    steps_per_bar,
                    steps_per_beat,
                    panel.scale_root,
                    panel.scale,
                );
            }
            ui.separator();
            let region_span_ticks = region.loop_length_steps * TICKS_PER_STEP;
            egui::ScrollArea::vertical().max_height(automation_reserved.max(60.0)).show(
                ui,
                |ui| {
                    automation_lanes_ui(
                        ui,
                        &mut region.automation,
                        region_span_ticks,
                        index,
                        &track_effects_snapshot,
                        panel.track_effect_slots,
                        &other_tracks_snapshot,
                        &song.sends,
                        panel.send_effect_slots,
                        &song.master_effects,
                        panel.master_effect_slots,
                        *panel.piano_roll_zoom,
                        panel.automation_drag,
                    );
                },
            );
        }
        Some((index, RegionEditTarget::SessionSlot(slot_index))) => {
            let color = track_color(index);
            // Same reservation as the `Region` arm above, now that a session slot gets its own
            // automation panel too (see `SessionClip::automation`'s doc comment).
            let automation_reserved = 110.0;
            let visible_height =
                (ui.available_height() - automation_reserved).max(PIANO_ROLL_HEIGHT_MIN);
            let steps_per_bar = song.steps_per_bar();
            let steps_per_beat = song.steps_per_beat();
            // Same pre-borrow snapshot as the `Region` arm above, for the same reason.
            let other_tracks_snapshot: Vec<(String, Vec<TrackEffectConfig>)> = song
                .tracks
                .iter()
                .map(|t| (t.name.clone(), t.effects.clone()))
                .collect();
            let next_note_id = &mut song.next_note_id;
            let track = &mut song.tracks[index];
            let track_effects_snapshot = track.effects.clone();
            let default_note_length_ticks = &mut track.default_note_length_ticks;
            let Some(Some(clip)) = track.session_clips.get_mut(slot_index) else {
                ui.weak("Clip no longer exists.");
                return;
            };
            if let SessionClipContent::Region { content, content_length_steps, .. } = &mut clip.content
                && let RegionContent::PianoRoll(notes) = content
            {
                piano_roll_quantize_humanize_groove_ui(ui, notes, panel);
                piano_roll_ui(
                    ui,
                    notes,
                    next_note_id,
                    default_note_length_ticks,
                    content_length_steps,
                    current_tick,
                    panel.piano_roll_drag,
                    panel.selected_notes,
                    *panel.piano_roll_zoom,
                    visible_height,
                    color,
                    panel.scroll_to,
                    steps_per_bar,
                    steps_per_beat,
                    panel.scale_root,
                    panel.scale,
                );
            }
            ui.separator();
            let clip_span_ticks = match &clip.content {
                SessionClipContent::Region { loop_length_steps, .. } => {
                    loop_length_steps * TICKS_PER_STEP
                }
                SessionClipContent::Audio(_) | SessionClipContent::Recording(_) => 0,
            };
            egui::ScrollArea::vertical().max_height(automation_reserved.max(60.0)).show(
                ui,
                |ui| {
                    automation_lanes_ui(
                        ui,
                        &mut clip.automation,
                        clip_span_ticks,
                        index,
                        &track_effects_snapshot,
                        panel.track_effect_slots,
                        &other_tracks_snapshot,
                        &song.sends,
                        panel.send_effect_slots,
                        &song.master_effects,
                        panel.master_effect_slots,
                        *panel.piano_roll_zoom,
                        panel.automation_drag,
                    );
                },
            );
        }
    }
}

/// A proper piano roll: notes are drawn and edited freely (click-drag to draw,
/// drag a note's body to move it, drag its right edge to resize it, right-click
/// to delete), rather than toggling cells on a fixed 16-step grid. Velocity is
/// edited in the lane below (`velocity_lane_ui`), synced to the same time axis.
fn piano_roll_ui(
    ui: &mut egui::Ui,
    notes: &mut Vec<Note>,
    next_note_id: &mut u64,
    default_note_length_ticks: &mut usize,
    length_steps: &mut usize,
    current_tick: Option<usize>,
    drag: &mut Option<PianoRollDrag>,
    selected: &mut HashSet<u64>,
    zoom: f32,
    visible_height: f32,
    note_color: egui::Color32,
    scroll_to: &mut Option<usize>,
    steps_per_bar: usize,
    steps_per_beat: usize,
    scale_root: &mut u8,
    scale: &mut PianoRollScale,
) {
    // The declared pattern length always covers the furthest note (used for playback
    // looping/export); the visible/clickable canvas adds one more empty bar past that
    // so there's always room to click in a new note further out, which then grows the
    // declared length again next frame.
    model::grow_length_to_fit_notes(length_steps, notes, steps_per_bar);
    let display_steps = *length_steps + steps_per_bar;
    let length_ticks_total = (display_steps * TICKS_PER_STEP).max(1);
    let canvas_width = tick_to_x(length_ticks_total, zoom);
    let row_h = row_height(zoom);
    let canvas_height = (PIANO_ROLL_HIGH - PIANO_ROLL_LOW + 1) as f32 * row_h;

    // The note grid is much taller than the old fixed 1.7-octave range, so it scrolls
    // vertically on its own. The first time this runs for a given piano roll, jump to
    // the old default range's center rather than dropping the user at the very top of
    // the MIDI range.
    //
    // The grid needs to scroll both ways (vertically through pitches, horizontally
    // through time) while the key-label column to its left only follows the vertical
    // axis (frozen horizontally, so pitch names stay put as the music scrolls by) and
    // the velocity lane below only follows the horizontal axis (frozen vertically, so
    // it's always visible rather than buried under the pitch range). egui has no
    // built-in "frozen row/column" scroll area, so this is three separate `ScrollArea`s
    // — the grid is the one the user actually drags/wheels over, and the key column +
    // velocity lane are forced to mirror its offset every frame (one frame of lag for
    // the key column, since it's rendered before the grid reports its fresh offset;
    // zero lag for the velocity lane, rendered after).
    let scroll_offset_id = ui.id().with("piano-roll-grid-offset");
    let known_offset = ui
        .ctx()
        .data(|d| d.get_temp::<egui::Vec2>(scroll_offset_id))
        .unwrap_or_default();

    let centered_id = ui.id().with("piano-roll-vscroll-centered");
    let already_centered = ui
        .ctx()
        .data(|d| d.get_temp::<bool>(centered_id))
        .unwrap_or(false);
    let initial_offset_y = if already_centered {
        known_offset.y
    } else {
        let visible_rows = visible_height / row_h;
        let center_row = (PIANO_ROLL_HIGH - PIANO_ROLL_DEFAULT_CENTER_PITCH) as f32;
        let offset_y = ((center_row - visible_rows / 2.0) * row_h).max(0.0);
        ui.ctx().data_mut(|d| d.insert_temp(centered_id, true));
        offset_y
    };

    // While playing, keep the moving playhead in view: if it's about to run off the
    // right edge of the visible area (or isn't visible at all), jump the horizontal
    // scroll forward so it reappears near the left with room to see what's coming.
    // Only forces a scroll when actually needed, so manual scrolling while paused
    // (or while the playhead is already on-screen) is left alone.
    let mut grid_hscroll = egui::ScrollArea::horizontal().id_salt("piano-roll-grid-hscroll");
    let grid_vscroll = egui::ScrollArea::vertical()
        .id_salt("piano-roll-grid-vscroll")
        .max_height(visible_height)
        .vertical_scroll_offset(initial_offset_y);
    let playhead_x = current_tick.map(|tick| tick_to_x(tick % length_ticks_total, zoom));
    let margin = 60.0;
    // A pending scroll request (a Playlist double-click landing partway through a looped
    // region) wins over the playhead-follow logic below and is consumed immediately, so it
    // only ever repositions the view once rather than fighting manual scrolling afterward.
    if let Some(tick) = scroll_to.take() {
        let target_x = tick_to_x(tick % length_ticks_total, zoom);
        grid_hscroll = grid_hscroll.horizontal_scroll_offset((target_x - margin).max(0.0));
    } else if let Some(playhead_x) = playhead_x {
        let viewport_width = ui.available_width();
        if playhead_x < known_offset.x + margin
            || playhead_x > known_offset.x + viewport_width - margin
        {
            grid_hscroll = grid_hscroll.horizontal_scroll_offset((playhead_x - margin).max(0.0));
        }
    }

    let keys_scroll = egui::ScrollArea::vertical()
        .id_salt("piano-roll-keys-scroll")
        .max_height(visible_height)
        .max_width(KEY_LABEL_WIDTH)
        .vertical_scroll_offset(initial_offset_y)
        .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysHidden);

    ui.horizontal(|ui| {
        ui.vertical(|ui| {
            ui.set_width(96.0);
            ui.weak("ℹ").on_hover_text("Click empty space for a quick note; drag to draw one of any length. Drag a note's edge to resize, its body to move. Click a note to select it, Ctrl/Cmd-click to add/remove one from the selection, Shift-drag empty space to box-select. Drag any selected note to move the whole selection together. Right-click or Delete/Backspace removes the selection (or just the note under the cursor if nothing's selected). The roll grows automatically as notes reach its right edge.");
            ui.weak("Length:");
            for (label, steps) in [("1/16", 1usize), ("1/8", 2), ("1/4", 4), ("1/2", 8)] {
                let ticks = steps * TICKS_PER_STEP;
                if ui
                    .selectable_label(*default_note_length_ticks == ticks, label)
                    .clicked()
                {
                    *default_note_length_ticks = ticks;
                }
            }
            ui.add_space(6.0);
            ui.weak("Scale:")
                .on_hover_text("Highlights in-scale rows in the piano roll's background as a visual guide. Doesn't restrict note placement.");
            egui::ComboBox::from_id_salt("piano-roll-scale-root")
                .selected_text(pitch_class_name(*scale_root))
                .show_ui(ui, |ui| {
                    for pitch_class in 0u8..12 {
                        ui.selectable_value(scale_root, pitch_class, pitch_class_name(pitch_class));
                    }
                });
            egui::ComboBox::from_id_salt("piano-roll-scale-kind")
                .selected_text(scale.label())
                .show_ui(ui, |ui| {
                    for &option in PianoRollScale::ALL.iter() {
                        ui.selectable_value(scale, option, option.label());
                    }
                });
        });
        ui.separator();
        ui.vertical(|ui| {
            let mut grid_offset = known_offset;

            // Bar/beat ruler above the grid: frozen vertically (it never scrolls with the
            // pitch range) but follows the grid's horizontal scroll. Rendered before the
            // grid below, so — like the key column — it trails the grid's horizontal
            // offset by one frame (`known_offset.x` rather than this frame's fresh value,
            // which isn't available until the grid itself has been shown).
            ui.horizontal(|ui| {
                ui.add_space(KEY_LABEL_WIDTH);
                egui::ScrollArea::horizontal()
                    .id_salt("piano-roll-ruler-scroll")
                    .horizontal_scroll_offset(known_offset.x)
                    .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysHidden)
                    .show(ui, |ui| {
                        let (response, painter) = ui.allocate_painter(
                            egui::vec2(canvas_width, PIANO_ROLL_RULER_HEIGHT),
                            egui::Sense::hover(),
                        );
                        let rect = response.rect;
                        painter.rect_filled(rect, 0u8, ui.visuals().extreme_bg_color);
                        painter.line_segment(
                            [rect.left_bottom(), rect.right_bottom()],
                            egui::Stroke::new(1.0, ui.visuals().weak_text_color()),
                        );
                        for step in 0..=display_steps {
                            let x = rect.left() + tick_to_x(step * TICKS_PER_STEP, zoom);
                            let is_bar = step % steps_per_bar == 0;
                            let stroke = if is_bar {
                                egui::Stroke::new(1.5, ui.visuals().text_color())
                            } else if step % steps_per_beat == 0 {
                                egui::Stroke::new(1.0, ui.visuals().weak_text_color())
                            } else {
                                continue;
                            };
                            let tick_top = if is_bar {
                                rect.top() + 4.0
                            } else {
                                rect.top() + PIANO_ROLL_RULER_HEIGHT * 0.6
                            };
                            painter.line_segment(
                                [egui::pos2(x, tick_top), egui::pos2(x, rect.bottom())],
                                stroke,
                            );
                            if is_bar {
                                painter.text(
                                    egui::pos2(x + 3.0, rect.top() + 2.0),
                                    egui::Align2::LEFT_TOP,
                                    format!("{}", step / steps_per_bar + 1),
                                    egui::FontId::proportional(10.0),
                                    ui.visuals().text_color(),
                                );
                            }
                        }
                        if let Some(playhead_x) = playhead_x {
                            let x = rect.left() + playhead_x;
                            painter.line_segment(
                                [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
                                egui::Stroke::new(2.0, FL_ACCENT_GREEN),
                            );
                        }
                    });
            });

            ui.horizontal(|ui| {
                // A `ScrollArea` sizes itself from `ui.available_rect_before_wrap()`, which
                // for the *first* child in a `ui.horizontal` defaults to a single default
                // row's height (it only grows once children report their real size — too
                // late for a `ScrollArea` already committed to that first small guess). Pin
                // the row's height up front so both scroll areas below get the full budget.
                ui.set_height(visible_height);
                keys_scroll.show(ui, |ui| {
                    // Painted directly (rather than a `Label` per row) so each row can get a
                    // black/white piano-key-shaped background instead of plain text on the
                    // panel's flat background — a small FL Studio–style touch.
                    let (response, painter) = ui.allocate_painter(
                        egui::vec2(KEY_LABEL_WIDTH, canvas_height),
                        egui::Sense::hover(),
                    );
                    let rect = response.rect;
                    for (row, pitch) in (PIANO_ROLL_LOW..=PIANO_ROLL_HIGH).rev().enumerate() {
                        let y = rect.top() + row as f32 * row_h;
                        let key_rect = egui::Rect::from_min_size(
                            egui::pos2(rect.left(), y),
                            egui::vec2(KEY_LABEL_WIDTH, row_h),
                        );
                        let is_black_key = matches!(pitch % 12, 1 | 3 | 6 | 8 | 10);
                        let (bg, text_color) = if is_black_key {
                            (egui::Color32::from_rgb(22, 22, 22), egui::Color32::from_rgb(190, 190, 190))
                        } else {
                            (egui::Color32::from_rgb(210, 210, 206), egui::Color32::from_rgb(30, 30, 30))
                        };
                        painter.rect_filled(key_rect, 0u8, bg);
                        painter.line_segment(
                            [key_rect.left_bottom(), key_rect.right_bottom()],
                            egui::Stroke::new(0.5, egui::Color32::from_black_alpha(120)),
                        );
                        painter.text(
                            key_rect.right_center() - egui::vec2(4.0, 0.0),
                            egui::Align2::RIGHT_CENTER,
                            note_name(pitch),
                            egui::FontId::proportional(9.0),
                            text_color,
                        );
                    }
                });

                let vscroll_output = grid_vscroll.show(ui, |ui| {
                    grid_hscroll.show(ui, |ui| {
                        let (response, painter) = ui.allocate_painter(
                            egui::vec2(canvas_width, canvas_height),
                            egui::Sense::click_and_drag(),
                        );
                        let rect = response.rect;

                        for (row, pitch) in (PIANO_ROLL_LOW..=PIANO_ROLL_HIGH).rev().enumerate() {
                            let y = rect.top() + row as f32 * row_h;
                            let row_rect = egui::Rect::from_min_size(
                                egui::pos2(rect.left(), y),
                                egui::vec2(canvas_width, row_h),
                            );
                            let is_black_key = matches!(pitch % 12, 1 | 3 | 6 | 8 | 10);
                            let bg = if is_black_key {
                                ui.visuals().faint_bg_color
                            } else {
                                ui.visuals().extreme_bg_color
                            };
                            // Off means "no highlight": leave the plain black/white key coloring
                            // alone. Otherwise tint in-scale rows toward the region's own color
                            // and dim out-of-scale ones, so the scale reads at a glance.
                            let bg = match *scale {
                                PianoRollScale::Off => bg,
                                s if s.contains(*scale_root, pitch) => {
                                    blend_color(bg, note_color, 0.3)
                                }
                                _ => bg.gamma_multiply(0.5),
                            };
                            painter.rect_filled(row_rect, 0u8, bg);
                        }

                        for step in 0..=display_steps {
                            let x = rect.left() + tick_to_x(step * TICKS_PER_STEP, zoom);
                            let stroke = if step % steps_per_bar == 0 {
                                egui::Stroke::new(1.5, ui.visuals().text_color())
                            } else if step % steps_per_beat == 0 {
                                egui::Stroke::new(1.0, ui.visuals().weak_text_color())
                            } else {
                                egui::Stroke::new(0.5, ui.visuals().faint_bg_color)
                            };
                            painter.line_segment(
                                [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
                                stroke,
                            );
                        }

                        if let Some(playhead_x) = playhead_x {
                            let x = rect.left() + playhead_x;
                            painter.line_segment(
                                [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
                                egui::Stroke::new(2.0, FL_ACCENT_GREEN),
                            );
                        }

                        for note in notes.iter() {
                            let x = rect.left() + tick_to_x(note.start_tick, zoom);
                            let w = tick_to_x(note.length_ticks, zoom).max(3.0);
                            let y = rect.top() + (PIANO_ROLL_HIGH - note.pitch) as f32 * row_h;
                            let note_rect = egui::Rect::from_min_size(
                                egui::pos2(x, y + 1.0),
                                egui::vec2(w, row_h - 2.0),
                            );
                            let intensity = note.velocity as f32 / 127.0;
                            let color = note_color.gamma_multiply(0.45 + 0.55 * intensity);
                            painter.rect_filled(note_rect, 2u8, color);
                            let is_selected = selected.contains(&note.id);
                            let outline = if is_selected {
                                egui::Stroke::new(2.0, egui::Color32::WHITE)
                            } else {
                                egui::Stroke::new(1.0, egui::Color32::BLACK)
                            };
                            painter.rect_stroke(note_rect, 2u8, outline, egui::StrokeKind::Inside);
                        }

                        if let Some(PianoRollDrag { mode: PianoRollDragMode::BoxSelect { start_local } }) =
                            drag.as_ref()
                        {
                            if let Some(pos) = response.interact_pointer_pos() {
                                let (lx, ly) = (pos.x - rect.left(), pos.y - rect.top());
                                let box_rect = egui::Rect::from_two_pos(
                                    egui::pos2(rect.left() + start_local.x, rect.top() + start_local.y),
                                    egui::pos2(rect.left() + lx, rect.top() + ly),
                                );
                                painter.rect_filled(
                                    box_rect,
                                    0u8,
                                    egui::Color32::from_rgba_unmultiplied(120, 170, 230, 40),
                                );
                                painter.rect_stroke(
                                    box_rect,
                                    0u8,
                                    egui::Stroke::new(1.0, egui::Color32::from_rgb(120, 170, 230)),
                                    egui::StrokeKind::Inside,
                                );
                            }
                        }

                        handle_piano_roll_interaction(
                            ui,
                            &response,
                            rect,
                            notes,
                            next_note_id,
                            selected,
                            *default_note_length_ticks,
                            length_ticks_total,
                            drag,
                            zoom,
                        );
                    })
                });
                grid_offset = egui::vec2(vscroll_output.inner.state.offset.x, vscroll_output.state.offset.y);
            });

            // The velocity lane mirrors the grid's fresh (this-frame) horizontal offset
            // computed just above, so it tracks with zero lag, but is never given a
            // vertical one — it always stays visible under the grid rather than
            // scrolling away with the pitch range.
            ui.horizontal(|ui| {
                ui.add_space(KEY_LABEL_WIDTH);
                egui::ScrollArea::horizontal()
                    .id_salt("piano-roll-velocity-scroll")
                    .horizontal_scroll_offset(grid_offset.x)
                    .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysHidden)
                    .show(ui, |ui| {
                        velocity_lane_ui(ui, notes, canvas_width, drag, selected, zoom, note_color);
                    });
            });

            ui.ctx().data_mut(|d| d.insert_temp(scroll_offset_id, grid_offset));
        });
    });
}

/// Hit-tests and applies click/drag gestures against `notes`, driven by the
/// canvas's own `Response` (so this only ever reacts to input inside the
/// piano-roll area it was drawn in). Overlap resolution (two notes on the
/// same pitch covering the same time) is deferred to `drag_stopped` rather
/// than applied every frame, so notes don't flicker away mid-drag.
///
/// `selected` holds the note ids currently selected for group move/delete.
/// Plain click replaces the selection; Ctrl/Cmd-click toggles one note in or
/// out of it; Shift-drag from empty space rubber-bands a whole rectangle of
/// notes into it. Dragging a note that's part of a >1-note selection moves
/// the whole group together; dragging any other note moves just that one
/// (and replaces the selection with it, matching typical DAW behavior).
fn handle_piano_roll_interaction(
    ui: &egui::Ui,
    response: &egui::Response,
    rect: egui::Rect,
    notes: &mut Vec<Note>,
    next_note_id: &mut u64,
    selected: &mut HashSet<u64>,
    default_length_ticks: usize,
    length_ticks_total: usize,
    drag: &mut Option<PianoRollDrag>,
    zoom: f32,
) {
    let local = |p: egui::Pos2| (p.x - rect.left(), p.y - rect.top());
    let note_at = |notes: &[Note], tick: usize, pitch: u8| {
        notes
            .iter()
            .find(|n| {
                n.pitch == pitch && tick >= n.start_tick && tick < n.start_tick + n.length_ticks
            })
            .copied()
    };
    let near_right_edge = |note: &Note, local_x: f32| {
        let end_x = tick_to_x(note.start_tick + note.length_ticks, zoom);
        (local_x - end_x).abs() <= RESIZE_HANDLE_PX
    };
    // `note_at` requires the pointer's tick to fall strictly inside the note's
    // range, which excludes the pixels right at (or just past) its right edge —
    // exactly where the resize handle needs to be detected. Look up by pitch and
    // proximity to the edge instead, independent of tick containment.
    let note_at_right_edge = |notes: &[Note], local_x: f32, pitch: u8| {
        notes
            .iter()
            .find(|n| n.pitch == pitch && near_right_edge(n, local_x))
            .copied()
    };
    let delete_selection_or =
        |notes: &mut Vec<Note>, selected: &mut HashSet<u64>, fallback_id: u64| {
            if selected.contains(&fallback_id) && selected.len() > 1 {
                for id in selected.iter().copied().collect::<Vec<_>>() {
                    remove_note(notes, id);
                }
                selected.clear();
            } else {
                remove_note(notes, fallback_id);
                selected.remove(&fallback_id);
            }
        };

    if response.secondary_clicked() {
        if let Some(pos) = response.interact_pointer_pos() {
            let (lx, ly) = local(pos);
            if let Some(hit) = note_at(notes, x_to_tick(lx, zoom), y_to_pitch(ly, zoom)) {
                delete_selection_or(notes, selected, hit.id);
            }
        }
    }

    let has_keyboard_focus_elsewhere = ui.ctx().memory(|m| m.focused().is_some());
    if !selected.is_empty() && !has_keyboard_focus_elsewhere {
        let delete_pressed =
            ui.input(|i| i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace));
        if delete_pressed {
            let ids: Vec<u64> = selected
                .iter()
                .copied()
                .filter(|id| notes.iter().any(|n| n.id == *id))
                .collect();
            for id in &ids {
                remove_note(notes, *id);
                selected.remove(id);
            }
        }
    }

    if response.clicked() {
        if let Some(pos) = response.interact_pointer_pos() {
            let (lx, ly) = local(pos);
            let tick = x_to_tick(lx, zoom).min(length_ticks_total.saturating_sub(1));
            let pitch = y_to_pitch(ly, zoom);
            let modifiers = ui.input(|i| i.modifiers);
            if let Some(hit) = note_at(notes, tick, pitch) {
                if modifiers.command {
                    if !selected.remove(&hit.id) {
                        selected.insert(hit.id);
                    }
                } else {
                    selected.clear();
                    selected.insert(hit.id);
                }
            } else if !modifiers.shift {
                selected.clear();
                add_note(
                    notes,
                    next_note_id,
                    pitch,
                    tick,
                    default_length_ticks.max(1),
                    100,
                );
            }
        }
    }

    if drag.is_none() && response.drag_started() {
        if let Some(pos) = response.interact_pointer_pos() {
            let (lx, ly) = local(pos);
            let tick = x_to_tick(lx, zoom).min(length_ticks_total.saturating_sub(1));
            let pitch = y_to_pitch(ly, zoom);
            let modifiers = ui.input(|i| i.modifiers);

            if let Some(edge_hit) = note_at_right_edge(notes, lx, pitch) {
                selected.clear();
                selected.insert(edge_hit.id);
                *drag = Some(PianoRollDrag {
                    mode: PianoRollDragMode::Resize {
                        note_id: edge_hit.id,
                    },
                });
            } else if let Some(hit) = note_at(notes, tick, pitch) {
                if selected.len() > 1 && selected.contains(&hit.id) {
                    let origin: Vec<(u64, usize, u8)> = notes
                        .iter()
                        .filter(|n| selected.contains(&n.id))
                        .map(|n| (n.id, n.start_tick, n.pitch))
                        .collect();
                    *drag = Some(PianoRollDrag {
                        mode: PianoRollDragMode::MoveSelection {
                            anchor_id: hit.id,
                            grab_tick_offset: tick as i64 - hit.start_tick as i64,
                            start_pitch: pitch as i32,
                            origin,
                        },
                    });
                } else {
                    selected.clear();
                    selected.insert(hit.id);
                    *drag = Some(PianoRollDrag {
                        mode: PianoRollDragMode::Move {
                            note_id: hit.id,
                            grab_tick_offset: tick as i64 - hit.start_tick as i64,
                        },
                    });
                }
            } else if modifiers.shift {
                *drag = Some(PianoRollDrag {
                    mode: PianoRollDragMode::BoxSelect {
                        start_local: egui::pos2(lx, ly),
                    },
                });
            } else {
                selected.clear();
                let id = add_note(
                    notes,
                    next_note_id,
                    pitch,
                    tick,
                    default_length_ticks.max(1),
                    100,
                );
                *drag = Some(PianoRollDrag {
                    mode: PianoRollDragMode::Create {
                        note_id: id,
                        start_tick: tick,
                        pitch,
                    },
                });
            }
        }
    }

    if let Some(state) = drag {
        match &state.mode {
            PianoRollDragMode::Velocity { .. } => {}
            PianoRollDragMode::BoxSelect { start_local } => {
                if response.drag_stopped() {
                    if let Some(pos) = response.interact_pointer_pos() {
                        let (lx, ly) = local(pos);
                        let x_min = start_local.x.min(lx).max(0.0);
                        let x_max = start_local.x.max(lx).max(0.0);
                        let y_min = start_local.y.min(ly);
                        let y_max = start_local.y.max(ly);
                        let tick_lo = x_to_tick(x_min, zoom);
                        let tick_hi = x_to_tick(x_max, zoom);
                        let pitch_hi = y_to_pitch(y_min, zoom);
                        let pitch_lo = y_to_pitch(y_max, zoom);
                        if !ui.input(|i| i.modifiers.command) {
                            selected.clear();
                        }
                        for note in notes.iter() {
                            let time_overlap = note.start_tick <= tick_hi
                                && note.start_tick + note.length_ticks > tick_lo;
                            let pitch_in_range = note.pitch >= pitch_lo && note.pitch <= pitch_hi;
                            if time_overlap && pitch_in_range {
                                selected.insert(note.id);
                            }
                        }
                    }
                    *drag = None;
                }
            }
            PianoRollDragMode::MoveSelection {
                anchor_id,
                grab_tick_offset,
                start_pitch,
                origin,
            } => {
                if response.dragged() {
                    if let Some(pos) = response.interact_pointer_pos() {
                        let (lx, ly) = local(pos);
                        let tick = x_to_tick(lx.max(0.0), zoom) as i64;
                        let pitch = y_to_pitch(ly, zoom) as i32;
                        let anchor_orig = origin
                            .iter()
                            .find(|(id, _, _)| id == anchor_id)
                            .map(|(_, t, _)| *t as i64)
                            .unwrap_or(0);
                        let new_anchor_start = (tick - grab_tick_offset).max(0);
                        let delta_tick = new_anchor_start - anchor_orig;
                        let delta_pitch = pitch - start_pitch;
                        for (id, orig_tick, orig_pitch) in origin {
                            if let Some(note) = find_note_mut(notes, *id) {
                                let new_tick = (*orig_tick as i64 + delta_tick).max(0) as usize;
                                note.start_tick =
                                    new_tick.min(length_ticks_total.saturating_sub(1));
                                let new_pitch = (*orig_pitch as i32 + delta_pitch)
                                    .clamp(PIANO_ROLL_LOW as i32, PIANO_ROLL_HIGH as i32);
                                note.pitch = new_pitch as u8;
                            }
                        }
                    }
                }
                if response.drag_stopped() {
                    for (id, _, _) in origin {
                        if let Some(note) = notes.iter().find(|n| n.id == *id).copied() {
                            clear_overlaps(
                                notes,
                                note.id,
                                note.pitch,
                                note.start_tick,
                                note.length_ticks,
                            );
                        }
                    }
                    *drag = None;
                }
            }
            PianoRollDragMode::Move {
                note_id,
                grab_tick_offset,
            } if notes.iter().any(|n| n.id == *note_id) => {
                let note_id = *note_id;
                let grab_tick_offset = *grab_tick_offset;
                if response.dragged() {
                    if let Some(pos) = response.interact_pointer_pos() {
                        let (lx, ly) = local(pos);
                        let tick = x_to_tick(lx.max(0.0), zoom);
                        let pitch = y_to_pitch(ly, zoom);
                        if let Some(note) = find_note_mut(notes, note_id) {
                            let new_start = (tick as i64 - grab_tick_offset).max(0) as usize;
                            note.start_tick = new_start.min(length_ticks_total.saturating_sub(1));
                            note.pitch = pitch;
                        }
                    }
                }
                if response.drag_stopped() {
                    if let Some(note) = notes.iter().find(|n| n.id == note_id).copied() {
                        clear_overlaps(
                            notes,
                            note.id,
                            note.pitch,
                            note.start_tick,
                            note.length_ticks,
                        );
                    }
                    *drag = None;
                }
            }
            PianoRollDragMode::Resize { note_id } if notes.iter().any(|n| n.id == *note_id) => {
                let note_id = *note_id;
                if response.dragged() {
                    if let Some(pos) = response.interact_pointer_pos() {
                        let (lx, _ly) = local(pos);
                        let tick = x_to_tick(lx.max(0.0), zoom);
                        if let Some(note) = find_note_mut(notes, note_id) {
                            let new_len =
                                (tick as i64 - note.start_tick as i64 + 1).max(1) as usize;
                            note.length_ticks = new_len;
                        }
                    }
                }
                if response.drag_stopped() {
                    if let Some(note) = notes.iter().find(|n| n.id == note_id).copied() {
                        clear_overlaps(
                            notes,
                            note.id,
                            note.pitch,
                            note.start_tick,
                            note.length_ticks,
                        );
                    }
                    *drag = None;
                }
            }
            PianoRollDragMode::Create {
                note_id,
                start_tick,
                pitch,
            } if notes.iter().any(|n| n.id == *note_id) => {
                let note_id = *note_id;
                let start_tick = *start_tick;
                let pitch = *pitch;
                if response.dragged() {
                    if let Some(pos) = response.interact_pointer_pos() {
                        let (lx, _ly) = local(pos);
                        let tick = x_to_tick(lx.max(0.0), zoom);
                        if let Some(note) = find_note_mut(notes, note_id) {
                            let end = tick.max(start_tick + 1);
                            note.start_tick = start_tick;
                            note.length_ticks = end - start_tick;
                            note.pitch = pitch;
                        }
                    }
                }
                if response.drag_stopped() {
                    if let Some(note) = notes.iter().find(|n| n.id == note_id).copied() {
                        clear_overlaps(
                            notes,
                            note.id,
                            note.pitch,
                            note.start_tick,
                            note.length_ticks,
                        );
                    }
                    *drag = None;
                }
            }
            // Note behind a Move/Resize/Create drag was deleted mid-drag (e.g. via the
            // keyboard-delete handling above) — just drop the now-dangling drag state.
            PianoRollDragMode::Move { .. }
            | PianoRollDragMode::Resize { .. }
            | PianoRollDragMode::Create { .. } => {
                if response.drag_stopped() {
                    *drag = None;
                }
            }
        }
    }

    if drag.is_none() {
        if let Some(pos) = response.hover_pos() {
            let (lx, ly) = local(pos);
            let pitch = y_to_pitch(ly, zoom);
            let hovering_edge = note_at_right_edge(notes, lx, pitch).is_some();
            if hovering_edge {
                ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
            }
        }
    }
}

/// The Playlist's contents: a fixed-width, non-scrolling name/color header column (see
/// `draw_playlist_row_header`) on the left — one row per track, `StepGrid`/`PianoRoll` tracks
/// first (their own independent `regions`) then `Audio` tracks (their `audio_clips`) — beside a
/// horizontally-scrolling canvas where each row draws that one track's own clips, positioned/sized
/// by `start_tick`/`loop_length_steps` (or the audio equivalent). Toggled by `playlist_open`, and
/// either docked into the main window's central area or popped into its own OS window via
/// `playlist_detached` (see `ui` in `impl eframe::App for SimpleDawApp`) — same dock/detach split
/// as the Channel Rack. This is also the *only* place a region gets opened for editing:
/// double-click one to open it in Piano Roll/Beats (see
/// `PlaylistEditorTargets`) — there's no Channel Rack button or in-window picker for it.
/// Draws a small preview of a region's musical content inside its Playlist clip rect — thin bars
/// for piano-roll notes (stacked by pitch) or step-grid hits (stacked by lane row). Tiled across
/// `rect`'s width the same way `audio.rs`'s `Sequencer::tick` repeats/truncates the content within
/// `loop_length_steps` (`(tick_index - region.start_tick) % region.content_length_ticks()`), so the
/// preview matches what's actually played at each point along the clip rather than just the first
/// pass through the content.
pub(crate) fn draw_region_note_preview(painter: &egui::Painter, rect: egui::Rect, region: &Region) {
    let content_ticks = region.content_length_ticks();
    let loop_ticks = region.loop_length_steps * TICKS_PER_STEP;
    if content_ticks == 0 || loop_ticks == 0 || rect.width() < 6.0 || rect.height() < 3.0 {
        return;
    }
    let px_per_tick = rect.width() / loop_ticks as f32;
    let color = egui::Color32::from_white_alpha(235);

    let draw_hit = |row: usize, rows: usize, start_tick: usize, length_ticks: usize| {
        let row_h = (rect.height() / rows.max(1) as f32).max(1.0);
        let mut tile_start = 0usize;
        while tile_start < loop_ticks {
            let abs_start = tile_start + start_tick;
            if abs_start < loop_ticks {
                let abs_end = (abs_start + length_ticks.max(1)).min(loop_ticks);
                let x0 = rect.left() + abs_start as f32 * px_per_tick;
                let x1 = rect.left() + abs_end as f32 * px_per_tick;
                let y = rect.top() + row as f32 * row_h;
                let hit_rect = egui::Rect::from_min_size(
                    egui::pos2(x0, y),
                    egui::vec2((x1 - x0).max(1.0), row_h),
                );
                painter.rect_filled(hit_rect, 0.0, color);
            }
            tile_start += content_ticks;
        }
    };

    match &region.content {
        RegionContent::PianoRoll(notes) => {
            if notes.is_empty() {
                return;
            }
            let min_pitch = notes.iter().map(|n| n.pitch).min().unwrap();
            let max_pitch = notes.iter().map(|n| n.pitch).max().unwrap();
            let rows = (max_pitch - min_pitch) as usize + 1;
            for note in notes {
                let row = (max_pitch - note.pitch) as usize;
                draw_hit(row, rows, note.start_tick, note.length_ticks);
            }
        }
        RegionContent::StepGrid(lanes) => {
            if lanes.is_empty() {
                return;
            }
            let rows = lanes.len();
            for (row, lane) in lanes.iter().enumerate() {
                for (step, hit) in lane.steps.iter().enumerate() {
                    if hit.is_some() {
                        draw_hit(row, rows, step * TICKS_PER_STEP, TICKS_PER_STEP);
                    }
                }
            }
        }
    }
}

/// A thin strip below the note canvas showing one draggable bar per note,
/// height proportional to velocity — the standard piano-roll way to see and
/// adjust velocity without cluttering the note itself.
fn velocity_lane_ui(
    ui: &mut egui::Ui,
    notes: &mut Vec<Note>,
    canvas_width: f32,
    drag: &mut Option<PianoRollDrag>,
    selected: &HashSet<u64>,
    zoom: f32,
    note_color: egui::Color32,
) {
    ui.horizontal(|ui| {
        let (response, painter) = ui.allocate_painter(
            egui::vec2(canvas_width, VELOCITY_LANE_HEIGHT),
            egui::Sense::click_and_drag(),
        );
        let rect = response.rect;
        painter.rect_filled(rect, 0u8, ui.visuals().extreme_bg_color);

        for note in notes.iter() {
            let x = rect.left() + tick_to_x(note.start_tick, zoom);
            let w = tick_to_x(note.length_ticks, zoom).max(3.0).min(14.0);
            let h = (note.velocity as f32 / 127.0) * VELOCITY_LANE_HEIGHT;
            let bar_rect =
                egui::Rect::from_min_size(egui::pos2(x, rect.bottom() - h), egui::vec2(w, h));
            let color = if selected.contains(&note.id) {
                egui::Color32::WHITE
            } else {
                note_color
            };
            painter.rect_filled(bar_rect, 1u8, color);
        }

        let velocity_from_y = |y: f32| {
            (((rect.bottom() - y) / VELOCITY_LANE_HEIGHT) * 127.0)
                .round()
                .clamp(1.0, 127.0) as u8
        };

        if drag.is_none() && response.drag_started() {
            if let Some(pos) = response.interact_pointer_pos() {
                let tick = x_to_tick(pos.x - rect.left(), zoom);
                if let Some(note) = notes
                    .iter()
                    .find(|n| tick >= n.start_tick && tick < n.start_tick + n.length_ticks)
                {
                    *drag = Some(PianoRollDrag {
                        mode: PianoRollDragMode::Velocity { note_id: note.id },
                    });
                }
            }
        }

        if let Some(state) = drag {
            if let PianoRollDragMode::Velocity { note_id } = state.mode {
                if response.dragged() {
                    if let Some(pos) = response.interact_pointer_pos() {
                        let velocity = velocity_from_y(pos.y);
                        if let Some(note) = find_note_mut(notes, note_id) {
                            note.velocity = velocity;
                        }
                    }
                }
                if response.drag_stopped() {
                    *drag = None;
                }
            }
        }
    });
}
