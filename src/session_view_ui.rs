//! Session View: an Ableton-style clip-launching grid (tracks as columns, clip slots as rows) —
//! a mode-switch alternative to the Playlist arrangement timeline, never playing at the same time
//! (see `audio::Transport::session_mode`). A new file rather than folded into main.rs's otherwise
//! universal everything-in-one-file convention, an explicit exception made when this feature was
//! scoped given main.rs's size.
//!
//! v1 authoring is deliberately narrow: a slot's content is *assigned* by copying an
//! already-authored Playlist `Region`/`AudioClip` (see `SessionClip::from_region`/
//! `from_audio_clip`), not composed fresh here — so this file never touches the Piano Roll/Beats
//! editor windows.
//!
//! Scene rows (`Song::scenes`, `Scene`) can each recall a tempo on launch (`launch_scene`). Unlike
//! clip launches — which have to round-trip through `Track::session_launch_requests` because the
//! audio thread only ever sees stale per-callback clones of `Song` (see "Data model vs. real-time
//! engine" in the project's CLAUDE.md) — a scene's tempo change is just data (`Song::tempo_map`),
//! so `launch_scene` writes it straight into the live `Song` here on the UI thread: every function
//! in this file already receives `song: &mut Song` as an already-locked `MutexGuard` deref, the
//! same one every other UI-thread edit in this app mutates directly.

use crate::audio::{SessionSlotHandles, Transport};
use crate::model::{
    FollowAction, LaunchIntent, LaunchMode, RegionContent, SessionClip, SessionClipContent,
    SessionLaunchRequest, SessionQuantize, Song, TrackKind,
};
use crate::session::{next_quantize_boundary, SlotState};
use crate::RegionEditTarget;

/// Tempo range for a scene's own `DragValue` — matches the Tempo Track panel's own tempo fields
/// (`tempo_track_ui` in `main.rs`).
const SCENE_TEMPO_RANGE: std::ops::RangeInclusive<f32> = 20.0..=300.0;

/// Rows shown even when every track's `Track::session_clips` is shorter than this — so the grid
/// always offers somewhere to assign a first clip rather than starting at zero height.
const MIN_VISIBLE_SLOTS: usize = 8;

/// Every `SessionQuantize` value, in display order — for iterating the quantize picker(s). The
/// enum itself lives in `model.rs` (see its doc comment on why) since `SessionClip::
/// quantize_override` needs it too, not just this grid-wide picker.
const SESSION_QUANTIZE_OPTIONS: [SessionQuantize; 6] = [
    SessionQuantize::None,
    SessionQuantize::QuarterBar,
    SessionQuantize::HalfBar,
    SessionQuantize::OneBar,
    SessionQuantize::TwoBar,
    SessionQuantize::FourBar,
];

fn session_quantize_label(quantize: SessionQuantize) -> &'static str {
    match quantize {
        SessionQuantize::None => "None",
        SessionQuantize::QuarterBar => "1/4 Bar",
        SessionQuantize::HalfBar => "1/2 Bar",
        SessionQuantize::OneBar => "1 Bar",
        SessionQuantize::TwoBar => "2 Bar",
        SessionQuantize::FourBar => "4 Bar",
    }
}

/// Draws the Session View panel: the Arrangement/Session play-mode switch, the quantize picker,
/// and the clip-slot grid itself. `session_slots` is the audio thread's live queued/playing/
/// stopped state (see `SessionSlotHandles`'s doc comment) — locked once per frame here, briefly,
/// the same read pattern the Mixer's meters already use.
#[allow(clippy::too_many_arguments)]
pub fn session_view_contents_ui(
    ui: &mut egui::Ui,
    song: &mut Song,
    transport: &Transport,
    session_slots: &SessionSlotHandles,
    quantize: &mut SessionQuantize,
    follow_action_editor: &mut Option<(usize, usize)>,
    flex_editor: &mut Option<(usize, usize)>,
    selected_track: &mut Option<usize>,
    piano_roll_target: &mut Option<RegionEditTarget>,
    selected_beats_track: &mut Option<usize>,
    beats_target: &mut Option<RegionEditTarget>,
    detached: &mut bool,
) {
    ui.horizontal(|ui| {
        let session_mode = transport.is_session_mode();
        if ui
            .selectable_label(!session_mode, "▶ Arrangement")
            .on_hover_text("Transport plays the Playlist arrangement")
            .clicked()
        {
            transport.set_session_mode(false);
        }
        if ui
            .selectable_label(session_mode, "▶ Session")
            .on_hover_text("Transport plays this Session View grid instead of the Playlist")
            .clicked()
        {
            transport.set_session_mode(true);
        }
        ui.add_space(12.0);
        ui.label("Quantize:");
        egui::ComboBox::from_id_salt("session_quantize")
            .selected_text(session_quantize_label(*quantize))
            .show_ui(ui, |ui| {
                for option in SESSION_QUANTIZE_OPTIONS {
                    ui.selectable_value(quantize, option, session_quantize_label(option));
                }
            });
        transport.set_session_quantize_ticks(quantize.ticks(song));
        if ui
            .small_button(if *detached { "⏷ Dock" } else { "⧉ Detach" })
            .clicked()
        {
            *detached = !*detached;
        }
    });
    ui.separator();

    if song.tracks.is_empty() {
        ui.label("No tracks yet — add one from the Channel Rack.");
        return;
    }

    let live_slots = session_slots.lock().ok();
    let slot_count = song
        .tracks
        .iter()
        .map(|track| track.session_clips.len())
        .max()
        .unwrap_or(0)
        .max(MIN_VISIBLE_SLOTS);
    let track_count = song.tracks.len();

    egui::ScrollArea::both().auto_shrink([false, false]).show(ui, |ui| {
        egui::Grid::new("session_view_grid")
            .num_columns(track_count + 1)
            .spacing(egui::vec2(6.0, 6.0))
            .striped(true)
            .show(ui, |ui| {
                ui.label("");
                for track in &song.tracks {
                    ui.label(egui::RichText::new(&track.name).strong());
                }
                ui.end_row();

                for slot_index in 0..slot_count {
                    scene_header_cell_ui(ui, song, transport, slot_index);
                    for track_index in 0..track_count {
                        let live_state = live_slots
                            .as_deref()
                            .and_then(|tracks| tracks.get(track_index))
                            .and_then(|slots| slots.get(slot_index))
                            .copied();
                        session_slot_cell_ui(
                            ui,
                            song,
                            track_index,
                            slot_index,
                            live_state,
                            follow_action_editor,
                            flex_editor,
                            selected_track,
                            piano_roll_target,
                            selected_beats_track,
                            beats_target,
                        );
                    }
                    ui.end_row();
                }
            });
    });

    follow_action_editor_window_ui(ui.ctx(), song, follow_action_editor);
}

/// One scene row's header cell: an editable name, an optional tempo to recall on launch (see
/// `model::Scene`), and the launch button itself.
fn scene_header_cell_ui(ui: &mut egui::Ui, song: &mut Song, transport: &Transport, slot_index: usize) {
    // Read before `ensure_scene` takes a mutable borrow of `song` below — `bpm_at` needs `&song`.
    let current_bpm = song.bpm_at(transport.current_tick());
    ui.horizontal(|ui| {
        if ui
            .button("▶")
            .on_hover_text(format!(
                "Launch Scene {}: every track's clip at this row (tracks with nothing here are \
                 left alone), plus this scene's own tempo if it has one",
                slot_index + 1
            ))
            .clicked()
        {
            launch_scene(song, transport, slot_index);
        }
        let scene = song.ensure_scene(slot_index);
        ui.add(
            egui::TextEdit::singleline(&mut scene.name)
                .hint_text(format!("Scene {}", slot_index + 1))
                .desired_width(70.0),
        );
        let mut has_tempo = scene.tempo.is_some();
        if ui.checkbox(&mut has_tempo, "").on_hover_text("Recall a tempo when this scene launches").changed() {
            scene.tempo = has_tempo.then_some(current_bpm);
        }
        if let Some(tempo) = scene.tempo.as_mut() {
            ui.add(egui::DragValue::new(tempo).range(SCENE_TEMPO_RANGE).suffix(" BPM"));
        }
    });
}

/// Sends a `Play` request to every track whose `session_clips[slot_index]` is filled — tracks
/// with nothing at that row are left untouched (Ableton's own scene-launch behavior: a scene
/// launch never stops a track that has no clip in that row). If this row's `Scene` has its own
/// tempo set, also inserts it as a real `TempoPoint` at the same launch-quantize boundary tick the
/// clip launches themselves resolve to — tick-accurate and rewind-safe, the same guarantee the
/// Tempo Track panel's own points get, since `song` here is already the locked live `Song` (see
/// this module's doc comment on why no audio-thread hop is needed for this, unlike clip launches).
fn launch_scene(song: &mut Song, transport: &Transport, slot_index: usize) {
    for track_index in 0..song.tracks.len() {
        let has_clip = song.tracks[track_index]
            .session_clips
            .get(slot_index)
            .is_some_and(|slot| slot.is_some());
        if has_clip {
            send_launch_request(song, track_index, slot_index, LaunchIntent::Play);
        }
    }
    if let Some(bpm) = song.scenes.get(slot_index).and_then(|scene| scene.tempo) {
        let boundary_tick =
            next_quantize_boundary(transport.current_tick(), transport.session_quantize_ticks());
        song.set_tempo_at(boundary_tick, bpm);
    }
}

/// One clip slot's button: empty slots offer "Assign from Playlist" on right-click, filled slots
/// show their name/state and launch/stop on left-click.
#[allow(clippy::too_many_arguments)]
fn session_slot_cell_ui(
    ui: &mut egui::Ui,
    song: &mut Song,
    track_index: usize,
    slot_index: usize,
    live_state: Option<SlotState>,
    follow_action_editor: &mut Option<(usize, usize)>,
    flex_editor: &mut Option<(usize, usize)>,
    selected_track: &mut Option<usize>,
    piano_roll_target: &mut Option<RegionEditTarget>,
    selected_beats_track: &mut Option<usize>,
    beats_target: &mut Option<RegionEditTarget>,
) {
    let has_clip = song.tracks[track_index]
        .session_clips
        .get(slot_index)
        .is_some_and(|slot| slot.is_some());

    let (label, fill) = if !has_clip {
        ("".to_string(), ui.visuals().extreme_bg_color)
    } else {
        let name = song.tracks[track_index].session_clips[slot_index]
            .as_ref()
            .map(|clip| clip.name.clone())
            .unwrap_or_default();
        match live_state {
            Some(SlotState::Playing { .. }) => (format!("▶ {name}"), crate::FL_ACCENT_GREEN),
            Some(SlotState::Queued { .. }) => (format!("⏳ {name}"), crate::FL_ACCENT_ORANGE),
            Some(SlotState::QueuedStop { .. }) => (format!("⏹ {name}"), crate::FL_ACCENT_ORANGE),
            Some(SlotState::Stopped) | None => (name, ui.visuals().widgets.inactive.bg_fill),
        }
    };

    let launch_mode = song.tracks[track_index]
        .session_clips
        .get(slot_index)
        .and_then(|slot| slot.as_ref())
        .map(|clip| clip.launch_mode)
        .unwrap_or_default();

    let button = egui::Button::new(label).fill(fill).min_size(egui::vec2(96.0, 28.0));
    let response = ui.add(button);

    if has_clip {
        match launch_mode {
            // Gate/Repeat play only while held — a continuous signal (see
            // `SessionLaunchRequest::held`'s doc comment), read fresh every frame rather than
            // waiting for a `clicked()` edge, so a held-down button keeps the clip going even
            // across frames where nothing "changed" from egui's point of view.
            LaunchMode::Gate | LaunchMode::Repeat => {
                set_held(song, track_index, slot_index, response.is_pointer_button_down_on());
            }
            // Trigger always (re)starts on click — never stops itself; stopping needs the
            // context menu's "Stop" entry, a scene, or another exclusive launch on this track.
            LaunchMode::Trigger => {
                if response.clicked() {
                    send_launch_request(song, track_index, slot_index, LaunchIntent::Play);
                }
            }
            LaunchMode::Toggle => {
                if response.clicked() {
                    let intent = match live_state {
                        Some(SlotState::Playing { .. } | SlotState::Queued { .. }) => LaunchIntent::Stop,
                        _ => LaunchIntent::Play,
                    };
                    send_launch_request(song, track_index, slot_index, intent);
                }
            }
        }
    }

    response.context_menu(|ui| {
        if has_clip {
            if ui.button("Stop").clicked() {
                send_launch_request(song, track_index, slot_index, LaunchIntent::Stop);
                ui.close();
            }
            ui.menu_button("Launch Mode ▸", |ui| {
                launch_mode_menu_ui(ui, song, track_index, slot_index);
            });
            ui.menu_button("Quantize ▸", |ui| {
                quantize_override_menu_ui(ui, song, track_index, slot_index);
            });
            let is_audio_clip = song.tracks[track_index].session_clips[slot_index]
                .as_ref()
                .is_some_and(|clip| matches!(clip.content, SessionClipContent::Audio(_)));
            if is_audio_clip && ui.button("Flex Time / Pitch…").clicked() {
                *flex_editor = Some((track_index, slot_index));
                ui.close();
            }
            let is_piano_roll_region = song.tracks[track_index].kind == TrackKind::PianoRoll
                && song.tracks[track_index].session_clips[slot_index].as_ref().is_some_and(|clip| {
                    matches!(
                        clip.content,
                        SessionClipContent::Region { content: RegionContent::PianoRoll(_), .. }
                    )
                });
            if is_piano_roll_region {
                if ui.button("Edit in Piano Roll…").clicked() {
                    *selected_track = Some(track_index);
                    *piano_roll_target = Some(RegionEditTarget::SessionSlot(slot_index));
                    ui.close();
                }
                if ui
                    .button("Send to Playlist")
                    .on_hover_text(
                        "Places a copy of this clip as a new Region at the end of this track's \
                         Playlist content — the slot keeps its own copy too.",
                    )
                    .clicked()
                {
                    send_clip_to_playlist(song, track_index, slot_index);
                    ui.close();
                }
            }
            let is_step_grid_region = song.tracks[track_index].kind == TrackKind::StepGrid
                && song.tracks[track_index].session_clips[slot_index].as_ref().is_some_and(|clip| {
                    matches!(
                        clip.content,
                        SessionClipContent::Region { content: RegionContent::StepGrid(_), .. }
                    )
                });
            if is_step_grid_region {
                if ui.button("Edit in Beats…").clicked() {
                    *selected_beats_track = Some(track_index);
                    *beats_target = Some(RegionEditTarget::SessionSlot(slot_index));
                    ui.close();
                }
                if ui
                    .button("Send to Playlist")
                    .on_hover_text(
                        "Places a copy of this clip as a new Region at the end of this track's \
                         Playlist content — the slot keeps its own copy too.",
                    )
                    .clicked()
                {
                    send_clip_to_playlist(song, track_index, slot_index);
                    ui.close();
                }
            }
            if ui.button("Follow Action…").clicked() {
                *follow_action_editor = Some((track_index, slot_index));
                ui.close();
            }
            let legato = song.tracks[track_index].session_clips[slot_index]
                .as_ref()
                .is_some_and(|clip| clip.legato);
            if ui.button(if legato { "Legato ✓" } else { "Legato" }).clicked() {
                if let Some(Some(clip)) = song.tracks[track_index].session_clips.get_mut(slot_index) {
                    clip.legato = !clip.legato;
                }
                ui.close();
            }
            if ui.button("Clear").clicked() {
                if let Some(slot) = song.tracks[track_index].session_clips.get_mut(slot_index) {
                    *slot = None;
                }
                // Reset any stale `held` signal so a future clip assigned to this same slot
                // doesn't inherit a leftover Gate/Repeat hold from before it was cleared.
                set_held(song, track_index, slot_index, false);
                if *follow_action_editor == Some((track_index, slot_index)) {
                    *follow_action_editor = None;
                }
                if *flex_editor == Some((track_index, slot_index)) {
                    *flex_editor = None;
                }
                // Both `piano_roll_target`/`beats_target` are only meaningful alongside their
                // own `selected_track`/`selected_beats_track` — a bare slot-index match isn't
                // enough, since the same slot index could be open for a *different* track.
                if *selected_track == Some(track_index)
                    && *piano_roll_target == Some(RegionEditTarget::SessionSlot(slot_index))
                {
                    *piano_roll_target = None;
                }
                if *selected_beats_track == Some(track_index)
                    && *beats_target == Some(RegionEditTarget::SessionSlot(slot_index))
                {
                    *beats_target = None;
                }
                ui.close();
            }
        } else {
            assign_from_playlist_menu_ui(ui, song, track_index, slot_index);
            if song.tracks[track_index].kind == TrackKind::PianoRoll
                && ui.button("Compose New Region…").clicked()
            {
                let steps_per_bar = song.steps_per_bar();
                let clip = SessionClip::new_piano_roll("New Region", steps_per_bar);
                assign_clip(song, track_index, slot_index, clip);
                *selected_track = Some(track_index);
                *piano_roll_target = Some(RegionEditTarget::SessionSlot(slot_index));
                ui.close();
            }
            if song.tracks[track_index].kind == TrackKind::StepGrid
                && ui.button("Compose New Beats Lane…").clicked()
            {
                let steps_per_bar = song.steps_per_bar();
                let clip = SessionClip::new_step_grid("New Beats Lane", steps_per_bar);
                assign_clip(song, track_index, slot_index, clip);
                *selected_beats_track = Some(track_index);
                *beats_target = Some(RegionEditTarget::SessionSlot(slot_index));
                ui.close();
            }
        }
    });
}

/// Places a copy of `track_index`/`slot_index`'s clip as a new Region at the end of that track's
/// Playlist content — see `SessionClip::to_region`/`Track::end_of_regions_tick`. A no-op if the
/// slot is empty or isn't `Region` content (e.g. `Audio`, out of scope for this action for now).
fn send_clip_to_playlist(song: &mut Song, track_index: usize, slot_index: usize) {
    let Some(track) = song.tracks.get(track_index) else { return };
    let Some(Some(clip)) = track.session_clips.get(slot_index) else { return };
    let Some(region) = clip.to_region(track.end_of_regions_tick()) else { return };
    song.tracks[track_index].regions.push(region);
}

/// "Assign from Playlist ▸" submenu contents: every region/audio clip already authored on this
/// track, each copying its content into the slot when picked — see `SessionClip::from_region`/
/// `from_audio_clip`.
fn assign_from_playlist_menu_ui(
    ui: &mut egui::Ui,
    song: &mut Song,
    track_index: usize,
    slot_index: usize,
) {
    let region_count = song.tracks[track_index].regions.len();
    let clip_count = song.tracks[track_index].audio_clips.len();
    if region_count == 0 && clip_count == 0 {
        ui.label("No Playlist content on this track yet");
        return;
    }
    for region_index in 0..region_count {
        let name = song.tracks[track_index].regions[region_index].name.clone();
        if ui.button(format!("Region: {name}")).clicked() {
            let clip = SessionClip::from_region(&song.tracks[track_index].regions[region_index]);
            assign_clip(song, track_index, slot_index, clip);
            ui.close();
        }
    }
    for clip_index in 0..clip_count {
        let label = song.tracks[track_index].audio_clips[clip_index]
            .file_path
            .clone();
        if ui.button(format!("Audio: {label}")).clicked() {
            let clip = SessionClip::from_audio_clip(&song.tracks[track_index].audio_clips[clip_index]);
            assign_clip(song, track_index, slot_index, clip);
            ui.close();
        }
    }
}

fn assign_clip(song: &mut Song, track_index: usize, slot_index: usize, clip: SessionClip) {
    let slots = &mut song.tracks[track_index].session_clips;
    if slots.len() <= slot_index {
        slots.resize(slot_index + 1, None);
    }
    slots[slot_index] = Some(clip);
}

/// Bumps `track.session_launch_requests[slot_index]`'s generation with `intent` — the click
/// handler's whole job (see `model::SessionLaunchRequest`'s doc comment on why this is an
/// edge-triggered counter rather than a direct state write).
fn send_launch_request(song: &mut Song, track_index: usize, slot_index: usize, intent: LaunchIntent) {
    let requests = &mut song.tracks[track_index].session_launch_requests;
    if requests.len() <= slot_index {
        requests.resize(slot_index + 1, SessionLaunchRequest::default());
    }
    requests[slot_index].generation += 1;
    requests[slot_index].intent = intent;
}

/// Writes `SessionLaunchRequest::held` directly — a level signal read fresh every buffer, so
/// (unlike `send_launch_request`) there's no generation to bump; the plain field write is enough
/// (see that field's doc comment).
fn set_held(song: &mut Song, track_index: usize, slot_index: usize, held: bool) {
    let requests = &mut song.tracks[track_index].session_launch_requests;
    if requests.len() <= slot_index {
        requests.resize(slot_index + 1, SessionLaunchRequest::default());
    }
    requests[slot_index].held = held;
}

/// Contents of a slot's "Launch Mode ▸" context-menu entry — see `model::LaunchMode`.
fn launch_mode_menu_ui(ui: &mut egui::Ui, song: &mut Song, track_index: usize, slot_index: usize) {
    let Some(Some(clip)) = song.tracks[track_index].session_clips.get_mut(slot_index) else { return };
    for mode in [LaunchMode::Toggle, LaunchMode::Trigger, LaunchMode::Gate, LaunchMode::Repeat] {
        let label = match mode {
            LaunchMode::Toggle => "Toggle",
            LaunchMode::Trigger => "Trigger",
            LaunchMode::Gate => "Gate",
            LaunchMode::Repeat => "Repeat",
        };
        if ui.selectable_label(clip.launch_mode == mode, label).clicked() {
            clip.launch_mode = mode;
            ui.close();
        }
    }
}

/// Contents of a slot's "Quantize ▸" context-menu entry — see `model::SessionClip::
/// quantize_override`. "Use Grid Default" clears the override (`None`), falling back to the
/// grid-wide quantize picker's own setting; the rest force this clip to always launch/stop at
/// that specific boundary regardless of the grid-wide setting.
fn quantize_override_menu_ui(ui: &mut egui::Ui, song: &mut Song, track_index: usize, slot_index: usize) {
    let Some(Some(clip)) = song.tracks[track_index].session_clips.get_mut(slot_index) else { return };
    if ui.selectable_label(clip.quantize_override.is_none(), "Use Grid Default").clicked() {
        clip.quantize_override = None;
        ui.close();
    }
    for option in SESSION_QUANTIZE_OPTIONS {
        if ui
            .selectable_label(clip.quantize_override == Some(option), session_quantize_label(option))
            .clicked()
        {
            clip.quantize_override = Some(option);
            ui.close();
        }
    }
}

/// Canonical display value for each `FollowAction` variant — `Other` shown at row `0` here since
/// the combo box only needs something to compare discriminants against; the actual row index is
/// edited by the `DragValue` next to it (see `follow_action_row_ui`).
const FOLLOW_ACTION_VARIANTS: [FollowAction; 9] = [
    FollowAction::None,
    FollowAction::Stop,
    FollowAction::Again,
    FollowAction::Previous,
    FollowAction::Next,
    FollowAction::First,
    FollowAction::Last,
    FollowAction::Any,
    FollowAction::Other(0),
];

fn follow_action_label(action: FollowAction) -> &'static str {
    match action {
        FollowAction::None => "None",
        FollowAction::Stop => "Stop",
        FollowAction::Again => "Again (restart)",
        FollowAction::Previous => "Previous",
        FollowAction::Next => "Next",
        FollowAction::First => "First",
        FollowAction::Last => "Last",
        FollowAction::Any => "Any (random)",
        FollowAction::Other(_) => "Other row…",
    }
}

/// The small popup opened from a filled slot's "Follow Action…" context-menu entry — `times`,
/// and the two weighted candidate actions (`action_a`/`chance_a`, `action_b`/`chance_b`), matching
/// Ableton's own two-follow-action model (see `model::FollowActionConfig`). Closing the window (or
/// the target slot being cleared, see `session_slot_cell_ui`'s "Clear") clears `editor_target`.
fn follow_action_editor_window_ui(
    ctx: &egui::Context,
    song: &mut Song,
    editor_target: &mut Option<(usize, usize)>,
) {
    let Some((track_index, slot_index)) = *editor_target else { return };
    let slot_count = song.tracks.get(track_index).map_or(0, |t| t.session_clips.len());
    let Some(clip) = song
        .tracks
        .get_mut(track_index)
        .and_then(|track| track.session_clips.get_mut(slot_index))
        .and_then(|slot| slot.as_mut())
    else {
        *editor_target = None;
        return;
    };

    let mut open = true;
    egui::Window::new(format!("Follow Action: {}", clip.name))
        .id(egui::Id::new(("follow-action-editor", track_index, slot_index)))
        .open(&mut open)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Times:");
                ui.add(egui::DragValue::new(&mut clip.follow_action.times).range(1..=64));
            });
            ui.separator();
            follow_action_row_ui(
                ui,
                "A",
                &mut clip.follow_action.action_a,
                &mut clip.follow_action.chance_a,
                slot_count,
            );
            follow_action_row_ui(
                ui,
                "B",
                &mut clip.follow_action.action_b,
                &mut clip.follow_action.chance_b,
                slot_count,
            );
        });
    if !open {
        *editor_target = None;
    }
}

fn follow_action_row_ui(
    ui: &mut egui::Ui,
    label: &str,
    action: &mut FollowAction,
    chance: &mut f32,
    slot_count: usize,
) {
    ui.horizontal(|ui| {
        ui.label(label);
        egui::ComboBox::from_id_salt(("follow_action_combo", label))
            .selected_text(follow_action_label(*action))
            .show_ui(ui, |ui| {
                for option in FOLLOW_ACTION_VARIANTS {
                    let selected = std::mem::discriminant(action) == std::mem::discriminant(&option);
                    if ui.selectable_label(selected, follow_action_label(option)).clicked() {
                        *action = option;
                    }
                }
            });
        if let FollowAction::Other(index) = action {
            let mut row_display = *index + 1;
            if ui
                .add(egui::DragValue::new(&mut row_display).range(1..=slot_count.max(1)))
                .changed()
            {
                *index = row_display.saturating_sub(1);
            }
        }
        ui.label("Chance:");
        let mut percent = *chance * 100.0;
        if ui.add(egui::Slider::new(&mut percent, 0.0..=100.0).suffix("%")).changed() {
            *chance = percent / 100.0;
        }
    });
}
