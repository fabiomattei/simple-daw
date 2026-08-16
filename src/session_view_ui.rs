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

use crate::audio::{SessionSlotHandles, Transport};
use crate::model::{
    FollowAction, LaunchIntent, SessionClip, SessionLaunchRequest, Song, TICKS_PER_STEP,
};
use crate::session::SlotState;

/// Rows shown even when every track's `Track::session_clips` is shorter than this — so the grid
/// always offers somewhere to assign a first clip rather than starting at zero height.
const MIN_VISIBLE_SLOTS: usize = 8;

/// How many ticks a Session View launch/stop click snaps forward to — see
/// `session::next_quantize_boundary`. Live UI state (`SimpleDawApp::session_quantize`), not song
/// data; `None` launches instantly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SessionQuantize {
    None,
    QuarterBar,
    HalfBar,
    #[default]
    OneBar,
    TwoBar,
    FourBar,
}

impl SessionQuantize {
    const ALL: [SessionQuantize; 6] = [
        Self::None,
        Self::QuarterBar,
        Self::HalfBar,
        Self::OneBar,
        Self::TwoBar,
        Self::FourBar,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::None => "None",
            Self::QuarterBar => "1/4 Bar",
            Self::HalfBar => "1/2 Bar",
            Self::OneBar => "1 Bar",
            Self::TwoBar => "2 Bar",
            Self::FourBar => "4 Bar",
        }
    }

    /// Ticks per quantize unit at `song`'s own time signature — see `Song::steps_per_bar`.
    fn ticks(self, song: &Song) -> usize {
        let bar_ticks = song.steps_per_bar() * TICKS_PER_STEP;
        match self {
            Self::None => 0,
            Self::QuarterBar => bar_ticks / 4,
            Self::HalfBar => bar_ticks / 2,
            Self::OneBar => bar_ticks,
            Self::TwoBar => bar_ticks * 2,
            Self::FourBar => bar_ticks * 4,
        }
    }
}

/// Draws the Session View panel: the Arrangement/Session play-mode switch, the quantize picker,
/// and the clip-slot grid itself. `session_slots` is the audio thread's live queued/playing/
/// stopped state (see `SessionSlotHandles`'s doc comment) — locked once per frame here, briefly,
/// the same read pattern the Mixer's meters already use.
pub fn session_view_contents_ui(
    ui: &mut egui::Ui,
    song: &mut Song,
    transport: &Transport,
    session_slots: &SessionSlotHandles,
    quantize: &mut SessionQuantize,
    follow_action_editor: &mut Option<(usize, usize)>,
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
            .selected_text(quantize.label())
            .show_ui(ui, |ui| {
                for option in SessionQuantize::ALL {
                    ui.selectable_value(quantize, option, option.label());
                }
            });
        transport.set_session_quantize_ticks(quantize.ticks(song));
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
                    if ui
                        .button("▶")
                        .on_hover_text(format!(
                            "Launch Scene {}: every track's clip at this row (tracks with \
                             nothing here are left alone)",
                            slot_index + 1
                        ))
                        .clicked()
                    {
                        launch_scene(song, slot_index);
                    }
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
                        );
                    }
                    ui.end_row();
                }
            });
    });

    follow_action_editor_window_ui(ui.ctx(), song, follow_action_editor);
}

/// Sends a `Play` request to every track whose `session_clips[slot_index]` is filled — tracks
/// with nothing at that row are left untouched (Ableton's own scene-launch behavior: a scene
/// launch never stops a track that has no clip in that row).
fn launch_scene(song: &mut Song, slot_index: usize) {
    for track_index in 0..song.tracks.len() {
        let has_clip = song.tracks[track_index]
            .session_clips
            .get(slot_index)
            .is_some_and(|slot| slot.is_some());
        if has_clip {
            send_launch_request(song, track_index, slot_index, LaunchIntent::Play);
        }
    }
}

/// One clip slot's button: empty slots offer "Assign from Playlist" on right-click, filled slots
/// show their name/state and launch/stop on left-click.
fn session_slot_cell_ui(
    ui: &mut egui::Ui,
    song: &mut Song,
    track_index: usize,
    slot_index: usize,
    live_state: Option<SlotState>,
    follow_action_editor: &mut Option<(usize, usize)>,
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

    let button = egui::Button::new(label).fill(fill).min_size(egui::vec2(96.0, 28.0));
    let response = ui.add(button);

    if has_clip && response.clicked() {
        let intent = match live_state {
            Some(SlotState::Playing { .. } | SlotState::Queued { .. }) => LaunchIntent::Stop,
            _ => LaunchIntent::Play,
        };
        send_launch_request(song, track_index, slot_index, intent);
    }

    response.context_menu(|ui| {
        if has_clip {
            if ui.button("Stop").clicked() {
                send_launch_request(song, track_index, slot_index, LaunchIntent::Stop);
                ui.close();
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
                if *follow_action_editor == Some((track_index, slot_index)) {
                    *follow_action_editor = None;
                }
                ui.close();
            }
        } else {
            assign_from_playlist_menu_ui(ui, song, track_index, slot_index);
        }
    });
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
