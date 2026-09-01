//! Everything for editing an already-placed `AudioClip`/`TakeFolder` in the Playlist: drag/click/
//! context-menu handling (`handle_audio_clip_interaction`, `handle_take_folder_interaction`,
//! `apply_strip_silence`), and the two editor windows opened from a clip's/folder's context menu —
//! Flex Time/Pitch (`flex_editor_window_ui`, plus its Session View counterpart
//! `session_flex_editor_window_ui`) and the Take Folder comp editor (`take_folder_editor_window_ui`).

use std::path::Path;
use std::sync::Arc;

use crate::audio;
use crate::automation_panel::{AutomationDrag, automation_lanes_ui};
use crate::model::{AudioClip, SessionClipContent, Song, TakeFolder, TrackEffectConfig};
use crate::pitch;
use crate::plugin_host::{MasterEffectSlots, SendEffectSlots, TrackEffectSlots};
use crate::playlist_panel::draw_audio_clip_waveform;
use crate::sample::SampleBuffer;
use crate::stretch;
use crate::transient_detection;
use crate::{
    AudioClipContextMenuTarget, FL_ACCENT_GREEN, FL_ACCENT_ORANGE, PLAYLIST_LANE_HEIGHT, RESIZE_HANDLE_PX,
    TakeFolderContextMenuTarget, audio_clip_length_ticks, tick_to_x, x_to_tick,
};

/// What the currently in-progress audio-clip drag (if any) is doing — the `AudioClip` counterpart
/// of `PlaylistDragMode`. Clips are only ever created by recording/import, never drawn out on the
/// timeline, so there's no `Create` mode. Every arm's `clip_index` re-checks bounds every frame, in
/// case the clip was removed (right-click) since the drag began.
enum AudioClipDragMode {
    /// Dragging an existing clip's body: changes `start_tick` only.
    Move {
        track_index: usize,
        clip_index: usize,
        grab_tick_offset: i64,
    },
    /// Dragging an existing clip's left edge: changes `start_tick`/`source_start_frame`/
    /// `length_ticks` together, keeping the clip's on-timeline end point fixed in place.
    TrimStart {
        track_index: usize,
        clip_index: usize,
    },
    /// Dragging an existing clip's right edge: changes `length_ticks` only.
    TrimEnd {
        track_index: usize,
        clip_index: usize,
    },
    /// Dragging the fade-in handle: changes `AudioClip::fade_in_ticks` only.
    FadeIn {
        track_index: usize,
        clip_index: usize,
    },
    /// Dragging the fade-out handle: changes `AudioClip::fade_out_ticks` only.
    FadeOut {
        track_index: usize,
        clip_index: usize,
    },
}

pub(crate) struct AudioClipDrag {
    mode: AudioClipDragMode,
}

/// An in-progress "paint `take_index` over this stretch" drag inside the take-folder comp editor
/// (`take_folder_editor_window_ui`) — `start_tick` is fixed at drag start; the current drag end is
/// read fresh from the pointer position each frame and applied live via
/// `TakeFolder::assign_take_to_range`, the same "mutate live on `dragged()`" pattern
/// `handle_audio_clip_interaction`'s trim/fade drags already use.
pub(crate) struct TakeFolderCompDrag {
    take_index: usize,
    start_tick: usize,
}

/// Which tab of the Flex editor window (`flex_editor_window_ui`) is showing.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum FlexEditorMode {
    Time,
    Pitch,
}

/// An in-progress "drag this warp marker's output position" drag in the Flex editor's Time tab.
/// `marker_index` addresses `AudioClip::warp_markers` directly (seeded with start/end anchors —
/// see `ensure_warp_anchors` — before this drag begins, so the index is always valid once
/// dragging starts). `live_output_frame` is the drag's own not-yet-committed position, read fresh
/// from the pointer every frame for the visual preview; the model (and the clip's baked buffer)
/// only update on `drag_stopped()` — re-baking via `AudioClip::load` on every drag frame would
/// mean re-running WSOLA on every mouse-move, which could visibly stutter for a longer clip.
pub(crate) struct FlexMarkerDrag {
    marker_index: usize,
    live_output_frame: usize,
}

/// An in-progress "drag this detected note's target pitch" drag in the Flex editor's Pitch tab.
/// `start_frame`/`end_frame` are the dragged `pitch::DetectedNote`'s own span (used to find-or-
/// create its matching `AudioClip::pitch_corrections` entry); `start_semitones` is whatever
/// correction (or `0.0`) was already in effect before this drag began; `drag_start_y` is the
/// pointer's canvas-local y at `drag_started()`, against which every later frame's y computes a
/// live semitone delta — same "live preview, commit on `drag_stopped()`" reasoning as
/// `FlexMarkerDrag`.
pub(crate) struct FlexNoteDrag {
    start_frame: usize,
    end_frame: usize,
    start_semitones: f32,
    drag_start_y: f32,
}

/// "Strip Silence" defaults (see `apply_strip_silence`) — a fixed floor rather than a per-clip
/// dial, so the context-menu action is a single click with no extra dialog. -40dBFS/100ms/50ms are
/// the same rough ballpark other DAWs default their strip-silence tools to.
const STRIP_SILENCE_THRESHOLD_DB: f32 = -40.0;
const STRIP_SILENCE_MIN_SILENCE_SECONDS: f32 = 0.1;
const STRIP_SILENCE_MIN_SEGMENT_SECONDS: f32 = 0.05;

/// Replaces the `AudioClip` at `song.tracks[track_index].audio_clips[clip_index]` with one clip
/// per non-silent segment `transient_detection::detect_non_silent_segments` finds in its current
/// trim window — each new clip keeps the same `file_path`/`gain`, re-anchored to the same absolute
/// song tick the audio originally occupied (not shifted to close the gap, so sync with anything
/// else on the timeline survives), with fresh trim (`source_start_frame`/`length_ticks`) and no
/// fades (a fade at the original clip's own edges doesn't carry any meaning at a newly cut edge).
/// A no-op if the clip is missing, unloaded, or entirely silent (no non-silent segments found).
fn apply_strip_silence(song: &mut Song, track_index: usize, clip_index: usize) {
    let Some(clip) = song
        .tracks
        .get(track_index)
        .and_then(|t| t.audio_clips.get(clip_index))
        .cloned()
    else {
        return;
    };
    let Some(buffer) = clip.buffer.clone() else {
        return;
    };
    let tps = audio::ticks_per_second(song.bpm_at(clip.start_tick));
    let frames_per_tick = buffer.sample_rate as f64 / tps;
    let window_start = clip.source_start_frame.min(buffer.mono.len());
    let window_end = window_start
        .saturating_add((clip.effective_length_ticks(tps) as f64 * frames_per_tick).round() as usize)
        .min(buffer.mono.len());
    let window = &buffer.mono[window_start..window_end];

    let segments = transient_detection::detect_non_silent_segments(
        window,
        buffer.sample_rate,
        STRIP_SILENCE_THRESHOLD_DB,
        STRIP_SILENCE_MIN_SILENCE_SECONDS,
        STRIP_SILENCE_MIN_SEGMENT_SECONDS,
    );
    if segments.is_empty() {
        return;
    }

    let new_clips: Vec<AudioClip> = segments
        .into_iter()
        .map(|(seg_start, seg_end)| {
            let mut new_clip = clip.clone();
            let offset_ticks = (seg_start as f64 / frames_per_tick).round() as usize;
            new_clip.start_tick = clip.start_tick + offset_ticks;
            new_clip.source_start_frame = window_start + seg_start;
            new_clip.length_ticks = (((seg_end - seg_start) as f64 / frames_per_tick).round() as usize).max(1);
            new_clip.fade_in_ticks = 0;
            new_clip.fade_out_ticks = 0;
            new_clip
        })
        .collect();

    if let Some(track) = song.tracks.get_mut(track_index) {
        if clip_index < track.audio_clips.len() {
            track.audio_clips.splice(clip_index..=clip_index, new_clips);
        }
    }
}

/// Height of the Flex editor's Time-tab waveform canvas and Pitch-tab note-lane strip.
const FLEX_WAVEFORM_HEIGHT: f32 = 120.0;
const FLEX_PITCH_STRIP_HEIGHT: f32 = 160.0;
/// Vertical pixels per semitone in the Pitch tab — how far a note bar visibly moves per semitone
/// of retargeting.
const PX_PER_SEMITONE: f32 = 6.0;
/// Furthest a note can be retargeted from its detected pitch, either direction — a generous range
/// for correcting an off-pitch take, not a full remapping tool.
const MAX_PITCH_CORRECTION_SEMITONES: f32 = 24.0;

/// Ensures `clip.warp_markers` has at least the two span-boundary anchors (`0 -> 0`, `raw_len ->
/// raw_len` — an identity, audibly-unchanged mapping) before an actual edit needs a real index to
/// mutate. Called at the start of a Time-tab mutation (dragging an anchor, adding a marker), never
/// just from opening the editor window, so merely opening/closing it without touching anything
/// stays a true no-op — see `stretch::warp_buffer`'s "fewer than 2 markers" sentinel.
fn ensure_warp_anchors(clip: &mut AudioClip, raw_len: usize) {
    if clip.warp_markers.len() < 2 {
        clip.warp_markers = vec![
            stretch::WarpMarker { source_frame: 0, output_frame: 0 },
            stretch::WarpMarker { source_frame: raw_len, output_frame: raw_len },
        ];
    }
}

/// The markers to actually *display*: `clip.warp_markers` verbatim once it has real ones, else the
/// virtual identity anchors `ensure_warp_anchors` would seed — so the Time tab's waveform/handles
/// render correctly even before the model has committed anything.
fn effective_warp_markers(clip: &AudioClip, raw_len: usize) -> Vec<stretch::WarpMarker> {
    if clip.warp_markers.len() >= 2 {
        clip.warp_markers.clone()
    } else {
        vec![
            stretch::WarpMarker { source_frame: 0, output_frame: 0 },
            stretch::WarpMarker { source_frame: raw_len, output_frame: raw_len },
        ]
    }
}

/// Linearly interpolates `markers` (sorted by `source_frame`) at `source_frame`, clamping to the
/// nearest edge marker's `output_frame` outside their span — used both to place a newly-added
/// marker without changing the audio (interpolating its starting `output_frame` from its
/// soon-to-be neighbors) and to estimate where an as-yet-unplaced transient would currently land
/// in output time (for "which transient is nearest this click" hit-testing).
fn interpolate_output_frame(markers: &[stretch::WarpMarker], source_frame: usize) -> usize {
    let Some(first) = markers.first() else {
        return source_frame;
    };
    if source_frame <= first.source_frame {
        return first.output_frame;
    }
    let last = markers.last().unwrap();
    if source_frame >= last.source_frame {
        return last.output_frame;
    }
    for pair in markers.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        if source_frame >= a.source_frame && source_frame <= b.source_frame {
            let span = (b.source_frame - a.source_frame).max(1) as f64;
            let frac = (source_frame - a.source_frame) as f64 / span;
            let out_span = b.output_frame as f64 - a.output_frame as f64;
            return (a.output_frame as f64 + frac * out_span).round() as usize;
        }
    }
    source_frame
}

/// The Time tab's own canvas: the raw waveform drawn piecewise per inter-marker span (each span's
/// slice of `raw_buffer` stretched/compressed, visually, to fill its own output-frame width — the
/// same "per-segment slice into its own rect" approach `playlist_contents_ui` already uses for
/// take-folder comp segments), detected transients as candidate snap points, and each warp
/// marker as a draggable vertical handle at its output position. Live drag feedback comes from
/// `marker_drag`, not the model — see `FlexMarkerDrag`'s doc comment on why the model (and the
/// clip's re-baked buffer) only update on `drag_stopped()`.
fn flex_time_tab_ui(
    ui: &mut egui::Ui,
    clip: &mut AudioClip,
    raw_buffer: &SampleBuffer,
    raw_len: usize,
    sample_rate: Option<u32>,
    marker_drag: &mut Option<FlexMarkerDrag>,
) {
    let available_width = ui.available_width().max(100.0);
    let markers = effective_warp_markers(clip, raw_len);
    let total_output = markers.last().map_or(raw_len, |m| m.output_frame).max(1);
    let px_per_output_frame = available_width / total_output as f32;

    let (response, painter) = ui.allocate_painter(
        egui::vec2(available_width, FLEX_WAVEFORM_HEIGHT),
        egui::Sense::click_and_drag(),
    );
    let rect = response.rect;
    painter.rect_filled(rect, 0u8, egui::Color32::from_gray(20));

    let display_markers: Vec<stretch::WarpMarker> = markers
        .iter()
        .enumerate()
        .map(|(i, m)| match marker_drag {
            Some(state) if state.marker_index == i => stretch::WarpMarker {
                source_frame: m.source_frame,
                output_frame: state.live_output_frame,
            },
            _ => *m,
        })
        .collect();

    for pair in display_markers.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        if b.output_frame <= a.output_frame || b.source_frame <= a.source_frame {
            continue;
        }
        let seg_rect = egui::Rect::from_min_size(
            rect.left_top() + egui::vec2(a.output_frame as f32 * px_per_output_frame, 0.0),
            egui::vec2(
                (b.output_frame - a.output_frame) as f32 * px_per_output_frame,
                FLEX_WAVEFORM_HEIGHT,
            ),
        );
        draw_audio_clip_waveform(&painter, seg_rect, raw_buffer, a.source_frame, b.source_frame);
    }

    let transients = transient_detection::detect_transients(&raw_buffer.mono, raw_buffer.sample_rate);
    for &t in &transients {
        let x = rect.left() + interpolate_output_frame(&display_markers, t) as f32 * px_per_output_frame;
        painter.line_segment(
            [
                egui::pos2(x, rect.bottom() - 10.0),
                egui::pos2(x, rect.bottom()),
            ],
            egui::Stroke::new(1.0, egui::Color32::from_rgba_unmultiplied(255, 220, 60, 160)),
        );
    }

    for marker in &display_markers {
        let x = rect.left() + marker.output_frame as f32 * px_per_output_frame;
        painter.line_segment(
            [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
            egui::Stroke::new(2.0, FL_ACCENT_ORANGE),
        );
    }

    let near_marker = |markers: &[stretch::WarpMarker], local_x: f32| {
        markers
            .iter()
            .enumerate()
            .map(|(i, m)| (i, (local_x - m.output_frame as f32 * px_per_output_frame).abs()))
            .filter(|&(_, dist)| dist <= RESIZE_HANDLE_PX)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| i)
    };

    if marker_drag.is_none() && response.drag_started() {
        if let Some(pos) = response.interact_pointer_pos() {
            let lx = pos.x - rect.left();
            if let Some(marker_index) = near_marker(&display_markers, lx) {
                ensure_warp_anchors(clip, raw_len);
                *marker_drag = Some(FlexMarkerDrag {
                    marker_index,
                    live_output_frame: clip.warp_markers[marker_index].output_frame,
                });
            }
        }
    }
    if let Some(state) = marker_drag {
        if response.dragged() {
            if let Some(pos) = response.interact_pointer_pos() {
                let lx = (pos.x - rect.left()).max(0.0);
                state.live_output_frame = (lx / px_per_output_frame).round().max(0.0) as usize;
            }
        }
        if response.drag_stopped() {
            let index = state.marker_index;
            let lower = if index > 0 {
                clip.warp_markers[index - 1].output_frame + 1
            } else {
                0
            };
            let upper = clip.warp_markers.get(index + 1).map(|m| m.output_frame.saturating_sub(1));
            let clamped = match upper {
                Some(u) => state.live_output_frame.clamp(lower, u.max(lower)),
                None => state.live_output_frame.max(lower),
            };
            clip.warp_markers[index].output_frame = clamped;
            clip.load(sample_rate.unwrap_or(48_000));
            *marker_drag = None;
        }
    }

    if response.clicked() {
        if let Some(pos) = response.interact_pointer_pos() {
            let lx = pos.x - rect.left();
            if near_marker(&display_markers, lx).is_none() {
                let click_output_frame = (lx / px_per_output_frame).round().max(0.0) as usize;
                let nearest = transients.iter().copied().min_by_key(|&t| {
                    interpolate_output_frame(&display_markers, t).abs_diff(click_output_frame)
                });
                if let Some(source_frame) = nearest {
                    let approx_output = interpolate_output_frame(&display_markers, source_frame);
                    let dist_px = (approx_output as f32 - click_output_frame as f32).abs() * px_per_output_frame;
                    if dist_px <= 40.0 {
                        ensure_warp_anchors(clip, raw_len);
                        let output_frame = interpolate_output_frame(&clip.warp_markers, source_frame);
                        clip.warp_markers.push(stretch::WarpMarker { source_frame, output_frame });
                        clip.warp_markers.sort_by_key(|m| m.source_frame);
                        clip.load(sample_rate.unwrap_or(48_000));
                    }
                }
            }
        }
    }

    if response.secondary_clicked() {
        if let Some(pos) = response.interact_pointer_pos() {
            let lx = pos.x - rect.left();
            if let Some(index) = near_marker(&display_markers, lx) {
                // The first/last markers are the span's own anchors — not deletable.
                if index != 0 && index != clip.warp_markers.len().saturating_sub(1) {
                    clip.warp_markers.remove(index);
                    clip.load(sample_rate.unwrap_or(48_000));
                }
            }
        }
    }
}

/// `note`'s pitch offset to draw/commit right now: the in-progress drag's live value if `note` is
/// the one being dragged, else whatever's already saved in `AudioClip::pitch_corrections`. A free
/// function (not a closure capturing `note_drag`) so `flex_pitch_tab_ui` can call it from inside
/// its drawing loop without holding any borrow of `note_drag` past that single call — it needs a
/// plain `&mut` on it again afterward, for the drag-continuation logic.
fn live_note_semitones(
    note: &pitch::DetectedNote,
    saved_semitones: f32,
    note_drag: Option<&FlexNoteDrag>,
    pointer_local_y: Option<f32>,
) -> f32 {
    match (note_drag, pointer_local_y) {
        (Some(state), Some(ly))
            if state.start_frame == note.start_frame && state.end_frame == note.end_frame =>
        {
            (state.start_semitones + (state.drag_start_y - ly) / PX_PER_SEMITONE)
                .round()
                .clamp(-MAX_PITCH_CORRECTION_SEMITONES, MAX_PITCH_CORRECTION_SEMITONES)
        }
        _ => saved_semitones,
    }
}

/// The Pitch tab's own canvas: `pitch::detect_notes` segments drawn as horizontal bars against the
/// raw waveform's own frame axis, vertically offset by their current (live-dragged or saved)
/// pitch correction around a center "no change" gridline. Dragging a bar vertically retargets that
/// note; the model only updates on `drag_stopped()` (see `FlexNoteDrag`'s doc comment).
fn flex_pitch_tab_ui(
    ui: &mut egui::Ui,
    clip: &mut AudioClip,
    raw_buffer: &SampleBuffer,
    sample_rate: Option<u32>,
    note_drag: &mut Option<FlexNoteDrag>,
) {
    let available_width = ui.available_width().max(100.0);
    let px_per_frame = available_width / raw_buffer.mono.len().max(1) as f32;
    let center_y = FLEX_PITCH_STRIP_HEIGHT / 2.0;

    let (response, painter) = ui.allocate_painter(
        egui::vec2(available_width, FLEX_PITCH_STRIP_HEIGHT),
        egui::Sense::click_and_drag(),
    );
    let rect = response.rect;
    painter.rect_filled(rect, 0u8, egui::Color32::from_gray(20));
    painter.line_segment(
        [
            egui::pos2(rect.left(), rect.top() + center_y),
            egui::pos2(rect.right(), rect.top() + center_y),
        ],
        egui::Stroke::new(1.0, egui::Color32::from_gray(60)),
    );

    let notes = pitch::detect_notes(&raw_buffer.mono, raw_buffer.sample_rate);
    let saved_semitones = |note: &pitch::DetectedNote| -> f32 {
        clip.pitch_corrections
            .iter()
            .find(|c| c.start_frame == note.start_frame && c.end_frame == note.end_frame)
            .map_or(0.0, |c| c.target_semitones)
    };
    let pointer_local_y = response.interact_pointer_pos().map(|p| p.y - rect.top());

    for note in &notes {
        let semitones = live_note_semitones(
            note,
            saved_semitones(note),
            note_drag.as_ref(),
            pointer_local_y,
        );
        let x = rect.left() + note.start_frame as f32 * px_per_frame;
        let w = ((note.end_frame - note.start_frame) as f32 * px_per_frame).max(2.0);
        let y = rect.top() + center_y - semitones * PX_PER_SEMITONE;
        let note_rect = egui::Rect::from_min_size(egui::pos2(x, y - 6.0), egui::vec2(w, 12.0));
        let color = if semitones == 0.0 { FL_ACCENT_GREEN } else { FL_ACCENT_ORANGE };
        painter.rect_filled(note_rect, 2u8, color);
    }

    let note_at = |lx: f32| {
        notes.iter().find(|n| {
            let x = n.start_frame as f32 * px_per_frame;
            let w = ((n.end_frame - n.start_frame) as f32 * px_per_frame).max(2.0);
            lx >= x && lx < x + w
        })
    };

    if note_drag.is_none() && response.drag_started() {
        if let Some(pos) = response.interact_pointer_pos() {
            let lx = pos.x - rect.left();
            let ly = pos.y - rect.top();
            if let Some(note) = note_at(lx) {
                *note_drag = Some(FlexNoteDrag {
                    start_frame: note.start_frame,
                    end_frame: note.end_frame,
                    start_semitones: saved_semitones(note),
                    drag_start_y: ly,
                });
            }
        }
    }
    if let Some(state) = note_drag {
        if response.drag_stopped() {
            let target = pointer_local_y.map_or(state.start_semitones, |ly| {
                (state.start_semitones + (state.drag_start_y - ly) / PX_PER_SEMITONE)
                    .round()
                    .clamp(-MAX_PITCH_CORRECTION_SEMITONES, MAX_PITCH_CORRECTION_SEMITONES)
            });
            match clip
                .pitch_corrections
                .iter_mut()
                .find(|c| c.start_frame == state.start_frame && c.end_frame == state.end_frame)
            {
                Some(existing) => existing.target_semitones = target,
                None if target != 0.0 => clip.pitch_corrections.push(pitch::PitchCorrection {
                    start_frame: state.start_frame,
                    end_frame: state.end_frame,
                    target_semitones: target,
                }),
                None => {}
            }
            clip.pitch_corrections.retain(|c| c.target_semitones != 0.0);
            clip.load(sample_rate.unwrap_or(48_000));
            *note_drag = None;
        }
    }
}

/// The Flex Time/Pitch editor window for whichever `AudioClip` `editor_target` names — opened from
/// that clip's right-click context menu (see `handle_audio_clip_interaction`). Loads its own
/// independent, unwarped/unshifted copy of the clip's decoded audio into `raw_cache` (keyed by
/// target, reloaded when it changes) since `AudioClip::buffer` is already the *edited* result once
/// `warp_markers`/`pitch_corrections` are set — the editor always places/drags things against the
/// original recording. Mirrors `take_folder_editor_window_ui`'s window-open/close pattern.
#[allow(clippy::too_many_arguments)]
pub(crate) fn flex_editor_window_ui(
    ctx: &egui::Context,
    song: &mut Song,
    sample_rate: Option<u32>,
    editor_target: &mut Option<(usize, usize)>,
    mode: &mut FlexEditorMode,
    raw_cache: &mut Option<((usize, usize), Arc<SampleBuffer>)>,
    marker_drag: &mut Option<FlexMarkerDrag>,
    note_drag: &mut Option<FlexNoteDrag>,
) {
    let Some((track_index, clip_index)) = *editor_target else {
        return;
    };
    let Some(file_path) = song
        .tracks
        .get(track_index)
        .and_then(|t| t.audio_clips.get(clip_index))
        .map(|c| c.file_path.clone())
    else {
        *editor_target = None;
        return;
    };

    let target = (track_index, clip_index);
    if raw_cache.as_ref().map(|(key, _)| *key) != Some(target) {
        let rate = sample_rate.unwrap_or(48_000);
        *raw_cache = SampleBuffer::load_wav_resampled(Path::new(&file_path), rate)
            .ok()
            .map(|buffer| (target, Arc::new(buffer)));
    }

    let mut open = true;
    egui::Window::new("Flex Time / Pitch")
        .id(egui::Id::new(("flex-editor", track_index, clip_index)))
        .collapsible(false)
        .resizable(true)
        .default_width(760.0)
        .open(&mut open)
        .show(ctx, |ui| {
            let Some((_, raw_buffer)) = raw_cache.clone() else {
                ui.weak("Couldn't decode this clip's audio file.");
                return;
            };
            let Some(clip) = song
                .tracks
                .get_mut(track_index)
                .and_then(|t| t.audio_clips.get_mut(clip_index))
            else {
                ui.weak("Clip no longer exists.");
                return;
            };
            let raw_len = raw_buffer.mono.len();

            ui.horizontal(|ui| {
                if ui.selectable_label(*mode == FlexEditorMode::Time, "Time").clicked() {
                    *mode = FlexEditorMode::Time;
                }
                if ui.selectable_label(*mode == FlexEditorMode::Pitch, "Pitch").clicked() {
                    *mode = FlexEditorMode::Pitch;
                }
            });

            match *mode {
                FlexEditorMode::Time => {
                    ui.weak(
                        "Click a yellow transient tick to add a warp point; drag an orange point \
                         to stretch the audio around it. Right-click a point to remove it.",
                    );
                    if ui.button("Reset (remove all warp points)").clicked() {
                        clip.warp_markers.clear();
                        clip.load(sample_rate.unwrap_or(48_000));
                    }
                    flex_time_tab_ui(ui, clip, &raw_buffer, raw_len, sample_rate, marker_drag);
                }
                FlexEditorMode::Pitch => {
                    ui.weak("Drag a detected note up/down to retarget its pitch.");
                    if ui.button("Reset (remove all pitch corrections)").clicked() {
                        clip.pitch_corrections.clear();
                        clip.load(sample_rate.unwrap_or(48_000));
                    }
                    flex_pitch_tab_ui(ui, clip, &raw_buffer, sample_rate, note_drag);
                }
            }
        });
    if !open {
        *editor_target = None;
        *marker_drag = None;
        *note_drag = None;
    }
}

/// Session View's counterpart of `flex_editor_window_ui` — same window shell and the exact same
/// `flex_time_tab_ui`/`flex_pitch_tab_ui` tab-rendering (both already operate on a plain
/// `&mut AudioClip`, addressing-agnostic), just resolving `editor_target`'s `(track_index,
/// slot_index)` into `Track::session_clips[slot_index]`'s `SessionClipContent::Audio` clip
/// instead of a Playlist `Track::audio_clips` entry. Opened from that slot's right-click "Flex
/// Time / Pitch…" context-menu entry (see `session_view_ui::session_slot_cell_ui`) — never shown
/// for a `SessionClipContent::Region`/`Recording` slot, neither of which has a plain `AudioClip`
/// to edit (a `Recording`'s `TakeFolder` has no Flex editor of its own in v1 — see that variant's
/// doc comment).
#[allow(clippy::too_many_arguments)]
pub(crate) fn session_flex_editor_window_ui(
    ctx: &egui::Context,
    song: &mut Song,
    sample_rate: Option<u32>,
    editor_target: &mut Option<(usize, usize)>,
    mode: &mut FlexEditorMode,
    raw_cache: &mut Option<((usize, usize), Arc<SampleBuffer>)>,
    marker_drag: &mut Option<FlexMarkerDrag>,
    note_drag: &mut Option<FlexNoteDrag>,
    track_effect_slots: &TrackEffectSlots,
    send_effect_slots: &SendEffectSlots,
    master_effect_slots: &MasterEffectSlots,
    automation_drag: &mut Option<AutomationDrag>,
) {
    let Some((track_index, slot_index)) = *editor_target else {
        return;
    };
    let Some(file_path) = song
        .tracks
        .get(track_index)
        .and_then(|t| t.session_clips.get(slot_index))
        .and_then(|slot| slot.as_ref())
        .and_then(|clip| match &clip.content {
            SessionClipContent::Audio(audio) => Some(audio.file_path.clone()),
            SessionClipContent::Region { .. } | SessionClipContent::Recording(_) => None,
        })
    else {
        *editor_target = None;
        return;
    };

    let target = (track_index, slot_index);
    if raw_cache.as_ref().map(|(key, _)| *key) != Some(target) {
        let rate = sample_rate.unwrap_or(48_000);
        *raw_cache = SampleBuffer::load_wav_resampled(Path::new(&file_path), rate)
            .ok()
            .map(|buffer| (target, Arc::new(buffer)));
    }

    let mut open = true;
    egui::Window::new("Flex Time / Pitch (Session View)")
        .id(egui::Id::new(("session-flex-editor", track_index, slot_index)))
        .collapsible(false)
        .resizable(true)
        .default_width(760.0)
        .open(&mut open)
        .show(ctx, |ui| {
            let Some((_, raw_buffer)) = raw_cache.clone() else {
                ui.weak("Couldn't decode this clip's audio file.");
                return;
            };
            // Same pre-borrow snapshot as `piano_roll_contents_ui`'s automation panel, for the
            // same reason — `automation_lanes_ui`'s "Other Track" targets need every track.
            let other_tracks_snapshot: Vec<(String, Vec<TrackEffectConfig>)> = song
                .tracks
                .iter()
                .map(|t| (t.name.clone(), t.effects.clone()))
                .collect();
            let track_effects_snapshot =
                song.tracks.get(track_index).map(|t| t.effects.clone()).unwrap_or_default();
            let ticks_per_second = audio::ticks_per_second(song.bpm);
            let Some(session_clip) = song
                .tracks
                .get_mut(track_index)
                .and_then(|t| t.session_clips.get_mut(slot_index))
                .and_then(|slot| slot.as_mut())
            else {
                ui.weak("Clip no longer exists.");
                return;
            };
            let SessionClipContent::Audio(clip) = &mut session_clip.content else {
                ui.weak("Clip no longer exists.");
                return;
            };
            let raw_len = raw_buffer.mono.len();
            let clip_span_ticks = clip.effective_length_ticks(ticks_per_second);

            ui.horizontal(|ui| {
                if ui.selectable_label(*mode == FlexEditorMode::Time, "Time").clicked() {
                    *mode = FlexEditorMode::Time;
                }
                if ui.selectable_label(*mode == FlexEditorMode::Pitch, "Pitch").clicked() {
                    *mode = FlexEditorMode::Pitch;
                }
            });

            match *mode {
                FlexEditorMode::Time => {
                    ui.weak(
                        "Click a yellow transient tick to add a warp point; drag an orange point \
                         to stretch the audio around it. Right-click a point to remove it.",
                    );
                    if ui.button("Reset (remove all warp points)").clicked() {
                        clip.warp_markers.clear();
                        clip.load(sample_rate.unwrap_or(48_000));
                    }
                    flex_time_tab_ui(ui, clip, &raw_buffer, raw_len, sample_rate, marker_drag);
                }
                FlexEditorMode::Pitch => {
                    ui.weak("Drag a detected note up/down to retarget its pitch.");
                    if ui.button("Reset (remove all pitch corrections)").clicked() {
                        clip.pitch_corrections.clear();
                        clip.load(sample_rate.unwrap_or(48_000));
                    }
                    flex_pitch_tab_ui(ui, clip, &raw_buffer, sample_rate, note_drag);
                }
            }

            ui.separator();
            egui::ScrollArea::vertical().max_height(110.0).show(ui, |ui| {
                automation_lanes_ui(
                    ui,
                    &mut session_clip.automation,
                    clip_span_ticks,
                    track_index,
                    &track_effects_snapshot,
                    track_effect_slots,
                    &other_tracks_snapshot,
                    &song.sends,
                    send_effect_slots,
                    &song.master_effects,
                    master_effect_slots,
                    1.0,
                    automation_drag,
                );
            });
        });
    if !open {
        *editor_target = None;
        *marker_drag = None;
        *note_drag = None;
        *automation_drag = None;
    }
}

/// Hit-tests and applies click/drag/right-click gestures against every `Audio`-kind track's
/// `audio_clips`, rendered in the same Playlist canvas as `handle_playlist_interaction` but in the
/// rows below it (`audio_rows_top` onward — see `playlist_contents_ui`). Mirrors
/// `handle_playlist_interaction`'s structure (click/drag_started/dragged/drag_stopped) but for
/// clips instead of regions, and with no `Create` mode — a clip is only ever created by
/// recording/import, never drawn out on the timeline. Right-clicking a clip opens a context menu
/// ("Strip Silence"/"Delete"/"Flex Time / Pitch…" — see `apply_strip_silence` and
/// `flex_editor_window_ui`) instead of deleting immediately.
#[allow(clippy::too_many_arguments)]
pub(crate) fn handle_audio_clip_interaction(
    response: &egui::Response,
    rect: egui::Rect,
    song: &mut Song,
    audio_track_indices: &[usize],
    audio_rows_top: f32,
    drag: &mut Option<AudioClipDrag>,
    context_menu_target: &mut Option<AudioClipContextMenuTarget>,
    flex_editor: &mut Option<(usize, usize)>,
    zoom: f32,
) {
    let local = |p: egui::Pos2| (p.x - rect.left(), p.y - rect.top());
    let row_count = audio_track_indices.len();
    let y_to_track_row = |y: f32| -> Option<usize> {
        if y < audio_rows_top {
            return None;
        }
        let row = ((y - audio_rows_top) / PLAYLIST_LANE_HEIGHT)
            .floor()
            .max(0.0) as usize;
        (row < row_count).then_some(row)
    };
    let row_frac_at = |ly: f32, row: usize| -> f32 {
        ((ly - audio_rows_top) / PLAYLIST_LANE_HEIGHT) - row as f32
    };
    // Snapshot before any `&mut song.tracks[...]` borrow below, mirroring the same pre-borrow
    // pattern used elsewhere (e.g. `piano_roll_contents_ui`'s `other_tracks_snapshot`) — a plain
    // closure calling `song.bpm_at(...)` here would hold `song` captured for this whole
    // function's remaining borrows, conflicting with those later mutable ones.
    let (base_bpm, tempo_map) = (song.bpm, song.tempo_map.clone());
    let bpm_at_tick = |tick: usize| -> f32 {
        tempo_map
            .iter()
            .rev()
            .find(|point| point.tick <= tick)
            .map_or(base_bpm, |point| point.bpm)
    };
    let clip_span_ticks = |c: &AudioClip| {
        audio_clip_length_ticks(c, audio::ticks_per_second(bpm_at_tick(c.start_tick)))
    };
    let clip_at = |clips: &[AudioClip], tick: usize| {
        clips
            .iter()
            .position(|c| tick >= c.start_tick && tick < c.start_tick + clip_span_ticks(c))
    };
    // Trim/fade handles sit at the clip's own left/right edges or, for fades, at the point its
    // ramp ends — see the matching drawing code in `playlist_contents_ui`
    // (`draw_audio_clip_fade_overlays`). Fade handles are restricted to the top half of the row
    // (`row_frac`) so a fade handle at fade_*_ticks == 0 (sitting right at the clip's corner)
    // doesn't shadow the whole-height trim/move hit-tests below, mirroring
    // `handle_playlist_interaction`'s `near_fade_in_handle`/`near_fade_out_handle`.
    let near_trim_start_handle = |clip: &AudioClip, local_x: f32| {
        (local_x - tick_to_x(clip.start_tick, zoom)).abs() <= RESIZE_HANDLE_PX
    };
    let near_trim_end_handle = |clip: &AudioClip, local_x: f32| {
        let end_x = tick_to_x(clip.start_tick + clip_span_ticks(clip), zoom);
        (local_x - end_x).abs() <= RESIZE_HANDLE_PX
    };
    let near_fade_in_handle = |clip: &AudioClip, local_x: f32| {
        let span_ticks = clip_span_ticks(clip);
        let fade_ticks = clip.fade_in_ticks.min(span_ticks);
        let x = tick_to_x(clip.start_tick + fade_ticks, zoom);
        (local_x - x).abs() <= RESIZE_HANDLE_PX
    };
    let near_fade_out_handle = |clip: &AudioClip, local_x: f32| {
        let span_ticks = clip_span_ticks(clip);
        let fade_ticks = clip.fade_out_ticks.min(span_ticks);
        let x = tick_to_x(clip.start_tick + span_ticks - fade_ticks, zoom);
        (local_x - x).abs() <= RESIZE_HANDLE_PX
    };

    if response.secondary_clicked() {
        *context_menu_target = response.interact_pointer_pos().and_then(|pos| {
            let (lx, ly) = local(pos);
            let row = y_to_track_row(ly)?;
            let track_index = audio_track_indices[row];
            let clip_index = clip_at(&song.tracks[track_index].audio_clips, x_to_tick(lx, zoom))?;
            Some(AudioClipContextMenuTarget {
                track_index,
                clip_index,
            })
        });
    }

    // Rendered every frame (not gated on `secondary_clicked()`) since `egui::Response::context_menu`
    // owns its own open/close state internally, keyed off `response`'s id — it needs to be called
    // every frame to keep drawing an already-open menu, not just on the click that opened it.
    // `context_menu_target` (set above) says which clip it's acting on; empty when the right-click
    // that opened it didn't land on a clip.
    response.context_menu(|ui| {
        let Some(target) = *context_menu_target else {
            return;
        };
        if ui.button("Strip Silence").clicked() {
            apply_strip_silence(song, target.track_index, target.clip_index);
            ui.close();
        }
        if ui.button("Flex Time / Pitch…").clicked() {
            *flex_editor = Some((target.track_index, target.clip_index));
            ui.close();
        }
        if ui.button("Delete").clicked() {
            if let Some(track) = song.tracks.get_mut(target.track_index) {
                if target.clip_index < track.audio_clips.len() {
                    track.audio_clips.remove(target.clip_index);
                }
            }
            ui.close();
        }
    });

    if drag.is_none() && response.drag_started() {
        if let Some(pos) = response.interact_pointer_pos() {
            let (lx, ly) = local(pos);
            if let Some(row) = y_to_track_row(ly) {
                let track_index = audio_track_indices[row];
                let row_frac = row_frac_at(ly, row);
                let tick = x_to_tick(lx, zoom);
                let clips = &song.tracks[track_index].audio_clips;
                let hovered_clip = clip_at(clips, tick);
                if let Some(clip_index) = hovered_clip
                    .filter(|&i| row_frac <= 0.5 && near_fade_in_handle(&clips[i], lx))
                {
                    *drag = Some(AudioClipDrag {
                        mode: AudioClipDragMode::FadeIn {
                            track_index,
                            clip_index,
                        },
                    });
                } else if let Some(clip_index) = hovered_clip
                    .filter(|&i| row_frac <= 0.5 && near_fade_out_handle(&clips[i], lx))
                {
                    *drag = Some(AudioClipDrag {
                        mode: AudioClipDragMode::FadeOut {
                            track_index,
                            clip_index,
                        },
                    });
                } else if let Some(clip_index) =
                    hovered_clip.filter(|&i| near_trim_end_handle(&clips[i], lx))
                {
                    *drag = Some(AudioClipDrag {
                        mode: AudioClipDragMode::TrimEnd {
                            track_index,
                            clip_index,
                        },
                    });
                } else if let Some(clip_index) =
                    hovered_clip.filter(|&i| near_trim_start_handle(&clips[i], lx))
                {
                    *drag = Some(AudioClipDrag {
                        mode: AudioClipDragMode::TrimStart {
                            track_index,
                            clip_index,
                        },
                    });
                } else if let Some(clip_index) = hovered_clip {
                    let grab_tick_offset = tick as i64 - clips[clip_index].start_tick as i64;
                    *drag = Some(AudioClipDrag {
                        mode: AudioClipDragMode::Move {
                            track_index,
                            clip_index,
                            grab_tick_offset,
                        },
                    });
                }
            }
        }
    }

    if let Some(state) = drag {
        let (track_index, clip_index) = match &state.mode {
            AudioClipDragMode::Move {
                track_index,
                clip_index,
                ..
            }
            | AudioClipDragMode::TrimStart {
                track_index,
                clip_index,
            }
            | AudioClipDragMode::TrimEnd {
                track_index,
                clip_index,
            }
            | AudioClipDragMode::FadeIn {
                track_index,
                clip_index,
            }
            | AudioClipDragMode::FadeOut {
                track_index,
                clip_index,
            } => (*track_index, *clip_index),
        };
        let clips = song
            .tracks
            .get_mut(track_index)
            .map(|t| &mut t.audio_clips);
        let Some(clips) = clips.filter(|c| clip_index < c.len()) else {
            // The clip behind this drag was removed mid-drag (right-click) — drop the dangling state.
            *drag = None;
            return;
        };
        match &state.mode {
            AudioClipDragMode::Move {
                grab_tick_offset, ..
            } => {
                let grab_tick_offset = *grab_tick_offset;
                if response.dragged() {
                    if let Some(pos) = response.interact_pointer_pos() {
                        let (lx, _ly) = local(pos);
                        let tick = x_to_tick(lx.max(0.0), zoom) as i64;
                        clips[clip_index].start_tick = (tick - grab_tick_offset).max(0) as usize;
                    }
                }
                if response.drag_stopped() {
                    *drag = None;
                }
            }
            AudioClipDragMode::TrimStart { .. } => {
                if response.dragged() {
                    if let Some(pos) = response.interact_pointer_pos() {
                        let (lx, _ly) = local(pos);
                        let clip = &mut clips[clip_index];
                        if let Some(buffer) = clip.buffer.clone() {
                            let tps = audio::ticks_per_second(bpm_at_tick(clip.start_tick));
                            let frames_per_tick = buffer.sample_rate as f64 / tps;
                            let old_start_tick = clip.start_tick;
                            let end_tick = old_start_tick + clip.effective_length_ticks(tps);
                            let new_start_tick =
                                x_to_tick(lx.max(0.0), zoom).min(end_tick.saturating_sub(1));
                            let delta_ticks = new_start_tick as i64 - old_start_tick as i64;
                            let delta_frames =
                                (delta_ticks as f64 * frames_per_tick).round() as i64;
                            clip.source_start_frame =
                                (clip.source_start_frame as i64 + delta_frames).max(0) as usize;
                            clip.start_tick = new_start_tick;
                            clip.length_ticks = end_tick.saturating_sub(new_start_tick).max(1);
                        }
                    }
                }
                if response.drag_stopped() {
                    *drag = None;
                }
            }
            AudioClipDragMode::TrimEnd { .. } => {
                if response.dragged() {
                    if let Some(pos) = response.interact_pointer_pos() {
                        let (lx, _ly) = local(pos);
                        let clip = &mut clips[clip_index];
                        let tps = audio::ticks_per_second(bpm_at_tick(clip.start_tick));
                        let max_ticks = clip.full_length_ticks(tps).max(1);
                        let tick = x_to_tick(lx.max(0.0), zoom);
                        clip.length_ticks =
                            tick.saturating_sub(clip.start_tick).clamp(1, max_ticks);
                    }
                }
                if response.drag_stopped() {
                    *drag = None;
                }
            }
            AudioClipDragMode::FadeIn { .. } => {
                if response.dragged() {
                    if let Some(pos) = response.interact_pointer_pos() {
                        let (lx, _ly) = local(pos);
                        let tick = x_to_tick(lx.max(0.0), zoom);
                        let clip = &mut clips[clip_index];
                        let tps = audio::ticks_per_second(bpm_at_tick(clip.start_tick));
                        let span_ticks = clip.effective_length_ticks(tps);
                        let offset = tick.saturating_sub(clip.start_tick);
                        clip.fade_in_ticks = offset.min(span_ticks);
                    }
                }
                if response.drag_stopped() {
                    *drag = None;
                }
            }
            AudioClipDragMode::FadeOut { .. } => {
                if response.dragged() {
                    if let Some(pos) = response.interact_pointer_pos() {
                        let (lx, _ly) = local(pos);
                        let tick = x_to_tick(lx.max(0.0), zoom);
                        let clip = &mut clips[clip_index];
                        let tps = audio::ticks_per_second(bpm_at_tick(clip.start_tick));
                        let span_ticks = clip.effective_length_ticks(tps);
                        let end_tick = clip.start_tick + span_ticks;
                        clip.fade_out_ticks = end_tick.saturating_sub(tick).min(span_ticks);
                    }
                }
                if response.drag_stopped() {
                    *drag = None;
                }
            }
        }
    }
}

/// Hit-tests right-clicks against every `Audio`-kind track's `take_folders`, in the same row
/// `handle_audio_clip_interaction` uses for that track's `audio_clips` (see
/// `playlist_contents_ui`). Right-clicking a folder opens a context menu to pick which take is
/// comped for the whole folder, or delete it; double-clicking opens the segment-level comp editor
/// (`take_folder_editor_window_ui`) — no move/trim drag yet, unlike plain audio clips.
#[allow(clippy::too_many_arguments)]
pub(crate) fn handle_take_folder_interaction(
    response: &egui::Response,
    rect: egui::Rect,
    song: &mut Song,
    audio_track_indices: &[usize],
    audio_rows_top: f32,
    context_menu_target: &mut Option<TakeFolderContextMenuTarget>,
    editor_target: &mut Option<(usize, usize)>,
    zoom: f32,
) {
    let local = |p: egui::Pos2| (p.x - rect.left(), p.y - rect.top());
    let row_count = audio_track_indices.len();
    let y_to_track_row = |y: f32| -> Option<usize> {
        if y < audio_rows_top {
            return None;
        }
        let row = ((y - audio_rows_top) / PLAYLIST_LANE_HEIGHT)
            .floor()
            .max(0.0) as usize;
        (row < row_count).then_some(row)
    };
    let folder_at = |folders: &[TakeFolder], tick: usize| {
        folders
            .iter()
            .position(|f| tick >= f.start_tick && tick < f.start_tick + f.length_ticks)
    };
    let hit_test = |pos: egui::Pos2| -> Option<(usize, usize)> {
        let (lx, ly) = local(pos);
        let row = y_to_track_row(ly)?;
        let track_index = audio_track_indices[row];
        let folder_index = folder_at(&song.tracks[track_index].take_folders, x_to_tick(lx, zoom))?;
        Some((track_index, folder_index))
    };

    if response.secondary_clicked() {
        *context_menu_target = response
            .interact_pointer_pos()
            .and_then(hit_test)
            .map(|(track_index, folder_index)| TakeFolderContextMenuTarget {
                track_index,
                folder_index,
            });
    }
    if response.double_clicked() {
        if let Some(pos) = response.interact_pointer_pos() {
            if let Some(target) = hit_test(pos) {
                *editor_target = Some(target);
            }
        }
    }

    // Rendered every frame — see `handle_audio_clip_interaction`'s identical comment on why.
    response.context_menu(|ui| {
        let Some(target) = *context_menu_target else {
            return;
        };
        let Some(folder) = song
            .tracks
            .get_mut(target.track_index)
            .and_then(|t| t.take_folders.get_mut(target.folder_index))
        else {
            return;
        };
        let active_take_index = folder.comp.first().map_or(0, |s| s.take_index);
        for take_index in 0..folder.takes.len() {
            let label = if take_index == active_take_index {
                format!("\u{2713} Take {}", take_index + 1)
            } else {
                format!("Take {}", take_index + 1)
            };
            if ui.button(label).clicked() {
                folder.set_active_take(take_index);
                ui.close();
            }
        }
        ui.separator();
        if ui.button("Delete").clicked() {
            if let Some(track) = song.tracks.get_mut(target.track_index) {
                if target.folder_index < track.take_folders.len() {
                    track.take_folders.remove(target.folder_index);
                }
            }
            ui.close();
        }
    });
}

/// Height of one take's lane in the comp editor's stacked-lanes canvas — see
/// `take_folder_editor_window_ui`.
const TAKE_LANE_HEIGHT: f32 = 48.0;

/// The segment-level "quick-swipe" comp editor for whichever take folder `editor_target` names —
/// opened by double-clicking a take folder in the Playlist (see `handle_take_folder_interaction`).
/// One horizontal lane per take, each showing that take's own full waveform across the folder's
/// span; the current `comp` is drawn as a bright outline over whichever lane/stretch it currently
/// points at. Dragging horizontally within a lane reassigns that stretch to that lane's take, live,
/// via `TakeFolder::assign_take_to_range` — mirrors the window-open/close and canvas-drag patterns
/// already used elsewhere (`self.effect_editor`'s "FX Params" `egui::Window`,
/// `handle_audio_clip_interaction`'s live-drag-then-`drag_stopped()` pattern), rather than
/// introducing a new one.
pub(crate) fn take_folder_editor_window_ui(
    ctx: &egui::Context,
    song: &mut Song,
    editor_target: &mut Option<(usize, usize)>,
    comp_drag: &mut Option<TakeFolderCompDrag>,
) {
    let Some((track_index, folder_index)) = *editor_target else {
        return;
    };
    let mut open = true;
    egui::Window::new("Take Folder")
        .id(egui::Id::new(("take-folder-editor", track_index, folder_index)))
        .collapsible(false)
        .resizable(true)
        .default_width(700.0)
        .open(&mut open)
        .show(ctx, |ui| {
            let Some(start_tick) = song
                .tracks
                .get(track_index)
                .and_then(|t| t.take_folders.get(folder_index))
                .map(|f| f.start_tick)
            else {
                ui.weak("Take folder no longer exists.");
                *comp_drag = None;
                return;
            };
            ui.weak("Drag across a take's lane to comp that stretch of the folder to it.");
            let folder_ticks_per_second = audio::ticks_per_second(song.bpm_at(start_tick));
            let folder = &mut song.tracks[track_index].take_folders[folder_index];
            let take_count = folder.takes.len().max(1);
            let available_width = ui.available_width().max(100.0);
            let px_per_tick = available_width / folder.length_ticks.max(1) as f32;

            let (response, painter) = ui.allocate_painter(
                egui::vec2(available_width, TAKE_LANE_HEIGHT * take_count as f32),
                egui::Sense::click_and_drag(),
            );
            let rect = response.rect;

            for (take_index, take) in folder.takes.iter().enumerate() {
                let lane_rect = egui::Rect::from_min_size(
                    rect.left_top() + egui::vec2(0.0, take_index as f32 * TAKE_LANE_HEIGHT),
                    egui::vec2(available_width, TAKE_LANE_HEIGHT),
                );
                painter.rect_filled(lane_rect, 0u8, egui::Color32::from_gray(30));
                if let Some(buffer) = &take.buffer {
                    let frames_per_tick = buffer.sample_rate as f64 / folder_ticks_per_second;
                    let end_frame = (folder.length_ticks as f64 * frames_per_tick).round() as usize;
                    draw_audio_clip_waveform(&painter, lane_rect, buffer, 0, end_frame);
                }
                painter.rect_stroke(
                    lane_rect,
                    0u8,
                    egui::Stroke::new(1.0, egui::Color32::from_gray(70)),
                    egui::StrokeKind::Inside,
                );
                painter.text(
                    lane_rect.left_top() + egui::vec2(3.0, 2.0),
                    egui::Align2::LEFT_TOP,
                    format!("Take {}", take_index + 1),
                    egui::FontId::proportional(9.0),
                    egui::Color32::LIGHT_GRAY,
                );
            }
            // The current comp, on top of the lanes: a bright outline over each segment's own
            // take's lane, at that segment's own tick range.
            for segment in &folder.comp {
                if segment.take_index >= take_count {
                    continue;
                }
                let seg_rect = egui::Rect::from_min_size(
                    rect.left_top()
                        + egui::vec2(
                            segment.start_tick as f32 * px_per_tick,
                            segment.take_index as f32 * TAKE_LANE_HEIGHT,
                        ),
                    egui::vec2(
                        (segment.end_tick - segment.start_tick) as f32 * px_per_tick,
                        TAKE_LANE_HEIGHT,
                    ),
                );
                painter.rect_stroke(
                    seg_rect,
                    0u8,
                    egui::Stroke::new(2.0, FL_ACCENT_GREEN),
                    egui::StrokeKind::Inside,
                );
            }

            let local = |p: egui::Pos2| (p.x - rect.left(), p.y - rect.top());
            let tick_at =
                |lx: f32| ((lx / px_per_tick).round().max(0.0) as usize).min(folder.length_ticks);
            let take_at = |ly: f32| {
                ((ly / TAKE_LANE_HEIGHT).floor().max(0.0) as usize).min(take_count.saturating_sub(1))
            };

            if comp_drag.is_none() && response.drag_started() {
                if let Some(pos) = response.interact_pointer_pos() {
                    let (lx, ly) = local(pos);
                    *comp_drag = Some(TakeFolderCompDrag {
                        take_index: take_at(ly),
                        start_tick: tick_at(lx),
                    });
                }
            }
            if let Some(state) = comp_drag {
                if response.dragged() {
                    if let Some(pos) = response.interact_pointer_pos() {
                        let (lx, _ly) = local(pos);
                        let end_tick = tick_at(lx);
                        let (lo, hi) = if state.start_tick <= end_tick {
                            (state.start_tick, end_tick)
                        } else {
                            (end_tick, state.start_tick)
                        };
                        folder.assign_take_to_range(state.take_index, lo, hi);
                    }
                }
                if response.drag_stopped() {
                    *comp_drag = None;
                }
            }
        });
    if !open {
        *editor_target = None;
        *comp_drag = None;
    }
}
