//! The Playlist: the song arrangement timeline — one lane per track, showing piano-roll/step-grid
//! Region blocks, AudioClip blocks, and TakeFolder blocks, plus the Tempo Track panel.
//! `playlist_contents_ui` is the entry point (shared by the docked and detached-window
//! renderings, see `SimpleDawApp::update`); `handle_playlist_interaction` is its drag/click/
//! context-menu logic for Region blocks specifically (`handle_audio_clip_interaction`/
//! `handle_take_folder_interaction`, not yet extracted, handle their own block kinds and are
//! called from here). `draw_audio_clip_waveform` is pub(crate) since the Flex editor and Take
//! Folder comp editor (also not yet extracted) reuse it for their own waveform drawing.

use std::path::Path;

use crate::audio;
use crate::model::{AudioClip, Region, Song, TICKS_PER_STEP, TrackKind};
use crate::sample::SampleBuffer;
use crate::transient_detection;
use crate::{
    AudioClipContextMenuTarget, AudioClipDrag, FL_ACCENT_GREEN, PLAYLIST_LANE_HEIGHT, RESIZE_HANDLE_PX,
    RegionEditTarget, TakeFolderContextMenuTarget, audio_clip_length_ticks, draw_region_note_preview,
    handle_audio_clip_interaction, handle_take_folder_interaction, tick_to_x, track_color, x_to_tick,
};

/// Playlist timeline zoom range — same shape as `PIANO_ROLL_ZOOM_MIN`/`MAX`, a separate range
/// since the Playlist is a different view with its own reasonable default scale.
const PLAYLIST_ZOOM_MIN: f32 = 0.25;
const PLAYLIST_ZOOM_MAX: f32 = 3.0;
/// Height of the Playlist canvas's bar/step ruler row, and of each pattern-placement lane below it.
const PLAYLIST_RULER_HEIGHT: f32 = 20.0;
/// Width of the Playlist's fixed (non-scrolling) row-header column — the named/colored labels
/// down the left side, FL Studio–style, that stay put while the timeline canvas scrolls under them.
const PLAYLIST_HEADER_WIDTH: f32 = 120.0;
/// Playlist block fill for a `TakeFolder` — distinct from a plain `AudioClip`'s `track_color` fill
/// so a recording that can be re-comped is visually distinguishable at a glance from an import.
const TAKE_FOLDER_COLOR: egui::Color32 = egui::Color32::from_rgb(196, 152, 219);

/// What the currently in-progress Playlist drag (if any) is doing — the region counterpart of
/// `PianoRollDragMode`. A region is addressed by which track owns it plus its index into that
/// track's own `regions` (rather than a stable id, unlike `Note::id`); every arm below re-checks
/// that index is still in bounds before using it, in case the region was removed (right-click)
/// since the drag began. There's no cross-track drag — a region's `track_index` never changes
/// once created, only its position/span within that one row.
enum PlaylistDragMode {
    /// Dragging an existing region's body: changes `start_tick` only.
    Move {
        track_index: usize,
        region_index: usize,
        grab_step_offset: i64,
    },
    /// Dragging an existing region's right edge: changes `loop_length_steps` only.
    Resize {
        track_index: usize,
        region_index: usize,
    },
    /// Drawing a brand-new region out from a click on empty space.
    Create {
        track_index: usize,
        region_index: usize,
    },
    /// Dragging the fade-in handle: changes `Region::fade_in_ticks` only.
    FadeIn {
        track_index: usize,
        region_index: usize,
    },
    /// Dragging the fade-out handle: changes `Region::fade_out_ticks` only.
    FadeOut {
        track_index: usize,
        region_index: usize,
    },
}

pub(crate) struct PlaylistDrag {
    mode: PlaylistDragMode,
}

/// The Piano Roll's/Beats' "which region is open" state, bundled so `handle_playlist_interaction`
/// can set either pair on a double-click without a long individual-borrow parameter list. Setting
/// `selected_track`/`piano_roll_region` (or the Beats equivalent) is the *only* way either editor
/// window opens or changes which region it shows — there's no in-window picker, and the Channel
/// Rack has no "open editor" button; see `playlist_contents_ui`'s doc comment.
pub(crate) struct PlaylistEditorTargets<'a> {
    pub(crate) selected_track: &'a mut Option<usize>,
    pub(crate) piano_roll_region: &'a mut Option<RegionEditTarget>,
    /// See `SimpleDawApp::piano_roll_scroll_to`. Set alongside `piano_roll_region` on a
    /// double-click, to the content-local tick under the click.
    pub(crate) piano_roll_scroll_to: &'a mut Option<usize>,
    pub(crate) selected_beats_track: &'a mut Option<usize>,
    pub(crate) beats_region: &'a mut Option<RegionEditTarget>,
}

/// Draws `region`'s fade-in/fade-out ramps as the usual DAW convention: a semi-transparent
/// triangular wedge over the faded portion of the clip, tapering from full shade at the region's
/// own edge down to none at the point `region.fade_gain_at` reaches 1.0 — dragging that point
/// (see `handle_playlist_interaction`'s `near_fade_in_handle`/`near_fade_out_handle`) is how
/// `fade_in_ticks`/`fade_out_ticks` get set in the first place. Draws nothing for a fade of 0.
fn draw_region_fade_overlays(
    painter: &egui::Painter,
    rect: egui::Rect,
    region: &Region,
    zoom: f32,
) {
    let span_ticks = region.loop_length_steps * TICKS_PER_STEP;
    if span_ticks == 0 {
        return;
    }
    let fade_shade = egui::Color32::from_black_alpha(110);
    if region.fade_in_ticks > 0 {
        let fade_w = tick_to_x(region.fade_in_ticks.min(span_ticks), zoom).min(rect.width());
        painter.add(egui::Shape::convex_polygon(
            vec![
                rect.left_top(),
                egui::pos2(rect.left() + fade_w, rect.top()),
                rect.left_bottom(),
            ],
            fade_shade,
            egui::Stroke::NONE,
        ));
    }
    if region.fade_out_ticks > 0 {
        let fade_w = tick_to_x(region.fade_out_ticks.min(span_ticks), zoom).min(rect.width());
        painter.add(egui::Shape::convex_polygon(
            vec![
                rect.right_top(),
                egui::pos2(rect.right() - fade_w, rect.top()),
                rect.right_bottom(),
            ],
            fade_shade,
            egui::Stroke::NONE,
        ));
    }
}

/// Draws `clip`'s fade-in/fade-out ramps, the `AudioClip` counterpart of
/// `draw_region_fade_overlays` — same wedge convention, but against `span_ticks`
/// (`AudioClip::effective_length_ticks`) rather than a region's `loop_length_steps`. Dragging the
/// point where the wedge ends (see `handle_audio_clip_interaction`'s `near_fade_in_handle`/
/// `near_fade_out_handle`) is how `fade_in_ticks`/`fade_out_ticks` get set. Draws nothing for a
/// fade of 0.
fn draw_audio_clip_fade_overlays(
    painter: &egui::Painter,
    rect: egui::Rect,
    clip: &AudioClip,
    span_ticks: usize,
    zoom: f32,
) {
    if span_ticks == 0 {
        return;
    }
    let fade_shade = egui::Color32::from_black_alpha(110);
    if clip.fade_in_ticks > 0 {
        let fade_w = tick_to_x(clip.fade_in_ticks.min(span_ticks), zoom).min(rect.width());
        painter.add(egui::Shape::convex_polygon(
            vec![
                rect.left_top(),
                egui::pos2(rect.left() + fade_w, rect.top()),
                rect.left_bottom(),
            ],
            fade_shade,
            egui::Stroke::NONE,
        ));
    }
    if clip.fade_out_ticks > 0 {
        let fade_w = tick_to_x(clip.fade_out_ticks.min(span_ticks), zoom).min(rect.width());
        painter.add(egui::Shape::convex_polygon(
            vec![
                rect.right_top(),
                egui::pos2(rect.right() - fade_w, rect.top()),
                rect.right_bottom(),
            ],
            fade_shade,
            egui::Stroke::NONE,
        ));
    }
}

/// Draws a Logic-style min/max waveform for an `Audio`-track clip's trimmed window
/// (`start_frame..end_frame` into the decoded buffer, per `AudioClip::source_start_frame`/
/// `effective_length_ticks`), stretched across `rect` — one column of pixels covers a proportional
/// slice of that window's samples, not the whole buffer.
pub(crate) fn draw_audio_clip_waveform(
    painter: &egui::Painter,
    rect: egui::Rect,
    buffer: &SampleBuffer,
    start_frame: usize,
    end_frame: usize,
) {
    let len = buffer.mono.len();
    let samples = &buffer.mono[start_frame.min(len)..end_frame.min(len)];
    let width_px = rect.width().round() as usize;
    if samples.is_empty() || width_px == 0 {
        return;
    }
    let mid_y = rect.center().y;
    let half_h = rect.height() / 2.0;
    let stroke = egui::Stroke::new(1.0, egui::Color32::from_black_alpha(180));
    for px in 0..width_px {
        let start = samples.len() * px / width_px;
        let end = (samples.len() * (px + 1) / width_px)
            .max(start + 1)
            .min(samples.len());
        let (min_v, max_v) = samples[start..end]
            .iter()
            .fold((0.0f32, 0.0f32), |(lo, hi), &s| (lo.min(s), hi.max(s)));
        let x = rect.left() + px as f32 + 0.5;
        let y0 = mid_y - max_v.clamp(-1.0, 1.0) * half_h;
        let y1 = (mid_y - min_v.clamp(-1.0, 1.0) * half_h).max(y0 + 0.5);
        painter.line_segment([egui::pos2(x, y0), egui::pos2(x, y1)], stroke);
    }
}

/// Draws a short tick mark at each detected attack (`transient_detection::detect_transients`)
/// within a clip's trimmed window (`start_frame..end_frame`, same window
/// `draw_audio_clip_waveform` draws), scaled across `rect` the same way that waveform is —
/// visual-only, not persisted on `AudioClip` and not (yet) usable to slice the clip; recomputed
/// on every draw the same way the waveform itself is, rather than cached.
fn draw_audio_clip_transient_markers(
    painter: &egui::Painter,
    rect: egui::Rect,
    buffer: &SampleBuffer,
    start_frame: usize,
    end_frame: usize,
) {
    let len = buffer.mono.len();
    let start_frame = start_frame.min(len);
    let end_frame = end_frame.min(len);
    let window_len = end_frame.saturating_sub(start_frame);
    if window_len == 0 {
        return;
    }
    let samples = &buffer.mono[start_frame..end_frame];
    let stroke = egui::Stroke::new(1.0, egui::Color32::from_rgba_unmultiplied(255, 220, 60, 200));
    for marker in transient_detection::detect_transients(samples, buffer.sample_rate) {
        let x = rect.left() + (marker as f32 / window_len as f32) * rect.width();
        painter.line_segment(
            [
                egui::pos2(x, rect.top()),
                egui::pos2(x, rect.top() + rect.height() * 0.25),
            ],
            stroke,
        );
    }
}

/// A list editor for `Song::tempo_map`: each row is an existing tempo-change point's tick
/// (read-only — moving a point means removing and re-inserting it, not dragging it) and its BPM
/// (editable in place), plus a remove button. "+ Insert Tempo Change at Playhead" adds a new
/// point at the transport's current tick, defaulting its BPM to whatever's already in effect
/// there (`Song::bpm_at`) so inserting one is a no-op until the value's actually changed.
/// Simpler than the Piano Roll's draggable automation graph (`automation_lane_graph_ui`) since a
/// tempo map is a handful of precise step-function points, not a continuously-dragged curve.
fn tempo_track_ui(ui: &mut egui::Ui, song: &mut Song, current_tick: Option<usize>) {
    ui.horizontal(|ui| {
        ui.label("Starting tempo:");
        ui.add(
            egui::DragValue::new(&mut song.bpm)
                .range(20.0..=300.0)
                .suffix(" BPM"),
        );
        ui.weak("(same field as the transport LCD's TEMPO)");
    });
    if ui
        .button("+ Insert Tempo Change at Playhead")
        .on_hover_text("Adds a tempo-change point at the transport's current position")
        .clicked()
    {
        let tick = current_tick.unwrap_or(0);
        song.set_tempo_at(tick, song.bpm_at(tick));
    }
    let mut remove_index = None;
    for (index, point) in song.tempo_map.iter_mut().enumerate() {
        ui.horizontal(|ui| {
            ui.label(format!("Tick {}", point.tick));
            ui.add(egui::DragValue::new(&mut point.bpm).range(20.0..=300.0).suffix(" BPM"));
            if ui.small_button("🗑").on_hover_text("Remove this tempo change").clicked() {
                remove_index = Some(index);
            }
        });
    }
    if let Some(index) = remove_index {
        song.remove_tempo_point(index);
    }
    if song.tempo_map.is_empty() {
        ui.weak("No tempo changes yet — the song plays at the starting tempo throughout.");
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn playlist_contents_ui(
    ui: &mut egui::Ui,
    song: &mut Song,
    current_tick: Option<usize>,
    zoom: &mut f32,
    drag: &mut Option<PlaylistDrag>,
    audio_clip_drag: &mut Option<AudioClipDrag>,
    audio_clip_context_menu: &mut Option<AudioClipContextMenuTarget>,
    take_folder_context_menu: &mut Option<TakeFolderContextMenuTarget>,
    take_folder_editor: &mut Option<(usize, usize)>,
    flex_editor: &mut Option<(usize, usize)>,
    editor_targets: &mut PlaylistEditorTargets,
    detached: &mut bool,
) {
    ui.horizontal(|ui| {
        ui.heading("Playlist");
        ui.separator();
        ui.label("Zoom");
        ui.add(
            egui::Slider::new(zoom, PLAYLIST_ZOOM_MIN..=PLAYLIST_ZOOM_MAX)
                .fixed_decimals(2)
                .suffix("x"),
        );
        if ui
            .small_button(if *detached { "⏷ Dock" } else { "⧉ Detach" })
            .clicked()
        {
            *detached = !*detached;
        }
    });
    ui.weak(
        "Click empty space on a track's row to create a region there; drag its right edge to \
         resize (shorter truncates it, longer loops it); drag its body to move it in time. \
         Double-click a region to edit it in the Piano Roll/Beats; right-click removes it.",
    );
    ui.separator();
    ui.collapsing("Tempo Track", |ui| {
        tempo_track_ui(ui, song, current_tick);
    });
    ui.separator();
    let zoom = *zoom;

    // `StepGrid`/`PianoRoll` tracks get a row for their own `regions`; `Audio` tracks get a row
    // below those for their `audio_clips` instead — the two content kinds never share a row.
    let lane_track_indices: Vec<usize> = song
        .tracks
        .iter()
        .enumerate()
        .filter(|(_, t)| t.kind != TrackKind::Audio)
        .map(|(i, _)| i)
        .collect();
    let audio_track_indices: Vec<usize> = song
        .tracks
        .iter()
        .enumerate()
        .filter(|(_, t)| t.kind == TrackKind::Audio)
        .map(|(i, _)| i)
        .collect();

    if lane_track_indices.is_empty() && audio_track_indices.is_empty() {
        ui.centered_and_justified(|ui| {
            ui.weak("Add a track in the Channel Rack to start arranging the song.");
        });
        return;
    }

    let steps_per_bar = song.steps_per_bar();
    let steps_per_beat = song.steps_per_beat();
    let max_region_step = lane_track_indices
        .iter()
        .flat_map(|&i| song.tracks[i].regions.iter())
        .map(|r| (r.start_tick + r.loop_length_steps * TICKS_PER_STEP) / TICKS_PER_STEP)
        .max()
        .unwrap_or(0);
    // Each clip's own starting tempo (`Song::bpm_at`), not one flat rate for all of them — same
    // approximation `audio::arrangement_length_ticks` uses for the same reason.
    let max_audio_step = audio_track_indices
        .iter()
        .flat_map(|&i| song.tracks[i].audio_clips.iter())
        .map(|clip| {
            let ticks_per_second = audio::ticks_per_second(song.bpm_at(clip.start_tick));
            (clip.start_tick + audio_clip_length_ticks(clip, ticks_per_second)) / TICKS_PER_STEP
        })
        .max()
        .unwrap_or(0);
    let display_steps = max_region_step.max(max_audio_step) + steps_per_bar;
    let canvas_width = tick_to_x(display_steps * TICKS_PER_STEP, zoom);
    let audio_rows_top =
        PLAYLIST_RULER_HEIGHT + lane_track_indices.len() as f32 * PLAYLIST_LANE_HEIGHT;
    let canvas_height = audio_rows_top + audio_track_indices.len() as f32 * PLAYLIST_LANE_HEIGHT;
    let total_ticks = (display_steps * TICKS_PER_STEP).max(1);

    // While playing, keep the moving playhead in view: if it's about to run off the right
    // edge of the visible area (or isn't visible at all), jump the horizontal scroll forward
    // so it reappears near the left with room to see what's coming. Only forces a scroll when
    // actually needed, so manual scrolling while paused (or while the playhead is already
    // on-screen) is left alone. Mirrors the piano roll grid's auto-scroll (see
    // `piano_roll_grid_ui`).
    let scroll_offset_id = ui.id().with("playlist-scroll-offset");
    let known_offset_x = ui
        .ctx()
        .data(|d| d.get_temp::<f32>(scroll_offset_id))
        .unwrap_or(0.0);
    let mut playlist_hscroll = egui::ScrollArea::horizontal().id_salt("playlist-scroll");
    if let Some(tick) = current_tick {
        let playhead_x = tick_to_x(tick % total_ticks, zoom);
        let viewport_width = (ui.available_width() - PLAYLIST_HEADER_WIDTH).max(0.0);
        let margin = 60.0;
        if playhead_x < known_offset_x + margin
            || playhead_x > known_offset_x + viewport_width - margin
        {
            playlist_hscroll =
                playlist_hscroll.horizontal_scroll_offset((playhead_x - margin).max(0.0));
        }
    }

    ui.horizontal(|ui| {
        let (header_response, header_painter) = ui.allocate_painter(
            egui::vec2(PLAYLIST_HEADER_WIDTH, canvas_height),
            egui::Sense::hover(),
        );
        let header_rect = header_response.rect;
        header_painter.rect_filled(header_rect, 0u8, ui.visuals().extreme_bg_color);
        for (row, &track_index) in lane_track_indices.iter().enumerate() {
            let y = header_rect.top() + PLAYLIST_RULER_HEIGHT + row as f32 * PLAYLIST_LANE_HEIGHT;
            draw_playlist_row_header(
                &header_painter,
                header_rect,
                y,
                &song.tracks[track_index].name,
                track_color(track_index),
            );
        }
        for (row, &track_index) in audio_track_indices.iter().enumerate() {
            let y = header_rect.top() + audio_rows_top + row as f32 * PLAYLIST_LANE_HEIGHT;
            draw_playlist_row_header(
                &header_painter,
                header_rect,
                y,
                &song.tracks[track_index].name,
                track_color(track_index),
            );
        }

        let scroll_output = playlist_hscroll.show(ui, |ui| {
            let (response, painter) = ui.allocate_painter(
                egui::vec2(canvas_width, canvas_height),
                egui::Sense::click_and_drag(),
            );
            let rect = response.rect;

            let total_rows = lane_track_indices.len() + audio_track_indices.len();
            for row in 0..total_rows {
                let y = rect.top() + PLAYLIST_RULER_HEIGHT + row as f32 * PLAYLIST_LANE_HEIGHT;
                let row_rect = egui::Rect::from_min_size(
                    egui::pos2(rect.left(), y),
                    egui::vec2(canvas_width, PLAYLIST_LANE_HEIGHT),
                );
                let bg = if row % 2 == 0 {
                    ui.visuals().extreme_bg_color
                } else {
                    ui.visuals().faint_bg_color
                };
                painter.rect_filled(row_rect, 0u8, bg);
            }

            let ruler_rect = egui::Rect::from_min_size(
                rect.left_top(),
                egui::vec2(canvas_width, PLAYLIST_RULER_HEIGHT),
            );
            painter.rect_filled(ruler_rect, 0u8, ui.visuals().extreme_bg_color);
            painter.line_segment(
                [ruler_rect.left_bottom(), ruler_rect.right_bottom()],
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
                    ruler_rect.top() + 4.0
                } else {
                    ruler_rect.top() + PLAYLIST_RULER_HEIGHT * 0.6
                };
                painter.line_segment(
                    [egui::pos2(x, tick_top), egui::pos2(x, rect.bottom())],
                    stroke,
                );
                if is_bar {
                    painter.text(
                        egui::pos2(x + 3.0, ruler_rect.top() + 2.0),
                        egui::Align2::LEFT_TOP,
                        format!("{}", step / steps_per_bar + 1),
                        egui::FontId::proportional(10.0),
                        ui.visuals().text_color(),
                    );
                }
            }

            if let Some(tick) = current_tick {
                let x = rect.left() + tick_to_x(tick % total_ticks, zoom);
                painter.line_segment(
                    [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
                    egui::Stroke::new(2.0, FL_ACCENT_GREEN),
                );
            }

            for (row, &track_index) in lane_track_indices.iter().enumerate() {
                let y = rect.top() + PLAYLIST_RULER_HEIGHT + row as f32 * PLAYLIST_LANE_HEIGHT;
                let color = track_color(track_index);
                for region in &song.tracks[track_index].regions {
                    let x = rect.left() + tick_to_x(region.start_tick, zoom);
                    let w = tick_to_x(region.loop_length_steps * TICKS_PER_STEP, zoom).max(3.0);
                    let region_rect = egui::Rect::from_min_size(
                        egui::pos2(x, y + 1.0),
                        egui::vec2(w, PLAYLIST_LANE_HEIGHT - 2.0),
                    );
                    painter.rect_filled(region_rect, 2u8, color);
                    let label_h = 10.0_f32.min(region_rect.height() * 0.5);
                    let preview_rect = egui::Rect::from_min_size(
                        region_rect.min + egui::vec2(0.0, label_h),
                        egui::vec2(
                            region_rect.width(),
                            (region_rect.height() - label_h).max(0.0),
                        ),
                    );
                    draw_region_note_preview(&painter, preview_rect, region);
                    draw_region_fade_overlays(&painter, region_rect, region, zoom);
                    painter.rect_stroke(
                        region_rect,
                        2u8,
                        egui::Stroke::new(1.0, egui::Color32::BLACK),
                        egui::StrokeKind::Inside,
                    );
                    painter.text(
                        region_rect.left_top() + egui::vec2(4.0, 1.0),
                        egui::Align2::LEFT_TOP,
                        &region.name,
                        egui::FontId::proportional(9.0),
                        egui::Color32::BLACK,
                    );
                }
            }

            for (row, &track_index) in audio_track_indices.iter().enumerate() {
                let y = rect.top() + audio_rows_top + row as f32 * PLAYLIST_LANE_HEIGHT;
                let track = &song.tracks[track_index];
                let color = track_color(track_index);
                for clip in &track.audio_clips {
                    let x = rect.left() + tick_to_x(clip.start_tick, zoom);
                    let clip_ticks_per_second = audio::ticks_per_second(song.bpm_at(clip.start_tick));
                    let span_ticks = audio_clip_length_ticks(clip, clip_ticks_per_second);
                    let w = tick_to_x(span_ticks, zoom).max(3.0);
                    let clip_rect = egui::Rect::from_min_size(
                        egui::pos2(x, y + 1.0),
                        egui::vec2(w, PLAYLIST_LANE_HEIGHT - 2.0),
                    );
                    painter.rect_filled(clip_rect, 2u8, color);
                    if let Some(buffer) = &clip.buffer {
                        let frames_per_tick = buffer.sample_rate as f64 / clip_ticks_per_second;
                        let end_frame = clip.source_start_frame.saturating_add(
                            (span_ticks as f64 * frames_per_tick).round() as usize,
                        );
                        draw_audio_clip_waveform(
                            &painter,
                            clip_rect,
                            buffer,
                            clip.source_start_frame,
                            end_frame,
                        );
                        draw_audio_clip_transient_markers(
                            &painter,
                            clip_rect,
                            buffer,
                            clip.source_start_frame,
                            end_frame,
                        );
                    }
                    draw_audio_clip_fade_overlays(&painter, clip_rect, clip, span_ticks, zoom);
                    painter.rect_stroke(
                        clip_rect,
                        2u8,
                        egui::Stroke::new(1.0, egui::Color32::BLACK),
                        egui::StrokeKind::Inside,
                    );
                    let label = Path::new(&clip.file_path)
                        .file_name()
                        .map_or(clip.file_path.as_str(), |n| {
                            n.to_str().unwrap_or(clip.file_path.as_str())
                        });
                    painter.text(
                        clip_rect.left_center() + egui::vec2(4.0, 0.0),
                        egui::Align2::LEFT_CENTER,
                        label,
                        egui::FontId::proportional(10.0),
                        egui::Color32::BLACK,
                    );
                    if clip.load_error.is_some() {
                        painter.text(
                            clip_rect.right_center() - egui::vec2(4.0, 0.0),
                            egui::Align2::RIGHT_CENTER,
                            "⚠",
                            egui::FontId::proportional(11.0),
                            egui::Color32::RED,
                        );
                    }
                }
                for folder in &track.take_folders {
                    let folder_ticks_per_second = audio::ticks_per_second(song.bpm_at(folder.start_tick));
                    let folder_rect = egui::Rect::from_min_size(
                        egui::pos2(rect.left() + tick_to_x(folder.start_tick, zoom), y + 1.0),
                        egui::vec2(
                            tick_to_x(folder.length_ticks, zoom).max(3.0),
                            PLAYLIST_LANE_HEIGHT - 2.0,
                        ),
                    );
                    // Comping-by-take-selection only (see `TakeFolderContextMenuTarget`'s doc
                    // comment), so `comp` is always exactly one segment spanning the whole folder
                    // in this phase — draw whichever take that segment points at.
                    painter.rect_filled(folder_rect, 2u8, TAKE_FOLDER_COLOR);
                    for segment in &folder.comp {
                        let Some(buffer) = folder
                            .takes
                            .get(segment.take_index)
                            .and_then(|t| t.buffer.as_ref())
                        else {
                            continue;
                        };
                        let seg_rect = egui::Rect::from_min_size(
                            egui::pos2(
                                rect.left() + tick_to_x(folder.start_tick + segment.start_tick, zoom),
                                y + 1.0,
                            ),
                            egui::vec2(
                                tick_to_x(segment.end_tick - segment.start_tick, zoom).max(1.0),
                                PLAYLIST_LANE_HEIGHT - 2.0,
                            ),
                        );
                        let frames_per_tick = buffer.sample_rate as f64 / folder_ticks_per_second;
                        let start_frame = (segment.start_tick as f64 * frames_per_tick).round() as usize;
                        let end_frame = (segment.end_tick as f64 * frames_per_tick).round() as usize;
                        draw_audio_clip_waveform(&painter, seg_rect, buffer, start_frame, end_frame);
                    }
                    painter.rect_stroke(
                        folder_rect,
                        2u8,
                        egui::Stroke::new(1.0, egui::Color32::BLACK),
                        egui::StrokeKind::Inside,
                    );
                    let active_take_index = folder.comp.first().map_or(0, |s| s.take_index);
                    let label = format!(
                        "Take {}/{}",
                        active_take_index + 1,
                        folder.takes.len().max(1)
                    );
                    painter.text(
                        folder_rect.left_center() + egui::vec2(4.0, 0.0),
                        egui::Align2::LEFT_CENTER,
                        label,
                        egui::FontId::proportional(10.0),
                        egui::Color32::BLACK,
                    );
                }
            }

            handle_playlist_interaction(
                ui,
                &response,
                rect,
                song,
                &lane_track_indices,
                drag,
                zoom,
                editor_targets,
                steps_per_bar,
            );
            handle_audio_clip_interaction(
                &response,
                rect,
                song,
                &audio_track_indices,
                audio_rows_top,
                audio_clip_drag,
                audio_clip_context_menu,
                flex_editor,
                zoom,
            );
            handle_take_folder_interaction(
                &response,
                rect,
                song,
                &audio_track_indices,
                audio_rows_top,
                take_folder_context_menu,
                take_folder_editor,
                zoom,
            );
        });
        ui.ctx()
            .data_mut(|d| d.insert_temp(scroll_offset_id, scroll_output.state.offset.x));
    });
}

/// Draws one row's name/color header in the Playlist's fixed-left column (see
/// `PLAYLIST_HEADER_WIDTH`): a color swatch plus the row's label, vertically centered on a
/// `PLAYLIST_LANE_HEIGHT`-tall band starting at `y` — kept as its own function since it's called
/// once per track's region row and once per audio-track row, with the same layout either way.
fn draw_playlist_row_header(
    painter: &egui::Painter,
    header_rect: egui::Rect,
    y: f32,
    name: &str,
    color: egui::Color32,
) {
    let swatch_rect = egui::Rect::from_min_size(
        egui::pos2(header_rect.left() + 4.0, y + 4.0),
        egui::vec2(6.0, PLAYLIST_LANE_HEIGHT - 8.0),
    );
    painter.rect_filled(swatch_rect, 1u8, color);
    painter.text(
        egui::pos2(swatch_rect.right() + 5.0, y + PLAYLIST_LANE_HEIGHT / 2.0),
        egui::Align2::LEFT_CENTER,
        name,
        egui::FontId::proportional(11.0),
        egui::Color32::from_rgb(220, 220, 220),
    );
}

/// Hit-tests and applies click/drag gestures against every `StepGrid`/`PianoRoll` track's own
/// `regions`, mirroring `handle_piano_roll_interaction`'s structure (click/drag_started/dragged/
/// drag_stopped) but for regions instead of notes, and with no multi-select/box-select — a region
/// only ever moves, resizes, or gets created/removed one at a time. `lane_track_indices[row]` maps
/// a row to the track it belongs to; double-clicking an existing region routes through
/// `editor_targets` to open it in the Piano Roll or Beats window, whichever matches that track's
/// kind — the only way either window opens or changes region (see `PlaylistEditorTargets`).
#[allow(clippy::too_many_arguments)]
fn handle_playlist_interaction(
    ui: &egui::Ui,
    response: &egui::Response,
    rect: egui::Rect,
    song: &mut Song,
    lane_track_indices: &[usize],
    drag: &mut Option<PlaylistDrag>,
    zoom: f32,
    editor_targets: &mut PlaylistEditorTargets,
    steps_per_bar: usize,
) {
    let local = |p: egui::Pos2| (p.x - rect.left(), p.y - rect.top());
    let x_to_step = |x: f32| -> usize { x_to_tick(x, zoom) / TICKS_PER_STEP };
    let row_count = lane_track_indices.len();
    // Bounded to `row_count`: below that is the audio-track clip rows (see
    // `handle_audio_clip_interaction`), which must not be mistaken for region rows here.
    let y_to_row = |y: f32| -> Option<usize> {
        if y < PLAYLIST_RULER_HEIGHT {
            None
        } else {
            let row = ((y - PLAYLIST_RULER_HEIGHT) / PLAYLIST_LANE_HEIGHT)
                .floor()
                .max(0.0) as usize;
            (row < row_count).then_some(row)
        }
    };
    let region_at = |song: &Song, track_index: usize, step: usize| -> Option<usize> {
        song.tracks[track_index].regions.iter().position(|r| {
            let start_step = r.start_tick / TICKS_PER_STEP;
            step >= start_step && step < start_step + r.loop_length_steps
        })
    };
    let near_right_edge = |song: &Song, track_index: usize, region_index: usize, local_x: f32| {
        let region = &song.tracks[track_index].regions[region_index];
        let end_x = tick_to_x(
            region.start_tick + region.loop_length_steps * TICKS_PER_STEP,
            zoom,
        );
        (local_x - end_x).abs() <= RESIZE_HANDLE_PX
    };
    let region_at_right_edge = |song: &Song, track_index: usize, local_x: f32| -> Option<usize> {
        (0..song.tracks[track_index].regions.len())
            .find(|&i| near_right_edge(song, track_index, i, local_x))
    };
    // Fade handles sit at the point on the region's top edge where its fade ramp ends (fade-in)
    // or begins (fade-out) — see the matching drawing code in `playlist_contents_ui`. Restricted
    // to the top half of the row (`row_frac`) so a fade handle at fade_*_ticks == 0 (sitting right
    // at the region's corner) doesn't shadow the whole-height Move/Resize hit-tests below.
    let near_fade_in_handle = |song: &Song, track_index: usize, region_index: usize, local_x: f32| {
        let region = &song.tracks[track_index].regions[region_index];
        let span_ticks = region.loop_length_steps * TICKS_PER_STEP;
        let fade_ticks = region.fade_in_ticks.min(span_ticks);
        let x = tick_to_x(region.start_tick + fade_ticks, zoom);
        (local_x - x).abs() <= RESIZE_HANDLE_PX
    };
    let near_fade_out_handle = |song: &Song, track_index: usize, region_index: usize, local_x: f32| {
        let region = &song.tracks[track_index].regions[region_index];
        let span_ticks = region.loop_length_steps * TICKS_PER_STEP;
        let fade_ticks = region.fade_out_ticks.min(span_ticks);
        let x = tick_to_x(region.start_tick + span_ticks - fade_ticks, zoom);
        (local_x - x).abs() <= RESIZE_HANDLE_PX
    };
    let region_at_fade_in_handle =
        |song: &Song, track_index: usize, local_x: f32, row_frac: f32| -> Option<usize> {
            if row_frac > 0.5 {
                return None;
            }
            (0..song.tracks[track_index].regions.len())
                .find(|&i| near_fade_in_handle(song, track_index, i, local_x))
        };
    let region_at_fade_out_handle =
        |song: &Song, track_index: usize, local_x: f32, row_frac: f32| -> Option<usize> {
            if row_frac > 0.5 {
                return None;
            }
            (0..song.tracks[track_index].regions.len())
                .find(|&i| near_fade_out_handle(song, track_index, i, local_x))
        };
    let row_frac_at = |ly: f32, row: usize| -> f32 {
        ((ly - PLAYLIST_RULER_HEIGHT) / PLAYLIST_LANE_HEIGHT) - row as f32
    };

    if response.secondary_clicked() {
        if let Some(pos) = response.interact_pointer_pos() {
            let (lx, ly) = local(pos);
            if let Some(row) = y_to_row(ly) {
                let track_index = lane_track_indices[row];
                if let Some(region_index) = region_at(song, track_index, x_to_step(lx)) {
                    song.tracks[track_index].regions.remove(region_index);
                }
            }
        }
    }

    if response.double_clicked() {
        if let Some(pos) = response.interact_pointer_pos() {
            let (lx, ly) = local(pos);
            if let Some(row) = y_to_row(ly) {
                let track_index = lane_track_indices[row];
                let step = x_to_step(lx);
                if let Some(region_index) = region_at(song, track_index, step) {
                    match song.tracks[track_index].kind {
                        TrackKind::PianoRoll => {
                            let region = &song.tracks[track_index].regions[region_index];
                            let start_step = region.start_tick / TICKS_PER_STEP;
                            let content_length_steps = region.content_length_steps.max(1);
                            let local_step = step.saturating_sub(start_step) % content_length_steps;
                            *editor_targets.selected_track = Some(track_index);
                            *editor_targets.piano_roll_region =
                                Some(RegionEditTarget::Region(region_index));
                            *editor_targets.piano_roll_scroll_to =
                                Some(local_step * TICKS_PER_STEP);
                        }
                        TrackKind::StepGrid => {
                            *editor_targets.selected_beats_track = Some(track_index);
                            *editor_targets.beats_region =
                                Some(RegionEditTarget::Region(region_index));
                        }
                        TrackKind::Audio => {}
                    }
                }
            }
        }
    }

    if response.clicked() {
        if let Some(pos) = response.interact_pointer_pos() {
            let (lx, ly) = local(pos);
            if let Some(row) = y_to_row(ly) {
                let track_index = lane_track_indices[row];
                let step = x_to_step(lx);
                if region_at(song, track_index, step).is_none() {
                    song.tracks[track_index].add_region(step, steps_per_bar);
                }
            }
        }
    }

    if drag.is_none() && response.drag_started() {
        if let Some(pos) = response.interact_pointer_pos() {
            let (lx, ly) = local(pos);
            if let Some(row) = y_to_row(ly) {
                let track_index = lane_track_indices[row];
                let step = x_to_step(lx);
                let row_frac = row_frac_at(ly, row);
                if let Some(region_index) =
                    region_at_fade_in_handle(song, track_index, lx, row_frac)
                {
                    *drag = Some(PlaylistDrag {
                        mode: PlaylistDragMode::FadeIn {
                            track_index,
                            region_index,
                        },
                    });
                } else if let Some(region_index) =
                    region_at_fade_out_handle(song, track_index, lx, row_frac)
                {
                    *drag = Some(PlaylistDrag {
                        mode: PlaylistDragMode::FadeOut {
                            track_index,
                            region_index,
                        },
                    });
                } else if let Some(region_index) = region_at_right_edge(song, track_index, lx) {
                    *drag = Some(PlaylistDrag {
                        mode: PlaylistDragMode::Resize {
                            track_index,
                            region_index,
                        },
                    });
                } else if let Some(region_index) = region_at(song, track_index, step) {
                    let grab_step_offset = step as i64
                        - (song.tracks[track_index].regions[region_index].start_tick
                            / TICKS_PER_STEP) as i64;
                    *drag = Some(PlaylistDrag {
                        mode: PlaylistDragMode::Move {
                            track_index,
                            region_index,
                            grab_step_offset,
                        },
                    });
                } else {
                    let region_index = song.tracks[track_index].add_region(step, steps_per_bar);
                    *drag = Some(PlaylistDrag {
                        mode: PlaylistDragMode::Create {
                            track_index,
                            region_index,
                        },
                    });
                }
            }
        }
    }

    if let Some(state) = drag {
        let (track_index, region_index) = match &state.mode {
            PlaylistDragMode::Move {
                track_index,
                region_index,
                ..
            }
            | PlaylistDragMode::Resize {
                track_index,
                region_index,
            }
            | PlaylistDragMode::Create {
                track_index,
                region_index,
            }
            | PlaylistDragMode::FadeIn {
                track_index,
                region_index,
            }
            | PlaylistDragMode::FadeOut {
                track_index,
                region_index,
            } => (*track_index, *region_index),
        };
        let region_count = song.tracks.get(track_index).map_or(0, |t| t.regions.len());
        if region_index >= region_count {
            // The region behind this drag was removed mid-drag (right-click) — drop the dangling state.
            *drag = None;
        } else {
            match &state.mode {
                PlaylistDragMode::Move {
                    grab_step_offset, ..
                } => {
                    let grab_step_offset = *grab_step_offset;
                    if response.dragged() {
                        if let Some(pos) = response.interact_pointer_pos() {
                            let (lx, _ly) = local(pos);
                            let step = x_to_step(lx.max(0.0)) as i64;
                            let new_start_step = (step - grab_step_offset).max(0) as usize;
                            song.tracks[track_index].regions[region_index].start_tick =
                                new_start_step * TICKS_PER_STEP;
                        }
                    }
                    if response.drag_stopped() {
                        *drag = None;
                    }
                }
                PlaylistDragMode::Resize { .. } | PlaylistDragMode::Create { .. } => {
                    if response.dragged() {
                        if let Some(pos) = response.interact_pointer_pos() {
                            let (lx, _ly) = local(pos);
                            let step = x_to_step(lx.max(0.0));
                            let region = &mut song.tracks[track_index].regions[region_index];
                            let start_step = region.start_tick / TICKS_PER_STEP;
                            region.loop_length_steps = step.max(start_step + 1) - start_step;
                        }
                    }
                    if response.drag_stopped() {
                        *drag = None;
                    }
                }
                PlaylistDragMode::FadeIn { .. } => {
                    if response.dragged() {
                        if let Some(pos) = response.interact_pointer_pos() {
                            let (lx, _ly) = local(pos);
                            let tick = x_to_tick(lx.max(0.0), zoom);
                            let region = &mut song.tracks[track_index].regions[region_index];
                            let span_ticks = region.loop_length_steps * TICKS_PER_STEP;
                            let offset = tick.saturating_sub(region.start_tick);
                            region.fade_in_ticks = offset.min(span_ticks);
                        }
                    }
                    if response.drag_stopped() {
                        *drag = None;
                    }
                }
                PlaylistDragMode::FadeOut { .. } => {
                    if response.dragged() {
                        if let Some(pos) = response.interact_pointer_pos() {
                            let (lx, _ly) = local(pos);
                            let tick = x_to_tick(lx.max(0.0), zoom);
                            let region = &mut song.tracks[track_index].regions[region_index];
                            let span_ticks = region.loop_length_steps * TICKS_PER_STEP;
                            let end_tick = region.start_tick + span_ticks;
                            region.fade_out_ticks = end_tick.saturating_sub(tick).min(span_ticks);
                        }
                    }
                    if response.drag_stopped() {
                        *drag = None;
                    }
                }
            }
        }
    }

    if drag.is_none() {
        if let Some(pos) = response.hover_pos() {
            let (lx, ly) = local(pos);
            if let Some(row) = y_to_row(ly) {
                let track_index = lane_track_indices[row];
                if region_at_right_edge(song, track_index, lx).is_some() {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
                }
            }
        }
    }
}
