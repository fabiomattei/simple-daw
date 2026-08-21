mod audio;
mod audio_input;
mod builtin_fx;
mod factory_presets;
mod groove;
mod gui_embed;
#[cfg(unix)]
mod mcp_control;
mod metering;
mod midi_import;
mod model;
mod pitch;
mod plugin_host;
mod sample;
mod session;
mod session_view_ui;
mod stretch;
mod tempo;
mod tempo_detection;
mod transient_detection;
mod wavetable;

use std::collections::HashSet;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use audio::{AudioEngine, Transport};
use builtin_fx::{BuiltInEffect, automatable_params_for_config};
use clack_host::prelude::PluginInstance;
use factory_presets::factory_presets;
use groove::GROOVE_TEMPLATES;
use metering::{MeterHandles, MeterReadings};
use model::{
    AudioClip, AutomationLane, AutomationPoint, AutomationTarget, CurveShape, EffectParamKey, EqBandType,
    FilterMode, FilterRouting, FilterSlope, FilterType, Lane, LfoTarget, MAX_STEP_TIMING_OFFSET_TICKS,
    ModSlot, ModSource, ModTarget, Note, ProjectPlugin, Region, RegionContent, SendBus, Song,
    StepData, SubmixBus, SynthEngine, SynthParams, SynthPreset, SynthWaveform, TICKS_PER_STEP,
    TakeFolder, Track, TrackEffectConfig, TrackKind,
    TrackOutput, TrineParams, WaveModSlot, WaveModSource, WaveModTarget, WaveParams, add_note,
    clear_overlaps, find_note_mut, remove_note,
};
use plugin_host::{
    DawHost, EffectInstance, LoadedEffect, MasterEffectSlots, PluginGuiHandle, PluginParamInfo,
    SendEffectSlots, SubmixEffectSlots, TrackEffectSlots,
};
use raw_window_handle::{HasWindowHandle, RawWindowHandle};
use sample::SampleBuffer;
use wavetable::{WaveWarpMode, WavetableId};

/// Piano-roll pitch range: the full MIDI note range, since a melodic part can
/// use any pitch. The canvas is taller than any screen at this range, so the
/// note grid (not the velocity lane) scrolls vertically — see `piano_roll_ui`.
const PIANO_ROLL_LOW: u8 = 0;
const PIANO_ROLL_HIGH: u8 = 127;
/// Old default range's center (was 28..=48), used to pick a sensible initial
/// vertical scroll position instead of dropping the user at MIDI note 127.
const PIANO_ROLL_DEFAULT_CENTER_PITCH: u8 = 38;

const ROW_HEIGHT: f32 = 15.0;
/// Pixels per 16th-note step; ticks are drawn at a fraction of this.
const PIXELS_PER_STEP: f32 = 40.0;
const PIXELS_PER_TICK: f32 = PIXELS_PER_STEP / TICKS_PER_STEP as f32;
const KEY_LABEL_WIDTH: f32 = 42.0;
const VELOCITY_LANE_HEIGHT: f32 = 46.0;
/// How close (in canvas pixels) a press has to be to a note's right edge to
/// resize it instead of moving it.
const RESIZE_HANDLE_PX: f32 = 6.0;

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

/// Playlist timeline zoom range — same shape as `PIANO_ROLL_ZOOM_MIN`/`MAX`, a separate range
/// since the Playlist is a different view with its own reasonable default scale.
const PLAYLIST_ZOOM_MIN: f32 = 0.25;
const PLAYLIST_ZOOM_MAX: f32 = 3.0;
/// Height of the Playlist canvas's bar/step ruler row, and of each pattern-placement lane below it.
const PLAYLIST_RULER_HEIGHT: f32 = 20.0;
const PLAYLIST_LANE_HEIGHT: f32 = 26.0;
/// Width of the Playlist's fixed (non-scrolling) row-header column — the named/colored labels
/// down the left side, FL Studio–style, that stay put while the timeline canvas scrolls under them.
const PLAYLIST_HEADER_WIDTH: f32 = 120.0;

/// FL Studio–style accent green: playback, active steps/LEDs, the piano-roll playhead. `pub(crate)`
/// so `session_view_ui` can reuse it for a playing Session View slot, the same "active" meaning.
pub(crate) const FL_ACCENT_GREEN: egui::Color32 = egui::Color32::from_rgb(139, 198, 63);
/// FL Studio–style accent orange: warnings, recording, clipping. `pub(crate)` so `session_view_ui`
/// can reuse it for a queued Session View slot.
pub(crate) const FL_ACCENT_ORANGE: egui::Color32 = egui::Color32::from_rgb(242, 169, 59);
/// Playlist block fill for a `TakeFolder` — distinct from a plain `AudioClip`'s `track_color` fill
/// so a recording that can be re-comped is visually distinguishable at a glance from an import.
const TAKE_FOLDER_COLOR: egui::Color32 = egui::Color32::from_rgb(196, 152, 219);
/// Accent yellow: an active track solo, distinct from mute's orange.
const FL_ACCENT_YELLOW: egui::Color32 = egui::Color32::from_rgb(235, 210, 64);

/// A dark theme approximating FL Studio's default color scheme, applied once at startup
/// (see `main`). Nearly every custom-painted widget in this file (piano-roll grid, ADSR/
/// filter/LFO previews, oscillator scopes) already reads colors from `ui.visuals()` rather
/// than hardcoding them, so this one override cascades through the whole app.
fn fl_studio_visuals() -> egui::Visuals {
    let mut visuals = egui::Visuals::dark();

    visuals.override_text_color = Some(egui::Color32::from_rgb(228, 228, 228));
    visuals.weak_text_color = Some(egui::Color32::from_rgb(140, 140, 140));
    visuals.hyperlink_color = FL_ACCENT_GREEN;
    visuals.faint_bg_color = egui::Color32::from_rgb(46, 46, 46);
    visuals.extreme_bg_color = egui::Color32::from_rgb(16, 16, 16);
    visuals.code_bg_color = egui::Color32::from_rgb(40, 40, 40);
    visuals.warn_fg_color = FL_ACCENT_ORANGE;

    visuals.window_fill = egui::Color32::from_rgb(36, 36, 36);
    visuals.window_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(16, 16, 16));
    visuals.panel_fill = egui::Color32::from_rgb(36, 36, 36);

    visuals.selection.bg_fill = FL_ACCENT_GREEN.gamma_multiply(0.55);
    visuals.selection.stroke = egui::Stroke::new(1.0, FL_ACCENT_GREEN);

    visuals.widgets.noninteractive.bg_fill = egui::Color32::from_rgb(36, 36, 36);
    visuals.widgets.noninteractive.weak_bg_fill = egui::Color32::from_rgb(36, 36, 36);
    visuals.widgets.noninteractive.bg_stroke =
        egui::Stroke::new(1.0, egui::Color32::from_rgb(16, 16, 16));
    visuals.widgets.noninteractive.fg_stroke =
        egui::Stroke::new(1.0, egui::Color32::from_rgb(180, 180, 180));

    visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(58, 58, 58);
    visuals.widgets.inactive.weak_bg_fill = egui::Color32::from_rgb(58, 58, 58);
    visuals.widgets.inactive.fg_stroke =
        egui::Stroke::new(1.0, egui::Color32::from_rgb(210, 210, 210));

    visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(72, 72, 72);
    visuals.widgets.hovered.weak_bg_fill = egui::Color32::from_rgb(72, 72, 72);
    visuals.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, FL_ACCENT_GREEN);
    visuals.widgets.hovered.fg_stroke = egui::Stroke::new(1.5, egui::Color32::WHITE);

    visuals.widgets.active.bg_fill = FL_ACCENT_GREEN;
    visuals.widgets.active.weak_bg_fill = FL_ACCENT_GREEN;
    visuals.widgets.active.bg_stroke = egui::Stroke::new(1.0, FL_ACCENT_GREEN);
    visuals.widgets.active.fg_stroke = egui::Stroke::new(1.5, egui::Color32::BLACK);

    visuals.widgets.open.bg_fill = egui::Color32::from_rgb(44, 44, 44);
    visuals.widgets.open.weak_bg_fill = egui::Color32::from_rgb(44, 44, 44);
    visuals.widgets.open.bg_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(16, 16, 16));

    visuals
}

/// A fixed palette of hues cycled by track index, used to color each track's Channel Rack
/// swatch, its step-grid's active steps, and its piano-roll notes — so a track stays visually
/// identifiable across the rack and the note editor without adding a persisted "color" field
/// to `Track`.
fn track_color(index: usize) -> egui::Color32 {
    const PALETTE: [egui::Color32; 10] = [
        egui::Color32::from_rgb(139, 198, 63),  // green
        egui::Color32::from_rgb(90, 170, 250),  // blue
        egui::Color32::from_rgb(242, 169, 59),  // orange
        egui::Color32::from_rgb(220, 100, 130), // rose
        egui::Color32::from_rgb(180, 130, 230), // purple
        egui::Color32::from_rgb(90, 220, 200),  // teal
        egui::Color32::from_rgb(230, 210, 80),  // yellow
        egui::Color32::from_rgb(230, 120, 80),  // red-orange
        egui::Color32::from_rgb(120, 200, 130), // mint
        egui::Color32::from_rgb(150, 160, 220), // periwinkle
    ];
    PALETTE[index % PALETTE.len()]
}

fn tick_to_x(tick: usize, zoom: f32) -> f32 {
    tick as f32 * PIXELS_PER_TICK * zoom
}

fn x_to_tick(x: f32, zoom: f32) -> usize {
    (x / (PIXELS_PER_TICK * zoom)).round().max(0.0) as usize
}

/// An `AudioClip`'s on-timeline length in ticks, for drawing/hit-testing its block in the
/// Playlist — `AudioClip::effective_length_ticks` at the song's current tempo (`ticks_per_second`,
/// from `audio::ticks_per_second`), same as `audio::arrangement_length_ticks` does for looping.
fn audio_clip_length_ticks(clip: &AudioClip, ticks_per_second: f64) -> usize {
    if clip.buffer.is_some() {
        clip.effective_length_ticks(ticks_per_second).max(1)
    } else {
        // Still loading (or failed to load) — a minimal placeholder width keeps a broken clip
        // visible/selectable to move or delete, rather than invisible.
        TICKS_PER_STEP
    }
}

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
enum PianoRollScale {
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

struct PianoRollDrag {
    mode: PianoRollDragMode,
}

/// At most one automation lane point is being dragged at a time, shared by the Piano Roll's and
/// Beats' automation panels (see `automation_lanes_ui`) the same way `PlaylistDrag` is shared by
/// the region-move/resize/fade gestures. `lane_index`/`point_index` re-check bounds every frame,
/// in case the lane or point was removed (right-click) since the drag began.
struct AutomationDrag {
    lane_index: usize,
    point_index: usize,
}

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

struct PlaylistDrag {
    mode: PlaylistDragMode,
}

/// The Piano Roll's/Beats' "which region is open" state, bundled so `handle_playlist_interaction`
/// can set either pair on a double-click without a long individual-borrow parameter list. Setting
/// `selected_track`/`piano_roll_region` (or the Beats equivalent) is the *only* way either editor
/// window opens or changes which region it shows — there's no in-window picker, and the Channel
/// Rack has no "open editor" button; see `playlist_contents_ui`'s doc comment.
struct PlaylistEditorTargets<'a> {
    selected_track: &'a mut Option<usize>,
    piano_roll_region: &'a mut Option<usize>,
    /// See `SimpleDawApp::piano_roll_scroll_to`. Set alongside `piano_roll_region` on a
    /// double-click, to the content-local tick under the click.
    piano_roll_scroll_to: &'a mut Option<usize>,
    selected_beats_track: &'a mut Option<usize>,
    beats_region: &'a mut Option<usize>,
}

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

struct AudioClipDrag {
    mode: AudioClipDragMode,
}

/// Which audio clip's right-click context menu (if any) is currently open, and where — set by
/// `handle_audio_clip_interaction` on `secondary_clicked()`, read back by `playlist_contents_ui`'s
/// `response.context_menu` closure to know which clip "Strip Silence"/"Delete" apply to. Kept
/// alongside `audio_clip_drag` in `SimpleDawApp` rather than folded into it, since a context menu
/// being open has nothing to do with a drag being in progress.
#[derive(Clone, Copy)]
struct AudioClipContextMenuTarget {
    track_index: usize,
    clip_index: usize,
}

/// Which take folder's right-click context menu (if any) is currently open, and where — the
/// `TakeFolder` counterpart of `AudioClipContextMenuTarget`. This phase's take-folder editing is
/// context-menu-only (pick a take, or delete the whole folder) — no move/trim drag yet, unlike
/// plain audio clips (see `handle_take_folder_interaction`).
#[derive(Clone, Copy)]
struct TakeFolderContextMenuTarget {
    track_index: usize,
    folder_index: usize,
}

/// An in-progress "paint `take_index` over this stretch" drag inside the take-folder comp editor
/// (`take_folder_editor_window_ui`) — `start_tick` is fixed at drag start; the current drag end is
/// read fresh from the pointer position each frame and applied live via
/// `TakeFolder::assign_take_to_range`, the same "mutate live on `dragged()`" pattern
/// `handle_audio_clip_interaction`'s trim/fade drags already use.
struct TakeFolderCompDrag {
    take_index: usize,
    start_tick: usize,
}

/// Which tab of the Flex editor window (`flex_editor_window_ui`) is showing.
#[derive(Clone, Copy, PartialEq)]
enum FlexEditorMode {
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
struct FlexMarkerDrag {
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
struct FlexNoteDrag {
    start_frame: usize,
    end_frame: usize,
    start_semitones: f32,
    drag_start_y: f32,
}

/// Which effect's parameter-editor window (if any) is currently open. There's only ever one such
/// window at a time, shared by the master bus, every track, and every send bus. `Master(slot_index)`/
/// `Track(track_index, slot_index)`/`Send(send_index, slot_index)` identify one slot within the
/// master chain/that track's chain/that send bus's chain respectively.
#[derive(Clone, Copy, PartialEq, Eq)]
enum EffectEditorTarget {
    Master(usize),
    Track(usize, usize),
    Send(usize, usize),
    Submix(usize, usize),
}

/// Which chain `TrackFxUi` is editing — only changes which `EffectEditorTarget` variant its
/// "Params" button opens (see `fx_chain_ui`), so each location's editor state doesn't collide with
/// the others'. The chain's own index within `slots` (a real track's index, or a send bus's index;
/// meaningless — always 0 — for `Master`) still comes from `TrackFxUi::track_index`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum FxChainKind {
    Track,
    Master,
    Send,
    Submix,
}

/// Which track's or step-grid lane's instrument + effect chain the always-visible bottom Device
/// Panel (see `device_panel_contents_ui`) is currently showing — the Bitwig/Ableton-style docked
/// device rack, replacing the old per-track/per-lane synth-settings windows this app used to open
/// on a "🎹" click. `Lane` is one level deeper than `Track` since a lane's synth belongs to a
/// specific region's lane rather than the whole track (see `Lane::synth_override`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum DeviceChainFocus {
    Track(usize),
    Lane { track_index: usize, region_index: usize, lane_index: usize },
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

/// Set by a track row's Freeze/Unfreeze/Bounce button; applied by the caller once `song.tracks`'
/// mutable-iterator borrow from the row loop has ended — same "signal during the loop, apply
/// after" pattern `track_to_remove` already uses, since `freeze_track`/`bounce_track_in_place`
/// need `&mut Song` as a whole, not just the one `&mut Track` a row has borrowed.
enum TrackFreezeAction {
    Freeze,
    Unfreeze,
    Bounce,
}

struct SimpleDawApp {
    engine: anyhow::Result<AudioEngine>,
    song: Arc<Mutex<Song>>,
    transport: Transport,
    /// Tap-tempo state for the transport LCD's Tap button — transient UI state, not song data
    /// (unlike `Song::bpm`, which a tap's resulting estimate is written into).
    tap_tempo: tempo::TapTempo,
    /// `None` if the audio engine failed to start; lanes can't be pre-resampled without it.
    sample_rate: Option<u32>,
    song_path: String,
    /// `song_path` as of the last time the window title was set — lets the title update track
    /// `song_path` (on load/save) without re-issuing the viewport title command every frame.
    titled_song_path: String,
    /// (was the last save/load successful, message to show)
    song_message: Option<(bool, String)>,
    show_save_as: bool,
    save_as_path: String,
    show_export_dialog: bool,
    export_path: String,
    export_loops: u32,
    /// (was this export successful, message to show)
    export_message: Option<(bool, String)>,
    show_import_midi: bool,
    import_midi_path: String,
    /// Whether to also overwrite `Song::bpm` with the tempo detected in the imported file (if
    /// any). Defaults off since "Import" (unlike "Load") adds to the current song rather than
    /// replacing it, and silently retempoing the whole song out from under existing tracks would
    /// be surprising.
    import_midi_apply_bpm: bool,
    /// (was the last import successful, message to show)
    import_midi_message: Option<(bool, String)>,
    /// Whether the "Detect Tempo" dialog (analyze a WAV file's audio content — not a MIDI file's
    /// header, see `import_midi_apply_bpm` — and estimate its BPM via `tempo_detection`) is open.
    show_detect_tempo: bool,
    detect_tempo_path: String,
    /// The last detection's estimate, kept separate from `detect_tempo_message` (a string) so the
    /// dialog's "Apply to Song Tempo" button doesn't need to re-parse it.
    detect_tempo_bpm: Option<f32>,
    /// (was the last detection successful, message to show)
    detect_tempo_message: Option<(bool, String)>,
    /// Whether the "Plugins" window (project-level CLAP plugin library — import, load onto
    /// master, remove) is open.
    show_plugins_panel: bool,
    /// The master bus's own effect-chain bookkeeping — same shape as the per-track
    /// `track_effect_*` fields below, just flat (one implicit chain) rather than one `Vec` per
    /// track, since there's only ever one master bus. `master_effect_slots` is the live chain
    /// shared with the audio thread (see `plugin_host::MasterEffectSlots`), always exactly one
    /// "track" long at the outer level.
    master_effect_paths: Vec<String>,
    master_effect_slots: MasterEffectSlots,
    /// Kept alive for the app's lifetime once loaded — see phase 7 plan for
    /// why there's no unload/deactivate support yet.
    master_effect_instances: Vec<Option<PluginInstance<DawHost>>>,
    /// The master effect's floating GUI window state, if a plugin implementing the `gui`
    /// extension is loaded — see `plugin_host::PluginGuiHandle`.
    master_effect_guis: Vec<Option<PluginGuiHandle>>,
    master_effect_messages: Vec<Option<(bool, String)>>,
    track_effect_slots: TrackEffectSlots,
    /// Kept alive for the app's lifetime once loaded, same as `master_effect_instances`. Outer
    /// index is the track (same as `song.tracks`), inner index is the slot within that track's
    /// effect chain (same as `Track::effects`/`track_effect_slots`'s inner `Vec`); kept in sync
    /// via `resize_track_effects`/`remove_track_effects`.
    track_effect_instances: Vec<Vec<Option<PluginInstance<DawHost>>>>,
    /// Per-slot floating GUI window state, indexed the same as `track_effect_instances`.
    track_effect_guis: Vec<Vec<Option<PluginGuiHandle>>>,
    track_effect_paths: Vec<Vec<String>>,
    track_effect_messages: Vec<Vec<Option<(bool, String)>>>,
    /// Every send bus's own effect-chain bookkeeping — same shape as the per-track `track_effect_*`
    /// fields above (one entry per `Song::sends` row), kept in sync via `resize_track_effects`/
    /// `remove_track_effects`, the same helpers the per-track fields use. `send_effect_slots` is
    /// the live chain shared with the audio thread (see `plugin_host::SendEffectSlots`).
    send_effect_slots: SendEffectSlots,
    send_effect_instances: Vec<Vec<Option<PluginInstance<DawHost>>>>,
    send_effect_guis: Vec<Vec<Option<PluginGuiHandle>>>,
    send_effect_paths: Vec<Vec<String>>,
    send_effect_messages: Vec<Vec<Option<(bool, String)>>>,
    /// Every submix bus's own effect-chain bookkeeping — same shape/sync mechanism as
    /// `send_effect_*` above, just one entry per `Song::submixes` row. `submix_effect_slots` is
    /// the live chain shared with the audio thread (see `plugin_host::SubmixEffectSlots`).
    submix_effect_slots: SubmixEffectSlots,
    submix_effect_instances: Vec<Vec<Option<PluginInstance<DawHost>>>>,
    submix_effect_guis: Vec<Vec<Option<PluginGuiHandle>>>,
    submix_effect_paths: Vec<Vec<String>>,
    submix_effect_messages: Vec<Vec<Option<(bool, String)>>>,
    /// Live peak/RMS/LUFS readings published by the audio thread — one entry per track, kept in
    /// sync with `song.tracks` via `resize_track_meters`/`remove_track_meter` the same way
    /// `track_effect_slots` is kept in sync via `resize_track_effects`/`remove_track_effects`. The
    /// master bus's own meter (`master_meter`) is the same type pinned to a single row, mirroring
    /// `master_effect_slots`/`MasterEffectSlots`. `submix_meters` is one entry per `Song::submixes`
    /// row, kept in sync the same way `track_meters` is.
    track_meters: MeterHandles,
    master_meter: MeterHandles,
    submix_meters: MeterHandles,
    /// Live Session View clip-slot playback state (queued/playing/stopped), published by the audio
    /// thread once per callback — see `audio::SessionSlotHandles`'s doc comment. Self-managing
    /// (the audio thread resizes it to match `Song::tracks`/`Track::session_clips` itself), unlike
    /// `track_meters`, so there's no `resize_session_slots` counterpart to call on track add/remove.
    session_slots: audio::SessionSlotHandles,
    /// Session View's launch-quantize setting (e.g. "1 Bar") — live UI state, not song data, read
    /// each frame to compute `Transport::session_quantize_ticks` from the current song's own
    /// `Song::steps_per_bar`. See `session_view_ui::SessionQuantize`.
    session_quantize: session_view_ui::SessionQuantize,
    /// Which Session View slot's Follow Action editor window is open, if any: `(track_index,
    /// slot_index)` — same `Option<(usize, usize)>`-keyed-window idiom as `take_folder_editor`.
    follow_action_editor: Option<(usize, usize)>,
    /// Which effect's parameter-editor window (if any) is currently open.
    effect_editor: Option<EffectEditorTarget>,
    /// Which slot's embedded plugin GUI (if any) currently owns the one reserved panel inside the
    /// "FX Params" window (see `plugin_gui_button_ui`) — at most one embedded GUI is ever shown at
    /// once, since there's only ever one such window. Opening a different target's embedded GUI
    /// closes whichever one this points at first; floating GUIs aren't tracked here since they're
    /// independent OS windows that can already coexist.
    active_embedded_gui: Option<EffectEditorTarget>,
    /// Which track's or lane's instrument + effect chain the bottom Device Panel is showing, if
    /// any — see `DeviceChainFocus`. Unlike `effect_editor`, a `Track` focus operates straight on
    /// `Song::tracks[..].synth`/`.trine`/`.wave` (model data, no live plugin instance to juggle),
    /// so the panel body just borrows `song` directly for that part.
    device_chain_focus: Option<DeviceChainFocus>,
    /// Text field backing the "Save as preset" button in the synth editor window — shared across
    /// tracks/sessions like `save_as_path` is for song files, since only one synth editor window
    /// (and thus one preset-name input) can be open at a time.
    new_preset_name: String,
    /// Result of the last preset import/export attempt, shown in the synth editor window —
    /// same (ok, message) shape as `import_midi_message`.
    preset_message: Option<(bool, String)>,
    /// At most one piano-roll note is being dragged at a time, across every
    /// track's piano roll (there's only one mouse).
    piano_roll_drag: Option<PianoRollDrag>,
    /// See `AutomationDrag` — shared by the Piano Roll's and Beats' region-scoped automation
    /// panels the same way `piano_roll_drag` is shared across every track's piano roll.
    automation_drag: Option<AutomationDrag>,
    /// Same as `automation_drag`, for the separate track-wide automation panel (`Track::automation`,
    /// not tied to a region) each window also shows whenever a track is selected. A distinct field
    /// so dragging a point in one panel can't be confused with the other's drag state.
    track_automation_drag: Option<AutomationDrag>,
    /// Currently selected piano-roll note ids, shared across every track's piano roll (like
    /// `piano_roll_drag`, there's only one selection active at a time). Note ids are unique
    /// across the whole song, so a selection only ever matches notes in the one track it was
    /// made in — other tracks' `Vec<Note>` simply won't contain those ids.
    selected_notes: HashSet<u64>,
    /// Grid size for the Piano Roll's Quantize/Groove Template toolbar, in ticks — one of a fixed
    /// set of note-length choices (see `QUANTIZE_GRID_CHOICES`), not freely adjustable.
    groove_quantize_grid_ticks: usize,
    /// How fully Quantize snaps notes to `groove_quantize_grid_ticks` (0.0 = no change, 1.0 =
    /// fully snapped) — see `groove::quantize_notes`.
    groove_quantize_strength: f32,
    /// Max random timing nudge (in ticks) Humanize applies — see `groove::humanize_notes`.
    groove_humanize_timing_ticks: usize,
    /// Max random velocity nudge Humanize applies — see `groove::humanize_notes`.
    groove_humanize_velocity: u8,
    /// Which `groove::GROOVE_TEMPLATES` entry the Groove Template toolbar button applies.
    groove_template_index: usize,
    /// Shared zoom level for every track's piano roll (1.0 = normal size), so
    /// switching tracks doesn't reset your zoom. Scales both axes together —
    /// see `tick_to_x`/`row_height`. Lives in the top toolbar (not per-roll)
    /// since it applies to every track's piano roll at once.
    piano_roll_zoom: f32,
    /// Root note (pitch class 0=C..11=B) for the piano roll's scale-highlight background — see
    /// `piano_roll_scale`. Shared across every track's piano roll, like `piano_roll_zoom`.
    piano_roll_scale_root: u8,
    /// Which modal/pentatonic scale (if any) to highlight in the piano roll's row background,
    /// selected from the left panel next to the note-length fractions. `PianoRollScale::Off`
    /// disables highlighting entirely, restoring the plain black/white key row coloring.
    piano_roll_scale: PianoRollScale,
    /// Index of the track whose Piano Roll window is open — set only by double-clicking one of
    /// its regions in the Playlist (see `PlaylistEditorTargets`), cleared when that window is
    /// closed. `None` means no Piano Roll window is open. Only meaningful when it points at a
    /// piano-roll track.
    selected_track: Option<usize>,
    /// Which of `selected_track`'s own `regions` the Piano Roll is showing/editing — the region
    /// counterpart of `selected_track` picking the track. `None` (or pointing past the end after
    /// the region was deleted) shows a "double-click a region in the Playlist" placeholder instead.
    piano_roll_region: Option<usize>,
    /// A content-local tick the Piano Roll should scroll to on its next render, set alongside
    /// `piano_roll_region`/`selected_track` by a Playlist double-click so the grid opens on the
    /// section that was actually clicked rather than always at the start. Consumed (cleared) by
    /// `piano_roll_ui` once applied, so it doesn't fight manual scrolling afterward.
    piano_roll_scroll_to: Option<usize>,
    /// Index of the track whose Beats window is open — same lifecycle as `selected_track`, but
    /// for step-grid tracks.
    selected_beats_track: Option<usize>,
    /// Which of `selected_beats_track`'s own `regions` the Beats window is showing/editing — the
    /// Beats counterpart of `piano_roll_region`.
    beats_region: Option<usize>,
    /// Whether the Channel Rack is popped out into its own native OS window (via
    /// `egui::Context::show_viewport_immediate`) instead of docked as the left `egui::Panel`.
    channel_rack_detached: bool,
    /// Whether the Playlist (arrangement timeline) window is open — toggled from the toolbar,
    /// always detached like the Piano Roll/Beats windows (no docked mode).
    playlist_open: bool,
    /// Whether Session View (the clip-launching grid) is showing in the central area instead of
    /// the Playlist — toggled from the toolbar, mutually exclusive with `playlist_open` (both want
    /// the same central area; unlike Playlist vs. Mixer, which dock to different regions and can
    /// coexist). See `session_view_ui::session_view_contents_ui`.
    session_view_open: bool,
    /// Whether the Mixer (classic vertical channel-strip view — one strip per track plus a Master
    /// strip) is visible at all, toggled from the toolbar. Same dock/detach split as the Channel
    /// Rack (see `mixer_detached`), but unlike the Channel Rack it can be hidden entirely, since
    /// the same volume/mute/solo/FX controls already live inline on each Channel Rack row.
    mixer_open: bool,
    /// Whether the (visible) Mixer is popped out into its own native OS window instead of docked
    /// as a bottom `egui::Panel` — see `channel_rack_detached`.
    mixer_detached: bool,
    /// Zoom for the Playlist timeline, independent of `piano_roll_zoom` since it's a separate view.
    playlist_zoom: f32,
    /// At most one Playlist clip is being dragged at a time — see `piano_roll_drag`.
    playlist_drag: Option<PlaylistDrag>,
    /// At most one audio-track clip is being dragged at a time — see `playlist_drag`.
    audio_clip_drag: Option<AudioClipDrag>,
    /// Which audio clip's right-click context menu is currently open, if any — see
    /// `AudioClipContextMenuTarget`.
    audio_clip_context_menu: Option<AudioClipContextMenuTarget>,
    /// Which take folder's right-click context menu (pick a take/delete) is currently open, if
    /// any — see `TakeFolderContextMenuTarget`.
    take_folder_context_menu: Option<TakeFolderContextMenuTarget>,
    /// Which take folder's segment-level comp-editor window is open, if any: `(track_index,
    /// folder_index)`. Set by double-clicking a take folder in the Playlist (mirrors
    /// `piano_roll_region`'s "double-click opens the editor" convention) — see
    /// `take_folder_editor_window_ui`.
    take_folder_editor: Option<(usize, usize)>,
    /// In-progress "paint this take over this stretch" drag inside the take-folder comp editor —
    /// see `TakeFolderCompDrag`.
    take_folder_comp_drag: Option<TakeFolderCompDrag>,
    /// Which `AudioClip`'s Flex Time/Pitch editor window is open, if any: `(track_index,
    /// clip_index)` — set from that clip's right-click context menu (see
    /// `handle_audio_clip_interaction`). Used by `flex_editor_window_ui`.
    flex_editor: Option<(usize, usize)>,
    /// Time vs. Pitch tab within the open Flex editor window.
    flex_editor_mode: FlexEditorMode,
    /// The Flex editor's own independently-decoded *unwarped, unshifted* buffer for whichever
    /// clip `flex_editor` names, keyed by that same `(track_index, clip_index)` so it's reloaded
    /// when the target changes — the editor places/drags markers and note segments against the
    /// original recording, never against `AudioClip::buffer` (which is already the edited result
    /// once `warp_markers`/`pitch_corrections` are non-empty, see `AudioClip::load`).
    flex_editor_raw: Option<((usize, usize), Arc<SampleBuffer>)>,
    /// In-progress "drag this warp marker's output position" drag in the Flex editor's Time tab —
    /// see `FlexMarkerDrag`.
    flex_marker_drag: Option<FlexMarkerDrag>,
    /// In-progress "drag this detected note's target pitch" drag in the Flex editor's Pitch tab —
    /// see `FlexNoteDrag`.
    flex_note_drag: Option<FlexNoteDrag>,
    /// Index of the `Audio`-kind track armed for recording, if any — set from the Channel Rack's
    /// record-arm toggle, cleared if that track is deleted. Session/UI state, not song data (see
    /// `RecordingSession`).
    record_armed_track: Option<usize>,
    /// Name of the input device to record from, as returned by `audio_input::list_input_devices`.
    /// `None` means "use the host's default input device" (see `audio_input::InputRecorder::start`).
    selected_input_device: Option<String>,
    /// The in-progress recording, if the transport's Record button is currently engaged.
    recording: Option<RecordingSession>,
    /// (was the last recording successfully turned into a clip, message to show)
    recording_message: Option<(bool, String)>,
    /// Name of the output device to play through, as returned by `audio::list_output_devices`.
    /// `None` means "use the host's default output device" (see `AudioEngine::start`).
    selected_output_device: Option<String>,
    /// Sample rate to run the output stream at. `None` means "use the selected device's own
    /// default rate" (see `AudioEngine::start`).
    selected_output_sample_rate: Option<u32>,
    /// (did the last device/rate switch succeed, message to show)
    output_device_message: Option<(bool, String)>,
    /// Queued requests from the `simple-daw-mcp` companion binary (see `mcp_control`), drained
    /// and applied at the top of `ui()` each frame. `None` on non-Unix builds, where MCP control
    /// isn't compiled in.
    #[cfg(unix)]
    mcp_rx: std::sync::mpsc::Receiver<mcp_control::McpRequest>,
}

/// State for an in-progress recording started from the toolbar's Record button — see
/// `SimpleDawApp::recording`. Torn down (WAV written, `AudioClip` pushed onto the armed track) by
/// the same button on the next click.
struct RecordingSession {
    track_index: usize,
    recorder: audio_input::InputRecorder,
    start_tick: usize,
}

impl SimpleDawApp {
    fn new() -> Self {
        let song = Arc::new(Mutex::new(Song::demo()));
        let transport = Transport::new();
        let master_effect_slots = plugin_host::new_master_effect_slots();
        let track_count = song.lock().unwrap().tracks.len();
        let track_effect_slots = plugin_host::new_track_effect_slots(track_count);
        let send_count = song.lock().unwrap().sends.len();
        let send_effect_slots = plugin_host::new_track_effect_slots(send_count);
        let submix_count = song.lock().unwrap().submixes.len();
        let submix_effect_slots = plugin_host::new_track_effect_slots(submix_count);
        let track_meters = metering::new_track_meter_handles(track_count);
        let master_meter = metering::new_master_meter_handles();
        let submix_meters = metering::new_track_meter_handles(submix_count);
        let session_slots = audio::new_session_slot_handles();
        let engine = AudioEngine::start(
            song.clone(),
            transport.clone(),
            master_effect_slots.clone(),
            track_effect_slots.clone(),
            send_effect_slots.clone(),
            submix_effect_slots.clone(),
            track_meters.clone(),
            master_meter.clone(),
            submix_meters.clone(),
            session_slots.clone(),
            None,
            None,
        );
        let sample_rate = engine.as_ref().ok().map(|e| e.status.sample_rate);
        let engine_config = engine
            .as_ref()
            .ok()
            .map(|e| (e.status.sample_rate as f64, e.status.min_frames, e.status.max_frames));

        if let Some(sample_rate) = sample_rate {
            preload_demo_samples(&song, sample_rate);
        }

        // `Song::demo()` seeds a default master-bus limiter (see its own doc comment) — build and
        // load it into the live chain now, the same way loading a song file does, so a fresh
        // session actually hears it rather than just recording it in `Song::master_effects`.
        let master_specs = song.lock().unwrap().master_effects.clone();
        let (
            master_effect_paths,
            master_effect_instances,
            master_effect_guis,
            master_effect_messages,
            master_chain,
        ) = build_effect_chain(master_specs, engine_config);
        if let Ok(mut slots) = master_effect_slots.lock()
            && let Some(slot) = slots.get_mut(0)
        {
            *slot = master_chain;
        }

        Self {
            engine,
            song,
            transport,
            tap_tempo: tempo::TapTempo::default(),
            sample_rate,
            song_path: "song.json".to_string(),
            titled_song_path: String::new(),
            song_message: None,
            show_save_as: false,
            save_as_path: String::new(),
            show_export_dialog: false,
            export_path: "export.wav".to_string(),
            export_loops: 4,
            export_message: None,
            show_import_midi: false,
            import_midi_path: String::new(),
            import_midi_apply_bpm: false,
            import_midi_message: None,
            show_detect_tempo: false,
            detect_tempo_path: String::new(),
            detect_tempo_bpm: None,
            detect_tempo_message: None,
            show_plugins_panel: false,
            master_effect_paths,
            master_effect_slots,
            master_effect_instances,
            master_effect_guis,
            master_effect_messages,
            track_effect_slots,
            track_effect_instances: (0..track_count).map(|_| Vec::new()).collect(),
            track_effect_guis: (0..track_count).map(|_| Vec::new()).collect(),
            track_effect_paths: (0..track_count).map(|_| Vec::new()).collect(),
            track_effect_messages: (0..track_count).map(|_| Vec::new()).collect(),
            send_effect_slots,
            send_effect_instances: (0..send_count).map(|_| Vec::new()).collect(),
            send_effect_guis: (0..send_count).map(|_| Vec::new()).collect(),
            send_effect_paths: (0..send_count).map(|_| Vec::new()).collect(),
            send_effect_messages: (0..send_count).map(|_| Vec::new()).collect(),
            submix_effect_slots,
            submix_effect_instances: (0..submix_count).map(|_| Vec::new()).collect(),
            submix_effect_guis: (0..submix_count).map(|_| Vec::new()).collect(),
            submix_effect_paths: (0..submix_count).map(|_| Vec::new()).collect(),
            submix_effect_messages: (0..submix_count).map(|_| Vec::new()).collect(),
            track_meters,
            master_meter,
            submix_meters,
            session_slots,
            session_quantize: session_view_ui::SessionQuantize::default(),
            follow_action_editor: None,
            effect_editor: None,
            active_embedded_gui: None,
            device_chain_focus: None,
            new_preset_name: String::new(),
            preset_message: None,
            piano_roll_drag: None,
            automation_drag: None,
            track_automation_drag: None,
            selected_notes: HashSet::new(),
            groove_quantize_grid_ticks: TICKS_PER_STEP,
            groove_quantize_strength: 1.0,
            groove_humanize_timing_ticks: 6,
            groove_humanize_velocity: 12,
            groove_template_index: 0,
            piano_roll_zoom: 1.0,
            piano_roll_scale_root: 0,
            piano_roll_scale: PianoRollScale::Off,
            selected_track: None,
            piano_roll_region: None,
            piano_roll_scroll_to: None,
            selected_beats_track: None,
            beats_region: None,
            channel_rack_detached: false,
            playlist_open: true,
            session_view_open: false,
            mixer_open: false,
            mixer_detached: false,
            playlist_zoom: 1.0,
            playlist_drag: None,
            audio_clip_drag: None,
            audio_clip_context_menu: None,
            take_folder_context_menu: None,
            take_folder_editor: None,
            take_folder_comp_drag: None,
            flex_editor: None,
            flex_editor_mode: FlexEditorMode::Time,
            flex_editor_raw: None,
            flex_marker_drag: None,
            flex_note_drag: None,
            record_armed_track: None,
            selected_input_device: None,
            recording: None,
            recording_message: None,
            selected_output_device: None,
            selected_output_sample_rate: None,
            output_device_message: None,
            #[cfg(unix)]
            mcp_rx: mcp_control::spawn(),
        }
    }
}

/// Bundles the Channel Rack's mutable app-state borrows so `channel_rack_contents_ui` (shared
/// between the docked `egui::Panel::left` rendering and the detached-window rendering — see
/// `SimpleDawApp::ui`) doesn't need a dozen positional parameters. `song.lock()` is held for the
/// rest of `SimpleDawApp::ui`, so this borrows individual fields rather than `&mut self` — a
/// method taking `&mut self` would conflict with that outstanding lock guard.
struct ChannelRackUi<'a> {
    selected_track: &'a Option<usize>,
    selected_beats_track: &'a Option<usize>,
    detached: &'a mut bool,
    track_effect_slots: &'a TrackEffectSlots,
    track_effect_instances: &'a mut Vec<Vec<Option<PluginInstance<DawHost>>>>,
    track_effect_guis: &'a mut Vec<Vec<Option<PluginGuiHandle>>>,
    track_effect_paths: &'a mut Vec<Vec<String>>,
    track_effect_messages: &'a mut Vec<Vec<Option<(bool, String)>>>,
    track_meters: &'a MeterHandles,
    effect_editor: &'a mut Option<EffectEditorTarget>,
    device_chain_focus: &'a mut Option<DeviceChainFocus>,
    /// The record-armed `Audio`-kind track (if any) and its chosen input device — see
    /// `SimpleDawApp::record_armed_track`/`selected_input_device`.
    record_armed_track: &'a mut Option<usize>,
    selected_input_device: &'a mut Option<String>,
}

/// The Channel Rack's heading/"+ Add" menu/track-row list, including the Detach/Dock toggle
/// button — shared by the docked and detached-window renderings so the two stay in sync.
fn channel_rack_contents_ui(
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

/// Bundles the Mixer's mutable app-state borrows, for the same reason as `ChannelRackUi` — reused
/// between the docked and detached-window renderings. Also carries the master bus's own
/// effect-chain bookkeeping (see `SimpleDawApp::master_effect_paths` and friends) so the Mixer can
/// show a Master strip alongside the per-track ones, the same chain the "Plugins" window's "Master
/// bus FX chain" section edits.
struct MixerUi<'a> {
    detached: &'a mut bool,
    track_effect_slots: &'a TrackEffectSlots,
    track_effect_instances: &'a mut Vec<Vec<Option<PluginInstance<DawHost>>>>,
    track_effect_guis: &'a mut Vec<Vec<Option<PluginGuiHandle>>>,
    track_effect_paths: &'a mut Vec<Vec<String>>,
    track_effect_messages: &'a mut Vec<Vec<Option<(bool, String)>>>,
    track_meters: &'a MeterHandles,
    effect_editor: &'a mut Option<EffectEditorTarget>,
    master_effect_paths: &'a mut Vec<String>,
    master_effect_slots: MasterEffectSlots,
    master_effect_instances: &'a mut Vec<Option<PluginInstance<DawHost>>>,
    master_effect_guis: &'a mut Vec<Option<PluginGuiHandle>>,
    master_effect_messages: &'a mut Vec<Option<(bool, String)>>,
    master_meter: &'a MeterHandles,
    /// Every send bus's own effect-chain bookkeeping — same shape as `track_effect_*` (one entry
    /// per `Song::sends` row), kept in sync via `resize_track_effects`/`remove_track_effects`, the
    /// same helpers `track_effect_*`/`send_effect_*` both reuse (see `SimpleDawApp::send_effect_slots`).
    send_effect_slots: &'a SendEffectSlots,
    send_effect_instances: &'a mut Vec<Vec<Option<PluginInstance<DawHost>>>>,
    send_effect_guis: &'a mut Vec<Vec<Option<PluginGuiHandle>>>,
    send_effect_paths: &'a mut Vec<Vec<String>>,
    send_effect_messages: &'a mut Vec<Vec<Option<(bool, String)>>>,
    /// Every submix bus's own effect-chain bookkeeping — same shape/sync mechanism as
    /// `send_effect_*` above (one entry per `Song::submixes` row).
    submix_effect_slots: &'a SubmixEffectSlots,
    submix_effect_instances: &'a mut Vec<Vec<Option<PluginInstance<DawHost>>>>,
    submix_effect_guis: &'a mut Vec<Vec<Option<PluginGuiHandle>>>,
    submix_effect_paths: &'a mut Vec<Vec<String>>,
    submix_effect_messages: &'a mut Vec<Vec<Option<(bool, String)>>>,
    submix_meters: &'a MeterHandles,
}

/// The Mixer's heading/Detach toggle plus one classic vertical channel strip per track, ending in
/// a Master strip — shared by the docked and detached-window renderings so the two stay in sync.
/// Unlike the Channel Rack's compact horizontal rows, each strip lays its controls out top to
/// bottom (name, FX, pan, mute/solo, then a tall fader) the way a hardware/DAW mixer console does.
/// These are the same `Track::volume`/`pan`/`muted`/`solo`/`effects` the Channel Rack row already
/// edits inline — the Mixer is an additional view onto the same data, not a separate copy of it.
fn mixer_contents_ui(
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

/// Bundles the Piano Roll's mutable app-state borrows for the same reason as `ChannelRackUi`.
struct PianoRollPanelUi<'a> {
    selected_track: Option<usize>,
    piano_roll_drag: &'a mut Option<PianoRollDrag>,
    selected_notes: &'a mut HashSet<u64>,
    /// See `SimpleDawApp::groove_quantize_grid_ticks` and its sibling `groove_*` fields — bundled
    /// into one borrow the same way the other toolbar controls above are.
    groove_quantize_grid_ticks: &'a mut usize,
    groove_quantize_strength: &'a mut f32,
    groove_humanize_timing_ticks: &'a mut usize,
    groove_humanize_velocity: &'a mut u8,
    groove_template_index: &'a mut usize,
    piano_roll_zoom: &'a mut f32,
    /// See `SimpleDawApp::piano_roll_scale_root`.
    scale_root: &'a mut u8,
    /// See `SimpleDawApp::piano_roll_scale`.
    scale: &'a mut PianoRollScale,
    /// Which of `selected_track`'s own `regions` to show/edit — see
    /// `SimpleDawApp::piano_roll_region`. Set only by double-clicking a region in the Playlist.
    editing_region_index: &'a mut Option<usize>,
    /// See `SimpleDawApp::piano_roll_scroll_to`.
    scroll_to: &'a mut Option<usize>,
    /// The open region's own track's live effect chain — read by `automation_lanes_ui`'s "+ Add
    /// Lane" menu to offer a currently-loaded CLAP plugin's real parameter names.
    track_effect_slots: &'a TrackEffectSlots,
    /// Every send bus's and the master bus's live effect chains — same reason as
    /// `track_effect_slots`, for `automation_lanes_ui`'s "Send FX"/"Master FX" cross-bus targets.
    send_effect_slots: &'a SendEffectSlots,
    master_effect_slots: &'a MasterEffectSlots,
    /// See `AutomationDrag`.
    automation_drag: &'a mut Option<AutomationDrag>,
    /// See `SimpleDawApp::track_automation_drag`.
    track_automation_drag: &'a mut Option<AutomationDrag>,
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

/// The Piano Roll's header (selected track name/mute badge) and note grid, rendered inside the
/// always-detached Piano Roll window (see `ui` in `impl eframe::App for SimpleDawApp`). Unlike
/// the Channel Rack, the Piano Roll has no docked mode: it only exists when a piano-roll track is
/// selected, and closing its window clears the selection instead of re-docking it. There's no
/// picker here to switch regions — double-click a different region in the Playlist instead (see
/// `PlaylistEditorTargets`).
fn piano_roll_contents_ui(
    ui: &mut egui::Ui,
    song: &mut Song,
    current_tick: Option<usize>,
    panel: &mut PianoRollPanelUi,
) {
    let selected = panel
        .selected_track
        .filter(|&i| i < song.tracks.len())
        .filter(|&i| song.tracks[i].kind == TrackKind::PianoRoll);
    let region = selected.and_then(|index| {
        let region_index = (*panel.editing_region_index)?;
        (region_index < song.tracks[index].regions.len()).then_some((index, region_index))
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
            if let Some((_, region_index)) = region {
                ui.separator();
                ui.weak(&song.tracks[index].regions[region_index].name);
            }
        }
        None => {
            ui.heading("Piano Roll");
        }
    });
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
        Some((index, region_index)) => {
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
    }
}

/// Resizes every per-track effect bookkeeping collection to match `track_count` — called after
/// loading a song, since the new song can have a different number of tracks than the old one.
fn resize_track_effects(
    slots: &TrackEffectSlots,
    instances: &mut Vec<Vec<Option<PluginInstance<DawHost>>>>,
    guis: &mut Vec<Vec<Option<PluginGuiHandle>>>,
    paths: &mut Vec<Vec<String>>,
    messages: &mut Vec<Vec<Option<(bool, String)>>>,
    track_count: usize,
) {
    if let Ok(mut guard) = slots.lock() {
        guard.resize_with(track_count, Vec::new);
    }
    instances.resize_with(track_count, Vec::new);
    guis.resize_with(track_count, Vec::new);
    paths.resize_with(track_count, Vec::new);
    messages.resize_with(track_count, Vec::new);
}

/// Removes the bookkeeping entry at `index` from every per-track effect collection, keeping them
/// aligned with `song.tracks` after a track is deleted. Unlike `resize_track_effects` (which only
/// grows/shrinks from the end, correct for an append-at-the-end "add track"), deleting from the
/// middle needs each collection's entry at `index` removed, not just truncated.
fn remove_track_effects(
    slots: &TrackEffectSlots,
    instances: &mut Vec<Vec<Option<PluginInstance<DawHost>>>>,
    guis: &mut Vec<Vec<Option<PluginGuiHandle>>>,
    paths: &mut Vec<Vec<String>>,
    messages: &mut Vec<Vec<Option<(bool, String)>>>,
    index: usize,
) {
    if let Ok(mut guard) = slots.lock() {
        if index < guard.len() {
            guard.remove(index);
        }
    }
    if index < instances.len() {
        instances.remove(index);
    }
    if index < guis.len() {
        guis.remove(index);
    }
    if index < paths.len() {
        paths.remove(index);
    }
    if index < messages.len() {
        messages.remove(index);
    }
}

/// Resizes a per-track `MeterHandles` (see `metering`'s module doc) to `track_count` entries,
/// mirroring `resize_track_effects` — called at the same sites, right alongside it, whenever
/// `song.tracks` grows.
fn resize_track_meters(meters: &MeterHandles, track_count: usize) {
    if let Ok(mut guard) = meters.lock() {
        guard.resize_with(track_count, || Arc::new(metering::ChannelMeterAtomics::new()));
    }
}

/// Removes the meter entry at `index`, mirroring `remove_track_effects` — called alongside it
/// whenever a track is deleted from the middle of `song.tracks`.
fn remove_track_meter(meters: &MeterHandles, index: usize) {
    if let Ok(mut guard) = meters.lock()
        && index < guard.len()
    {
        guard.remove(index);
    }
}

/// Bundles the pieces of `SimpleDawApp` an MCP command handler needs, mirroring `ChannelRackUi`'s
/// "disjoint field borrows" pattern (see above) — `song` in `apply_mcp_command` is borrowed
/// straight from `self.song.lock()`, not through `self`, so a real `&mut self` method can't be
/// called alongside it; constructing this struct from individual `self.field` borrows can.
#[cfg(unix)]
struct McpContext<'a> {
    transport: &'a Transport,
    sample_rate: Option<u32>,
    engine_config: Option<(f64, u32, u32)>,
    song_path: &'a mut String,
    master_effect_paths: &'a mut Vec<String>,
    master_effect_slots: &'a MasterEffectSlots,
    master_effect_instances: &'a mut Vec<Option<PluginInstance<DawHost>>>,
    master_effect_guis: &'a mut Vec<Option<PluginGuiHandle>>,
    master_effect_messages: &'a mut Vec<Option<(bool, String)>>,
    track_effect_slots: &'a TrackEffectSlots,
    track_effect_instances: &'a mut Vec<Vec<Option<PluginInstance<DawHost>>>>,
    track_effect_guis: &'a mut Vec<Vec<Option<PluginGuiHandle>>>,
    track_effect_paths: &'a mut Vec<Vec<String>>,
    track_effect_messages: &'a mut Vec<Vec<Option<(bool, String)>>>,
    send_effect_slots: &'a SendEffectSlots,
    send_effect_instances: &'a mut Vec<Vec<Option<PluginInstance<DawHost>>>>,
    send_effect_guis: &'a mut Vec<Vec<Option<PluginGuiHandle>>>,
    send_effect_paths: &'a mut Vec<Vec<String>>,
    send_effect_messages: &'a mut Vec<Vec<Option<(bool, String)>>>,
    submix_effect_slots: &'a SubmixEffectSlots,
    submix_effect_instances: &'a mut Vec<Vec<Option<PluginInstance<DawHost>>>>,
    submix_effect_guis: &'a mut Vec<Vec<Option<PluginGuiHandle>>>,
    submix_effect_paths: &'a mut Vec<Vec<String>>,
    submix_effect_messages: &'a mut Vec<Vec<Option<(bool, String)>>>,
    track_meters: &'a MeterHandles,
    submix_meters: &'a MeterHandles,
}

#[cfg(unix)]
fn parse_track_kind(s: &str) -> Result<TrackKind, String> {
    match s {
        "step_grid" => Ok(TrackKind::StepGrid),
        "piano_roll" => Ok(TrackKind::PianoRoll),
        "audio" => Ok(TrackKind::Audio),
        other => Err(format!(
            "unknown track kind \"{other}\" (expected step_grid, piano_roll, or audio)"
        )),
    }
}

#[cfg(unix)]
fn track_kind_str(kind: TrackKind) -> &'static str {
    match kind {
        TrackKind::StepGrid => "step_grid",
        TrackKind::PianoRoll => "piano_roll",
        TrackKind::Audio => "audio",
    }
}

#[cfg(unix)]
fn parse_synth_engine(s: &str) -> Result<SynthEngine, String> {
    match s {
        "simple" => Ok(SynthEngine::Simple),
        "trine" => Ok(SynthEngine::Trine),
        "wave" => Ok(SynthEngine::Wave),
        other => Err(format!(
            "unknown synth engine \"{other}\" (expected simple, trine, or wave)"
        )),
    }
}

#[cfg(unix)]
fn synth_engine_str(engine: SynthEngine) -> &'static str {
    match engine {
        SynthEngine::Simple => "simple",
        SynthEngine::Trine => "trine",
        SynthEngine::Wave => "wave",
    }
}

#[cfg(unix)]
fn mcp_track_mut(song: &mut Song, index: usize) -> Result<&mut Track, String> {
    let track_count = song.tracks.len();
    song.tracks
        .get_mut(index)
        .ok_or_else(|| format!("no track at index {index} (song has {track_count} tracks)"))
}

#[cfg(unix)]
#[derive(serde::Deserialize)]
struct McpAddTrackParams {
    name: String,
    kind: String,
    #[serde(default)]
    midi_channel: Option<u8>,
}

#[cfg(unix)]
#[derive(serde::Deserialize)]
struct McpTrackIndexParams {
    track: usize,
}

#[cfg(unix)]
#[derive(serde::Deserialize)]
struct McpSetTrackVolumeParams {
    track: usize,
    volume: f32,
}

#[cfg(unix)]
#[derive(serde::Deserialize)]
struct McpSetTrackMuteParams {
    track: usize,
    muted: bool,
}

#[cfg(unix)]
#[derive(serde::Deserialize)]
struct McpSetTrackSoloParams {
    track: usize,
    solo: bool,
}

#[cfg(unix)]
#[derive(serde::Deserialize)]
struct McpAddRegionParams {
    track: usize,
    start_step: usize,
}

#[cfg(unix)]
#[derive(serde::Deserialize)]
struct McpAddLaneParams {
    track: usize,
    name: String,
    pitch: u8,
}

#[cfg(unix)]
#[derive(serde::Deserialize)]
struct McpSetStepParams {
    track: usize,
    region: usize,
    lane: usize,
    step: usize,
    /// 0 clears the step; 1-127 sets it with that velocity (mirrors `Lane::steps`' own encoding).
    velocity: u8,
}

#[cfg(unix)]
#[derive(serde::Deserialize)]
struct McpAddNoteParams {
    track: usize,
    region: usize,
    pitch: u8,
    start_step: usize,
    length_steps: usize,
    velocity: u8,
}

#[cfg(unix)]
#[derive(serde::Deserialize, Default)]
struct McpListPresetsParams {
    #[serde(default)]
    engine: Option<String>,
}

#[cfg(unix)]
#[derive(serde::Deserialize)]
struct McpApplyPresetParams {
    track: usize,
    preset_name: String,
}

#[cfg(unix)]
#[derive(serde::Deserialize)]
struct McpSetBpmParams {
    bpm: f32,
}

#[cfg(unix)]
#[derive(serde::Deserialize)]
struct McpSaveSongParams {
    path: String,
}

#[cfg(unix)]
#[derive(serde::Deserialize)]
struct McpLoadSongParams {
    path: String,
}

#[cfg(unix)]
fn default_export_loops() -> u32 {
    1
}

#[cfg(unix)]
#[derive(serde::Deserialize)]
struct McpExportWavParams {
    path: String,
    #[serde(default = "default_export_loops")]
    loops: u32,
}

/// Parses `params` into `T`, mapping a schema mismatch into the same `Result<_, String>` shape
/// every other MCP command handler returns — so a bad tool call from the LLM comes back as a
/// normal tool error instead of panicking the socket-handling thread.
#[cfg(unix)]
fn parse_mcp_params<T: serde::de::DeserializeOwned>(params: serde_json::Value) -> Result<T, String> {
    serde_json::from_value(params).map_err(|err| format!("invalid parameters: {err}"))
}

/// Applies one queued MCP command (see `mcp_control::McpRequest`) to the live song, returning
/// either a JSON result payload or a human-readable error — the latter is sent back to
/// `simple-daw-mcp`, which surfaces it to the LLM as a tool error rather than a crash. Every
/// mutation here goes through the same model methods/helpers a UI button would use, so behavior
/// (and invariants like keeping `track_effect_*` aligned with `song.tracks`) stays identical to
/// driving the app by hand. Keep the tool names/params here in sync with the schemas declared in
/// `src/bin/simple-daw-mcp.rs`'s `tool_definitions()`.
#[cfg(unix)]
fn apply_mcp_command(
    cmd: &str,
    params: serde_json::Value,
    song: &mut Song,
    ctx: &mut McpContext,
) -> Result<serde_json::Value, String> {
    use serde_json::json;

    match cmd {
        "list_tracks" => {
            let tracks: Vec<serde_json::Value> = song
                .tracks
                .iter()
                .enumerate()
                .map(|(index, track)| {
                    json!({
                        "index": index,
                        "name": track.name,
                        "kind": track_kind_str(track.kind),
                        "midi_channel": track.midi_channel,
                        "muted": track.muted,
                        "solo": track.solo,
                        "volume": track.volume,
                        "synth_engine": synth_engine_str(track.synth_engine),
                        "region_count": track.regions.len(),
                    })
                })
                .collect();
            Ok(json!({ "tracks": tracks }))
        }
        "add_track" => {
            let p: McpAddTrackParams = parse_mcp_params(params)?;
            let kind = parse_track_kind(&p.kind)?;
            let midi_channel = p
                .midi_channel
                .unwrap_or((song.tracks.len() as u8 % 16) + 1);
            let index = song.add_track(p.name, midi_channel, kind);
            resize_track_effects(
                ctx.track_effect_slots,
                ctx.track_effect_instances,
                ctx.track_effect_guis,
                ctx.track_effect_paths,
                ctx.track_effect_messages,
                song.tracks.len(),
            );
            resize_track_meters(ctx.track_meters, song.tracks.len());
            Ok(json!({ "index": index }))
        }
        "remove_track" => {
            let p: McpTrackIndexParams = parse_mcp_params(params)?;
            if p.track >= song.tracks.len() {
                return Err(format!(
                    "no track at index {} (song has {} tracks)",
                    p.track,
                    song.tracks.len()
                ));
            }
            song.remove_track(p.track);
            remove_track_effects(
                ctx.track_effect_slots,
                ctx.track_effect_instances,
                ctx.track_effect_guis,
                ctx.track_effect_paths,
                ctx.track_effect_messages,
                p.track,
            );
            remove_track_meter(ctx.track_meters, p.track);
            Ok(json!({}))
        }
        "set_track_volume" => {
            let p: McpSetTrackVolumeParams = parse_mcp_params(params)?;
            mcp_track_mut(song, p.track)?.volume = p.volume.max(0.0);
            Ok(json!({}))
        }
        "set_track_mute" => {
            let p: McpSetTrackMuteParams = parse_mcp_params(params)?;
            mcp_track_mut(song, p.track)?.muted = p.muted;
            Ok(json!({}))
        }
        "set_track_solo" => {
            let p: McpSetTrackSoloParams = parse_mcp_params(params)?;
            mcp_track_mut(song, p.track)?.solo = p.solo;
            Ok(json!({}))
        }
        "add_region" => {
            let p: McpAddRegionParams = parse_mcp_params(params)?;
            let steps_per_bar = song.steps_per_bar();
            let track = mcp_track_mut(song, p.track)?;
            if track.kind == TrackKind::Audio {
                return Err("audio tracks use audio_clips, not regions".to_string());
            }
            let region_index = track.add_region(p.start_step, steps_per_bar);
            Ok(json!({ "region_index": region_index }))
        }
        "add_lane" => {
            let p: McpAddLaneParams = parse_mcp_params(params)?;
            let track = mcp_track_mut(song, p.track)?;
            if track.kind != TrackKind::StepGrid {
                return Err("add_lane only applies to step_grid tracks".to_string());
            }
            track.add_lane(p.name, p.pitch);
            let lane_index = track
                .regions
                .first()
                .and_then(|region| match &region.content {
                    RegionContent::StepGrid(lanes) => Some(lanes.len().saturating_sub(1)),
                    _ => None,
                })
                .unwrap_or(0);
            Ok(json!({ "lane_index": lane_index }))
        }
        "set_step" => {
            let p: McpSetStepParams = parse_mcp_params(params)?;
            let track = mcp_track_mut(song, p.track)?;
            let region = track
                .regions
                .get_mut(p.region)
                .ok_or_else(|| format!("no region {} on track {}", p.region, p.track))?;
            let RegionContent::StepGrid(lanes) = &mut region.content else {
                return Err("region is not a step-grid region".to_string());
            };
            let lane = lanes
                .get_mut(p.lane)
                .ok_or_else(|| format!("no lane {} in region {}", p.lane, p.region))?;
            if p.step >= lane.steps.len() {
                return Err(format!(
                    "step {} out of range (region has {} steps)",
                    p.step,
                    lane.steps.len()
                ));
            }
            if p.velocity == 0 {
                lane.steps[p.step] = None;
            } else {
                lane.set_step(p.step, p.velocity.min(127));
            }
            Ok(json!({}))
        }
        "add_note" => {
            let p: McpAddNoteParams = parse_mcp_params(params)?;
            let steps_per_bar = song.steps_per_bar();
            let next_note_id = &mut song.next_note_id;
            let track_count = song.tracks.len();
            let track = song
                .tracks
                .get_mut(p.track)
                .ok_or_else(|| format!("no track at index {} (song has {track_count} tracks)", p.track))?;
            let region = track
                .regions
                .get_mut(p.region)
                .ok_or_else(|| format!("no region {} on track {}", p.region, p.track))?;
            let RegionContent::PianoRoll(notes) = &mut region.content else {
                return Err("region is not a piano-roll region".to_string());
            };
            let id = add_note(
                notes,
                next_note_id,
                p.pitch,
                p.start_step * TICKS_PER_STEP,
                p.length_steps * TICKS_PER_STEP,
                p.velocity.min(127),
            );
            model::grow_length_to_fit_notes(&mut region.content_length_steps, notes, steps_per_bar);
            Ok(json!({ "note_id": id }))
        }
        "list_presets" => {
            let p: McpListPresetsParams = parse_mcp_params(params)?;
            let engine_filter = p.engine.as_deref().map(parse_synth_engine).transpose()?;
            let presets: Vec<serde_json::Value> = factory_presets()
                .into_iter()
                .chain(song.synth_presets.clone())
                .filter(|preset| engine_filter.map_or(true, |e| preset.engine == e))
                .map(|preset| json!({"name": preset.name, "engine": synth_engine_str(preset.engine)}))
                .collect();
            Ok(json!({ "presets": presets }))
        }
        "apply_preset" => {
            let p: McpApplyPresetParams = parse_mcp_params(params)?;
            let preset = factory_presets()
                .into_iter()
                .chain(song.synth_presets.clone())
                .find(|preset| preset.name == p.preset_name)
                .ok_or_else(|| format!("no preset named \"{}\"", p.preset_name))?;
            let track = mcp_track_mut(song, p.track)?;
            track.synth_engine = preset.engine;
            match preset.engine {
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
            Ok(json!({}))
        }
        "set_bpm" => {
            let p: McpSetBpmParams = parse_mcp_params(params)?;
            if p.bpm <= 0.0 {
                return Err("bpm must be positive".to_string());
            }
            song.bpm = p.bpm;
            Ok(json!({}))
        }
        "play" => {
            ctx.transport.set_playing(true);
            Ok(json!({}))
        }
        "stop" => {
            ctx.transport.set_playing(false);
            Ok(json!({}))
        }
        "get_playback_state" => Ok(json!({
            "playing": ctx.transport.is_playing(),
            "current_tick": ctx.transport.current_tick(),
            "bpm": song.bpm,
            "song_name": song.name,
        })),
        "save_song" => {
            let p: McpSaveSongParams = parse_mcp_params(params)?;
            sync_song_effects(
                song,
                ctx.master_effect_paths,
                ctx.master_effect_slots,
                ctx.track_effect_paths,
                ctx.track_effect_slots,
                ctx.send_effect_paths,
                ctx.send_effect_slots,
                ctx.submix_effect_paths,
                ctx.submix_effect_slots,
            );
            let path = p.path.trim().to_string();
            let (ok, message) = perform_save(song, &path);
            if ok {
                *ctx.song_path = path;
                Ok(json!({ "message": message }))
            } else {
                Err(message)
            }
        }
        "load_song" => {
            let p: McpLoadSongParams = parse_mcp_params(params)?;
            let path = p.path.trim().to_string();
            let loaded = perform_load(&path, ctx.sample_rate)?;
            let track_count = loaded.tracks.len();
            let send_count = loaded.sends.len();
            let submix_count = loaded.submixes.len();
            let master_effect_specs = loaded.master_effects.clone();
            let track_effect_specs: Vec<Vec<TrackEffectConfig>> =
                loaded.tracks.iter().map(|t| t.effects.clone()).collect();
            let send_effect_specs: Vec<Vec<TrackEffectConfig>> =
                loaded.sends.iter().map(|s| s.effects.clone()).collect();
            let submix_effect_specs: Vec<Vec<TrackEffectConfig>> =
                loaded.submixes.iter().map(|s| s.effects.clone()).collect();
            *song = loaded;
            *ctx.song_path = path;
            resize_track_effects(
                ctx.track_effect_slots,
                ctx.track_effect_instances,
                ctx.track_effect_guis,
                ctx.track_effect_paths,
                ctx.track_effect_messages,
                track_count,
            );
            resize_track_meters(ctx.track_meters, track_count);
            resize_track_effects(
                ctx.send_effect_slots,
                ctx.send_effect_instances,
                ctx.send_effect_guis,
                ctx.send_effect_paths,
                ctx.send_effect_messages,
                send_count,
            );
            resize_track_effects(
                ctx.submix_effect_slots,
                ctx.submix_effect_instances,
                ctx.submix_effect_guis,
                ctx.submix_effect_paths,
                ctx.submix_effect_messages,
                submix_count,
            );
            resize_track_meters(ctx.submix_meters, submix_count);
            apply_loaded_effects(
                ctx.master_effect_paths,
                ctx.master_effect_instances,
                ctx.master_effect_guis,
                ctx.master_effect_slots,
                ctx.master_effect_messages,
                master_effect_specs,
                ctx.track_effect_paths,
                ctx.track_effect_instances,
                ctx.track_effect_guis,
                ctx.track_effect_messages,
                ctx.track_effect_slots,
                track_effect_specs,
                ctx.send_effect_paths,
                ctx.send_effect_instances,
                ctx.send_effect_guis,
                ctx.send_effect_messages,
                ctx.send_effect_slots,
                send_effect_specs,
                ctx.submix_effect_paths,
                ctx.submix_effect_instances,
                ctx.submix_effect_guis,
                ctx.submix_effect_messages,
                ctx.submix_effect_slots,
                submix_effect_specs,
                ctx.engine_config,
            );
            Ok(json!({ "track_count": track_count }))
        }
        "export_wav" => {
            let p: McpExportWavParams = parse_mcp_params(params)?;
            let sample_rate = ctx
                .sample_rate
                .ok_or_else(|| "no audio engine available (can't determine sample rate)".to_string())?;
            let path = std::path::Path::new(p.path.trim());
            audio::render_song_to_wav(song, sample_rate, p.loops.max(1), path)
                .map_err(|err| format!("{err:#}"))?;
            Ok(json!({ "path": p.path }))
        }
        other => Err(format!("unknown command \"{other}\"")),
    }
}

/// Renders sliders for `effect`'s parameters (if any are loaded), writing changes straight into
/// the plugin via `LoadedEffect::set_param`. Shared by both the master-bus and per-track windows.
fn effect_params_ui(ui: &mut egui::Ui, effect: Option<&mut LoadedEffect>) {
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
fn plugin_gui_button_ui(
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
fn close_effect_gui(
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
fn built_in_effect_params_ui(ui: &mut egui::Ui, effect: &mut BuiltInEffect) {
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

/// Renders the waveform picker and attack/decay sliders for a track's built-in synth voice,
/// shown inside the bottom Device Panel (see `device_panel_contents_ui`). Laid out as two
/// columns (oscillators | envelope/filter/LFO) to keep the panel from growing too tall.
fn synth_params_ui(ui: &mut egui::Ui, synth: &mut SynthParams) {
    ui.columns(2, |columns| {
        synth_oscillators_ui(&mut columns[0], synth);
        synth_modulation_ui(&mut columns[1], synth);
    });
}

/// Sample of a raw oscillator cycle in `[-1, 1]` for `phase` running `0..1`, for the small
/// waveform-preview canvases below. Mirrors `audio::waveform_sample` (kept private to the
/// real-time engine) since this is purely for drawing, not audio.
fn waveform_shape_sample(waveform: SynthWaveform, phase: f32, pulse_width: f32) -> f32 {
    match waveform {
        SynthWaveform::Sine => (phase * std::f32::consts::TAU).sin(),
        SynthWaveform::Saw => 2.0 * phase - 1.0,
        SynthWaveform::Square => {
            if phase < pulse_width {
                1.0
            } else {
                -1.0
            }
        }
        SynthWaveform::Triangle => 4.0 * (phase - 0.5).abs() - 1.0,
        // Mirrors `audio::hash_to_bipolar` — good enough for a jagged-looking preview.
        SynthWaveform::Noise => {
            let mut h = phase.to_bits();
            h ^= h >> 16;
            h = h.wrapping_mul(0x7feb_352d);
            h ^= h >> 15;
            h = h.wrapping_mul(0x846c_a68b);
            h ^= h >> 16;
            (h as f32 / u32::MAX as f32) * 2.0 - 1.0
        }
    }
}

/// Draws `cycles` repetitions of `waveform`'s shape across `rect`, scaled by `amplitude` (0..1)
/// and vertically centered. Used by the oscillator and LFO preview canvases.
fn paint_waveform_shape(
    painter: &egui::Painter,
    rect: egui::Rect,
    waveform: SynthWaveform,
    pulse_width: f32,
    amplitude: f32,
    cycles: f32,
    stroke: egui::Stroke,
) {
    let samples = 200;
    let mid_y = rect.center().y;
    let half_h = rect.height() * 0.5 * 0.9;
    let points: Vec<egui::Pos2> = (0..=samples)
        .map(|i| {
            let t = i as f32 / samples as f32;
            let phase = (t * cycles).fract();
            let sample = waveform_shape_sample(waveform, phase, pulse_width) * amplitude;
            egui::pos2(rect.left() + t * rect.width(), mid_y - sample * half_h)
        })
        .collect();
    painter.add(egui::Shape::line(points, stroke));
}

/// Samples one oscillator's shape across `cycles` of a reference oscillator's period, optionally
/// hard-synced to it — the same math `oscillator2_preview_ui` uses for Oscillator 2, generalized
/// so `TrineParams`'s three oscillators can share it (see `trine_oscillators_preview_ui`).
/// `ratio` is this oscillator's frequency relative to the reference (from semitone/cent tuning);
/// when `sync` is true the phase re-zeroes every reference cycle (`(fract(t) * ratio).fract()`),
/// mirroring `audio::Voice`'s hard-sync; when false it free-runs (`(t * ratio).fract()`).
fn synced_oscillator_points(
    rect: egui::Rect,
    waveform: SynthWaveform,
    pulse_width: f32,
    ratio: f32,
    sync: bool,
    amplitude: f32,
    cycles: f32,
    samples: usize,
) -> Vec<egui::Pos2> {
    let mid_y = rect.center().y;
    let half_h = rect.height() * 0.5 * 0.9;
    (0..=samples)
        .map(|i| {
            let t = i as f32 / samples as f32 * cycles;
            let phase = if sync {
                (t.fract() * ratio).fract()
            } else {
                (t * ratio).fract()
            };
            let sample = waveform_shape_sample(waveform, phase, pulse_width) * amplitude;
            egui::pos2(
                rect.left() + i as f32 / samples as f32 * rect.width(),
                mid_y - sample * half_h,
            )
        })
        .collect()
}

/// Small canvas previewing the combined oscillator output: Oscillator 1 in the accent color,
/// Oscillator 2 faded in proportion to its mix, and the sub-oscillator as a thin low-amplitude
/// line when its level is above zero.
fn oscillator_preview_ui(ui: &mut egui::Ui, synth: &SynthParams) {
    let (response, painter) =
        ui.allocate_painter(egui::vec2(ui.available_width(), 60.0), egui::Sense::hover());
    let rect = response.rect;
    painter.rect_filled(rect, 2.0, ui.visuals().extreme_bg_color);
    painter.line_segment(
        [
            egui::pos2(rect.left(), rect.center().y),
            egui::pos2(rect.right(), rect.center().y),
        ],
        egui::Stroke::new(0.5, ui.visuals().weak_text_color()),
    );

    if synth.sub_osc_mix > 0.0 {
        paint_waveform_shape(
            &painter,
            rect,
            SynthWaveform::Sine,
            0.5,
            synth.sub_osc_mix,
            1.0,
            egui::Stroke::new(1.0, ui.visuals().weak_text_color()),
        );
    }
    if synth.osc2_mix > 0.0 {
        let osc2_color =
            egui::Color32::from_rgb(230, 160, 90).gamma_multiply(synth.osc2_mix.max(0.25));
        paint_waveform_shape(
            &painter,
            rect,
            synth.osc2_waveform,
            0.5,
            1.0,
            2.0,
            egui::Stroke::new(1.5, osc2_color),
        );
    }
    let osc1_amplitude = if synth.osc2_mix > 0.0 {
        1.0 - synth.osc2_mix
    } else {
        1.0
    };
    paint_waveform_shape(
        &painter,
        rect,
        synth.waveform,
        synth.pulse_width,
        osc1_amplitude.max(0.15),
        2.0,
        egui::Stroke::new(2.0, egui::Color32::from_rgb(120, 170, 230)),
    );
}

/// Small canvas dedicated to Oscillator 2: a faint reference line for Oscillator 1 (one cycle
/// per `cycles`) and Oscillator 2's own shape overlaid in orange. Both phases are computed
/// analytically from the elapsed fraction of Oscillator 1's cycle, so this mirrors
/// `audio::Voice::next_sample`'s hard-sync math exactly without needing to simulate sample by
/// sample: free-running osc2 phase is `(t * ratio).fract()`, and — since sync resets osc2 to 0 at
/// every osc1 wrap — synced osc2 phase is `(fract(t) * ratio).fract()`, i.e. re-zeroed each cycle.
fn oscillator2_preview_ui(ui: &mut egui::Ui, synth: &SynthParams) {
    let (response, painter) =
        ui.allocate_painter(egui::vec2(ui.available_width(), 60.0), egui::Sense::hover());
    let rect = response.rect;
    painter.rect_filled(rect, 2.0, ui.visuals().extreme_bg_color);
    painter.line_segment(
        [
            egui::pos2(rect.left(), rect.center().y),
            egui::pos2(rect.right(), rect.center().y),
        ],
        egui::Stroke::new(0.5, ui.visuals().weak_text_color()),
    );

    let ratio =
        2f32.powf(synth.osc2_semitones as f32 / 12.0) * 2f32.powf(synth.osc2_detune_cents / 1200.0);
    let cycles = 3.0;
    let samples = 300;
    let mid_y = rect.center().y;
    let half_h = rect.height() * 0.5 * 0.9;

    let osc1_points: Vec<egui::Pos2> = (0..=samples)
        .map(|i| {
            let t = i as f32 / samples as f32 * cycles;
            let sample = waveform_shape_sample(synth.waveform, t.fract(), synth.pulse_width) * 0.6;
            egui::pos2(
                rect.left() + i as f32 / samples as f32 * rect.width(),
                mid_y - sample * half_h,
            )
        })
        .collect();
    painter.add(egui::Shape::line(
        osc1_points,
        egui::Stroke::new(1.0, ui.visuals().weak_text_color()),
    ));

    let osc2_points: Vec<egui::Pos2> = (0..=samples)
        .map(|i| {
            let t = i as f32 / samples as f32 * cycles;
            let phase = if synth.osc2_sync {
                (t.fract() * ratio).fract()
            } else {
                (t * ratio).fract()
            };
            let sample = waveform_shape_sample(synth.osc2_waveform, phase, 0.5);
            egui::pos2(
                rect.left() + i as f32 / samples as f32 * rect.width(),
                mid_y - sample * half_h,
            )
        })
        .collect();
    painter.add(egui::Shape::line(
        osc2_points,
        egui::Stroke::new(2.0, egui::Color32::from_rgb(230, 160, 90)),
    ));
}

fn synth_oscillators_ui(ui: &mut egui::Ui, synth: &mut SynthParams) {
    ui.strong("Oscillator");
    ui.horizontal(|ui| {
        ui.label("Waveform:");
        for (label, waveform) in [
            ("Sine", SynthWaveform::Sine),
            ("Saw", SynthWaveform::Saw),
            ("Square", SynthWaveform::Square),
            ("Triangle", SynthWaveform::Triangle),
        ] {
            if ui
                .selectable_label(synth.waveform == waveform, label)
                .clicked()
            {
                synth.waveform = waveform;
            }
        }
    });
    ui.add_enabled(
        synth.waveform == SynthWaveform::Square,
        egui::Slider::new(&mut synth.pulse_width, 0.05..=0.95).text("Pulse width"),
    )
    .on_hover_text("Duty cycle of the Square wave; only applies to that waveform.");
    ui.horizontal(|ui| {
        ui.label("Unison:");
        for voices in 1..=3u8 {
            if ui
                .selectable_label(synth.unison_voices == voices, voices.to_string())
                .clicked()
            {
                synth.unison_voices = voices;
            }
        }
    });
    ui.add_enabled(
        synth.unison_voices > 1,
        egui::Slider::new(&mut synth.unison_detune_cents, 0.0..=50.0)
            .text("Detune")
            .suffix(" cents"),
    );
    ui.add_enabled(
        synth.unison_voices > 1,
        egui::Slider::new(&mut synth.unison_width, 0.0..=1.0).text("Width"),
    )
    .on_hover_text("Spreads unison voices across the stereo field. 0 keeps them centered.");
    oscillator_preview_ui(ui, synth);

    ui.separator();
    ui.strong("Oscillator 2");
    ui.horizontal(|ui| {
        ui.label("Waveform:");
        for (label, waveform) in [
            ("Sine", SynthWaveform::Sine),
            ("Saw", SynthWaveform::Saw),
            ("Square", SynthWaveform::Square),
            ("Triangle", SynthWaveform::Triangle),
        ] {
            if ui
                .selectable_label(synth.osc2_waveform == waveform, label)
                .clicked()
            {
                synth.osc2_waveform = waveform;
            }
        }
    });
    ui.add(
        egui::Slider::new(&mut synth.osc2_semitones, -24..=24)
            .text("Coarse tune")
            .suffix(" st"),
    );
    ui.add(
        egui::Slider::new(&mut synth.osc2_detune_cents, -50.0..=50.0)
            .text("Fine tune")
            .suffix(" cents"),
    );
    ui.add(egui::Slider::new(&mut synth.osc2_mix, 0.0..=1.0).text("Mix"));
    ui.weak("Mix crossfades between Oscillator 1 (0) and Oscillator 2 (1); 0 sounds exactly like before this existed.");
    ui.checkbox(&mut synth.osc2_sync, "Sync to Oscillator 1")
        .on_hover_text("Resets Oscillator 2's phase every time Oscillator 1 completes a cycle, locking it to Oscillator 1's pitch and truncating its waveform for a bright, buzzy timbre.");
    oscillator2_preview_ui(ui, synth);
    ui.add(egui::Slider::new(&mut synth.sub_osc_mix, 0.0..=1.0).text("Sub-osc level"));
    ui.weak("A fixed sine one octave below the note's pitch, mixed in on top (not crossfaded).");
}

/// Draws the ADSR envelope shape. Since Sustain has no fixed duration (it holds until note-off),
/// its plateau is drawn as a fixed-width visual segment rather than to scale; Attack, Decay and
/// Release segments are sized proportionally to their actual values relative to one another.
/// Generic over the raw ADSR values so `TrineParams`/`WaveParams`'s multiple envelopes (which
/// aren't wrapped in a `SynthParams`) can reuse it — see `envelope_preview_ui` for the
/// `SynthParams` convenience wrapper.
fn adsr_preview_ui(
    ui: &mut egui::Ui,
    attack: f32,
    decay: f32,
    sustain: f32,
    release: f32,
    color: egui::Color32,
) {
    let (response, painter) =
        ui.allocate_painter(egui::vec2(ui.available_width(), 60.0), egui::Sense::hover());
    let rect = response.rect;
    painter.rect_filled(rect, 2.0, ui.visuals().extreme_bg_color);

    let attack = attack.max(0.01);
    let decay = decay.max(0.01);
    let release = release.max(0.01);
    let sustain_hold = 0.4; // fixed visual width standing in for the undefined hold duration
    let total = attack + decay + sustain_hold + release;

    let pad = 4.0;
    let x0 = rect.left() + pad;
    let usable_w = rect.width() - 2.0 * pad;
    let x_attack = x0 + usable_w * (attack / total);
    let x_decay = x_attack + usable_w * (decay / total);
    let x_hold = x_decay + usable_w * (sustain_hold / total);
    let x_release = x_hold + usable_w * (release / total);
    let y_bottom = rect.bottom() - pad;
    let y_top = rect.top() + pad;
    let y_sustain = y_bottom - (y_bottom - y_top) * sustain.clamp(0.0, 1.0);

    let points = vec![
        egui::pos2(x0, y_bottom),
        egui::pos2(x_attack, y_top),
        egui::pos2(x_decay, y_sustain),
        egui::pos2(x_hold, y_sustain),
        egui::pos2(x_release, y_bottom),
    ];
    painter.add(egui::Shape::line(points, egui::Stroke::new(2.0, color)));
}

fn envelope_preview_ui(ui: &mut egui::Ui, synth: &SynthParams) {
    adsr_preview_ui(
        ui,
        synth.attack_seconds,
        synth.decay_seconds,
        synth.sustain_level,
        synth.release_seconds,
        egui::Color32::from_rgb(120, 220, 140),
    );
}

/// Frequency-response magnitude curve for the per-voice TPT state-variable filter, using the
/// standard analog 2-pole lowpass/highpass/bandpass/notch prototype that filter approximates
/// (see `audio::Voice::process`'s SVF for the actual real-time DSP). Purely illustrative — close
/// enough to communicate cutoff/resonance shape without re-simulating the exact discrete filter.
fn filter_response_db(
    filter_type: FilterType,
    freq_hz: f32,
    cutoff_hz: f32,
    resonance: f32,
) -> f32 {
    let x = freq_hz / cutoff_hz.max(1.0);
    let q = resonance.max(0.05);
    let denom = ((1.0 - x * x).powi(2) + (x / q).powi(2)).sqrt().max(1e-6);
    let magnitude = match filter_type {
        FilterType::Lowpass => 1.0 / denom,
        FilterType::Highpass => (x * x) / denom,
        FilterType::Bandpass => (x / q) / denom,
        FilterType::Notch => (1.0 - x * x).abs() / denom,
    };
    20.0 * magnitude.max(1e-6).log10()
}

/// Draws one filter's frequency response across 20Hz-20kHz (log-scaled x-axis) with a marker at
/// the current cutoff. Generic over the raw filter values — see `filter_preview_ui` for the
/// `SynthParams` wrapper and `dual_filter_preview_ui` for `TrineParams`/`WaveParams`'s two-filter
/// version.
fn filter_response_preview_ui(
    ui: &mut egui::Ui,
    filter_type: FilterType,
    cutoff_hz: f32,
    resonance: f32,
) {
    let (response, painter) =
        ui.allocate_painter(egui::vec2(ui.available_width(), 60.0), egui::Sense::hover());
    let rect = response.rect;
    painter.rect_filled(rect, 2.0, ui.visuals().extreme_bg_color);

    let min_db = -36.0;
    let max_db = 18.0;
    let log_min = 20.0f32.log10();
    let log_max = 20_000.0f32.log10();
    let db_to_y = |db: f32| {
        let t = ((db - min_db) / (max_db - min_db)).clamp(0.0, 1.0);
        rect.bottom() - t * rect.height()
    };
    // 0 dB reference line
    painter.line_segment(
        [
            egui::pos2(rect.left(), db_to_y(0.0)),
            egui::pos2(rect.right(), db_to_y(0.0)),
        ],
        egui::Stroke::new(0.5, ui.visuals().weak_text_color()),
    );

    let samples = 150;
    let points: Vec<egui::Pos2> = (0..=samples)
        .map(|i| {
            let t = i as f32 / samples as f32;
            let freq = 10f32.powf(log_min + t * (log_max - log_min));
            let db =
                filter_response_db(filter_type, freq, cutoff_hz, resonance).clamp(min_db, max_db);
            egui::pos2(rect.left() + t * rect.width(), db_to_y(db))
        })
        .collect();
    painter.add(egui::Shape::line(
        points,
        egui::Stroke::new(2.0, egui::Color32::from_rgb(230, 140, 140)),
    ));

    let cutoff_t = ((cutoff_hz.max(20.0).log10() - log_min) / (log_max - log_min)).clamp(0.0, 1.0);
    let cutoff_x = rect.left() + cutoff_t * rect.width();
    painter.line_segment(
        [
            egui::pos2(cutoff_x, rect.top()),
            egui::pos2(cutoff_x, rect.bottom()),
        ],
        egui::Stroke::new(1.0, egui::Color32::YELLOW),
    );
}

fn filter_preview_ui(ui: &mut egui::Ui, synth: &SynthParams) {
    filter_response_preview_ui(
        ui,
        synth.filter_type,
        synth.filter_cutoff_hz,
        synth.filter_resonance,
    );
}

/// Draws the combined frequency response of `TrineParams`/`WaveParams`'s two filters, respecting
/// `FilterRouting`: `Off` shows filter1 alone, `Series` sums the two responses in dB (equivalent
/// to multiplying their linear magnitudes), `Parallel` sums their linear magnitudes before
/// converting back to dB. A second marker line (in filter2's color) appears next to filter1's
/// whenever filter2 is actually in the signal path.
#[allow(clippy::too_many_arguments)]
fn dual_filter_preview_ui(
    ui: &mut egui::Ui,
    filter1_type: FilterType,
    cutoff1_hz: f32,
    resonance1: f32,
    filter2_type: FilterType,
    cutoff2_hz: f32,
    resonance2: f32,
    routing: FilterRouting,
) {
    let (response, painter) =
        ui.allocate_painter(egui::vec2(ui.available_width(), 60.0), egui::Sense::hover());
    let rect = response.rect;
    painter.rect_filled(rect, 2.0, ui.visuals().extreme_bg_color);

    let min_db = -36.0;
    let max_db = 18.0;
    let log_min = 20.0f32.log10();
    let log_max = 20_000.0f32.log10();
    let db_to_y = |db: f32| {
        let t = ((db - min_db) / (max_db - min_db)).clamp(0.0, 1.0);
        rect.bottom() - t * rect.height()
    };
    painter.line_segment(
        [
            egui::pos2(rect.left(), db_to_y(0.0)),
            egui::pos2(rect.right(), db_to_y(0.0)),
        ],
        egui::Stroke::new(0.5, ui.visuals().weak_text_color()),
    );

    let samples = 150;
    let points: Vec<egui::Pos2> = (0..=samples)
        .map(|i| {
            let t = i as f32 / samples as f32;
            let freq = 10f32.powf(log_min + t * (log_max - log_min));
            let db1 = filter_response_db(filter1_type, freq, cutoff1_hz, resonance1);
            let db = match routing {
                FilterRouting::Off => db1,
                FilterRouting::Series => {
                    db1 + filter_response_db(filter2_type, freq, cutoff2_hz, resonance2)
                }
                FilterRouting::Parallel => {
                    let db2 = filter_response_db(filter2_type, freq, cutoff2_hz, resonance2);
                    let linear_sum = 10f32.powf(db1 / 20.0) + 10f32.powf(db2 / 20.0);
                    20.0 * linear_sum.max(1e-6).log10()
                }
            };
            egui::pos2(
                rect.left() + t * rect.width(),
                db_to_y(db.clamp(min_db, max_db)),
            )
        })
        .collect();
    painter.add(egui::Shape::line(
        points,
        egui::Stroke::new(2.0, egui::Color32::from_rgb(230, 140, 140)),
    ));

    let cutoff_x = |cutoff_hz: f32| {
        let t = ((cutoff_hz.max(20.0).log10() - log_min) / (log_max - log_min)).clamp(0.0, 1.0);
        rect.left() + t * rect.width()
    };
    let x1 = cutoff_x(cutoff1_hz);
    painter.line_segment(
        [egui::pos2(x1, rect.top()), egui::pos2(x1, rect.bottom())],
        egui::Stroke::new(1.0, egui::Color32::YELLOW),
    );
    if routing != FilterRouting::Off {
        let x2 = cutoff_x(cutoff2_hz);
        painter.line_segment(
            [egui::pos2(x2, rect.top()), egui::pos2(x2, rect.bottom())],
            egui::Stroke::new(1.0, egui::Color32::from_rgb(120, 200, 255)),
        );
    }
}

fn synth_modulation_ui(ui: &mut egui::Ui, synth: &mut SynthParams) {
    ui.strong("Envelope");
    ui.add(
        egui::Slider::new(&mut synth.attack_seconds, 0.0..=0.5)
            .text("Attack")
            .suffix(" s"),
    );
    ui.add(
        egui::Slider::new(&mut synth.decay_seconds, 0.02..=2.0)
            .text("Decay")
            .suffix(" s"),
    );
    ui.add(egui::Slider::new(&mut synth.sustain_level, 0.0..=1.0).text("Sustain"));
    ui.add(
        egui::Slider::new(&mut synth.release_seconds, 0.02..=2.0)
            .text("Release")
            .suffix(" s"),
    );
    ui.weak(
        "Piano-roll notes hold Sustain for their drawn length, then Release. Step-grid hits have \
         no natural length, so they treat Attack + Decay as their held time, then Release.",
    );
    ui.add(
        egui::Slider::new(&mut synth.glide_seconds, 0.0..=1.0)
            .text("Glide")
            .suffix(" s"),
    );
    ui.weak("Portamento from the previously played pitch. Only applies to piano-roll notes, not step-grid hits.");
    envelope_preview_ui(ui, synth);

    ui.separator();
    ui.strong("Filter");
    ui.horizontal(|ui| {
        ui.label("Type:");
        for (label, filter_type) in [
            ("Lowpass", FilterType::Lowpass),
            ("Highpass", FilterType::Highpass),
            ("Bandpass", FilterType::Bandpass),
            ("Notch", FilterType::Notch),
        ] {
            if ui
                .selectable_label(synth.filter_type == filter_type, label)
                .clicked()
            {
                synth.filter_type = filter_type;
            }
        }
    });
    ui.add(
        egui::Slider::new(&mut synth.filter_cutoff_hz, 20.0..=20_000.0)
            .logarithmic(true)
            .text("Cutoff")
            .suffix(" Hz"),
    );
    ui.add(egui::Slider::new(&mut synth.filter_resonance, 0.3..=10.0).text("Resonance"));
    ui.add(
        egui::Slider::new(&mut synth.filter_env_amount_hz, -10_000.0..=10_000.0)
            .text("Env amount")
            .suffix(" Hz"),
    );
    ui.weak("Env amount sweeps the cutoff from note-on, decaying over the same time as the amplitude Decay above.");
    filter_preview_ui(ui, synth);

    ui.separator();
    ui.strong("LFO");
    ui.horizontal(|ui| {
        ui.label("Waveform:");
        for (label, waveform) in [
            ("Sine", SynthWaveform::Sine),
            ("Saw", SynthWaveform::Saw),
            ("Square", SynthWaveform::Square),
            ("Triangle", SynthWaveform::Triangle),
        ] {
            if ui
                .selectable_label(synth.lfo_waveform == waveform, label)
                .clicked()
            {
                synth.lfo_waveform = waveform;
            }
        }
    });
    ui.add(
        egui::Slider::new(&mut synth.lfo_rate_hz, 0.1..=20.0)
            .text("Rate")
            .suffix(" Hz"),
    );
    ui.horizontal(|ui| {
        ui.label("Target:");
        for (label, target) in [
            ("Off", LfoTarget::None),
            ("Pitch", LfoTarget::Pitch),
            ("Amplitude", LfoTarget::Amplitude),
            ("Filter cutoff", LfoTarget::FilterCutoff),
        ] {
            if ui
                .selectable_label(synth.lfo_target == target, label)
                .clicked()
            {
                synth.lfo_target = target;
            }
        }
    });
    ui.add_enabled(
        synth.lfo_target != LfoTarget::None,
        egui::Slider::new(&mut synth.lfo_depth, 0.0..=1.0).text("Depth"),
    );
    lfo_preview_ui(ui, synth);
}

/// Draws a few cycles of an LFO's waveform, scaled by `depth`; grayed out when `active` is false.
/// Generic over the raw values — see `lfo_preview_ui` for the `SynthParams` wrapper.
fn lfo_shape_preview_ui(ui: &mut egui::Ui, waveform: SynthWaveform, active: bool, depth: f32) {
    let (response, painter) =
        ui.allocate_painter(egui::vec2(ui.available_width(), 60.0), egui::Sense::hover());
    let rect = response.rect;
    painter.rect_filled(rect, 2.0, ui.visuals().extreme_bg_color);
    painter.line_segment(
        [
            egui::pos2(rect.left(), rect.center().y),
            egui::pos2(rect.right(), rect.center().y),
        ],
        egui::Stroke::new(0.5, ui.visuals().weak_text_color()),
    );

    let color = if active {
        egui::Color32::from_rgb(200, 140, 230)
    } else {
        ui.visuals().weak_text_color()
    };
    let amplitude = if active { depth.max(0.05) } else { 0.6 };
    paint_waveform_shape(
        &painter,
        rect,
        waveform,
        0.5,
        amplitude,
        4.0,
        egui::Stroke::new(2.0, color),
    );
}

fn lfo_preview_ui(ui: &mut egui::Ui, synth: &SynthParams) {
    lfo_shape_preview_ui(
        ui,
        synth.lfo_waveform,
        synth.lfo_target != LfoTarget::None,
        synth.lfo_depth,
    );
}

/// A row of selectable labels for every `SynthWaveform` variant, mirroring the picker rows in
/// `synth_oscillators_ui` — shared here since Trine has five of these (three oscillators, two LFOs).
fn waveform_picker_ui(ui: &mut egui::Ui, current: &mut SynthWaveform) {
    ui.horizontal(|ui| {
        ui.label("Waveform:");
        for (label, waveform) in [
            ("Sine", SynthWaveform::Sine),
            ("Saw", SynthWaveform::Saw),
            ("Square", SynthWaveform::Square),
            ("Triangle", SynthWaveform::Triangle),
            ("Noise", SynthWaveform::Noise),
        ] {
            if ui.selectable_label(*current == waveform, label).clicked() {
                *current = waveform;
            }
        }
    });
}

/// Renders the Trine engine's settings, shown inside the bottom Device Panel when
/// `Track::synth_engine == SynthEngine::Trine` (see `device_panel_contents_ui`). Laid out as
/// three columns (oscillators | filter + modulation matrix | LFOs + envelopes) so every section
/// is visible at once instead of stacked behind collapsing headers, mirroring `synth_params_ui`'s
/// two-column layout but wider since Trine has considerably more surface.
fn trine_params_ui(ui: &mut egui::Ui, trine: &mut TrineParams) {
    ui.weak(
        "A second, independent synth engine: 3 oscillators, a dual filter, and a free \
         modulation matrix. Envelope 3 always drives amplitude; everything else only does \
         something once routed in the Modulation Matrix section below.",
    );
    ui.separator();
    ui.columns(3, |columns| {
        trine_oscillators_ui(&mut columns[0], trine);
        trine_filter_matrix_ui(&mut columns[1], trine);
        trine_lfos_envelopes_ui(&mut columns[2], trine);
    });
}

/// Small canvas overlaying all three of Trine's oscillators: Oscillator 1 in blue (an undetuned,
/// 3-cycle reference), Oscillator 2 in orange and Oscillator 3 in green, both pitch- and
/// sync-accurate against Oscillator 1 using the same math as `oscillator2_preview_ui`, each
/// scaled by its own Level. FM and ring mod aren't reflected — they're audio-rate interactions a
/// static per-oscillator shape can't show — only mix level and pitch/sync relationships are.
fn trine_oscillators_preview_ui(ui: &mut egui::Ui, trine: &TrineParams) {
    let (response, painter) =
        ui.allocate_painter(egui::vec2(ui.available_width(), 70.0), egui::Sense::hover());
    let rect = response.rect;
    painter.rect_filled(rect, 2.0, ui.visuals().extreme_bg_color);
    painter.line_segment(
        [
            egui::pos2(rect.left(), rect.center().y),
            egui::pos2(rect.right(), rect.center().y),
        ],
        egui::Stroke::new(0.5, ui.visuals().weak_text_color()),
    );

    let cycles = 3.0;
    let samples = 300;
    if trine.osc1_level > 0.0 {
        let points = synced_oscillator_points(
            rect,
            trine.osc1_waveform,
            trine.pulse_width,
            1.0,
            false,
            trine.osc1_level.max(0.15),
            cycles,
            samples,
        );
        painter.add(egui::Shape::line(
            points,
            egui::Stroke::new(2.0, egui::Color32::from_rgb(120, 170, 230)),
        ));
    }
    if trine.osc2_level > 0.0 {
        let ratio = 2f32.powf(trine.osc2_semitones as f32 / 12.0)
            * 2f32.powf(trine.osc2_detune_cents / 1200.0);
        let points = synced_oscillator_points(
            rect,
            trine.osc2_waveform,
            trine.pulse_width,
            ratio,
            trine.osc2_sync,
            trine.osc2_level,
            cycles,
            samples,
        );
        let color =
            egui::Color32::from_rgb(230, 160, 90).gamma_multiply(trine.osc2_level.max(0.25));
        painter.add(egui::Shape::line(points, egui::Stroke::new(1.5, color)));
    }
    if trine.osc3_level > 0.0 {
        let ratio = 2f32.powf(trine.osc3_semitones as f32 / 12.0)
            * 2f32.powf(trine.osc3_detune_cents / 1200.0);
        let points = synced_oscillator_points(
            rect,
            trine.osc3_waveform,
            trine.pulse_width,
            ratio,
            trine.osc3_sync,
            trine.osc3_level,
            cycles,
            samples,
        );
        let color =
            egui::Color32::from_rgb(140, 220, 170).gamma_multiply(trine.osc3_level.max(0.25));
        painter.add(egui::Shape::line(points, egui::Stroke::new(1.5, color)));
    }
}

/// Small canvas dedicated to one of Trine's Oscillator 2/3: a faint reference line for
/// Oscillator 1 and this oscillator's own shape overlaid in its accent color, pitch- and
/// sync-accurate against Oscillator 1 via `synced_oscillator_points` — the same math
/// `oscillator2_preview_ui` uses for the base synth, and the same per-oscillator math
/// `trine_oscillators_preview_ui` overlays for all three at once.
fn trine_oscillator_n_preview_ui(
    ui: &mut egui::Ui,
    trine: &TrineParams,
    waveform: SynthWaveform,
    semitones: i32,
    detune_cents: f32,
    sync: bool,
    color: egui::Color32,
) {
    let (response, painter) =
        ui.allocate_painter(egui::vec2(ui.available_width(), 60.0), egui::Sense::hover());
    let rect = response.rect;
    painter.rect_filled(rect, 2.0, ui.visuals().extreme_bg_color);
    painter.line_segment(
        [
            egui::pos2(rect.left(), rect.center().y),
            egui::pos2(rect.right(), rect.center().y),
        ],
        egui::Stroke::new(0.5, ui.visuals().weak_text_color()),
    );

    let cycles = 3.0;
    let samples = 300;
    let osc1_points = synced_oscillator_points(
        rect,
        trine.osc1_waveform,
        trine.pulse_width,
        1.0,
        false,
        0.6,
        cycles,
        samples,
    );
    painter.add(egui::Shape::line(
        osc1_points,
        egui::Stroke::new(1.0, ui.visuals().weak_text_color()),
    ));

    let ratio = 2f32.powf(semitones as f32 / 12.0) * 2f32.powf(detune_cents / 1200.0);
    let osc_points = synced_oscillator_points(
        rect,
        waveform,
        trine.pulse_width,
        ratio,
        sync,
        1.0,
        cycles,
        samples,
    );
    painter.add(egui::Shape::line(osc_points, egui::Stroke::new(2.0, color)));
}

fn trine_oscillators_ui(ui: &mut egui::Ui, trine: &mut TrineParams) {
    ui.strong("Oscillators");
    trine_oscillators_preview_ui(ui, trine);
    ui.separator();
    ui.strong("Oscillator 1");
    waveform_picker_ui(ui, &mut trine.osc1_waveform);
    ui.add(egui::Slider::new(&mut trine.osc1_level, 0.0..=1.0).text("Level"));

    ui.separator();
    ui.strong("Oscillator 2");
    waveform_picker_ui(ui, &mut trine.osc2_waveform);
    ui.add(
        egui::Slider::new(&mut trine.osc2_semitones, -24..=24)
            .text("Coarse tune")
            .suffix(" st"),
    );
    ui.add(
        egui::Slider::new(&mut trine.osc2_detune_cents, -50.0..=50.0)
            .text("Fine tune")
            .suffix(" cents"),
    );
    ui.add(egui::Slider::new(&mut trine.osc2_level, 0.0..=1.0).text("Level"));
    ui.checkbox(&mut trine.osc2_sync, "Sync to Oscillator 1");
    trine_oscillator_n_preview_ui(
        ui,
        trine,
        trine.osc2_waveform,
        trine.osc2_semitones,
        trine.osc2_detune_cents,
        trine.osc2_sync,
        egui::Color32::from_rgb(230, 160, 90),
    );

    ui.separator();
    ui.strong("Oscillator 3");
    waveform_picker_ui(ui, &mut trine.osc3_waveform);
    ui.add(
        egui::Slider::new(&mut trine.osc3_semitones, -24..=24)
            .text("Coarse tune")
            .suffix(" st"),
    );
    ui.add(
        egui::Slider::new(&mut trine.osc3_detune_cents, -50.0..=50.0)
            .text("Fine tune")
            .suffix(" cents"),
    );
    ui.add(egui::Slider::new(&mut trine.osc3_level, 0.0..=1.0).text("Level"));
    ui.checkbox(&mut trine.osc3_sync, "Sync to Oscillator 1");
    trine_oscillator_n_preview_ui(
        ui,
        trine,
        trine.osc3_waveform,
        trine.osc3_semitones,
        trine.osc3_detune_cents,
        trine.osc3_sync,
        egui::Color32::from_rgb(140, 220, 170),
    );

    ui.separator();
    ui.add(
        egui::Slider::new(&mut trine.pulse_width, 0.05..=0.95)
            .text("Pulse width")
            .suffix(" (Square oscillators)"),
    );
    ui.add(egui::Slider::new(&mut trine.fm_amount, 0.0..=4.0).text("FM amount"))
        .on_hover_text("Oscillator 2 frequency-modulates Oscillator 1, independent of Oscillator 2's own level.");
    ui.add(egui::Slider::new(&mut trine.ring_mod_mix, 0.0..=1.0).text("Ring mod mix"))
        .on_hover_text(
            "Oscillator 1 x Oscillator 2, mixed in on top of the regular oscillator sum.",
        );
    ui.add(egui::Slider::new(&mut trine.analog_drift, 0.0..=1.0).text("Analog drift"))
        .on_hover_text(
            "Slow per-voice random pitch wander, emulating analog oscillator instability.",
        );
}

/// Combines `trine_filter_ui` and `trine_matrix_ui` into Trine's middle column.
fn trine_filter_matrix_ui(ui: &mut egui::Ui, trine: &mut TrineParams) {
    ui.strong("Filter");
    trine_filter_ui(ui, trine);
    ui.separator();
    ui.strong("Modulation Matrix");
    trine_matrix_ui(ui, trine);
}

fn trine_filter_ui(ui: &mut egui::Ui, trine: &mut TrineParams) {
    ui.horizontal(|ui| {
        ui.label("Routing:");
        for (label, routing) in [
            ("Off", FilterRouting::Off),
            ("Series", FilterRouting::Series),
            ("Parallel", FilterRouting::Parallel),
        ] {
            if ui
                .selectable_label(trine.filter_routing == routing, label)
                .clicked()
            {
                trine.filter_routing = routing;
            }
        }
    });
    ui.weak("Off uses Filter 1 alone; Series feeds Filter 1 into Filter 2; Parallel sums both filters' output.");

    ui.strong("Filter 1");
    filter_stage_ui(
        ui,
        &mut trine.filter1_cutoff_hz,
        &mut trine.filter1_resonance,
        &mut trine.filter1_type,
        &mut trine.filter1_slope,
    );

    ui.add_enabled_ui(trine.filter_routing != FilterRouting::Off, |ui| {
        ui.separator();
        ui.strong("Filter 2");
        filter_stage_ui(
            ui,
            &mut trine.filter2_cutoff_hz,
            &mut trine.filter2_resonance,
            &mut trine.filter2_type,
            &mut trine.filter2_slope,
        );
    });

    ui.separator();
    ui.add(egui::Slider::new(&mut trine.filter_drive, 0.0..=1.0).text("Drive"))
        .on_hover_text("Soft-clip saturation applied before Filter 1.");
    ui.add(egui::Slider::new(&mut trine.filter_fm_amount, 0.0..=1.0).text("Filter FM"))
        .on_hover_text("Filter 1 cutoff modulated directly by Oscillator 2's instantaneous output (audio-rate).");

    dual_filter_preview_ui(
        ui,
        trine.filter1_type,
        trine.filter1_cutoff_hz,
        trine.filter1_resonance,
        trine.filter2_type,
        trine.filter2_cutoff_hz,
        trine.filter2_resonance,
        trine.filter_routing,
    );
}

fn filter_stage_ui(
    ui: &mut egui::Ui,
    cutoff_hz: &mut f32,
    resonance: &mut f32,
    filter_type: &mut FilterType,
    slope: &mut FilterSlope,
) {
    ui.horizontal(|ui| {
        ui.label("Type:");
        for (label, ft) in [
            ("Lowpass", FilterType::Lowpass),
            ("Highpass", FilterType::Highpass),
            ("Bandpass", FilterType::Bandpass),
            ("Notch", FilterType::Notch),
        ] {
            if ui.selectable_label(*filter_type == ft, label).clicked() {
                *filter_type = ft;
            }
        }
    });
    ui.horizontal(|ui| {
        ui.label("Slope:");
        for (label, s) in [
            ("12 dB/oct", FilterSlope::Slope12),
            ("24 dB/oct", FilterSlope::Slope24),
        ] {
            if ui.selectable_label(*slope == s, label).clicked() {
                *slope = s;
            }
        }
    });
    ui.add(
        egui::Slider::new(cutoff_hz, 20.0..=20_000.0)
            .logarithmic(true)
            .text("Cutoff")
            .suffix(" Hz"),
    );
    ui.add(egui::Slider::new(resonance, 0.3..=10.0).text("Resonance"));
}

fn trine_matrix_ui(ui: &mut egui::Ui, trine: &mut TrineParams) {
    ui.weak("Route a modulation source to a target with a bipolar amount. Empty by default.");
    let mut to_remove = None;
    for (index, slot) in trine.mod_slots.iter_mut().enumerate() {
        ui.horizontal(|ui| {
            egui::ComboBox::from_id_salt(("trine-mod-source", index))
                .selected_text(mod_source_label(slot.source))
                .show_ui(ui, |ui| {
                    for source in [
                        ModSource::None,
                        ModSource::Lfo1,
                        ModSource::Lfo2,
                        ModSource::Env1,
                        ModSource::Env2,
                        ModSource::Velocity,
                    ] {
                        ui.selectable_value(&mut slot.source, source, mod_source_label(source));
                    }
                });
            ui.label("->");
            egui::ComboBox::from_id_salt(("trine-mod-target", index))
                .selected_text(mod_target_label(slot.target))
                .show_ui(ui, |ui| {
                    for target in [
                        ModTarget::None,
                        ModTarget::Pitch,
                        ModTarget::Osc1Level,
                        ModTarget::Osc2Level,
                        ModTarget::Osc3Level,
                        ModTarget::PulseWidth,
                        ModTarget::FilterCutoff,
                        ModTarget::Filter2Cutoff,
                        ModTarget::FilterResonance,
                        ModTarget::FmAmount,
                        ModTarget::RingModMix,
                    ] {
                        ui.selectable_value(&mut slot.target, target, mod_target_label(target));
                    }
                });
            ui.add(egui::Slider::new(&mut slot.amount, -1.0..=1.0).text("Amount"));
            if ui.button("🗑").on_hover_text("Remove this mod slot").clicked() {
                to_remove = Some(index);
            }
        });
    }
    if let Some(index) = to_remove {
        trine.mod_slots.remove(index);
    }
    if ui.button("+ Add slot").clicked() {
        trine.mod_slots.push(ModSlot::default());
    }
}

fn mod_source_label(source: ModSource) -> &'static str {
    match source {
        ModSource::None => "— none —",
        ModSource::Lfo1 => "LFO 1",
        ModSource::Lfo2 => "LFO 2",
        ModSource::Env1 => "Envelope 1",
        ModSource::Env2 => "Envelope 2",
        ModSource::Velocity => "Velocity",
    }
}

fn mod_target_label(target: ModTarget) -> &'static str {
    match target {
        ModTarget::None => "— none —",
        ModTarget::Pitch => "Pitch",
        ModTarget::Osc1Level => "Osc 1 Level",
        ModTarget::Osc2Level => "Osc 2 Level",
        ModTarget::Osc3Level => "Osc 3 Level",
        ModTarget::PulseWidth => "Pulse Width",
        ModTarget::FilterCutoff => "Filter 1 Cutoff",
        ModTarget::Filter2Cutoff => "Filter 2 Cutoff",
        ModTarget::FilterResonance => "Filter 1 Resonance",
        ModTarget::FmAmount => "FM Amount",
        ModTarget::RingModMix => "Ring Mod Mix",
    }
}

/// Whether `source` is actually wired to something in `mod_slots` (routed to a non-`None` target
/// with a non-zero amount) and, if so, the largest magnitude it's routed at — used to decide
/// whether an LFO's preview canvas should render as "live" or grayed-out, since Trine/Wave's LFOs
/// have no target/depth of their own the way `SynthParams`'s single LFO does.
fn trine_lfo_active_depth(mod_slots: &[ModSlot], source: ModSource) -> (bool, f32) {
    let depth = mod_slots
        .iter()
        .filter(|slot| slot.source == source && slot.target != ModTarget::None)
        .map(|slot| slot.amount.abs())
        .fold(0.0f32, f32::max);
    (depth > 0.001, depth)
}

/// Combines Trine's LFOs and envelopes into the third column.
fn trine_lfos_envelopes_ui(ui: &mut egui::Ui, trine: &mut TrineParams) {
    ui.strong("LFOs");
    trine_lfos_ui(ui, trine);
    ui.separator();
    ui.strong("Envelopes");
    trine_envelopes_ui(ui, trine);
}

fn trine_lfos_ui(ui: &mut egui::Ui, trine: &mut TrineParams) {
    ui.strong("LFO 1");
    waveform_picker_ui(ui, &mut trine.lfo1_waveform);
    ui.add(
        egui::Slider::new(&mut trine.lfo1_rate_hz, 0.1..=20.0)
            .text("Rate")
            .suffix(" Hz"),
    );
    let (active1, depth1) = trine_lfo_active_depth(&trine.mod_slots, ModSource::Lfo1);
    lfo_shape_preview_ui(ui, trine.lfo1_waveform, active1, depth1);

    ui.separator();
    ui.strong("LFO 2");
    waveform_picker_ui(ui, &mut trine.lfo2_waveform);
    ui.add(
        egui::Slider::new(&mut trine.lfo2_rate_hz, 0.1..=20.0)
            .text("Rate")
            .suffix(" Hz"),
    );
    let (active2, depth2) = trine_lfo_active_depth(&trine.mod_slots, ModSource::Lfo2);
    lfo_shape_preview_ui(ui, trine.lfo2_waveform, active2, depth2);
}

fn trine_envelopes_ui(ui: &mut egui::Ui, trine: &mut TrineParams) {
    ui.strong("Envelope 1")
        .on_hover_text("Free — only audible once routed in the Modulation Matrix.");
    ui.add(
        egui::Slider::new(&mut trine.env1_attack_seconds, 0.0..=2.0)
            .text("Attack")
            .suffix(" s"),
    );
    ui.add(
        egui::Slider::new(&mut trine.env1_decay_seconds, 0.02..=2.0)
            .text("Decay")
            .suffix(" s"),
    );
    ui.add(egui::Slider::new(&mut trine.env1_sustain_level, 0.0..=1.0).text("Sustain"));
    ui.add(
        egui::Slider::new(&mut trine.env1_release_seconds, 0.02..=2.0)
            .text("Release")
            .suffix(" s"),
    );
    adsr_preview_ui(
        ui,
        trine.env1_attack_seconds,
        trine.env1_decay_seconds,
        trine.env1_sustain_level,
        trine.env1_release_seconds,
        egui::Color32::from_rgb(200, 140, 230),
    );

    ui.separator();
    ui.strong("Envelope 2")
        .on_hover_text("Free — only audible once routed in the Modulation Matrix.");
    ui.add(
        egui::Slider::new(&mut trine.env2_attack_seconds, 0.0..=2.0)
            .text("Attack")
            .suffix(" s"),
    );
    ui.add(
        egui::Slider::new(&mut trine.env2_decay_seconds, 0.02..=2.0)
            .text("Decay")
            .suffix(" s"),
    );
    ui.add(egui::Slider::new(&mut trine.env2_sustain_level, 0.0..=1.0).text("Sustain"));
    ui.add(
        egui::Slider::new(&mut trine.env2_release_seconds, 0.02..=2.0)
            .text("Release")
            .suffix(" s"),
    );
    adsr_preview_ui(
        ui,
        trine.env2_attack_seconds,
        trine.env2_decay_seconds,
        trine.env2_sustain_level,
        trine.env2_release_seconds,
        egui::Color32::from_rgb(200, 140, 230),
    );

    ui.separator();
    ui.strong("Envelope 3 (Volume)")
        .on_hover_text("Always active — directly drives amplitude.");
    ui.add(
        egui::Slider::new(&mut trine.env3_attack_seconds, 0.0..=2.0)
            .text("Attack")
            .suffix(" s"),
    );
    ui.add(
        egui::Slider::new(&mut trine.env3_decay_seconds, 0.02..=2.0)
            .text("Decay")
            .suffix(" s"),
    );
    ui.add(egui::Slider::new(&mut trine.env3_sustain_level, 0.0..=1.0).text("Sustain"));
    ui.add(
        egui::Slider::new(&mut trine.env3_release_seconds, 0.02..=2.0)
            .text("Release")
            .suffix(" s"),
    );
    adsr_preview_ui(
        ui,
        trine.env3_attack_seconds,
        trine.env3_decay_seconds,
        trine.env3_sustain_level,
        trine.env3_release_seconds,
        egui::Color32::from_rgb(120, 220, 140),
    );
}

/// A row of selectable labels for every `WavetableId` variant — see `waveform_picker_ui`, the
/// equivalent for classic waveforms.
fn wavetable_picker_ui(ui: &mut egui::Ui, current: &mut WavetableId) {
    ui.horizontal(|ui| {
        ui.label("Table:");
        for id in WavetableId::ALL {
            if ui.selectable_label(*current == id, id.label()).clicked() {
                *current = id;
            }
        }
    });
}

/// A row of selectable labels for every `WaveWarpMode` variant, plus an amount slider (disabled
/// while `Off`, matching `filter_stage_ui`'s "Off" no-op precedent elsewhere in this file).
fn warp_mode_picker_ui(ui: &mut egui::Ui, mode: &mut WaveWarpMode, amount: &mut f32) {
    ui.horizontal(|ui| {
        ui.label("Warp:");
        for (label, m) in [
            ("Off", WaveWarpMode::Off),
            ("Bend", WaveWarpMode::Bend),
            ("Sync", WaveWarpMode::Sync),
            ("Mirror", WaveWarpMode::Mirror),
            ("FM", WaveWarpMode::Fm),
        ] {
            if ui.selectable_label(*mode == m, label).clicked() {
                *mode = m;
            }
        }
    });
    ui.add_enabled(
        *mode != WaveWarpMode::Off,
        egui::Slider::new(amount, 0.0..=1.0).text("Warp amount"),
    );
}

/// Renders the Wave engine's settings, shown inside a track's synth window when
/// `Track::synth_engine == SynthEngine::Wave`. Laid out as three columns (oscillators | filter +
/// modulation matrix | LFOs + envelopes), the same structure as `trine_params_ui`.
fn wave_params_ui(ui: &mut egui::Ui, wave: &mut WaveParams) {
    ui.weak(
        "A third synth engine loosely inspired by wavetable synths like Serum: 2 wavetable \
         oscillators (each scanning its table's frames, with an optional phase-warp), a sub \
         oscillator, noise, a dual filter, and a free modulation matrix. The amplitude envelope \
         always drives volume; everything else only does something once routed in the \
         Modulation Matrix section below.",
    );
    ui.separator();
    ui.columns(3, |columns| {
        wave_oscillators_ui(&mut columns[0], wave);
        wave_filter_matrix_ui(&mut columns[1], wave);
        wave_lfos_envelopes_ui(&mut columns[2], wave);
    });
}

/// Samples one full cycle (`phase` 0..1) of a `WaveParams` oscillator directly from its actual
/// wavetable data — `wavetable::sample` at mip level 0 (the highest-fidelity mip; previews aren't
/// played back at pitch, so aliasing doesn't apply), through `wavetable::warp_phase` first so
/// Bend/Sync/Mirror/FM warp modes show up exactly as they'd sound, not an approximation.
fn wave_oscillator_points(
    rect: egui::Rect,
    table: WavetableId,
    position: f32,
    warp_mode: WaveWarpMode,
    warp_amount: f32,
    amplitude: f32,
    samples: usize,
) -> Vec<egui::Pos2> {
    let mid_y = rect.center().y;
    let half_h = rect.height() * 0.5 * 0.9;
    (0..=samples)
        .map(|i| {
            let phase = i as f32 / samples as f32;
            let warped = wavetable::warp_phase(phase, warp_mode, warp_amount);
            let sample = wavetable::sample(table, position, warped, 0) * amplitude;
            egui::pos2(rect.left() + phase * rect.width(), mid_y - sample * half_h)
        })
        .collect()
}

/// Small canvas overlaying Wave's two oscillators, each sampled straight from its actual
/// wavetable (see `wave_oscillator_points`): Oscillator 1 in blue, Oscillator 2 faded in
/// proportion to its Level in orange. Since a wavetable oscillator is periodic in phase 0..1
/// regardless of pitch, both are drawn as one cycle rather than pitch/sync-accurate like
/// `trine_oscillators_preview_ui` — this shows table/position/warp timbre, not tuning.
fn wave_oscillators_preview_ui(ui: &mut egui::Ui, wave: &WaveParams) {
    let (response, painter) =
        ui.allocate_painter(egui::vec2(ui.available_width(), 70.0), egui::Sense::hover());
    let rect = response.rect;
    painter.rect_filled(rect, 2.0, ui.visuals().extreme_bg_color);
    painter.line_segment(
        [
            egui::pos2(rect.left(), rect.center().y),
            egui::pos2(rect.right(), rect.center().y),
        ],
        egui::Stroke::new(0.5, ui.visuals().weak_text_color()),
    );

    let samples = 300;
    if wave.osc1_level > 0.0 {
        let points = wave_oscillator_points(
            rect,
            wave.osc1_table,
            wave.osc1_position,
            wave.osc1_warp_mode,
            wave.osc1_warp_amount,
            wave.osc1_level.max(0.15),
            samples,
        );
        painter.add(egui::Shape::line(
            points,
            egui::Stroke::new(2.0, egui::Color32::from_rgb(120, 170, 230)),
        ));
    }
    if wave.osc2_level > 0.0 {
        let points = wave_oscillator_points(
            rect,
            wave.osc2_table,
            wave.osc2_position,
            wave.osc2_warp_mode,
            wave.osc2_warp_amount,
            wave.osc2_level,
            samples,
        );
        let color = egui::Color32::from_rgb(230, 160, 90).gamma_multiply(wave.osc2_level.max(0.25));
        painter.add(egui::Shape::line(points, egui::Stroke::new(1.5, color)));
    }
}

/// Small canvas dedicated to Wave's Oscillator 2: a faint reference cycle for Oscillator 1 and
/// Oscillator 2's own shape overlaid in orange, both sampled directly from their wavetable data
/// via `wave_oscillator_points` — the single-oscillator counterpart to
/// `wave_oscillators_preview_ui`'s combined overlay.
fn wave_oscillator2_preview_ui(ui: &mut egui::Ui, wave: &WaveParams) {
    let (response, painter) =
        ui.allocate_painter(egui::vec2(ui.available_width(), 60.0), egui::Sense::hover());
    let rect = response.rect;
    painter.rect_filled(rect, 2.0, ui.visuals().extreme_bg_color);
    painter.line_segment(
        [
            egui::pos2(rect.left(), rect.center().y),
            egui::pos2(rect.right(), rect.center().y),
        ],
        egui::Stroke::new(0.5, ui.visuals().weak_text_color()),
    );

    let samples = 300;
    let osc1_points = wave_oscillator_points(
        rect,
        wave.osc1_table,
        wave.osc1_position,
        wave.osc1_warp_mode,
        wave.osc1_warp_amount,
        0.6,
        samples,
    );
    painter.add(egui::Shape::line(
        osc1_points,
        egui::Stroke::new(1.0, ui.visuals().weak_text_color()),
    ));

    let osc2_points = wave_oscillator_points(
        rect,
        wave.osc2_table,
        wave.osc2_position,
        wave.osc2_warp_mode,
        wave.osc2_warp_amount,
        1.0,
        samples,
    );
    painter.add(egui::Shape::line(
        osc2_points,
        egui::Stroke::new(2.0, egui::Color32::from_rgb(230, 160, 90)),
    ));
}

fn wave_oscillators_ui(ui: &mut egui::Ui, wave: &mut WaveParams) {
    ui.strong("Oscillators");
    wave_oscillators_preview_ui(ui, wave);
    ui.separator();
    ui.strong("Oscillator 1");
    wavetable_picker_ui(ui, &mut wave.osc1_table);
    ui.add(egui::Slider::new(&mut wave.osc1_position, 0.0..=1.0).text("Position"));
    warp_mode_picker_ui(ui, &mut wave.osc1_warp_mode, &mut wave.osc1_warp_amount);
    ui.add(egui::Slider::new(&mut wave.osc1_level, 0.0..=1.0).text("Level"));
    ui.horizontal(|ui| {
        ui.label("Unison:");
        for voices in 1..=3u8 {
            if ui
                .selectable_label(wave.unison_voices == voices, voices.to_string())
                .clicked()
            {
                wave.unison_voices = voices;
            }
        }
    });
    ui.add_enabled(
        wave.unison_voices > 1,
        egui::Slider::new(&mut wave.unison_detune_cents, 0.0..=50.0)
            .text("Unison detune")
            .suffix(" cents"),
    );
    ui.add_enabled(
        wave.unison_voices > 1,
        egui::Slider::new(&mut wave.unison_width, 0.0..=1.0).text("Unison width"),
    )
    .on_hover_text("Spreads unison voices across the stereo field. 0 keeps them centered.");

    ui.separator();
    ui.strong("Oscillator 2");
    wavetable_picker_ui(ui, &mut wave.osc2_table);
    ui.add(egui::Slider::new(&mut wave.osc2_position, 0.0..=1.0).text("Position"));
    warp_mode_picker_ui(ui, &mut wave.osc2_warp_mode, &mut wave.osc2_warp_amount);
    ui.add(
        egui::Slider::new(&mut wave.osc2_semitones, -24..=24)
            .text("Coarse tune")
            .suffix(" st"),
    );
    ui.add(
        egui::Slider::new(&mut wave.osc2_detune_cents, -50.0..=50.0)
            .text("Fine tune")
            .suffix(" cents"),
    );
    ui.add(egui::Slider::new(&mut wave.osc2_level, 0.0..=1.0).text("Level"));
    wave_oscillator2_preview_ui(ui, wave);

    ui.separator();
    ui.strong("Sub / Noise");
    ui.add(
        egui::Slider::new(&mut wave.sub_osc_semitones, -24..=0)
            .text("Sub tune")
            .suffix(" st"),
    );
    ui.add(egui::Slider::new(&mut wave.sub_osc_level, 0.0..=1.0).text("Sub level"));
    ui.add(egui::Slider::new(&mut wave.noise_level, 0.0..=1.0).text("Noise level"));
}

/// Combines `wave_filter_ui` and `wave_matrix_ui` into Wave's middle column.
fn wave_filter_matrix_ui(ui: &mut egui::Ui, wave: &mut WaveParams) {
    ui.strong("Filter");
    wave_filter_ui(ui, wave);
    ui.separator();
    ui.strong("Modulation Matrix");
    wave_matrix_ui(ui, wave);
}

fn wave_filter_ui(ui: &mut egui::Ui, wave: &mut WaveParams) {
    ui.horizontal(|ui| {
        ui.label("Routing:");
        for (label, routing) in [
            ("Off", FilterRouting::Off),
            ("Series", FilterRouting::Series),
            ("Parallel", FilterRouting::Parallel),
        ] {
            if ui
                .selectable_label(wave.filter_routing == routing, label)
                .clicked()
            {
                wave.filter_routing = routing;
            }
        }
    });
    ui.weak("Off uses Filter 1 alone; Series feeds Filter 1 into Filter 2; Parallel sums both filters' output.");

    ui.strong("Filter 1");
    filter_stage_ui(
        ui,
        &mut wave.filter1_cutoff_hz,
        &mut wave.filter1_resonance,
        &mut wave.filter1_type,
        &mut wave.filter1_slope,
    );

    ui.add_enabled_ui(wave.filter_routing != FilterRouting::Off, |ui| {
        ui.separator();
        ui.strong("Filter 2");
        filter_stage_ui(
            ui,
            &mut wave.filter2_cutoff_hz,
            &mut wave.filter2_resonance,
            &mut wave.filter2_type,
            &mut wave.filter2_slope,
        );
    });

    ui.separator();
    ui.add(egui::Slider::new(&mut wave.filter_drive, 0.0..=1.0).text("Drive"))
        .on_hover_text("Soft-clip saturation applied before Filter 1.");

    dual_filter_preview_ui(
        ui,
        wave.filter1_type,
        wave.filter1_cutoff_hz,
        wave.filter1_resonance,
        wave.filter2_type,
        wave.filter2_cutoff_hz,
        wave.filter2_resonance,
        wave.filter_routing,
    );
}

fn wave_matrix_ui(ui: &mut egui::Ui, wave: &mut WaveParams) {
    ui.weak("Route a modulation source to a target with a bipolar amount. Empty by default.");
    let mut to_remove = None;
    for (index, slot) in wave.mod_slots.iter_mut().enumerate() {
        ui.horizontal(|ui| {
            egui::ComboBox::from_id_salt(("wave-mod-source", index))
                .selected_text(wave_mod_source_label(slot.source))
                .show_ui(ui, |ui| {
                    for source in [
                        WaveModSource::None,
                        WaveModSource::Lfo1,
                        WaveModSource::Lfo2,
                        WaveModSource::Env1,
                        WaveModSource::Env2,
                        WaveModSource::Velocity,
                    ] {
                        ui.selectable_value(
                            &mut slot.source,
                            source,
                            wave_mod_source_label(source),
                        );
                    }
                });
            ui.label("->");
            egui::ComboBox::from_id_salt(("wave-mod-target", index))
                .selected_text(wave_mod_target_label(slot.target))
                .show_ui(ui, |ui| {
                    for target in [
                        WaveModTarget::None,
                        WaveModTarget::Pitch,
                        WaveModTarget::Osc1Position,
                        WaveModTarget::Osc2Position,
                        WaveModTarget::Osc1WarpAmount,
                        WaveModTarget::Osc2WarpAmount,
                        WaveModTarget::FilterCutoff,
                        WaveModTarget::Filter2Cutoff,
                        WaveModTarget::FilterResonance,
                    ] {
                        ui.selectable_value(
                            &mut slot.target,
                            target,
                            wave_mod_target_label(target),
                        );
                    }
                });
            ui.add(egui::Slider::new(&mut slot.amount, -1.0..=1.0).text("Amount"));
            if ui.button("🗑").on_hover_text("Remove this mod slot").clicked() {
                to_remove = Some(index);
            }
        });
    }
    if let Some(index) = to_remove {
        wave.mod_slots.remove(index);
    }
    if ui.button("+ Add slot").clicked() {
        wave.mod_slots.push(WaveModSlot::default());
    }
}

fn wave_mod_source_label(source: WaveModSource) -> &'static str {
    match source {
        WaveModSource::None => "— none —",
        WaveModSource::Lfo1 => "LFO 1",
        WaveModSource::Lfo2 => "LFO 2",
        WaveModSource::Env1 => "Envelope 1",
        WaveModSource::Env2 => "Envelope 2",
        WaveModSource::Velocity => "Velocity",
    }
}

fn wave_mod_target_label(target: WaveModTarget) -> &'static str {
    match target {
        WaveModTarget::None => "— none —",
        WaveModTarget::Pitch => "Pitch",
        WaveModTarget::Osc1Position => "Osc 1 Position",
        WaveModTarget::Osc2Position => "Osc 2 Position",
        WaveModTarget::Osc1WarpAmount => "Osc 1 Warp Amount",
        WaveModTarget::Osc2WarpAmount => "Osc 2 Warp Amount",
        WaveModTarget::FilterCutoff => "Filter 1 Cutoff",
        WaveModTarget::Filter2Cutoff => "Filter 2 Cutoff",
        WaveModTarget::FilterResonance => "Filter 1 Resonance",
    }
}

/// Whether `source` is actually wired to something in `mod_slots` and, if so, the largest
/// magnitude it's routed at — see `trine_lfo_active_depth`, the `ModSlot` equivalent.
fn wave_lfo_active_depth(mod_slots: &[WaveModSlot], source: WaveModSource) -> (bool, f32) {
    let depth = mod_slots
        .iter()
        .filter(|slot| slot.source == source && slot.target != WaveModTarget::None)
        .map(|slot| slot.amount.abs())
        .fold(0.0f32, f32::max);
    (depth > 0.001, depth)
}

/// Combines Wave's LFOs and envelopes into the third column.
fn wave_lfos_envelopes_ui(ui: &mut egui::Ui, wave: &mut WaveParams) {
    ui.strong("LFOs");
    wave_lfos_ui(ui, wave);
    ui.separator();
    ui.strong("Envelopes");
    wave_envelopes_ui(ui, wave);
}

fn wave_lfos_ui(ui: &mut egui::Ui, wave: &mut WaveParams) {
    ui.strong("LFO 1");
    waveform_picker_ui(ui, &mut wave.lfo1_waveform);
    ui.add(
        egui::Slider::new(&mut wave.lfo1_rate_hz, 0.1..=20.0)
            .text("Rate")
            .suffix(" Hz"),
    );
    let (active1, depth1) = wave_lfo_active_depth(&wave.mod_slots, WaveModSource::Lfo1);
    lfo_shape_preview_ui(ui, wave.lfo1_waveform, active1, depth1);

    ui.separator();
    ui.strong("LFO 2");
    waveform_picker_ui(ui, &mut wave.lfo2_waveform);
    ui.add(
        egui::Slider::new(&mut wave.lfo2_rate_hz, 0.1..=20.0)
            .text("Rate")
            .suffix(" Hz"),
    );
    let (active2, depth2) = wave_lfo_active_depth(&wave.mod_slots, WaveModSource::Lfo2);
    lfo_shape_preview_ui(ui, wave.lfo2_waveform, active2, depth2);
}

fn wave_envelopes_ui(ui: &mut egui::Ui, wave: &mut WaveParams) {
    ui.strong("Envelope 1")
        .on_hover_text("Free — only audible once routed in the Modulation Matrix.");
    ui.add(
        egui::Slider::new(&mut wave.env1_attack_seconds, 0.0..=2.0)
            .text("Attack")
            .suffix(" s"),
    );
    ui.add(
        egui::Slider::new(&mut wave.env1_decay_seconds, 0.02..=2.0)
            .text("Decay")
            .suffix(" s"),
    );
    ui.add(egui::Slider::new(&mut wave.env1_sustain_level, 0.0..=1.0).text("Sustain"));
    ui.add(
        egui::Slider::new(&mut wave.env1_release_seconds, 0.02..=2.0)
            .text("Release")
            .suffix(" s"),
    );
    adsr_preview_ui(
        ui,
        wave.env1_attack_seconds,
        wave.env1_decay_seconds,
        wave.env1_sustain_level,
        wave.env1_release_seconds,
        egui::Color32::from_rgb(200, 140, 230),
    );

    ui.separator();
    ui.strong("Envelope 2")
        .on_hover_text("Free — only audible once routed in the Modulation Matrix.");
    ui.add(
        egui::Slider::new(&mut wave.env2_attack_seconds, 0.0..=2.0)
            .text("Attack")
            .suffix(" s"),
    );
    ui.add(
        egui::Slider::new(&mut wave.env2_decay_seconds, 0.02..=2.0)
            .text("Decay")
            .suffix(" s"),
    );
    ui.add(egui::Slider::new(&mut wave.env2_sustain_level, 0.0..=1.0).text("Sustain"));
    ui.add(
        egui::Slider::new(&mut wave.env2_release_seconds, 0.02..=2.0)
            .text("Release")
            .suffix(" s"),
    );
    adsr_preview_ui(
        ui,
        wave.env2_attack_seconds,
        wave.env2_decay_seconds,
        wave.env2_sustain_level,
        wave.env2_release_seconds,
        egui::Color32::from_rgb(200, 140, 230),
    );

    ui.separator();
    ui.strong("Amplitude Envelope")
        .on_hover_text("Always active — directly drives amplitude.");
    ui.add(
        egui::Slider::new(&mut wave.amp_attack_seconds, 0.0..=2.0)
            .text("Attack")
            .suffix(" s"),
    );
    ui.add(
        egui::Slider::new(&mut wave.amp_decay_seconds, 0.02..=2.0)
            .text("Decay")
            .suffix(" s"),
    );
    ui.add(egui::Slider::new(&mut wave.amp_sustain_level, 0.0..=1.0).text("Sustain"));
    ui.add(
        egui::Slider::new(&mut wave.amp_release_seconds, 0.02..=2.0)
            .text("Release")
            .suffix(" s"),
    );
    adsr_preview_ui(
        ui,
        wave.amp_attack_seconds,
        wave.amp_decay_seconds,
        wave.amp_sustain_level,
        wave.amp_release_seconds,
        egui::Color32::from_rgb(120, 220, 140),
    );
}

/// One entry in a factory-sample category: (display label, path relative to
/// `factory_samples_dir()`).
type FactorySampleEntry = (&'static str, &'static str);

const KICK_SAMPLES: &[FactorySampleEntry] = &[
    ("Kick", "kick/kick.wav"),
    ("Kick (Tight)", "kick/kick_tight.wav"),
    ("Kick (808)", "kick/kick_808.wav"),
    ("Kick (Sub)", "kick/kick_sub.wav"),
    ("Kick (Punchy)", "kick/kick_punchy.wav"),
    ("Kick (Distorted)", "kick/kick_distorted.wav"),
    ("Kick (Lo-Fi)", "kick/kick_lofi.wav"),
    ("Kick (Boom)", "kick/kick_boom.wav"),
    ("Kick (Click)", "kick/kick_click.wav"),
    ("Kick (B)", "kick/kick_b.wav"),
    ("Kick (Tight, B)", "kick/kick_tight_b.wav"),
    ("Kick (808, B)", "kick/kick_808_b.wav"),
    ("Kick (Sub, B)", "kick/kick_sub_b.wav"),
    ("Kick (Punchy, B)", "kick/kick_punchy_b.wav"),
    ("Kick (Distorted, B)", "kick/kick_distorted_b.wav"),
    ("Kick (Lo-Fi, B)", "kick/kick_lofi_b.wav"),
    ("Kick (Boom, B)", "kick/kick_boom_b.wav"),
    ("Kick (Click, B)", "kick/kick_click_b.wav"),
];

const SNARE_SAMPLES: &[FactorySampleEntry] = &[
    ("Snare", "snare/snare.wav"),
    ("Snare (Tight)", "snare/snare_tight.wav"),
    ("Snare (Fat)", "snare/snare_fat.wav"),
    ("Snare (Rimshot)", "snare/snare_rimshot.wav"),
    ("Snare (Lo-Fi)", "snare/snare_lofi.wav"),
    ("Snare (808)", "snare/snare_808.wav"),
    ("Snare (Cross-stick)", "snare/snare_crossstick.wav"),
    ("Snare (B)", "snare/snare_b.wav"),
    ("Snare (Tight, B)", "snare/snare_tight_b.wav"),
    ("Snare (Fat, B)", "snare/snare_fat_b.wav"),
    ("Snare (Rimshot, B)", "snare/snare_rimshot_b.wav"),
    ("Snare (Lo-Fi, B)", "snare/snare_lofi_b.wav"),
    ("Snare (808, B)", "snare/snare_808_b.wav"),
    ("Snare (Cross-stick, B)", "snare/snare_crossstick_b.wav"),
];

const HAT_SAMPLES: &[FactorySampleEntry] = &[
    ("Closed Hat", "hat/hat_closed.wav"),
    ("Open Hat", "hat/hat_open.wav"),
    ("Pedal Hat", "hat/hat_pedal.wav"),
    ("Tight Hat", "hat/hat_tight.wav"),
    ("Metallic Hat", "hat/hat_metallic.wav"),
    ("Sizzle Hat", "hat/hat_sizzle.wav"),
    ("Closed Hat (B)", "hat/hat_closed_b.wav"),
    ("Open Hat (B)", "hat/hat_open_b.wav"),
    ("Pedal Hat (B)", "hat/hat_pedal_b.wav"),
    ("Tight Hat (B)", "hat/hat_tight_b.wav"),
    ("Metallic Hat (B)", "hat/hat_metallic_b.wav"),
    ("Sizzle Hat (B)", "hat/hat_sizzle_b.wav"),
];

const CLAP_SAMPLES: &[FactorySampleEntry] = &[
    ("Clap", "clap/clap.wav"),
    ("Clap (Tight)", "clap/clap_tight.wav"),
    ("Clap (808)", "clap/clap_808.wav"),
    ("Clap (Wide)", "clap/clap_wide.wav"),
    ("Clap (B)", "clap/clap_b.wav"),
    ("Clap (Tight, B)", "clap/clap_tight_b.wav"),
    ("Clap (808, B)", "clap/clap_808_b.wav"),
    ("Clap (Wide, B)", "clap/clap_wide_b.wav"),
];

const TOM_SAMPLES: &[FactorySampleEntry] = &[
    ("Low Tom", "tom/tom_low.wav"),
    ("Mid Tom", "tom/tom_mid.wav"),
    ("High Tom", "tom/tom_high.wav"),
    ("Floor Tom", "tom/tom_floor.wav"),
    ("Synth Tom", "tom/tom_synth.wav"),
    ("Low Tom (B)", "tom/tom_low_b.wav"),
    ("Mid Tom (B)", "tom/tom_mid_b.wav"),
    ("High Tom (B)", "tom/tom_high_b.wav"),
    ("Floor Tom (B)", "tom/tom_floor_b.wav"),
    ("Synth Tom (B)", "tom/tom_synth_b.wav"),
];

const CYMBAL_SAMPLES: &[FactorySampleEntry] = &[
    ("Crash", "cymbal/crash.wav"),
    ("Ride", "cymbal/ride.wav"),
    ("Splash", "cymbal/splash.wav"),
    ("China", "cymbal/china.wav"),
    ("Bell", "cymbal/bell.wav"),
    ("Crash (Reverse)", "cymbal/crash_reverse.wav"),
    ("Crash 2", "cymbal/crash_2.wav"),
    ("Sizzle", "cymbal/sizzle.wav"),
    ("Crash (B)", "cymbal/crash_b.wav"),
    ("Ride (B)", "cymbal/ride_b.wav"),
    ("Splash (B)", "cymbal/splash_b.wav"),
    ("China (B)", "cymbal/china_b.wav"),
    ("Bell (B)", "cymbal/bell_b.wav"),
    ("Crash (Reverse, B)", "cymbal/crash_reverse_b.wav"),
    ("Crash 2 (B)", "cymbal/crash_2_b.wav"),
    ("Sizzle (B)", "cymbal/sizzle_b.wav"),
];

const PERC_SAMPLES: &[FactorySampleEntry] = &[
    ("Rim", "perc/rim.wav"),
    ("Cowbell", "perc/cowbell.wav"),
    ("Shaker", "perc/shaker.wav"),
    ("Tambourine", "perc/tambourine.wav"),
    ("Snap", "perc/snap.wav"),
    ("Woodblock", "perc/woodblock.wav"),
    ("Conga (High)", "perc/conga_high.wav"),
    ("Conga (Low)", "perc/conga_low.wav"),
    ("Bongo (High)", "perc/bongo_high.wav"),
    ("Bongo (Low)", "perc/bongo_low.wav"),
    ("Clave", "perc/clave.wav"),
    ("Triangle", "perc/triangle.wav"),
    ("Timbale (High)", "perc/timbale_high.wav"),
    ("Timbale (Low)", "perc/timbale_low.wav"),
    ("Agogo (High)", "perc/agogo_high.wav"),
    ("Agogo (Low)", "perc/agogo_low.wav"),
    ("Castanet", "perc/castanet.wav"),
    ("Sleigh Bells", "perc/sleigh_bells.wav"),
    ("Djembe", "perc/djembe.wav"),
    ("Taiko", "perc/taiko.wav"),
    ("Frame Drum", "perc/frame_drum.wav"),
    ("Rim (B)", "perc/rim_b.wav"),
    ("Cowbell (B)", "perc/cowbell_b.wav"),
    ("Shaker (B)", "perc/shaker_b.wav"),
    ("Tambourine (B)", "perc/tambourine_b.wav"),
    ("Conga (High, B)", "perc/conga_high_b.wav"),
    ("Conga (Low, B)", "perc/conga_low_b.wav"),
    ("Bongo (High, B)", "perc/bongo_high_b.wav"),
    ("Bongo (Low, B)", "perc/bongo_low_b.wav"),
    ("Clave (B)", "perc/clave_b.wav"),
    ("Snap (B)", "perc/snap_b.wav"),
    ("Woodblock (B)", "perc/woodblock_b.wav"),
    ("Triangle (B)", "perc/triangle_b.wav"),
    ("Timbale (High, B)", "perc/timbale_high_b.wav"),
    ("Timbale (Low, B)", "perc/timbale_low_b.wav"),
    ("Agogo (High, B)", "perc/agogo_high_b.wav"),
    ("Agogo (Low, B)", "perc/agogo_low_b.wav"),
    ("Castanet (B)", "perc/castanet_b.wav"),
    ("Sleigh Bells (B)", "perc/sleigh_bells_b.wav"),
    ("Djembe (B)", "perc/djembe_b.wav"),
    ("Taiko (B)", "perc/taiko_b.wav"),
    ("Frame Drum (B)", "perc/frame_drum_b.wav"),
];

/// Non-melodic Mallet entries — the pitched instruments are generated by `MELODIC_INSTRUMENTS`
/// below instead of hand-listed here.
const MALLET_EXTRA_SAMPLES: &[FactorySampleEntry] =
    &[("Bell Tree", "mallet/bell_tree.wav"), ("Bell Tree (B)", "mallet/bell_tree_b.wav")];

/// Non-melodic Orchestral entries — Pizzicato/Brass Stab/Horn Hit are generated instead.
const ORCHESTRAL_EXTRA_SAMPLES: &[FactorySampleEntry] = &[
    ("Choir Hit", "orchestral/choir_hit.wav"),
    ("String Swell", "orchestral/string_swell.wav"),
    ("Timpani", "orchestral/timpani.wav"),
];

const CHIP_FX_SAMPLES: &[FactorySampleEntry] = &[
    ("Coin", "chip/coin.wav"),
    ("Jump", "chip/jump.wav"),
    ("Laser", "chip/laser.wav"),
    ("Powerup", "chip/powerup.wav"),
    ("Blip", "chip/blip.wav"),
    ("Explosion", "chip/explosion.wav"),
    ("Hurt", "chip/hurt.wav"),
    ("Select", "chip/select.wav"),
    ("Level Up", "chip/level_up.wav"),
    ("Game Over", "chip/game_over.wav"),
    ("Checkpoint", "chip/checkpoint.wav"),
    ("Footstep", "chip/footstep.wav"),
    ("Menu Back", "chip/menu_back.wav"),
    ("Error", "chip/error.wav"),
    ("1-Up", "chip/one_up.wav"),
    ("Countdown", "chip/countdown.wav"),
    ("Power Down", "chip/power_down.wav"),
    ("Teleport", "chip/teleport.wav"),
    ("Charge Up", "chip/charge_up.wav"),
    ("Warp", "chip/warp.wav"),
    ("Item Get", "chip/item_get.wav"),
    ("Boss Alert", "chip/boss_alert.wav"),
    ("Coin (B)", "chip/coin_b.wav"),
    ("Jump (B)", "chip/jump_b.wav"),
    ("Laser (B)", "chip/laser_b.wav"),
    ("Powerup (B)", "chip/powerup_b.wav"),
    ("Blip (B)", "chip/blip_b.wav"),
    ("Explosion (B)", "chip/explosion_b.wav"),
    ("Hurt (B)", "chip/hurt_b.wav"),
    ("Select (B)", "chip/select_b.wav"),
    ("Level Up (B)", "chip/level_up_b.wav"),
    ("Game Over (B)", "chip/game_over_b.wav"),
    ("Checkpoint (B)", "chip/checkpoint_b.wav"),
    ("Footstep (B)", "chip/footstep_b.wav"),
    ("Menu Back (B)", "chip/menu_back_b.wav"),
    ("Error (B)", "chip/error_b.wav"),
    ("1-Up (B)", "chip/one_up_b.wav"),
    ("Countdown (B)", "chip/countdown_b.wav"),
    ("Power Down (B)", "chip/power_down_b.wav"),
    ("Teleport (B)", "chip/teleport_b.wav"),
    ("Charge Up (B)", "chip/charge_up_b.wav"),
    ("Warp (B)", "chip/warp_b.wav"),
    ("Item Get (B)", "chip/item_get_b.wav"),
    ("Boss Alert (B)", "chip/boss_alert_b.wav"),
];

const FX_SAMPLES: &[FactorySampleEntry] = &[
    ("Riser", "fx/riser.wav"),
    ("Downlifter", "fx/downlifter.wav"),
    ("Impact", "fx/impact.wav"),
    ("Whoosh", "fx/whoosh.wav"),
    ("Sweep Up", "fx/sweep_up.wav"),
    ("Sweep Down", "fx/sweep_down.wav"),
    ("Braam", "fx/braam.wav"),
    ("Sub Boom", "fx/sub_boom.wav"),
    ("Glitch", "fx/glitch.wav"),
    ("Reverse Snare", "fx/reverse_snare.wav"),
    ("Riser (B)", "fx/riser_b.wav"),
    ("Downlifter (B)", "fx/downlifter_b.wav"),
    ("Impact (B)", "fx/impact_b.wav"),
    ("Whoosh (B)", "fx/whoosh_b.wav"),
    ("Sweep Up (B)", "fx/sweep_up_b.wav"),
    ("Sweep Down (B)", "fx/sweep_down_b.wav"),
    ("Braam (B)", "fx/braam_b.wav"),
    ("Sub Boom (B)", "fx/sub_boom_b.wav"),
    ("Glitch (B)", "fx/glitch_b.wav"),
    ("Reverse Snare (B)", "fx/reverse_snare_b.wav"),
];

/// Non-melodic Bass entries — Bass Pluck/Slap Bass/Reese Bass/FM Bass are generated instead.
const BASS_EXTRA_SAMPLES: &[FactorySampleEntry] = &[
    ("Sub (Low)", "bass/sub_low.wav"),
    ("Sub (Mid)", "bass/sub_mid.wav"),
    ("Sub (High)", "bass/sub_high.wav"),
    ("Sub Drop", "bass/sub_drop.wav"),
    ("Bass Growl", "bass/bass_growl.wav"),
    ("Sub (Low, B)", "bass/sub_low_b.wav"),
    ("Sub (Mid, B)", "bass/sub_mid_b.wav"),
    ("Sub (High, B)", "bass/sub_high_b.wav"),
    ("Sub Drop (B)", "bass/sub_drop_b.wav"),
    ("Bass Growl (B)", "bass/bass_growl_b.wav"),
];

/// The 20 melodic (pitched, one-shot-per-note) instruments across Mallet/Pluck/Orchestral/Bass:
/// (category, file-name stem, display name). Each gets a root sample (no suffix) plus every
/// `CHROMATIC_NOTE_SUFFIXES` note, generated below rather than hand-listed — hand-listing ~260
/// near-identical entries is exactly how the "m7"/"M7" note collided on macOS's
/// case-insensitive filesystem (both saved to the same path) without anyone noticing.
const MELODIC_INSTRUMENTS: &[(&str, &str, &str)] = &[
    ("Mallet", "marimba", "Marimba"),
    ("Mallet", "xylophone", "Xylophone"),
    ("Mallet", "kalimba", "Kalimba"),
    ("Mallet", "glockenspiel", "Glockenspiel"),
    ("Mallet", "vibraphone", "Vibraphone"),
    ("Mallet", "steel_drum", "Steel Drum"),
    ("Mallet", "celesta", "Celesta"),
    ("Pluck", "guitar_pluck", "Guitar Pluck"),
    ("Pluck", "harp_pluck", "Harp Pluck"),
    ("Pluck", "koto", "Koto"),
    ("Pluck", "music_box", "Music Box"),
    ("Pluck", "nylon_pluck", "Nylon Pluck"),
    ("Pluck", "banjo", "Banjo"),
    ("Orchestral", "pizzicato", "Pizzicato"),
    ("Orchestral", "brass_stab", "Brass Stab"),
    ("Orchestral", "horn_hit", "Horn Hit"),
    ("Bass", "bass_pluck", "Bass Pluck"),
    ("Bass", "slap_bass", "Slap Bass"),
    ("Bass", "reese_bass", "Reese Bass"),
    ("Bass", "fm_bass", "FM Bass"),
];

/// A full 2-octave chromatic range around each melodic instrument's root (root itself has no
/// suffix and isn't listed here): (file-name suffix, display label), low to high.
const CHROMATIC_NOTE_SUFFIXES: &[(&str, &str)] = &[
    ("low8ve", "-8ve"),
    ("m2", "m2"),
    ("2nd", "2nd"),
    ("m3", "m3"),
    ("3rd", "3rd"),
    ("4th", "4th"),
    ("tt", "TT"),
    ("5th", "5th"),
    ("m6", "m6"),
    ("6th", "6th"),
    ("min7", "m7"),
    ("maj7", "M7"),
    ("8ve", "8ve"),
];

/// Built once and cached: per-category factory sample lists, combining the hand-listed
/// non-melodic categories above with generated note entries for every `MELODIC_INSTRUMENTS`
/// instrument (see its doc comment for why those are generated rather than hand-listed).
fn factory_drum_samples() -> &'static [(&'static str, Vec<(String, String)>)] {
    static SAMPLES: std::sync::LazyLock<Vec<(&'static str, Vec<(String, String)>)>> =
        std::sync::LazyLock::new(|| {
            fn owned(entries: &[FactorySampleEntry]) -> Vec<(String, String)> {
                entries.iter().map(|(label, path)| (label.to_string(), path.to_string())).collect()
            }
            let mut categories: Vec<(&'static str, Vec<(String, String)>)> = vec![
                ("Kick", owned(KICK_SAMPLES)),
                ("Snare", owned(SNARE_SAMPLES)),
                ("Hat", owned(HAT_SAMPLES)),
                ("Clap", owned(CLAP_SAMPLES)),
                ("Tom", owned(TOM_SAMPLES)),
                ("Cymbal", owned(CYMBAL_SAMPLES)),
                ("Perc", owned(PERC_SAMPLES)),
                ("Mallet", owned(MALLET_EXTRA_SAMPLES)),
                ("Pluck", Vec::new()),
                ("Orchestral", owned(ORCHESTRAL_EXTRA_SAMPLES)),
                ("Chip FX", owned(CHIP_FX_SAMPLES)),
                ("FX", owned(FX_SAMPLES)),
                ("Bass", owned(BASS_EXTRA_SAMPLES)),
            ];
            for (category, stem, display) in MELODIC_INSTRUMENTS {
                let folder = category.to_lowercase();
                let entry = categories
                    .iter_mut()
                    .find(|(name, _)| name == category)
                    .expect("MELODIC_INSTRUMENTS category must exist in `categories` above");
                entry.1.push((display.to_string(), format!("{folder}/{stem}.wav")));
                for (suffix, label) in CHROMATIC_NOTE_SUFFIXES {
                    entry.1.push((format!("{display} ({label})"), format!("{folder}/{stem}_{suffix}.wav")));
                }
            }
            categories
        });
    &SAMPLES
}

fn factory_samples_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/samples")
}

/// Points the demo song's drum lanes at the bundled placeholder one-shots (matching the lane
/// names `Song::demo()` sets up, in order), so opening the app for the first time plays real
/// samples, not just the synth.
fn preload_demo_samples(song: &Arc<Mutex<Song>>, sample_rate: u32) {
    let assets = factory_samples_dir();
    let mut song = song.lock().unwrap();
    let Some(drums_index) = song.tracks.iter().position(|t| t.name == "Drums") else {
        return;
    };
    let Some(region) = song.tracks[drums_index].regions.first_mut() else {
        return;
    };
    let RegionContent::StepGrid(lanes) = &mut region.content else {
        return;
    };
    let lane_samples = [
        (0, "kick/kick.wav"),        // Kick
        (1, "snare/snare.wav"),      // Snare
        (2, "hat/hat_closed.wav"),   // Closed Hat
        (3, "hat/hat_open.wav"),     // Open Hat
        (4, "clap/clap.wav"),        // Clap
        (5, "perc/rim.wav"),         // Rim
        (6, "tom/tom_low.wav"),      // Low Tom
        (7, "cymbal/crash.wav"),     // Crash
        (8, "tom/tom_mid.wav"),      // Mid Tom
        (9, "tom/tom_high.wav"),     // High Tom
        (10, "cymbal/ride.wav"),     // Ride
        (11, "perc/cowbell.wav"),    // Cowbell
        (12, "perc/shaker.wav"),     // Shaker
        (13, "perc/tambourine.wav"), // Tambourine
        (14, "perc/woodblock.wav"),  // Woodblock
        (15, "perc/triangle.wav"),   // Triangle
    ];
    for (lane_index, filename) in lane_samples {
        if let Some(lane) = lanes.get_mut(lane_index) {
            lane.sample_path = assets.join(filename).display().to_string();
            lane.load_sample(sample_rate);
        }
    }
}

fn perform_save(song: &Song, path: &str) -> (bool, String) {
    let path = std::path::Path::new(path.trim());
    match song.save_to_file(path) {
        Ok(()) => (true, format!("Saved to {}", path.display())),
        Err(err) => (false, format!("{err:#}")),
    }
}

fn perform_load(path: &str, sample_rate: Option<u32>) -> Result<Song, String> {
    let path = std::path::Path::new(path.trim());
    Song::load_from_file(path, sample_rate).map_err(|err| format!("{err:#}"))
}

/// Opens a native file picker seeded to `current_path`'s parent directory (if any). Passing
/// `save_as` picks a save dialog with that suggested filename; `None` picks an open dialog.
/// Returns the chosen path's display string, or `None` if the dialog was cancelled.
fn browse_for_file(
    current_path: &str,
    filter_label: &str,
    extensions: &[&str],
    save_as: Option<&str>,
) -> Option<String> {
    let mut dialog = rfd::FileDialog::new().add_filter(filter_label, extensions);
    if let Some(dir) = Path::new(current_path.trim())
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
    {
        dialog = dialog.set_directory(dir);
    }
    let path = match save_as {
        Some(name) => dialog.set_file_name(name).save_file(),
        None => dialog.pick_file(),
    };
    path.map(|p| p.display().to_string())
}

fn perform_export(song: &Song, sample_rate: u32, loops: u32, path: &str) -> (bool, String) {
    let path = std::path::Path::new(path.trim());
    match audio::render_song_to_wav(song, sample_rate, loops, path) {
        Ok(()) => (
            true,
            format!("Exported {loops} loop(s) to {}", path.display()),
        ),
        Err(err) => (false, format!("{err:#}")),
    }
}

/// Writes a just-recorded take to a new mono 16-bit WAV file under `recordings/` (created if
/// needed) and returns its path — same on-disk format `audio::render_song_to_wav` bounces to.
/// Named by track index and a timestamp rather than the track's name, since track names are
/// user-editable free text and not guaranteed to be filesystem-safe.
fn write_recording_wav(
    track_index: usize,
    samples: &[f32],
    sample_rate: u32,
) -> Result<std::path::PathBuf, String> {
    let dir = Path::new("recordings");
    std::fs::create_dir_all(dir).map_err(|err| format!("{err:#}"))?;
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let path = dir.join(format!("track{track_index}_{timestamp}.wav"));
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&path, spec).map_err(|err| format!("{err:#}"))?;
    for &sample in samples {
        let value = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        writer
            .write_sample(value)
            .map_err(|err| format!("{err:#}"))?;
    }
    writer.finalize().map_err(|err| format!("{err:#}"))?;
    Ok(path)
}

/// Turns a just-stopped recording into a saved WAV file plus a take on `track_index` — the
/// toolbar Record button's stop-side logic, pulled out so it can be tested and read on its own.
/// Re-recording from the exact same `start_tick` on the same track (rewind, hit record again)
/// joins the existing `TakeFolder` there as a new take, comped in immediately — see
/// `TakeFolder::add_take_and_activate`; anything else starts a fresh folder, its span frozen at
/// this take's own recorded duration. `engine_sample_rate` is `None` only if the playback engine
/// failed to start (see `SimpleDawApp::sample_rate`), in which case the take is added unloaded,
/// same as a `Lane` sample would be.
fn finish_recording(
    song: &mut Song,
    track_index: usize,
    start_tick: usize,
    samples: &[f32],
    captured_sample_rate: u32,
    engine_sample_rate: Option<u32>,
) -> (bool, String) {
    if samples.is_empty() {
        return (false, "No audio captured".to_string());
    }
    let path = match write_recording_wav(track_index, samples, captured_sample_rate) {
        Ok(path) => path,
        Err(err) => return (false, err),
    };
    let file_path = path.to_string_lossy().to_string();

    let Some(track) = song
        .tracks
        .get(track_index)
        .filter(|t| t.kind == TrackKind::Audio)
    else {
        return (
            false,
            format!(
                "Recorded {}, but its track no longer exists",
                path.display()
            ),
        );
    };
    let existing_folder_index = track
        .take_folders
        .iter()
        .position(|f| f.start_tick == start_tick);
    let duration_seconds = samples.len() as f64 / captured_sample_rate.max(1) as f64;
    let length_ticks = (duration_seconds * audio::ticks_per_second(song.bpm_at(start_tick)))
        .ceil()
        .max(1.0) as usize;

    let track = &mut song.tracks[track_index];
    let (folder_index, take_index) = match existing_folder_index {
        Some(folder_index) => {
            let take_index = track.take_folders[folder_index].add_take_and_activate(file_path);
            (folder_index, take_index)
        }
        None => {
            track
                .take_folders
                .push(TakeFolder::new(start_tick, length_ticks, file_path));
            (track.take_folders.len() - 1, 0)
        }
    };
    if let Some(rate) = engine_sample_rate {
        track.take_folders[folder_index].takes[take_index].load(rate);
    }
    (true, format!("Recorded {}", path.display()))
}

/// Writes a track's baked-down render (`audio::render_track_to_buffer`'s output) to a stereo WAV
/// file, the freeze/bounce-in-place counterpart of `write_recording_wav` — same "own cache
/// directory, timestamped filename" convention, kept separate (`frozen/` vs `recordings/`) since
/// the two aren't the same kind of asset. Written stereo (unlike a recorded take's mono WAV) for
/// full fidelity on disk even though `SampleBuffer::load_wav_resampled` downmixes to mono on load
/// same as it already does for any imported stereo sample — see `render_track_to_buffer`'s doc.
fn write_frozen_track_wav(
    track_index: usize,
    l: &[f32],
    r: &[f32],
    sample_rate: u32,
) -> Result<std::path::PathBuf, String> {
    let dir = Path::new("frozen");
    std::fs::create_dir_all(dir).map_err(|err| format!("{err:#}"))?;
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    // A nanosecond timestamp alone isn't guaranteed unique under heavy parallelism (e.g. multiple
    // tests freezing the same track index in the same process) — an atomic counter closes that gap.
    static SEQUENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let sequence = SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = dir.join(format!("track{track_index}_{timestamp}_{sequence}.wav"));
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&path, spec).map_err(|err| format!("{err:#}"))?;
    for (&l, &r) in l.iter().zip(r) {
        writer
            .write_sample((l.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
            .map_err(|err| format!("{err:#}"))?;
        writer
            .write_sample((r.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
            .map_err(|err| format!("{err:#}"))?;
    }
    writer.finalize().map_err(|err| format!("{err:#}"))?;
    Ok(path)
}

/// Bakes `track_index`'s current live content (notes/steps/audio clips, run through its own
/// effect chain — see `audio::render_track_to_buffer`) to a WAV file and points `Track::frozen_clip`
/// at it, setting `Track::frozen`. Deletes the previous freeze's file first, if any, so repeated
/// freeze/unfreeze cycles don't leak files into `frozen/`. `engine_sample_rate` is `None` only if
/// the playback engine failed to start, in which case the clip is left unloaded (same tolerance
/// `finish_recording` already has).
fn freeze_track(
    song: &mut Song,
    track_index: usize,
    render_sample_rate: u32,
    engine_sample_rate: Option<u32>,
) -> (bool, String) {
    let Some((l, r)) = audio::render_track_to_buffer(song, track_index, render_sample_rate) else {
        return (false, "Nothing to freeze".to_string());
    };
    let path = match write_frozen_track_wav(track_index, &l, &r, render_sample_rate) {
        Ok(path) => path,
        Err(err) => return (false, err),
    };
    let Some(track) = song.tracks.get_mut(track_index) else {
        return (false, "Track no longer exists".to_string());
    };
    if let Some(old_clip) = track.frozen_clip.take() {
        std::fs::remove_file(&old_clip.file_path).ok();
    }
    let mut clip = AudioClip::new(0, path.to_string_lossy().to_string());
    if let Some(rate) = engine_sample_rate {
        clip.load(rate);
    }
    track.frozen = true;
    track.frozen_clip = Some(clip);
    (true, format!("Froze {}", path.display()))
}

/// Reverts `track_index` to live synthesis, deleting the frozen WAV file `freeze_track` wrote.
fn unfreeze_track(song: &mut Song, track_index: usize) {
    let Some(track) = song.tracks.get_mut(track_index) else {
        return;
    };
    if let Some(clip) = track.frozen_clip.take() {
        std::fs::remove_file(&clip.file_path).ok();
    }
    track.frozen = false;
}

/// Destructively replaces `track_index`'s notes/steps/audio-clips/take-folders/effects with one
/// baked `AudioClip` (via the same `render_track_to_buffer`/`write_frozen_track_wav` primitives
/// `freeze_track` uses) and converts it to an audio track — the permanent counterpart of freeze.
/// Unfreezes first if the track was already frozen, so the bake reflects its live content, not a
/// stale previous freeze.
fn bounce_track_in_place(
    song: &mut Song,
    track_index: usize,
    render_sample_rate: u32,
    engine_sample_rate: Option<u32>,
) -> (bool, String) {
    if song.tracks.get(track_index).is_some_and(|t| t.frozen) {
        unfreeze_track(song, track_index);
    }
    let Some((l, r)) = audio::render_track_to_buffer(song, track_index, render_sample_rate) else {
        return (false, "Nothing to bounce".to_string());
    };
    let path = match write_frozen_track_wav(track_index, &l, &r, render_sample_rate) {
        Ok(path) => path,
        Err(err) => return (false, err),
    };
    let Some(track) = song.tracks.get_mut(track_index) else {
        return (false, "Track no longer exists".to_string());
    };
    let mut clip = AudioClip::new(0, path.to_string_lossy().to_string());
    if let Some(rate) = engine_sample_rate {
        clip.load(rate);
    }
    track.kind = TrackKind::Audio;
    track.regions.clear();
    track.audio_clips = vec![clip];
    track.take_folders.clear();
    track.effects.clear();
    (true, format!("Bounced {} to audio", path.display()))
}

/// Snapshots one live effect chain (the master bus, or a single track) back into its
/// `TrackEffectConfig` list form for persisting to a song file — the shared body behind
/// `sync_song_effects`'s master and per-track cases, which differ only in which chain/paths
/// they're reading.
fn chain_to_config(chain: &[Option<EffectInstance>], paths: &[String]) -> Vec<TrackEffectConfig> {
    chain
        .iter()
        .enumerate()
        .map(|(slot_index, slot)| match slot {
            Some(EffectInstance::Clap(effect)) => TrackEffectConfig::Clap {
                path: paths.get(slot_index).cloned().unwrap_or_default(),
                params: effect.param_snapshot(),
                sidechain_source: effect.sidechain_source,
            },
            Some(EffectInstance::BuiltIn(effect)) => effect.to_config(),
            None => TrackEffectConfig::Clap {
                path: paths.get(slot_index).cloned().unwrap_or_default(),
                params: Vec::new(),
                sidechain_source: None,
            },
        })
        .collect()
}

/// Writes the app's live effect state (master bus + every track's effect chain + every send bus's
/// and submix bus's effect chain) into `song`'s `master_effects`/`Track::effects`/
/// `SendBus::effects`/`SubmixBus::effects` fields so `save_to_file` captures it.
#[allow(clippy::too_many_arguments)]
fn sync_song_effects(
    song: &mut Song,
    master_effect_paths: &[String],
    master_effect_slots: &MasterEffectSlots,
    track_effect_paths: &[Vec<String>],
    track_effect_slots: &TrackEffectSlots,
    send_effect_paths: &[Vec<String>],
    send_effect_slots: &SendEffectSlots,
    submix_effect_paths: &[Vec<String>],
    submix_effect_slots: &SubmixEffectSlots,
) {
    if let Ok(chains) = master_effect_slots.lock() {
        song.master_effects = chains
            .first()
            .map(|chain| chain_to_config(chain, master_effect_paths))
            .unwrap_or_default();
    }

    if let Ok(chains) = track_effect_slots.lock() {
        for (index, track) in song.tracks.iter_mut().enumerate() {
            let paths = track_effect_paths.get(index).map(Vec::as_slice).unwrap_or(&[]);
            track.effects = chains
                .get(index)
                .map(|chain| chain_to_config(chain, paths))
                .unwrap_or_default();
        }
    }

    if let Ok(chains) = send_effect_slots.lock() {
        for (index, send) in song.sends.iter_mut().enumerate() {
            let paths = send_effect_paths.get(index).map(Vec::as_slice).unwrap_or(&[]);
            send.effects = chains
                .get(index)
                .map(|chain| chain_to_config(chain, paths))
                .unwrap_or_default();
        }
    }

    if let Ok(chains) = submix_effect_slots.lock() {
        for (index, submix) in song.submixes.iter_mut().enumerate() {
            let paths = submix_effect_paths.get(index).map(Vec::as_slice).unwrap_or(&[]);
            submix.effects = chains
                .get(index)
                .map(|chain| chain_to_config(chain, paths))
                .unwrap_or_default();
        }
    }
}

/// Loads a CLAP plugin at `path` and re-applies previously-saved `params` (by CLAP id) to it.
fn load_effect(
    path: &str,
    params: &[(u32, f64)],
    engine_config: Option<(f64, u32, u32)>,
) -> Result<(PluginInstance<DawHost>, LoadedEffect, PluginGuiHandle), String> {
    let Some((sample_rate, min_frames, max_frames)) = engine_config else {
        return Err("audio engine not running".to_string());
    };
    let plugin_path = std::path::Path::new(path.trim());
    let (instance, mut effect, gui) =
        plugin_host::load_and_activate(plugin_path, sample_rate, min_frames, max_frames)
            .map_err(|err| format!("{err:#}"))?;
    for (id, value) in params {
        effect.set_param_by_id(*id, *value);
    }
    Ok((instance, effect, gui))
}

/// Builds one live effect chain (the master bus, or a single track) from its saved
/// `TrackEffectConfig`s — loading any referenced CLAP plugin and instantiating any built-in DSP
/// effect. The shared body behind `apply_loaded_effects`'s master and per-track cases, and behind
/// `SimpleDawApp::new()`'s startup load of `Song::demo()`'s default master effects.
#[allow(clippy::type_complexity)]
fn build_effect_chain(
    specs: Vec<TrackEffectConfig>,
    engine_config: Option<(f64, u32, u32)>,
) -> (
    Vec<String>,
    Vec<Option<PluginInstance<DawHost>>>,
    Vec<Option<PluginGuiHandle>>,
    Vec<Option<(bool, String)>>,
    Vec<Option<EffectInstance>>,
) {
    let sample_rate = engine_config.map(|(sr, _, _)| sr as f32);
    let mut paths = Vec::with_capacity(specs.len());
    let mut instances = Vec::with_capacity(specs.len());
    let mut guis = Vec::with_capacity(specs.len());
    let mut messages = Vec::with_capacity(specs.len());
    let mut chain: Vec<Option<EffectInstance>> = Vec::with_capacity(specs.len());
    for spec in specs {
        match spec {
            TrackEffectConfig::Clap { path, params, sidechain_source } => {
                paths.push(path.clone());
                if path.trim().is_empty() {
                    instances.push(None);
                    guis.push(None);
                    chain.push(None);
                    messages.push(None);
                } else {
                    match load_effect(&path, &params, engine_config) {
                        Ok((instance, mut effect, gui)) => {
                            effect.sidechain_source = sidechain_source;
                            instances.push(Some(instance));
                            guis.push(Some(gui));
                            chain.push(Some(EffectInstance::Clap(effect)));
                            messages.push(Some((true, format!("Loaded {path}"))));
                        }
                        Err(err) => {
                            instances.push(None);
                            guis.push(None);
                            chain.push(None);
                            messages.push(Some((false, err)));
                        }
                    }
                }
            }
            builtin_spec => {
                paths.push(String::new());
                instances.push(None);
                guis.push(None);
                match sample_rate.and_then(|sr| BuiltInEffect::from_config(&builtin_spec, sr)) {
                    Some(effect) => {
                        chain.push(Some(EffectInstance::BuiltIn(effect)));
                        messages.push(None);
                    }
                    None => {
                        chain.push(None);
                        messages.push(Some((false, "Audio engine not running".to_string())));
                    }
                }
            }
        }
    }
    (paths, instances, guis, messages, chain)
}

/// Rebuilds one indexed chain slot (a real track's, or a send bus's) from its saved specs and
/// writes the result into that index's bookkeeping entries — the shared body behind
/// `apply_loaded_effects`'s per-track and per-send loops, which differ only in which parallel
/// `Vec`s/`slots` they're writing into.
#[allow(clippy::too_many_arguments)]
fn apply_chain_specs_at(
    index: usize,
    specs: Vec<TrackEffectConfig>,
    engine_config: Option<(f64, u32, u32)>,
    slots: &TrackEffectSlots,
    paths_out: &mut [Vec<String>],
    instances_out: &mut [Vec<Option<PluginInstance<DawHost>>>],
    guis_out: &mut [Vec<Option<PluginGuiHandle>>],
    messages_out: &mut [Vec<Option<(bool, String)>>],
) {
    let (paths, instances, guis, messages, chain) = build_effect_chain(specs, engine_config);
    if let Ok(mut slots) = slots.lock()
        && let Some(slot) = slots.get_mut(index)
    {
        *slot = chain;
    }
    if let Some(field) = paths_out.get_mut(index) {
        *field = paths;
    }
    if let Some(field) = instances_out.get_mut(index) {
        *field = instances;
    }
    if let Some(field) = guis_out.get_mut(index) {
        *field = guis;
    }
    if let Some(field) = messages_out.get_mut(index) {
        *field = messages;
    }
}

/// Re-loads the master bus's, every track's, and every send bus's effect chain after a `Song` is
/// loaded from a file, restoring each CLAP plugin's saved parameter values and re-instantiating
/// every built-in effect. Takes the loaded specs by value (extracted from the `Song` before it's
/// swapped into place) rather than the `Song` itself, so it can run as a free function alongside
/// the caller's `&mut Song` borrow.
#[allow(clippy::too_many_arguments)]
fn apply_loaded_effects(
    master_effect_paths: &mut Vec<String>,
    master_effect_instances: &mut Vec<Option<PluginInstance<DawHost>>>,
    master_effect_guis: &mut Vec<Option<PluginGuiHandle>>,
    master_effect_slots: &MasterEffectSlots,
    master_effect_messages: &mut Vec<Option<(bool, String)>>,
    loaded_master_specs: Vec<TrackEffectConfig>,
    track_effect_paths: &mut [Vec<String>],
    track_effect_instances: &mut [Vec<Option<PluginInstance<DawHost>>>],
    track_effect_guis: &mut [Vec<Option<PluginGuiHandle>>],
    track_effect_messages: &mut [Vec<Option<(bool, String)>>],
    track_effect_slots: &TrackEffectSlots,
    loaded_track_specs: Vec<Vec<TrackEffectConfig>>,
    send_effect_paths: &mut [Vec<String>],
    send_effect_instances: &mut [Vec<Option<PluginInstance<DawHost>>>],
    send_effect_guis: &mut [Vec<Option<PluginGuiHandle>>],
    send_effect_messages: &mut [Vec<Option<(bool, String)>>],
    send_effect_slots: &SendEffectSlots,
    loaded_send_specs: Vec<Vec<TrackEffectConfig>>,
    submix_effect_paths: &mut [Vec<String>],
    submix_effect_instances: &mut [Vec<Option<PluginInstance<DawHost>>>],
    submix_effect_guis: &mut [Vec<Option<PluginGuiHandle>>],
    submix_effect_messages: &mut [Vec<Option<(bool, String)>>],
    submix_effect_slots: &SubmixEffectSlots,
    loaded_submix_specs: Vec<Vec<TrackEffectConfig>>,
    engine_config: Option<(f64, u32, u32)>,
) {
    let (paths, instances, guis, messages, chain) =
        build_effect_chain(loaded_master_specs, engine_config);
    *master_effect_paths = paths;
    *master_effect_instances = instances;
    *master_effect_guis = guis;
    *master_effect_messages = messages;
    if let Ok(mut slots) = master_effect_slots.lock()
        && let Some(slot) = slots.get_mut(0)
    {
        *slot = chain;
    }

    for (index, track_specs) in loaded_track_specs.into_iter().enumerate() {
        apply_chain_specs_at(
            index,
            track_specs,
            engine_config,
            track_effect_slots,
            track_effect_paths,
            track_effect_instances,
            track_effect_guis,
            track_effect_messages,
        );
    }

    for (index, send_specs) in loaded_send_specs.into_iter().enumerate() {
        apply_chain_specs_at(
            index,
            send_specs,
            engine_config,
            send_effect_slots,
            send_effect_paths,
            send_effect_instances,
            send_effect_guis,
            send_effect_messages,
        );
    }

    for (index, submix_specs) in loaded_submix_specs.into_iter().enumerate() {
        apply_chain_specs_at(
            index,
            submix_specs,
            engine_config,
            submix_effect_slots,
            submix_effect_paths,
            submix_effect_instances,
            submix_effect_guis,
            submix_effect_messages,
        );
    }
}

/// One Bar/Beat/Div/Tick (or Tempo/Sig) cell of `transport_lcd_ui`: a zero-padded number with
/// its leading padding digits dimmed (so "004" reads as a bright "4"), and a small caption
/// underneath. `width` is the total digit count; pass 1 for single-digit fields (no padding).
fn lcd_segment(ui: &mut egui::Ui, value: usize, width: usize, label: &str) {
    ui.vertical(|ui| {
        let text = format!("{value:0>width$}");
        let dim_count = text
            .chars()
            .take_while(|c| *c == '0')
            .count()
            .min(width - 1);
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 0.0;
            if dim_count > 0 {
                ui.label(
                    egui::RichText::new(&text[..dim_count])
                        .monospace()
                        .size(14.0)
                        .color(egui::Color32::from_gray(90)),
                );
            }
            ui.label(
                egui::RichText::new(&text[dim_count..])
                    .monospace()
                    .size(14.0)
                    .color(egui::Color32::WHITE),
            );
        });
        ui.label(
            egui::RichText::new(label)
                .size(8.0)
                .color(egui::Color32::from_gray(140)),
        );
    });
}

/// Thin vertical rule between `lcd_segment` cells.
fn lcd_divider(ui: &mut egui::Ui) {
    ui.add_space(6.0);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(1.0, 20.0), egui::Sense::hover());
    ui.painter()
        .rect_filled(rect, 0.0, egui::Color32::from_rgb(60, 62, 66));
    ui.add_space(6.0);
}

/// A visually distinct cluster within the toolbar — a subtly raised rounded panel that groups
/// related controls (transport, zoom, device picker, …) so the toolbar reads as separate
/// sections instead of one undifferentiated strip of widgets. Purely presentational: callers
/// place their existing widgets inside `add_contents` unchanged.
fn toolbar_group(ui: &mut egui::Ui, add_contents: impl FnOnce(&mut egui::Ui)) {
    egui::Frame::new()
        .fill(egui::Color32::from_rgb(38, 38, 38))
        .corner_radius(5.0)
        .inner_margin(egui::Margin::symmetric(10, 5))
        .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(20, 20, 20)))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 8.0;
                add_contents(ui);
            });
        });
}

/// Denominators reachable from the SIG picker — restricted to values that evenly divide a
/// sixteenth-note step (see `model::Song::steps_per_beat`), so `steps_per_bar`/`steps_per_beat`
/// never need to round or reject an input.
const TIME_SIGNATURE_DENOMINATORS: [u8; 5] = [1, 2, 4, 8, 16];

/// Logic Pro–style transport LCD: Bar/Beat/Div/Tick derived from the absolute tick counter and
/// the song's own time signature, plus editable Tempo/Signature fields, in one dark rounded
/// panel.
fn transport_lcd_ui(
    ui: &mut egui::Ui,
    tick: usize,
    song: &mut Song,
    tap_tempo: &mut tempo::TapTempo,
) {
    let steps_per_beat = song.steps_per_beat();
    let ticks_per_beat = steps_per_beat * TICKS_PER_STEP;
    let ticks_per_bar = song.steps_per_bar() * TICKS_PER_STEP;

    let bar = tick / ticks_per_bar + 1;
    let tick_in_bar = tick % ticks_per_bar;
    let beat = tick_in_bar / ticks_per_beat + 1;
    let tick_in_beat = tick_in_bar % ticks_per_beat;
    let division = tick_in_beat / TICKS_PER_STEP + 1;
    let sub_tick = tick_in_beat % TICKS_PER_STEP;

    egui::Frame::new()
        .fill(egui::Color32::from_rgb(35, 37, 40))
        .corner_radius(6.0)
        .inner_margin(egui::Margin::symmetric(10, 4))
        .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(15, 15, 15)))
        .show(ui, |ui| {
            ui.with_layout(egui::Layout::left_to_right(egui::Align::Min), |ui| {
                lcd_segment(ui, bar, 3, "BAR");
                lcd_divider(ui);
                lcd_segment(ui, beat, 1, "BEAT");
                lcd_divider(ui);
                lcd_segment(ui, division, 1, "DIV");
                lcd_divider(ui);
                lcd_segment(ui, sub_tick, 3, "TICK");

                ui.add_space(8.0);
                egui::Frame::new()
                    .fill(egui::Color32::from_rgb(46, 49, 53))
                    .corner_radius(4.0)
                    .inner_margin(egui::Margin {
                        left: 8,
                        right: 8,
                        top: 0,
                        bottom: 6,
                    })
                    .show(ui, |ui| {
                        ui.with_layout(egui::Layout::left_to_right(egui::Align::Min), |ui| {
                            ui.spacing_mut().button_padding.y = 0.0;
                            ui.vertical(|ui| {
                                ui.scope(|ui| {
                                    ui.style_mut().override_font_id =
                                        Some(egui::FontId::monospace(14.0));
                                    ui.add(
                                        egui::DragValue::new(&mut song.bpm)
                                            .range(20.0..=300.0)
                                            .fixed_decimals(0),
                                    );
                                });
                                ui.label(
                                    egui::RichText::new("TEMPO")
                                        .size(8.0)
                                        .color(egui::Color32::from_gray(140)),
                                );
                            });
                            lcd_divider(ui);
                            ui.vertical(|ui| {
                                if ui
                                    .add(egui::Button::new("TAP").small())
                                    .on_hover_text("Click on the beat a few times to set tempo")
                                    .clicked()
                                    && let Some(bpm) = tap_tempo.tap(std::time::Instant::now())
                                {
                                    song.bpm = bpm.clamp(20.0, 300.0);
                                }
                                ui.label(
                                    egui::RichText::new("TAP")
                                        .size(8.0)
                                        .color(egui::Color32::from_gray(140)),
                                );
                            });
                            lcd_divider(ui);
                            ui.vertical(|ui| {
                                ui.scope(|ui| {
                                    ui.spacing_mut().item_spacing.x = 2.0;
                                    ui.style_mut().override_font_id =
                                        Some(egui::FontId::monospace(13.0));
                                    ui.horizontal(|ui| {
                                        ui.add(
                                            egui::DragValue::new(
                                                &mut song.time_signature_numerator,
                                            )
                                            .range(1..=32),
                                        );
                                        ui.label("/");
                                        egui::ComboBox::from_id_salt("time_signature_denominator")
                                            .selected_text(format!(
                                                "{}",
                                                song.time_signature_denominator
                                            ))
                                            .width(36.0)
                                            .show_ui(ui, |ui| {
                                                for denominator in TIME_SIGNATURE_DENOMINATORS {
                                                    ui.selectable_value(
                                                        &mut song.time_signature_denominator,
                                                        denominator,
                                                        format!("{denominator}"),
                                                    );
                                                }
                                            });
                                    });
                                });
                                ui.label(
                                    egui::RichText::new("SIG")
                                        .size(8.0)
                                        .color(egui::Color32::from_gray(140)),
                                );
                            });
                        });
                    });
            });
        });
}

impl eframe::App for SimpleDawApp {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let playing = self.transport.is_playing();
        if playing {
            ui.ctx().request_repaint_after(Duration::from_millis(33));
        }

        if self.song_path != self.titled_song_path {
            let file_name = Path::new(self.song_path.trim())
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .filter(|name| !name.is_empty())
                .unwrap_or_else(|| "Untitled".to_string());
            ui.ctx()
                .send_viewport_cmd(egui::ViewportCommand::Title(format!(
                    "simple-daw — {file_name}"
                )));
            self.titled_song_path = self.song_path.clone();
        }

        let mut song_guard = self.song.lock().unwrap();
        // A plain `&mut Song` (rather than the `MutexGuard` itself) so the
        // borrow checker can see `tracks` and `next_note_id` as disjoint
        // fields below — through the guard's `Deref` it can't.
        let song: &mut Song = &mut song_guard;

        // Apply any commands queued by the `simple-daw-mcp` companion binary since the last
        // frame, before anything below reads `song` — see `mcp_control` and `apply_mcp_command`.
        #[cfg(unix)]
        {
            mcp_control::set_repaint_context(ui.ctx().clone());
            while let Ok(req) = self.mcp_rx.try_recv() {
                let engine_config = self.engine.as_ref().ok().map(|e| {
                    (
                        e.status.sample_rate as f64,
                        e.status.min_frames,
                        e.status.max_frames,
                    )
                });
                let mut mcp_ctx = McpContext {
                    transport: &self.transport,
                    sample_rate: self.sample_rate,
                    engine_config,
                    song_path: &mut self.song_path,
                    master_effect_paths: &mut self.master_effect_paths,
                    master_effect_slots: &self.master_effect_slots,
                    master_effect_instances: &mut self.master_effect_instances,
                    master_effect_guis: &mut self.master_effect_guis,
                    master_effect_messages: &mut self.master_effect_messages,
                    track_effect_slots: &self.track_effect_slots,
                    track_effect_instances: &mut self.track_effect_instances,
                    track_effect_guis: &mut self.track_effect_guis,
                    track_effect_paths: &mut self.track_effect_paths,
                    track_effect_messages: &mut self.track_effect_messages,
                    send_effect_slots: &self.send_effect_slots,
                    send_effect_instances: &mut self.send_effect_instances,
                    send_effect_guis: &mut self.send_effect_guis,
                    send_effect_paths: &mut self.send_effect_paths,
                    send_effect_messages: &mut self.send_effect_messages,
                    submix_effect_slots: &self.submix_effect_slots,
                    submix_effect_instances: &mut self.submix_effect_instances,
                    submix_effect_guis: &mut self.submix_effect_guis,
                    submix_effect_paths: &mut self.submix_effect_paths,
                    submix_effect_messages: &mut self.submix_effect_messages,
                    track_meters: &self.track_meters,
                    submix_meters: &self.submix_meters,
                };
                let result = apply_mcp_command(&req.cmd, req.params, song, &mut mcp_ctx);
                let _ = req.reply.send(result);
            }
        }

        egui::Panel::top("menu_bar")
            .frame(
                egui::Frame::new()
                    .fill(egui::Color32::from_rgb(24, 24, 24))
                    .inner_margin(egui::Margin::symmetric(10, 4))
                    .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(12, 12, 12))),
            )
            .show(ui, |ui| {
                egui::MenuBar::new().ui(ui, |ui| {
                    ui.label(
                        egui::RichText::new("simple-daw")
                            .strong()
                            .size(14.0)
                            .color(FL_ACCENT_GREEN),
                    );
                    ui.add_space(12.0);
                    ui.menu_button("File", |ui| {
                        let path_set = !self.song_path.trim().is_empty();

                        if ui.button("Load…").clicked() {
                            ui.close();
                            if let Some(path) =
                                browse_for_file(&self.song_path, "Song (JSON)", &["json"], None)
                            {
                                self.song_path = path;
                                match perform_load(&self.song_path, self.sample_rate) {
                                    Ok(loaded) => {
                                        let track_count = loaded.tracks.len();
                                        let send_count = loaded.sends.len();
                                        let submix_count = loaded.submixes.len();
                                        let master_effect_specs = loaded.master_effects.clone();
                                        let track_effect_specs: Vec<Vec<TrackEffectConfig>> =
                                            loaded
                                                .tracks
                                                .iter()
                                                .map(|t| t.effects.clone())
                                                .collect();
                                        let send_effect_specs: Vec<Vec<TrackEffectConfig>> =
                                            loaded
                                                .sends
                                                .iter()
                                                .map(|s| s.effects.clone())
                                                .collect();
                                        let submix_effect_specs: Vec<Vec<TrackEffectConfig>> =
                                            loaded
                                                .submixes
                                                .iter()
                                                .map(|s| s.effects.clone())
                                                .collect();
                                        *song = loaded;
                                        self.transport.set_playing(false);
                                        self.piano_roll_drag = None;
                                        self.selected_notes.clear();
                                        self.effect_editor = None;
                                        // Region indices from the old song don't carry over — close
                                        // whichever Piano Roll/Beats window was open, if any; re-open by
                                        // double-clicking a region in the loaded song's Playlist.
                                        self.selected_track = None;
                                        self.piano_roll_region = None;
                                        self.selected_beats_track = None;
                                        self.beats_region = None;
                                        resize_track_effects(
                                            &self.track_effect_slots,
                                            &mut self.track_effect_instances,
                                            &mut self.track_effect_guis,
                                            &mut self.track_effect_paths,
                                            &mut self.track_effect_messages,
                                            track_count,
                                        );
                                        resize_track_meters(&self.track_meters, track_count);
                                        resize_track_effects(
                                            &self.send_effect_slots,
                                            &mut self.send_effect_instances,
                                            &mut self.send_effect_guis,
                                            &mut self.send_effect_paths,
                                            &mut self.send_effect_messages,
                                            send_count,
                                        );
                                        resize_track_effects(
                                            &self.submix_effect_slots,
                                            &mut self.submix_effect_instances,
                                            &mut self.submix_effect_guis,
                                            &mut self.submix_effect_paths,
                                            &mut self.submix_effect_messages,
                                            submix_count,
                                        );
                                        resize_track_meters(&self.submix_meters, submix_count);
                                        let engine_config = self.engine.as_ref().ok().map(|e| {
                                            (
                                                e.status.sample_rate as f64,
                                                e.status.min_frames,
                                                e.status.max_frames,
                                            )
                                        });
                                        apply_loaded_effects(
                                            &mut self.master_effect_paths,
                                            &mut self.master_effect_instances,
                                            &mut self.master_effect_guis,
                                            &self.master_effect_slots,
                                            &mut self.master_effect_messages,
                                            master_effect_specs,
                                            &mut self.track_effect_paths,
                                            &mut self.track_effect_instances,
                                            &mut self.track_effect_guis,
                                            &mut self.track_effect_messages,
                                            &self.track_effect_slots,
                                            track_effect_specs,
                                            &mut self.send_effect_paths,
                                            &mut self.send_effect_instances,
                                            &mut self.send_effect_guis,
                                            &mut self.send_effect_messages,
                                            &self.send_effect_slots,
                                            send_effect_specs,
                                            &mut self.submix_effect_paths,
                                            &mut self.submix_effect_instances,
                                            &mut self.submix_effect_guis,
                                            &mut self.submix_effect_messages,
                                            &self.submix_effect_slots,
                                            submix_effect_specs,
                                            engine_config,
                                        );
                                        self.song_message = Some((
                                            true,
                                            format!("Loaded {}", self.song_path.trim()),
                                        ));
                                    }
                                    Err(err) => self.song_message = Some((false, err)),
                                }
                            }
                        }
                        if ui
                            .add_enabled(path_set, egui::Button::new("Save"))
                            .clicked()
                        {
                            sync_song_effects(
                                song,
                                &self.master_effect_paths,
                                &self.master_effect_slots,
                                &self.track_effect_paths,
                                &self.track_effect_slots,
                                &self.send_effect_paths,
                                &self.send_effect_slots,
                                &self.submix_effect_paths,
                                &self.submix_effect_slots,
                            );
                            self.song_message = Some(perform_save(song, &self.song_path));
                            ui.close();
                        }
                        if ui.button("Save As…").clicked() {
                            self.save_as_path = self.song_path.clone();
                            self.show_save_as = true;
                            ui.close();
                        }

                        ui.separator();

                        if ui.button("Import MIDI…").clicked() {
                            self.show_import_midi = true;
                            ui.close();
                        }
                        if ui.button("Detect Tempo…").clicked() {
                            self.show_detect_tempo = true;
                            ui.close();
                        }

                        ui.separator();

                        if ui.button("Export…").clicked() {
                            self.show_export_dialog = true;
                            ui.close();
                        }
                    });
                    ui.add_space(6.0);
                    if ui.button("Plugins").clicked() {
                        self.show_plugins_panel = true;
                    }
                });
            });

        egui::Panel::top("toolbar")
            .frame(
                egui::Frame::new()
                    .fill(egui::Color32::from_rgb(30, 30, 30))
                    .inner_margin(egui::Margin::symmetric(10, 6))
                    .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(12, 12, 12))),
            )
            .show(ui, |ui| {
                ui.columns(3, |columns| {
                    columns[0].horizontal(|ui| {
                        toolbar_group(ui, |ui| {
                            let play_button = egui::Button::new(
                                egui::RichText::new(if playing { "⏹" } else { "▶" }).size(18.0),
                            )
                            .fill(if playing {
                                FL_ACCENT_GREEN
                            } else {
                                ui.visuals().widgets.inactive.bg_fill
                            })
                            .min_size(egui::vec2(36.0, 30.0));
                            if ui.add(play_button).clicked() {
                                self.transport.set_playing(!playing);
                            }

                            ui.add_space(6.0);
                            let is_recording = self.recording.is_some();
                            let record_enabled = is_recording
                                || self.record_armed_track.is_some_and(|i| {
                                    song.tracks
                                        .get(i)
                                        .is_some_and(|t| t.kind == TrackKind::Audio)
                                });
                            let record_button =
                                egui::Button::new(egui::RichText::new("⏺").size(18.0))
                                    .fill(if is_recording {
                                        FL_ACCENT_ORANGE
                                    } else {
                                        ui.visuals().widgets.inactive.bg_fill
                                    })
                                    .min_size(egui::vec2(36.0, 30.0));
                            let record_response = ui
                                .add_enabled(record_enabled, record_button)
                                .on_hover_text(if is_recording {
                                    "Stop recording"
                                } else {
                                    "Record the armed audio track"
                                });
                            if record_response.clicked() {
                                if let Some(session) = self.recording.take() {
                                    self.transport.set_playing(false);
                                    let RecordingSession {
                                        track_index,
                                        recorder,
                                        start_tick,
                                    } = session;
                                    let captured_sample_rate = recorder.sample_rate;
                                    let samples = recorder.stop();
                                    self.recording_message = Some(finish_recording(
                                        song,
                                        track_index,
                                        start_tick,
                                        &samples,
                                        captured_sample_rate,
                                        self.sample_rate,
                                    ));
                                } else if let Some(track_index) = self.record_armed_track {
                                    let input_gain = song
                                        .tracks
                                        .get(track_index)
                                        .map_or(1.0, |t| t.input_gain);
                                    match audio_input::InputRecorder::start(
                                        self.selected_input_device.as_deref(),
                                        input_gain,
                                    ) {
                                        Ok(recorder) => {
                                            self.transport.set_playing(true);
                                            let start_tick = self.transport.current_tick();
                                            self.recording = Some(RecordingSession {
                                                track_index,
                                                recorder,
                                                start_tick,
                                            });
                                            self.recording_message = None;
                                        }
                                        Err(err) => {
                                            self.recording_message =
                                                Some((false, format!("{err:#}")))
                                        }
                                    }
                                }
                            }
                            if is_recording {
                                let level =
                                    self.recording.as_ref().map_or(0.0, |s| s.recorder.level());
                                let (meter_rect, _) = ui.allocate_exact_size(
                                    egui::vec2(46.0, 10.0),
                                    egui::Sense::hover(),
                                );
                                ui.painter().rect_filled(
                                    meter_rect,
                                    2.0,
                                    ui.visuals().extreme_bg_color,
                                );
                                let mut filled_rect = meter_rect;
                                filled_rect.set_width(meter_rect.width() * level.clamp(0.0, 1.0));
                                ui.painter().rect_filled(filled_rect, 2.0, FL_ACCENT_ORANGE);
                            }
                            if let Some((ok, message)) = &self.recording_message {
                                let color = if *ok {
                                    FL_ACCENT_GREEN
                                } else {
                                    egui::Color32::RED
                                };
                                ui.colored_label(color, message);
                            }

                            ui.add_space(6.0);
                            let metronome_enabled = self.transport.is_metronome_enabled();
                            let metronome_button =
                                egui::Button::new(egui::RichText::new("🔔").size(18.0))
                                    .fill(if metronome_enabled {
                                        FL_ACCENT_GREEN
                                    } else {
                                        ui.visuals().widgets.inactive.bg_fill
                                    })
                                    .min_size(egui::vec2(36.0, 30.0));
                            if ui
                                .add(metronome_button)
                                .on_hover_text("Metronome")
                                .clicked()
                            {
                                self.transport.set_metronome_enabled(!metronome_enabled);
                            }
                        });

                        ui.add_space(8.0);
                        toolbar_group(ui, |ui| {
                            if ui
                                .selectable_label(self.playlist_open, "🎵 Playlist")
                                .clicked()
                            {
                                self.playlist_open = !self.playlist_open;
                                if self.playlist_open {
                                    self.session_view_open = false;
                                }
                            }
                            // Session View and the Playlist both dock into the central area (see
                            // `session_view_open`'s doc comment) — clicking one closes the other,
                            // the same "one central-area view at a time" idea Ableton's own
                            // Session/Arrangement tab switcher uses.
                            if ui
                                .selectable_label(self.session_view_open, "🎛 Session")
                                .clicked()
                            {
                                self.session_view_open = !self.session_view_open;
                                if self.session_view_open {
                                    self.playlist_open = false;
                                }
                            }
                            if ui.selectable_label(self.mixer_open, "🎚 Mixer").clicked() {
                                self.mixer_open = !self.mixer_open;
                            }
                        });
                    });

                    columns[1].with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
                        transport_lcd_ui(
                            ui,
                            self.transport.current_tick(),
                            song,
                            &mut self.tap_tempo,
                        );
                    });

                    columns[2].horizontal(|ui| {
                        toolbar_group(ui, |ui| {
                            let mut restart_output = false;
                            egui::ComboBox::from_id_salt("output_device_picker")
                                .selected_text(
                                    self.selected_output_device.as_deref().unwrap_or("Default"),
                                )
                                .show_ui(ui, |ui| {
                                    if ui
                                        .selectable_label(
                                            self.selected_output_device.is_none(),
                                            "Default",
                                        )
                                        .clicked()
                                        && self.selected_output_device.is_some()
                                    {
                                        self.selected_output_device = None;
                                        self.selected_output_sample_rate = None;
                                        restart_output = true;
                                    }
                                    for name in audio::list_output_devices() {
                                        let selected = self.selected_output_device.as_deref()
                                            == Some(name.as_str());
                                        if ui.selectable_label(selected, &name).clicked()
                                            && !selected
                                        {
                                            self.selected_output_device = Some(name);
                                            self.selected_output_sample_rate = None;
                                            restart_output = true;
                                        }
                                    }
                                });
                            egui::ComboBox::from_id_salt("output_sample_rate_picker")
                                .selected_text(
                                    self.selected_output_sample_rate
                                        .map(|rate| format!("{rate} Hz"))
                                        .unwrap_or_else(|| "Default".to_string()),
                                )
                                .show_ui(ui, |ui| {
                                    if ui
                                        .selectable_label(
                                            self.selected_output_sample_rate.is_none(),
                                            "Default",
                                        )
                                        .clicked()
                                        && self.selected_output_sample_rate.is_some()
                                    {
                                        self.selected_output_sample_rate = None;
                                        restart_output = true;
                                    }
                                    for rate in audio::list_output_sample_rates(
                                        self.selected_output_device.as_deref(),
                                    ) {
                                        let selected =
                                            self.selected_output_sample_rate == Some(rate);
                                        if ui
                                            .selectable_label(selected, format!("{rate} Hz"))
                                            .clicked()
                                            && !selected
                                        {
                                            self.selected_output_sample_rate = Some(rate);
                                            restart_output = true;
                                        }
                                    }
                                });

                            // A free-standing block rather than a `&mut self` method: `song` here is
                            // reborrowed from `self.song.lock()` (held for this whole `ui` call), and a
                            // `&mut self` method call would conflict with that outstanding borrow of the
                            // `self.song` field. Operating on `self.<field>` directly alongside `song`
                            // instead (same pattern the Load-song flow above already uses) borrows only
                            // the disjoint fields actually touched, which the borrow checker allows.
                            if restart_output {
                                sync_song_effects(
                                    song,
                                    &self.master_effect_paths,
                                    &self.master_effect_slots,
                                    &self.track_effect_paths,
                                    &self.track_effect_slots,
                                    &self.send_effect_paths,
                                    &self.send_effect_slots,
                                    &self.submix_effect_paths,
                                    &self.submix_effect_slots,
                                );
                                // The old engine (if any) is kept alive until the new one succeeds, so a
                                // bad device/rate doesn't leave the app silent.
                                match AudioEngine::start(
                                    self.song.clone(),
                                    self.transport.clone(),
                                    self.master_effect_slots.clone(),
                                    self.track_effect_slots.clone(),
                                    self.send_effect_slots.clone(),
                                    self.submix_effect_slots.clone(),
                                    self.track_meters.clone(),
                                    self.master_meter.clone(),
                                    self.submix_meters.clone(),
                                    self.session_slots.clone(),
                                    self.selected_output_device.as_deref(),
                                    self.selected_output_sample_rate,
                                ) {
                                    Ok(engine) => {
                                        let sample_rate = engine.status.sample_rate;
                                        let engine_config = Some((
                                            sample_rate as f64,
                                            engine.status.min_frames,
                                            engine.status.max_frames,
                                        ));
                                        self.engine = Ok(engine);
                                        self.sample_rate = Some(sample_rate);
                                        song.reload_samples(sample_rate);
                                        apply_loaded_effects(
                                            &mut self.master_effect_paths,
                                            &mut self.master_effect_instances,
                                            &mut self.master_effect_guis,
                                            &self.master_effect_slots,
                                            &mut self.master_effect_messages,
                                            song.master_effects.clone(),
                                            &mut self.track_effect_paths,
                                            &mut self.track_effect_instances,
                                            &mut self.track_effect_guis,
                                            &mut self.track_effect_messages,
                                            &self.track_effect_slots,
                                            song.tracks.iter().map(|t| t.effects.clone()).collect(),
                                            &mut self.send_effect_paths,
                                            &mut self.send_effect_instances,
                                            &mut self.send_effect_guis,
                                            &mut self.send_effect_messages,
                                            &self.send_effect_slots,
                                            song.sends.iter().map(|s| s.effects.clone()).collect(),
                                            &mut self.submix_effect_paths,
                                            &mut self.submix_effect_instances,
                                            &mut self.submix_effect_guis,
                                            &mut self.submix_effect_messages,
                                            &self.submix_effect_slots,
                                            song.submixes.iter().map(|s| s.effects.clone()).collect(),
                                            engine_config,
                                        );
                                        self.output_device_message = None;
                                    }
                                    Err(err) => {
                                        self.output_device_message =
                                            Some((false, format!("{err:#}")));
                                    }
                                }
                            }
                            match &self.engine {
                                Ok(engine) => {
                                    ui.weak(format!(
                                        "{} · {} Hz",
                                        engine.status.device_name, engine.status.sample_rate
                                    ));
                                }
                                Err(err) => {
                                    ui.colored_label(
                                        FL_ACCENT_ORANGE,
                                        format!("Audio engine error: {err:#}"),
                                    );
                                }
                            }
                            if let Some((ok, message)) = &self.output_device_message {
                                let color = if *ok {
                                    FL_ACCENT_GREEN
                                } else {
                                    egui::Color32::RED
                                };
                                ui.colored_label(color, message);
                            }
                        });

                        ui.add_space(8.0);
                        toolbar_group(ui, |ui| {
                            ui.weak(format!("Song: {}", self.song_path));
                            if let Some((ok, message)) = &self.song_message {
                                let color = if *ok {
                                    FL_ACCENT_GREEN
                                } else {
                                    egui::Color32::RED
                                };
                                ui.colored_label(color, message);
                            }
                        });
                    });
                });
            });

        if self.show_save_as {
            let mut open = true;
            egui::Window::new("Save As")
                .collapsible(false)
                .resizable(false)
                .open(&mut open)
                .show(ui.ctx(), |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Path:");
                        ui.add_sized(
                            [240.0, 22.0],
                            egui::TextEdit::singleline(&mut self.save_as_path)
                                .hint_text("song.json"),
                        );
                        if ui.button("Browse…").clicked() {
                            if let Some(path) = browse_for_file(
                                &self.save_as_path,
                                "Song (JSON)",
                                &["json"],
                                Some("song.json"),
                            ) {
                                self.save_as_path = path;
                            }
                        }
                    });
                    ui.horizontal(|ui| {
                        let can_save = !self.save_as_path.trim().is_empty();
                        if ui
                            .add_enabled(can_save, egui::Button::new("Save"))
                            .clicked()
                        {
                            self.song_path = self.save_as_path.clone();
                            sync_song_effects(
                                song,
                                &self.master_effect_paths,
                                &self.master_effect_slots,
                                &self.track_effect_paths,
                                &self.track_effect_slots,
                                &self.send_effect_paths,
                                &self.send_effect_slots,
                                &self.submix_effect_paths,
                                &self.submix_effect_slots,
                            );
                            self.song_message = Some(perform_save(song, &self.song_path));
                            self.show_save_as = false;
                        }
                        if ui.button("Cancel").clicked() {
                            self.show_save_as = false;
                        }
                    });
                });
            if !open {
                self.show_save_as = false;
            }
        }

        if self.show_import_midi {
            let mut open = true;
            egui::Window::new("Import MIDI")
                .collapsible(false)
                .resizable(false)
                .open(&mut open)
                .show(ui.ctx(), |ui| {
                    ui.horizontal(|ui| {
                        ui.label("File:");
                        ui.add_sized(
                            [240.0, 22.0],
                            egui::TextEdit::singleline(&mut self.import_midi_path)
                                .hint_text("song.mid"),
                        );
                        if ui.button("Browse…").clicked() {
                            if let Some(path) = browse_for_file(
                                &self.import_midi_path,
                                "MIDI",
                                &["mid", "midi"],
                                None,
                            ) {
                                self.import_midi_path = path;
                            }
                        }
                    });
                    ui.checkbox(
                        &mut self.import_midi_apply_bpm,
                        "Also set song tempo (BPM) from the file, if it has one",
                    );
                    ui.horizontal(|ui| {
                        let can_import = !self.import_midi_path.trim().is_empty();
                        if ui
                            .add_enabled(can_import, egui::Button::new("Import"))
                            .clicked()
                        {
                            let path = std::path::Path::new(self.import_midi_path.trim());
                            let steps_per_bar = song.steps_per_bar();
                            match midi_import::import_midi_file(
                                path,
                                &mut song.next_note_id,
                                steps_per_bar,
                            ) {
                                Ok(imported) => {
                                    let added = imported.tracks.len();
                                    for imported_track in imported.tracks {
                                        let track_index = song.add_track(
                                            imported_track.name,
                                            imported_track.midi_channel,
                                            TrackKind::PianoRoll,
                                        );
                                        let length_steps =
                                            imported_track.length_steps.max(steps_per_bar);
                                        song.tracks[track_index].regions.push(Region {
                                            name: "Imported".to_string(),
                                            start_tick: 0,
                                            content_length_steps: length_steps,
                                            loop_length_steps: length_steps,
                                            content: RegionContent::PianoRoll(imported_track.notes),
                                            fade_in_ticks: 0,
                                            fade_out_ticks: 0,
                                            automation: Vec::new(),
                                        });
                                    }
                                    resize_track_effects(
                                        &self.track_effect_slots,
                                        &mut self.track_effect_instances,
                                        &mut self.track_effect_guis,
                                        &mut self.track_effect_paths,
                                        &mut self.track_effect_messages,
                                        song.tracks.len(),
                                    );
                                    resize_track_meters(&self.track_meters, song.tracks.len());
                                    let mut message = format!(
                                        "Imported {added} track(s) from {}",
                                        self.import_midi_path.trim()
                                    );
                                    if self.import_midi_apply_bpm {
                                        if let Some(bpm) = imported.detected_bpm {
                                            song.bpm = bpm;
                                            message
                                                .push_str(&format!(", set tempo to {bpm:.1} BPM"));
                                        }
                                    }
                                    self.import_midi_message = Some((true, message));
                                    self.show_import_midi = false;
                                }
                                Err(err) => {
                                    self.import_midi_message = Some((false, format!("{err:#}")));
                                }
                            }
                        }
                        if ui.button("Cancel").clicked() {
                            self.show_import_midi = false;
                        }
                    });
                    if let Some((ok, message)) = &self.import_midi_message {
                        let color = if *ok {
                            egui::Color32::from_rgb(120, 220, 140)
                        } else {
                            egui::Color32::RED
                        };
                        ui.colored_label(color, message);
                    }
                });
            if !open {
                self.show_import_midi = false;
            }
        }

        if self.show_detect_tempo {
            let mut open = true;
            egui::Window::new("Detect Tempo")
                .collapsible(false)
                .resizable(false)
                .open(&mut open)
                .show(ui.ctx(), |ui| {
                    ui.horizontal(|ui| {
                        ui.label("File:");
                        ui.add_sized(
                            [240.0, 22.0],
                            egui::TextEdit::singleline(&mut self.detect_tempo_path)
                                .hint_text("loop.wav"),
                        );
                        if ui.button("Browse…").clicked()
                            && let Some(path) = browse_for_file(
                                &self.detect_tempo_path,
                                "WAV audio",
                                &["wav"],
                                None,
                            )
                        {
                            self.detect_tempo_path = path;
                        }
                    });
                    ui.horizontal(|ui| {
                        let can_detect = !self.detect_tempo_path.trim().is_empty();
                        if ui
                            .add_enabled(can_detect, egui::Button::new("Detect"))
                            .clicked()
                        {
                            let path = std::path::Path::new(self.detect_tempo_path.trim());
                            match SampleBuffer::load_wav(path) {
                                Ok(buffer) => {
                                    match tempo_detection::detect_bpm(
                                        &buffer.mono,
                                        buffer.sample_rate,
                                    ) {
                                        Some(bpm) => {
                                            self.detect_tempo_bpm = Some(bpm);
                                            self.detect_tempo_message =
                                                Some((true, format!("Detected {bpm:.1} BPM")));
                                        }
                                        None => {
                                            self.detect_tempo_bpm = None;
                                            self.detect_tempo_message = Some((
                                                false,
                                                "Couldn't find a steady tempo in this file"
                                                    .to_string(),
                                            ));
                                        }
                                    }
                                }
                                Err(err) => {
                                    self.detect_tempo_bpm = None;
                                    self.detect_tempo_message = Some((false, format!("{err:#}")));
                                }
                            }
                        }
                        if let Some(bpm) = self.detect_tempo_bpm
                            && ui
                                .button(format!("Apply to Song Tempo ({bpm:.1} BPM)"))
                                .clicked()
                        {
                            song.bpm = bpm;
                        }
                        if ui.button("Cancel").clicked() {
                            self.show_detect_tempo = false;
                        }
                    });
                    if let Some((ok, message)) = &self.detect_tempo_message {
                        let color = if *ok {
                            egui::Color32::from_rgb(120, 220, 140)
                        } else {
                            egui::Color32::RED
                        };
                        ui.colored_label(color, message);
                    }
                });
            if !open {
                self.show_detect_tempo = false;
            }
        }

        if self.show_plugins_panel {
            let mut open = true;
            let engine_config = self.engine.as_ref().ok().map(|e| {
                (
                    e.status.sample_rate as f64,
                    e.status.min_frames,
                    e.status.max_frames,
                )
            });
            egui::Window::new("Plugins")
                .collapsible(false)
                .resizable(true)
                .open(&mut open)
                .show(ui.ctx(), |ui| {
                    ui.weak("Project plugin library — import a CLAP effect once, then load it onto the master bus or any track's FX chain.");
                    ui.add_space(4.0);
                    if ui.button("Import Plugin…").clicked() {
                        if let Some(path) = browse_for_file("", "CLAP Plugin", &["clap"], None) {
                            if !song.plugins.iter().any(|p| p.path == path) {
                                let name = Path::new(&path)
                                    .file_stem()
                                    .map(|n| n.to_string_lossy().to_string())
                                    .unwrap_or_else(|| path.clone());
                                song.plugins.push(ProjectPlugin { name, path });
                            }
                        }
                    }
                    ui.separator();

                    let mut plugin_to_remove: Option<usize> = None;
                    egui::ScrollArea::vertical().max_height(150.0).show(ui, |ui| {
                        for (index, plugin) in song.plugins.iter_mut().enumerate() {
                            ui.horizontal(|ui| {
                                ui.add_sized(
                                    [110.0, 20.0],
                                    egui::TextEdit::singleline(&mut plugin.name).hint_text("Name"),
                                )
                                .on_hover_text(plugin.path.as_str());
                                if ui
                                    .small_button("🗑")
                                    .on_hover_text("Remove from the project plugin list")
                                    .clicked()
                                {
                                    plugin_to_remove = Some(index);
                                }
                            });
                        }
                    });
                    if let Some(index) = plugin_to_remove {
                        song.plugins.remove(index);
                    }

                    ui.separator();
                    ui.strong("Master bus FX chain");
                    // Unused by `fx_chain_ui` itself (only `channel_rack_row_ui`'s own buttons
                    // touch these) — the master bus has no synth to open and can't be deleted, but
                    // `TrackFxUi` needs somewhere to point since it's shared with tracks.
                    let mut unused_device_chain_focus: Option<DeviceChainFocus> = None;
                    let mut unused_remove_requested: Option<usize> = None;
                    let track_names: Vec<String> =
                        song.tracks.iter().map(|t| t.name.clone()).collect();
                    let mut master_fx = TrackFxUi {
                        track_index: 0,
                        chain_kind: FxChainKind::Master,
                        paths: &mut self.master_effect_paths,
                        messages: &mut self.master_effect_messages,
                        slots: self.master_effect_slots.clone(),
                        instances: &mut self.master_effect_instances,
                        guis: &mut self.master_effect_guis,
                        engine_config,
                        known_plugins: &song.plugins,
                        track_names: &track_names,
                        editor: &mut self.effect_editor,
                        device_chain_focus: &mut unused_device_chain_focus,
                        remove_requested: &mut unused_remove_requested,
                        inline_params: false,
                    };
                    fx_chain_ui(ui, &mut master_fx);
                });
            if !open {
                self.show_plugins_panel = false;
            }
        }

        if self.show_export_dialog {
            let mut open = true;
            egui::Window::new("Export")
                .collapsible(false)
                .resizable(false)
                .open(&mut open)
                .show(ui.ctx(), |ui| {
                    ui.horizontal(|ui| {
                        ui.label("File:");
                        ui.add_sized(
                            [200.0, 22.0],
                            egui::TextEdit::singleline(&mut self.export_path)
                                .hint_text("output.wav"),
                        );
                        ui.label("loops:");
                        ui.add(egui::DragValue::new(&mut self.export_loops).range(1..=32));
                    });
                    ui.horizontal(|ui| {
                        let can_export =
                            self.sample_rate.is_some() && !self.export_path.trim().is_empty();
                        if ui
                            .add_enabled(can_export, egui::Button::new("Export"))
                            .clicked()
                        {
                            if let Some(rate) = self.sample_rate {
                                self.export_message = Some(perform_export(
                                    song,
                                    rate,
                                    self.export_loops,
                                    &self.export_path,
                                ));
                            }
                        }
                        if ui.button("Close").clicked() {
                            self.show_export_dialog = false;
                        }
                    });
                    if let Some((ok, message)) = &self.export_message {
                        let color = if *ok {
                            egui::Color32::from_rgb(120, 220, 140)
                        } else {
                            egui::Color32::RED
                        };
                        ui.colored_label(color, message);
                    }
                });
            if !open {
                self.show_export_dialog = false;
            }
        }

        if let Some(target) = self.effect_editor {
            // Only one embedded GUI panel can exist at a time (there's only ever one "FX Params"
            // window) — if a *different* slot's embedded GUI is still marked active (e.g. the
            // user clicked a different slot's "Params" button while a GUI was open), close it
            // before this window opens its own, so its container view doesn't linger unpositioned.
            if let Some(other) = self.active_embedded_gui
                && other != target
            {
                close_effect_gui(
                    other,
                    &mut self.master_effect_instances,
                    &mut self.master_effect_guis,
                    &mut self.track_effect_instances,
                    &mut self.track_effect_guis,
                    &mut self.send_effect_instances,
                    &mut self.send_effect_guis,
                    &mut self.submix_effect_instances,
                    &mut self.submix_effect_guis,
                );
                self.active_embedded_gui = None;
            }

            let embed_target = frame.window_handle().ok().map(|handle| {
                let scale_factor = frame.winit_window().map(|w| w.scale_factor()).unwrap_or(1.0);
                (handle.as_raw(), scale_factor)
            });

            let title = match target {
                EffectEditorTarget::Master(slot_index) => {
                    format!("Master FX {} Params", slot_index + 1)
                }
                EffectEditorTarget::Track(track_index, slot_index) => {
                    format!("Track {} FX {} Params", track_index + 1, slot_index + 1)
                }
                EffectEditorTarget::Send(send_index, slot_index) => {
                    format!("Send {} FX {} Params", send_index + 1, slot_index + 1)
                }
                EffectEditorTarget::Submix(submix_index, slot_index) => {
                    format!("Submix {} FX {} Params", submix_index + 1, slot_index + 1)
                }
            };
            let gui_title = title.clone();
            let mut open = true;
            // No live resize renegotiation with an embedded plugin GUI (see `plugin_host`'s doc
            // comment on `open_plugin_gui`'s embedded path) — disable dragging the window's own
            // edges while one is open, rather than let the user resize a panel nothing then keeps
            // in sync with the plugin.
            let resizable = self.active_embedded_gui != Some(target);
            egui::Window::new(title)
                .collapsible(false)
                .resizable(resizable)
                .open(&mut open)
                .show(ui.ctx(), |ui| match target {
                    EffectEditorTarget::Master(slot_index) => {
                        if let Ok(mut guard) = self.master_effect_slots.lock() {
                            let slot = guard
                                .first_mut()
                                .and_then(|chain| chain.get_mut(slot_index))
                                .and_then(|slot| slot.as_mut());
                            match slot {
                                Some(EffectInstance::Clap(effect)) => {
                                    effect_params_ui(ui, Some(effect))
                                }
                                Some(EffectInstance::BuiltIn(effect)) => {
                                    built_in_effect_params_ui(ui, effect)
                                }
                                None => effect_params_ui(ui, None),
                            }
                        }
                        if let (Some(instance), Some(gui)) = (
                            self.master_effect_instances
                                .get_mut(slot_index)
                                .and_then(|instance| instance.as_mut()),
                            self.master_effect_guis
                                .get_mut(slot_index)
                                .and_then(|gui| gui.as_mut()),
                        ) {
                            plugin_gui_button_ui(
                                ui,
                                instance,
                                gui,
                                &gui_title,
                                embed_target,
                                target,
                                &mut self.active_embedded_gui,
                            );
                        }
                    }
                    EffectEditorTarget::Track(track_index, slot_index) => {
                        if let Ok(mut guard) = self.track_effect_slots.lock() {
                            let slot = guard
                                .get_mut(track_index)
                                .and_then(|chain| chain.get_mut(slot_index))
                                .and_then(|slot| slot.as_mut());
                            match slot {
                                Some(EffectInstance::Clap(effect)) => {
                                    effect_params_ui(ui, Some(effect))
                                }
                                Some(EffectInstance::BuiltIn(effect)) => {
                                    built_in_effect_params_ui(ui, effect)
                                }
                                None => effect_params_ui(ui, None),
                            }
                        }
                        if let (Some(instance), Some(gui)) = (
                            self.track_effect_instances
                                .get_mut(track_index)
                                .and_then(|slots| slots.get_mut(slot_index))
                                .and_then(|instance| instance.as_mut()),
                            self.track_effect_guis
                                .get_mut(track_index)
                                .and_then(|slots| slots.get_mut(slot_index))
                                .and_then(|gui| gui.as_mut()),
                        ) {
                            plugin_gui_button_ui(
                                ui,
                                instance,
                                gui,
                                &gui_title,
                                embed_target,
                                target,
                                &mut self.active_embedded_gui,
                            );
                        }
                    }
                    EffectEditorTarget::Send(send_index, slot_index) => {
                        if let Ok(mut guard) = self.send_effect_slots.lock() {
                            let slot = guard
                                .get_mut(send_index)
                                .and_then(|chain| chain.get_mut(slot_index))
                                .and_then(|slot| slot.as_mut());
                            match slot {
                                Some(EffectInstance::Clap(effect)) => {
                                    effect_params_ui(ui, Some(effect))
                                }
                                Some(EffectInstance::BuiltIn(effect)) => {
                                    built_in_effect_params_ui(ui, effect)
                                }
                                None => effect_params_ui(ui, None),
                            }
                        }
                        if let (Some(instance), Some(gui)) = (
                            self.send_effect_instances
                                .get_mut(send_index)
                                .and_then(|slots| slots.get_mut(slot_index))
                                .and_then(|instance| instance.as_mut()),
                            self.send_effect_guis
                                .get_mut(send_index)
                                .and_then(|slots| slots.get_mut(slot_index))
                                .and_then(|gui| gui.as_mut()),
                        ) {
                            plugin_gui_button_ui(
                                ui,
                                instance,
                                gui,
                                &gui_title,
                                embed_target,
                                target,
                                &mut self.active_embedded_gui,
                            );
                        }
                    }
                    EffectEditorTarget::Submix(submix_index, slot_index) => {
                        if let Ok(mut guard) = self.submix_effect_slots.lock() {
                            let slot = guard
                                .get_mut(submix_index)
                                .and_then(|chain| chain.get_mut(slot_index))
                                .and_then(|slot| slot.as_mut());
                            match slot {
                                Some(EffectInstance::Clap(effect)) => {
                                    effect_params_ui(ui, Some(effect))
                                }
                                Some(EffectInstance::BuiltIn(effect)) => {
                                    built_in_effect_params_ui(ui, effect)
                                }
                                None => effect_params_ui(ui, None),
                            }
                        }
                        if let (Some(instance), Some(gui)) = (
                            self.submix_effect_instances
                                .get_mut(submix_index)
                                .and_then(|slots| slots.get_mut(slot_index))
                                .and_then(|instance| instance.as_mut()),
                            self.submix_effect_guis
                                .get_mut(submix_index)
                                .and_then(|slots| slots.get_mut(slot_index))
                                .and_then(|gui| gui.as_mut()),
                        ) {
                            plugin_gui_button_ui(
                                ui,
                                instance,
                                gui,
                                &gui_title,
                                embed_target,
                                target,
                                &mut self.active_embedded_gui,
                            );
                        }
                    }
                });
            if !open {
                // The window that reserved this embedded GUI's panel is gone — nothing keeps its
                // container view positioned anymore, so close it rather than leave it stranded.
                if self.active_embedded_gui == Some(target) {
                    close_effect_gui(
                        target,
                        &mut self.master_effect_instances,
                        &mut self.master_effect_guis,
                        &mut self.track_effect_instances,
                        &mut self.track_effect_guis,
                        &mut self.send_effect_instances,
                        &mut self.send_effect_guis,
                        &mut self.submix_effect_instances,
                        &mut self.submix_effect_guis,
                    );
                    self.active_embedded_gui = None;
                }
                self.effect_editor = None;
            }
        }

        take_folder_editor_window_ui(
            ui.ctx(),
            song,
            &mut self.take_folder_editor,
            &mut self.take_folder_comp_drag,
        );

        flex_editor_window_ui(
            ui.ctx(),
            song,
            self.sample_rate,
            &mut self.flex_editor,
            &mut self.flex_editor_mode,
            &mut self.flex_editor_raw,
            &mut self.flex_marker_drag,
            &mut self.flex_note_drag,
        );

        let current_tick = playing.then(|| self.transport.current_tick());
        let engine_config = self.engine.as_ref().ok().map(|e| {
            (
                e.status.sample_rate as f64,
                e.status.min_frames,
                e.status.max_frames,
            )
        });

        let mut track_to_remove: Option<usize> = None;
        let mut freeze_requested: Option<(usize, TrackFreezeAction)> = None;

        let channel_rack_frame = || {
            egui::Frame::new()
                .fill(egui::Color32::from_rgb(26, 26, 26))
                .inner_margin(egui::Margin::same(6))
                .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(12, 12, 12)))
        };
        let channel_rack_already_detached = self.channel_rack_detached;
        let mut rack = ChannelRackUi {
            selected_track: &self.selected_track,
            selected_beats_track: &self.selected_beats_track,
            detached: &mut self.channel_rack_detached,
            track_effect_slots: &self.track_effect_slots,
            track_effect_instances: &mut self.track_effect_instances,
            track_effect_guis: &mut self.track_effect_guis,
            track_effect_paths: &mut self.track_effect_paths,
            track_effect_messages: &mut self.track_effect_messages,
            track_meters: &self.track_meters,
            effect_editor: &mut self.effect_editor,
            device_chain_focus: &mut self.device_chain_focus,
            record_armed_track: &mut self.record_armed_track,
            selected_input_device: &mut self.selected_input_device,
        };

        if channel_rack_already_detached {
            let ctx = ui.ctx().clone();
            let mut still_open = true;
            ctx.show_viewport_immediate(
                egui::ViewportId::from_hash_of("channel_rack_viewport"),
                egui::ViewportBuilder::default()
                    .with_title("Channel Rack")
                    .with_inner_size(egui::vec2(420.0, 640.0)),
                |ui, _class| {
                    egui::CentralPanel::default()
                        .frame(channel_rack_frame())
                        .show(ui, |ui| {
                            channel_rack_contents_ui(
                                ui,
                                song,
                                engine_config,
                                &mut track_to_remove,
                                &mut freeze_requested,
                                &mut rack,
                            );
                        });
                    if ui.ctx().input(|i| i.viewport().close_requested()) {
                        still_open = false;
                    }
                },
            );
            if !still_open {
                *rack.detached = false;
            }
        } else {
            egui::Panel::left("channel_rack")
                .default_size(400.0)
                .size_range(260.0..=620.0)
                .frame(channel_rack_frame())
                .show(ui, |ui| {
                    channel_rack_contents_ui(
                        ui,
                        song,
                        engine_config,
                        &mut track_to_remove,
                        &mut freeze_requested,
                        &mut rack,
                    );
                });
        }

        if let Some(index) = track_to_remove {
            song.remove_track(index);
            self.piano_roll_drag = None;
            self.selected_notes.clear();
            if matches!(self.effect_editor, Some(EffectEditorTarget::Track(t, _)) if t == index) {
                self.effect_editor = None;
            }
            let focus_points_at_removed_track = match self.device_chain_focus {
                Some(DeviceChainFocus::Track(t)) => t == index,
                Some(DeviceChainFocus::Lane { track_index, .. }) => track_index == index,
                None => false,
            };
            if focus_points_at_removed_track {
                self.device_chain_focus = None;
            }
            remove_track_effects(
                &self.track_effect_slots,
                &mut self.track_effect_instances,
                &mut self.track_effect_guis,
                &mut self.track_effect_paths,
                &mut self.track_effect_messages,
                index,
            );
            remove_track_meter(&self.track_meters, index);
            // The removed track shifts every later index down by one; if it was the selected
            // one, fall back to the next available piano-roll track (or none).
            self.selected_track = match self.selected_track {
                Some(t) if t == index => song
                    .tracks
                    .iter()
                    .position(|tr| tr.kind == TrackKind::PianoRoll),
                Some(t) if t > index => Some(t - 1),
                other => other,
            };
            self.selected_beats_track = match self.selected_beats_track {
                Some(t) if t == index => song
                    .tracks
                    .iter()
                    .position(|tr| tr.kind == TrackKind::StepGrid),
                Some(t) if t > index => Some(t - 1),
                other => other,
            };
            self.record_armed_track = match self.record_armed_track {
                Some(t) if t == index => None,
                Some(t) if t > index => Some(t - 1),
                other => other,
            };
        }

        if let Some((track_index, action)) = freeze_requested {
            let engine_sample_rate = engine_config.map(|(sr, _, _)| sr as u32);
            let render_sample_rate = engine_sample_rate.unwrap_or(48_000);
            let (ok, message) = match action {
                TrackFreezeAction::Freeze => {
                    freeze_track(song, track_index, render_sample_rate, engine_sample_rate)
                }
                TrackFreezeAction::Unfreeze => {
                    unfreeze_track(song, track_index);
                    (true, "Unfroze track".to_string())
                }
                TrackFreezeAction::Bounce => {
                    let result = bounce_track_in_place(
                        song,
                        track_index,
                        render_sample_rate,
                        engine_sample_rate,
                    );
                    // `bounce_track_in_place` cleared `track.effects` in the model (its processing
                    // is already baked into the new clip) — the *runtime* chain still holds the
                    // old plugin instances until resynced, which would otherwise double-process
                    // the now-already-wet bounced audio.
                    if result.0 {
                        apply_chain_specs_at(
                            track_index,
                            Vec::new(),
                            engine_config,
                            &self.track_effect_slots,
                            &mut self.track_effect_paths,
                            &mut self.track_effect_instances,
                            &mut self.track_effect_guis,
                            &mut self.track_effect_messages,
                        );
                    }
                    result
                }
            };
            if !ok {
                eprintln!("freeze/bounce failed for track {track_index}: {message}");
            }
        }

        if self.mixer_open {
            let mixer_frame = || {
                egui::Frame::new()
                    .fill(egui::Color32::from_rgb(26, 26, 26))
                    .inner_margin(egui::Margin::same(6))
                    .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(12, 12, 12)))
            };
            let mixer_already_detached = self.mixer_detached;
            let mut mixer_ui_state = MixerUi {
                detached: &mut self.mixer_detached,
                track_effect_slots: &self.track_effect_slots,
                track_effect_instances: &mut self.track_effect_instances,
                track_effect_guis: &mut self.track_effect_guis,
                track_effect_paths: &mut self.track_effect_paths,
                track_effect_messages: &mut self.track_effect_messages,
                track_meters: &self.track_meters,
                effect_editor: &mut self.effect_editor,
                master_effect_paths: &mut self.master_effect_paths,
                master_effect_slots: self.master_effect_slots.clone(),
                master_effect_instances: &mut self.master_effect_instances,
                master_effect_guis: &mut self.master_effect_guis,
                master_effect_messages: &mut self.master_effect_messages,
                master_meter: &self.master_meter,
                send_effect_slots: &self.send_effect_slots,
                send_effect_instances: &mut self.send_effect_instances,
                send_effect_guis: &mut self.send_effect_guis,
                send_effect_paths: &mut self.send_effect_paths,
                send_effect_messages: &mut self.send_effect_messages,
                submix_effect_slots: &self.submix_effect_slots,
                submix_effect_instances: &mut self.submix_effect_instances,
                submix_effect_guis: &mut self.submix_effect_guis,
                submix_effect_paths: &mut self.submix_effect_paths,
                submix_effect_messages: &mut self.submix_effect_messages,
                submix_meters: &self.submix_meters,
            };

            if mixer_already_detached {
                let ctx = ui.ctx().clone();
                let mut still_open = true;
                ctx.show_viewport_immediate(
                    egui::ViewportId::from_hash_of("mixer_viewport"),
                    egui::ViewportBuilder::default()
                        .with_title("Mixer")
                        .with_inner_size(egui::vec2(900.0, 320.0)),
                    |ui, _class| {
                        egui::CentralPanel::default().frame(mixer_frame()).show(ui, |ui| {
                            mixer_contents_ui(ui, song, engine_config, &mut mixer_ui_state);
                        });
                        if ui.ctx().input(|i| i.viewport().close_requested()) {
                            still_open = false;
                        }
                    },
                );
                if !still_open {
                    self.mixer_open = false;
                }
            } else {
                egui::Panel::bottom("mixer")
                    .default_size(220.0)
                    .size_range(160.0..=420.0)
                    .frame(mixer_frame())
                    .show(ui, |ui| {
                        mixer_contents_ui(ui, song, engine_config, &mut mixer_ui_state);
                    });
            }
        }

        let device_panel_frame = || {
            egui::Frame::new()
                .fill(egui::Color32::from_rgb(30, 30, 30))
                .inner_margin(egui::Margin::same(8))
                .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(12, 12, 12)))
        };
        let mut device_panel = DevicePanelUi {
            focus: &mut self.device_chain_focus,
            track_effect_slots: &self.track_effect_slots,
            track_effect_instances: &mut self.track_effect_instances,
            track_effect_guis: &mut self.track_effect_guis,
            track_effect_paths: &mut self.track_effect_paths,
            track_effect_messages: &mut self.track_effect_messages,
            effect_editor: &mut self.effect_editor,
            new_preset_name: &mut self.new_preset_name,
            preset_message: &mut self.preset_message,
        };
        egui::Panel::bottom("device_panel")
            .resizable(true)
            .default_size(220.0)
            .size_range(120.0..=480.0)
            .frame(device_panel_frame())
            .show(ui, |ui| {
                ui.heading("Device Panel");
                ui.separator();
                egui::ScrollArea::vertical().show(ui, |ui| {
                    device_panel_contents_ui(ui, song, engine_config, &mut device_panel);
                });
            });

        let piano_roll_open = self
            .selected_track
            .filter(|&i| i < song.tracks.len())
            .filter(|&i| song.tracks[i].kind == TrackKind::PianoRoll)
            .is_some();

        if piano_roll_open {
            let piano_roll_frame = || {
                egui::Frame::new()
                    .fill(egui::Color32::from_rgb(36, 36, 36))
                    .inner_margin(egui::Margin::same(8))
            };
            let mut panel = PianoRollPanelUi {
                selected_track: self.selected_track,
                piano_roll_drag: &mut self.piano_roll_drag,
                selected_notes: &mut self.selected_notes,
                groove_quantize_grid_ticks: &mut self.groove_quantize_grid_ticks,
                groove_quantize_strength: &mut self.groove_quantize_strength,
                groove_humanize_timing_ticks: &mut self.groove_humanize_timing_ticks,
                groove_humanize_velocity: &mut self.groove_humanize_velocity,
                groove_template_index: &mut self.groove_template_index,
                piano_roll_zoom: &mut self.piano_roll_zoom,
                scale_root: &mut self.piano_roll_scale_root,
                scale: &mut self.piano_roll_scale,
                editing_region_index: &mut self.piano_roll_region,
                scroll_to: &mut self.piano_roll_scroll_to,
                track_effect_slots: &self.track_effect_slots,
                send_effect_slots: &self.send_effect_slots,
                master_effect_slots: &self.master_effect_slots,
                automation_drag: &mut self.automation_drag,
                track_automation_drag: &mut self.track_automation_drag,
            };
            let ctx = ui.ctx().clone();
            let mut still_open = true;
            ctx.show_viewport_immediate(
                egui::ViewportId::from_hash_of("piano_roll_viewport"),
                egui::ViewportBuilder::default()
                    .with_title("Piano Roll")
                    .with_inner_size(egui::vec2(900.0, 500.0)),
                |ui, _class| {
                    egui::CentralPanel::default()
                        .frame(piano_roll_frame())
                        .show(ui, |ui| {
                            piano_roll_contents_ui(ui, song, current_tick, &mut panel);
                        });
                    if ui.ctx().input(|i| i.viewport().close_requested()) {
                        still_open = false;
                    }
                },
            );
            if !still_open {
                self.selected_track = None;
                self.piano_roll_region = None;
            }
        }

        let beats_open = self
            .selected_beats_track
            .filter(|&i| i < song.tracks.len())
            .filter(|&i| song.tracks[i].kind == TrackKind::StepGrid)
            .is_some();

        if beats_open {
            let beats_frame = || {
                egui::Frame::new()
                    .fill(egui::Color32::from_rgb(36, 36, 36))
                    .inner_margin(egui::Margin::same(8))
            };
            let selected_beats_track = self.selected_beats_track;
            let sample_rate = self.sample_rate;
            let beats_region = &mut self.beats_region;
            let device_chain_focus = &mut self.device_chain_focus;
            let track_effect_slots = &self.track_effect_slots;
            let send_effect_slots = &self.send_effect_slots;
            let master_effect_slots = &self.master_effect_slots;
            let automation_drag = &mut self.automation_drag;
            let track_automation_drag = &mut self.track_automation_drag;
            let mut groove = StepGrooveUi {
                humanize_timing_ticks: &mut self.groove_humanize_timing_ticks,
                humanize_velocity: &mut self.groove_humanize_velocity,
                template_index: &mut self.groove_template_index,
            };
            let ctx = ui.ctx().clone();
            let mut still_open = true;
            ctx.show_viewport_immediate(
                egui::ViewportId::from_hash_of("beats_viewport"),
                egui::ViewportBuilder::default()
                    .with_title("Beats")
                    .with_inner_size(egui::vec2(1000.0, 650.0)),
                |ui, _class| {
                    egui::CentralPanel::default()
                        .frame(beats_frame())
                        .show(ui, |ui| {
                            beats_contents_ui(
                                ui,
                                song,
                                current_tick,
                                sample_rate,
                                selected_beats_track,
                                beats_region,
                                device_chain_focus,
                                track_effect_slots,
                                send_effect_slots,
                                master_effect_slots,
                                automation_drag,
                                track_automation_drag,
                                &mut groove,
                            );
                        });
                    if ui.ctx().input(|i| i.viewport().close_requested()) {
                        still_open = false;
                    }
                },
            );
            if !still_open {
                self.selected_beats_track = None;
                self.beats_region = None;
            }
        }

        if self.playlist_open {
            // Docked into the main window's remaining central area (to the right of the Channel
            // Rack panel), rather than an always-detached viewport like Piano Roll/Beats — the
            // Playlist is the song-arrangement overview, so it reads best as part of the main
            // screen instead of a window you have to keep finding and re-positioning.
            egui::Frame::new()
                .fill(egui::Color32::from_rgb(30, 30, 30))
                .inner_margin(egui::Margin::same(8))
                .show(ui, |ui| {
                    let mut editor_targets = PlaylistEditorTargets {
                        selected_track: &mut self.selected_track,
                        piano_roll_region: &mut self.piano_roll_region,
                        piano_roll_scroll_to: &mut self.piano_roll_scroll_to,
                        selected_beats_track: &mut self.selected_beats_track,
                        beats_region: &mut self.beats_region,
                    };
                    playlist_contents_ui(
                        ui,
                        song,
                        current_tick,
                        &mut self.playlist_zoom,
                        &mut self.playlist_drag,
                        &mut self.audio_clip_drag,
                        &mut self.audio_clip_context_menu,
                        &mut self.take_folder_context_menu,
                        &mut self.take_folder_editor,
                        &mut self.flex_editor,
                        &mut editor_targets,
                    );
                });
        } else if self.session_view_open {
            egui::Frame::new()
                .fill(egui::Color32::from_rgb(30, 30, 30))
                .inner_margin(egui::Margin::same(8))
                .show(ui, |ui| {
                    session_view_ui::session_view_contents_ui(
                        ui,
                        song,
                        &self.transport,
                        &self.session_slots,
                        &mut self.session_quantize,
                        &mut self.follow_action_editor,
                    );
                });
        }
    }
}

/// Per-track (or master-bus) CLAP/built-in effect-chain UI state, bundled to keep `track_ui`'s
/// parameter list manageable. `paths`/`instances`/`messages` are this one chain, indexed the same
/// as `Track::effects`/`Song::master_effects` (slot 0 first, feeding into slot 1, and so on) — one
/// entry per effect slot, whether or not that slot has successfully loaded a plugin yet.
struct TrackFxUi<'a> {
    /// Row into `slots` this chain lives at — always 0 when `chain_kind` is `Master` (see
    /// `plugin_host::MasterEffectSlots`'s doc comment on why the master chain still uses the
    /// per-track `TrackEffectSlots` shape, just pinned to one row).
    track_index: usize,
    /// Which location this chain belongs to — see `FxChainKind`.
    chain_kind: FxChainKind,
    paths: &'a mut Vec<String>,
    messages: &'a mut Vec<Option<(bool, String)>>,
    slots: TrackEffectSlots,
    instances: &'a mut Vec<Option<PluginInstance<DawHost>>>,
    guis: &'a mut Vec<Option<PluginGuiHandle>>,
    /// (sample_rate, min_frames, max_frames) from the running audio engine, or `None` if it
    /// failed to start — plugins can't be activated without a device to size buffers for.
    engine_config: Option<(f64, u32, u32)>,
    /// The project's imported CLAP plugin library (`Song::plugins`), offered by mnemonic name —
    /// both as a picker next to each CLAP slot's path field, and as one-click entries in
    /// "+ Add Effect" — so paths don't need retyping.
    known_plugins: &'a [ProjectPlugin],
    /// Every track's name, in `Song::tracks` order — offered as sidechain key-source choices for
    /// a CLAP/Compressor/NoiseGate slot's sidechain picker (see `fx_chain_ui`), regardless of
    /// which chain (a track's, a send's, a submix's, or the master's) is being edited.
    track_names: &'a [String],
    editor: &'a mut Option<EffectEditorTarget>,
    /// Set by channel_rack_row_ui's "🎹" button to focus the bottom Device Panel on that track.
    /// Unused (and meaningless) for the master bus, which has no synth.
    device_chain_focus: &'a mut Option<DeviceChainFocus>,
    /// Set by channel_rack_row_ui's "🗑" button; applied by the caller after the track loop ends
    /// (can't remove from `song.tracks` mid-iteration since it's borrowed via `iter_mut`). Unused
    /// for the master bus, which can't be deleted.
    remove_requested: &'a mut Option<usize>,
    /// Whether `fx_chain_ui` should draw a built-in effect's own knobs right under its slot row
    /// (only set by the bottom Device Panel — see `device_panel_track_fx_chain_ui`) rather than
    /// requiring the separate "FX Params" window every other caller of `fx_chain_ui` still uses.
    /// A CLAP plugin's params still only open in that window even here — inlining those would need
    /// the embedded/floating-GUI machinery `effect_editor`'s window already carries (window handle,
    /// `active_embedded_gui`), which the Device Panel doesn't have access to.
    inline_params: bool,
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
fn pan_label(pan: f32) -> String {
    if pan.abs() < 0.01 {
        "C".to_string()
    } else if pan < 0.0 {
        format!("{:.0}% L", -pan * 100.0)
    } else {
        format!("{:.0}% R", pan * 100.0)
    }
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

/// Bundles the Beats window's shared groove/humanize controls — the same underlying
/// `SimpleDawApp` fields the Piano Roll toolbar uses (see `PianoRollPanelUi`'s `groove_*`
/// fields), reused here since "how much to humanize by" is a general preference, not a
/// per-lane setting.
struct StepGrooveUi<'a> {
    humanize_timing_ticks: &'a mut usize,
    humanize_velocity: &'a mut u8,
    template_index: &'a mut usize,
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
fn step_grid_lanes_ui(
    ui: &mut egui::Ui,
    lanes: &mut [Lane],
    current_tick: Option<usize>,
    sample_rate: Option<u32>,
    color: egui::Color32,
    track_index: usize,
    region_index: usize,
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
            let is_focused = *device_chain_focus
                == Some(DeviceChainFocus::Lane { track_index, region_index, lane_index });
            let synth_button = egui::Button::new("🎹").selected(lane.synth_override || is_focused);
            if ui
                .add(synth_button)
                .on_hover_text("Show this lane's own synth (overrides the track synth) in the Device Panel")
                .clicked()
            {
                *device_chain_focus =
                    Some(DeviceChainFocus::Lane { track_index, region_index, lane_index });
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

/// The Beats window's header (selected track name/mute badge) and step grid, rendered inside the
/// always-detached Beats window (see `ui` in `impl eframe::App for SimpleDawApp`) — the step-grid
/// counterpart of `piano_roll_contents_ui`, including the "no in-window picker, double-click a
/// region in the Playlist instead" behavior.
#[allow(clippy::too_many_arguments)]
fn beats_contents_ui(
    ui: &mut egui::Ui,
    song: &mut Song,
    current_tick: Option<usize>,
    sample_rate: Option<u32>,
    selected_beats_track: Option<usize>,
    editing_region_index: &mut Option<usize>,
    device_chain_focus: &mut Option<DeviceChainFocus>,
    track_effect_slots: &TrackEffectSlots,
    send_effect_slots: &SendEffectSlots,
    master_effect_slots: &MasterEffectSlots,
    automation_drag: &mut Option<AutomationDrag>,
    track_automation_drag: &mut Option<AutomationDrag>,
    groove: &mut StepGrooveUi,
) {
    let selected = selected_beats_track
        .filter(|&i| i < song.tracks.len())
        .filter(|&i| song.tracks[i].kind == TrackKind::StepGrid);
    let region = selected.and_then(|index| {
        let region_index = (*editing_region_index)?;
        (region_index < song.tracks[index].regions.len()).then_some((index, region_index))
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
            if let Some((_, region_index)) = region {
                ui.separator();
                ui.weak(&song.tracks[index].regions[region_index].name);
            }
        }
        None => {
            ui.heading("Beats");
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
        Some((index, region_index)) => {
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
                    region_index,
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
    }
}

/// Bundles the bottom Device Panel's mutable app-state borrows (see `device_panel_contents_ui`),
/// the panel counterpart of `ChannelRackUi`/`MixerUi` — a plain struct rather than a dozen
/// positional parameters, for the same reason those have one.
struct DevicePanelUi<'a> {
    focus: &'a mut Option<DeviceChainFocus>,
    track_effect_slots: &'a TrackEffectSlots,
    track_effect_instances: &'a mut Vec<Vec<Option<PluginInstance<DawHost>>>>,
    track_effect_guis: &'a mut Vec<Vec<Option<PluginGuiHandle>>>,
    track_effect_paths: &'a mut Vec<Vec<String>>,
    track_effect_messages: &'a mut Vec<Vec<Option<(bool, String)>>>,
    effect_editor: &'a mut Option<EffectEditorTarget>,
    new_preset_name: &'a mut String,
    preset_message: &'a mut Option<(bool, String)>,
}

/// The always-visible bottom Device Panel's contents — whichever track's or step-grid lane's
/// instrument + effect chain `panel.focus` (see `DeviceChainFocus`) currently points at, laid out
/// inline rather than behind a separate window/menu (Bitwig/Ableton's docked device-rack pattern).
/// Shows a placeholder until a track/lane's "🎹" button is clicked at least once.
fn device_panel_contents_ui(
    ui: &mut egui::Ui,
    song: &mut Song,
    engine_config: Option<(f64, u32, u32)>,
    panel: &mut DevicePanelUi,
) {
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
                ui.horizontal(|ui| {
                    ui.label("Engine:");
                    let track = &mut song.tracks[index];
                    if ui
                        .selectable_label(track.synth_engine == SynthEngine::Simple, "Simple Synth")
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
                        synth_preset_bar_ui(ui, song, index, panel.new_preset_name, panel.preset_message);
                        synth_params_ui(ui, &mut song.tracks[index].synth);
                    }
                    SynthEngine::Trine => {
                        synth_preset_bar_ui(ui, song, index, panel.new_preset_name, panel.preset_message);
                        trine_params_ui(ui, &mut song.tracks[index].trine);
                    }
                    SynthEngine::Wave => {
                        synth_preset_bar_ui(ui, song, index, panel.new_preset_name, panel.preset_message);
                        wave_params_ui(ui, &mut song.tracks[index].wave);
                    }
                }
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
            ui.checkbox(&mut lane.synth_override, "Override the track synth for this lane");
            if !lane.sample_path.trim().is_empty() {
                ui.weak(
                    "This lane has a sample loaded — the sample takes priority and plays \
                     instead of any synth until it's cleared.",
                );
            }
            if lane.synth_override {
                ui.horizontal(|ui| {
                    ui.label("Engine:");
                    if ui
                        .selectable_label(lane.synth_engine == SynthEngine::Simple, "Simple Synth")
                        .clicked()
                    {
                        lane.synth_engine = SynthEngine::Simple;
                    }
                    if ui
                        .selectable_label(lane.synth_engine == SynthEngine::Trine, "Trine")
                        .clicked()
                    {
                        lane.synth_engine = SynthEngine::Trine;
                    }
                    if ui
                        .selectable_label(lane.synth_engine == SynthEngine::Wave, "Wave")
                        .clicked()
                    {
                        lane.synth_engine = SynthEngine::Wave;
                    }
                });
                match lane.synth_engine {
                    SynthEngine::Simple => synth_params_ui(ui, &mut lane.synth),
                    SynthEngine::Trine => trine_params_ui(ui, &mut lane.trine),
                    SynthEngine::Wave => wave_params_ui(ui, &mut lane.wave),
                }
            } else {
                ui.weak("Unchecked: this lane plays the track's own synth.");
            }
            ui.separator();
            // A lane has no effect chain of its own — effects are track-scoped — so the panel
            // shows the owning track's chain too, since that's still part of the sound being heard.
            device_panel_track_fx_chain_ui(ui, song, track_index, engine_config, panel);
        }
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

/// Renders the "+ Add Effect" menu and the ordered list of the track's effect-chain slots
/// (CLAP path/Load or built-in label, Params button, status message, remove button). Opened
/// from each Channel Rack row's "FX" popup menu, the Mixer's per-strip "FX" menu, and the bottom
/// Device Panel (see `channel_rack_row_ui`/`device_panel_track_fx_chain_ui`).
fn fx_chain_ui(ui: &mut egui::Ui, fx: &mut TrackFxUi) {
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
    for slot_index in 0..fx.paths.len() {
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
            ui.weak(format!("{}.", slot_index + 1));
            match slot_kind {
                FxSlotKind::Clap => {
                    ui.add_sized(
                        [150.0, 20.0],
                        egui::TextEdit::singleline(&mut fx.paths[slot_index])
                            .hint_text("effect.clap"),
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
            if let Some((ok, message)) = fx.messages[slot_index].as_ref() {
                let color = if *ok {
                    egui::Color32::from_rgb(120, 220, 140)
                } else {
                    egui::Color32::RED
                };
                ui.colored_label(color, message);
            }
            if ui
                .small_button("🗑")
                .on_hover_text("Remove this effect from the chain")
                .clicked()
            {
                fx_slot_to_remove = Some(slot_index);
            }
        });
        // Bitwig/Ableton-style inline knobs, only for the Device Panel (see `TrackFxUi::inline_params`'s
        // doc comment for why a CLAP slot's params still only open in the separate "FX Params" window).
        if fx.inline_params
            && let FxSlotKind::BuiltIn(_) = slot_kind
        {
            egui::Frame::group(ui.style()).show(ui, |ui| {
                if let Ok(mut slots) = fx.slots.lock()
                    && let Some(chain) = slots.get_mut(fx.track_index)
                    && let Some(Some(EffectInstance::BuiltIn(effect))) = chain.get_mut(slot_index)
                {
                    built_in_effect_params_ui(ui, effect);
                }
            });
        }
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
/// by `start_tick`/`loop_length_steps` (or the audio equivalent). Docked into the main window's
/// central area, toggled by `playlist_open` (see `ui` in `impl eframe::App for SimpleDawApp`) —
/// unlike Piano Roll/Beats, this isn't a detached viewport. This is also the *only* place a region
/// gets opened for editing: double-click one to open it in Piano Roll/Beats (see
/// `PlaylistEditorTargets`) — there's no Channel Rack button or in-window picker for it.
/// Draws a small preview of a region's musical content inside its Playlist clip rect — thin bars
/// for piano-roll notes (stacked by pitch) or step-grid hits (stacked by lane row). Tiled across
/// `rect`'s width the same way `audio.rs`'s `Sequencer::tick` repeats/truncates the content within
/// `loop_length_steps` (`(tick_index - region.start_tick) % region.content_length_ticks()`), so the
/// preview matches what's actually played at each point along the clip rather than just the first
/// pass through the content.
fn draw_region_note_preview(painter: &egui::Painter, rect: egui::Rect, region: &Region) {
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
fn draw_audio_clip_waveform(
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
fn playlist_contents_ui(
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
                            *editor_targets.piano_roll_region = Some(region_index);
                            *editor_targets.piano_roll_scroll_to =
                                Some(local_step * TICKS_PER_STEP);
                        }
                        TrackKind::StepGrid => {
                            *editor_targets.selected_beats_track = Some(track_index);
                            *editor_targets.beats_region = Some(region_index);
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
fn flex_editor_window_ui(
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

/// Hit-tests and applies click/drag/right-click gestures against every `Audio`-kind track's
/// `audio_clips`, rendered in the same Playlist canvas as `handle_playlist_interaction` but in the
/// rows below it (`audio_rows_top` onward — see `playlist_contents_ui`). Mirrors
/// `handle_playlist_interaction`'s structure (click/drag_started/dragged/drag_stopped) but for
/// clips instead of regions, and with no `Create` mode — a clip is only ever created by
/// recording/import, never drawn out on the timeline. Right-clicking a clip opens a context menu
/// ("Strip Silence"/"Delete"/"Flex Time / Pitch…" — see `apply_strip_silence` and
/// `flex_editor_window_ui`) instead of deleting immediately.
#[allow(clippy::too_many_arguments)]
fn handle_audio_clip_interaction(
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
fn handle_take_folder_interaction(
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
fn take_folder_editor_window_ui(
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
fn automation_lanes_ui(
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

fn note_name(pitch: u8) -> String {
    const NAMES: [&str; 12] = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    let octave = (pitch as i32 / 12) - 1;
    format!("{}{}", NAMES[(pitch % 12) as usize], octave)
}

/// Name of a pitch class (0=C..11=B), independent of octave — for the piano roll's scale-root
/// picker, where only the class matters (see `PianoRollScale::contains`).
fn pitch_class_name(pitch_class: u8) -> &'static str {
    const NAMES: [&str; 12] = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    NAMES[(pitch_class % 12) as usize]
}

fn lane_sample_controls(ui: &mut egui::Ui, lane: &mut Lane, sample_rate: Option<u32>) {
    ui.add_sized(
        [160.0, 22.0],
        egui::TextEdit::singleline(&mut lane.sample_path).hint_text("path/to/sample.wav"),
    );

    let can_load = sample_rate.is_some();
    if can_load {
        ui.menu_button("🎵", |ui| {
            for (category, sounds) in factory_drum_samples() {
                ui.menu_button(*category, |ui| {
                    for (label, filename) in sounds {
                        if ui.button(label).clicked() {
                            lane.sample_path =
                                factory_samples_dir().join(filename).display().to_string();
                            if let Some(rate) = sample_rate {
                                lane.load_sample(rate);
                            }
                            ui.close();
                        }
                    }
                });
            }
        })
        .response
        .on_hover_text("Load a built-in drum sample");
    }
    if ui
        .add_enabled(can_load, egui::Button::new("Browse"))
        .clicked()
    {
        if let Some(path) = browse_for_file(&lane.sample_path, "WAV sample", &["wav"], None) {
            lane.sample_path = path;
            if let Some(rate) = sample_rate {
                lane.load_sample(rate);
            }
        }
    }
    if ui
        .add_enabled(can_load, egui::Button::new("Load"))
        .clicked()
    {
        if let Some(rate) = sample_rate {
            lane.load_sample(rate);
        }
    }

    // Always reserve space for the status dot and the remove-sample button, even when neither
    // applies yet (a freshly added lane has no sample and no error) — otherwise this row's step
    // buttons end up shifted left relative to every other lane's row.
    let status_color = if lane.sample.is_some() {
        egui::Color32::from_rgb(120, 220, 140)
    } else if lane.sample_error.is_some() {
        egui::Color32::RED
    } else {
        egui::Color32::TRANSPARENT
    };
    let status_dot = ui.colored_label(status_color, "⏺");
    if let Some(err) = &lane.sample_error {
        status_dot.on_hover_text(err);
    } else if lane.sample.is_some() {
        status_dot.on_hover_text("Sample loaded");
    }
    if ui
        .add_enabled(lane.sample.is_some(), egui::Button::new("🗑").small())
        .on_hover_text("Remove sample, use synth")
        .clicked()
    {
        lane.clear_sample();
    }
}

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_min_inner_size([640.0, 480.0]),
        ..Default::default()
    };
    eframe::run_native(
        "simple-daw",
        native_options,
        Box::new(|cc| {
            cc.egui_ctx.set_visuals(fl_studio_visuals());
            // egui's default resize-handle hit zone (3px) is too thin to reliably grab with a
            // mouse, especially on the Channel Rack/Piano Roll panel border introduced by this
            // layout — widen it app-wide.
            cc.egui_ctx.all_styles_mut(|style| {
                style.interaction.resize_grab_radius_side = 10.0;
            });
            Ok(Box::new(SimpleDawApp::new()))
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sustained_note_track(name: &str, midi_channel: u8) -> Track {
        let mut track = Track::new_piano_roll(name, midi_channel);
        track.synth.attack_seconds = 0.0;
        track.synth.decay_seconds = 0.0;
        track.synth.sustain_level = 1.0;
        track.regions.push(model::Region {
            name: "Hit".to_string(),
            start_tick: 0,
            content_length_steps: 4,
            loop_length_steps: 4,
            content: RegionContent::PianoRoll(vec![model::Note {
                id: 0,
                pitch: 60,
                start_tick: 0,
                length_ticks: 4 * model::TICKS_PER_STEP,
                velocity: 127,
            }]),
            fade_in_ticks: 0,
            fade_out_ticks: 0,
            automation: Vec::new(),
        });
        track
    }

    #[test]
    fn freeze_track_bakes_content_and_silences_live_triggering() {
        let mut song = Song::demo();
        song.tracks.push(sustained_note_track("Lead", 1));
        let track_index = song.tracks.len() - 1;

        let (ok, message) = freeze_track(&mut song, track_index, 48_000, Some(48_000));
        assert!(ok, "message: {message}");
        assert!(song.tracks[track_index].frozen);
        let clip = song.tracks[track_index]
            .frozen_clip
            .as_ref()
            .expect("freeze should populate frozen_clip");
        assert!(clip.buffer.is_some(), "frozen clip should have loaded its baked audio");
        assert!(
            !song.tracks[track_index].regions.is_empty(),
            "freeze must not touch the track's original regions — only bounce-in-place does"
        );

        std::fs::remove_file(&clip.file_path).ok();
    }

    #[test]
    fn unfreeze_track_clears_frozen_state_and_deletes_its_file() {
        let mut song = Song::demo();
        song.tracks.push(sustained_note_track("Lead", 1));
        let track_index = song.tracks.len() - 1;
        let (ok, _) = freeze_track(&mut song, track_index, 48_000, Some(48_000));
        assert!(ok);
        let file_path = song.tracks[track_index].frozen_clip.as_ref().unwrap().file_path.clone();
        assert!(std::path::Path::new(&file_path).exists());

        unfreeze_track(&mut song, track_index);

        assert!(!song.tracks[track_index].frozen);
        assert!(song.tracks[track_index].frozen_clip.is_none());
        assert!(
            !std::path::Path::new(&file_path).exists(),
            "unfreeze should delete the frozen WAV file it owns"
        );
    }

    #[test]
    fn bounce_track_in_place_replaces_content_with_one_baked_clip() {
        let mut song = Song::demo();
        song.tracks.push(sustained_note_track("Lead", 1));
        let track_index = song.tracks.len() - 1;

        let (ok, message) =
            bounce_track_in_place(&mut song, track_index, 48_000, Some(48_000));
        assert!(ok, "message: {message}");

        let track = &song.tracks[track_index];
        assert_eq!(track.kind, TrackKind::Audio);
        assert!(track.regions.is_empty(), "bounce should clear the original MIDI content");
        assert!(track.effects.is_empty(), "bounce should clear the chain now baked into the clip");
        assert_eq!(track.audio_clips.len(), 1);
        assert!(track.audio_clips[0].buffer.is_some());
        assert!(!track.frozen, "bounce is permanent, not a reversible freeze");

        std::fs::remove_file(&track.audio_clips[0].file_path).ok();
    }

    #[test]
    fn finish_recording_rejects_an_empty_capture() {
        let mut song = Song::demo();
        let track_index = song.add_track("Vocals", 5, TrackKind::Audio);
        let (ok, message) = finish_recording(&mut song, track_index, 0, &[], 48_000, Some(48_000));
        assert!(!ok, "message: {message}");
        assert!(song.tracks[track_index].take_folders.is_empty());
    }

    #[test]
    fn finish_recording_creates_a_new_take_folder_for_a_fresh_recording() {
        let mut song = Song::demo();
        let track_index = song.add_track("Vocals", 5, TrackKind::Audio);
        let samples = vec![0.5f32; 48_000]; // 1 second at 48kHz
        let (ok, _) = finish_recording(&mut song, track_index, 0, &samples, 48_000, Some(48_000));
        assert!(ok);

        let folders = &song.tracks[track_index].take_folders;
        assert_eq!(folders.len(), 1);
        assert_eq!(folders[0].takes.len(), 1);
        assert_eq!(folders[0].comp.len(), 1);
        assert_eq!(folders[0].comp[0].take_index, 0);
        assert_eq!(folders[0].comp[0].start_tick, 0);
        assert_eq!(folders[0].comp[0].end_tick, folders[0].length_ticks);
    }

    #[test]
    fn finish_recording_at_the_same_start_tick_joins_the_existing_folder_as_a_new_take() {
        let mut song = Song::demo();
        let track_index = song.add_track("Vocals", 5, TrackKind::Audio);
        let samples = vec![0.5f32; 48_000];
        finish_recording(&mut song, track_index, 480, &samples, 48_000, Some(48_000));
        finish_recording(&mut song, track_index, 480, &samples, 48_000, Some(48_000));

        let folders = &song.tracks[track_index].take_folders;
        assert_eq!(folders.len(), 1, "should not create a second overlapping folder");
        assert_eq!(folders[0].takes.len(), 2);
        assert_eq!(
            folders[0].comp[0].take_index, 1,
            "the just-recorded take should be the one comped in"
        );
    }

    #[test]
    fn finish_recording_at_a_different_start_tick_creates_a_separate_folder() {
        let mut song = Song::demo();
        let track_index = song.add_track("Vocals", 5, TrackKind::Audio);
        let samples = vec![0.5f32; 48_000];
        finish_recording(&mut song, track_index, 0, &samples, 48_000, Some(48_000));
        finish_recording(&mut song, track_index, 9600, &samples, 48_000, Some(48_000));

        let folders = &song.tracks[track_index].take_folders;
        assert_eq!(folders.len(), 2);
        assert_eq!(folders[0].takes.len(), 1);
        assert_eq!(folders[1].takes.len(), 1);
    }
}
