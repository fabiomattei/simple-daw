mod audio;
mod audio_clip_interaction;
mod audio_input;
mod automation_panel;
mod beats_panel;
mod builtin_fx;
mod channel_rack_panel;
mod device_panel;
mod factory_presets;
mod file_ops;
mod groove;
mod gui_embed;
#[cfg(unix)]
mod mcp_bridge;
#[cfg(unix)]
mod mcp_control;
mod metering;
mod midi_import;
mod mixer_panel;
mod model;
mod piano_roll_panel;
mod pitch;
mod playlist_panel;
mod plugin_host;
mod sample;
mod session;
mod session_view_ui;
mod stretch;
mod synth_preview_widgets;
mod synth_simple_panel;
mod synth_trine_panel;
mod synth_wave_panel;
mod tempo;
mod tempo_detection;
mod transient_detection;
mod transport_lcd;
mod wavetable;

use std::collections::HashSet;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use audio::{AudioEngine, Transport};
use audio_clip_interaction::{
    AudioClipDrag, FlexEditorMode, FlexMarkerDrag, FlexNoteDrag, TakeFolderCompDrag, flex_editor_window_ui,
    session_flex_editor_window_ui, take_folder_editor_window_ui,
};
use automation_panel::AutomationDrag;
use beats_panel::beats_contents_ui;
use channel_rack_panel::channel_rack_contents_ui;
use clack_host::prelude::PluginInstance;
use device_panel::{
    DeviceChainFocus, DevicePanelUi, EffectEditorTarget, FxChainKind, TrackFxUi, built_in_effect_params_ui,
    close_effect_gui, device_panel_contents_ui, effect_params_ui, fx_chain_ui, plugin_gui_button_ui,
};
use file_ops::{
    apply_chain_specs_at, apply_loaded_effects, bounce_track_in_place, browse_for_file, build_effect_chain,
    finish_recording, freeze_track, handle_session_record_click, perform_export, perform_load, perform_save,
    sync_song_effects, unfreeze_track,
};
use metering::MeterHandles;
use mixer_panel::mixer_contents_ui;
use piano_roll_panel::{PianoRollDrag, PianoRollScale, draw_region_note_preview, piano_roll_contents_ui};
use playlist_panel::{PlaylistDrag, PlaylistEditorTargets, playlist_contents_ui};
use model::{
    AudioClip,
    Lane,
    ProjectPlugin, Region, RegionContent,
    SessionQuantize, Song,
    TICKS_PER_STEP,
    TrackEffectConfig, TrackKind,
};
use plugin_host::{
    DawHost, EffectInstance, MasterEffectSlots, PluginGuiHandle,
    SendEffectSlots, SubmixEffectSlots, TrackEffectSlots,
};
use raw_window_handle::HasWindowHandle;
use sample::SampleBuffer;
use transport_lcd::{toolbar_group, transport_lcd_ui};

/// Pixels per 16th-note step; ticks are drawn at a fraction of this.
const PIXELS_PER_STEP: f32 = 40.0;
const PIXELS_PER_TICK: f32 = PIXELS_PER_STEP / TICKS_PER_STEP as f32;
/// How close (in canvas pixels) a press has to be to a note's/region's right edge to resize it
/// instead of moving it — shared by the Piano Roll's note-edge hit-testing and the Playlist's
/// region-edge/fade-handle hit-testing.
pub(crate) const RESIZE_HANDLE_PX: f32 = 6.0;
/// Height of one lane row in the Playlist's per-track canvas — shared with the not-yet-extracted
/// Flex editor/Take Folder comp editor windows, which lay out their own rows at the same height.
pub(crate) const PLAYLIST_LANE_HEIGHT: f32 = 26.0;

/// FL Studio–style accent green: playback, active steps/LEDs, the piano-roll playhead. `pub(crate)`
/// so `session_view_ui` can reuse it for a playing Session View slot, the same "active" meaning.
pub(crate) const FL_ACCENT_GREEN: egui::Color32 = egui::Color32::from_rgb(139, 198, 63);
/// FL Studio–style accent orange: warnings, recording, clipping. `pub(crate)` so `session_view_ui`
/// can reuse it for a queued Session View slot.
pub(crate) const FL_ACCENT_ORANGE: egui::Color32 = egui::Color32::from_rgb(242, 169, 59);
/// Accent yellow: an active track solo, distinct from mute's orange. `pub(crate)` so
/// `mixer_panel` can reuse it for the same "active solo" meaning on a channel strip.
pub(crate) const FL_ACCENT_YELLOW: egui::Color32 = egui::Color32::from_rgb(235, 210, 64);

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
pub(crate) fn track_color(index: usize) -> egui::Color32 {
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

pub(crate) fn tick_to_x(tick: usize, zoom: f32) -> f32 {
    tick as f32 * PIXELS_PER_TICK * zoom
}

pub(crate) fn x_to_tick(x: f32, zoom: f32) -> usize {
    (x / (PIXELS_PER_TICK * zoom)).round().max(0.0) as usize
}

/// An `AudioClip`'s on-timeline length in ticks, for drawing/hit-testing its block in the
/// Playlist — `AudioClip::effective_length_ticks` at the song's current tempo (`ticks_per_second`,
/// from `audio::ticks_per_second`), same as `audio::arrangement_length_ticks` does for looping.
pub(crate) fn audio_clip_length_ticks(clip: &AudioClip, ticks_per_second: f64) -> usize {
    if clip.buffer.is_some() {
        clip.effective_length_ticks(ticks_per_second).max(1)
    } else {
        // Still loading (or failed to load) — a minimal placeholder width keeps a broken clip
        // visible/selectable to move or delete, rather than invisible.
        TICKS_PER_STEP
    }
}

/// What the Piano Roll or Beats window is currently bound to — a Playlist `Region`
/// (`Track::regions`, addressed by index) or a Session View slot's own `RegionContent`
/// (`Track::session_clips`, addressed by slot index) — see `SimpleDawApp::piano_roll_region`/
/// `beats_region`. `piano_roll_contents_ui`/`beats_contents_ui` each branch on the variant
/// directly (not through a shared abstraction) since the two live in differently-shaped
/// containers: `Region` has its own top-level `content`/`content_length_steps`/
/// `loop_length_steps`/`automation` fields, while a session slot's equivalents nest inside
/// `SessionClipContent::Region` and it has no `automation` of its own yet (see the Session View
/// parity plan's "per-clip envelopes" item).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RegionEditTarget {
    Region(usize),
    SessionSlot(usize),
}

/// Which audio clip's right-click context menu (if any) is currently open, and where — set by
/// `handle_audio_clip_interaction` on `secondary_clicked()`, read back by `playlist_contents_ui`'s
/// `response.context_menu` closure to know which clip "Strip Silence"/"Delete" apply to. Kept
/// alongside `audio_clip_drag` in `SimpleDawApp` rather than folded into it, since a context menu
/// being open has nothing to do with a drag being in progress.
#[derive(Clone, Copy)]
pub(crate) struct AudioClipContextMenuTarget {
    pub(crate) track_index: usize,
    pub(crate) clip_index: usize,
}

/// Which take folder's right-click context menu (if any) is currently open, and where — the
/// `TakeFolder` counterpart of `AudioClipContextMenuTarget`. This phase's take-folder editing is
/// context-menu-only (pick a take, or delete the whole folder) — no move/trim drag yet, unlike
/// plain audio clips (see `handle_take_folder_interaction`).
#[derive(Clone, Copy)]
pub(crate) struct TakeFolderContextMenuTarget {
    pub(crate) track_index: usize,
    pub(crate) folder_index: usize,
}

/// Set by a track row's Freeze/Unfreeze/Bounce button; applied by the caller once `song.tracks`'
/// mutable-iterator borrow from the row loop has ended — same "signal during the loop, apply
/// after" pattern `track_to_remove` already uses, since `freeze_track`/`bounce_track_in_place`
/// need `&mut Song` as a whole, not just the one `&mut Track` a row has borrowed.
pub(crate) enum TrackFreezeAction {
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
    /// Session View performance log while the toolbar's "Capture" button is armed, published by
    /// the audio thread once per callback — see `audio::CaptureLogHandle`'s doc comment. Read once,
    /// when the button is turned back off (`handle_capture_toggle`), and handed to `Song::
    /// insert_captured_session_performance` to materialize.
    capture_log: audio::CaptureLogHandle,
    /// Session View's grid-wide launch-quantize setting (e.g. "1 Bar") — live UI state, not song
    /// data, read each frame to compute `Transport::session_quantize_ticks` from the current
    /// song's own `Song::steps_per_bar`. A `SessionClip::quantize_override` can override this per
    /// clip. See `model::SessionQuantize`.
    session_quantize: SessionQuantize,
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
    /// Which of `selected_track`'s own regions or Session View slots the Piano Roll is showing/
    /// editing — see `RegionEditTarget`. `None` (or pointing past the end after the target was
    /// deleted) shows a "double-click a region in the Playlist" placeholder instead.
    piano_roll_region: Option<RegionEditTarget>,
    /// A content-local tick the Piano Roll should scroll to on its next render, set alongside
    /// `piano_roll_region`/`selected_track` by a Playlist double-click so the grid opens on the
    /// section that was actually clicked rather than always at the start. Consumed (cleared) by
    /// `piano_roll_ui` once applied, so it doesn't fight manual scrolling afterward.
    piano_roll_scroll_to: Option<usize>,
    /// Whether the (open) Piano Roll is docked into the central area (alongside Playlist/Session
    /// View/Beats, taking priority over them while it's open — see the `ui` central-area block in
    /// `impl eframe::App for SimpleDawApp`) instead of its own native OS window. Defaults to `true`
    /// to preserve this window's original always-floating behavior.
    piano_roll_detached: bool,
    /// Index of the track whose Beats window is open — same lifecycle as `selected_track`, but
    /// for step-grid tracks.
    selected_beats_track: Option<usize>,
    /// Which of `selected_beats_track`'s own regions or Session View slots the Beats window is
    /// showing/editing — the Beats counterpart of `piano_roll_region`. See `RegionEditTarget`.
    beats_region: Option<RegionEditTarget>,
    /// Whether the (open) Beats window is docked into the central area instead of its own native
    /// OS window — see `piano_roll_detached`. When both Piano Roll and Beats are open and docked
    /// at once, Piano Roll takes priority for the shared central area.
    beats_detached: bool,
    /// Whether the Channel Rack is popped out into its own native OS window (via
    /// `egui::Context::show_viewport_immediate`) instead of docked as the left `egui::Panel`.
    channel_rack_detached: bool,
    /// Whether the Playlist (arrangement timeline) window is open — toggled from the toolbar.
    playlist_open: bool,
    /// Whether the (open) Playlist is popped out into its own native OS window instead of docked
    /// into the central area — same dock/detach split as the Channel Rack, see
    /// `channel_rack_detached`. Mutually exclusive with `session_view_detached` the same way
    /// `playlist_open`/`session_view_open` are: only one central-area view detaches at a time.
    playlist_detached: bool,
    /// Whether Session View (the clip-launching grid) is showing in the central area instead of
    /// the Playlist — toggled from the toolbar, mutually exclusive with `playlist_open` (both want
    /// the same central area; unlike Playlist vs. Mixer, which dock to different regions and can
    /// coexist). See `session_view_ui::session_view_contents_ui`.
    session_view_open: bool,
    /// Whether the (open) Session View is popped out into its own native OS window instead of
    /// docked into the central area — see `playlist_detached`.
    session_view_detached: bool,
    /// Whether the Mixer (classic vertical channel-strip view — one strip per track plus a Master
    /// strip) is visible at all, toggled from the toolbar. Same dock/detach split as the Channel
    /// Rack (see `mixer_detached`), but unlike the Channel Rack it can be hidden entirely, since
    /// the same volume/mute/solo/FX controls already live inline on each Channel Rack row.
    mixer_open: bool,
    /// Whether the (visible) Mixer is popped out into its own native OS window instead of docked
    /// as a bottom `egui::Panel` — see `channel_rack_detached`.
    mixer_detached: bool,
    /// Whether the Device Panel is popped out into its own native OS window instead of docked as
    /// a bottom `egui::Panel` — see `channel_rack_detached`.
    device_panel_detached: bool,
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
    /// Session View's counterpart of `flex_editor`: which slot's own `AudioClip` (inside a
    /// `SessionClipContent::Audio` slot) has its Flex Time/Pitch editor window open, if any —
    /// `(track_index, slot_index)` into `Track::session_clips`. A fully separate window/state
    /// (not widened addressing on `flex_editor` itself) since a Playlist clip and a session
    /// slot's own clip are independent things that can each have their own editor window open at
    /// once — see `session_flex_editor_window_ui`.
    session_flex_editor: Option<(usize, usize)>,
    /// Time vs. Pitch tab within the open Session View Flex editor window.
    session_flex_editor_mode: FlexEditorMode,
    /// The Session View Flex editor's own independently-decoded raw buffer — same reasoning as
    /// `flex_editor_raw`, just keyed by `(track_index, slot_index)` instead.
    session_flex_editor_raw: Option<((usize, usize), Arc<SampleBuffer>)>,
    /// In-progress warp-marker drag in the Session View Flex editor's Time tab — see
    /// `flex_marker_drag`.
    session_flex_marker_drag: Option<FlexMarkerDrag>,
    /// In-progress pitch-correction drag in the Session View Flex editor's Pitch tab — see
    /// `flex_note_drag`.
    session_flex_note_drag: Option<FlexNoteDrag>,
    /// In-progress automation-point drag in the Session View Flex editor's own automation panel
    /// (an `Audio`-content `SessionClip`'s only editor surface, so its automation panel lives here
    /// rather than in the Piano Roll/Beats windows — see `SessionClip::automation`'s doc comment).
    session_flex_automation_drag: Option<AutomationDrag>,
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
    /// (did the last "Capture to Arrangement" (see `capture_log`) produce anything, message to
    /// show) — set when the toolbar's Capture button is turned off.
    capture_message: Option<(bool, String)>,
    /// The in-progress Session View slot recording, if a slot's own record-arm button is
    /// currently engaged — see `session_view_ui::session_slot_cell_ui`'s record button. Mutually
    /// exclusive with `recording`: only one `InputRecorder` (one input device) can run at a time,
    /// so each recording kind's own start button is disabled while the other is active.
    session_recording: Option<SessionRecordingSession>,
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

/// State for an in-progress recording started from a Session View slot's own record button — the
/// session-target counterpart of `RecordingSession`. `start_tick` is only used to compute the
/// recorded duration in ticks (for bar-rounding, see `finish_session_recording`) — unlike
/// `RecordingSession::start_tick`, it isn't where the result gets placed on a timeline, since a
/// session slot has no absolute song position.
pub(crate) struct SessionRecordingSession {
    pub(crate) track_index: usize,
    pub(crate) slot_index: usize,
    pub(crate) recorder: audio_input::InputRecorder,
    pub(crate) start_tick: usize,
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
        let capture_log = audio::new_capture_log_handle();
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
            capture_log.clone(),
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
            capture_log,
            session_quantize: SessionQuantize::default(),
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
            piano_roll_detached: true,
            selected_beats_track: None,
            beats_region: None,
            beats_detached: true,
            channel_rack_detached: false,
            playlist_open: true,
            playlist_detached: false,
            session_view_open: false,
            session_view_detached: false,
            mixer_open: false,
            mixer_detached: false,
            device_panel_detached: false,
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
            session_flex_editor: None,
            session_flex_editor_mode: FlexEditorMode::Time,
            session_flex_editor_raw: None,
            session_flex_marker_drag: None,
            session_flex_note_drag: None,
            session_flex_automation_drag: None,
            record_armed_track: None,
            selected_input_device: None,
            recording: None,
            recording_message: None,
            capture_message: None,
            session_recording: None,
            selected_output_device: None,
            selected_output_sample_rate: None,
            output_device_message: None,
            #[cfg(unix)]
            mcp_rx: mcp_control::spawn(),
        }
    }
}

/// Resizes every per-track effect bookkeeping collection to match `track_count` — called after
/// loading a song, since the new song can have a different number of tracks than the old one.
pub(crate) fn resize_track_effects(
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
pub(crate) fn remove_track_effects(
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
pub(crate) fn resize_track_meters(meters: &MeterHandles, track_count: usize) {
    if let Ok(mut guard) = meters.lock() {
        guard.resize_with(track_count, || Arc::new(metering::ChannelMeterAtomics::new()));
    }
}

/// Removes the meter entry at `index`, mirroring `remove_track_effects` — called alongside it
/// whenever a track is deleted from the middle of `song.tracks`.
pub(crate) fn remove_track_meter(meters: &MeterHandles, index: usize) {
    if let Ok(mut guard) = meters.lock()
        && index < guard.len()
    {
        guard.remove(index);
    }
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
                let mut mcp_ctx = mcp_bridge::McpContext {
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
                let result = mcp_bridge::apply_mcp_command(&req.cmd, req.params, song, &mut mcp_ctx);
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
                                || (self.session_recording.is_none()
                                    && self.record_armed_track.is_some_and(|i| {
                                        song.tracks
                                            .get(i)
                                            .is_some_and(|t| t.kind == TrackKind::Audio)
                                    }));
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
                            let is_capturing = self.transport.is_capturing();
                            let capture_enabled = is_capturing
                                || (self.transport.is_session_mode() && playing);
                            let capture_button =
                                egui::Button::new(egui::RichText::new("⏺ Capture").size(13.0))
                                    .fill(if is_capturing {
                                        FL_ACCENT_ORANGE
                                    } else {
                                        ui.visuals().widgets.inactive.bg_fill
                                    })
                                    .min_size(egui::vec2(76.0, 30.0));
                            let capture_response = ui
                                .add_enabled(capture_enabled, capture_button)
                                .on_hover_text(if is_capturing {
                                    "Stop capturing and insert this performance into the Playlist"
                                } else {
                                    "Capture this Session View performance into the Playlist \
                                     (needs Session Mode on and the transport playing)"
                                });
                            if capture_response.clicked() {
                                if is_capturing {
                                    self.transport.set_capturing(false);
                                    let (events, final_tick) = self
                                        .capture_log
                                        .lock()
                                        .map(|log| log.clone())
                                        .unwrap_or_default();
                                    if events.is_empty() {
                                        self.capture_message =
                                            Some((false, "Nothing was captured".to_string()));
                                    } else {
                                        let ticks_per_second = audio::ticks_per_second(
                                            song.bpm_at(self.transport.current_tick()),
                                        );
                                        song.insert_captured_session_performance(
                                            &events,
                                            final_tick,
                                            ticks_per_second,
                                        );
                                        self.capture_message = Some((
                                            true,
                                            "Captured this performance into the Playlist"
                                                .to_string(),
                                        ));
                                    }
                                } else {
                                    self.transport.set_capturing(true);
                                    self.capture_message = None;
                                }
                            }
                            if let Some((ok, message)) = &self.capture_message {
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
                                    self.capture_log.clone(),
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

        session_flex_editor_window_ui(
            ui.ctx(),
            song,
            self.sample_rate,
            &mut self.session_flex_editor,
            &mut self.session_flex_editor_mode,
            &mut self.session_flex_editor_raw,
            &mut self.session_flex_marker_drag,
            &mut self.session_flex_note_drag,
            &self.track_effect_slots,
            &self.send_effect_slots,
            &self.master_effect_slots,
            &mut self.session_flex_automation_drag,
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
        let mut rack = channel_rack_panel::ChannelRackUi {
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
                Some(
                    DeviceChainFocus::Lane { track_index, .. }
                    | DeviceChainFocus::SessionSlotLane { track_index, .. },
                ) => track_index == index,
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
            let mut mixer_ui_state = mixer_panel::MixerUi {
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
            detached: &mut self.device_panel_detached,
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
        if *device_panel.detached {
            let ctx = ui.ctx().clone();
            let mut still_open = true;
            ctx.show_viewport_immediate(
                egui::ViewportId::from_hash_of("device_panel_viewport"),
                egui::ViewportBuilder::default()
                    .with_title("Device Panel")
                    .with_inner_size(egui::vec2(420.0, 480.0)),
                |ui, _class| {
                    egui::CentralPanel::default()
                        .frame(device_panel_frame())
                        .show(ui, |ui| {
                            egui::ScrollArea::vertical().show(ui, |ui| {
                                device_panel_contents_ui(ui, song, engine_config, &mut device_panel);
                            });
                        });
                    if ui.ctx().input(|i| i.viewport().close_requested()) {
                        still_open = false;
                    }
                },
            );
            if !still_open {
                *device_panel.detached = false;
            }
        } else {
            egui::Panel::bottom("device_panel")
                .resizable(true)
                .default_size(220.0)
                .size_range(120.0..=480.0)
                .frame(device_panel_frame())
                .show(ui, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        device_panel_contents_ui(ui, song, engine_config, &mut device_panel);
                    });
                });
        }

        let piano_roll_open = self
            .selected_track
            .filter(|&i| i < song.tracks.len())
            .filter(|&i| song.tracks[i].kind == TrackKind::PianoRoll)
            .is_some();
        // Whether the (open) Piano Roll claims the shared central-area slot this frame — it always
        // wins that slot over a docked Beats window, see `beats_docked` below.
        let piano_roll_docked = piano_roll_open && !self.piano_roll_detached;

        if piano_roll_open {
            let piano_roll_frame = || {
                egui::Frame::new()
                    .fill(egui::Color32::from_rgb(36, 36, 36))
                    .inner_margin(egui::Margin::same(8))
            };
            let mut panel = piano_roll_panel::PianoRollPanelUi {
                detached: &mut self.piano_roll_detached,
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
                editing_target: &mut self.piano_roll_region,
                scroll_to: &mut self.piano_roll_scroll_to,
                track_effect_slots: &self.track_effect_slots,
                send_effect_slots: &self.send_effect_slots,
                master_effect_slots: &self.master_effect_slots,
                automation_drag: &mut self.automation_drag,
                track_automation_drag: &mut self.track_automation_drag,
            };
            if *panel.detached {
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
            } else {
                piano_roll_frame().show(ui, |ui| {
                    piano_roll_contents_ui(ui, song, current_tick, &mut panel);
                });
            }
        }

        let beats_open = self
            .selected_beats_track
            .filter(|&i| i < song.tracks.len())
            .filter(|&i| song.tracks[i].kind == TrackKind::StepGrid)
            .is_some();
        // If Piano Roll already claimed the shared central-area slot this frame, a docked Beats
        // window temporarily renders as a floating window instead of disappearing — its own
        // `beats_detached` flag is left untouched, so it docks again as soon as the slot frees up.
        let beats_docked = beats_open && !self.beats_detached && !piano_roll_docked;

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
            let mut groove = beats_panel::StepGrooveUi {
                humanize_timing_ticks: &mut self.groove_humanize_timing_ticks,
                humanize_velocity: &mut self.groove_humanize_velocity,
                template_index: &mut self.groove_template_index,
            };
            let beats_detached = &mut self.beats_detached;
            if beats_docked {
                beats_frame().show(ui, |ui| {
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
                        beats_detached,
                    );
                });
            } else {
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
                                    beats_detached,
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
        }

        // Piano Roll/Beats already rendered themselves above (either into their own floating
        // window or straight into this central area) when docked — Playlist/Session View only get
        // the shared central-area slot when neither of them is occupying it.
        if piano_roll_docked || beats_docked {
            // Nothing left to draw here this frame.
        } else if self.playlist_open {
            // Docked into the main window's remaining central area (to the right of the Channel
            // Rack panel) by default, or popped into its own OS window via `playlist_detached` —
            // same dock/detach split as the Channel Rack/Mixer.
            let playlist_frame = || {
                egui::Frame::new()
                    .fill(egui::Color32::from_rgb(30, 30, 30))
                    .inner_margin(egui::Margin::same(8))
            };
            let mut editor_targets = PlaylistEditorTargets {
                selected_track: &mut self.selected_track,
                piano_roll_region: &mut self.piano_roll_region,
                piano_roll_scroll_to: &mut self.piano_roll_scroll_to,
                selected_beats_track: &mut self.selected_beats_track,
                beats_region: &mut self.beats_region,
            };
            if self.playlist_detached {
                let ctx = ui.ctx().clone();
                let mut still_open = true;
                ctx.show_viewport_immediate(
                    egui::ViewportId::from_hash_of("playlist_viewport"),
                    egui::ViewportBuilder::default()
                        .with_title("Playlist")
                        .with_inner_size(egui::vec2(1000.0, 500.0)),
                    |ui, _class| {
                        egui::CentralPanel::default()
                            .frame(playlist_frame())
                            .show(ui, |ui| {
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
                                    &mut self.playlist_detached,
                                );
                            });
                        if ui.ctx().input(|i| i.viewport().close_requested()) {
                            still_open = false;
                        }
                    },
                );
                if !still_open {
                    self.playlist_detached = false;
                }
            } else {
                playlist_frame().show(ui, |ui| {
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
                        &mut self.playlist_detached,
                    );
                });
            }
        } else if self.session_view_open {
            let session_frame = || {
                egui::Frame::new()
                    .fill(egui::Color32::from_rgb(30, 30, 30))
                    .inner_margin(egui::Margin::same(8))
            };
            let mut session_record_click: Option<(usize, usize)> = None;
            // Hides every slot's record button outright while a Playlist recording is already in
            // progress — only one `InputRecorder` can run at a time, and this is simpler than a
            // second "blocked" reason alongside `session_slot_cell_ui`'s existing sibling-slot one.
            let record_armed_track = if self.recording.is_some() { None } else { self.record_armed_track };
            let session_recording_slot =
                self.session_recording.as_ref().map(|s| (s.track_index, s.slot_index));
            if self.session_view_detached {
                let ctx = ui.ctx().clone();
                let mut still_open = true;
                ctx.show_viewport_immediate(
                    egui::ViewportId::from_hash_of("session_view_viewport"),
                    egui::ViewportBuilder::default()
                        .with_title("Session View")
                        .with_inner_size(egui::vec2(900.0, 500.0)),
                    |ui, _class| {
                        egui::CentralPanel::default()
                            .frame(session_frame())
                            .show(ui, |ui| {
                                session_view_ui::session_view_contents_ui(
                                    ui,
                                    song,
                                    &self.transport,
                                    &self.session_slots,
                                    &mut self.session_quantize,
                                    &mut self.follow_action_editor,
                                    &mut self.session_flex_editor,
                                    &mut self.selected_track,
                                    &mut self.piano_roll_region,
                                    &mut self.selected_beats_track,
                                    &mut self.beats_region,
                                    &mut self.session_view_detached,
                                    record_armed_track,
                                    session_recording_slot,
                                    &mut session_record_click,
                                );
                            });
                        if ui.ctx().input(|i| i.viewport().close_requested()) {
                            still_open = false;
                        }
                    },
                );
                if !still_open {
                    self.session_view_detached = false;
                }
            } else {
                session_frame().show(ui, |ui| {
                    session_view_ui::session_view_contents_ui(
                        ui,
                        song,
                        &self.transport,
                        &self.session_slots,
                        &mut self.session_quantize,
                        &mut self.follow_action_editor,
                        &mut self.session_flex_editor,
                        &mut self.selected_track,
                        &mut self.piano_roll_region,
                        &mut self.selected_beats_track,
                        &mut self.beats_region,
                        &mut self.session_view_detached,
                        record_armed_track,
                        session_recording_slot,
                        &mut session_record_click,
                    );
                });
            }
            if let Some((track_index, slot_index)) = session_record_click {
                handle_session_record_click(
                    song,
                    track_index,
                    slot_index,
                    &mut self.session_recording,
                    &mut self.recording_message,
                    self.selected_input_device.as_deref(),
                    self.sample_rate,
                    self.transport.current_tick(),
                );
            }
        }
    }
}

pub(crate) fn note_name(pitch: u8) -> String {
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

pub(crate) fn lane_sample_controls(ui: &mut egui::Ui, lane: &mut Lane, sample_rate: Option<u32>) {
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
    use crate::file_ops::finish_session_recording;
    use crate::model::{SessionClip, SessionClipContent, Track};

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

    #[test]
    fn finish_session_recording_rejects_an_empty_capture() {
        let mut song = Song::demo();
        let track_index = song.add_track("Loop Vox", 5, TrackKind::Audio);
        let (ok, message) = finish_session_recording(&mut song, track_index, 0, 0, &[], 48_000, Some(48_000));
        assert!(!ok, "message: {message}");
        assert!(song.tracks[track_index].session_clips.is_empty());
    }

    #[test]
    fn finish_session_recording_creates_a_recording_clip_for_a_fresh_recording() {
        let mut song = Song::demo();
        let track_index = song.add_track("Loop Vox", 5, TrackKind::Audio);
        let samples = vec![0.5f32; 48_000]; // 1 second at 48kHz
        let (ok, _) = finish_session_recording(&mut song, track_index, 0, 0, &samples, 48_000, Some(48_000));
        assert!(ok);

        let clip = song.tracks[track_index].session_clips[0]
            .as_ref()
            .expect("slot should hold the new recording");
        let SessionClipContent::Recording(folder) = &clip.content else {
            panic!("expected Recording content");
        };
        assert_eq!(folder.takes.len(), 1);
        assert_eq!(folder.comp.len(), 1);
        assert_eq!(folder.comp[0].take_index, 0);
        let bar_ticks = song.steps_per_bar() * TICKS_PER_STEP;
        assert_eq!(folder.length_ticks % bar_ticks, 0, "loop length should be a whole number of bars");
        assert!(folder.length_ticks > 0);
    }

    #[test]
    fn finish_session_recording_rounds_a_short_recording_up_to_one_full_bar() {
        let mut song = Song::demo();
        let track_index = song.add_track("Loop Vox", 5, TrackKind::Audio);
        let samples = vec![0.5f32; 4_800]; // 0.1s at 48kHz — much shorter than a bar at 120bpm
        let (ok, _) = finish_session_recording(&mut song, track_index, 0, 0, &samples, 48_000, Some(48_000));
        assert!(ok);
        let clip = song.tracks[track_index].session_clips[0].as_ref().unwrap();
        let SessionClipContent::Recording(folder) = &clip.content else {
            panic!("expected Recording content");
        };
        let bar_ticks = song.steps_per_bar() * TICKS_PER_STEP;
        assert_eq!(folder.length_ticks, bar_ticks);
    }

    #[test]
    fn finish_session_recording_into_an_existing_recording_slot_joins_as_a_new_take() {
        let mut song = Song::demo();
        let track_index = song.add_track("Loop Vox", 5, TrackKind::Audio);
        let samples = vec![0.5f32; 48_000];
        finish_session_recording(&mut song, track_index, 0, 0, &samples, 48_000, Some(48_000));
        finish_session_recording(&mut song, track_index, 0, 0, &samples, 48_000, Some(48_000));

        let clip = song.tracks[track_index].session_clips[0].as_ref().unwrap();
        let SessionClipContent::Recording(folder) = &clip.content else {
            panic!("expected Recording content");
        };
        assert_eq!(folder.takes.len(), 2);
        assert_eq!(folder.comp[0].take_index, 1, "the just-recorded take should be the one comped in");
    }

    #[test]
    fn finish_session_recording_refuses_to_overwrite_a_slot_with_region_content() {
        let mut song = Song::demo();
        let track_index = song.add_track("Loop Vox", 5, TrackKind::Audio);
        let region = Region {
            name: "Existing".to_string(),
            start_tick: 0,
            content_length_steps: 4,
            loop_length_steps: 4,
            content: RegionContent::PianoRoll(Vec::new()),
            fade_in_ticks: 0,
            fade_out_ticks: 0,
            automation: Vec::new(),
        };
        song.tracks[track_index].session_clips.push(Some(SessionClip::from_region(&region)));

        let samples = vec![0.5f32; 48_000];
        let (ok, message) = finish_session_recording(&mut song, track_index, 0, 0, &samples, 48_000, Some(48_000));
        assert!(!ok, "message: {message}");
        assert!(matches!(
            song.tracks[track_index].session_clips[0].as_ref().unwrap().content,
            SessionClipContent::Region { .. }
        ));
    }
}
