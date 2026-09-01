//! Save/load/export, recording (Playlist and Session View), freeze/bounce-in-place, and
//! effect-chain persistence (loading a `Song`'s saved `TrackEffectConfig`s into live CLAP/
//! built-in effect instances, and snapshotting live effect state back for `save_to_file`) — the
//! file-system-and-engine-state side of the toolbar/File-menu/Session-View-record-button actions
//! driven from `SimpleDawApp::update`.

use std::path::Path;

use crate::builtin_fx::BuiltInEffect;
use crate::model::{
    AudioClip, LaunchIntent, SessionClip, SessionClipContent, Song, TICKS_PER_STEP, TakeFolder, TrackEffectConfig,
    TrackKind,
};
use crate::plugin_host::{
    DawHost, EffectInstance, LoadedEffect, MasterEffectSlots, PluginGuiHandle, SendEffectSlots, SubmixEffectSlots,
    TrackEffectSlots,
};
use crate::{SessionRecordingSession, audio, audio_input, plugin_host, session_view_ui};
use clack_host::prelude::PluginInstance;

pub(crate) fn perform_save(song: &Song, path: &str) -> (bool, String) {
    let path = std::path::Path::new(path.trim());
    match song.save_to_file(path) {
        Ok(()) => (true, format!("Saved to {}", path.display())),
        Err(err) => (false, format!("{err:#}")),
    }
}

pub(crate) fn perform_load(path: &str, sample_rate: Option<u32>) -> Result<Song, String> {
    let path = std::path::Path::new(path.trim());
    Song::load_from_file(path, sample_rate).map_err(|err| format!("{err:#}"))
}

/// Opens a native file picker seeded to `current_path`'s parent directory (if any). Passing
/// `save_as` picks a save dialog with that suggested filename; `None` picks an open dialog.
/// Returns the chosen path's display string, or `None` if the dialog was cancelled.
pub(crate) fn browse_for_file(
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

pub(crate) fn perform_export(song: &Song, sample_rate: u32, loops: u32, path: &str) -> (bool, String) {
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
pub(crate) fn finish_recording(
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

/// The Session View slot record button's stop-side logic — `finish_recording`'s counterpart for a
/// session slot instead of a Playlist `start_tick`. A slot that already holds `SessionClipContent::
/// Recording` joins the existing folder as a new take, comped in immediately (overdub-style
/// re-record, same behavior as re-recording onto the same Playlist `start_tick`); an empty or
/// differently-typed slot gets a fresh one. The recorded loop length is rounded up to the next
/// whole bar (at the tempo in effect when recording started) rather than left as the raw captured
/// duration — a session loop needs *some* bar-aligned length to advance a launch-quantized
/// playhead against, and rounding up keeps every recorded frame rather than truncating the tail
/// (accepting a trailing silence gap if recording was stopped a little early).
pub(crate) fn finish_session_recording(
    song: &mut Song,
    track_index: usize,
    slot_index: usize,
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
    let existing_slot = track.session_clips.get(slot_index).and_then(|slot| slot.as_ref());
    let has_existing_recording =
        existing_slot.is_some_and(|clip| matches!(clip.content, SessionClipContent::Recording(_)));
    // Refuse to land into a slot already holding `Region`/`Audio` content — recording only ever
    // starts into an empty slot or overdubs an existing `Recording` (see `SessionClipContent::
    // Recording`'s doc comment); silently overwriting anything else would destroy it. The UI layer
    // (`session_view_ui::session_slot_cell_ui`'s `can_record_here`) already keeps this from being
    // reachable through the record button, but this function shouldn't rely on that alone.
    if existing_slot.is_some() && !has_existing_recording {
        return (
            false,
            format!("Recorded {}, but slot {} already holds other content", path.display(), slot_index + 1),
        );
    }

    let duration_seconds = samples.len() as f64 / captured_sample_rate.max(1) as f64;
    let bar_ticks = (song.steps_per_bar() * TICKS_PER_STEP).max(1);
    let raw_ticks = (duration_seconds * audio::ticks_per_second(song.bpm_at(start_tick)))
        .ceil()
        .max(1.0) as usize;
    let length_ticks = raw_ticks.div_ceil(bar_ticks) * bar_ticks;

    let track = &mut song.tracks[track_index];
    if track.session_clips.len() <= slot_index {
        track.session_clips.resize(slot_index + 1, None);
    }
    let take_index = if has_existing_recording {
        let Some(Some(clip)) = track.session_clips.get_mut(slot_index) else {
            return (false, format!("Recorded {}, but its slot no longer exists", path.display()));
        };
        let SessionClipContent::Recording(folder) = &mut clip.content else {
            return (false, format!("Recorded {}, but its slot no longer exists", path.display()));
        };
        folder.add_take_and_activate(file_path)
    } else {
        let folder = TakeFolder::new(0, length_ticks, file_path);
        let clip = SessionClip::from_recording(format!("Recording {}", slot_index + 1), folder);
        track.session_clips[slot_index] = Some(clip);
        0
    };
    if let Some(rate) = engine_sample_rate
        && let Some(SessionClipContent::Recording(folder)) = track
            .session_clips
            .get_mut(slot_index)
            .and_then(|slot| slot.as_mut())
            .map(|clip| &mut clip.content)
    {
        folder.takes[take_index].load(rate);
    }
    (true, format!("Recorded {}", path.display()))
}

/// A session slot's own record button's whole click handler (`session_view_ui::
/// session_slot_cell_ui`'s button, surfaced back here via `record_click` — see that param's doc
/// comment on why this module owns the actual `InputRecorder` instead of `session_view_ui.rs`).
/// A click on the slot already being recorded into stops it (`finish_session_recording`, then
/// immediately queues a `Play` launch for the slot so the just-recorded loop starts right away,
/// matching Ableton's own "recording stops, playback begins" handoff); a click on any other
/// eligible slot starts a fresh capture, first sending `Stop` to every *other* slot on that same
/// track — the confirmed design: starting a recording is itself a launch, so it follows the same
/// per-track exclusivity any other launch does, no special exemption. A click while
/// `session_recording` already targets a different slot is unreachable in practice (that slot's
/// button renders disabled — see `session_slot_cell_ui`) but is still a safe no-op here regardless.
#[allow(clippy::too_many_arguments)]
pub(crate) fn handle_session_record_click(
    song: &mut Song,
    track_index: usize,
    slot_index: usize,
    session_recording: &mut Option<SessionRecordingSession>,
    recording_message: &mut Option<(bool, String)>,
    selected_input_device: Option<&str>,
    engine_sample_rate: Option<u32>,
    current_tick: usize,
) {
    let is_this_slot = session_recording
        .as_ref()
        .is_some_and(|s| s.track_index == track_index && s.slot_index == slot_index);
    if is_this_slot {
        let Some(session) = session_recording.take() else { return };
        let SessionRecordingSession { track_index, slot_index, recorder, start_tick } = session;
        let captured_sample_rate = recorder.sample_rate;
        let samples = recorder.stop();
        *recording_message = Some(finish_session_recording(
            song,
            track_index,
            slot_index,
            start_tick,
            &samples,
            captured_sample_rate,
            engine_sample_rate,
        ));
        session_view_ui::send_launch_request(song, track_index, slot_index, LaunchIntent::Play);
        return;
    }
    if session_recording.is_some() {
        return;
    }
    let input_gain = song.tracks.get(track_index).map_or(1.0, |t| t.input_gain);
    match audio_input::InputRecorder::start(selected_input_device, input_gain) {
        Ok(recorder) => {
            let other_slot_count =
                song.tracks.get(track_index).map_or(0, |t| t.session_clips.len());
            for other_slot_index in 0..other_slot_count {
                if other_slot_index != slot_index {
                    session_view_ui::send_launch_request(
                        song,
                        track_index,
                        other_slot_index,
                        LaunchIntent::Stop,
                    );
                }
            }
            *session_recording = Some(SessionRecordingSession {
                track_index,
                slot_index,
                recorder,
                start_tick: current_tick,
            });
            *recording_message = None;
        }
        Err(err) => *recording_message = Some((false, format!("{err:#}"))),
    }
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
pub(crate) fn freeze_track(
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
pub(crate) fn unfreeze_track(song: &mut Song, track_index: usize) {
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
pub(crate) fn bounce_track_in_place(
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
pub(crate) fn sync_song_effects(
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
pub(crate) fn load_effect(
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
pub(crate) fn build_effect_chain(
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
pub(crate) fn apply_chain_specs_at(
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
pub(crate) fn apply_loaded_effects(
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
