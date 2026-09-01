//! The tick clock and trigger engine: `Sequencer::process` synthesizes one buffer's worth of dry,
//! unclipped mono-per-track samples, triggering notes/steps/audio clips/take folders as tick
//! boundaries are crossed and (in Session View mode) launching/advancing clip slots — the exact
//! same synthesis path shared by the real-time `build_playback_stream` callback and the offline
//! `render_song_to_wav`/`render_track_to_buffer` bounce, so a bounce sounds like what plays live.
//! `ticks_per_second`/`arrangement_length_ticks` (tick-domain conversions) and
//! `SessionSlotHandles`/`CaptureLogHandle` (the audio-thread-publishes-a-clone handles for the
//! Session View grid and "Capture to Arrangement") live here too since they're this module's own
//! public surface.

use std::sync::{Arc, Mutex};

use crate::model::{
    CaptureEvent, CaptureEventKind, FollowAction, Lane, LaunchIntent, LaunchMode, Note, RegionContent, SessionClip,
    SessionClipContent, Song, StepData, SynthEngine, TICKS_PER_STEP, Track, TrackKind, TrackOutput,
};
use crate::session::{self, SlotState};

use super::sample_voice::TrackVoices;
use super::voice_dsp::pitch_to_freq;
use super::{SAMPLE_VOICE_COUNT, STEPS_PER_BEAT, VOICE_COUNT};

/// A piano-roll note's gate time (how long it's "held" before Release begins — see
/// `SynthParams`) is its own length in seconds; this is just a floor against a degenerate
/// zero-length note.
const MIN_NOTE_GATE_SECONDS: f32 = 0.01;
/// Fixed fade-in/out applied at every `TakeFolder` comp-segment boundary (see `Sequencer::process`'s
/// take-folder trigger loop) so switching which take is heard mid-folder doesn't click — short
/// enough to be inaudible as a fade, long enough to smooth a hard amplitude discontinuity.
const TAKE_FOLDER_CROSSFADE_SECONDS: f32 = 0.005;


/// Owns each track's voice pools and the shared tick clock. `process` synthesizes one buffer's
/// worth of mono samples *per track* (dry, unclipped — no gain/soft-clip applied here), triggering
/// notes as tick boundaries are crossed (see `model::TICKS_PER_STEP` — step-grid lanes trigger
/// every `TICKS_PER_STEP`-th tick, piano-roll notes trigger on their exact tick). Callers are
/// responsible for summing tracks (optionally through per-track effects) into a master bus and
/// applying `MASTER_GAIN`/soft-clipping — see `build_playback_stream` and `render_song_to_wav`.
/// Shared between the real-time cpal callback and the offline WAV exporter so the two never drift
/// apart into subtly different playback.
/// Ticks-per-second at `bpm` — the conversion an `AudioClip`'s decoded real-time duration needs
/// to become a tick span. Exposed (rather than kept private to `arrangement_length_ticks`) so the
/// Playlist UI's audio-clip block width (`main.rs`) uses this exact same formula instead of a
/// second copy that could drift out of sync with it.
pub fn ticks_per_second(bpm: f32) -> f64 {
    (bpm.max(1.0) as f64) * STEPS_PER_BEAT * TICKS_PER_STEP as f64 / 60.0
}

/// The song's total loop length in ticks: the furthest point any region or audio clip reaches.
/// Both live playback (`Sequencer::process`) and `render_song_to_wav` derive their loop/song
/// length from this single formula, so they can never drift apart. `pub(crate)` (rather than
/// private) so `main.rs`'s track-wide automation panel can use the same span for its graph's
/// x-axis that a `Track::automation` lane's absolute ticks are actually evaluated against, instead
/// of a second formula that could drift out of sync with it.
pub(crate) fn arrangement_length_ticks(song: &Song) -> usize {
    let pattern_end = song
        .tracks
        .iter()
        .flat_map(|track| track.regions.iter())
        .map(|region| region.start_tick + region.loop_length_steps * TICKS_PER_STEP)
        .max()
        .unwrap_or(0);

    // An untrimmed audio clip's duration is however long its decoded buffer is, in real seconds,
    // converted to ticks at the tempo in effect where the clip starts (see `Song::bpm_at`) so a
    // recording never gets truncated by the arrangement looping underneath it — a trimmed clip
    // uses its own stored `length_ticks` instead (see `AudioClip::effective_length_ticks`). If a
    // tempo change lands partway through the clip, its tick length is still computed at its own
    // starting tempo throughout — a documented approximation, the same kind
    // `render_song_to_wav`'s per-chunk-not-per-sample tempo resolution already accepts.
    let audio_end = song
        .tracks
        .iter()
        .flat_map(|track| track.audio_clips.iter())
        .filter_map(|clip| {
            clip.buffer.as_ref()?;
            let duration_ticks =
                clip.effective_length_ticks(ticks_per_second(song.bpm_at(clip.start_tick)));
            Some(clip.start_tick + duration_ticks)
        })
        .max()
        .unwrap_or(0);

    // A `TakeFolder`'s span is explicit (`length_ticks`, frozen at the first take's own recorded
    // duration — see `model::TakeFolder`), unlike a plain `AudioClip`'s implicit-until-trimmed one.
    let take_folder_end = song
        .tracks
        .iter()
        .flat_map(|track| track.take_folders.iter())
        .map(|folder| folder.start_tick + folder.length_ticks)
        .max()
        .unwrap_or(0);

    pattern_end
        .max(audio_end)
        .max(take_folder_end)
        .max(TICKS_PER_STEP)
        .max(1)
}

/// Session View clip-slot playback state, published once per audio callback (see
/// `build_playback_stream`) for the UI thread to poll each frame — the `Sequencer::session_slots`
/// counterpart of `metering::MeterHandles`, same "audio thread owns and publishes, UI thread reads
/// a cheap clone" split, since the live queued/playing/stopped state genuinely only exists on the
/// audio thread (see `model::SessionLaunchRequest`'s doc comment). Outer index is track, inner is
/// slot, same shape as `Sequencer::session_slots` itself.
pub type SessionSlotHandles = Arc<Mutex<Vec<Vec<SlotState>>>>;

/// A fresh, empty `SessionSlotHandles` — mirrors `metering::new_track_meter_handles`.
pub fn new_session_slot_handles() -> SessionSlotHandles {
    Arc::new(Mutex::new(Vec::new()))
}

/// `Sequencer::capture_log` plus `Sequencer::capture_tick` (the tick to close any still-open
/// interval at — see `model::Song::insert_captured_session_performance`'s `final_relative_tick`),
/// published once per callback for `main.rs`'s "Capture to Arrangement" toolbar button to read the
/// moment it turns capturing back off — same "audio thread owns and publishes, UI thread reads a
/// cheap clone" split as `SessionSlotHandles`.
pub type CaptureLogHandle = Arc<Mutex<(Vec<CaptureEvent>, usize)>>;

/// A fresh, empty `CaptureLogHandle`.
pub fn new_capture_log_handle() -> CaptureLogHandle {
    Arc::new(Mutex::new((Vec::new(), 0)))
}

pub(crate) struct Sequencer {
    sample_rate: f32,
    pub(crate) track_voices: Vec<TrackVoices>,
    tick_index: usize,
    samples_until_next_tick: f64,
    last_triggered_tick: usize,
    /// This track's current region-fade gain (0.0..1.0, see `Region::fade_gain_at`), recomputed
    /// once per tick (in lockstep with the trigger loop, not per-sample — see `process`'s doc
    /// comment on why tick granularity is smooth enough here) and held across every sample until
    /// the next tick. Reverts to 1.0 whenever no region is currently active on that track, even if
    /// a still-ringing voice's own envelope (untouched by this) plays on past the region's end —
    /// this engine already tolerates that same "notes ring past their trigger" gap for mute/regions
    /// generally, not something fades need to newly solve. Indexed the same as `track_voices`.
    track_fade_gain: Vec<f32>,
    /// How many samples into its decaying click envelope the metronome currently is —
    /// `>= metronome_click_len` means silent (see `next_metronome_click_sample`).
    metronome_click_pos: usize,
    metronome_click_len: usize,
    metronome_click_freq: f32,
    /// Session View clip-slot playback state, owned here rather than in `Song` since it must
    /// survive across the per-callback `Song` snapshot clone (see `model::SessionLaunchRequest`'s
    /// doc comment). Outer index is track, inner is slot — resized to match
    /// `Track::session_clips` each `trigger_session_clips` call, the same per-callback resize
    /// pattern `track_voices` already uses.
    pub(crate) session_slots: Vec<Vec<SlotState>>,
    /// The last-seen `SessionLaunchRequest::generation` per track/slot, index-aligned with
    /// `session_slots` — see `model::SessionLaunchRequest`'s doc comment on the edge-triggered
    /// click protocol this implements.
    session_last_seen_generation: Vec<Vec<u64>>,
    /// Which `TrackVoices::sample_voices` index is currently looping a slot's `SessionClipContent::
    /// Audio` playback, index-aligned with `session_slots` — `None` for an empty/non-audio slot or
    /// one that isn't playing. Unlike step-grid/piano-roll content (which never needs cancelling —
    /// a triggered synth voice or one-shot sample just rings out on its own), a looping
    /// `SampleVoice` never stops itself, so `trigger_session_clips` needs this handle to hard-cut
    /// it the moment a slot's state reaches `SlotState::Stopped`.
    session_audio_voice: Vec<Vec<Option<usize>>>,
    /// Whether `Transport::capturing` read as true on the *previous* `trigger_session_clips` call —
    /// lets that function detect the off→on/on→off edges itself (resetting `capture_log`/
    /// `capture_tick` on the former, nothing special on the latter — `main.rs` reads whatever's
    /// last published right after flipping the flag off) without `main.rs` needing its own
    /// round-trip through `Song` the way a slot click does (see `model::SessionLaunchRequest`'s
    /// doc comment on why *that* needs one — this doesn't, since nothing here needs to survive a
    /// `Song` snapshot clone).
    was_capturing: bool,
    /// Ticks since capturing was last armed — reset to `0` on the off→on edge. Deliberately
    /// separate from `tick_index`, which wraps around the Playlist's own current length even in
    /// Session View (see `process`'s doc comment) — a capture spanning multiple wraps needs a
    /// counter that doesn't.
    pub(crate) capture_tick: usize,
    /// Every slot start/stop logged since capturing was last armed — see `model::CaptureEvent`.
    /// Published each callback via `CaptureLogHandle`, the same "audio thread owns and publishes,
    /// UI thread reads a cheap clone" split `SessionSlotHandles` already uses.
    pub(crate) capture_log: Vec<CaptureEvent>,
}

/// One beat's worth of ticks (see `STEPS_PER_BEAT`/`TICKS_PER_STEP`) — the metronome clicks once
/// per beat, not once per step.
const METRONOME_BEAT_TICKS: usize = STEPS_PER_BEAT as usize * TICKS_PER_STEP;
const METRONOME_CLICK_SECONDS: f32 = 0.03;
const METRONOME_CLICK_HZ: f32 = 1000.0;
const METRONOME_ACCENT_HZ: f32 = 1600.0;
const METRONOME_GAIN: f32 = 0.5;

impl Sequencer {
    pub(crate) fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            track_voices: Vec::new(),
            tick_index: 0,
            samples_until_next_tick: 0.0,
            last_triggered_tick: 0,
            track_fade_gain: Vec::new(),
            metronome_click_pos: 0,
            metronome_click_len: 0,
            metronome_click_freq: 0.0,
            session_slots: Vec::new(),
            session_last_seen_generation: Vec::new(),
            session_audio_voice: Vec::new(),
            was_capturing: false,
            capture_tick: 0,
            capture_log: Vec::new(),
        }
    }

    /// Rewinds the tick clock to the start without touching in-flight voices.
    pub(crate) fn reset_position(&mut self) {
        self.tick_index = 0;
        self.samples_until_next_tick = 0.0;
        self.last_triggered_tick = 0;
        self.metronome_click_pos = 0;
    }

    /// Starts a fresh decaying click envelope — `accent` picks a higher pitch for the downbeat
    /// (tick 0), matching the usual "first beat sounds different" metronome convention.
    fn trigger_metronome_click(&mut self, accent: bool) {
        self.metronome_click_pos = 0;
        self.metronome_click_len = (self.sample_rate * METRONOME_CLICK_SECONDS) as usize;
        self.metronome_click_freq = if accent {
            METRONOME_ACCENT_HZ
        } else {
            METRONOME_CLICK_HZ
        };
    }

    /// Renders the next metronome sample: a short decaying sine burst, or silence once the
    /// current click has fully decayed.
    fn next_metronome_click_sample(&mut self) -> f32 {
        if self.metronome_click_pos >= self.metronome_click_len {
            return 0.0;
        }
        let t = self.metronome_click_pos as f32 / self.sample_rate;
        let envelope = 1.0 - (self.metronome_click_pos as f32 / self.metronome_click_len as f32);
        let sample = (2.0 * std::f32::consts::PI * self.metronome_click_freq * t).sin()
            * envelope
            * METRONOME_GAIN;
        self.metronome_click_pos += 1;
        sample
    }

    /// The tick most recently triggered (for UI playhead display).
    pub(crate) fn current_tick(&self) -> usize {
        self.last_triggered_tick
    }

    /// Renders `frames` samples, writing one dry stereo mix per track into `track_out_l[i]`/
    /// `track_out_r[i]` (both resized to match `snapshot.tracks`). Track count can change between
    /// calls (e.g. after loading a different song) — `track_voices` is resized to match,
    /// discarding in-flight voices for any removed track.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn process(
        &mut self,
        snapshot: &Song,
        frames: usize,
        track_out_l: &mut Vec<Vec<f32>>,
        track_out_r: &mut Vec<Vec<f32>>,
        metronome_enabled: bool,
        metronome_out: &mut Vec<f32>,
        session_mode: bool,
        session_quantize_ticks: usize,
        capturing: bool,
    ) {
        while self.track_voices.len() < snapshot.tracks.len() {
            self.track_voices.push(TrackVoices::new());
        }
        self.track_voices.truncate(snapshot.tracks.len());
        self.track_fade_gain.resize(snapshot.tracks.len(), 1.0);
        if session_mode {
            // The region-fade reset below only runs on the Playlist-arrangement path (it lives
            // inside the region loop `trigger_session_clips` replaces) — Session View has no
            // fades in v1, so reset explicitly here instead, once per buffer rather than per
            // tick, to avoid a stale fade value left over from a previous Arrangement-mode
            // session bleeding into Session playback.
            self.track_fade_gain.fill(1.0);
        }

        track_out_l.resize_with(snapshot.tracks.len(), Vec::new);
        track_out_r.resize_with(snapshot.tracks.len(), Vec::new);
        for buf in track_out_l.iter_mut().chain(track_out_r.iter_mut()) {
            buf.clear();
            buf.resize(frames, 0.0);
        }
        metronome_out.clear();
        metronome_out.resize(frames, 0.0);

        let arrangement_len_ticks = arrangement_length_ticks(snapshot);
        self.tick_index %= arrangement_len_ticks;
        // Re-derived at every tick boundary below (not just once per buffer) from
        // `Song::bpm_at(self.tick_index)`, so a `Song::tempo_map` change takes effect at exactly
        // the tick it's placed on — full per-tick precision for note/step triggering, unlike the
        // buffer-granularity precision `build_playback_stream`'s continuous automation and
        // `mix_song_to_wav_buffer`'s offline mixdown settle for (see their own comments).
        let mut samples_per_tick =
            samples_per_tick_at(self.sample_rate as f64, snapshot.bpm_at(self.tick_index));
        // When any track *or submix bus* is soloed, only soloed tracks (and tracks routed into a
        // soloed submix) are audible; every other track goes silent regardless of its own mute
        // state — the same "solo wins" rule extended to submix groups, so soloing a submix acts
        // like soloing every one of its member tracks at once. Silencing at this synthesis stage
        // (rather than only gating the submix's own summed output later in the mixdown) means a
        // muted/non-soloed-out submix costs nothing beyond this check — its member tracks simply
        // never render.
        let any_solo = snapshot.tracks.iter().any(|track| track.solo)
            || snapshot.submixes.iter().any(|submix| submix.solo);
        let submix_for = |track: &Track| match track.output {
            TrackOutput::Submix(index) => snapshot.submixes.get(index),
            TrackOutput::Master => None,
        };
        let track_silent = |track: &Track| {
            let soloed = track.solo || submix_for(track).is_some_and(|submix| submix.solo);
            if any_solo {
                !soloed
            } else {
                track.muted || submix_for(track).is_some_and(|submix| submix.muted)
            }
        };

        for sample_index in 0..frames {
            if self.samples_until_next_tick <= 0.0 {
                self.last_triggered_tick = self.tick_index;
                if metronome_enabled && self.tick_index % METRONOME_BEAT_TICKS == 0 {
                    self.trigger_metronome_click(self.tick_index == 0);
                }
                if session_mode {
                    // Session View is a mode switch, not an overlay (see `Transport::session_mode`'s
                    // doc comment): while active, none of the Playlist arrangement's regions/audio
                    // clips/take folders below trigger — only Session View's own clip slots do.
                    self.trigger_session_clips(snapshot, session_quantize_ticks, &track_silent, capturing);
                } else {
                // A track's regions are independently positioned and may overlap in time with
                // each other (unusual, but not prevented) — every region active at this tick on
                // this track contributes; tracks with nothing placed at this tick stay silent.
                for (track_index, track) in snapshot.tracks.iter().enumerate() {
                    if track_silent(track) {
                        continue;
                    }
                    // No region active this tick reverts to unfaded (1.0) — see
                    // `track_fade_gain`'s doc comment. Recomputed from scratch each tick rather
                    // than only lowered, so raising a region's fade values back down (or moving
                    // off a region entirely) takes effect immediately, not just on the next fade-in.
                    self.track_fade_gain[track_index] = 1.0;
                    // A frozen track's own notes/steps are silenced — `frozen_clip`, triggered
                    // below alongside audio clips, already carries this track's baked-down output.
                    if track.frozen {
                        continue;
                    }
                    for region in track.regions.iter().filter(|region| {
                        self.tick_index >= region.start_tick
                            && self.tick_index
                                < region.start_tick + region.loop_length_steps * TICKS_PER_STEP
                    }) {
                        // Offset from the region's on-timeline start — feeds both the fade curve
                        // (against the on-timeline span) and, modulo the content's own length
                        // below, which step/note is playing right now.
                        let region_offset_ticks = self.tick_index - region.start_tick;
                        self.track_fade_gain[track_index] = self.track_fade_gain[track_index]
                            .min(region.fade_gain_at(region_offset_ticks));
                        // The region's on-timeline span may be shorter than its own content
                        // (truncating it) or longer (looping it) — both fall out of this modulo.
                        let region_local_tick =
                            region_offset_ticks % region.content_length_ticks().max(1);
                        let tv = &mut self.track_voices[track_index];
                        match &region.content {
                            RegionContent::StepGrid(lanes) => {
                                for lane in lanes {
                                    if let Some(step) = step_triggering_at(lane, region_local_tick) {
                                        trigger_lane_step(tv, track, lane, step, self.sample_rate);
                                    }
                                }
                            }
                            RegionContent::PianoRoll(notes) => {
                                for note in notes {
                                    if note.start_tick != region_local_tick {
                                        continue;
                                    }
                                    trigger_piano_roll_note(
                                        tv,
                                        track,
                                        note,
                                        self.sample_rate,
                                        samples_per_tick,
                                    );
                                }
                            }
                        }
                    }
                }

                // Audio clips live directly on their track at an absolute song tick (see
                // `model::AudioClip`), not inside a `Region` — so unlike step-grid/piano-roll
                // content above, this doesn't go through the per-track region loop at all.
                // `frozen_clip` (a track's own baked-down audio, see `Track::frozen`) is triggered
                // the same way, right here, whichever track kind it's attached to — a frozen track
                // never also triggers its own regions/audio_clips/take_folders (see the guards
                // below and the region loop above), so exactly one of "live content" or "frozen
                // clip" plays for any given track.
                for (track_index, track) in snapshot.tracks.iter().enumerate() {
                    if track_silent(track) {
                        continue;
                    }
                    let tv = &mut self.track_voices[track_index];
                    if track.frozen {
                        let Some(clip) = &track.frozen_clip else { continue };
                        if clip.start_tick != self.tick_index {
                            continue;
                        }
                        let Some(buffer) = &clip.buffer else { continue };
                        let tps = ticks_per_second(snapshot.bpm_at(self.tick_index));
                        let frames_per_tick = buffer.sample_rate as f64 / tps;
                        let length_frames =
                            (clip.effective_length_ticks(tps) as f64 * frames_per_tick).round()
                                as usize;
                        let start_frame = clip.source_start_frame;
                        let end_frame = start_frame.saturating_add(length_frames);
                        let fade_in_frames =
                            (clip.fade_in_ticks as f64 * frames_per_tick).round() as usize;
                        let fade_out_frames =
                            (clip.fade_out_ticks as f64 * frames_per_tick).round() as usize;
                        tv.sample_voices[tv.next_sample_voice].trigger_clip(
                            buffer.clone(),
                            clip.gain,
                            start_frame,
                            end_frame,
                            fade_in_frames,
                            fade_out_frames,
                            false,
                        );
                        tv.next_sample_voice = (tv.next_sample_voice + 1) % SAMPLE_VOICE_COUNT;
                        continue;
                    }
                    if track.kind != TrackKind::Audio {
                        continue;
                    }
                    for clip in &track.audio_clips {
                        if clip.start_tick != self.tick_index {
                            continue;
                        }
                        let Some(buffer) = &clip.buffer else { continue };
                        // Trim/fade are stored in ticks (source-offset excepted — see
                        // `model::AudioClip`) and converted to frames here, at the tempo in effect
                        // where the clip starts, matching `arrangement_length_ticks`'s and
                        // `effective_length_ticks`'s own documented approximation.
                        let tps = ticks_per_second(snapshot.bpm_at(self.tick_index));
                        let frames_per_tick = buffer.sample_rate as f64 / tps;
                        let length_frames =
                            (clip.effective_length_ticks(tps) as f64 * frames_per_tick).round()
                                as usize;
                        let start_frame = clip.source_start_frame;
                        let end_frame = start_frame.saturating_add(length_frames);
                        let fade_in_frames =
                            (clip.fade_in_ticks as f64 * frames_per_tick).round() as usize;
                        let fade_out_frames =
                            (clip.fade_out_ticks as f64 * frames_per_tick).round() as usize;
                        tv.sample_voices[tv.next_sample_voice].trigger_clip(
                            buffer.clone(),
                            clip.gain,
                            start_frame,
                            end_frame,
                            fade_in_frames,
                            fade_out_frames,
                            false,
                        );
                        tv.next_sample_voice = (tv.next_sample_voice + 1) % SAMPLE_VOICE_COUNT;
                    }
                    // Take Folders (see `model::TakeFolder`) trigger one `SampleVoice` per comp
                    // segment that starts on this tick, from the same `sample_voices` pool plain
                    // audio clips use above — a comp segment is windowed into its take's buffer
                    // exactly like a trimmed `AudioClip` is windowed into its own (same
                    // `trigger_clip`), with a small fixed crossfade at every segment's edges so
                    // switching takes mid-folder doesn't click.
                    for folder in &track.take_folders {
                        let tps = ticks_per_second(snapshot.bpm_at(folder.start_tick));
                        let frames_per_tick_for = |take: &crate::model::Take| {
                            take.buffer.as_ref().map(|b| b.sample_rate as f64 / tps)
                        };
                        for segment in &folder.comp {
                            let abs_start_tick = folder.start_tick + segment.start_tick;
                            if abs_start_tick != self.tick_index {
                                continue;
                            }
                            let Some(take) = folder.takes.get(segment.take_index) else {
                                continue;
                            };
                            let Some(buffer) = &take.buffer else { continue };
                            let Some(frames_per_tick) = frames_per_tick_for(take) else {
                                continue;
                            };
                            let start_frame =
                                (segment.start_tick as f64 * frames_per_tick).round() as usize;
                            let end_frame =
                                (segment.end_tick as f64 * frames_per_tick).round() as usize;
                            let crossfade_frames =
                                (TAKE_FOLDER_CROSSFADE_SECONDS * buffer.sample_rate as f32) as usize;
                            tv.sample_voices[tv.next_sample_voice].trigger_clip(
                                buffer.clone(),
                                folder.gain,
                                start_frame,
                                end_frame,
                                crossfade_frames,
                                crossfade_frames,
                                false,
                            );
                            tv.next_sample_voice = (tv.next_sample_voice + 1) % SAMPLE_VOICE_COUNT;
                        }
                    }
                }
                }
                // The tempo *at the tick that just fired* (not the one it's about to advance to)
                // governs how long that tick lasts — otherwise the single tick immediately before
                // a `Song::tempo_map` point would borrow the new tempo one tick early.
                samples_per_tick =
                    samples_per_tick_at(self.sample_rate as f64, snapshot.bpm_at(self.tick_index));
                self.tick_index = (self.tick_index + 1) % arrangement_len_ticks;
                self.samples_until_next_tick += samples_per_tick;
            }
            self.samples_until_next_tick -= 1.0;

            for (track_index, tv) in self.track_voices.iter_mut().enumerate() {
                let mut mixed_l = 0.0f32;
                let mut mixed_r = 0.0f32;
                for voice in tv.voices.iter_mut() {
                    let (l, r) = voice.next_sample();
                    mixed_l += l;
                    mixed_r += r;
                }
                let mod_slots = &snapshot.tracks[track_index].trine.mod_slots;
                for voice in tv.trine_voices.iter_mut() {
                    let (l, r) = voice.next_sample(mod_slots);
                    mixed_l += l;
                    mixed_r += r;
                }
                let wave_mod_slots = &snapshot.tracks[track_index].wave.mod_slots;
                for voice in tv.wave_voices.iter_mut() {
                    let (l, r) = voice.next_sample(wave_mod_slots);
                    mixed_l += l;
                    mixed_r += r;
                }
                // Audio-clip playback is still a mono source (see `SampleVoice`) — centered by
                // adding equally to both channels, same as before this feature.
                for voice in tv.sample_voices.iter_mut() {
                    let s = voice.next_sample();
                    mixed_l += s;
                    mixed_r += s;
                }
                // Region fades (see `track_fade_gain`) scale this track's whole mixed output for
                // the sample, same point pan/volume apply further downstream in the mixdown — not
                // per-voice, since a voice doesn't know which region (if any) triggered it.
                let fade_gain = self.track_fade_gain[track_index];
                track_out_l[track_index][sample_index] = mixed_l * fade_gain;
                track_out_r[track_index][sample_index] = mixed_r * fade_gain;
            }
            metronome_out[sample_index] = self.next_metronome_click_sample();
        }
    }

    /// Session View's per-tick trigger step — the mode-switch counterpart of the region/audio-clip
    /// loops `process` runs when `session_mode` is false (see `Transport::session_mode`'s doc
    /// comment). Called once per fired tick, for every track's every clip slot: folds in any new
    /// `Track::session_launch_requests` click, advances that slot's `session::SlotState`, and
    /// triggers its content into the same `TrackVoices` pool arrangement playback would use — safe
    /// to share since the two never run within the same `process` call.
    ///
    /// Session View plays at most one clip per track at a time (see `stop_other_session_slots`),
    /// so within one track's own slots this runs in two passes: the first (mirroring the original,
    /// pre-follow-action shape of this function) advances every slot and triggers ordinary
    /// content, collecting any slot whose loop just completed into `pending_follow_actions`; the
    /// second resolves and immediately triggers those — a follow action fires at the exact tick a
    /// clip's loop count is satisfied, not queued to a future launch-quantize boundary the way a
    /// manual click is.
    fn trigger_session_clips(
        &mut self,
        snapshot: &Song,
        session_quantize_ticks: usize,
        track_silent: &impl Fn(&Track) -> bool,
        capturing: bool,
    ) {
        while self.session_slots.len() < snapshot.tracks.len() {
            self.session_slots.push(Vec::new());
            self.session_last_seen_generation.push(Vec::new());
            self.session_audio_voice.push(Vec::new());
        }
        self.session_slots.truncate(snapshot.tracks.len());
        self.session_last_seen_generation.truncate(snapshot.tracks.len());
        self.session_audio_voice.truncate(snapshot.tracks.len());

        // The off→on/on→off edges of `Transport::capturing` — see `was_capturing`'s doc comment.
        if capturing && !self.was_capturing {
            self.capture_log.clear();
            self.capture_tick = 0;
        } else if capturing {
            self.capture_tick += 1;
        }
        self.was_capturing = capturing;

        let tick_now = self.tick_index;
        let ticks_per_second = ticks_per_second(snapshot.bpm_at(tick_now));
        let samples_per_tick = samples_per_tick_at(self.sample_rate as f64, snapshot.bpm_at(tick_now));

        for (track_index, track) in snapshot.tracks.iter().enumerate() {
            self.session_slots[track_index].resize(track.session_clips.len(), SlotState::default());
            self.session_last_seen_generation[track_index].resize(track.session_clips.len(), 0);
            self.session_audio_voice[track_index].resize(track.session_clips.len(), None);

            if track_silent(track) {
                continue;
            }

            let slot_count = track.session_clips.len();
            let mut pending_follow_actions: Vec<(usize, FollowAction, u32)> = Vec::new();

            for (slot_index, maybe_clip) in track.session_clips.iter().enumerate() {
                let Some(clip) = maybe_clip else { continue };

                // A clip's own `quantize_override` (if set) replaces the grid-wide quantize
                // setting for every boundary this slot resolves below — see that field's doc
                // comment on why it's stored as the `SessionQuantize` enum rather than raw ticks
                // (so it stays correct across a mid-song time-signature change too).
                let quantize_ticks = clip
                    .quantize_override
                    .map(|quantize| quantize.ticks(snapshot))
                    .unwrap_or(session_quantize_ticks);

                let mut state = self.session_slots[track_index][slot_index];
                let mut forced_retrigger = false;

                if matches!(clip.launch_mode, LaunchMode::Gate | LaunchMode::Repeat) {
                    // Gate/Repeat are driven by the continuous `held` signal, not a discrete
                    // click — see `SessionLaunchRequest::held`'s doc comment. An explicit Stop
                    // (e.g. the slot's context-menu "Stop" entry) is still honored on top of
                    // that, the same generation-edge check the click path below uses, so there's
                    // always a way to force-stop a held slot even if the UI's hold signal gets
                    // stuck (e.g. losing the pointer-up event).
                    let held = track
                        .session_launch_requests
                        .get(slot_index)
                        .is_some_and(|request| request.held);
                    let (new_state, force) = session::advance_held_slot(
                        state,
                        held,
                        clip.launch_mode,
                        tick_now,
                        quantize_ticks,
                    );
                    state = new_state;
                    forced_retrigger = force;
                    if let Some(request) = track.session_launch_requests.get(slot_index) {
                        let seen = &mut self.session_last_seen_generation[track_index][slot_index];
                        if request.generation != *seen {
                            *seen = request.generation;
                            if request.intent == LaunchIntent::Stop {
                                state = SlotState::Stopped;
                            }
                        }
                    }
                } else if let Some(request) = track.session_launch_requests.get(slot_index) {
                    let seen = &mut self.session_last_seen_generation[track_index][slot_index];
                    if request.generation != *seen {
                        *seen = request.generation;
                        state = session::apply_launch_request(
                            state,
                            request.intent,
                            clip.launch_mode,
                            tick_now,
                            quantize_ticks,
                        );
                    }
                }

                let loop_length_ticks = clip.loop_length_ticks(ticks_per_second);
                let before = state;
                let mut after = session::advance_slot(state, tick_now, loop_length_ticks);

                let started_playing = session::just_started_playing(before, after);
                if started_playing {
                    // Legato: continue whatever phase a sibling on this same track was already
                    // at, instead of restarting at local_tick 0 — read before the exclusivity
                    // stop below wipes that sibling's state. Still waits for the normal
                    // launch-quantize boundary (already resolved by `advance_slot` above); only
                    // the starting phase changes, matching Ableton's own Legato.
                    if clip.legato
                        && let Some(sibling_local_tick) =
                            self.playing_sibling_local_tick(track_index, slot_index)
                    {
                        after = SlotState::Playing {
                            local_tick: sibling_local_tick % loop_length_ticks.max(1),
                            loop_count: 0,
                        };
                    }
                    self.stop_other_session_slots(track_index, slot_index);
                }
                self.session_slots[track_index][slot_index] = after;

                let just_stopped = matches!(after, SlotState::Stopped)
                    && !matches!(before, SlotState::Stopped | SlotState::Queued { .. });
                if just_stopped {
                    self.stop_session_slot_audio(track_index, slot_index);
                }

                if capturing {
                    if started_playing {
                        self.capture_log.push(CaptureEvent {
                            relative_tick: self.capture_tick,
                            track_index,
                            kind: CaptureEventKind::Started { clip: Box::new(clip.clone()) },
                        });
                    }
                    if just_stopped {
                        self.capture_log.push(CaptureEvent {
                            relative_tick: self.capture_tick,
                            track_index,
                            kind: CaptureEventKind::Stopped,
                        });
                    }
                }

                let local_tick = match after {
                    SlotState::Playing { local_tick, .. } => local_tick,
                    SlotState::QueuedStop { local_tick, .. } => local_tick,
                    SlotState::Stopped | SlotState::Queued { .. } => continue,
                };
                self.trigger_session_slot_content(
                    track,
                    track_index,
                    slot_index,
                    clip,
                    local_tick,
                    session::just_started_playing(before, after) || forced_retrigger,
                    ticks_per_second,
                    samples_per_tick,
                );

                if let SlotState::Playing { loop_count, .. } = after
                    && session::just_completed_a_loop(before, after)
                {
                    let seed = follow_action_seed(tick_now, track_index, slot_index, loop_count);
                    if let Some(action) =
                        session::resolve_follow_action(loop_count, &clip.follow_action, seed)
                        && action != FollowAction::None
                    {
                        pending_follow_actions.push((slot_index, action, loop_count));
                    }
                }
            }

            for (slot_index, action, loop_count) in pending_follow_actions {
                let seed = follow_action_seed(tick_now, track_index, slot_index, loop_count) ^ 0x5EED;
                let target_index = session::follow_action_target(action, slot_index, slot_count, seed);
                self.stop_session_slot_audio(track_index, slot_index);
                self.session_slots[track_index][slot_index] = SlotState::Stopped;
                if capturing {
                    self.capture_log.push(CaptureEvent {
                        relative_tick: self.capture_tick,
                        track_index,
                        kind: CaptureEventKind::Stopped,
                    });
                }

                let Some(target_index) = target_index else { continue };
                let Some(target_clip) = &track.session_clips[target_index] else { continue };
                self.session_slots[track_index][target_index] =
                    SlotState::Playing { local_tick: 0, loop_count: 0 };
                if capturing {
                    self.capture_log.push(CaptureEvent {
                        relative_tick: self.capture_tick,
                        track_index,
                        kind: CaptureEventKind::Started { clip: Box::new(target_clip.clone()) },
                    });
                }
                self.trigger_session_slot_content(
                    track,
                    track_index,
                    target_index,
                    target_clip,
                    0,
                    true,
                    ticks_per_second,
                    samples_per_tick,
                );
            }
        }
    }

    /// The `local_tick` of whatever other slot on `track_index` is currently `Playing`/
    /// `QueuedStop`, if any — see `trigger_session_clips`'s legato handling. Since Session View
    /// plays at most one clip per track (`stop_other_session_slots`), there's at most one such
    /// sibling to find.
    fn playing_sibling_local_tick(&self, track_index: usize, exclude_slot_index: usize) -> Option<usize> {
        self.session_slots[track_index]
            .iter()
            .enumerate()
            .find_map(|(idx, state)| {
                if idx == exclude_slot_index {
                    return None;
                }
                match state {
                    SlotState::Playing { local_tick, .. } | SlotState::QueuedStop { local_tick, .. } => {
                        Some(*local_tick)
                    }
                    _ => None,
                }
            })
    }

    /// Stops every slot on `track_index` other than `keep_slot_index` — Session View plays at
    /// most one clip per track at a time, so launching one always stops whatever else was
    /// playing (or queued) on the same track, the same exclusive-per-track model Ableton's
    /// Session View uses. Hard-cuts any looping audio voice those slots were using, same cleanup
    /// `trigger_session_clips`'s ordinary manual-stop path already does.
    fn stop_other_session_slots(&mut self, track_index: usize, keep_slot_index: usize) {
        for idx in 0..self.session_slots[track_index].len() {
            if idx == keep_slot_index {
                continue;
            }
            if !matches!(self.session_slots[track_index][idx], SlotState::Stopped) {
                self.session_slots[track_index][idx] = SlotState::Stopped;
            }
            self.stop_session_slot_audio(track_index, idx);
        }
    }

    /// Hard-cuts the looping `SampleVoice` (if any) `track_index`/`slot_index` was using — a
    /// looping voice never stops itself (see `SampleVoice::looping`), so every path that fully
    /// stops a slot (a manual stop, exclusivity, a `Stop`/empty-target follow action) needs to
    /// call this. A no-op for a slot with no audio voice (empty, or step-grid/piano-roll content,
    /// which never needs cancelling — see `trigger_session_clips`'s doc comment on that).
    fn stop_session_slot_audio(&mut self, track_index: usize, slot_index: usize) {
        if let Some(voice_index) = self.session_audio_voice[track_index][slot_index].take() {
            self.track_voices[track_index].sample_voices[voice_index].buffer = None;
        }
    }

    /// Triggers `clip`'s content for `track_index`/`slot_index` at `local_tick`. Step-grid/
    /// piano-roll content is checked every call (driven purely by matching `local_tick` against
    /// grid/note positions, same as the arrangement region loop); `SessionClipContent::Audio`/
    /// `Recording` only trigger when `fresh_start` is true — a `SampleVoice` sets up once and then
    /// loops on its own (`SampleVoice::looping`), so it's only ever (re)triggered at the moment a
    /// slot starts, not every tick it continues playing. `local_tick` shifts an `Audio`/`Recording`
    /// clip's start position forward by that many ticks' worth of frames (a no-op when
    /// `local_tick == 0`, the ordinary case) — what legato/follow-action restarts need to join an
    /// already-playing phase;
    /// see `trigger_session_clips`'s doc comment on why every subsequent loop then also starts
    /// from that same shifted point rather than the clip's true frame `0` (a documented
    /// simplification, not a bug).
    ///
    /// Shared by `trigger_session_clips`'s ordinary per-tick pass and its follow-action second
    /// pass so content sitting exactly at `local_tick == 0` isn't silently missed the way it
    /// would be if the second pass only set state and waited for the next tick's ordinary pass to
    /// reach it (`advance_slot` will have already moved `local_tick` past `0` by then).
    #[allow(clippy::too_many_arguments)]
    fn trigger_session_slot_content(
        &mut self,
        track: &Track,
        track_index: usize,
        slot_index: usize,
        clip: &SessionClip,
        local_tick: usize,
        fresh_start: bool,
        ticks_per_second: f64,
        samples_per_tick: f64,
    ) {
        match &clip.content {
            SessionClipContent::Region { content, .. } => {
                let tv = &mut self.track_voices[track_index];
                match content {
                    RegionContent::StepGrid(lanes) => {
                        for lane in lanes {
                            if let Some(step) = step_triggering_at(lane, local_tick) {
                                trigger_lane_step(tv, track, lane, step, self.sample_rate);
                            }
                        }
                    }
                    RegionContent::PianoRoll(notes) => {
                        for note in notes {
                            if note.start_tick != local_tick {
                                continue;
                            }
                            trigger_piano_roll_note(tv, track, note, self.sample_rate, samples_per_tick);
                        }
                    }
                }
            }
            SessionClipContent::Audio(audio_clip) => {
                if !fresh_start {
                    return;
                }
                let Some(buffer) = &audio_clip.buffer else { return };
                let frames_per_tick = buffer.sample_rate as f64 / ticks_per_second;
                let length_frames = (audio_clip.effective_length_ticks(ticks_per_second) as f64
                    * frames_per_tick)
                    .round() as usize;
                let end_frame = audio_clip.source_start_frame.saturating_add(length_frames);
                let phase_frames = (local_tick as f64 * frames_per_tick).round() as usize;
                let start_frame = audio_clip.source_start_frame.saturating_add(phase_frames).min(end_frame);
                let fade_in_frames = (audio_clip.fade_in_ticks as f64 * frames_per_tick).round() as usize;
                let fade_out_frames = (audio_clip.fade_out_ticks as f64 * frames_per_tick).round() as usize;
                let tv = &mut self.track_voices[track_index];
                let voice_index = tv.next_sample_voice;
                tv.sample_voices[voice_index].trigger_clip(
                    buffer.clone(),
                    audio_clip.gain,
                    start_frame,
                    end_frame,
                    fade_in_frames,
                    fade_out_frames,
                    true,
                );
                tv.next_sample_voice = (voice_index + 1) % SAMPLE_VOICE_COUNT;
                self.session_audio_voice[track_index][slot_index] = Some(voice_index);
            }
            SessionClipContent::Recording(folder) => {
                if !fresh_start {
                    return;
                }
                // `comp` is always a single whole-span segment for a session recording (see
                // `SessionClipContent::Recording`'s doc comment) — the active take is whichever
                // one that segment points at, falling back to take 0 for an empty/corrupt comp.
                let take_index = folder.comp.first().map_or(0, |segment| segment.take_index);
                let Some(buffer) = folder.takes.get(take_index).and_then(|take| take.buffer.as_ref())
                else {
                    return;
                };
                let frames_per_tick = buffer.sample_rate as f64 / ticks_per_second;
                let length_frames = (folder.length_ticks as f64 * frames_per_tick).round() as usize;
                let phase_frames = (local_tick as f64 * frames_per_tick).round() as usize;
                let start_frame = phase_frames.min(length_frames);
                let crossfade_frames = (TAKE_FOLDER_CROSSFADE_SECONDS * buffer.sample_rate as f32) as usize;
                let tv = &mut self.track_voices[track_index];
                let voice_index = tv.next_sample_voice;
                tv.sample_voices[voice_index].trigger_clip(
                    buffer.clone(),
                    folder.gain,
                    start_frame,
                    length_frames,
                    crossfade_frames,
                    crossfade_frames,
                    true,
                );
                tv.next_sample_voice = (voice_index + 1) % SAMPLE_VOICE_COUNT;
                self.session_audio_voice[track_index][slot_index] = Some(voice_index);
            }
        }
    }
}

/// A varying seed for `session::resolve_follow_action`/`session::follow_action_target`'s
/// dependency-free hash — mixes the tick a loop completed with the track/slot/loop-count so two
/// different slots (or the same slot on a later loop) don't roll the same "random" outcome.
fn follow_action_seed(tick_now: usize, track_index: usize, slot_index: usize, loop_count: u32) -> u64 {
    (tick_now as u64)
        ^ ((track_index as u64) << 40)
        ^ ((slot_index as u64) << 24)
        ^ ((loop_count as u64) << 8)
}

/// Samples per sequencer tick at `sample_rate`/`bpm` — the shared clock-rate formula every
/// tick-position calculation in this file (`Sequencer::process`, `render_song_to_wav`, and
/// `build_playback_stream`'s per-sample automation lookups) must agree on exactly, or automation/
/// fades would drift out of sync with what's actually playing.
pub(crate) fn samples_per_tick_at(sample_rate: f64, bpm: f32) -> f64 {
    (sample_rate * 60.0 / (bpm.max(1.0) as f64) / STEPS_PER_BEAT / TICKS_PER_STEP as f64).max(1.0)
}

/// Total sample count spanned by ticks `0..span_ticks` at `song`'s tempo — `song.bpm` alone if
/// `song.tempo_map` is empty (or has no points inside the span), otherwise the sum of each
/// constant-tempo segment's own duration. `render_song_to_wav` uses this instead of one flat
/// `samples_per_tick_at(sample_rate, song.bpm) * span_ticks` so a bounce comes out the right total
/// length even when the tempo changes partway through.
pub(crate) fn samples_for_tick_span(song: &Song, sample_rate: f64, span_ticks: usize) -> f64 {
    let mut boundaries: Vec<usize> = std::iter::once(0)
        .chain(song.tempo_map.iter().map(|point| point.tick).filter(|&tick| tick < span_ticks))
        .chain(std::iter::once(span_ticks))
        .collect();
    boundaries.sort_unstable();
    boundaries.dedup();
    boundaries
        .windows(2)
        .map(|w| {
            let (start, end) = (w[0], w[1]);
            (end - start) as f64 * samples_per_tick_at(sample_rate, song.bpm_at(start))
        })
        .sum()
}

/// The step (if any) of `lane` that fires exactly at `region_local_tick`, honoring each active
/// step's `StepData::timing_offset_ticks` nudge off its own grid position. Only the two step
/// boundaries nearest `region_local_tick` can possibly match — `StepData::timing_offset_ticks` is
/// kept within `+/-MAX_STEP_TIMING_OFFSET_TICKS` (under half a step) by every setter, so a step
/// can never be nudged far enough to land near any other boundary.
pub(crate) fn step_triggering_at(lane: &Lane, region_local_tick: usize) -> Option<StepData> {
    let floor_step = region_local_tick / TICKS_PER_STEP;
    [floor_step, floor_step + 1].into_iter().find_map(|step_index| {
        let step = (*lane.steps.get(step_index)?)?;
        let target_tick = (step_index * TICKS_PER_STEP) as i64 + step.timing_offset_ticks as i64;
        (target_tick == region_local_tick as i64).then_some(step)
    })
}

/// Triggers `lane`'s synth/sample voice for `step` (already looked up via `step_triggering_at`)
/// into `tv` — the step-grid trigger-dispatch body shared by `Sequencer::process`'s arrangement
/// region loop and its Session View clip loop (`trigger_session_clips`), so a synth-engine change
/// only ever needs updating in one place.
fn trigger_lane_step(tv: &mut TrackVoices, track: &Track, lane: &Lane, step: StepData, sample_rate: f32) {
    let velocity = step.velocity;
    if let Some(sample) = &lane.sample {
        tv.sample_voices[tv.next_sample_voice].trigger(sample.clone(), velocity);
        tv.next_sample_voice = (tv.next_sample_voice + 1) % SAMPLE_VOICE_COUNT;
        return;
    }
    let freq = pitch_to_freq(lane.pitch);
    // A lane with its own synth (see `Lane::synth_override`) renders with that instead of the
    // track's — lets a step-grid track mix synth patches per lane (kick on one, hi-hat on
    // another).
    let (engine, synth, trine, wave) = if lane.synth_override {
        (lane.synth_engine, &lane.synth, &lane.trine, &lane.wave)
    } else {
        (track.synth_engine, &track.synth, &track.trine, &track.wave)
    };
    // Step-grid hits have no explicit length, unlike a piano-roll note — treat "attack + decay"
    // as the gate time, so Release begins right as Decay would otherwise have finished settling
    // at the sustain level.
    match engine {
        SynthEngine::Simple => {
            let gate_seconds = synth.attack_seconds + synth.decay_seconds;
            // Step-grid hits never glide — see `SynthParams::glide_seconds`.
            tv.voices[tv.next_voice].trigger(freq, velocity, sample_rate, gate_seconds, synth, None);
            tv.next_voice = (tv.next_voice + 1) % VOICE_COUNT;
        }
        SynthEngine::Trine => {
            let gate_seconds = trine.env3_attack_seconds + trine.env3_decay_seconds;
            tv.trine_voices[tv.next_trine_voice].trigger(freq, velocity, sample_rate, gate_seconds, trine);
            tv.next_trine_voice = (tv.next_trine_voice + 1) % VOICE_COUNT;
        }
        SynthEngine::Wave => {
            let gate_seconds = wave.amp_attack_seconds + wave.amp_decay_seconds;
            tv.wave_voices[tv.next_wave_voice].trigger(freq, velocity, sample_rate, gate_seconds, wave);
            tv.next_wave_voice = (tv.next_wave_voice + 1) % VOICE_COUNT;
        }
    }
}

/// Triggers `track`'s synth voice for `note` into `tv` — the piano-roll trigger-dispatch body
/// shared by `Sequencer::process`'s arrangement region loop and its Session View clip loop
/// (`trigger_session_clips`), same reasoning as `trigger_lane_step`.
fn trigger_piano_roll_note(
    tv: &mut TrackVoices,
    track: &Track,
    note: &Note,
    sample_rate: f32,
    samples_per_tick: f64,
) {
    let freq = pitch_to_freq(note.pitch);
    // The note's own length is its gate time: it holds through Attack/Decay/Sustain for exactly
    // this long before Release begins.
    let gate_seconds = ((note.length_ticks as f64 * samples_per_tick / sample_rate as f64) as f32)
        .max(MIN_NOTE_GATE_SECONDS);
    match track.synth_engine {
        SynthEngine::Simple => {
            let glide_from = if track.synth.glide_seconds > 0.0 { tv.last_freq } else { None };
            tv.voices[tv.next_voice].trigger(
                freq,
                note.velocity,
                sample_rate,
                gate_seconds,
                &track.synth,
                glide_from,
            );
            tv.next_voice = (tv.next_voice + 1) % VOICE_COUNT;
        }
        SynthEngine::Trine => {
            // Glide isn't part of the Trine engine in this pass.
            tv.trine_voices[tv.next_trine_voice].trigger(
                freq,
                note.velocity,
                sample_rate,
                gate_seconds,
                &track.trine,
            );
            tv.next_trine_voice = (tv.next_trine_voice + 1) % VOICE_COUNT;
        }
        SynthEngine::Wave => {
            // Glide isn't part of the Wave engine in this pass.
            tv.wave_voices[tv.next_wave_voice].trigger(
                freq,
                note.velocity,
                sample_rate,
                gate_seconds,
                &track.wave,
            );
            tv.next_wave_voice = (tv.next_wave_voice + 1) % VOICE_COUNT;
        }
    }
    tv.last_freq = Some(freq);
}

/// Equal-power left/right gains for a `Track::pan` value (-1.0 hard left, 0.0 center, 1.0 hard
/// right) — the standard constant-power law (`cos`/`sin` of a quarter-turn sweep) so a centered
/// track doesn't get louder or quieter than a hard-panned one when summed to mono.
pub(crate) fn equal_power_pan_gains(pan: f32) -> (f32, f32) {
    let theta = (pan.clamp(-1.0, 1.0) + 1.0) * std::f32::consts::FRAC_PI_4;
    (theta.cos(), theta.sin())
}
