use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{BufferSize, FromSample, SampleFormat, SizedSample, Stream, StreamConfig};

use crate::metering::{LoudnessMeter, MeterHandles};
use crate::model::{Song, TrackOutput};
use crate::plugin_host::{
    self, MasterEffectSlots, SendEffectSlots, SubmixEffectSlots, TrackEffectSlots,
};
use crate::wavetable;

mod automation;
mod offline_render;
mod sample_voice;
mod sequencer;
mod simple_voice;
mod trine_voice;
mod voice_dsp;
mod wave_voice;
pub use offline_render::{render_song_to_wav, render_track_to_buffer};
pub use sequencer::{CaptureLogHandle, SessionSlotHandles, new_capture_log_handle, new_session_slot_handles, ticks_per_second};
pub(crate) use sequencer::arrangement_length_ticks;

use sample_voice::TrackVoices;
use sequencer::{Sequencer, equal_power_pan_gains, samples_for_tick_span, samples_per_tick_at};

use automation::{
    DelayLine, MasterAutomationOverride, collect_automation, pdc_delay_samples_per_track,
    process_chain_with_automation,
};

/// 16th-note grid: 4 steps per beat.
const STEPS_PER_BEAT: f64 = 4.0;
pub(crate) const VOICE_COUNT: usize = 32;
pub(crate) const SAMPLE_VOICE_COUNT: usize = 32;
pub(crate) const MASTER_GAIN: f32 = 0.3;
/// Level (relative to a voice's starting amplitude) considered inaudible; below this a voice is freed.
pub(crate) const ENVELOPE_FLOOR: f32 = 0.0005;
/// Oscillator copies stacked per voice for `SynthParams::unison_voices` (capped at 3).
pub(crate) const MAX_UNISON_VOICES: usize = 3;

pub struct AudioStatus {
    pub device_name: String,
    pub sample_rate: u32,
    /// The range of frame counts a callback may be asked to render, needed to
    /// activate a CLAP plugin with a compatible `PluginAudioConfiguration`.
    pub min_frames: u32,
    pub max_frames: u32,
}

/// Playback control shared between the UI thread and the audio callback.
/// Cheap to clone: internally just a couple of `Arc`s.
#[derive(Clone)]
pub struct Transport {
    playing: Arc<AtomicBool>,
    current_tick: Arc<AtomicUsize>,
    metronome_enabled: Arc<AtomicBool>,
    /// Whether the transport is currently playing the Session View grid instead of the Playlist
    /// arrangement — a mode switch, never both at once (see `Sequencer::process`'s `session_mode`
    /// parameter). Transport state, not song data, same category as `playing`/`metronome_enabled`.
    session_mode: Arc<AtomicBool>,
    /// How many ticks a Session View slot launch/stop click snaps forward to (see
    /// `session::next_quantize_boundary`) — `0` means "launch immediately, no quantization." Set
    /// by the UI from the current song's own `Song::steps_per_bar` (main.rs's quantize picker), not
    /// computed here, since only the UI has a `Song` to read a time signature from at click time.
    session_quantize_ticks: Arc<AtomicUsize>,
    /// Whether the toolbar's "Capture" button is currently armed — while true (and only while
    /// `session_mode`/`playing` are also true), `Sequencer::trigger_session_clips` logs every slot
    /// launch/stop into its own `capture_log` for `main.rs`'s "Capture to Arrangement" workflow to
    /// materialize onto the Playlist once this is turned back off. Transport state, not song data,
    /// same category as `session_mode`.
    capturing: Arc<AtomicBool>,
}

impl Transport {
    /// A stopped, metronome-off, arrangement-mode transport at tick 0.
    pub fn new() -> Self {
        Self {
            playing: Arc::new(AtomicBool::new(false)),
            current_tick: Arc::new(AtomicUsize::new(0)),
            metronome_enabled: Arc::new(AtomicBool::new(false)),
            session_mode: Arc::new(AtomicBool::new(false)),
            session_quantize_ticks: Arc::new(AtomicUsize::new(0)),
            capturing: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Whether the transport is currently playing Session View clips instead of the Playlist
    /// arrangement.
    pub fn is_session_mode(&self) -> bool {
        self.session_mode.load(Ordering::Relaxed)
    }

    /// Switches between Session View and Playlist-arrangement playback.
    pub fn set_session_mode(&self, session_mode: bool) {
        self.session_mode.store(session_mode, Ordering::Relaxed);
    }

    /// See `Transport::session_quantize_ticks`.
    pub fn session_quantize_ticks(&self) -> usize {
        self.session_quantize_ticks.load(Ordering::Relaxed)
    }

    /// See `Transport::session_quantize_ticks`.
    pub fn set_session_quantize_ticks(&self, ticks: usize) {
        self.session_quantize_ticks.store(ticks, Ordering::Relaxed);
    }

    /// Whether the sequencer is currently running.
    pub fn is_playing(&self) -> bool {
        self.playing.load(Ordering::Relaxed)
    }

    /// Starts or stops the sequencer.
    pub fn set_playing(&self, playing: bool) {
        self.playing.store(playing, Ordering::Relaxed);
    }

    /// Whether the metronome click is currently enabled.
    pub fn is_metronome_enabled(&self) -> bool {
        self.metronome_enabled.load(Ordering::Relaxed)
    }

    /// Enables or disables the metronome click.
    pub fn set_metronome_enabled(&self, enabled: bool) {
        self.metronome_enabled.store(enabled, Ordering::Relaxed);
    }

    /// The most recently triggered tick (see `model::TICKS_PER_STEP`), for
    /// the UI's playhead. Divide by `TICKS_PER_STEP` for step-grid display.
    pub fn current_tick(&self) -> usize {
        self.current_tick.load(Ordering::Relaxed)
    }

    /// See `Transport::capturing`.
    pub fn is_capturing(&self) -> bool {
        self.capturing.load(Ordering::Relaxed)
    }

    /// See `Transport::capturing`.
    pub fn set_capturing(&self, capturing: bool) {
        self.capturing.store(capturing, Ordering::Relaxed);
    }
}

/// Lists audio output device names, in host-enumeration order, for a UI picker (see
/// `AudioEngine::start`'s `device_name`). Mirrors `audio_input::list_input_devices`. Best-effort:
/// a device whose name can't be queried is skipped, since a name-only listing has no other way to
/// represent it.
pub fn list_output_devices() -> Vec<String> {
    let Ok(devices) = cpal::default_host().output_devices() else {
        return Vec::new();
    };
    devices
        .filter_map(|d| Some(d.description().ok()?.name().to_string()))
        .collect()
}

/// Every sample rate `device_name` (or the host's default output device, if `None`) actually
/// supports, deduped and sorted — so a UI picker built from this can never offer a rate
/// `AudioEngine::start` would have to reject.
pub fn list_output_sample_rates(device_name: Option<&str>) -> Vec<u32> {
    let host = cpal::default_host();
    let named_device = device_name.and_then(|name| {
        host.output_devices().ok()?.find(|d| {
            d.description()
                .map(|desc| desc.name() == name)
                .unwrap_or(false)
        })
    });
    let Some(device) = named_device.or_else(|| host.default_output_device()) else {
        return Vec::new();
    };
    let Ok(configs) = device.supported_output_configs() else {
        return Vec::new();
    };
    let mut rates: Vec<u32> = configs
        .flat_map(|range| [range.min_sample_rate(), range.max_sample_rate()])
        .collect();
    rates.sort_unstable();
    rates.dedup();
    rates
}

pub struct AudioEngine {
    _stream: Stream,
    pub status: AudioStatus,
}

impl AudioEngine {
    /// Starts the playback engine on `device_name` (falling back to the host's default output
    /// device if `None` or if no device with that name is found), at `sample_rate` (falling back
    /// to the device's own default rate if `None`, or if the device doesn't actually support the
    /// requested rate — see `list_output_sample_rates` for a picker that only ever offers rates
    /// that don't need this fallback).
    #[allow(clippy::too_many_arguments)]
    pub fn start(
        song: Arc<Mutex<Song>>,
        transport: Transport,
        master_effects: MasterEffectSlots,
        track_effects: TrackEffectSlots,
        send_effects: SendEffectSlots,
        submix_effects: SubmixEffectSlots,
        track_meters: MeterHandles,
        master_meter: MeterHandles,
        submix_meters: MeterHandles,
        session_slots: SessionSlotHandles,
        capture_log: CaptureLogHandle,
        device_name: Option<&str>,
        sample_rate: Option<u32>,
    ) -> Result<Self> {
        // Forces the Wave engine's procedural wavetables to generate now, on this (non-real-time)
        // setup thread, instead of lazily the first time a `WaveVoice` triggers — which could
        // otherwise happen on the audio callback thread and cause an audible dropout.
        wavetable::warm_up();

        let host = cpal::default_host();
        let named_device = device_name.and_then(|name| {
            host.output_devices().ok()?.find(|d| {
                d.description()
                    .map(|desc| desc.name() == name)
                    .unwrap_or(false)
            })
        });
        let device = named_device
            .or_else(|| host.default_output_device())
            .context("no output audio device available")?;
        let device_name = device.to_string();

        let supported_config = match sample_rate {
            Some(rate) => device
                .supported_output_configs()
                .ok()
                .and_then(|mut configs| configs.find_map(|range| range.try_with_sample_rate(rate)))
                .or_else(|| device.default_output_config().ok())
                .context("no output config available at the requested sample rate")?,
            None => device
                .default_output_config()
                .context("no default output config available")?,
        };
        let sample_format = supported_config.sample_format();
        let config: StreamConfig = supported_config.into();

        let (min_frames, max_frames) = match config.buffer_size {
            BufferSize::Fixed(n) => (n, n),
            BufferSize::Default => (1, 8192),
        };

        let status = AudioStatus {
            device_name,
            sample_rate: config.sample_rate,
            min_frames,
            max_frames,
        };

        let stream = match sample_format {
            SampleFormat::F32 => build_playback_stream::<f32>(
                &device,
                &config,
                song,
                transport,
                master_effects,
                track_effects,
                send_effects,
                submix_effects,
                track_meters,
                master_meter,
                submix_meters,
                session_slots,
                capture_log,
                max_frames as usize,
            )?,
            SampleFormat::I16 => build_playback_stream::<i16>(
                &device,
                &config,
                song,
                transport,
                master_effects,
                track_effects,
                send_effects,
                submix_effects,
                track_meters,
                master_meter,
                submix_meters,
                session_slots,
                capture_log,
                max_frames as usize,
            )?,
            SampleFormat::U16 => build_playback_stream::<u16>(
                &device,
                &config,
                song,
                transport,
                master_effects,
                track_effects,
                send_effects,
                submix_effects,
                track_meters,
                master_meter,
                submix_meters,
                session_slots,
                capture_log,
                max_frames as usize,
            )?,
            other => bail!("unsupported output sample format: {other:?}"),
        };

        stream.play().context("failed to start audio stream")?;

        Ok(Self {
            _stream: stream,
            status,
        })
    }
}

fn build_playback_stream<T>(
    device: &cpal::Device,
    config: &StreamConfig,
    song: Arc<Mutex<Song>>,
    transport: Transport,
    master_effects: MasterEffectSlots,
    track_effects: TrackEffectSlots,
    send_effects: SendEffectSlots,
    submix_effects: SubmixEffectSlots,
    track_meters: MeterHandles,
    master_meter: MeterHandles,
    submix_meters: MeterHandles,
    session_slots: SessionSlotHandles,
    capture_log: CaptureLogHandle,
    max_frames: usize,
) -> Result<Stream>
where
    T: SizedSample + FromSample<f32>,
{
    let sample_rate = config.sample_rate as f32;
    let channels = (config.channels as usize).max(1);

    let mut sequencer = Sequencer::new(sample_rate);
    let mut scratch_l: Vec<f32> = Vec::with_capacity(max_frames);
    let mut scratch_r: Vec<f32> = Vec::with_capacity(max_frames);
    let mut track_dry_l: Vec<Vec<f32>> = Vec::new();
    let mut track_dry_r: Vec<Vec<f32>> = Vec::new();
    let mut metronome_dry: Vec<f32> = Vec::with_capacity(max_frames);

    // One `LoudnessMeter` per track (post-fader/pan, resized in lockstep with `track_dry_l`
    // below) plus one for the master bus (post-master-FX) — audio-thread-owned real-time state,
    // published each buffer into `track_meters`/`master_meter` for the UI thread to poll (see
    // `metering`'s module doc).
    let mut track_loudness: Vec<LoudnessMeter> = Vec::new();
    let mut master_loudness = LoudnessMeter::new(sample_rate);
    let mut was_playing = false;
    // One PDC compensation `DelayLine` per track (resized in lockstep with `track_dry_l`, same as
    // `track_loudness` above) — see `pdc_delay_samples_per_track`.
    let mut track_pdc_delay: Vec<DelayLine> = Vec::new();

    // Per-track CLAP insert-effect-chain scratch (one `Vec<EffectScratch>` per track index, grown
    // lazily to match that track's chain length) and a pair of reusable stereo buffers plus the
    // chain's own in-flight stereo scratch for whichever track is currently being processed.
    let mut track_scratch: Vec<Vec<plugin_host::EffectScratch>> = Vec::new();
    let mut track_effect_out_l: Vec<f32> = Vec::with_capacity(max_frames);
    let mut track_effect_out_r: Vec<f32> = Vec::with_capacity(max_frames);
    let mut track_chain_run_l: Vec<f32> = Vec::with_capacity(max_frames);
    let mut track_chain_run_r: Vec<f32> = Vec::with_capacity(max_frames);
    // Reused scratch for whichever track's post-fader/pan signal is currently being fed to its
    // `LoudnessMeter` (see below) — not summed anywhere itself, just a metering tap.
    let mut track_meter_l: Vec<f32> = Vec::with_capacity(max_frames);
    let mut track_meter_r: Vec<f32> = Vec::with_capacity(max_frames);

    // One accumulation buffer per send bus (resized in lockstep with `Song::sends`), fed by every
    // track's post-fader/pan signal scaled by that track's `Track::send_levels` entry for this
    // send — the same tap point `track_meter_l/r` reads, just scaled per-send instead of summed
    // straight into the master mix. Plus per-send CLAP/built-in effect-chain scratch, mirroring
    // `track_scratch`'s per-track shape, and a pair of reusable output/run buffers (sends are
    // processed one at a time, so one reusable pair covers all of them per callback).
    let mut send_mix_l: Vec<Vec<f32>> = Vec::new();
    let mut send_mix_r: Vec<Vec<f32>> = Vec::new();
    let mut send_scratch: Vec<Vec<plugin_host::EffectScratch>> = Vec::new();
    let mut send_chain_out_l: Vec<f32> = Vec::with_capacity(max_frames);
    let mut send_chain_out_r: Vec<f32> = Vec::with_capacity(max_frames);
    let mut send_chain_run_l: Vec<f32> = Vec::with_capacity(max_frames);
    let mut send_chain_run_r: Vec<f32> = Vec::with_capacity(max_frames);

    // One accumulation buffer per submix bus (resized in lockstep with `Song::submixes`), fed
    // *instead of* `scratch_l/r` by every track whose `Track::output` targets that submix — unlike
    // `send_mix_l/r` above, this replaces a track's direct contribution to the master mix rather
    // than tapping it in parallel. Same per-submix effect-chain scratch/output/run-buffer shape as
    // sends, plus one `LoudnessMeter` per submix (mirroring `track_loudness`) since a submix has
    // its own fader and deserves its own meter in the Mixer.
    let mut submix_mix_l: Vec<Vec<f32>> = Vec::new();
    let mut submix_mix_r: Vec<Vec<f32>> = Vec::new();
    let mut submix_scratch: Vec<Vec<plugin_host::EffectScratch>> = Vec::new();
    let mut submix_chain_out_l: Vec<f32> = Vec::with_capacity(max_frames);
    let mut submix_chain_out_r: Vec<f32> = Vec::with_capacity(max_frames);
    let mut submix_chain_run_l: Vec<f32> = Vec::with_capacity(max_frames);
    let mut submix_chain_run_r: Vec<f32> = Vec::with_capacity(max_frames);
    let mut submix_loudness: Vec<LoudnessMeter> = Vec::new();

    // Scratch for the master bus's own effect chain — same shape as `track_scratch`'s per-track
    // entries (one `EffectScratch` per chain slot), since the master chain runs through the exact
    // same `process_effect_chain` call a track's chain does.
    let mut master_scratch: Vec<plugin_host::EffectScratch> = Vec::new();
    let mut master_chain_run_l: Vec<f32> = Vec::with_capacity(max_frames);
    let mut master_chain_run_r: Vec<f32> = Vec::with_capacity(max_frames);
    let mut plugin_out_l: Vec<f32> = Vec::with_capacity(max_frames);
    let mut plugin_out_r: Vec<f32> = Vec::with_capacity(max_frames);

    // Pre-warm every per-track buffer to the song's shape *right now* (rather than letting the
    // first real-time callback discover it needs to grow `Vec`s and allocate 32+32 `Voice`/
    // `SampleVoice` structs per track) — a debug build doing that heap work inside the very first
    // callback was slow enough to blow the backend's deadline and log a startup underrun.
    //
    // Also captures the first snapshot for `last_snapshot` below — blocking here at stream setup
    // is fine since it's not on the real-time path yet.
    let mut last_snapshot: Option<Song> = None;
    if let Ok(snapshot) = song.lock() {
        for _ in 0..snapshot.tracks.len() {
            sequencer.track_voices.push(TrackVoices::new());
        }
        track_dry_l.resize_with(snapshot.tracks.len(), || Vec::with_capacity(max_frames));
        track_dry_r.resize_with(snapshot.tracks.len(), || Vec::with_capacity(max_frames));
        last_snapshot = Some(snapshot.clone());
    }
    if let Ok(chains) = track_effects.lock() {
        for chain in chains.iter() {
            let mut stage_scratch = Vec::with_capacity(chain.len());
            for _ in chain {
                let mut s = plugin_host::EffectScratch::new();
                s.reserve(max_frames);
                stage_scratch.push(s);
            }
            track_scratch.push(stage_scratch);
        }
    }
    if let Ok(chains) = master_effects.lock()
        && let Some(chain) = chains.first()
    {
        for _ in chain {
            let mut s = plugin_host::EffectScratch::new();
            s.reserve(max_frames);
            master_scratch.push(s);
        }
    }
    if let Ok(chains) = send_effects.lock() {
        for chain in chains.iter() {
            let mut stage_scratch = Vec::with_capacity(chain.len());
            for _ in chain {
                let mut s = plugin_host::EffectScratch::new();
                s.reserve(max_frames);
                stage_scratch.push(s);
            }
            send_scratch.push(stage_scratch);
        }
    }
    if let Ok(chains) = submix_effects.lock() {
        for chain in chains.iter() {
            let mut stage_scratch = Vec::with_capacity(chain.len());
            for _ in chain {
                let mut s = plugin_host::EffectScratch::new();
                s.reserve(max_frames);
                stage_scratch.push(s);
            }
            submix_scratch.push(stage_scratch);
        }
    }

    let err_fn = |err| eprintln!("audio stream error: {err}");

    let stream = device.build_output_stream(
        *config,
        move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
            let frames = data.len() / channels;
            scratch_l.resize(frames, 0.0);
            scratch_r.resize(frames, 0.0);
            scratch_l.iter_mut().for_each(|s| *s = 0.0);
            scratch_r.iter_mut().for_each(|s| *s = 0.0);

            // Integrated LUFS measures "since the last time playback started" (see
            // `LoudnessMeter::reset`'s doc comment) — reset every meter exactly on the
            // playing-to-stopped edge, the same transition that resets the sequencer's position
            // below, rather than continuing to accumulate through a stop.
            let is_playing = transport.is_playing();
            if was_playing && !is_playing {
                master_loudness.reset();
                for meter in track_loudness.iter_mut() {
                    meter.reset();
                }
            }
            was_playing = is_playing;

            // Note: even when stopped, silence still runs through the master
            // effect below rather than short-circuiting straight to the
            // device — otherwise a delay/reverb tail would cut off instantly
            // on Stop instead of ringing out naturally, like in a real DAW.
            // (Per-track effects don't get this treatment: while stopped, no
            // track has anything playing through them, so there's no tail to
            // preserve there — only the master bus stays fed with silence.)
            //
            // Declared outside the `is_playing` branch (default/empty when not playing) since the
            // master chain runs unconditionally, below, after this `if`/`else` — see there.
            let mut master_automation = MasterAutomationOverride::default();
            let mut master_samples_per_tick = 1.0f64;
            // Same "declared outside, default/empty when not playing" reasoning as
            // `master_automation` above — the master chain (which can itself have a sidechain
            // source) runs unconditionally below, after this `if`/`else`.
            let mut all_track_dry: Vec<(&[f32], &[f32])> = Vec::new();
            if is_playing {
                // Snapshot the song once per callback (not per sample) so the real-time thread
                // only briefly touches the shared lock. Uses `try_lock`, not `lock`, and falls
                // back to the previous snapshot on contention: the UI thread holds this same
                // mutex for its whole paint pass (`SimpleDawApp::ui`), and painting a large song
                // (many tracks/notes, e.g. after a MIDI import) can take long enough that blocking
                // here would miss this callback's deadline and log a real buffer underrun — reusing
                // a slightly stale snapshot for one buffer is inaudible, a dropout isn't.
                if let Ok(guard) = song.try_lock() {
                    last_snapshot = Some(guard.clone());
                }
                let Some(snapshot) = last_snapshot.as_ref() else {
                    return;
                };
                // The tick in effect as of this buffer's very first sample — whatever tick most
                // recently triggered as of the end of the *previous* callback, since a tick's
                // state holds until the next one fires (same reasoning as `track_fade_gain`).
                // Captured before `process()` advances it, so per-sample automation lookups below
                // can walk forward from a correct starting point instead of the buffer's *last*
                // triggered tick (which `sequencer.current_tick()` would give after `process()`).
                let buffer_start_tick = sequencer.current_tick();
                sequencer.process(
                    snapshot,
                    frames,
                    &mut track_dry_l,
                    &mut track_dry_r,
                    transport.is_metronome_enabled(),
                    &mut metronome_dry,
                    transport.is_session_mode(),
                    transport.session_quantize_ticks(),
                    transport.is_capturing(),
                );
                transport
                    .current_tick
                    .store(sequencer.current_tick(), Ordering::Relaxed);
                // See `SessionSlotHandles`'s doc comment — published every buffer so the UI thread
                // can show live queued/playing/stopped state, the same "audio thread publishes, UI
                // thread reads a cheap clone" split `track_meters` already uses just below.
                if let Ok(mut published_session_slots) = session_slots.lock() {
                    published_session_slots.clone_from(&sequencer.session_slots);
                }
                // See `CaptureLogHandle`'s doc comment — published every buffer so `main.rs` can
                // read the latest log the moment it turns `Transport::capturing` back off.
                if let Ok(mut published_capture_log) = capture_log.lock() {
                    published_capture_log.0.clone_from(&sequencer.capture_log);
                    published_capture_log.1 = sequencer.capture_tick;
                }

                // Resolved once per buffer from the tempo at its first tick — unlike
                // `Sequencer::process`'s own per-tick-boundary resolution above, a `tempo_map`
                // change landing mid-buffer only takes effect for automation/fades starting next
                // buffer (at most a few milliseconds' latency), so `LaneRef::value_at`'s per-sample
                // tick conversion below can keep assuming one constant rate for the whole buffer.
                let samples_per_tick =
                    samples_per_tick_at(sample_rate as f64, snapshot.bpm_at(buffer_start_tick));
                master_samples_per_tick = samples_per_tick;
                // One automated-lane snapshot per track/send/master for this whole buffer,
                // evaluated per output sample below via `LaneRef::value_at` — see
                // `TrackAutomationOverride`'s doc comment.
                let (track_automation, master_override, send_automation) = collect_automation(
                    snapshot,
                    buffer_start_tick,
                    transport.is_session_mode(),
                    &sequencer.session_slots,
                );
                master_automation = master_override;

                // Track count can change between callbacks (tracks added/removed) — resize in
                // lockstep with `track_dry_l`, same as `sequencer.track_voices` above.
                while track_loudness.len() < track_dry_l.len() {
                    track_loudness.push(LoudnessMeter::new(sample_rate));
                }
                track_loudness.truncate(track_dry_l.len());
                let published_track_meters = track_meters.lock().ok();

                // Run each track's dry mix through its own CLAP/built-in insert effect chain (if
                // any are loaded there — the chain now carries real stereo between stages, see
                // `plugin_host::process_effect_chain`), apply that track's volume and pan (as an
                // equal-power gain split, the same point a channel strip's pan pot sits after its
                // inserts), then sum every track into the master bus. The same post-fader/pan
                // samples feed that track's `LoudnessMeter` — the natural tap point for a channel
                // strip's meter, distinct from both the raw synthesis (`track_dry_l/r`) and the
                // final master mix.
                track_effect_out_l.resize(frames, 0.0);
                track_effect_out_r.resize(frames, 0.0);
                track_meter_l.resize(frames, 0.0);
                track_meter_r.resize(frames, 0.0);

                // Send bus count can change between callbacks (buses added/removed) — resize in
                // lockstep with `Song::sends`, same as `track_loudness` above.
                send_mix_l.resize_with(snapshot.sends.len(), Vec::new);
                send_mix_r.resize_with(snapshot.sends.len(), Vec::new);
                for buf in send_mix_l.iter_mut().chain(send_mix_r.iter_mut()) {
                    buf.clear();
                    buf.resize(frames, 0.0);
                }
                send_scratch.resize_with(snapshot.sends.len(), Vec::new);

                // Submix bus count can change between callbacks (buses added/removed) — resize in
                // lockstep with `Song::submixes`, same as the send buffers above.
                submix_mix_l.resize_with(snapshot.submixes.len(), Vec::new);
                submix_mix_r.resize_with(snapshot.submixes.len(), Vec::new);
                for buf in submix_mix_l.iter_mut().chain(submix_mix_r.iter_mut()) {
                    buf.clear();
                    buf.resize(frames, 0.0);
                }
                submix_scratch.resize_with(snapshot.submixes.len(), Vec::new);
                while submix_loudness.len() < snapshot.submixes.len() {
                    submix_loudness.push(LoudnessMeter::new(sample_rate));
                }
                submix_loudness.truncate(snapshot.submixes.len());
                let published_submix_meters = submix_meters.lock().ok();

                // Every track's own pre-effects, pre-volume/pan signal for this buffer, indexed by
                // `Song::tracks` index — the sidechain key source pool any chain (a track's, a
                // send's, a submix's, or the master's) in this callback can route from. Built once
                // here since it doesn't depend on which chain is being processed; see
                // `plugin_host::process_effect_chain`'s doc.
                all_track_dry = track_dry_l
                    .iter()
                    .zip(track_dry_r.iter())
                    .map(|(l, r)| (l.as_slice(), r.as_slice()))
                    .collect();

                // Automatic plugin delay compensation: computed fresh each buffer from whichever
                // chains are currently loaded (see `pdc_delay_samples_per_track`), applied to each
                // track's own post-chain signal below via `track_pdc_delay`'s persistent state.
                let track_pdc_delay_samples: Vec<u32> =
                    match (track_effects.lock(), submix_effects.lock()) {
                        (Ok(track_chains), Ok(submix_chains)) => {
                            let track_latency: Vec<u32> = track_chains
                                .iter()
                                .map(|chain| plugin_host::chain_latency_samples(chain))
                                .collect();
                            let submix_latency: Vec<u32> = submix_chains
                                .iter()
                                .map(|chain| plugin_host::chain_latency_samples(chain))
                                .collect();
                            pdc_delay_samples_per_track(&snapshot.tracks, &track_latency, &submix_latency)
                        }
                        _ => vec![0; snapshot.tracks.len()],
                    };
                while track_pdc_delay.len() < snapshot.tracks.len() {
                    track_pdc_delay.push(DelayLine::new());
                }
                track_pdc_delay.truncate(snapshot.tracks.len());

                if let Ok(mut chains) = track_effects.lock() {
                    while track_scratch.len() < track_dry_l.len() {
                        track_scratch.push(Vec::new());
                    }
                    for (track_index, (dry_l, dry_r)) in
                        track_dry_l.iter().zip(track_dry_r.iter()).enumerate()
                    {
                        let track = snapshot.tracks.get(track_index);
                        let automation = track_automation.get(track_index);
                        let static_volume = track.map_or(1.0, |t| t.volume);
                        let static_pan = track.map_or(0.0, |t| t.pan);
                        // Per-output-sample volume/pan, sample-accurate when a lane is present
                        // (`LaneRef::value_at` at this sample's exact tick position) rather than
                        // one value held for the whole buffer.
                        let volume_at = |i: usize| {
                            automation
                                .and_then(|a| a.volume.map(|lane| lane.value_at(i, samples_per_tick)))
                                .unwrap_or(static_volume)
                        };
                        let pan_at = |i: usize| {
                            automation
                                .and_then(|a| a.pan.map(|lane| lane.value_at(i, samples_per_tick)))
                                .unwrap_or(static_pan)
                        };
                        let chain = chains
                            .get_mut(track_index)
                            .map_or(&mut [][..], Vec::as_mut_slice);
                        let stage_scratch = &mut track_scratch[track_index];
                        while stage_scratch.len() < chain.len() {
                            stage_scratch.push(plugin_host::EffectScratch::new());
                        }
                        track_effect_out_l.resize(frames, 0.0);
                        track_effect_out_r.resize(frames, 0.0);

                        let empty_effect_params = Vec::new();
                        let effect_params =
                            automation.map_or(&empty_effect_params, |a| &a.effect_params);
                        // A frozen track's own chain is skipped: `dry_l/r` already came from
                        // `frozen_clip`, which was baked *through* this same chain (see
                        // `render_track_to_buffer`) — running it again here would double-process.
                        let used = !track.is_some_and(|t| t.frozen)
                            && process_chain_with_automation(
                                chain,
                                effect_params,
                                samples_per_tick,
                                dry_l,
                                dry_r,
                                &mut track_effect_out_l,
                                &mut track_effect_out_r,
                                stage_scratch,
                                &mut track_chain_run_l,
                                &mut track_chain_run_r,
                                &all_track_dry,
                            );
                        let source_l = if used { &track_effect_out_l } else { dry_l };
                        let source_r = if used { &track_effect_out_r } else { dry_r };
                        for i in 0..frames {
                            let (pan_l, pan_r) = equal_power_pan_gains(pan_at(i));
                            track_meter_l[i] = volume_at(i) * pan_l * source_l[i];
                            track_meter_r[i] = volume_at(i) * pan_r * source_r[i];
                        }
                        track_pdc_delay[track_index].process(
                            &mut track_meter_l[..frames],
                            &mut track_meter_r[..frames],
                            track_pdc_delay_samples[track_index] as usize,
                        );
                        // This track's post-fader/pan signal sums into its `TrackOutput`
                        // target — straight to the master accumulator, or exclusively into its
                        // submix bus's own accumulator instead (see `SubmixBus`'s doc comment).
                        match track.map_or(TrackOutput::Master, |t| t.output) {
                            TrackOutput::Master => {
                                for i in 0..frames {
                                    scratch_l[i] += track_meter_l[i];
                                    scratch_r[i] += track_meter_r[i];
                                }
                            }
                            TrackOutput::Submix(index) => {
                                if let (Some(mix_l), Some(mix_r)) =
                                    (submix_mix_l.get_mut(index), submix_mix_r.get_mut(index))
                                {
                                    for i in 0..frames {
                                        mix_l[i] += track_meter_l[i];
                                        mix_r[i] += track_meter_r[i];
                                    }
                                }
                            }
                        }
                        let readings = track_loudness[track_index].process(&track_meter_l, &track_meter_r);
                        if let Some(handle) =
                            published_track_meters.as_ref().and_then(|h| h.get(track_index))
                        {
                            handle.publish(&readings);
                        }

                        // Feed this track's post-fader/pan signal (`track_meter_l/r`, just
                        // published to the meter above) into every send bus it has a nonzero
                        // level for — the same tap point a channel strip's send knob reads from.
                        // A `SendLevel` automation lane on that send overrides the static level,
                        // sample-accurately when present, same as volume/pan above.
                        if let Some(send_levels) = track.map(|t| t.send_levels.as_slice()) {
                            for (send_index, &static_level) in send_levels.iter().enumerate() {
                                let lane = automation.and_then(|a| {
                                    a.send_levels
                                        .iter()
                                        .find(|(i, _)| *i == send_index)
                                        .map(|(_, lane)| *lane)
                                });
                                let Some((mix_l, mix_r)) = send_mix_l
                                    .get_mut(send_index)
                                    .zip(send_mix_r.get_mut(send_index))
                                else {
                                    continue;
                                };
                                match lane {
                                    Some(lane) => {
                                        for i in 0..frames {
                                            let level = lane.value_at(i, samples_per_tick);
                                            mix_l[i] += track_meter_l[i] * level;
                                            mix_r[i] += track_meter_r[i] * level;
                                        }
                                    }
                                    None => {
                                        if static_level != 0.0 {
                                            for i in 0..frames {
                                                mix_l[i] += track_meter_l[i] * static_level;
                                                mix_r[i] += track_meter_r[i] * static_level;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    for (track_index, (dry_l, dry_r)) in
                        track_dry_l.iter().zip(track_dry_r.iter()).enumerate()
                    {
                        let track = snapshot.tracks.get(track_index);
                        let automation = track_automation.get(track_index);
                        let static_volume = track.map_or(1.0, |t| t.volume);
                        let static_pan = track.map_or(0.0, |t| t.pan);
                        for i in 0..frames {
                            let volume = automation
                                .and_then(|a| a.volume.map(|lane| lane.value_at(i, samples_per_tick)))
                                .unwrap_or(static_volume);
                            let pan = automation
                                .and_then(|a| a.pan.map(|lane| lane.value_at(i, samples_per_tick)))
                                .unwrap_or(static_pan);
                            let (pan_l, pan_r) = equal_power_pan_gains(pan);
                            track_meter_l[i] = volume * pan_l * dry_l[i];
                            track_meter_r[i] = volume * pan_r * dry_r[i];
                        }
                        track_pdc_delay[track_index].process(
                            &mut track_meter_l[..frames],
                            &mut track_meter_r[..frames],
                            track_pdc_delay_samples[track_index] as usize,
                        );
                        // Output-routing: see the matching `match` in the `Ok(mut chains)` branch above.
                        match track.map_or(TrackOutput::Master, |t| t.output) {
                            TrackOutput::Master => {
                                for i in 0..frames {
                                    scratch_l[i] += track_meter_l[i];
                                    scratch_r[i] += track_meter_r[i];
                                }
                            }
                            TrackOutput::Submix(index) => {
                                if let (Some(mix_l), Some(mix_r)) =
                                    (submix_mix_l.get_mut(index), submix_mix_r.get_mut(index))
                                {
                                    for i in 0..frames {
                                        mix_l[i] += track_meter_l[i];
                                        mix_r[i] += track_meter_r[i];
                                    }
                                }
                            }
                        }
                        let readings = track_loudness[track_index].process(&track_meter_l, &track_meter_r);
                        if let Some(handle) =
                            published_track_meters.as_ref().and_then(|h| h.get(track_index))
                        {
                            handle.publish(&readings);
                        }

                        // See the matching comment in the `Ok(mut chains)` branch above.
                        if let Some(send_levels) = track.map(|t| t.send_levels.as_slice()) {
                            for (send_index, &static_level) in send_levels.iter().enumerate() {
                                let lane = automation.and_then(|a| {
                                    a.send_levels
                                        .iter()
                                        .find(|(i, _)| *i == send_index)
                                        .map(|(_, lane)| *lane)
                                });
                                let Some((mix_l, mix_r)) = send_mix_l
                                    .get_mut(send_index)
                                    .zip(send_mix_r.get_mut(send_index))
                                else {
                                    continue;
                                };
                                match lane {
                                    Some(lane) => {
                                        for i in 0..frames {
                                            let level = lane.value_at(i, samples_per_tick);
                                            mix_l[i] += track_meter_l[i] * level;
                                            mix_r[i] += track_meter_r[i] * level;
                                        }
                                    }
                                    None => {
                                        if static_level != 0.0 {
                                            for i in 0..frames {
                                                mix_l[i] += track_meter_l[i] * static_level;
                                                mix_r[i] += track_meter_r[i] * static_level;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Run each send bus's own effect chain (same `process_effect_chain` machinery a
                // track's or the master's chain uses) over its accumulated `send_mix_l/r`, then
                // sum the result straight into the master mix — a send bus has no fader of its
                // own in this minimal model, just its chain and whatever level each track sent it.
                // A `SendEffectParam` automation lane (from any track's region) overrides that
                // send's own chain params, same breakpoint-chunked approximation a track's chain
                // gets — see `process_chain_with_automation`.
                send_chain_out_l.resize(frames, 0.0);
                send_chain_out_r.resize(frames, 0.0);
                let empty_send_effect_params = Vec::new();
                for (send_index, (mix_l, mix_r)) in
                    send_mix_l.iter().zip(send_mix_r.iter()).enumerate()
                {
                    if let Ok(mut chains) = send_effects.lock() {
                        let chain = chains
                            .get_mut(send_index)
                            .map_or(&mut [][..], Vec::as_mut_slice);
                        let stage_scratch = &mut send_scratch[send_index];
                        while stage_scratch.len() < chain.len() {
                            stage_scratch.push(plugin_host::EffectScratch::new());
                        }
                        let effect_params = send_automation
                            .get(send_index)
                            .map_or(&empty_send_effect_params, |s| &s.effect_params);
                        let used = process_chain_with_automation(
                            chain,
                            effect_params,
                            samples_per_tick,
                            mix_l,
                            mix_r,
                            &mut send_chain_out_l,
                            &mut send_chain_out_r,
                            stage_scratch,
                            &mut send_chain_run_l,
                            &mut send_chain_run_r,
                            &all_track_dry,
                        );
                        if used {
                            for i in 0..frames {
                                scratch_l[i] += send_chain_out_l[i];
                                scratch_r[i] += send_chain_out_r[i];
                            }
                            continue;
                        }
                    }
                    for i in 0..frames {
                        scratch_l[i] += mix_l[i];
                        scratch_r[i] += mix_r[i];
                    }
                }

                // Run each submix bus's own effect chain (same `process_effect_chain` machinery a
                // track's/send's chain uses) over its accumulated `submix_mix_l/r`, apply that
                // submix's `volume` fader (unlike a send bus, which has none — a submix stands in
                // for its member tracks' direct contribution to the mix), publish its own
                // `LoudnessMeter` reading (the post-chain, post-fader signal — the same tap point
                // a track's own meter reads), then sum into the master mix.
                submix_chain_out_l.resize(frames, 0.0);
                submix_chain_out_r.resize(frames, 0.0);
                for (submix_index, (mix_l, mix_r)) in
                    submix_mix_l.iter().zip(submix_mix_r.iter()).enumerate()
                {
                    let volume = snapshot.submixes.get(submix_index).map_or(1.0, |s| s.volume);
                    let (pan_l, pan_r) = equal_power_pan_gains(
                        snapshot.submixes.get(submix_index).map_or(0.0, |s| s.pan),
                    );
                    if let Ok(mut chains) = submix_effects.lock() {
                        let chain = chains
                            .get_mut(submix_index)
                            .map_or(&mut [][..], Vec::as_mut_slice);
                        let stage_scratch = &mut submix_scratch[submix_index];
                        while stage_scratch.len() < chain.len() {
                            stage_scratch.push(plugin_host::EffectScratch::new());
                        }
                        let used = plugin_host::process_effect_chain(
                            chain,
                            mix_l,
                            mix_r,
                            &mut submix_chain_out_l,
                            &mut submix_chain_out_r,
                            stage_scratch,
                            &mut submix_chain_run_l,
                            &mut submix_chain_run_r,
                            &all_track_dry,
                        );
                        if used {
                            for i in 0..frames {
                                submix_chain_out_l[i] *= volume * pan_l;
                                submix_chain_out_r[i] *= volume * pan_r;
                            }
                            let readings = submix_loudness[submix_index]
                                .process(&submix_chain_out_l, &submix_chain_out_r);
                            if let Some(handle) = published_submix_meters
                                .as_ref()
                                .and_then(|h| h.get(submix_index))
                            {
                                handle.publish(&readings);
                            }
                            for i in 0..frames {
                                scratch_l[i] += submix_chain_out_l[i];
                                scratch_r[i] += submix_chain_out_r[i];
                            }
                            continue;
                        }
                    }
                    for i in 0..frames {
                        submix_chain_out_l[i] = mix_l[i] * volume * pan_l;
                        submix_chain_out_r[i] = mix_r[i] * volume * pan_r;
                    }
                    let readings = submix_loudness[submix_index]
                        .process(&submix_chain_out_l, &submix_chain_out_r);
                    if let Some(handle) =
                        published_submix_meters.as_ref().and_then(|h| h.get(submix_index))
                    {
                        handle.publish(&readings);
                    }
                    for i in 0..frames {
                        scratch_l[i] += submix_chain_out_l[i];
                        scratch_r[i] += submix_chain_out_r[i];
                    }
                }

                for i in 0..frames {
                    scratch_l[i] += metronome_dry[i];
                    scratch_r[i] += metronome_dry[i];
                }

                for s in scratch_l.iter_mut() {
                    *s = (*s * MASTER_GAIN).tanh();
                }
                for s in scratch_r.iter_mut() {
                    *s = (*s * MASTER_GAIN).tanh();
                }
            } else {
                sequencer.reset_position();
                transport.current_tick.store(0, Ordering::Relaxed);
            }

            // Run the mix through the master bus's effect chain (CLAP and/or built-in stages, same
            // machinery a track's own chain uses — see `plugin_host::process_effect_chain`), if
            // any effects are loaded there. Falls back to the dry stereo mix if the chain is empty
            // or nothing in it actually processed. Channel counts for a CLAP stage come from what
            // the plugin actually declared via the `audio-ports` extension (see
            // `plugin_host::load_and_activate`) — assuming every effect is 2-in/2-out caused real
            // plugins (e.g. ZamDelay, which is mono-in) to read past their declared buffers.
            //
            // A `MasterEffectParam` automation lane (from any track's region, see
            // `AutomationTarget`) overrides the master chain's own params here, same
            // breakpoint-chunked approximation a track's/send's chain gets.
            let mut used_master_chain = false;
            if let Ok(mut chains) = master_effects.lock() {
                let chain = chains.get_mut(0).map_or(&mut [][..], Vec::as_mut_slice);
                while master_scratch.len() < chain.len() {
                    master_scratch.push(plugin_host::EffectScratch::new());
                }
                plugin_out_l.resize(frames, 0.0);
                plugin_out_r.resize(frames, 0.0);
                used_master_chain = process_chain_with_automation(
                    chain,
                    &master_automation.effect_params,
                    master_samples_per_tick,
                    &scratch_l,
                    &scratch_r,
                    &mut plugin_out_l,
                    &mut plugin_out_r,
                    &mut master_scratch,
                    &mut master_chain_run_l,
                    &mut master_chain_run_r,
                    &all_track_dry,
                );
            }

            let (left, right): (&[f32], &[f32]) = if used_master_chain {
                (&plugin_out_l, &plugin_out_r)
            } else {
                (&scratch_l, &scratch_r)
            };

            let master_readings = master_loudness.process(left, right);
            if let Ok(handles) = master_meter.lock()
                && let Some(handle) = handles.first()
            {
                handle.publish(&master_readings);
            }

            for (i, frame) in data.chunks_mut(channels).enumerate() {
                frame[0] = T::from_sample(left[i]);
                if channels > 1 {
                    let r = T::from_sample(right[i]);
                    for sample in &mut frame[1..] {
                        *sample = r;
                    }
                }
            }
        },
        err_fn,
        None,
    )?;
    Ok(stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{
        CurveShape, FilterRouting, FilterSlope, FilterType, FollowAction, Lane, LaunchMode, LfoTarget, ModSlot,
        ModSource, ModTarget, RegionContent, SynthParams, SynthWaveform, TICKS_PER_STEP, TrineParams,
        WaveModSlot, WaveModSource, WaveModTarget, WaveParams,
    };
    use crate::sample::SampleBuffer;
    use crate::session::SlotState;
    use crate::wavetable::{WaveWarpMode, WavetableId};
    use super::sample_voice::SampleVoice;
    use super::sequencer::step_triggering_at;
    use super::simple_voice::Voice;
    use super::trine_voice::TrineVoice;
    use super::voice_dsp::{pitch_to_freq, waveform_sample};
    use super::wave_voice::WaveVoice;

    #[test]
    fn pitch_to_freq_matches_concert_a() {
        assert!((pitch_to_freq(69) - 440.0).abs() < 0.001);
    }

    #[test]
    fn delay_line_shifts_an_impulse_by_exactly_the_requested_sample_count() {
        let mut delay = DelayLine::new();
        let mut l = vec![0.0f32; 10];
        let mut r = vec![0.0f32; 10];
        l[0] = 1.0;
        r[0] = 1.0;
        delay.process(&mut l, &mut r, 4);
        let expected_l: Vec<f32> = (0..10).map(|i| if i == 4 { 1.0 } else { 0.0 }).collect();
        assert_eq!(l, expected_l);
        assert_eq!(r, expected_l);
    }

    #[test]
    fn delay_line_of_zero_samples_is_the_identity() {
        let mut delay = DelayLine::new();
        let mut l = vec![0.1, 0.2, -0.3, 0.4];
        let mut r = l.clone();
        let original = l.clone();
        delay.process(&mut l, &mut r, 0);
        assert_eq!(l, original);
        assert_eq!(r, original);
    }

    #[test]
    fn delay_line_keeps_delaying_correctly_across_a_change_in_requested_delay() {
        // A plugin loading/unloading mid-session changes the requested delay live — the buffer
        // must grow/shrink to the new amount without corrupting in-flight samples' relative timing.
        let mut delay = DelayLine::new();
        let mut l = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut r = l.clone();
        delay.process(&mut l[..2], &mut r[..2], 2);
        delay.process(&mut l[2..], &mut r[2..], 5);
        let expected: Vec<f32> = (0..6).map(|i| if i == 2 { 1.0 } else { 0.0 }).collect();
        assert_eq!(l, expected, "the impulse should still land 2 samples after where it started");
    }

    #[test]
    fn pdc_delay_samples_per_track_aligns_every_track_to_the_slowest_one() {
        let tracks = vec![
            crate::model::Track::new_piano_roll("A", 1),
            crate::model::Track::new_piano_roll("B", 2),
            crate::model::Track::new_piano_roll("C", 3),
        ];
        let track_latency = vec![0, 64, 200];
        let delays = pdc_delay_samples_per_track(&tracks, &track_latency, &[]);
        assert_eq!(delays, vec![200, 136, 0], "every track should end up compensated to the max (200)");
    }

    #[test]
    fn pdc_delay_samples_per_track_adds_the_submixs_own_latency_for_a_routed_track() {
        let mut submix_track = crate::model::Track::new_piano_roll("Routed", 1);
        submix_track.output = TrackOutput::Submix(0);
        let direct_track = crate::model::Track::new_piano_roll("Direct", 2);
        let tracks = vec![submix_track, direct_track];
        // The routed track's own chain has 0 latency, but its submix's chain has 50 — its
        // effective latency (50) should still beat the direct track's higher own-chain latency
        // (30) once combined, so the direct track ends up with the larger compensation delay.
        let track_latency = vec![0, 30];
        let submix_latency = vec![50];
        let delays = pdc_delay_samples_per_track(&tracks, &track_latency, &submix_latency);
        assert_eq!(delays, vec![0, 20]);
    }

    #[test]
    fn step_triggering_at_fires_on_grid_when_offset_is_zero() {
        let mut lane = Lane::new("Kick", 36, 4);
        lane.set_step(1, 100);
        assert!(step_triggering_at(&lane, 0).is_none());
        let step = step_triggering_at(&lane, TICKS_PER_STEP)
            .expect("step 1 should trigger on its own boundary");
        assert_eq!(step.velocity, 100);
    }

    #[test]
    fn step_triggering_at_honors_positive_timing_offset() {
        let mut lane = Lane::new("Kick", 36, 4);
        lane.set_step(1, 100);
        lane.steps[1].as_mut().unwrap().timing_offset_ticks = 6;
        assert!(
            step_triggering_at(&lane, TICKS_PER_STEP).is_none(),
            "nudged step shouldn't fire on its unnudged boundary"
        );
        let step = step_triggering_at(&lane, TICKS_PER_STEP + 6)
            .expect("step should fire at its nudged tick");
        assert_eq!(step.velocity, 100);
    }

    #[test]
    fn step_triggering_at_honors_negative_timing_offset() {
        let mut lane = Lane::new("Kick", 36, 4);
        lane.set_step(2, 90);
        lane.steps[2].as_mut().unwrap().timing_offset_ticks = -5;
        let step = step_triggering_at(&lane, 2 * TICKS_PER_STEP - 5)
            .expect("step should fire early at its nudged tick");
        assert_eq!(step.velocity, 90);
    }

    #[test]
    fn step_triggering_at_returns_none_for_an_empty_step() {
        let lane = Lane::new("Kick", 36, 4);
        assert!(step_triggering_at(&lane, 0).is_none());
    }

    fn synth(overrides: impl FnOnce(&mut SynthParams)) -> SynthParams {
        let mut synth = SynthParams::default();
        overrides(&mut synth);
        synth
    }

    #[test]
    fn voice_is_audible_then_decays_to_silence() {
        let sample_rate = 48_000.0;
        let mut voice = Voice::default();
        let synth = synth(|s| s.decay_seconds = 0.25);
        voice.trigger(
            pitch_to_freq(60),
            100,
            sample_rate,
            synth.decay_seconds,
            &synth,
            None,
        );

        let mut early_peak = 0.0f32;
        for _ in 0..200 {
            early_peak = early_peak.max(voice.next_sample().0.abs());
        }
        assert!(
            early_peak > 0.05,
            "voice should be clearly audible right after trigger"
        );

        for _ in 0..(sample_rate as usize) {
            voice.next_sample().0;
        }
        assert!(
            !voice.active,
            "voice should have decayed to silence within 1 second"
        );
        assert_eq!(
            voice.next_sample().0,
            0.0,
            "an inactive voice must output silence"
        );
    }

    #[test]
    fn longer_decay_keeps_voice_active_longer() {
        let sample_rate = 48_000.0;
        let mut short = Voice::default();
        let short_synth = synth(|s| s.decay_seconds = 0.1);
        short.trigger(
            pitch_to_freq(60),
            100,
            sample_rate,
            short_synth.decay_seconds,
            &short_synth,
            None,
        );
        let mut long = Voice::default();
        let long_synth = synth(|s| s.decay_seconds = 1.0);
        long.trigger(
            pitch_to_freq(60),
            100,
            sample_rate,
            long_synth.decay_seconds,
            &long_synth,
            None,
        );

        for _ in 0..(sample_rate * 0.3) as usize {
            short.next_sample().0;
            long.next_sample().0;
        }
        assert!(!short.active, "short-decay voice should be done after 0.3s");
        assert!(
            long.active,
            "long-decay voice should still be sounding after 0.3s"
        );
    }

    #[test]
    fn attack_ramps_amplitude_up_instead_of_jumping_to_peak() {
        let sample_rate = 48_000.0;
        let mut voice = Voice::default();
        let synth = synth(|s| {
            s.waveform = SynthWaveform::Square;
            s.attack_seconds = 0.05;
            s.decay_seconds = 0.25;
        });
        voice.trigger(
            pitch_to_freq(60),
            127,
            sample_rate,
            synth.attack_seconds + synth.decay_seconds,
            &synth,
            None,
        );

        // Right after trigger, a Square wave at full amplitude would already be
        // near +/-1.0; with a 50ms attack ramping from 0, it should start much quieter.
        let first = voice.next_sample().0.abs();
        assert!(
            first < 0.1,
            "voice should start near silent during its attack ramp, got {first}"
        );

        // After the attack window elapses the voice should be in full swing.
        for _ in 0..(sample_rate * 0.05) as usize {
            voice.next_sample().0;
        }
        let mut peak = 0.0f32;
        for _ in 0..50 {
            peak = peak.max(voice.next_sample().0.abs());
        }
        assert!(
            peak > 0.9,
            "voice should reach near-full amplitude once attack completes, got {peak}"
        );
    }

    #[test]
    fn zero_attack_reaches_full_amplitude_almost_immediately() {
        let sample_rate = 48_000.0;
        let mut voice = Voice::default();
        let synth = synth(|s| s.waveform = SynthWaveform::Square);
        voice.trigger(
            pitch_to_freq(60),
            127,
            sample_rate,
            synth.decay_seconds,
            &synth,
            None,
        );
        // Skip the filter's brief startup transient (zero initial filter state means the first
        // handful of samples ramp toward the input rather than matching it outright).
        for _ in 0..20 {
            voice.next_sample().0;
        }
        let mut peak = 0.0f32;
        for _ in 0..50 {
            peak = peak.max(voice.next_sample().0.abs());
        }
        assert!(
            peak > 0.9,
            "voice should reach near-full amplitude almost immediately, got {peak}"
        );
    }

    #[test]
    fn each_waveform_produces_a_bounded_signal() {
        for waveform in [
            SynthWaveform::Sine,
            SynthWaveform::Saw,
            SynthWaveform::Square,
            SynthWaveform::Triangle,
        ] {
            let mut voice = Voice::default();
            let synth = synth(|s| {
                s.waveform = waveform;
                s.decay_seconds = 1.0;
            });
            voice.trigger(pitch_to_freq(60), 127, 48_000.0, 1.0, &synth, None);
            // Skip the filter's startup transient — a resonant lowpass can briefly overshoot a
            // waveform's own discontinuities (Saw/Square's sharp edges) before settling into a
            // stable, bounded periodic response.
            for _ in 0..500 {
                voice.next_sample().0;
            }
            for _ in 0..200 {
                let s = voice.next_sample().0;
                assert!(
                    (-1.5..=1.5).contains(&s),
                    "{waveform:?} sample out of range after settling: {s}"
                );
            }
        }
    }

    #[test]
    fn narrow_pulse_width_spends_less_time_at_positive_polarity() {
        let count_high = |width: f32| -> usize {
            (0..1000)
                .filter(|&i| waveform_sample(SynthWaveform::Square, i as f32 / 1000.0, width) > 0.0)
                .count()
        };
        let narrow = count_high(0.1);
        let wide = count_high(0.9);
        assert!(
            narrow < wide,
            "a narrower pulse width should spend less time high: narrow={narrow} wide={wide}"
        );
    }

    #[test]
    fn sustain_holds_amplitude_until_gate_closes_then_releases() {
        let sample_rate = 48_000.0;
        let synth = synth(|s| {
            s.waveform = SynthWaveform::Square;
            s.attack_seconds = 0.0;
            s.decay_seconds = 0.05;
            s.sustain_level = 0.6;
            s.release_seconds = 0.05;
        });
        let mut voice = Voice::default();
        let gate_seconds = 0.2;
        voice.trigger(
            pitch_to_freq(60),
            127,
            sample_rate,
            gate_seconds,
            &synth,
            None,
        );

        // Run past attack+decay, into the sustain plateau, well before the gate closes.
        for _ in 0..(sample_rate * 0.1) as usize {
            voice.next_sample().0;
        }
        let mut sustain_peak = 0.0f32;
        for _ in 0..50 {
            sustain_peak = sustain_peak.max(voice.next_sample().0.abs());
        }
        assert!(
            sustain_peak > 0.5,
            "should be holding near the sustain level, got {sustain_peak}"
        );
        assert!(voice.active, "voice should still be active while gated");

        // Run past the gate close (0.2s) plus the release tail.
        for _ in 0..(sample_rate * 0.2) as usize {
            voice.next_sample().0;
        }
        assert!(
            !voice.active,
            "voice should have released to silence after the gate closed"
        );
    }

    #[test]
    fn unison_voices_produce_bounded_output_without_panicking() {
        let synth = synth(|s| {
            s.unison_voices = 3;
            s.unison_detune_cents = 15.0;
            s.decay_seconds = 1.0;
        });
        let mut voice = Voice::default();
        voice.trigger(pitch_to_freq(60), 127, 48_000.0, 1.0, &synth, None);
        for _ in 0..200 {
            let s = voice.next_sample().0;
            assert!(
                (-1.01..=1.01).contains(&s),
                "unison output out of range: {s}"
            );
        }
    }

    #[test]
    fn unison_width_zero_keeps_left_and_right_identical() {
        let synth = synth(|s| {
            s.unison_voices = 3;
            s.unison_detune_cents = 15.0;
            s.unison_width = 0.0;
            s.decay_seconds = 1.0;
        });
        let mut voice = Voice::default();
        voice.trigger(pitch_to_freq(60), 127, 48_000.0, 1.0, &synth, None);
        for _ in 0..200 {
            let (l, r) = voice.next_sample();
            assert_eq!(l, r, "unison_width 0.0 should keep every channel identical");
        }
    }

    #[test]
    fn unison_width_above_zero_spreads_left_and_right_apart() {
        let synth = synth(|s| {
            s.unison_voices = 3;
            s.unison_detune_cents = 15.0;
            s.unison_width = 1.0;
            s.decay_seconds = 1.0;
        });
        let mut voice = Voice::default();
        voice.trigger(pitch_to_freq(60), 127, 48_000.0, 1.0, &synth, None);
        let differs = (0..200).any(|_| {
            let (l, r) = voice.next_sample();
            (l - r).abs() > 1e-4
        });
        assert!(
            differs,
            "unison_width 1.0 with 3 unison voices should produce a genuinely different left and right signal"
        );
    }

    #[test]
    fn lower_filter_cutoff_attenuates_the_signal_more() {
        let sample_rate = 48_000.0;
        let peak_after_settling = |cutoff_hz: f32| -> f32 {
            let synth = synth(|s| {
                s.waveform = SynthWaveform::Square;
                s.decay_seconds = 1.0;
                s.filter_cutoff_hz = cutoff_hz;
            });
            let mut voice = Voice::default();
            voice.trigger(pitch_to_freq(69), 127, sample_rate, 1.0, &synth, None); // A4, ~440 Hz
            for _ in 0..1000 {
                voice.next_sample().0; // let the filter settle
            }
            let mut peak = 0.0f32;
            for _ in 0..200 {
                peak = peak.max(voice.next_sample().0.abs());
            }
            peak
        };

        let bright = peak_after_settling(20_000.0);
        let dark = peak_after_settling(150.0);
        assert!(
            dark < bright * 0.5,
            "a 150Hz low-pass on a 440Hz square wave should attenuate it much more than a wide-open filter: bright={bright} dark={dark}"
        );
    }

    #[test]
    fn all_filter_types_produce_bounded_signal_without_panicking() {
        for filter_type in [
            FilterType::Lowpass,
            FilterType::Highpass,
            FilterType::Bandpass,
            FilterType::Notch,
        ] {
            let synth = synth(|s| {
                s.decay_seconds = 1.0;
                s.filter_resonance = 5.0; // stress-test stability at high resonance
                s.filter_type = filter_type;
            });
            let mut voice = Voice::default();
            voice.trigger(pitch_to_freq(60), 127, 48_000.0, 1.0, &synth, None);
            for _ in 0..500 {
                voice.next_sample().0; // let the filter settle
            }
            for _ in 0..200 {
                let s = voice.next_sample().0;
                assert!(
                    (-2.0..=2.0).contains(&s),
                    "{filter_type:?} sample out of range after settling: {s}"
                );
            }
        }
    }

    #[test]
    fn highpass_attenuates_low_frequency_more_than_lowpass() {
        let sample_rate = 48_000.0;
        let rms_after_settling = |filter_type: FilterType| -> f32 {
            let synth = synth(|s| {
                s.waveform = SynthWaveform::Square;
                s.decay_seconds = 1.0;
                s.filter_cutoff_hz = 1000.0;
                s.filter_type = filter_type;
            });
            let mut voice = Voice::default();
            voice.trigger(pitch_to_freq(45), 127, sample_rate, 1.0, &synth, None); // A2, ~110 Hz, well below cutoff
            for _ in 0..1000 {
                voice.next_sample().0; // let the filter settle
            }
            let n = 400;
            let sum_sq: f32 = (0..n)
                .map(|_| {
                    let s = voice.next_sample().0;
                    s * s
                })
                .sum();
            (sum_sq / n as f32).sqrt()
        };

        let low = rms_after_settling(FilterType::Lowpass);
        let high = rms_after_settling(FilterType::Highpass);
        assert!(
            high < low * 0.5,
            "a 1kHz highpass on a 110Hz tone should attenuate it far more than the same-cutoff lowpass: low={low} high={high}"
        );
    }

    #[test]
    fn lfo_amplitude_target_produces_tremolo() {
        let sample_rate = 48_000.0;
        let peak_windows = |lfo_target: LfoTarget, lfo_depth: f32| -> Vec<f32> {
            let synth = synth(|s| {
                s.waveform = SynthWaveform::Sine;
                s.attack_seconds = 0.0;
                s.decay_seconds = 0.01;
                s.sustain_level = 1.0;
                s.lfo_target = lfo_target;
                s.lfo_depth = lfo_depth;
                s.lfo_rate_hz = 10.0; // 100ms period
            });
            let mut voice = Voice::default();
            voice.trigger(pitch_to_freq(69), 127, sample_rate, 1.0, &synth, None);
            for _ in 0..500 {
                voice.next_sample().0; // settle past attack/decay
            }
            let window = (sample_rate / 40.0) as usize; // 25ms, 4 windows per LFO cycle
            (0..8)
                .map(|_| {
                    let mut peak = 0.0f32;
                    for _ in 0..window {
                        peak = peak.max(voice.next_sample().0.abs());
                    }
                    peak
                })
                .collect()
        };

        let modulated = peak_windows(LfoTarget::Amplitude, 1.0);
        let flat = peak_windows(LfoTarget::None, 0.0);

        let mod_min = modulated.iter().cloned().fold(f32::INFINITY, f32::min);
        let mod_max = modulated.iter().cloned().fold(0.0f32, f32::max);
        let flat_min = flat.iter().cloned().fold(f32::INFINITY, f32::min);
        let flat_max = flat.iter().cloned().fold(0.0f32, f32::max);

        assert!(
            mod_max - mod_min > 0.3,
            "an amplitude-target LFO at full depth should visibly swing peak amplitude across windows: {modulated:?}"
        );
        assert!(
            flat_max - flat_min < 0.1,
            "without an LFO, amplitude should stay essentially flat: {flat:?}"
        );
    }

    #[test]
    fn osc2_mix_changes_output_from_osc1_alone() {
        let sample_rate = 48_000.0;
        let render = |osc2_mix: f32| -> Vec<f32> {
            let synth = synth(|s| {
                s.waveform = SynthWaveform::Sine;
                s.osc2_waveform = SynthWaveform::Square;
                s.osc2_mix = osc2_mix;
                s.decay_seconds = 1.0;
            });
            let mut voice = Voice::default();
            voice.trigger(pitch_to_freq(60), 127, sample_rate, 1.0, &synth, None);
            for _ in 0..500 {
                voice.next_sample().0;
            }
            (0..400).map(|_| voice.next_sample().0).collect()
        };

        let osc1_only = render(0.0);
        let osc2_only = render(1.0);
        let sum_sq_diff: f32 = osc1_only
            .iter()
            .zip(&osc2_only)
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        assert!(
            sum_sq_diff > 1.0,
            "crossfading fully to a differently-shaped osc2 should audibly change the output, got sum_sq_diff={sum_sq_diff}"
        );
    }

    #[test]
    fn osc2_sync_changes_output_when_osc2_is_detuned_from_osc1() {
        let sample_rate = 48_000.0;
        let render = |osc2_sync: bool| -> Vec<f32> {
            let synth = synth(|s| {
                s.waveform = SynthWaveform::Sine;
                s.osc2_waveform = SynthWaveform::Saw;
                s.osc2_semitones = 7; // detuned from osc1, so sync actually truncates its cycle
                s.osc2_mix = 1.0; // isolate osc2 in the output
                s.osc2_sync = osc2_sync;
                s.decay_seconds = 1.0;
            });
            let mut voice = Voice::default();
            voice.trigger(pitch_to_freq(60), 127, sample_rate, 1.0, &synth, None);
            for _ in 0..500 {
                voice.next_sample().0;
            }
            (0..400).map(|_| voice.next_sample().0).collect()
        };

        let unsynced = render(false);
        let synced = render(true);
        let sum_sq_diff: f32 = unsynced
            .iter()
            .zip(&synced)
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        assert!(
            sum_sq_diff > 1.0,
            "hard-syncing a detuned osc2 to osc1 should audibly change the output, got sum_sq_diff={sum_sq_diff}"
        );
    }

    #[test]
    fn sub_osc_adds_energy() {
        let sample_rate = 48_000.0;
        let rms = |sub_osc_mix: f32| -> f32 {
            let synth = synth(|s| {
                s.waveform = SynthWaveform::Sine;
                s.sub_osc_mix = sub_osc_mix;
                s.decay_seconds = 1.0;
            });
            let mut voice = Voice::default();
            voice.trigger(pitch_to_freq(69), 127, sample_rate, 1.0, &synth, None);
            for _ in 0..500 {
                voice.next_sample().0;
            }
            let n = 400;
            let sum_sq: f32 = (0..n)
                .map(|_| {
                    let s = voice.next_sample().0;
                    s * s
                })
                .sum();
            (sum_sq / n as f32).sqrt()
        };

        let without_sub = rms(0.0);
        let with_sub = rms(1.0);
        assert!(
            with_sub > without_sub * 1.2,
            "mixing in a full-level sub oscillator should increase output energy: without={without_sub} with={with_sub}"
        );
    }

    /// Rough instantaneous-frequency estimate via rising zero-crossing rate, good enough to tell
    /// "close to A" from "close to B" without needing an FFT — same spirit as the pitch-tracking
    /// verification technique used for the piano roll (see session history), just simplified for
    /// a unit test on a single clean sine partial.
    fn rising_zero_crossing_freq(samples: &[f32], sample_rate: f32) -> f32 {
        let crossings = samples
            .windows(2)
            .filter(|w| w[0] <= 0.0 && w[1] > 0.0)
            .count();
        crossings as f32 * sample_rate / samples.len() as f32
    }

    #[test]
    fn glide_ramps_frequency_toward_target_instead_of_jumping() {
        let sample_rate = 48_000.0;
        let start_freq = pitch_to_freq(48); // C3, ~130.8 Hz
        let target_freq = pitch_to_freq(72); // C5, ~261.6 Hz
        let glide_seconds = 0.3;

        let synth = synth(|s| {
            s.waveform = SynthWaveform::Sine;
            s.attack_seconds = 0.0;
            s.decay_seconds = 0.01;
            s.sustain_level = 1.0; // flat amplitude so zero-crossing counting stays reliable
            s.glide_seconds = glide_seconds;
        });
        let mut voice = Voice::default();
        voice.trigger(target_freq, 127, sample_rate, 1.0, &synth, Some(start_freq));

        let window = 2000; // ~42ms
        let early: Vec<f32> = (0..window).map(|_| voice.next_sample().0).collect();
        let early_freq = rising_zero_crossing_freq(&early, sample_rate);
        assert!(
            (early_freq - start_freq).abs() < (target_freq - start_freq) * 0.5,
            "shortly after retriggering, pitch should still be close to the start frequency, not the target: \
             early_freq={early_freq} start={start_freq} target={target_freq}"
        );

        // Run out the rest of the glide plus a settling margin, then sample a late window.
        let glide_samples = (glide_seconds * sample_rate) as usize;
        for _ in 0..(glide_samples - window + 500) {
            voice.next_sample().0;
        }
        let late: Vec<f32> = (0..window).map(|_| voice.next_sample().0).collect();
        let late_freq = rising_zero_crossing_freq(&late, sample_rate);
        assert!(
            (late_freq - target_freq).abs() < target_freq * 0.05,
            "once the glide has finished, pitch should have settled exactly on the target frequency: \
             late_freq={late_freq} target={target_freq}"
        );
    }

    #[test]
    fn sample_voice_plays_buffer_then_goes_silent() {
        let buffer = Arc::new(SampleBuffer {
            sample_rate: 48_000,
            mono: vec![1.0, 0.5, -0.5],
        });
        let mut voice = SampleVoice::default();
        voice.trigger(buffer, 127);

        assert!((voice.next_sample() - 1.0).abs() < 0.001);
        assert!((voice.next_sample() - 0.5).abs() < 0.001);
        assert!((voice.next_sample() - -0.5).abs() < 0.001);
        assert_eq!(
            voice.next_sample(),
            0.0,
            "voice should be silent past the end of the buffer"
        );
        assert!(
            voice.buffer.is_none(),
            "voice should free itself once exhausted"
        );
    }

    #[test]
    fn trigger_clip_plays_only_the_trimmed_window() {
        let buffer = Arc::new(SampleBuffer {
            sample_rate: 48_000,
            mono: vec![1.0, 2.0, 3.0, 4.0, 5.0],
        });
        let mut voice = SampleVoice::default();
        // Trim to frames 1..4 (values 2.0, 3.0, 4.0), no fades.
        voice.trigger_clip(buffer, 1.0, 1, 4, 0, 0, false);

        assert_eq!(voice.next_sample(), 2.0);
        assert_eq!(voice.next_sample(), 3.0);
        assert_eq!(voice.next_sample(), 4.0);
        assert_eq!(
            voice.next_sample(),
            0.0,
            "voice should stop at end_frame, ignoring the rest of the buffer"
        );
    }

    #[test]
    fn trigger_clip_with_looping_wraps_back_to_start_instead_of_stopping() {
        let buffer = Arc::new(SampleBuffer {
            sample_rate: 48_000,
            mono: vec![1.0, 2.0, 3.0],
        });
        let mut voice = SampleVoice::default();
        voice.trigger_clip(buffer, 1.0, 0, 3, 0, 0, true);

        assert_eq!(voice.next_sample(), 1.0);
        assert_eq!(voice.next_sample(), 2.0);
        assert_eq!(voice.next_sample(), 3.0);
        assert_eq!(voice.next_sample(), 1.0, "should wrap back to the start, not go silent");
        assert_eq!(voice.next_sample(), 2.0);
    }

    #[test]
    fn trigger_clip_ramps_fade_in_and_fade_out_linearly() {
        let buffer = Arc::new(SampleBuffer {
            sample_rate: 48_000,
            mono: vec![1.0; 8],
        });
        let mut voice = SampleVoice::default();
        // 8-frame clip, 2-frame fade-in, 2-frame fade-out.
        voice.trigger_clip(buffer, 1.0, 0, 8, 2, 2, false);

        let samples: Vec<f32> = (0..8).map(|_| voice.next_sample()).collect();
        // Fade-in ramps over frames 0..2 (elapsed frames played so far); fade-out ramps over the
        // last 2 frames remaining before `end_position` — the last played frame (7) is always one
        // step short of `end_position`, so it never hits exactly 0.0.
        assert!((samples[0] - 0.0).abs() < 0.001, "frame 0: {}", samples[0]);
        assert!((samples[1] - 0.5).abs() < 0.001, "frame 1: {}", samples[1]);
        assert!((samples[2] - 1.0).abs() < 0.001, "frame 2: fully faded in");
        assert!((samples[5] - 1.0).abs() < 0.001, "frame 5: still full before fade-out");
        assert!((samples[6] - 1.0).abs() < 0.001, "frame 6: {}", samples[6]);
        assert!((samples[7] - 0.5).abs() < 0.001, "frame 7: {}", samples[7]);
    }

    /// De-interleaves a stereo `i16` sample stream (as `render_song_to_wav` now writes) back down
    /// to just its left channel, so tests written against the format's old mono sample-index math
    /// don't all need their index literals doubled.
    fn left_channel(samples: &[i16]) -> Vec<i16> {
        samples.iter().step_by(2).copied().collect()
    }

    /// The right-channel counterpart of `left_channel`.
    fn right_channel(samples: &[i16]) -> Vec<i16> {
        samples.iter().skip(1).step_by(2).copied().collect()
    }

    #[test]
    fn render_song_to_wav_produces_expected_length_and_nonsilent_audio() {
        let song = crate::model::Song::demo();
        let sample_rate = 48_000u32;
        let path = std::env::temp_dir().join(format!("simple_daw_test_{}.wav", std::process::id()));

        render_song_to_wav(&song, sample_rate, 1, &path).expect("export should succeed");

        let mut reader = hound::WavReader::open(&path).expect("exported wav should be readable");
        let spec = reader.spec();
        assert_eq!(spec.channels, 2);
        assert_eq!(spec.sample_rate, sample_rate);

        let samples: Vec<i16> = left_channel(
            &reader
                .samples::<i16>()
                .collect::<std::result::Result<Vec<_>, _>>()
                .unwrap(),
        );
        let samples_per_step = (sample_rate as f64 * 60.0 / 120.0 / STEPS_PER_BEAT).max(1.0);
        let expected_len = (16.0 * samples_per_step).round() as i64;
        assert!(
            (samples.len() as i64 - expected_len).abs() <= 1,
            "expected ~{expected_len} samples for one loop, got {}",
            samples.len()
        );

        let nonzero = samples.iter().filter(|&&s| s != 0).count();
        assert!(
            nonzero > samples.len() / 10,
            "exported audio should be clearly non-silent, only {nonzero}/{} nonzero samples",
            samples.len()
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn render_song_to_wav_produces_expected_length_with_a_tempo_map() {
        let mut song = crate::model::Song::demo();
        song.bpm = 120.0;
        let sample_rate = 48_000u32;
        let arrangement_len_ticks = arrangement_length_ticks(&song);
        let half = arrangement_len_ticks / 2;
        song.set_tempo_at(half, 240.0);
        let path = std::env::temp_dir()
            .join(format!("simple_daw_test_tempo_map_{}.wav", std::process::id()));

        render_song_to_wav(&song, sample_rate, 1, &path).expect("export should succeed");

        let reader = hound::WavReader::open(&path).expect("exported wav should be readable");
        let frame_count = reader.len() as usize / 2; // interleaved stereo
        std::fs::remove_file(&path).ok();

        let expected_len = (half as f64 * samples_per_tick_at(sample_rate as f64, 120.0)
            + (arrangement_len_ticks - half) as f64 * samples_per_tick_at(sample_rate as f64, 240.0))
            .round() as i64;
        assert!(
            (frame_count as i64 - expected_len).abs() <= 1,
            "expected ~{expected_len} frames with a tempo change halfway through, got {frame_count}"
        );
    }

    #[test]
    fn samples_for_tick_span_matches_the_flat_formula_with_no_tempo_map() {
        let song = crate::model::Song::demo();
        let sample_rate = 44_100.0;
        let span = 1000;
        let expected = span as f64 * samples_per_tick_at(sample_rate, song.bpm);
        assert!((samples_for_tick_span(&song, sample_rate, span) - expected).abs() < 0.001);
    }

    #[test]
    fn samples_for_tick_span_sums_each_segments_own_duration() {
        let mut song = crate::model::Song::demo();
        song.bpm = 120.0;
        song.set_tempo_at(400, 240.0);
        let sample_rate = 44_100.0;
        let span = 1000;
        let expected = 400.0 * samples_per_tick_at(sample_rate, 120.0)
            + 600.0 * samples_per_tick_at(sample_rate, 240.0);
        assert!((samples_for_tick_span(&song, sample_rate, span) - expected).abs() < 0.001);
    }

    #[test]
    fn samples_for_tick_span_ignores_tempo_points_beyond_the_span() {
        let mut song = crate::model::Song::demo();
        song.set_tempo_at(5000, 999.0);
        let sample_rate = 44_100.0;
        let span = 1000;
        let expected = span as f64 * samples_per_tick_at(sample_rate, song.bpm);
        assert!((samples_for_tick_span(&song, sample_rate, span) - expected).abs() < 0.001);
    }

    #[test]
    fn render_song_to_wav_applies_a_tracks_own_effect_chain() {
        let sample_rate = 48_000u32;
        let mut song = song_with_regions(vec![sustained_note_region_with_fade_in(8, 0)]);
        // Hold at full amplitude for the whole region, isolating the effect chain's influence
        // from the synth's own (by-default percussive) envelope shape — same reasoning as
        // `region_fade_in_silences_the_first_tick_then_reaches_full_volume`.
        song.tracks[0].synth.attack_seconds = 0.0;
        song.tracks[0].synth.decay_seconds = 0.0;
        song.tracks[0].synth.sustain_level = 1.0;
        song.tracks[0].effects = vec![crate::model::TrackEffectConfig::PhaseInvert {
            invert_left: true,
            invert_right: true,
        }];

        let path = std::env::temp_dir()
            .join(format!("simple-daw-test-wet-track-fx-{}.wav", std::process::id()));
        render_song_to_wav(&song, sample_rate, 1, &path).expect("export should succeed");
        let mut reader = hound::WavReader::open(&path).expect("exported wav should be readable");
        let inverted = left_channel(
            &reader
                .samples::<i16>()
                .collect::<std::result::Result<Vec<_>, _>>()
                .unwrap(),
        );
        std::fs::remove_file(&path).ok();

        let mut dry_song = song.clone();
        dry_song.tracks[0].effects.clear();
        let dry_path = std::env::temp_dir()
            .join(format!("simple-daw-test-wet-track-fx-dry-{}.wav", std::process::id()));
        render_song_to_wav(&dry_song, sample_rate, 1, &dry_path).expect("export should succeed");
        let mut dry_reader =
            hound::WavReader::open(&dry_path).expect("exported wav should be readable");
        let dry = left_channel(
            &dry_reader
                .samples::<i16>()
                .collect::<std::result::Result<Vec<_>, _>>()
                .unwrap(),
        );
        std::fs::remove_file(&dry_path).ok();

        assert_eq!(inverted.len(), dry.len());
        assert!(dry.iter().any(|&s| s.unsigned_abs() > 200), "dry render should be clearly audible");
        let mismatched_signs = inverted
            .iter()
            .zip(&dry)
            .filter(|(a, b)| b.unsigned_abs() > 200 && a.signum() != -b.signum())
            .count();
        assert_eq!(
            mismatched_signs, 0,
            "a track's own PhaseInvert effect chain should flip sign wherever the dry render is \
             clearly nonzero, confirming the bounce actually routes through a track's own chain \
             now instead of staying dry-only"
        );
    }

    #[test]
    fn render_song_to_wav_ducks_a_track_via_another_tracks_sidechain_source() {
        // Track 0 ("Kick") is loud; track 1 ("Bass") is quiet on its own — quiet enough that its
        // own Compressor, alone, never crosses the threshold below. Only when track 1's Compressor
        // is routed to key off track 0 (`sidechain_source: Some(0)`) should track 1 measurably
        // duck, proving `all_track_dry` threading actually resolves a cross-track key.
        //
        // Both tracks sum into the same master bounce, so isolating bass's own contribution means
        // rendering the same song three ways (kick-only, kick+bass unsidechained, kick+bass
        // sidechained) and subtracting out kick's identical contribution each time, rather than
        // reading the combined master output directly.
        let mut kick = crate::model::Track::new_piano_roll("Kick", 1);
        kick.regions.push(sustained_note_region_with_fade_in(8, 0));
        kick.synth.attack_seconds = 0.0;
        kick.synth.decay_seconds = 0.0;
        kick.synth.sustain_level = 1.0;

        let mut bass = crate::model::Track::new_piano_roll("Bass", 2);
        let mut bass_region = sustained_note_region_with_fade_in(8, 0);
        if let RegionContent::PianoRoll(notes) = &mut bass_region.content {
            notes[0].velocity = 20;
        }
        bass.regions.push(bass_region);
        bass.synth.attack_seconds = 0.0;
        bass.synth.decay_seconds = 0.0;
        bass.synth.sustain_level = 1.0;
        bass.effects = vec![crate::model::TrackEffectConfig::Compressor {
            threshold_db: -6.0,
            ratio: 20.0,
            attack_ms: 1.0,
            release_ms: 50.0,
            makeup_db: 0.0,
            sidechain_source: None,
        }];

        let sample_rate = 48_000u32;
        let mut song = song_with_regions(Vec::new());
        song.tracks = vec![kick, bass];

        fn render_left_channel(song: &crate::model::Song, sample_rate: u32, tag: &str) -> Vec<i16> {
            let path = std::env::temp_dir()
                .join(format!("simple-daw-test-sidechain-{tag}-{}.wav", std::process::id()));
            render_song_to_wav(song, sample_rate, 1, &path).expect("export should succeed");
            let mut reader = hound::WavReader::open(&path).unwrap();
            let samples = left_channel(
                &reader.samples::<i16>().collect::<std::result::Result<Vec<_>, _>>().unwrap(),
            );
            std::fs::remove_file(&path).ok();
            samples
        }

        let mut kick_only_song = song.clone();
        kick_only_song.tracks[1].regions.clear();
        let kick_only = render_left_channel(&kick_only_song, sample_rate, "kick-only");

        let unducked_master = render_left_channel(&song, sample_rate, "off");

        if let crate::model::TrackEffectConfig::Compressor { sidechain_source, .. } =
            &mut song.tracks[1].effects[0]
        {
            *sidechain_source = Some(0);
        }
        let ducked_master = render_left_channel(&song, sample_rate, "on");

        let settle = 3000;
        let bass_unducked = unducked_master[settle] as i32 - kick_only[settle] as i32;
        let bass_ducked = ducked_master[settle] as i32 - kick_only[settle] as i32;
        assert!(
            bass_unducked.unsigned_abs() > 100,
            "bass's own isolated contribution should be clearly audible before ducking: {bass_unducked}"
        );
        assert!(
            bass_ducked.unsigned_abs() < bass_unducked.unsigned_abs(),
            "routing the compressor's sidechain to the loud kick track should duck bass's own \
             isolated contribution: unducked={bass_unducked}, ducked={bass_ducked}"
        );
    }

    #[test]
    fn render_track_to_buffer_applies_that_tracks_own_effect_chain_in_isolation() {
        // Two tracks: track 0 is the one under test (sustained tone + PhaseInvert), track 1 is a
        // second, unrelated loud track — `render_track_to_buffer` must render track 0 alone, not
        // pick up any of track 1's signal, proving it isolates one track the way `Sequencer::process`
        // computes `track_dry_l/r` per track but this function bakes only one of them through its
        // own chain.
        let mut lead = crate::model::Track::new_piano_roll("Lead", 1);
        lead.regions.push(sustained_note_region_with_fade_in(8, 0));
        lead.synth.attack_seconds = 0.0;
        lead.synth.decay_seconds = 0.0;
        lead.synth.sustain_level = 1.0;
        lead.effects = vec![crate::model::TrackEffectConfig::PhaseInvert {
            invert_left: true,
            invert_right: true,
        }];

        let mut other = crate::model::Track::new_piano_roll("Other", 2);
        other.regions.push(sustained_note_region_with_fade_in(8, 0));
        other.synth.attack_seconds = 0.0;
        other.synth.decay_seconds = 0.0;
        other.synth.sustain_level = 1.0;

        let sample_rate = 48_000u32;
        let mut song = song_with_regions(Vec::new());
        song.tracks = vec![lead.clone(), other];

        let (inverted_l, _) = render_track_to_buffer(&song, 0, sample_rate)
            .expect("track 0 should render to a nonempty buffer");

        song.tracks[0].effects.clear();
        let (dry_l, _) = render_track_to_buffer(&song, 0, sample_rate)
            .expect("track 0 should render to a nonempty buffer");

        assert_eq!(inverted_l.len(), dry_l.len());
        assert!(dry_l.iter().any(|&s| s.abs() > 0.05), "dry render should be clearly audible");
        let mismatched_signs = inverted_l
            .iter()
            .zip(&dry_l)
            .filter(|(a, b)| b.abs() > 0.05 && a.signum() != -b.signum())
            .count();
        assert_eq!(
            mismatched_signs, 0,
            "render_track_to_buffer should apply track 0's own PhaseInvert, flipping sign \
             wherever the dry render is clearly nonzero"
        );
    }

    #[test]
    fn render_song_to_wav_applies_a_sends_own_effect_chain() {
        let sample_rate = 48_000u32;
        let mut song = song_with_regions(vec![sustained_note_region_with_fade_in(8, 0)]);
        song.tracks[0].synth.attack_seconds = 0.0;
        song.tracks[0].synth.decay_seconds = 0.0;
        song.tracks[0].synth.sustain_level = 1.0;
        song.tracks[0].send_levels = vec![1.0];
        song.sends.push(crate::model::SendBus {
            name: "Send A".to_string(),
            effects: vec![crate::model::TrackEffectConfig::PhaseInvert {
                invert_left: true,
                invert_right: true,
            }],
        });

        let render = |song: &crate::model::Song, tag: &str| -> Vec<i16> {
            let path = std::env::temp_dir()
                .join(format!("simple-daw-test-wet-send-fx-{tag}-{}.wav", std::process::id()));
            render_song_to_wav(song, sample_rate, 1, &path).expect("export should succeed");
            let mut reader =
                hound::WavReader::open(&path).expect("exported wav should be readable");
            let samples = left_channel(
                &reader
                    .samples::<i16>()
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .unwrap(),
            );
            std::fs::remove_file(&path).ok();
            samples
        };

        let with_send = render(&song, "with");
        // The track's own direct (dry, pan/volume-only) contribution to master is unchanged
        // either way — only the send's contribution toggles, so any difference below can only
        // come from the send bus's own chain actually running now, not being excluded.
        song.tracks[0].send_levels = vec![0.0];
        let without_send = render(&song, "without");

        assert_ne!(
            with_send, without_send,
            "a send's own effect chain should audibly change the bounce, confirming sends are no \
             longer excluded from it"
        );
    }

    #[test]
    fn render_song_to_wav_applies_track_wide_volume_automation_over_time() {
        let sample_rate = 48_000u32;
        let loop_length_steps = 64;
        let mut song =
            song_with_regions(vec![sustained_note_region_with_fade_in(loop_length_steps, 0)]);
        song.tracks[0].synth.attack_seconds = 0.0;
        song.tracks[0].synth.decay_seconds = 0.0;
        song.tracks[0].synth.sustain_level = 1.0;
        let span_ticks = loop_length_steps * TICKS_PER_STEP;
        song.tracks[0].automation = vec![crate::model::AutomationLane {
            target: crate::model::AutomationTarget::Volume,
            points: vec![
                crate::model::AutomationPoint { tick: 0, value: 0.0, curve: CurveShape::default() },
                crate::model::AutomationPoint { tick: span_ticks, value: 1.0, curve: CurveShape::default() },
            ],
        }];

        let path = std::env::temp_dir()
            .join(format!("simple-daw-test-wet-automation-{}.wav", std::process::id()));
        render_song_to_wav(&song, sample_rate, 1, &path).expect("export should succeed");
        let mut reader = hound::WavReader::open(&path).expect("exported wav should be readable");
        let samples = left_channel(
            &reader
                .samples::<i16>()
                .collect::<std::result::Result<Vec<_>, _>>()
                .unwrap(),
        );
        std::fs::remove_file(&path).ok();

        let quarter = samples.len() / 4;
        let early_peak = samples[..quarter].iter().map(|s| s.unsigned_abs()).max().unwrap_or(0);
        let late_peak =
            samples[samples.len() - quarter..].iter().map(|s| s.unsigned_abs()).max().unwrap_or(0);
        assert!(
            late_peak > early_peak * 2,
            "volume automation ramping 0.0 -> 1.0 across the render should make it noticeably \
             louder later than earlier, got early={early_peak} late={late_peak}"
        );
    }

    /// A one-track piano-roll song with explicit `regions` on that track, for testing
    /// `Sequencer::process`'s region-lookup logic directly (as opposed to `Song::demo()`'s
    /// implicit single-region shape, which every track always has content for).
    fn song_with_regions(regions: Vec<crate::model::Region>) -> crate::model::Song {
        let mut track = crate::model::Track::new_piano_roll("Lead", 1);
        track.regions = regions;
        crate::model::Song {
            name: "region test".to_string(),
            bpm: 120.0,
            tempo_map: Vec::new(),
            tracks: vec![track],
            next_note_id: 0,
            master_effects: Vec::new(),
            plugins: Vec::new(),
            synth_presets: Vec::new(),
            time_signature_numerator: 4,
            time_signature_denominator: 4,
            sends: Vec::new(),
            submixes: Vec::new(),
            scenes: Vec::new(),
        }
    }

    fn one_note_region(
        start_tick: usize,
        content_length_steps: usize,
        loop_length_steps: usize,
        note_start_tick: usize,
        note_length_ticks: usize,
    ) -> crate::model::Region {
        crate::model::Region {
            name: "Hit".to_string(),
            start_tick,
            content_length_steps,
            loop_length_steps,
            content: RegionContent::PianoRoll(vec![crate::model::Note {
                id: 0,
                pitch: 60,
                start_tick: note_start_tick,
                length_ticks: note_length_ticks,
                velocity: 127,
            }]),
            fade_in_ticks: 0,
            fade_out_ticks: 0,
            automation: Vec::new(),
        }
    }

    /// Same as `one_note_region`, but with a note sustained for the whole on-timeline span (so its
    /// amplitude envelope is already well past its own attack by the time the region's fade would
    /// matter) and an explicit `fade_in_ticks` — for testing `Sequencer::process`'s region-fade
    /// gain, isolated from the synth's own envelope shape.
    fn sustained_note_region_with_fade_in(
        loop_length_steps: usize,
        fade_in_ticks: usize,
    ) -> crate::model::Region {
        crate::model::Region {
            name: "Hit".to_string(),
            start_tick: 0,
            content_length_steps: loop_length_steps,
            loop_length_steps,
            content: RegionContent::PianoRoll(vec![crate::model::Note {
                id: 0,
                pitch: 60,
                start_tick: 0,
                length_ticks: loop_length_steps * TICKS_PER_STEP,
                velocity: 127,
            }]),
            fade_in_ticks,
            fade_out_ticks: 0,
            automation: Vec::new(),
        }
    }

    fn region_with_automation(
        loop_length_steps: usize,
        lanes: Vec<crate::model::AutomationLane>,
    ) -> crate::model::Region {
        crate::model::Region {
            name: "Automated".to_string(),
            start_tick: 0,
            content_length_steps: loop_length_steps,
            loop_length_steps,
            content: RegionContent::PianoRoll(Vec::new()),
            fade_in_ticks: 0,
            fade_out_ticks: 0,
            automation: lanes,
        }
    }

    #[test]
    fn collect_automation_is_default_with_no_active_region() {
        let song = song_with_regions(Vec::new());
        let (tracks, master, sends) = collect_automation(&song, 0, false, &[]);
        assert!(tracks[0].volume.is_none());
        assert!(tracks[0].pan.is_none());
        assert!(tracks[0].send_levels.is_empty());
        assert!(tracks[0].effect_params.is_empty());
        assert!(master.effect_params.is_empty());
        assert!(sends.is_empty());
    }

    #[test]
    fn collect_automation_is_default_when_the_active_region_has_no_lanes() {
        let song = song_with_regions(vec![region_with_automation(4, Vec::new())]);
        let (tracks, _, _) = collect_automation(&song, 10, false, &[]);
        assert!(tracks[0].volume.is_none());
        assert!(tracks[0].pan.is_none());
    }

    #[test]
    fn collect_automation_reads_volume_and_pan_from_the_active_region() {
        use crate::model::{AutomationLane, AutomationPoint, AutomationTarget};
        let song = song_with_regions(vec![region_with_automation(
            4,
            vec![
                AutomationLane {
                    target: AutomationTarget::Volume,
                    points: vec![
                        AutomationPoint { tick: 0, value: 0.0, curve: CurveShape::default() },
                        AutomationPoint { tick: 96, value: 1.0, curve: CurveShape::default() },
                    ],
                },
                AutomationLane {
                    target: AutomationTarget::Pan,
                    points: vec![AutomationPoint { tick: 0, value: -1.0, curve: CurveShape::default() }],
                },
            ],
        )]);
        let (tracks, _, _) = collect_automation(&song, 48, false, &[]);
        let volume = tracks[0].volume.unwrap().value_at(0, 1.0);
        assert!((volume - 0.5).abs() < 1e-6);
        let pan = tracks[0].pan.unwrap().value_at(0, 1.0);
        assert_eq!(pan, -1.0);
    }

    #[test]
    fn collect_automation_ignores_a_region_that_is_not_active_at_this_tick() {
        use crate::model::{AutomationLane, AutomationPoint, AutomationTarget};
        let song = song_with_regions(vec![region_with_automation(
            4,
            vec![AutomationLane {
                target: AutomationTarget::Volume,
                points: vec![AutomationPoint { tick: 0, value: 0.25, curve: CurveShape::default() }],
            }],
        )]);
        // Past the region's on-timeline span (4 steps * TICKS_PER_STEP).
        let (tracks, _, _) = collect_automation(&song, 4 * TICKS_PER_STEP + 1, false, &[]);
        assert!(tracks[0].volume.is_none());
    }

    #[test]
    fn collect_automation_collects_send_level_and_effect_param_lanes() {
        use crate::model::{AutomationLane, AutomationPoint, AutomationTarget, EffectParamKey};
        let song = song_with_regions(vec![region_with_automation(
            4,
            vec![
                AutomationLane {
                    target: AutomationTarget::SendLevel { send_index: 2 },
                    points: vec![AutomationPoint { tick: 0, value: 0.6, curve: CurveShape::default() }],
                },
                AutomationLane {
                    target: AutomationTarget::EffectParam {
                        slot_index: 0,
                        key: EffectParamKey::BuiltIn { param_name: "Mix".to_string() },
                    },
                    points: vec![AutomationPoint { tick: 0, value: 0.3, curve: CurveShape::default() }],
                },
            ],
        )]);
        let (tracks, _, _) = collect_automation(&song, 0, false, &[]);
        assert_eq!(tracks[0].send_levels.len(), 1);
        let (send_index, lane) = &tracks[0].send_levels[0];
        assert_eq!(*send_index, 2);
        assert_eq!(lane.value_at(0, 1.0), 0.6);
        assert_eq!(tracks[0].effect_params.len(), 1);
        let (slot_index, key, lane) = &tracks[0].effect_params[0];
        assert_eq!(*slot_index, 0);
        assert_eq!(**key, EffectParamKey::BuiltIn { param_name: "Mix".to_string() });
        assert_eq!(lane.value_at(0, 1.0), 0.3);
    }

    #[test]
    fn collect_automation_redirects_other_track_and_master_targets() {
        use crate::model::{AutomationLane, AutomationPoint, AutomationTarget, EffectParamKey};
        let mut song = song_with_regions(vec![region_with_automation(
            4,
            vec![
                AutomationLane {
                    target: AutomationTarget::OtherTrackVolume { track_index: 1 },
                    points: vec![AutomationPoint { tick: 0, value: 0.4, curve: CurveShape::default() }],
                },
                AutomationLane {
                    target: AutomationTarget::MasterEffectParam {
                        slot_index: 0,
                        key: EffectParamKey::BuiltIn { param_name: "Mix".to_string() },
                    },
                    points: vec![AutomationPoint { tick: 0, value: 0.8, curve: CurveShape::default() }],
                },
            ],
        )]);
        // A second track, with no automation of its own, to be the redirect target.
        song.tracks.push(crate::model::Track::new_piano_roll("Other", 1));

        let (tracks, master, _) = collect_automation(&song, 0, false, &[]);
        assert!(tracks[0].volume.is_none(), "lane redirects away from its own track");
        assert_eq!(tracks[1].volume.unwrap().value_at(0, 1.0), 0.4);
        assert_eq!(master.effect_params.len(), 1);
        let (slot_index, key, lane) = &master.effect_params[0];
        assert_eq!(*slot_index, 0);
        assert_eq!(**key, EffectParamKey::BuiltIn { param_name: "Mix".to_string() });
        assert_eq!(lane.value_at(0, 1.0), 0.8);
    }

    #[test]
    fn collect_automation_uses_track_wide_lane_where_no_region_is_active() {
        use crate::model::{AutomationLane, AutomationPoint, AutomationTarget};
        let mut song = song_with_regions(Vec::new());
        song.tracks[0].automation.push(AutomationLane {
            target: AutomationTarget::Volume,
            points: vec![
                AutomationPoint { tick: 0, value: 0.2, curve: CurveShape::default() },
                AutomationPoint { tick: 100, value: 1.0, curve: CurveShape::default() },
            ],
        });
        // No region at all, let alone one active at tick 50 — the track-wide lane still applies,
        // evaluated at the absolute tick (unlike a region lane, not offset by any region start).
        let (tracks, _, _) = collect_automation(&song, 50, false, &[]);
        assert_eq!(tracks[0].volume.unwrap().value_at(0, 1.0), 0.6);
    }

    #[test]
    fn collect_automation_lets_an_active_regions_lane_override_a_track_wide_lane() {
        use crate::model::{AutomationLane, AutomationPoint, AutomationTarget};
        let mut song = song_with_regions(vec![region_with_automation(
            4,
            vec![AutomationLane {
                target: AutomationTarget::Volume,
                points: vec![AutomationPoint { tick: 0, value: 0.9, curve: CurveShape::default() }],
            }],
        )]);
        song.tracks[0].automation.push(AutomationLane {
            target: AutomationTarget::Volume,
            points: vec![AutomationPoint { tick: 0, value: 0.1, curve: CurveShape::default() }],
        });
        // Tick 0 is inside the region's on-timeline span — its lane should win over the
        // track-wide one on the same target (Volume), per `Track::automation`'s doc comment.
        let (tracks, _, _) = collect_automation(&song, 0, false, &[]);
        assert_eq!(tracks[0].volume.unwrap().value_at(0, 1.0), 0.9);
    }

    fn song_with_session_clip(clip: crate::model::SessionClip) -> crate::model::Song {
        let mut track = crate::model::Track::new_piano_roll("Lead", 1);
        track.session_clips = vec![Some(clip)];
        crate::model::Song {
            name: "session clip automation test".to_string(),
            bpm: 120.0,
            tempo_map: Vec::new(),
            tracks: vec![track],
            next_note_id: 0,
            master_effects: Vec::new(),
            plugins: Vec::new(),
            synth_presets: Vec::new(),
            time_signature_numerator: 4,
            time_signature_denominator: 4,
            sends: Vec::new(),
            submixes: Vec::new(),
            scenes: Vec::new(),
        }
    }

    #[test]
    fn collect_automation_reads_a_playing_session_clips_lane_in_session_mode() {
        use crate::model::{AutomationLane, AutomationPoint, AutomationTarget};
        let mut clip = crate::model::SessionClip::new_piano_roll("Clip", 4);
        clip.automation.push(AutomationLane {
            target: AutomationTarget::Volume,
            points: vec![
                AutomationPoint { tick: 0, value: 0.0, curve: CurveShape::default() },
                AutomationPoint { tick: 96, value: 1.0, curve: CurveShape::default() },
            ],
        });
        let song = song_with_session_clip(clip);
        let session_slots = vec![vec![SlotState::Playing { local_tick: 48, loop_count: 0 }]];
        // The absolute tick (0) is irrelevant in session mode — only the slot's own `local_tick`
        // (48, halfway through the lane's 0..96 span) matters.
        let (tracks, _, _) = collect_automation(&song, 0, true, &session_slots);
        let volume = tracks[0].volume.unwrap().value_at(0, 1.0);
        assert!((volume - 0.5).abs() < 1e-6);
    }

    #[test]
    fn collect_automation_ignores_a_session_clips_lane_outside_session_mode() {
        use crate::model::{AutomationLane, AutomationPoint, AutomationTarget};
        let mut clip = crate::model::SessionClip::new_piano_roll("Clip", 4);
        clip.automation.push(AutomationLane {
            target: AutomationTarget::Volume,
            points: vec![AutomationPoint { tick: 0, value: 0.25, curve: CurveShape::default() }],
        });
        let song = song_with_session_clip(clip);
        let session_slots = vec![vec![SlotState::Playing { local_tick: 0, loop_count: 0 }]];
        // Same slot state as above, but `session_mode` is false — a session clip's own lanes
        // never apply outside Session View, matching `Sequencer::process`'s own mode switch.
        let (tracks, _, _) = collect_automation(&song, 0, false, &session_slots);
        assert!(tracks[0].volume.is_none());
    }

    #[test]
    fn collect_automation_ignores_a_stopped_slots_lane_in_session_mode() {
        use crate::model::{AutomationLane, AutomationPoint, AutomationTarget};
        let mut clip = crate::model::SessionClip::new_piano_roll("Clip", 4);
        clip.automation.push(AutomationLane {
            target: AutomationTarget::Volume,
            points: vec![AutomationPoint { tick: 0, value: 0.25, curve: CurveShape::default() }],
        });
        let song = song_with_session_clip(clip);
        let session_slots = vec![vec![SlotState::Stopped]];
        let (tracks, _, _) = collect_automation(&song, 0, true, &session_slots);
        assert!(tracks[0].volume.is_none());
    }

    #[test]
    fn region_fade_in_silences_the_first_tick_then_reaches_full_volume() {
        let sample_rate = 48_000.0;
        // A generous fade relative to one tick's worth of samples at a typical tempo/sample rate,
        // so "well past the fade" isn't razor-close to the boundary computed below.
        let fade_in_ticks = 20;
        let region = sustained_note_region_with_fade_in(8, fade_in_ticks);
        let mut song = song_with_regions(vec![region]);
        // The default `SynthParams` is percussive (sustain_level 0.0 — decays to silence after
        // `decay_seconds` regardless of gate length, see the type's own doc comment), which would
        // confound this test's "still audible well after the fade" check with the synth's own
        // decay. Hold at full amplitude instead, isolating the region fade's effect.
        song.tracks[0].synth.attack_seconds = 0.0;
        song.tracks[0].synth.decay_seconds = 0.0;
        song.tracks[0].synth.sustain_level = 1.0;

        let mut sequencer = Sequencer::new(sample_rate);
        let mut track_out_l = Vec::new();
        let mut track_out_r = Vec::new();
        let mut metronome_out = Vec::new();
        // Enough frames to cover the fade-in window several times over regardless of tempo.
        sequencer.process(
            &song,
            48_000,
            &mut track_out_l,
            &mut track_out_r,
            false,
            &mut metronome_out,
            false,
            0,
            false, // capturing
        );

        let samples_per_tick = (sample_rate as f64 * 60.0
            / (song.bpm.max(1.0) as f64)
            / STEPS_PER_BEAT
            / TICKS_PER_STEP as f64)
            .max(1.0);
        let first_tick_samples = samples_per_tick as usize;
        assert!(
            track_out_l[0][..first_tick_samples].iter().all(|&s| s == 0.0),
            "the region's first tick should be fully silenced by fade_gain_at(0) == 0.0"
        );

        let well_past_fade = ((fade_in_ticks + 4) as f64 * samples_per_tick) as usize;
        assert!(
            track_out_l[0][well_past_fade..]
                .iter()
                .any(|&s| s.abs() > 0.1),
            "well after fade_in_ticks the note should be audible at full gain"
        );
    }

    #[test]
    fn sequencer_process_honors_a_tempo_map_change_at_the_right_tick() {
        let sample_rate = 48_000.0;
        let tempo_change_tick = 4 * TICKS_PER_STEP;
        let note_tick = 6 * TICKS_PER_STEP; // two steps after the tempo change
        let region = one_note_region(0, 8, 8, note_tick, TICKS_PER_STEP);
        let mut song = song_with_regions(vec![region]);
        song.bpm = 120.0;
        song.set_tempo_at(tempo_change_tick, 240.0);
        // Hold at full amplitude immediately so the note's onset sample is unambiguous.
        song.tracks[0].synth.attack_seconds = 0.0;
        song.tracks[0].synth.decay_seconds = 0.0;
        song.tracks[0].synth.sustain_level = 1.0;

        let mut sequencer = Sequencer::new(sample_rate);
        let mut track_out_l = Vec::new();
        let mut track_out_r = Vec::new();
        let mut metronome_out = Vec::new();
        sequencer.process(
            &song,
            48_000,
            &mut track_out_l,
            &mut track_out_r,
            false,
            &mut metronome_out,
            false,
            0,
            false, // capturing
        );

        let expected_onset = (tempo_change_tick as f64
            * samples_per_tick_at(sample_rate as f64, 120.0)
            + (note_tick - tempo_change_tick) as f64
                * samples_per_tick_at(sample_rate as f64, 240.0))
        .round() as usize;
        let naive_flat_bpm_onset =
            (note_tick as f64 * samples_per_tick_at(sample_rate as f64, 120.0)).round() as usize;
        assert!(
            expected_onset < naive_flat_bpm_onset,
            "test sanity check: the tempo-aware onset should be earlier than the flat-120bpm one"
        );

        let onset = track_out_l[0]
            .iter()
            .position(|&s| s.abs() > 0.01)
            .expect("note should have triggered somewhere in the buffer");
        assert!(
            (onset as i64 - expected_onset as i64).abs() <= 2,
            "expected the note to trigger at sample ~{expected_onset} (tempo-map-aware), got {onset}"
        );
        assert!(
            (onset as i64 - naive_flat_bpm_onset as i64).abs() > 50,
            "onset {onset} should clearly differ from the flat-120bpm prediction {naive_flat_bpm_onset}"
        );
    }

    #[test]
    fn gap_between_regions_is_silent() {
        // Same 2-step region (a short note right at its start) placed at step 0..2, then a gap,
        // then again at step 6..8 — 8 steps total.
        let regions = vec![
            one_note_region(0, 2, 2, 0, TICKS_PER_STEP),
            one_note_region(6 * TICKS_PER_STEP, 2, 2, 0, TICKS_PER_STEP),
        ];
        let song = song_with_regions(regions);

        let sample_rate = 48_000u32;
        let path =
            std::env::temp_dir().join(format!("simple_daw_test_gap_{}.wav", std::process::id()));
        render_song_to_wav(&song, sample_rate, 1, &path).expect("export should succeed");
        let mut reader = hound::WavReader::open(&path).unwrap();
        let samples: Vec<i16> = left_channel(
            &reader
                .samples::<i16>()
                .collect::<std::result::Result<Vec<_>, _>>()
                .unwrap(),
        );
        std::fs::remove_file(&path).ok();

        // samples_per_tick = 250 at 120bpm/48kHz (see the formula in render_song_to_wav), so one
        // step = 6000 samples. region 0 spans samples [0, 12000); region 1 starts at step 6 = 36000.
        let early_region0 = &samples[0..3000];
        let deep_gap = &samples[20000..30000];
        let early_region1 = &samples[36000..39000];

        assert!(
            early_region0.iter().any(|&s| s != 0),
            "region 0's note should be audible right after it starts"
        );
        assert!(
            deep_gap.iter().all(|&s| s == 0),
            "the gap between regions should be silent"
        );
        assert!(
            early_region1.iter().any(|&s| s != 0),
            "region 1 should replay its own note"
        );
    }

    #[test]
    fn region_span_longer_than_its_content_loops_the_content() {
        // A region whose own content is 2 steps (a short note at its start), but whose
        // on-timeline span is 4 steps (2x its own length) — the content should repeat once
        // within that span, at step 2.
        let region = one_note_region(0, 2, 4, 0, TICKS_PER_STEP / 2);
        let song = song_with_regions(vec![region]);

        let sample_rate = 48_000u32;
        let path =
            std::env::temp_dir().join(format!("simple_daw_test_loop_{}.wav", std::process::id()));
        render_song_to_wav(&song, sample_rate, 1, &path).expect("export should succeed");
        let mut reader = hound::WavReader::open(&path).unwrap();
        let samples: Vec<i16> = left_channel(
            &reader
                .samples::<i16>()
                .collect::<std::result::Result<Vec<_>, _>>()
                .unwrap(),
        );
        std::fs::remove_file(&path).ok();

        // One step = 6000 samples; the region's own content is 2 steps = 12000 samples, so it
        // should retrigger at sample 12000 (the loop point inside the 4-step span).
        let first_trigger = &samples[0..2000];
        let between_triggers = &samples[6000..9000];
        let second_trigger = &samples[12000..14000];

        assert!(
            first_trigger.iter().any(|&s| s != 0),
            "the note should sound right at the region's start"
        );
        assert!(
            between_triggers.iter().all(|&s| s == 0),
            "the short note should have decayed to silence before the loop point"
        );
        assert!(
            second_trigger.iter().any(|&s| s != 0),
            "the content should retrigger when it loops within the span"
        );
    }

    #[test]
    fn regions_on_different_tracks_dont_bleed_into_each_other() {
        // Two tracks, each with its own region containing a note at tick 0 — since regions are
        // now inherently per-track (no shared pattern to scope into), each track's own note
        // should sound and nothing should leak onto the other track's output.
        let mut track_a = crate::model::Track::new_piano_roll("A", 1);
        track_a
            .regions
            .push(one_note_region(0, 2, 2, 0, TICKS_PER_STEP));
        let mut track_b = crate::model::Track::new_piano_roll("B", 2);
        track_b
            .regions
            .push(one_note_region(0, 2, 2, 0, TICKS_PER_STEP));
        let song = crate::model::Song {
            name: "track independence test".to_string(),
            bpm: 120.0,
            tempo_map: Vec::new(),
            tracks: vec![track_a, track_b],
            next_note_id: 0,
            master_effects: Vec::new(),
            plugins: Vec::new(),
            synth_presets: Vec::new(),
            time_signature_numerator: 4,
            time_signature_denominator: 4,
            sends: Vec::new(),
            submixes: Vec::new(),
            scenes: Vec::new(),
        };

        let mut sequencer = Sequencer::new(48_000.0);
        let mut track_out_l = Vec::new();
        let mut track_out_r = Vec::new();
        let mut metronome_out = Vec::new();
        // A few tick's worth of frames, enough for each track's note to trigger and decay somewhat.
        sequencer.process(
            &song,
            4096,
            &mut track_out_l,
            &mut track_out_r,
            false,
            &mut metronome_out,
            false,
            0,
            false, // capturing
        );

        assert!(
            track_out_l[0].iter().any(|&s| s != 0.0),
            "track A's own region should sound"
        );
        assert!(
            track_out_l[1].iter().any(|&s| s != 0.0),
            "track B's own region should sound"
        );
    }

    #[test]
    fn muted_submix_silences_every_member_track_at_the_synthesis_stage() {
        // Two tracks routed into the same submix bus; muting the bus should silence both, the
        // same way muting a track directly silences it — see `Sequencer::process`'s `track_silent`.
        let mut track_a = crate::model::Track::new_piano_roll("A", 1);
        track_a
            .regions
            .push(one_note_region(0, 2, 2, 0, TICKS_PER_STEP));
        track_a.output = crate::model::TrackOutput::Submix(0);
        let mut track_b = crate::model::Track::new_piano_roll("B", 2);
        track_b
            .regions
            .push(one_note_region(0, 2, 2, 0, TICKS_PER_STEP));
        track_b.output = crate::model::TrackOutput::Submix(0);
        let mut song = song_with_regions(Vec::new());
        song.tracks = vec![track_a, track_b];
        song.submixes = vec![crate::model::SubmixBus {
            name: "Bus".to_string(),
            effects: Vec::new(),
            volume: 1.0,
            pan: 0.0,
            muted: true,
            solo: false,
        }];

        let mut sequencer = Sequencer::new(48_000.0);
        let mut track_out_l = Vec::new();
        let mut track_out_r = Vec::new();
        let mut metronome_out = Vec::new();
        sequencer.process(
            &song, 4096, &mut track_out_l, &mut track_out_r, false, &mut metronome_out, false, 0,
            false, // capturing
        );

        assert!(
            track_out_l[0].iter().all(|&s| s == 0.0),
            "track A should be silent while its submix is muted"
        );
        assert!(
            track_out_l[1].iter().all(|&s| s == 0.0),
            "track B should be silent while its submix is muted"
        );
    }

    #[test]
    fn soloed_submix_silences_every_track_outside_it() {
        // Track A routes into a soloed submix, track B stays on Master — only A should sound,
        // the same "solo wins" rule a plain track solo already applies, extended to submix groups.
        let mut track_a = crate::model::Track::new_piano_roll("A", 1);
        track_a
            .regions
            .push(one_note_region(0, 2, 2, 0, TICKS_PER_STEP));
        track_a.output = crate::model::TrackOutput::Submix(0);
        let mut track_b = crate::model::Track::new_piano_roll("B", 2);
        track_b
            .regions
            .push(one_note_region(0, 2, 2, 0, TICKS_PER_STEP));
        let mut song = song_with_regions(Vec::new());
        song.tracks = vec![track_a, track_b];
        song.submixes = vec![crate::model::SubmixBus {
            name: "Bus".to_string(),
            effects: Vec::new(),
            volume: 1.0,
            pan: 0.0,
            muted: false,
            solo: true,
        }];

        let mut sequencer = Sequencer::new(48_000.0);
        let mut track_out_l = Vec::new();
        let mut track_out_r = Vec::new();
        let mut metronome_out = Vec::new();
        sequencer.process(
            &song, 4096, &mut track_out_l, &mut track_out_r, false, &mut metronome_out, false, 0,
            false, // capturing
        );

        assert!(
            track_out_l[0].iter().any(|&s| s != 0.0),
            "track A should sound: it's routed into the soloed submix"
        );
        assert!(
            track_out_l[1].iter().all(|&s| s == 0.0),
            "track B should be silent: solo is active and it's outside the soloed submix"
        );
    }

    #[test]
    fn hard_panned_submix_silences_the_opposite_channel_in_the_offline_bounce() {
        // A track routed into a submix panned hard left should produce audio only on the left
        // channel of the bounce — verifying `SubmixBus::pan` is applied in `mix_song_to_wav_buffer`
        // the same way `Track::pan` already is.
        let mut track = crate::model::Track::new_piano_roll("A", 1);
        track
            .regions
            .push(one_note_region(0, 2, 2, 0, TICKS_PER_STEP));
        track.output = crate::model::TrackOutput::Submix(0);
        let mut song = song_with_regions(Vec::new());
        song.tracks = vec![track];
        song.submixes = vec![crate::model::SubmixBus {
            name: "Bus".to_string(),
            effects: Vec::new(),
            volume: 1.0,
            pan: -1.0,
            muted: false,
            solo: false,
        }];

        let sample_rate = 48_000u32;
        let path = std::env::temp_dir().join(format!(
            "simple_daw_test_submix_pan_{}.wav",
            std::process::id()
        ));
        render_song_to_wav(&song, sample_rate, 1, &path).expect("export should succeed");
        let mut reader = hound::WavReader::open(&path).unwrap();
        let samples: Vec<i16> = reader
            .samples::<i16>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        std::fs::remove_file(&path).ok();

        assert!(
            left_channel(&samples).iter().any(|&s| s != 0),
            "left channel should be audible: the submix is panned hard left"
        );
        assert!(
            right_channel(&samples).iter().all(|&s| s == 0),
            "right channel should be silent: the submix is panned hard left"
        );
    }

    #[test]
    fn audio_clip_plays_back_at_its_start_tick_and_extends_the_loop_length() {
        let sample_rate = 48_000u32;
        // 0.1s of a constant tone, well above the "inaudible" floor used elsewhere in this file.
        let clip_samples: Vec<f32> = vec![0.5; (sample_rate as usize) / 10];
        let buffer = Arc::new(SampleBuffer {
            sample_rate,
            mono: clip_samples,
        });

        let mut track = crate::model::Track::new_audio("Vocals", 1);
        let mut clip = crate::model::AudioClip::new(0, "unused.wav");
        clip.buffer = Some(buffer);
        track.audio_clips.push(clip);

        let song = crate::model::Song {
            name: "audio clip test".to_string(),
            bpm: 120.0,
            tempo_map: Vec::new(),
            tracks: vec![track],
            next_note_id: 0,
            master_effects: Vec::new(),
            plugins: Vec::new(),
            synth_presets: Vec::new(),
            time_signature_numerator: 4,
            time_signature_denominator: 4,
            sends: Vec::new(),
            submixes: Vec::new(),
            scenes: Vec::new(),
        };

        // The clip alone (no regions at all) should still determine a nonzero loop length —
        // otherwise `render_song_to_wav` below would bounce zero samples.
        assert!(
            arrangement_length_ticks(&song) > 0,
            "an audio clip with no regions should still produce a nonzero loop length"
        );

        let path = std::env::temp_dir().join(format!(
            "simple_daw_test_audio_clip_{}.wav",
            std::process::id()
        ));
        render_song_to_wav(&song, sample_rate, 1, &path).expect("export should succeed");
        let mut reader = hound::WavReader::open(&path).unwrap();
        let samples: Vec<i16> = left_channel(
            &reader
                .samples::<i16>()
                .collect::<std::result::Result<Vec<_>, _>>()
                .unwrap(),
        );
        std::fs::remove_file(&path).ok();

        assert!(
            samples[0..1000].iter().any(|&s| s != 0),
            "the clip should be audible right at its start tick"
        );
    }

    #[test]
    fn session_mode_triggers_a_launched_step_grid_slot_and_ignores_the_playlist() {
        let mut track = crate::model::Track::new_step_grid("Drums", 1);
        // A Playlist region that would trigger a note at tick 0 in Arrangement mode — proves
        // Session View's mode switch actually suppresses it (see `Transport::session_mode`'s doc
        // comment: never both at once).
        track.regions.push(one_note_region(0, 4, 4, 0, TICKS_PER_STEP));
        let mut lane = crate::model::Lane::new("Kick", 60, 4);
        lane.set_step(0, 127);
        track.session_clips.push(Some(crate::model::SessionClip {
            name: "Slot".to_string(),
            content: crate::model::SessionClipContent::Region {
                content: RegionContent::StepGrid(vec![lane]),
                content_length_steps: 4,
                loop_length_steps: 4,
            },
            follow_action: crate::model::FollowActionConfig::default(),
            legato: false,
            launch_mode: LaunchMode::Toggle,
            quantize_override: None,
            automation: Vec::new(),
        }));
        track.session_launch_requests.push(crate::model::SessionLaunchRequest {
            generation: 1,
            intent: crate::model::LaunchIntent::Play,
            ..Default::default()
        });

        let song = crate::model::Song {
            name: "session view test".to_string(),
            bpm: 120.0,
            tempo_map: Vec::new(),
            tracks: vec![track],
            next_note_id: 0,
            master_effects: Vec::new(),
            plugins: Vec::new(),
            synth_presets: Vec::new(),
            time_signature_numerator: 4,
            time_signature_denominator: 4,
            sends: Vec::new(),
            submixes: Vec::new(),
            scenes: Vec::new(),
        };

        let mut sequencer = Sequencer::new(48_000.0);
        let mut track_out_l = Vec::new();
        let mut track_out_r = Vec::new();
        let mut metronome_out = Vec::new();
        sequencer.process(
            &song,
            4096,
            &mut track_out_l,
            &mut track_out_r,
            false,
            &mut metronome_out,
            true, // session_mode
            0,    // no quantization: launches immediately
            false, // capturing
        );

        assert!(
            track_out_l[0].iter().any(|&s| s != 0.0),
            "the launched step-grid session slot should be audible"
        );
    }

    #[test]
    fn session_mode_quantize_override_launches_immediately_despite_a_huge_grid_quantize() {
        let mut track = crate::model::Track::new_step_grid("Drums", 1);
        let mut lane = crate::model::Lane::new("Kick", 60, 4);
        lane.set_step(0, 127);
        track.session_clips.push(Some(crate::model::SessionClip {
            name: "Slot".to_string(),
            content: crate::model::SessionClipContent::Region {
                content: RegionContent::StepGrid(vec![lane]),
                content_length_steps: 4,
                loop_length_steps: 4,
            },
            follow_action: crate::model::FollowActionConfig::default(),
            legato: false,
            launch_mode: LaunchMode::Toggle,
            // Overrides the grid-wide quantize below with "None" (immediate) — without this,
            // the launch would never resolve within this test's short processed window.
            quantize_override: Some(crate::model::SessionQuantize::None),
            automation: Vec::new(),
        }));
        track.session_launch_requests.push(crate::model::SessionLaunchRequest {
            generation: 1,
            intent: crate::model::LaunchIntent::Play,
            ..Default::default()
        });

        let song = crate::model::Song {
            name: "session view quantize override test".to_string(),
            bpm: 120.0,
            tempo_map: Vec::new(),
            tracks: vec![track],
            next_note_id: 0,
            master_effects: Vec::new(),
            plugins: Vec::new(),
            synth_presets: Vec::new(),
            time_signature_numerator: 4,
            time_signature_denominator: 4,
            sends: Vec::new(),
            submixes: Vec::new(),
            scenes: Vec::new(),
        };

        let mut sequencer = Sequencer::new(48_000.0);
        let mut track_out_l = Vec::new();
        let mut track_out_r = Vec::new();
        let mut metronome_out = Vec::new();
        sequencer.process(
            &song,
            4096,
            &mut track_out_l,
            &mut track_out_r,
            false,
            &mut metronome_out,
            true,      // session_mode
            1_000_000, // grid-wide quantize: far beyond this test's window if not overridden
            false, // capturing
        );

        assert!(
            matches!(sequencer.session_slots[0][0], SlotState::Playing { .. }),
            "the clip's own quantize_override should have launched it immediately, got {:?}",
            sequencer.session_slots[0][0]
        );
        assert!(
            track_out_l[0].iter().any(|&s| s != 0.0),
            "the launched step-grid session slot should be audible"
        );
    }

    #[test]
    fn session_mode_looping_audio_clip_stops_hard_when_the_slot_is_stopped() {
        let sample_rate = 48_000u32;
        // A short clip so it wraps (loops) several times within one process() call.
        let buffer = Arc::new(SampleBuffer { sample_rate, mono: vec![0.5; 480] });
        let mut audio_clip = crate::model::AudioClip::new(0, "unused.wav");
        audio_clip.buffer = Some(buffer);

        let mut track = crate::model::Track::new_audio("Loop", 1);
        track.session_clips.push(Some(crate::model::SessionClip {
            name: "Loop Slot".to_string(),
            content: crate::model::SessionClipContent::Audio(audio_clip),
            follow_action: crate::model::FollowActionConfig::default(),
            legato: false,
            launch_mode: LaunchMode::Toggle,
            quantize_override: None,
            automation: Vec::new(),
        }));
        track.session_launch_requests.push(crate::model::SessionLaunchRequest {
            generation: 1,
            intent: crate::model::LaunchIntent::Play,
            ..Default::default()
        });

        let song = crate::model::Song {
            name: "session view loop test".to_string(),
            bpm: 120.0,
            tempo_map: Vec::new(),
            tracks: vec![track],
            next_note_id: 0,
            master_effects: Vec::new(),
            plugins: Vec::new(),
            synth_presets: Vec::new(),
            time_signature_numerator: 4,
            time_signature_denominator: 4,
            sends: Vec::new(),
            submixes: Vec::new(),
            scenes: Vec::new(),
        };

        let mut sequencer = Sequencer::new(sample_rate as f32);
        let mut track_out_l = Vec::new();
        let mut track_out_r = Vec::new();
        let mut metronome_out = Vec::new();
        // 4000 frames is well past the 480-frame clip's own length, so this only passes if the
        // clip actually looped (see `SampleVoice::looping`) instead of going silent after once.
        sequencer.process(
            &song, 4000, &mut track_out_l, &mut track_out_r, false, &mut metronome_out, true, 0,
            false, // capturing
        );
        assert!(
            track_out_l[0][3000..4000].iter().any(|&s| s != 0.0),
            "the clip should still be looping well past its own natural length"
        );

        let mut song_after_stop = song.clone();
        song_after_stop.tracks[0].session_launch_requests[0].generation = 2;
        song_after_stop.tracks[0].session_launch_requests[0].intent = crate::model::LaunchIntent::Stop;
        sequencer.process(
            &song_after_stop,
            4000,
            &mut track_out_l,
            &mut track_out_r,
            false,
            &mut metronome_out,
            true,
            0,
            false, // capturing
        );
        assert!(
            track_out_l[0][3000..4000].iter().all(|&s| s == 0.0),
            "stopping the slot should hard-cut the looping voice, not let it keep looping"
        );
    }

    #[test]
    fn session_mode_recording_clip_plays_its_active_take_and_loops() {
        let sample_rate = 48_000u32;
        // A short take so it wraps (loops) several times within one process() call, same shape as
        // `session_mode_looping_audio_clip_stops_hard_when_the_slot_is_stopped` above but for
        // `SessionClipContent::Recording` instead of `Audio`.
        let buffer = Arc::new(SampleBuffer { sample_rate, mono: vec![0.5; 480] });
        let mut folder = crate::model::TakeFolder::new(0, 480, "unused.wav");
        folder.takes[0].buffer = Some(buffer);

        let mut track = crate::model::Track::new_audio("Loop", 1);
        track.session_clips.push(Some(crate::model::SessionClip {
            name: "Recorded Loop".to_string(),
            content: crate::model::SessionClipContent::Recording(folder),
            follow_action: crate::model::FollowActionConfig::default(),
            legato: false,
            launch_mode: LaunchMode::Toggle,
            quantize_override: None,
            automation: Vec::new(),
        }));
        track.session_launch_requests.push(crate::model::SessionLaunchRequest {
            generation: 1,
            intent: crate::model::LaunchIntent::Play,
            ..Default::default()
        });

        let song = crate::model::Song {
            name: "session view recording loop test".to_string(),
            bpm: 120.0,
            tempo_map: Vec::new(),
            tracks: vec![track],
            next_note_id: 0,
            master_effects: Vec::new(),
            plugins: Vec::new(),
            synth_presets: Vec::new(),
            time_signature_numerator: 4,
            time_signature_denominator: 4,
            sends: Vec::new(),
            submixes: Vec::new(),
            scenes: Vec::new(),
        };

        let mut sequencer = Sequencer::new(sample_rate as f32);
        let mut track_out_l = Vec::new();
        let mut track_out_r = Vec::new();
        let mut metronome_out = Vec::new();
        // 4000 frames is well past the 480-frame take's own length, so this only passes if the
        // recording actually looped instead of going silent after playing once.
        sequencer.process(
            &song, 4000, &mut track_out_l, &mut track_out_r, false, &mut metronome_out, true, 0,
            false, // capturing
        );
        assert!(
            track_out_l[0][3000..4000].iter().any(|&s| s != 0.0),
            "the recorded take should still be looping well past its own natural length"
        );
    }

    fn step_grid_session_track(name: &str) -> crate::model::Track {
        crate::model::Track::new_step_grid(name, 1)
    }

    fn one_step_lane(pitch: u8) -> crate::model::Lane {
        let mut lane = crate::model::Lane::new("Kick", pitch, 1);
        lane.set_step(0, 127);
        lane
    }

    fn session_song(track: crate::model::Track) -> crate::model::Song {
        crate::model::Song {
            name: "session follow-action test".to_string(),
            bpm: 120.0,
            tempo_map: Vec::new(),
            tracks: vec![track],
            next_note_id: 0,
            master_effects: Vec::new(),
            plugins: Vec::new(),
            synth_presets: Vec::new(),
            time_signature_numerator: 4,
            time_signature_denominator: 4,
            sends: Vec::new(),
            submixes: Vec::new(),
            scenes: Vec::new(),
        }
    }

    #[test]
    fn session_mode_follow_action_next_advances_to_the_next_slot_and_stops_the_source() {
        let mut track = step_grid_session_track("Drums");
        track.session_clips.push(Some(crate::model::SessionClip {
            name: "Slot 0".to_string(),
            content: crate::model::SessionClipContent::Region {
                content: RegionContent::StepGrid(vec![one_step_lane(60)]),
                content_length_steps: 1,
                loop_length_steps: 1,
            },
            follow_action: crate::model::FollowActionConfig {
                times: 1,
                action_a: FollowAction::Next,
                chance_a: 1.0,
                action_b: FollowAction::None,
                chance_b: 0.0,
            },
            legato: false,
            launch_mode: LaunchMode::Toggle,
            quantize_override: None,
            automation: Vec::new(),
        }));
        track.session_clips.push(Some(crate::model::SessionClip {
            name: "Slot 1".to_string(),
            content: crate::model::SessionClipContent::Region {
                content: RegionContent::StepGrid(vec![one_step_lane(62)]),
                content_length_steps: 4,
                loop_length_steps: 4,
            },
            follow_action: crate::model::FollowActionConfig::default(),
            legato: false,
            launch_mode: LaunchMode::Toggle,
            quantize_override: None,
            automation: Vec::new(),
        }));
        track.session_launch_requests.push(crate::model::SessionLaunchRequest {
            generation: 1,
            intent: crate::model::LaunchIntent::Play,
            ..Default::default()
        });

        let song = session_song(track);
        let mut sequencer = Sequencer::new(48_000.0);
        let mut track_out_l = Vec::new();
        let mut track_out_r = Vec::new();
        let mut metronome_out = Vec::new();
        // Slot 0's whole loop is 1 step (TICKS_PER_STEP ticks) — comfortably less than one
        // second of audio at any reasonable tempo, so this covers several loops.
        sequencer.process(
            &song, 48_000, &mut track_out_l, &mut track_out_r, false, &mut metronome_out, true, 0,
            false, // capturing
        );

        assert_eq!(
            sequencer.session_slots[0][0],
            SlotState::Stopped,
            "the source slot should have stopped once its follow action fired"
        );
        assert!(
            matches!(sequencer.session_slots[0][1], SlotState::Playing { .. }),
            "Next should have launched slot 1, got {:?}",
            sequencer.session_slots[0][1]
        );
    }

    #[test]
    fn session_mode_legato_launch_continues_the_outgoing_slots_phase() {
        let mut track = step_grid_session_track("Drums");
        for pitch in [60, 62] {
            track.session_clips.push(Some(crate::model::SessionClip {
                name: format!("Slot {pitch}"),
                content: crate::model::SessionClipContent::Region {
                    content: RegionContent::StepGrid(vec![one_step_lane(pitch)]),
                    content_length_steps: 16,
                    loop_length_steps: 16,
                },
                follow_action: crate::model::FollowActionConfig::default(),
                legato: true,
                launch_mode: LaunchMode::Toggle,
                quantize_override: None,
                automation: Vec::new(),
            }));
        }
        track.session_launch_requests.push(crate::model::SessionLaunchRequest {
            generation: 1,
            intent: crate::model::LaunchIntent::Play,
            ..Default::default()
        });

        let song = session_song(track);
        let mut sequencer = Sequencer::new(48_000.0);
        let mut track_out_l = Vec::new();
        let mut track_out_r = Vec::new();
        let mut metronome_out = Vec::new();
        // A handful of ticks' worth of samples — enough for slot 0 to be clearly mid-loop, well
        // short of its own 16-step length.
        sequencer.process(
            &song, 4000, &mut track_out_l, &mut track_out_r, false, &mut metronome_out, true, 0,
            false, // capturing
        );
        let SlotState::Playing { local_tick: slot_0_local_tick, .. } = sequencer.session_slots[0][0] else {
            panic!("slot 0 should be playing by now: {:?}", sequencer.session_slots[0][0]);
        };
        assert!(slot_0_local_tick > 0, "slot 0 should be mid-loop, not just starting");

        let mut song_launch_slot_1 = song.clone();
        song_launch_slot_1.tracks[0].session_launch_requests.push(crate::model::SessionLaunchRequest {
            generation: 1,
            intent: crate::model::LaunchIntent::Play,
            ..Default::default()
        });
        sequencer.process(
            &song_launch_slot_1,
            1,
            &mut track_out_l,
            &mut track_out_r,
            false,
            &mut metronome_out,
            true,
            0,
            false, // capturing
        );

        assert_eq!(
            sequencer.session_slots[0][0],
            SlotState::Stopped,
            "legato still stops whatever else was playing on the same track"
        );
        let SlotState::Playing { local_tick: slot_1_local_tick, .. } = sequencer.session_slots[0][1] else {
            panic!("slot 1 should be playing: {:?}", sequencer.session_slots[0][1]);
        };
        assert!(
            slot_1_local_tick > 0,
            "legato should have continued slot 0's phase instead of restarting at 0, got {slot_1_local_tick}"
        );
    }

    fn session_launch(track: &mut crate::model::Track, pitch: u8, generation: u64) {
        track.session_clips.push(Some(crate::model::SessionClip {
            name: "Loop".to_string(),
            content: crate::model::SessionClipContent::Region {
                content: RegionContent::StepGrid(vec![one_step_lane(pitch)]),
                content_length_steps: 1,
                loop_length_steps: 1,
            },
            follow_action: crate::model::FollowActionConfig::default(),
            legato: false,
            launch_mode: LaunchMode::Toggle,
            quantize_override: None,
            automation: Vec::new(),
        }));
        track.session_launch_requests.push(crate::model::SessionLaunchRequest {
            generation,
            intent: crate::model::LaunchIntent::Play,
            ..Default::default()
        });
    }

    #[test]
    fn capturing_logs_a_started_event_on_launch_and_a_stopped_event_on_stop() {
        let mut track = step_grid_session_track("Drums");
        session_launch(&mut track, 36, 1);
        let song = session_song(track);

        let mut sequencer = Sequencer::new(48_000.0);
        let mut track_out_l = Vec::new();
        let mut track_out_r = Vec::new();
        let mut metronome_out = Vec::new();
        sequencer.process(
            &song, 4000, &mut track_out_l, &mut track_out_r, false, &mut metronome_out, true, 0,
            true, // capturing
        );

        assert_eq!(sequencer.capture_log.len(), 1);
        assert!(matches!(
            sequencer.capture_log[0].kind,
            crate::model::CaptureEventKind::Started { .. }
        ));

        let mut song_after_stop = song.clone();
        song_after_stop.tracks[0].session_launch_requests[0].generation = 2;
        song_after_stop.tracks[0].session_launch_requests[0].intent = crate::model::LaunchIntent::Stop;
        sequencer.process(
            &song_after_stop, 4000, &mut track_out_l, &mut track_out_r, false, &mut metronome_out, true, 0,
            true, // capturing
        );

        assert_eq!(sequencer.capture_log.len(), 2);
        assert!(matches!(sequencer.capture_log[1].kind, crate::model::CaptureEventKind::Stopped));
    }

    #[test]
    fn capturing_logs_nothing_when_the_flag_is_off() {
        let mut track = step_grid_session_track("Drums");
        session_launch(&mut track, 36, 1);
        let song = session_song(track);

        let mut sequencer = Sequencer::new(48_000.0);
        let mut track_out_l = Vec::new();
        let mut track_out_r = Vec::new();
        let mut metronome_out = Vec::new();
        sequencer.process(
            &song, 4000, &mut track_out_l, &mut track_out_r, false, &mut metronome_out, true, 0,
            false, // capturing
        );

        assert!(sequencer.capture_log.is_empty());
    }

    #[test]
    fn capturing_clears_the_log_on_the_off_to_on_edge() {
        let mut track = step_grid_session_track("Drums");
        session_launch(&mut track, 36, 1);
        let song = session_song(track);

        let mut sequencer = Sequencer::new(48_000.0);
        let mut track_out_l = Vec::new();
        let mut track_out_r = Vec::new();
        let mut metronome_out = Vec::new();
        // First arm: logs the launch, and advances `capture_tick` by however many ticks one
        // 4000-frame call fires.
        sequencer.process(
            &song, 4000, &mut track_out_l, &mut track_out_r, false, &mut metronome_out, true, 0,
            true, // capturing
        );
        assert_eq!(sequencer.capture_log.len(), 1);
        let capture_tick_after_first_window = sequencer.capture_tick;

        // Turned off, then back on (a fresh arm) — the old log shouldn't leak into the new one,
        // and `capture_tick` should restart near 0 rather than keep accumulating from before.
        // Large enough frame counts that a tick boundary (and so `trigger_session_clips`, which is
        // what actually detects the edge) definitely fires within each call.
        sequencer.process(
            &song, 4000, &mut track_out_l, &mut track_out_r, false, &mut metronome_out, true, 0,
            false, // capturing
        );
        sequencer.process(
            &song, 4000, &mut track_out_l, &mut track_out_r, false, &mut metronome_out, true, 0,
            true, // capturing
        );
        assert!(
            sequencer.capture_log.is_empty(),
            "re-arming capture should start from an empty log, not carry over the previous window's events"
        );
        assert!(
            sequencer.capture_tick <= capture_tick_after_first_window,
            "re-arming should reset capture_tick near 0, not keep accumulating across the gap: \
             got {} after re-arming vs {} after the first window alone",
            sequencer.capture_tick,
            capture_tick_after_first_window
        );
    }

    fn trine(overrides: impl FnOnce(&mut TrineParams)) -> TrineParams {
        let mut trine = TrineParams::default();
        overrides(&mut trine);
        trine
    }

    #[test]
    fn trine_voice_is_audible_then_decays_to_silence() {
        let sample_rate = 48_000.0;
        let mut voice = TrineVoice::default();
        let trine = trine(|p| p.env3_decay_seconds = 0.25);
        voice.trigger(
            pitch_to_freq(60),
            100,
            sample_rate,
            trine.env3_decay_seconds,
            &trine,
        );

        let mut early_peak = 0.0f32;
        for _ in 0..200 {
            early_peak = early_peak.max(voice.next_sample(&[]).0.abs());
        }
        assert!(
            early_peak > 0.05,
            "voice should be clearly audible right after trigger"
        );

        for _ in 0..(sample_rate as usize) {
            voice.next_sample(&[]).0;
        }
        assert!(
            !voice.active,
            "voice should have decayed to silence within 1 second"
        );
        assert_eq!(
            voice.next_sample(&[]).0,
            0.0,
            "an inactive voice must output silence"
        );
    }

    #[test]
    fn trine_analog_drift_zero_keeps_left_and_right_identical() {
        let sample_rate = 48_000.0;
        let mut voice = TrineVoice::default();
        let trine = trine(|p| {
            p.analog_drift = 0.0;
            p.env3_decay_seconds = 1.0;
        });
        voice.trigger(pitch_to_freq(60), 100, sample_rate, 1.0, &trine);
        for _ in 0..500 {
            let (l, r) = voice.next_sample(&[]);
            assert_eq!(l, r, "analog_drift 0.0 should keep every channel identical");
        }
    }

    #[test]
    fn trine_analog_drift_above_zero_spreads_left_and_right_apart() {
        let sample_rate = 48_000.0;
        let mut voice = TrineVoice::default();
        let trine = trine(|p| {
            p.analog_drift = 1.0;
            p.env3_decay_seconds = 1.0;
        });
        voice.trigger(pitch_to_freq(60), 100, sample_rate, 1.0, &trine);
        let differs = (0..2000).any(|_| {
            let (l, r) = voice.next_sample(&[]);
            (l - r).abs() > 1e-4
        });
        assert!(
            differs,
            "analog_drift 1.0 should eventually decorrelate left and right"
        );
    }

    #[test]
    fn trine_default_track_is_audible_with_no_mod_slots() {
        // A freshly-selected Trine track (empty mod_slots) must still play — env3 is always-on.
        let sample_rate = 48_000.0;
        let mut voice = TrineVoice::default();
        let trine = TrineParams::default();
        voice.trigger(pitch_to_freq(60), 100, sample_rate, 1.0, &trine);

        let mut peak = 0.0f32;
        for _ in 0..1000 {
            peak = peak.max(voice.next_sample(&[]).0.abs());
        }
        assert!(
            peak > 0.05,
            "a fresh Trine voice with no matrix routing should still be audible"
        );
    }

    #[test]
    fn trine_all_filter_routings_and_slopes_produce_bounded_signal_without_panicking() {
        let sample_rate = 48_000.0;
        for routing in [
            FilterRouting::Off,
            FilterRouting::Series,
            FilterRouting::Parallel,
        ] {
            for slope in [FilterSlope::Slope12, FilterSlope::Slope24] {
                let mut voice = TrineVoice::default();
                let trine = trine(|p| {
                    p.filter_routing = routing;
                    p.filter1_slope = slope;
                    p.filter2_slope = slope;
                    p.osc2_level = 0.5;
                    p.osc3_level = 0.5;
                    p.filter_drive = 0.5;
                });
                voice.trigger(pitch_to_freq(60), 127, sample_rate, 1.0, &trine);
                for _ in 0..sample_rate as usize {
                    let s = voice.next_sample(&[]).0;
                    assert!(
                        s.is_finite() && s.abs() < 10.0,
                        "routing {routing:?} slope {slope:?} produced an unbounded/non-finite sample: {s}"
                    );
                }
            }
        }
    }

    #[test]
    fn trine_noise_oscillator_produces_bounded_signal() {
        let sample_rate = 48_000.0;
        let mut voice = TrineVoice::default();
        let trine = trine(|p| p.osc1_waveform = SynthWaveform::Noise);
        voice.trigger(pitch_to_freq(60), 127, sample_rate, 1.0, &trine);
        for _ in 0..sample_rate as usize {
            let s = voice.next_sample(&[]).0;
            assert!(
                s.is_finite() && s.abs() <= 1.5,
                "noise oscillator sample out of range: {s}"
            );
        }
    }

    #[test]
    fn trine_matrix_slot_routing_env2_to_filter_cutoff_changes_output_from_unrouted() {
        let sample_rate = 48_000.0;
        let trine = trine(|p| {
            p.osc1_waveform = SynthWaveform::Saw;
            p.filter1_cutoff_hz = 500.0;
            p.filter1_resonance = 4.0;
            p.env2_decay_seconds = 0.05;
        });

        let render = |mod_slots: &[ModSlot]| -> Vec<f32> {
            let mut voice = TrineVoice::default();
            voice.trigger(pitch_to_freq(48), 127, sample_rate, 1.0, &trine);
            (0..2000).map(|_| voice.next_sample(mod_slots).0).collect()
        };

        let unrouted = render(&[]);
        let routed = render(&[ModSlot {
            source: ModSource::Env2,
            target: ModTarget::FilterCutoff,
            amount: 1.0,
        }]);

        let differs = unrouted
            .iter()
            .zip(&routed)
            .any(|(a, b)| (a - b).abs() > 1e-4);
        assert!(
            differs,
            "routing Env2 -> FilterCutoff should audibly change the output vs no routing"
        );
    }

    #[test]
    fn trine_track_selecting_simple_engine_leaves_simple_synth_untouched() {
        // Switching a track to `SynthEngine::Trine` and back must not perturb `SynthParams`/`Voice`
        // in any way — the two engines are fully independent.
        let mut song = crate::model::Song::demo();
        let synth_before = song.tracks[1].synth;
        song.tracks[1].synth_engine = crate::model::SynthEngine::Trine;
        song.tracks[1].synth_engine = crate::model::SynthEngine::Simple;
        assert_eq!(song.tracks[1].synth, synth_before);
    }

    fn wave(overrides: impl FnOnce(&mut WaveParams)) -> WaveParams {
        let mut wave = WaveParams::default();
        overrides(&mut wave);
        wave
    }

    #[test]
    fn wave_voice_is_audible_then_decays_to_silence() {
        let sample_rate = 48_000.0;
        let mut voice = WaveVoice::default();
        let wave = wave(|p| p.amp_decay_seconds = 0.25);
        voice.trigger(
            pitch_to_freq(60),
            100,
            sample_rate,
            wave.amp_decay_seconds,
            &wave,
        );

        let mut early_peak = 0.0f32;
        for _ in 0..200 {
            early_peak = early_peak.max(voice.next_sample(&[]).0.abs());
        }
        assert!(
            early_peak > 0.05,
            "voice should be clearly audible right after trigger"
        );

        for _ in 0..(sample_rate as usize) {
            voice.next_sample(&[]).0;
        }
        assert!(
            !voice.active,
            "voice should have decayed to silence within 1 second"
        );
        assert_eq!(
            voice.next_sample(&[]).0,
            0.0,
            "an inactive voice must output silence"
        );
    }

    #[test]
    fn wave_unison_width_zero_keeps_left_and_right_identical() {
        let sample_rate = 48_000.0;
        let mut voice = WaveVoice::default();
        let wave = wave(|p| {
            p.unison_voices = 3;
            p.unison_detune_cents = 15.0;
            p.unison_width = 0.0;
        });
        voice.trigger(pitch_to_freq(60), 100, sample_rate, 1.0, &wave);
        for _ in 0..200 {
            let (l, r) = voice.next_sample(&[]);
            assert_eq!(l, r, "unison_width 0.0 should keep every channel identical");
        }
    }

    #[test]
    fn wave_unison_width_above_zero_spreads_left_and_right_apart() {
        let sample_rate = 48_000.0;
        let mut voice = WaveVoice::default();
        let wave = wave(|p| {
            p.unison_voices = 3;
            p.unison_detune_cents = 15.0;
            p.unison_width = 1.0;
        });
        voice.trigger(pitch_to_freq(60), 100, sample_rate, 1.0, &wave);
        let differs = (0..200).any(|_| {
            let (l, r) = voice.next_sample(&[]);
            (l - r).abs() > 1e-4
        });
        assert!(
            differs,
            "unison_width 1.0 with 3 unison voices should produce a genuinely different left and right signal"
        );
    }

    #[test]
    fn wave_default_track_is_audible_with_no_mod_slots() {
        // A freshly-selected Wave track (empty mod_slots) must still play — the amp envelope is
        // always-on.
        let sample_rate = 48_000.0;
        let mut voice = WaveVoice::default();
        let wave = WaveParams::default();
        voice.trigger(pitch_to_freq(60), 100, sample_rate, 1.0, &wave);

        let mut peak = 0.0f32;
        for _ in 0..1000 {
            peak = peak.max(voice.next_sample(&[]).0.abs());
        }
        assert!(
            peak > 0.05,
            "a fresh Wave voice with no matrix routing should still be audible"
        );
    }

    #[test]
    fn wave_all_filter_routings_and_slopes_produce_bounded_signal_without_panicking() {
        let sample_rate = 48_000.0;
        for routing in [
            FilterRouting::Off,
            FilterRouting::Series,
            FilterRouting::Parallel,
        ] {
            for slope in [FilterSlope::Slope12, FilterSlope::Slope24] {
                let mut voice = WaveVoice::default();
                let wave = wave(|p| {
                    p.filter_routing = routing;
                    p.filter1_slope = slope;
                    p.filter2_slope = slope;
                    p.osc2_level = 0.5;
                    p.sub_osc_level = 0.5;
                    p.noise_level = 0.3;
                    p.filter_drive = 0.5;
                });
                voice.trigger(pitch_to_freq(60), 127, sample_rate, 1.0, &wave);
                for _ in 0..sample_rate as usize {
                    let s = voice.next_sample(&[]).0;
                    assert!(
                        s.is_finite() && s.abs() < 10.0,
                        "routing {routing:?} slope {slope:?} produced an unbounded/non-finite sample: {s}"
                    );
                }
            }
        }
    }

    #[test]
    fn wave_all_tables_and_warp_modes_produce_bounded_signal_without_panicking() {
        let sample_rate = 48_000.0;
        for table in WavetableId::ALL {
            for warp_mode in [
                WaveWarpMode::Off,
                WaveWarpMode::Bend,
                WaveWarpMode::Sync,
                WaveWarpMode::Mirror,
                WaveWarpMode::Fm,
            ] {
                let mut voice = WaveVoice::default();
                let wave = wave(|p| {
                    p.osc1_table = table;
                    p.osc2_table = table;
                    p.osc1_warp_mode = warp_mode;
                    p.osc1_warp_amount = 0.7;
                    p.osc2_warp_mode = warp_mode;
                    p.osc2_warp_amount = 0.7;
                    p.osc2_level = 0.5;
                });
                voice.trigger(pitch_to_freq(72), 127, sample_rate, 1.0, &wave);
                for _ in 0..1000 {
                    let s = voice.next_sample(&[]).0;
                    assert!(
                        s.is_finite() && s.abs() < 10.0,
                        "table {table:?} warp {warp_mode:?} produced an unbounded/non-finite sample: {s}"
                    );
                }
            }
        }
    }

    #[test]
    fn wave_matrix_slot_routing_env2_to_filter_cutoff_changes_output_from_unrouted() {
        let sample_rate = 48_000.0;
        let wave = wave(|p| {
            p.filter1_cutoff_hz = 500.0;
            p.filter1_resonance = 4.0;
            p.env2_decay_seconds = 0.05;
        });

        let render = |mod_slots: &[WaveModSlot]| -> Vec<f32> {
            let mut voice = WaveVoice::default();
            voice.trigger(pitch_to_freq(48), 127, sample_rate, 1.0, &wave);
            (0..2000).map(|_| voice.next_sample(mod_slots).0).collect()
        };

        let unrouted = render(&[]);
        let routed = render(&[WaveModSlot {
            source: WaveModSource::Env2,
            target: WaveModTarget::FilterCutoff,
            amount: 1.0,
        }]);

        let differs = unrouted
            .iter()
            .zip(&routed)
            .any(|(a, b)| (a - b).abs() > 1e-4);
        assert!(
            differs,
            "routing Env2 -> FilterCutoff should audibly change the output vs no routing"
        );
    }

    #[test]
    fn wave_track_selecting_simple_engine_leaves_simple_synth_untouched() {
        // Switching a track to `SynthEngine::Wave` and back must not perturb `SynthParams`/`Voice`
        // in any way — the engines are fully independent.
        let mut song = crate::model::Song::demo();
        let synth_before = song.tracks[1].synth;
        song.tracks[1].synth_engine = crate::model::SynthEngine::Wave;
        song.tracks[1].synth_engine = crate::model::SynthEngine::Simple;
        assert_eq!(song.tracks[1].synth, synth_before);
    }
}
