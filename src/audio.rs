use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{BufferSize, FromSample, SampleFormat, SizedSample, Stream, StreamConfig};

use crate::model::{
    FilterRouting, FilterSlope, FilterType, LfoTarget, ModSlot, ModSource, ModTarget,
    RegionContent, Song, SynthEngine, SynthParams, SynthWaveform, TICKS_PER_STEP, Track,
    TrackKind, TrineParams, WaveModSlot, WaveModSource, WaveModTarget, WaveParams,
};
use crate::plugin_host::{self, MasterEffectSlot, TrackEffectSlots};
use crate::sample::SampleBuffer;
use crate::wavetable::{self, WaveWarpMode, WavetableId};

/// 16th-note grid: 4 steps per beat.
const STEPS_PER_BEAT: f64 = 4.0;
const VOICE_COUNT: usize = 32;
const SAMPLE_VOICE_COUNT: usize = 32;
const MASTER_GAIN: f32 = 0.3;
/// Level (relative to a voice's starting amplitude) considered inaudible; below this a voice is freed.
const ENVELOPE_FLOOR: f32 = 0.0005;
/// A piano-roll note's gate time (how long it's "held" before Release begins — see
/// `SynthParams`) is its own length in seconds; this is just a floor against a degenerate
/// zero-length note.
const MIN_NOTE_GATE_SECONDS: f32 = 0.01;
/// Oscillator copies stacked per voice for `SynthParams::unison_voices` (capped at 3).
const MAX_UNISON_VOICES: usize = 3;
/// Hz swept by the LFO at full depth when `LfoTarget::FilterCutoff` is selected — deliberately
/// narrower than `filter_env_amount_hz`'s +/-10kHz range since an LFO sweep is felt continuously
/// rather than once per note.
const FILTER_LFO_RANGE_HZ: f32 = 4000.0;
/// Max pitch swing (cents) at full LFO depth when `LfoTarget::Pitch` is selected — one semitone,
/// a standard vibrato range.
const PITCH_LFO_MAX_CENTS: f32 = 100.0;

/// Max pitch swing (semitones) at full modulation depth when an `TrineVoice` matrix slot targets
/// `ModTarget::Pitch` — a full octave, wider than Simple Synth's LFO-only vibrato range since this
/// can be driven by an envelope for pitch sweeps, not just an LFO.
const TRINE_PITCH_MOD_MAX_SEMITONES: f32 = 12.0;
/// Hz swept at full modulation depth when a matrix slot targets `ModTarget::FilterCutoff` or
/// `Filter2Cutoff` — comparable in scale to Simple Synth's `FILTER_LFO_RANGE_HZ`/
/// `filter_env_amount_hz`.
const TRINE_FILTER_MOD_RANGE_HZ: f32 = 8000.0;
/// Resonance swept at full modulation depth when a matrix slot targets `ModTarget::FilterResonance`.
const TRINE_RESONANCE_MOD_RANGE: f32 = 5.0;
/// Hz swept at full `TrineParams::filter_fm_amount` — audio-rate filter FM driven directly by osc2's
/// instantaneous sample, independent of the modulation matrix.
const TRINE_FILTER_FM_RANGE_HZ: f32 = 6000.0;
/// One-pole smoothing coefficient turning per-sample hashed noise into `TrineVoice`'s slow analog-
/// drift random walk — small enough that the drift wanders over seconds, not per-sample jitter.
const TRINE_ANALOG_DRIFT_SMOOTHING: f32 = 0.0008;
/// Max pitch drift (cents) at `TrineParams::analog_drift == 1.0` and the random walk at its extreme.
const TRINE_ANALOG_DRIFT_MAX_CENTS: f32 = 15.0;

/// Max pitch swing (semitones) at full modulation depth when a `WaveVoice` matrix slot targets
/// `WaveModTarget::Pitch` — see `TRINE_PITCH_MOD_MAX_SEMITONES`, the equivalent for `TrineVoice`.
const WAVE_PITCH_MOD_MAX_SEMITONES: f32 = 12.0;
/// Hz swept at full modulation depth when a matrix slot targets `WaveModTarget::FilterCutoff` or
/// `Filter2Cutoff` — see `TRINE_FILTER_MOD_RANGE_HZ`.
const WAVE_FILTER_MOD_RANGE_HZ: f32 = 8000.0;
/// Resonance swept at full modulation depth when a matrix slot targets `WaveModTarget::FilterResonance`.
const WAVE_RESONANCE_MOD_RANGE: f32 = 5.0;

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
}

impl Transport {
    /// A stopped, metronome-off transport at tick 0.
    pub fn new() -> Self {
        Self {
            playing: Arc::new(AtomicBool::new(false)),
            current_tick: Arc::new(AtomicUsize::new(0)),
            metronome_enabled: Arc::new(AtomicBool::new(false)),
        }
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
    pub fn start(
        song: Arc<Mutex<Song>>,
        transport: Transport,
        master_effect: MasterEffectSlot,
        track_effects: TrackEffectSlots,
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
                master_effect,
                track_effects,
                max_frames as usize,
            )?,
            SampleFormat::I16 => build_playback_stream::<i16>(
                &device,
                &config,
                song,
                transport,
                master_effect,
                track_effects,
                max_frames as usize,
            )?,
            SampleFormat::U16 => build_playback_stream::<u16>(
                &device,
                &config,
                song,
                transport,
                master_effect,
                track_effects,
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

/// Raw oscillator output in `[-1, 1]` for one cycle, `phase` running `0..1`. `pulse_width` is the
/// Square wave's duty cycle (0.5 = classic 50/50 square); ignored by the other waveforms.
fn waveform_sample(waveform: SynthWaveform, phase: f32, pulse_width: f32) -> f32 {
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
        // No meaningful phase/cycle for noise — instead hash `phase`'s own bit pattern into a
        // broadband-looking value. `phase` still changes by a fresh amount every sample (it's the
        // oscillator's running phase accumulator), and a good integer hash avalanches even a
        // single differing bit into a very different output, so this needs no separate PRNG state
        // and drops straight into this function's existing pure signature.
        SynthWaveform::Noise => hash_to_bipolar(phase.to_bits()),
    }
}

/// Cheap, dependency-free integer hash (Murmur3-style finalizer) — see `waveform_sample`'s
/// `Noise` arm and `TrineVoice`'s analog-drift random walk. Not cryptographic, just needs to look
/// broadband/uncorrelated for audio purposes.
fn hash_to_bipolar(x: u32) -> f32 {
    let mut h = x;
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb_352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846c_a68b);
    h ^= h >> 16;
    (h as f32 / u32::MAX as f32) * 2.0 - 1.0
}

/// The stage of a `Voice`'s amplitude envelope. There's no live "note off" event in this
/// sequencer (see `SynthParams` docs) — `Voice` instead knows its whole gate time up front and
/// transitions itself from whichever stage it's in straight to `Release` once that gate elapses.
#[derive(Clone, Copy, PartialEq, Default)]
enum EnvelopeStage {
    #[default]
    Attack,
    Decay,
    Sustain,
    Release,
}

/// A single synthesized voice: up to `MAX_UNISON_VOICES` detuned copies of an oscillator (see
/// `SynthWaveform`), plus an optional second oscillator (crossfaded) and sub-oscillator (mixed
/// additively), summed and run through a real attack/decay/sustain/release envelope, an optional
/// LFO (pitch/amplitude/filter-cutoff), and a resonant filter (switchable type) with its own
/// decay-only envelope. Standing in for a real instrument until phase 4 (which added samples for
/// drums; this remains the melodic/synth path).
#[derive(Clone, Copy)]
struct Voice {
    active: bool,

    phases: [f32; MAX_UNISON_VOICES],
    /// Recomputed every sample from `current_freq * unison_ratios[i] * pitch_lfo_ratio` rather
    /// than fixed at trigger time — this is what lets glide and pitch-LFO modulate an in-flight
    /// voice's pitch (see `unison_ratios`/`current_freq` below).
    phase_incs: [f32; MAX_UNISON_VOICES],
    /// Per-partial frequency ratio baked in at trigger time from unison detune (1.0 = no
    /// detuning). Multiplying this by the (possibly gliding/LFO'd) base frequency each sample
    /// gives that partial's instantaneous frequency.
    unison_ratios: [f32; MAX_UNISON_VOICES],
    unison_count: usize,
    waveform: SynthWaveform,
    pulse_width: f32,

    /// The note's base frequency this sample, before per-partial unison/osc2/sub ratios are
    /// applied. Equal to `target_freq` once any glide has finished (or immediately, if
    /// `glide_seconds == 0`).
    current_freq: f32,
    target_freq: f32,
    /// `current_freq`'s glide state, tracked in log2-frequency space so the pitch sweep is
    /// musically linear (semitone-linear) rather than a linear-Hz sweep. `glide_remaining` counts
    /// down to 0, at which point `current_freq` is pinned exactly to `target_freq` to avoid float
    /// drift and to keep the `glide_seconds == 0` case bit-identical to the pre-glide behavior.
    current_log2: f32,
    glide_step_log2: f32,
    glide_remaining: u32,

    /// Second oscillator, crossfaded against osc1 by `osc2_mix` (0 = osc1 only). No unison
    /// stacking of its own — see `SynthParams::osc2_mix` docs.
    osc2_waveform: SynthWaveform,
    osc2_ratio: f32,
    osc2_phase: f32,
    osc2_mix: f32,
    /// Hard sync: reset `osc2_phase` to 0 whenever unison voice 0 of osc1 wraps — see
    /// `SynthParams::osc2_sync` docs.
    osc2_sync: bool,
    /// Fixed one-octave-down sine, mixed in additively by `sub_mix` (0 = off).
    sub_phase: f32,
    sub_mix: f32,

    /// Free-running per-voice LFO (phase resets at trigger, like `filter_env` below — nothing in
    /// this engine sustains modulation state across notes).
    lfo_waveform: SynthWaveform,
    lfo_phase: f32,
    lfo_phase_inc: f32,
    lfo_target: LfoTarget,
    lfo_depth: f32,

    stage: EnvelopeStage,
    elapsed_samples: u64,
    /// Sample count at which this voice transitions to `Release`, regardless of what stage it's
    /// currently in — computed once at trigger time from the caller-supplied gate duration.
    gate_samples: u64,
    peak_amp: f32,
    sustain_amp: f32,
    amp: f32,
    attack_per_sample: f32,
    /// Multiplicative per-sample factor pulling `amp` toward `sustain_amp` (not toward zero —
    /// that's what lets decay_seconds/ENVELOPE_FLOOR reach a nonzero sustain plateau rather than
    /// always bottoming out at silence).
    decay_per_sample: f32,
    release_per_sample: f32,
    release_samples: f32,

    /// Independent decay-only envelope (1.0 -> 0.0) added on top of the base filter cutoff,
    /// giving a classic filter "pluck"/"wow" sweep that keeps closing even while the amplitude
    /// envelope is holding at its sustain level.
    filter_env: f32,
    filter_cutoff_hz: f32,
    filter_resonance: f32,
    filter_env_amount_hz: f32,
    filter_type: FilterType,
    /// TPT state-variable filter integrator state (Zavalishin) — chosen over the more common
    /// Chamberlin SVF because it stays numerically stable even as cutoff approaches Nyquist,
    /// which matters here since the default cutoff (20kHz) is deliberately near-inaudible/near-
    /// bypass and must not blow up at typical device sample rates.
    filter_ic1eq: f32,
    filter_ic2eq: f32,

    sample_rate: f32,
}

impl Default for Voice {
    fn default() -> Self {
        Self {
            active: false,
            phases: [0.0; MAX_UNISON_VOICES],
            phase_incs: [0.0; MAX_UNISON_VOICES],
            unison_ratios: [1.0; MAX_UNISON_VOICES],
            unison_count: 1,
            waveform: SynthWaveform::default(),
            pulse_width: 0.5,
            current_freq: 0.0,
            target_freq: 0.0,
            current_log2: 0.0,
            glide_step_log2: 0.0,
            glide_remaining: 0,
            osc2_waveform: SynthWaveform::default(),
            osc2_ratio: 1.0,
            osc2_phase: 0.0,
            osc2_mix: 0.0,
            osc2_sync: false,
            sub_phase: 0.0,
            sub_mix: 0.0,
            lfo_waveform: SynthWaveform::default(),
            lfo_phase: 0.0,
            lfo_phase_inc: 0.0,
            lfo_target: LfoTarget::default(),
            lfo_depth: 0.0,
            stage: EnvelopeStage::default(),
            elapsed_samples: 0,
            gate_samples: 0,
            peak_amp: 0.0,
            sustain_amp: 0.0,
            amp: 0.0,
            attack_per_sample: 0.0,
            decay_per_sample: 0.0,
            release_per_sample: 0.0,
            release_samples: 1.0,
            filter_env: 0.0,
            filter_cutoff_hz: 20_000.0,
            filter_resonance: 0.707,
            filter_env_amount_hz: 0.0,
            filter_type: FilterType::default(),
            filter_ic1eq: 0.0,
            filter_ic2eq: 0.0,
            sample_rate: 48_000.0,
        }
    }
}

impl Voice {
    /// `gate_seconds` is how long the note stays "held" (Attack/Decay/Sustain) before Release
    /// begins — see `SynthParams` and the callers in `Sequencer::process` for how each pattern
    /// type derives it. `glide_from_freq`, when `Some`, is the previously played pitch on this
    /// track to portamento from (only ever passed for piano-roll notes — see `Sequencer::process`
    /// and `SynthParams::glide_seconds`); `None` (or `glide_seconds == 0`) triggers at `freq`
    /// immediately, exactly like before this feature existed.
    fn trigger(
        &mut self,
        freq: f32,
        velocity: u8,
        sample_rate: f32,
        gate_seconds: f32,
        synth: &SynthParams,
        glide_from_freq: Option<f32>,
    ) {
        self.active = true;
        self.sample_rate = sample_rate;
        self.waveform = synth.waveform;
        self.pulse_width = synth.pulse_width.clamp(0.02, 0.98);

        let unison = (synth.unison_voices as usize).clamp(1, MAX_UNISON_VOICES);
        self.unison_count = unison;
        let detune_ratio = 2f32.powf(synth.unison_detune_cents.max(0.0) / 1200.0);
        self.phases = [0.0; MAX_UNISON_VOICES];
        self.unison_ratios = [1.0; MAX_UNISON_VOICES];
        match unison {
            2 => {
                self.unison_ratios[0] = 1.0 / detune_ratio;
                self.unison_ratios[1] = detune_ratio;
            }
            3 => {
                self.unison_ratios[0] = 1.0 / detune_ratio;
                self.unison_ratios[2] = detune_ratio;
                // index 1 stays centered at the plain pitch.
            }
            _ => {}
        }

        self.target_freq = freq;
        match glide_from_freq.filter(|_| synth.glide_seconds > 0.0) {
            Some(from) => {
                self.current_log2 = from.max(1.0).log2();
                let target_log2 = freq.max(1.0).log2();
                let glide_samples = (synth.glide_seconds * sample_rate).max(1.0);
                self.glide_step_log2 = (target_log2 - self.current_log2) / glide_samples;
                self.glide_remaining = glide_samples as u32;
                self.current_freq = from;
            }
            None => {
                self.current_log2 = freq.max(1.0).log2();
                self.glide_step_log2 = 0.0;
                self.glide_remaining = 0;
                self.current_freq = freq;
            }
        }
        // phase_incs/osc2/sub phase_incs are (re)computed every sample in `next_sample` from
        // `current_freq`, so no need to seed them here beyond what the fields above drive.
        self.phase_incs = [0.0; MAX_UNISON_VOICES];

        self.osc2_waveform = synth.osc2_waveform;
        self.osc2_ratio = 2f32.powf(synth.osc2_semitones as f32 / 12.0)
            * 2f32.powf(synth.osc2_detune_cents / 1200.0);
        self.osc2_phase = 0.0;
        self.osc2_mix = synth.osc2_mix.clamp(0.0, 1.0);
        self.osc2_sync = synth.osc2_sync;
        self.sub_phase = 0.0;
        self.sub_mix = synth.sub_osc_mix.clamp(0.0, 1.0);

        self.lfo_waveform = synth.lfo_waveform;
        self.lfo_phase = 0.0;
        self.lfo_phase_inc = synth.lfo_rate_hz.max(0.0) / sample_rate;
        self.lfo_target = synth.lfo_target;
        self.lfo_depth = synth.lfo_depth.clamp(0.0, 1.0);

        self.peak_amp = (velocity as f32 / 127.0).clamp(0.0, 1.0);
        self.sustain_amp = self.peak_amp * synth.sustain_level.clamp(0.0, 1.0);

        let attack_samples = synth.attack_seconds.max(0.0) * sample_rate;
        if attack_samples < 1.0 {
            self.amp = self.peak_amp;
            self.stage = EnvelopeStage::Decay;
        } else {
            self.amp = 0.0;
            self.attack_per_sample = self.peak_amp / attack_samples;
            self.stage = EnvelopeStage::Attack;
        }

        let decay_samples = (synth.decay_seconds.max(0.0) * sample_rate).max(1.0);
        self.decay_per_sample = ENVELOPE_FLOOR.powf(1.0 / decay_samples);

        self.release_samples = (synth.release_seconds.max(0.0) * sample_rate).max(1.0);
        self.release_per_sample = 1.0;

        self.elapsed_samples = 0;
        self.gate_samples = (gate_seconds.max(0.0) * sample_rate) as u64;

        self.filter_env = 1.0;
        self.filter_cutoff_hz = synth.filter_cutoff_hz.max(20.0);
        self.filter_resonance = synth.filter_resonance.max(0.05);
        self.filter_env_amount_hz = synth.filter_env_amount_hz;
        self.filter_type = synth.filter_type;
        self.filter_ic1eq = 0.0;
        self.filter_ic2eq = 0.0;
    }

    fn next_sample(&mut self) -> f32 {
        if !self.active {
            return 0.0;
        }

        // LFO: one free-running sine/etc cycle per voice, computed once and reused by whichever
        // target is active (only one target applies at a time — see `LfoTarget`).
        let lfo_value = waveform_sample(self.lfo_waveform, self.lfo_phase, 0.5);
        self.lfo_phase += self.lfo_phase_inc;
        if self.lfo_phase >= 1.0 {
            self.lfo_phase -= 1.0;
        }
        let pitch_lfo_ratio = if self.lfo_target == LfoTarget::Pitch {
            2f32.powf(self.lfo_depth * PITCH_LFO_MAX_CENTS * lfo_value / 1200.0)
        } else {
            1.0
        };

        // Glide: step `current_freq` toward `target_freq` in log2 space (musically linear). A
        // no-glide trigger starts with `glide_remaining == 0`, so this is a no-op and
        // `current_freq` stays exactly at the triggered frequency, same as before this feature.
        if self.glide_remaining > 0 {
            self.current_log2 += self.glide_step_log2;
            self.glide_remaining -= 1;
            self.current_freq = if self.glide_remaining == 0 {
                self.target_freq
            } else {
                2f32.powf(self.current_log2)
            };
        }
        let base_freq = self.current_freq * pitch_lfo_ratio;

        let mut osc1 = 0.0;
        let mut osc1_master_wrapped = false;
        for i in 0..self.unison_count {
            self.phase_incs[i] = base_freq * self.unison_ratios[i] / self.sample_rate;
            osc1 += waveform_sample(self.waveform, self.phases[i], self.pulse_width);
            self.phases[i] += self.phase_incs[i];
            if self.phases[i] >= 1.0 {
                self.phases[i] -= 1.0;
                if i == 0 {
                    osc1_master_wrapped = true;
                }
            }
        }
        osc1 /= self.unison_count as f32;

        let osc2_inc = base_freq * self.osc2_ratio / self.sample_rate;
        let osc2 = waveform_sample(self.osc2_waveform, self.osc2_phase, self.pulse_width);
        if self.osc2_sync && osc1_master_wrapped {
            // Hard sync: snap osc2 back to the start of its cycle every time osc1 completes one,
            // truncating osc2's waveform when it's tuned away from osc1.
            self.osc2_phase = 0.0;
        } else {
            self.osc2_phase += osc2_inc;
            if self.osc2_phase >= 1.0 {
                self.osc2_phase -= 1.0;
            }
        }

        let sub_inc = base_freq * 0.5 / self.sample_rate;
        let sub = (self.sub_phase * std::f32::consts::TAU).sin();
        self.sub_phase += sub_inc;
        if self.sub_phase >= 1.0 {
            self.sub_phase -= 1.0;
        }

        let osc = osc1 * (1.0 - self.osc2_mix) + osc2 * self.osc2_mix + sub * self.sub_mix;

        match self.stage {
            EnvelopeStage::Attack => {
                self.amp += self.attack_per_sample;
                if self.amp >= self.peak_amp {
                    self.amp = self.peak_amp;
                    self.stage = EnvelopeStage::Decay;
                }
            }
            EnvelopeStage::Decay => {
                self.amp = self.sustain_amp + (self.amp - self.sustain_amp) * self.decay_per_sample;
                if (self.amp - self.sustain_amp).abs() < ENVELOPE_FLOOR {
                    self.amp = self.sustain_amp;
                    if self.sustain_amp < ENVELOPE_FLOOR {
                        // Sustaining at silence is indistinguishable from being done — free the
                        // voice now instead of occupying a slot until the (possibly distant) gate
                        // closes, which is what a default (sustain_level = 0) voice always does.
                        self.active = false;
                    } else {
                        self.stage = EnvelopeStage::Sustain;
                    }
                }
            }
            EnvelopeStage::Sustain => {
                self.amp = self.sustain_amp;
            }
            EnvelopeStage::Release => {
                self.amp *= self.release_per_sample;
            }
        }

        self.elapsed_samples += 1;
        if self.active
            && self.stage != EnvelopeStage::Release
            && self.elapsed_samples >= self.gate_samples
        {
            self.stage = EnvelopeStage::Release;
            self.release_per_sample = ENVELOPE_FLOOR.powf(1.0 / self.release_samples);
        }
        if self.stage == EnvelopeStage::Release && self.amp < ENVELOPE_FLOOR {
            self.active = false;
        }

        let amp_lfo = if self.lfo_target == LfoTarget::Amplitude {
            1.0 - self.lfo_depth * 0.5 * (1.0 - lfo_value)
        } else {
            1.0
        };
        let enveloped = osc * self.amp * amp_lfo;

        // TPT state-variable filter (Zavalishin's "Art of VA Filter Design"), cutoff modulated by
        // the independent filter envelope computed above and, if selected, the LFO.
        self.filter_env *= self.decay_per_sample;
        let filter_lfo_hz = if self.lfo_target == LfoTarget::FilterCutoff {
            self.lfo_depth * FILTER_LFO_RANGE_HZ * lfo_value
        } else {
            0.0
        };
        let cutoff =
            (self.filter_cutoff_hz + self.filter_env_amount_hz * self.filter_env + filter_lfo_hz)
                .clamp(20.0, self.sample_rate * 0.49);
        let g = (std::f32::consts::PI * cutoff / self.sample_rate).tan();
        let k = 1.0 / self.filter_resonance;
        let a1 = 1.0 / (1.0 + g * (g + k));
        let a2 = g * a1;
        let a3 = g * a2;
        let v3 = enveloped - self.filter_ic2eq;
        let v1 = a1 * self.filter_ic1eq + a2 * v3;
        let v2 = self.filter_ic2eq + a2 * self.filter_ic1eq + a3 * v3;
        self.filter_ic1eq = 2.0 * v1 - self.filter_ic1eq;
        self.filter_ic2eq = 2.0 * v2 - self.filter_ic2eq;

        match self.filter_type {
            FilterType::Lowpass => v2,
            FilterType::Bandpass => v1,
            FilterType::Highpass => enveloped - k * v1 - v2,
            FilterType::Notch => enveloped - k * v1,
        }
    }
}

fn pitch_to_freq(pitch: u8) -> f32 {
    440.0 * 2f32.powf((pitch as f32 - 69.0) / 12.0)
}

/// One TPT state-variable filter stage (Zavalishin's "Art of VA Filter Design") — factored out of
/// `Voice::next_sample`'s inlined version (left untouched) so `TrineVoice` can cascade it once
/// (12dB/octave) or twice (24dB/octave) per `FilterSlope`, and run it twice per sample (filter1 +
/// filter2).
fn svf_stage(
    input: f32,
    cutoff_hz: f32,
    resonance: f32,
    filter_type: FilterType,
    sample_rate: f32,
    ic1eq: &mut f32,
    ic2eq: &mut f32,
) -> f32 {
    let cutoff = cutoff_hz.clamp(20.0, sample_rate * 0.49);
    let g = (std::f32::consts::PI * cutoff / sample_rate).tan();
    let k = 1.0 / resonance.max(0.05);
    let a1 = 1.0 / (1.0 + g * (g + k));
    let a2 = g * a1;
    let a3 = g * a2;
    let v3 = input - *ic2eq;
    let v1 = a1 * *ic1eq + a2 * v3;
    let v2 = *ic2eq + a2 * *ic1eq + a3 * v3;
    *ic1eq = 2.0 * v1 - *ic1eq;
    *ic2eq = 2.0 * v2 - *ic2eq;
    match filter_type {
        FilterType::Lowpass => v2,
        FilterType::Bandpass => v1,
        FilterType::Highpass => input - k * v1 - v2,
        FilterType::Notch => input - k * v1,
    }
}

/// Runs `svf_stage` once (`Slope12`) or twice in series (`Slope24`, 24dB/octave), using
/// `ic1eq[0]`/`ic2eq[0]` for the first stage and `ic1eq[1]`/`ic2eq[1]` for the optional second.
fn run_filter_stage(
    input: f32,
    cutoff_hz: f32,
    resonance: f32,
    filter_type: FilterType,
    slope: FilterSlope,
    sample_rate: f32,
    ic1eq: &mut [f32; 2],
    ic2eq: &mut [f32; 2],
) -> f32 {
    let stage1 = svf_stage(
        input,
        cutoff_hz,
        resonance,
        filter_type,
        sample_rate,
        &mut ic1eq[0],
        &mut ic2eq[0],
    );
    match slope {
        FilterSlope::Slope12 => stage1,
        FilterSlope::Slope24 => svf_stage(
            stage1,
            cutoff_hz,
            resonance,
            filter_type,
            sample_rate,
            &mut ic1eq[1],
            &mut ic2eq[1],
        ),
    }
}

/// Reusable attack/decay/sustain/release generator (see `EnvelopeStage`) — three instances back
/// `TrineVoice`'s Env1/Env2/Env3, so they're one mechanism instead of copy-pasted fields (this is
/// the same machinery `Voice` hand-inlines into its own attack_per_sample/decay_per_sample/...
/// fields; `Voice` itself is left untouched since Simple Synth doesn't need three of these).
#[derive(Clone, Copy, Default)]
struct EnvGen {
    stage: EnvelopeStage,
    elapsed_samples: u64,
    gate_samples: u64,
    peak: f32,
    sustain: f32,
    value: f32,
    attack_per_sample: f32,
    decay_per_sample: f32,
    release_per_sample: f32,
    release_samples: f32,
}

impl EnvGen {
    /// `peak` is the envelope's target level at the top of Attack — pass a velocity-scaled value
    /// for an amplitude envelope (see `TrineVoice::trigger`'s `env3`), or `1.0` for a pure 0..1
    /// modulation-matrix source (`env1`/`env2`).
    #[allow(clippy::too_many_arguments)]
    fn trigger(
        &mut self,
        attack_seconds: f32,
        decay_seconds: f32,
        sustain_level: f32,
        release_seconds: f32,
        gate_samples: u64,
        sample_rate: f32,
        peak: f32,
    ) {
        self.peak = peak;
        self.sustain = peak * sustain_level.clamp(0.0, 1.0);
        let attack_samples = attack_seconds.max(0.0) * sample_rate;
        if attack_samples < 1.0 {
            self.value = self.peak;
            self.stage = EnvelopeStage::Decay;
        } else {
            self.value = 0.0;
            self.attack_per_sample = self.peak / attack_samples;
            self.stage = EnvelopeStage::Attack;
        }
        let decay_samples = (decay_seconds.max(0.0) * sample_rate).max(1.0);
        self.decay_per_sample = ENVELOPE_FLOOR.powf(1.0 / decay_samples);
        self.release_samples = (release_seconds.max(0.0) * sample_rate).max(1.0);
        self.release_per_sample = 1.0;
        self.elapsed_samples = 0;
        self.gate_samples = gate_samples;
    }

    fn advance(&mut self) -> f32 {
        match self.stage {
            EnvelopeStage::Attack => {
                self.value += self.attack_per_sample;
                if self.value >= self.peak {
                    self.value = self.peak;
                    self.stage = EnvelopeStage::Decay;
                }
            }
            EnvelopeStage::Decay => {
                self.value = self.sustain + (self.value - self.sustain) * self.decay_per_sample;
                if (self.value - self.sustain).abs() < ENVELOPE_FLOOR {
                    self.value = self.sustain;
                    self.stage = EnvelopeStage::Sustain;
                }
            }
            EnvelopeStage::Sustain => self.value = self.sustain,
            EnvelopeStage::Release => self.value *= self.release_per_sample,
        }
        self.elapsed_samples += 1;
        if self.stage != EnvelopeStage::Release && self.elapsed_samples >= self.gate_samples {
            self.stage = EnvelopeStage::Release;
            self.release_per_sample = ENVELOPE_FLOOR.powf(1.0 / self.release_samples);
        }
        self.value
    }
}

/// One Trine-engine voice — see `TrineParams`. Three oscillators (with FM, ring mod, and per-voice
/// analog drift) run into a dual filter (series/parallel/off routing, switchable slope, drive,
/// and audio-rate filter FM from osc2's raw sample), while up to five modulation sources (2 LFOs,
/// 2 free envelopes, velocity) are evaluated each sample and routed through the track's
/// `mod_slots` onto their targets. A third, always-on envelope (`env3`) drives amplitude directly
/// — mirroring Logic's ES2 and its own hardwired "ENV 3 Vol" — so a freshly-selected Trine track is
/// immediately audible without needing any matrix routing at all. `mod_slots` is passed into `next_sample`
/// rather than copied in at `trigger` time, since it's a `Vec` and cloning one on every note
/// trigger would allocate on the real-time audio thread.
#[derive(Clone, Copy)]
struct TrineVoice {
    active: bool,
    sample_rate: f32,
    base_freq: f32,

    osc1_waveform: SynthWaveform,
    osc1_phase: f32,
    osc1_level: f32,
    pulse_width: f32,

    osc2_waveform: SynthWaveform,
    osc2_ratio: f32,
    osc2_phase: f32,
    osc2_level: f32,
    osc2_sync: bool,

    osc3_waveform: SynthWaveform,
    osc3_ratio: f32,
    osc3_phase: f32,
    osc3_level: f32,
    osc3_sync: bool,

    fm_amount: f32,
    ring_mod_mix: f32,
    analog_drift: f32,
    /// Slowly-smoothed noise driving the per-voice pitch drift — see `TRINE_ANALOG_DRIFT_SMOOTHING`.
    drift_lp: f32,
    drift_seed: u32,

    filter1_cutoff_hz: f32,
    filter1_resonance: f32,
    filter1_type: FilterType,
    filter1_slope: FilterSlope,
    filter1_ic1eq: [f32; 2],
    filter1_ic2eq: [f32; 2],
    filter2_cutoff_hz: f32,
    filter2_resonance: f32,
    filter2_type: FilterType,
    filter2_slope: FilterSlope,
    filter2_ic1eq: [f32; 2],
    filter2_ic2eq: [f32; 2],
    filter_routing: FilterRouting,
    filter_drive: f32,
    filter_fm_amount: f32,

    lfo1_waveform: SynthWaveform,
    lfo1_phase: f32,
    lfo1_phase_inc: f32,
    lfo2_waveform: SynthWaveform,
    lfo2_phase: f32,
    lfo2_phase_inc: f32,

    env1: EnvGen,
    env2: EnvGen,
    env3: EnvGen,
    velocity: f32,
}

impl Default for TrineVoice {
    fn default() -> Self {
        Self {
            active: false,
            sample_rate: 48_000.0,
            base_freq: 0.0,
            osc1_waveform: SynthWaveform::default(),
            osc1_phase: 0.0,
            osc1_level: 1.0,
            pulse_width: 0.5,
            osc2_waveform: SynthWaveform::default(),
            osc2_ratio: 1.0,
            osc2_phase: 0.0,
            osc2_level: 0.0,
            osc2_sync: false,
            osc3_waveform: SynthWaveform::default(),
            osc3_ratio: 1.0,
            osc3_phase: 0.0,
            osc3_level: 0.0,
            osc3_sync: false,
            fm_amount: 0.0,
            ring_mod_mix: 0.0,
            analog_drift: 0.0,
            drift_lp: 0.0,
            drift_seed: 1,
            filter1_cutoff_hz: 20_000.0,
            filter1_resonance: 0.707,
            filter1_type: FilterType::default(),
            filter1_slope: FilterSlope::default(),
            filter1_ic1eq: [0.0; 2],
            filter1_ic2eq: [0.0; 2],
            filter2_cutoff_hz: 20_000.0,
            filter2_resonance: 0.707,
            filter2_type: FilterType::default(),
            filter2_slope: FilterSlope::default(),
            filter2_ic1eq: [0.0; 2],
            filter2_ic2eq: [0.0; 2],
            filter_routing: FilterRouting::default(),
            filter_drive: 0.0,
            filter_fm_amount: 0.0,
            lfo1_waveform: SynthWaveform::default(),
            lfo1_phase: 0.0,
            lfo1_phase_inc: 0.0,
            lfo2_waveform: SynthWaveform::default(),
            lfo2_phase: 0.0,
            lfo2_phase_inc: 0.0,
            env1: EnvGen::default(),
            env2: EnvGen::default(),
            env3: EnvGen::default(),
            velocity: 0.0,
        }
    }
}

impl TrineVoice {
    fn trigger(
        &mut self,
        freq: f32,
        velocity: u8,
        sample_rate: f32,
        gate_seconds: f32,
        trine: &TrineParams,
    ) {
        self.active = true;
        self.sample_rate = sample_rate;
        self.base_freq = freq;

        self.osc1_waveform = trine.osc1_waveform;
        self.osc1_phase = 0.0;
        self.osc1_level = trine.osc1_level.clamp(0.0, 1.0);
        self.pulse_width = trine.pulse_width.clamp(0.02, 0.98);

        self.osc2_waveform = trine.osc2_waveform;
        self.osc2_ratio = 2f32.powf(trine.osc2_semitones as f32 / 12.0)
            * 2f32.powf(trine.osc2_detune_cents / 1200.0);
        self.osc2_phase = 0.0;
        self.osc2_level = trine.osc2_level.clamp(0.0, 1.0);
        self.osc2_sync = trine.osc2_sync;

        self.osc3_waveform = trine.osc3_waveform;
        self.osc3_ratio = 2f32.powf(trine.osc3_semitones as f32 / 12.0)
            * 2f32.powf(trine.osc3_detune_cents / 1200.0);
        self.osc3_phase = 0.0;
        self.osc3_level = trine.osc3_level.clamp(0.0, 1.0);
        self.osc3_sync = trine.osc3_sync;

        self.fm_amount = trine.fm_amount.max(0.0);
        self.ring_mod_mix = trine.ring_mod_mix.clamp(0.0, 1.0);
        self.analog_drift = trine.analog_drift.max(0.0);
        self.drift_lp = 0.0;
        self.drift_seed = freq.to_bits() ^ 0xA5A5_5A5A;

        self.filter1_cutoff_hz = trine.filter1_cutoff_hz.max(20.0);
        self.filter1_resonance = trine.filter1_resonance.max(0.05);
        self.filter1_type = trine.filter1_type;
        self.filter1_slope = trine.filter1_slope;
        self.filter1_ic1eq = [0.0; 2];
        self.filter1_ic2eq = [0.0; 2];
        self.filter2_cutoff_hz = trine.filter2_cutoff_hz.max(20.0);
        self.filter2_resonance = trine.filter2_resonance.max(0.05);
        self.filter2_type = trine.filter2_type;
        self.filter2_slope = trine.filter2_slope;
        self.filter2_ic1eq = [0.0; 2];
        self.filter2_ic2eq = [0.0; 2];
        self.filter_routing = trine.filter_routing;
        self.filter_drive = trine.filter_drive.max(0.0);
        self.filter_fm_amount = trine.filter_fm_amount;

        self.lfo1_waveform = trine.lfo1_waveform;
        self.lfo1_phase = 0.0;
        self.lfo1_phase_inc = trine.lfo1_rate_hz.max(0.0) / sample_rate;
        self.lfo2_waveform = trine.lfo2_waveform;
        self.lfo2_phase = 0.0;
        self.lfo2_phase_inc = trine.lfo2_rate_hz.max(0.0) / sample_rate;

        let gate_samples = (gate_seconds.max(0.0) * sample_rate) as u64;
        let velocity_scalar = (velocity as f32 / 127.0).clamp(0.0, 1.0);
        self.velocity = velocity_scalar;
        self.env1.trigger(
            trine.env1_attack_seconds,
            trine.env1_decay_seconds,
            trine.env1_sustain_level,
            trine.env1_release_seconds,
            gate_samples,
            sample_rate,
            1.0,
        );
        self.env2.trigger(
            trine.env2_attack_seconds,
            trine.env2_decay_seconds,
            trine.env2_sustain_level,
            trine.env2_release_seconds,
            gate_samples,
            sample_rate,
            1.0,
        );
        self.env3.trigger(
            trine.env3_attack_seconds,
            trine.env3_decay_seconds,
            trine.env3_sustain_level,
            trine.env3_release_seconds,
            gate_samples,
            sample_rate,
            velocity_scalar,
        );
    }

    /// `mod_slots` is the owning track's live `TrineParams::mod_slots` — see the struct doc comment
    /// for why it's a parameter here instead of a field copied in at `trigger` time.
    fn next_sample(&mut self, mod_slots: &[ModSlot]) -> f32 {
        if !self.active {
            return 0.0;
        }

        self.drift_seed = self.drift_seed.wrapping_add(0x9E37_79B9);
        let drift_noise = hash_to_bipolar(self.drift_seed);
        self.drift_lp += (drift_noise - self.drift_lp) * TRINE_ANALOG_DRIFT_SMOOTHING;

        let lfo1_value = waveform_sample(self.lfo1_waveform, self.lfo1_phase, 0.5);
        self.lfo1_phase += self.lfo1_phase_inc;
        if self.lfo1_phase >= 1.0 {
            self.lfo1_phase -= 1.0;
        }
        let lfo2_value = waveform_sample(self.lfo2_waveform, self.lfo2_phase, 0.5);
        self.lfo2_phase += self.lfo2_phase_inc;
        if self.lfo2_phase >= 1.0 {
            self.lfo2_phase -= 1.0;
        }

        let env1_value = self.env1.advance();
        let env2_value = self.env2.advance();

        let mut pitch_semitones = 0.0f32;
        let mut osc1_level_delta = 0.0f32;
        let mut osc2_level_delta = 0.0f32;
        let mut osc3_level_delta = 0.0f32;
        let mut pulse_width_delta = 0.0f32;
        let mut filter1_cutoff_delta = 0.0f32;
        let mut filter2_cutoff_delta = 0.0f32;
        let mut filter1_resonance_delta = 0.0f32;
        let mut fm_amount_delta = 0.0f32;
        let mut ring_mod_delta = 0.0f32;

        for slot in mod_slots {
            if slot.target == ModTarget::None || slot.source == ModSource::None {
                continue;
            }
            let source_value = match slot.source {
                ModSource::None => 0.0,
                ModSource::Lfo1 => lfo1_value,
                ModSource::Lfo2 => lfo2_value,
                ModSource::Env1 => env1_value,
                ModSource::Env2 => env2_value,
                ModSource::Velocity => self.velocity,
            };
            let contribution = source_value * slot.amount;
            match slot.target {
                ModTarget::None => {}
                ModTarget::Pitch => pitch_semitones += contribution * TRINE_PITCH_MOD_MAX_SEMITONES,
                ModTarget::Osc1Level => osc1_level_delta += contribution,
                ModTarget::Osc2Level => osc2_level_delta += contribution,
                ModTarget::Osc3Level => osc3_level_delta += contribution,
                ModTarget::PulseWidth => pulse_width_delta += contribution * 0.45,
                ModTarget::FilterCutoff => {
                    filter1_cutoff_delta += contribution * TRINE_FILTER_MOD_RANGE_HZ
                }
                ModTarget::Filter2Cutoff => {
                    filter2_cutoff_delta += contribution * TRINE_FILTER_MOD_RANGE_HZ
                }
                ModTarget::FilterResonance => {
                    filter1_resonance_delta += contribution * TRINE_RESONANCE_MOD_RANGE
                }
                ModTarget::FmAmount => fm_amount_delta += contribution,
                ModTarget::RingModMix => ring_mod_delta += contribution,
            }
        }

        let drift_ratio =
            2f32.powf(self.analog_drift * self.drift_lp * TRINE_ANALOG_DRIFT_MAX_CENTS / 1200.0);
        let pitch_ratio = 2f32.powf(pitch_semitones / 12.0);
        let freq = self.base_freq * drift_ratio * pitch_ratio;

        let osc1_level = (self.osc1_level + osc1_level_delta).clamp(0.0, 1.0);
        let osc2_level = (self.osc2_level + osc2_level_delta).clamp(0.0, 1.0);
        let osc3_level = (self.osc3_level + osc3_level_delta).clamp(0.0, 1.0);
        let pulse_width = (self.pulse_width + pulse_width_delta).clamp(0.02, 0.98);
        let fm_amount = (self.fm_amount + fm_amount_delta).max(0.0);
        let ring_mod_mix = (self.ring_mod_mix + ring_mod_delta).clamp(0.0, 1.0);

        let osc1_inc = freq / self.sample_rate;
        let osc2_inc = freq * self.osc2_ratio / self.sample_rate;
        let osc3_inc = freq * self.osc3_ratio / self.sample_rate;

        let osc2_raw = waveform_sample(self.osc2_waveform, self.osc2_phase, pulse_width);
        let osc1_raw = waveform_sample(self.osc1_waveform, self.osc1_phase, pulse_width);

        // FM: osc2's raw sample perturbs osc1's phase increment for this sample only.
        self.osc1_phase += osc1_inc * (1.0 + fm_amount * osc2_raw);
        let osc1_wrapped = !(0.0..1.0).contains(&self.osc1_phase);
        if self.osc1_phase >= 1.0 {
            self.osc1_phase -= 1.0;
        } else if self.osc1_phase < 0.0 {
            self.osc1_phase += 1.0;
        }

        if self.osc2_sync && osc1_wrapped {
            self.osc2_phase = 0.0;
        } else {
            self.osc2_phase += osc2_inc;
            if self.osc2_phase >= 1.0 {
                self.osc2_phase -= 1.0;
            }
        }

        let osc3_raw = waveform_sample(self.osc3_waveform, self.osc3_phase, pulse_width);
        if self.osc3_sync && osc1_wrapped {
            self.osc3_phase = 0.0;
        } else {
            self.osc3_phase += osc3_inc;
            if self.osc3_phase >= 1.0 {
                self.osc3_phase -= 1.0;
            }
        }

        let ring = osc1_raw * osc2_raw * ring_mod_mix;
        let osc_sum = osc1_raw * osc1_level + osc2_raw * osc2_level + osc3_raw * osc3_level + ring;

        let driven = if self.filter_drive > 0.0 {
            (osc_sum * (1.0 + self.filter_drive * 4.0)).tanh()
        } else {
            osc_sum
        };

        let env3_value = self.env3.advance();
        let enveloped = driven * env3_value;

        let filter1_cutoff = (self.filter1_cutoff_hz
            + filter1_cutoff_delta
            + self.filter_fm_amount * TRINE_FILTER_FM_RANGE_HZ * osc2_raw)
            .clamp(20.0, self.sample_rate * 0.49);
        let filter1_resonance =
            (self.filter1_resonance + filter1_resonance_delta).clamp(0.05, 20.0);
        let filter2_cutoff =
            (self.filter2_cutoff_hz + filter2_cutoff_delta).clamp(20.0, self.sample_rate * 0.49);

        let filter1_out = run_filter_stage(
            enveloped,
            filter1_cutoff,
            filter1_resonance,
            self.filter1_type,
            self.filter1_slope,
            self.sample_rate,
            &mut self.filter1_ic1eq,
            &mut self.filter1_ic2eq,
        );

        let output = match self.filter_routing {
            FilterRouting::Off => filter1_out,
            FilterRouting::Series => run_filter_stage(
                filter1_out,
                filter2_cutoff,
                self.filter2_resonance,
                self.filter2_type,
                self.filter2_slope,
                self.sample_rate,
                &mut self.filter2_ic1eq,
                &mut self.filter2_ic2eq,
            ),
            FilterRouting::Parallel => {
                let filter2_out = run_filter_stage(
                    enveloped,
                    filter2_cutoff,
                    self.filter2_resonance,
                    self.filter2_type,
                    self.filter2_slope,
                    self.sample_rate,
                    &mut self.filter2_ic1eq,
                    &mut self.filter2_ic2eq,
                );
                filter1_out + filter2_out
            }
        };

        // Lifecycle mirrors `Voice`'s: once env3 (the amp envelope) has entered Release and decayed
        // below the floor, this voice is done.
        if self.env3.stage == EnvelopeStage::Release && self.env3.value < ENVELOPE_FLOOR {
            self.active = false;
        }

        output
    }
}

/// One Wave-engine voice — see `WaveParams`. Two wavetable oscillators (each scanning its table's
/// frames via `osc*_position`, with an optional phase-warp — see `wavetable::warp_phase`) run
/// into a dual filter (series/parallel/off routing, switchable slope, drive) shared with
/// `TrineVoice`'s implementation, while up to five modulation sources (2 LFOs, 2 free envelopes,
/// velocity) are evaluated each sample and routed through the track's `mod_slots`. A sub-
/// oscillator and a noise oscillator mix in additively alongside the two wavetable oscillators.
/// `mod_slots` is passed into `next_sample` rather than copied in at `trigger` time, for the same
/// reason as `TrineVoice`'s.
#[derive(Clone, Copy)]
struct WaveVoice {
    active: bool,
    sample_rate: f32,
    base_freq: f32,

    osc1_table: WavetableId,
    osc1_position: f32,
    osc1_warp_mode: WaveWarpMode,
    osc1_warp_amount: f32,
    osc1_level: f32,
    osc1_mip: usize,
    unison: usize,
    unison_phases: [f32; MAX_UNISON_VOICES],
    unison_ratios: [f32; MAX_UNISON_VOICES],

    osc2_table: WavetableId,
    osc2_position: f32,
    osc2_warp_mode: WaveWarpMode,
    osc2_warp_amount: f32,
    osc2_level: f32,
    osc2_ratio: f32,
    osc2_phase: f32,
    osc2_mip: usize,

    sub_phase: f32,
    sub_ratio: f32,
    sub_level: f32,
    noise_level: f32,
    noise_seed: u32,

    filter1_cutoff_hz: f32,
    filter1_resonance: f32,
    filter1_type: FilterType,
    filter1_slope: FilterSlope,
    filter1_ic1eq: [f32; 2],
    filter1_ic2eq: [f32; 2],
    filter2_cutoff_hz: f32,
    filter2_resonance: f32,
    filter2_type: FilterType,
    filter2_slope: FilterSlope,
    filter2_ic1eq: [f32; 2],
    filter2_ic2eq: [f32; 2],
    filter_routing: FilterRouting,
    filter_drive: f32,

    lfo1_waveform: SynthWaveform,
    lfo1_phase: f32,
    lfo1_phase_inc: f32,
    lfo2_waveform: SynthWaveform,
    lfo2_phase: f32,
    lfo2_phase_inc: f32,

    env1: EnvGen,
    env2: EnvGen,
    amp_env: EnvGen,
    velocity: f32,
}

impl Default for WaveVoice {
    fn default() -> Self {
        Self {
            active: false,
            sample_rate: 48_000.0,
            base_freq: 0.0,
            osc1_table: WavetableId::default(),
            osc1_position: 0.0,
            osc1_warp_mode: WaveWarpMode::default(),
            osc1_warp_amount: 0.0,
            osc1_level: 1.0,
            osc1_mip: 0,
            unison: 1,
            unison_phases: [0.0; MAX_UNISON_VOICES],
            unison_ratios: [1.0; MAX_UNISON_VOICES],
            osc2_table: WavetableId::default(),
            osc2_position: 0.0,
            osc2_warp_mode: WaveWarpMode::default(),
            osc2_warp_amount: 0.0,
            osc2_level: 0.0,
            osc2_ratio: 1.0,
            osc2_phase: 0.0,
            osc2_mip: 0,
            sub_phase: 0.0,
            sub_ratio: 0.5,
            sub_level: 0.0,
            noise_level: 0.0,
            noise_seed: 1,
            filter1_cutoff_hz: 20_000.0,
            filter1_resonance: 0.707,
            filter1_type: FilterType::default(),
            filter1_slope: FilterSlope::default(),
            filter1_ic1eq: [0.0; 2],
            filter1_ic2eq: [0.0; 2],
            filter2_cutoff_hz: 20_000.0,
            filter2_resonance: 0.707,
            filter2_type: FilterType::default(),
            filter2_slope: FilterSlope::default(),
            filter2_ic1eq: [0.0; 2],
            filter2_ic2eq: [0.0; 2],
            filter_routing: FilterRouting::default(),
            filter_drive: 0.0,
            lfo1_waveform: SynthWaveform::default(),
            lfo1_phase: 0.0,
            lfo1_phase_inc: 0.0,
            lfo2_waveform: SynthWaveform::default(),
            lfo2_phase: 0.0,
            lfo2_phase_inc: 0.0,
            env1: EnvGen::default(),
            env2: EnvGen::default(),
            amp_env: EnvGen::default(),
            velocity: 0.0,
        }
    }
}

impl WaveVoice {
    fn trigger(
        &mut self,
        freq: f32,
        velocity: u8,
        sample_rate: f32,
        gate_seconds: f32,
        wave: &WaveParams,
    ) {
        self.active = true;
        self.sample_rate = sample_rate;
        self.base_freq = freq;

        self.osc1_table = wave.osc1_table;
        self.osc1_position = wave.osc1_position.clamp(0.0, 1.0);
        self.osc1_warp_mode = wave.osc1_warp_mode;
        self.osc1_warp_amount = wave.osc1_warp_amount.clamp(0.0, 1.0);
        self.osc1_level = wave.osc1_level.clamp(0.0, 1.0);
        self.osc1_mip = wavetable::choose_mip_level(freq, sample_rate);

        let unison = (wave.unison_voices as usize).clamp(1, MAX_UNISON_VOICES);
        self.unison = unison;
        let detune_ratio = 2f32.powf(wave.unison_detune_cents.max(0.0) / 1200.0);
        self.unison_phases = [0.0; MAX_UNISON_VOICES];
        self.unison_ratios = [1.0; MAX_UNISON_VOICES];
        match unison {
            2 => {
                self.unison_ratios[0] = 1.0 / detune_ratio;
                self.unison_ratios[1] = detune_ratio;
            }
            3 => {
                self.unison_ratios[0] = 1.0 / detune_ratio;
                self.unison_ratios[2] = detune_ratio;
            }
            _ => {}
        }

        self.osc2_table = wave.osc2_table;
        self.osc2_position = wave.osc2_position.clamp(0.0, 1.0);
        self.osc2_warp_mode = wave.osc2_warp_mode;
        self.osc2_warp_amount = wave.osc2_warp_amount.clamp(0.0, 1.0);
        self.osc2_level = wave.osc2_level.clamp(0.0, 1.0);
        self.osc2_ratio = 2f32.powf(wave.osc2_semitones as f32 / 12.0)
            * 2f32.powf(wave.osc2_detune_cents / 1200.0);
        self.osc2_phase = 0.0;
        self.osc2_mip = wavetable::choose_mip_level(freq * self.osc2_ratio, sample_rate);

        self.sub_phase = 0.0;
        self.sub_ratio = 2f32.powf(wave.sub_osc_semitones as f32 / 12.0);
        self.sub_level = wave.sub_osc_level.clamp(0.0, 1.0);
        self.noise_level = wave.noise_level.clamp(0.0, 1.0);
        self.noise_seed = freq.to_bits() ^ 0x1234_5678;

        self.filter1_cutoff_hz = wave.filter1_cutoff_hz.max(20.0);
        self.filter1_resonance = wave.filter1_resonance.max(0.05);
        self.filter1_type = wave.filter1_type;
        self.filter1_slope = wave.filter1_slope;
        self.filter1_ic1eq = [0.0; 2];
        self.filter1_ic2eq = [0.0; 2];
        self.filter2_cutoff_hz = wave.filter2_cutoff_hz.max(20.0);
        self.filter2_resonance = wave.filter2_resonance.max(0.05);
        self.filter2_type = wave.filter2_type;
        self.filter2_slope = wave.filter2_slope;
        self.filter2_ic1eq = [0.0; 2];
        self.filter2_ic2eq = [0.0; 2];
        self.filter_routing = wave.filter_routing;
        self.filter_drive = wave.filter_drive.max(0.0);

        self.lfo1_waveform = wave.lfo1_waveform;
        self.lfo1_phase = 0.0;
        self.lfo1_phase_inc = wave.lfo1_rate_hz.max(0.0) / sample_rate;
        self.lfo2_waveform = wave.lfo2_waveform;
        self.lfo2_phase = 0.0;
        self.lfo2_phase_inc = wave.lfo2_rate_hz.max(0.0) / sample_rate;

        let gate_samples = (gate_seconds.max(0.0) * sample_rate) as u64;
        let velocity_scalar = (velocity as f32 / 127.0).clamp(0.0, 1.0);
        self.velocity = velocity_scalar;
        self.env1.trigger(
            wave.env1_attack_seconds,
            wave.env1_decay_seconds,
            wave.env1_sustain_level,
            wave.env1_release_seconds,
            gate_samples,
            sample_rate,
            1.0,
        );
        self.env2.trigger(
            wave.env2_attack_seconds,
            wave.env2_decay_seconds,
            wave.env2_sustain_level,
            wave.env2_release_seconds,
            gate_samples,
            sample_rate,
            1.0,
        );
        self.amp_env.trigger(
            wave.amp_attack_seconds,
            wave.amp_decay_seconds,
            wave.amp_sustain_level,
            wave.amp_release_seconds,
            gate_samples,
            sample_rate,
            velocity_scalar,
        );
    }

    /// `mod_slots` is the owning track's live `WaveParams::mod_slots` — see `TrineVoice::next_sample`'s
    /// doc comment for why it's a parameter here instead of a field copied in at `trigger` time.
    fn next_sample(&mut self, mod_slots: &[WaveModSlot]) -> f32 {
        if !self.active {
            return 0.0;
        }

        let lfo1_value = waveform_sample(self.lfo1_waveform, self.lfo1_phase, 0.5);
        self.lfo1_phase += self.lfo1_phase_inc;
        if self.lfo1_phase >= 1.0 {
            self.lfo1_phase -= 1.0;
        }
        let lfo2_value = waveform_sample(self.lfo2_waveform, self.lfo2_phase, 0.5);
        self.lfo2_phase += self.lfo2_phase_inc;
        if self.lfo2_phase >= 1.0 {
            self.lfo2_phase -= 1.0;
        }

        let env1_value = self.env1.advance();
        let env2_value = self.env2.advance();

        let mut pitch_semitones = 0.0f32;
        let mut osc1_position_delta = 0.0f32;
        let mut osc2_position_delta = 0.0f32;
        let mut osc1_warp_delta = 0.0f32;
        let mut osc2_warp_delta = 0.0f32;
        let mut filter1_cutoff_delta = 0.0f32;
        let mut filter2_cutoff_delta = 0.0f32;
        let mut filter1_resonance_delta = 0.0f32;

        for slot in mod_slots {
            if slot.target == WaveModTarget::None || slot.source == WaveModSource::None {
                continue;
            }
            let source_value = match slot.source {
                WaveModSource::None => 0.0,
                WaveModSource::Lfo1 => lfo1_value,
                WaveModSource::Lfo2 => lfo2_value,
                WaveModSource::Env1 => env1_value,
                WaveModSource::Env2 => env2_value,
                WaveModSource::Velocity => self.velocity,
            };
            let contribution = source_value * slot.amount;
            match slot.target {
                WaveModTarget::None => {}
                WaveModTarget::Pitch => {
                    pitch_semitones += contribution * WAVE_PITCH_MOD_MAX_SEMITONES
                }
                WaveModTarget::Osc1Position => osc1_position_delta += contribution,
                WaveModTarget::Osc2Position => osc2_position_delta += contribution,
                WaveModTarget::Osc1WarpAmount => osc1_warp_delta += contribution,
                WaveModTarget::Osc2WarpAmount => osc2_warp_delta += contribution,
                WaveModTarget::FilterCutoff => {
                    filter1_cutoff_delta += contribution * WAVE_FILTER_MOD_RANGE_HZ
                }
                WaveModTarget::Filter2Cutoff => {
                    filter2_cutoff_delta += contribution * WAVE_FILTER_MOD_RANGE_HZ
                }
                WaveModTarget::FilterResonance => {
                    filter1_resonance_delta += contribution * WAVE_RESONANCE_MOD_RANGE
                }
            }
        }

        let pitch_ratio = 2f32.powf(pitch_semitones / 12.0);
        let freq = self.base_freq * pitch_ratio;

        let osc1_position = (self.osc1_position + osc1_position_delta).clamp(0.0, 1.0);
        let osc2_position = (self.osc2_position + osc2_position_delta).clamp(0.0, 1.0);
        let osc1_warp_amount = (self.osc1_warp_amount + osc1_warp_delta).clamp(0.0, 1.0);
        let osc2_warp_amount = (self.osc2_warp_amount + osc2_warp_delta).clamp(0.0, 1.0);

        let mut osc1_raw = 0.0f32;
        for i in 0..self.unison {
            let phase_inc = freq * self.unison_ratios[i] / self.sample_rate;
            let warped =
                wavetable::warp_phase(self.unison_phases[i], self.osc1_warp_mode, osc1_warp_amount);
            osc1_raw += wavetable::sample(self.osc1_table, osc1_position, warped, self.osc1_mip);
            self.unison_phases[i] += phase_inc;
            if self.unison_phases[i] >= 1.0 {
                self.unison_phases[i] -= 1.0;
            }
        }
        osc1_raw /= self.unison as f32;

        let osc2_inc = freq * self.osc2_ratio / self.sample_rate;
        let osc2_warped =
            wavetable::warp_phase(self.osc2_phase, self.osc2_warp_mode, osc2_warp_amount);
        let osc2_raw =
            wavetable::sample(self.osc2_table, osc2_position, osc2_warped, self.osc2_mip);
        self.osc2_phase += osc2_inc;
        if self.osc2_phase >= 1.0 {
            self.osc2_phase -= 1.0;
        }

        let sub_inc = freq * self.sub_ratio / self.sample_rate;
        let sub_raw = (self.sub_phase * std::f32::consts::TAU).sin();
        self.sub_phase += sub_inc;
        if self.sub_phase >= 1.0 {
            self.sub_phase -= 1.0;
        }

        self.noise_seed = self.noise_seed.wrapping_add(0x9E37_79B9);
        let noise_raw = hash_to_bipolar(self.noise_seed);

        let osc_sum = osc1_raw * self.osc1_level
            + osc2_raw * self.osc2_level
            + sub_raw * self.sub_level
            + noise_raw * self.noise_level;

        let driven = if self.filter_drive > 0.0 {
            (osc_sum * (1.0 + self.filter_drive * 4.0)).tanh()
        } else {
            osc_sum
        };

        let amp_value = self.amp_env.advance();
        let enveloped = driven * amp_value;

        let filter1_cutoff =
            (self.filter1_cutoff_hz + filter1_cutoff_delta).clamp(20.0, self.sample_rate * 0.49);
        let filter1_resonance =
            (self.filter1_resonance + filter1_resonance_delta).clamp(0.05, 20.0);
        let filter2_cutoff =
            (self.filter2_cutoff_hz + filter2_cutoff_delta).clamp(20.0, self.sample_rate * 0.49);

        let filter1_out = run_filter_stage(
            enveloped,
            filter1_cutoff,
            filter1_resonance,
            self.filter1_type,
            self.filter1_slope,
            self.sample_rate,
            &mut self.filter1_ic1eq,
            &mut self.filter1_ic2eq,
        );

        let output = match self.filter_routing {
            FilterRouting::Off => filter1_out,
            FilterRouting::Series => run_filter_stage(
                filter1_out,
                filter2_cutoff,
                self.filter2_resonance,
                self.filter2_type,
                self.filter2_slope,
                self.sample_rate,
                &mut self.filter2_ic1eq,
                &mut self.filter2_ic2eq,
            ),
            FilterRouting::Parallel => {
                let filter2_out = run_filter_stage(
                    enveloped,
                    filter2_cutoff,
                    self.filter2_resonance,
                    self.filter2_type,
                    self.filter2_slope,
                    self.sample_rate,
                    &mut self.filter2_ic1eq,
                    &mut self.filter2_ic2eq,
                );
                filter1_out + filter2_out
            }
        };

        // Lifecycle mirrors `TrineVoice`'s: once the amp envelope has entered Release and decayed
        // below the floor, this voice is done.
        if self.amp_env.stage == EnvelopeStage::Release && self.amp_env.value < ENVELOPE_FLOOR {
            self.active = false;
        }

        output
    }
}

/// Plays back a pre-resampled one-shot sample from start to end.
#[derive(Clone, Default)]
struct SampleVoice {
    buffer: Option<Arc<SampleBuffer>>,
    position: usize,
    gain: f32,
}

impl SampleVoice {
    fn trigger(&mut self, buffer: Arc<SampleBuffer>, velocity: u8) {
        self.trigger_with_gain(buffer, (velocity as f32 / 127.0).clamp(0.0, 1.0));
    }

    /// Same as `trigger`, but with a continuous gain instead of a 0..127 velocity byte — used for
    /// `AudioClip` playback (see `model::AudioClip::gain`), which isn't velocity-triggered.
    fn trigger_with_gain(&mut self, buffer: Arc<SampleBuffer>, gain: f32) {
        self.buffer = Some(buffer);
        self.position = 0;
        self.gain = gain;
    }

    fn next_sample(&mut self) -> f32 {
        let Some(buffer) = &self.buffer else {
            return 0.0;
        };
        match buffer.mono.get(self.position) {
            Some(&s) => {
                self.position += 1;
                s * self.gain
            }
            None => {
                self.buffer = None;
                0.0
            }
        }
    }
}

/// One track's independent voice pools, so a busy drum track can never starve a melodic track's
/// polyphony (and so each track's dry signal can be kept separate for per-track CLAP effects).
struct TrackVoices {
    voices: [Voice; VOICE_COUNT],
    next_voice: usize,
    /// Independent voice pool for the Trine engine (see `SynthEngine::Trine`) — always allocated per
    /// track (a small, constant memory cost) but only ever triggered when that track's
    /// `synth_engine` is `Trine`; an idle pool just contributes silence to the mix.
    trine_voices: [TrineVoice; VOICE_COUNT],
    next_trine_voice: usize,
    /// Independent voice pool for the Wave engine (see `SynthEngine::Wave`) — same always-
    /// allocated-but-idle-when-unused arrangement as `trine_voices`.
    wave_voices: [WaveVoice; VOICE_COUNT],
    next_wave_voice: usize,
    sample_voices: [SampleVoice; SAMPLE_VOICE_COUNT],
    next_sample_voice: usize,
    /// Most recently triggered piano-roll pitch on this track, for `SynthParams::glide_seconds`
    /// to portamento from. Monophonic "last note" memory layered on top of the polyphonic voice
    /// pool — standard glide behavior even in an otherwise-polyphonic engine. Step-grid hits never
    /// read or write this (see `Sequencer::process`).
    last_freq: Option<f32>,
}

impl TrackVoices {
    fn new() -> Self {
        Self {
            voices: [Voice::default(); VOICE_COUNT],
            next_voice: 0,
            trine_voices: [TrineVoice::default(); VOICE_COUNT],
            next_trine_voice: 0,
            wave_voices: [WaveVoice::default(); VOICE_COUNT],
            next_wave_voice: 0,
            sample_voices: std::array::from_fn(|_| SampleVoice::default()),
            next_sample_voice: 0,
            last_freq: None,
        }
    }
}

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
/// length from this single formula, so they can never drift apart.
fn arrangement_length_ticks(song: &Song) -> usize {
    let pattern_end = song
        .tracks
        .iter()
        .flat_map(|track| track.regions.iter())
        .map(|region| region.start_tick + region.loop_length_steps * TICKS_PER_STEP)
        .max()
        .unwrap_or(0);

    // An audio clip has no stored length (see `model::AudioClip`) — its duration is however long
    // its decoded buffer is, in real seconds, converted to ticks at the song's current tempo so a
    // recording never gets truncated by the arrangement looping underneath it.
    let ticks_per_second = ticks_per_second(song.bpm);
    let audio_end = song
        .tracks
        .iter()
        .flat_map(|track| track.audio_clips.iter())
        .filter_map(|clip| {
            let buffer = clip.buffer.as_ref()?;
            let duration_seconds = buffer.mono.len() as f64 / buffer.sample_rate.max(1) as f64;
            let duration_ticks = (duration_seconds * ticks_per_second).ceil() as usize;
            Some(clip.start_tick + duration_ticks)
        })
        .max()
        .unwrap_or(0);

    pattern_end.max(audio_end).max(TICKS_PER_STEP).max(1)
}

struct Sequencer {
    sample_rate: f32,
    track_voices: Vec<TrackVoices>,
    tick_index: usize,
    samples_until_next_tick: f64,
    last_triggered_tick: usize,
    /// How many samples into its decaying click envelope the metronome currently is —
    /// `>= metronome_click_len` means silent (see `next_metronome_click_sample`).
    metronome_click_pos: usize,
    metronome_click_len: usize,
    metronome_click_freq: f32,
}

/// One beat's worth of ticks (see `STEPS_PER_BEAT`/`TICKS_PER_STEP`) — the metronome clicks once
/// per beat, not once per step.
const METRONOME_BEAT_TICKS: usize = STEPS_PER_BEAT as usize * TICKS_PER_STEP;
const METRONOME_CLICK_SECONDS: f32 = 0.03;
const METRONOME_CLICK_HZ: f32 = 1000.0;
const METRONOME_ACCENT_HZ: f32 = 1600.0;
const METRONOME_GAIN: f32 = 0.5;

impl Sequencer {
    fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            track_voices: Vec::new(),
            tick_index: 0,
            samples_until_next_tick: 0.0,
            last_triggered_tick: 0,
            metronome_click_pos: 0,
            metronome_click_len: 0,
            metronome_click_freq: 0.0,
        }
    }

    /// Rewinds the tick clock to the start without touching in-flight voices.
    fn reset_position(&mut self) {
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
    fn current_tick(&self) -> usize {
        self.last_triggered_tick
    }

    /// Renders `frames` samples, writing one dry mix per track into `track_out[i]` (resized to
    /// match `snapshot.tracks`). Track count can change between calls (e.g. after loading a
    /// different song) — `track_voices` is resized to match, discarding in-flight voices for any
    /// removed track.
    fn process(
        &mut self,
        snapshot: &Song,
        frames: usize,
        track_out: &mut Vec<Vec<f32>>,
        metronome_enabled: bool,
        metronome_out: &mut Vec<f32>,
    ) {
        while self.track_voices.len() < snapshot.tracks.len() {
            self.track_voices.push(TrackVoices::new());
        }
        self.track_voices.truncate(snapshot.tracks.len());

        track_out.resize_with(snapshot.tracks.len(), Vec::new);
        for buf in track_out.iter_mut() {
            buf.clear();
            buf.resize(frames, 0.0);
        }
        metronome_out.clear();
        metronome_out.resize(frames, 0.0);

        let arrangement_len_ticks = arrangement_length_ticks(snapshot);
        self.tick_index %= arrangement_len_ticks;
        let samples_per_tick = (self.sample_rate as f64 * 60.0
            / (snapshot.bpm.max(1.0) as f64)
            / STEPS_PER_BEAT
            / TICKS_PER_STEP as f64)
            .max(1.0);
        // When any track is soloed, only soloed tracks are audible (mute is ignored for them);
        // every other track goes silent regardless of its own mute state.
        let any_solo = snapshot.tracks.iter().any(|track| track.solo);
        let track_silent = |track: &Track| {
            if any_solo {
                !track.solo
            } else {
                track.muted
            }
        };

        for sample_index in 0..frames {
            if self.samples_until_next_tick <= 0.0 {
                self.last_triggered_tick = self.tick_index;
                if metronome_enabled && self.tick_index % METRONOME_BEAT_TICKS == 0 {
                    self.trigger_metronome_click(self.tick_index == 0);
                }
                // A track's regions are independently positioned and may overlap in time with
                // each other (unusual, but not prevented) — every region active at this tick on
                // this track contributes; tracks with nothing placed at this tick stay silent.
                for (track_index, track) in snapshot.tracks.iter().enumerate() {
                    if track_silent(track) {
                        continue;
                    }
                    for region in track.regions.iter().filter(|region| {
                        self.tick_index >= region.start_tick
                            && self.tick_index
                                < region.start_tick + region.loop_length_steps * TICKS_PER_STEP
                    }) {
                        // The region's on-timeline span may be shorter than its own content
                        // (truncating it) or longer (looping it) — both fall out of this modulo.
                        let region_local_tick = (self.tick_index - region.start_tick)
                            % region.content_length_ticks().max(1);
                        let tv = &mut self.track_voices[track_index];
                        match &region.content {
                            RegionContent::StepGrid(lanes) => {
                                if region_local_tick % TICKS_PER_STEP == 0 {
                                    let step_index = region_local_tick / TICKS_PER_STEP;
                                    for lane in lanes {
                                        if let Some(Some(velocity)) = lane.steps.get(step_index) {
                                            if let Some(sample) = &lane.sample {
                                                tv.sample_voices[tv.next_sample_voice]
                                                    .trigger(sample.clone(), *velocity);
                                                tv.next_sample_voice =
                                                    (tv.next_sample_voice + 1) % SAMPLE_VOICE_COUNT;
                                            } else {
                                                let freq = pitch_to_freq(lane.pitch);
                                                // A lane with its own synth (see
                                                // `Lane::synth_override`) renders with that instead
                                                // of the track's — lets a step-grid track mix synth
                                                // patches per lane (kick on one, hi-hat on another).
                                                let (engine, synth, trine, wave) =
                                                    if lane.synth_override {
                                                        (
                                                            lane.synth_engine,
                                                            &lane.synth,
                                                            &lane.trine,
                                                            &lane.wave,
                                                        )
                                                    } else {
                                                        (
                                                            track.synth_engine,
                                                            &track.synth,
                                                            &track.trine,
                                                            &track.wave,
                                                        )
                                                    };
                                                // Step-grid hits have no explicit length, unlike a
                                                // piano-roll note — treat "attack + decay" as the
                                                // gate time, so Release begins right as Decay would
                                                // otherwise have finished settling at the sustain level.
                                                match engine {
                                                    SynthEngine::Simple => {
                                                        let gate_seconds = synth.attack_seconds
                                                            + synth.decay_seconds;
                                                        tv.voices[tv.next_voice].trigger(
                                                            freq,
                                                            *velocity,
                                                            self.sample_rate,
                                                            gate_seconds,
                                                            synth,
                                                            // Step-grid hits never glide — see
                                                            // `SynthParams::glide_seconds`.
                                                            None,
                                                        );
                                                        tv.next_voice =
                                                            (tv.next_voice + 1) % VOICE_COUNT;
                                                    }
                                                    SynthEngine::Trine => {
                                                        let gate_seconds =
                                                            trine.env3_attack_seconds
                                                                + trine.env3_decay_seconds;
                                                        tv.trine_voices[tv.next_trine_voice]
                                                            .trigger(
                                                                freq,
                                                                *velocity,
                                                                self.sample_rate,
                                                                gate_seconds,
                                                                trine,
                                                            );
                                                        tv.next_trine_voice =
                                                            (tv.next_trine_voice + 1) % VOICE_COUNT;
                                                    }
                                                    SynthEngine::Wave => {
                                                        let gate_seconds =
                                                            wave.amp_attack_seconds
                                                                + wave.amp_decay_seconds;
                                                        tv.wave_voices[tv.next_wave_voice].trigger(
                                                            freq,
                                                            *velocity,
                                                            self.sample_rate,
                                                            gate_seconds,
                                                            wave,
                                                        );
                                                        tv.next_wave_voice =
                                                            (tv.next_wave_voice + 1) % VOICE_COUNT;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            RegionContent::PianoRoll(notes) => {
                                for note in notes {
                                    if note.start_tick != region_local_tick {
                                        continue;
                                    }
                                    let freq = pitch_to_freq(note.pitch);
                                    // The note's own length is its gate time: it holds through
                                    // Attack/Decay/Sustain for exactly this long before Release begins.
                                    let gate_seconds = ((note.length_ticks as f64
                                        * samples_per_tick
                                        / self.sample_rate as f64)
                                        as f32)
                                        .max(MIN_NOTE_GATE_SECONDS);
                                    match track.synth_engine {
                                        SynthEngine::Simple => {
                                            let glide_from = if track.synth.glide_seconds > 0.0 {
                                                tv.last_freq
                                            } else {
                                                None
                                            };
                                            tv.voices[tv.next_voice].trigger(
                                                freq,
                                                note.velocity,
                                                self.sample_rate,
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
                                                self.sample_rate,
                                                gate_seconds,
                                                &track.trine,
                                            );
                                            tv.next_trine_voice =
                                                (tv.next_trine_voice + 1) % VOICE_COUNT;
                                        }
                                        SynthEngine::Wave => {
                                            // Glide isn't part of the Wave engine in this pass.
                                            tv.wave_voices[tv.next_wave_voice].trigger(
                                                freq,
                                                note.velocity,
                                                self.sample_rate,
                                                gate_seconds,
                                                &track.wave,
                                            );
                                            tv.next_wave_voice =
                                                (tv.next_wave_voice + 1) % VOICE_COUNT;
                                        }
                                    }
                                    tv.last_freq = Some(freq);
                                }
                            }
                        }
                    }
                }

                // Audio clips live directly on their track at an absolute song tick (see
                // `model::AudioClip`), not inside a `Region` — so unlike step-grid/piano-roll
                // content above, this doesn't go through the per-track region loop at all.
                for (track_index, track) in snapshot.tracks.iter().enumerate() {
                    if track_silent(track) || track.kind != TrackKind::Audio {
                        continue;
                    }
                    let tv = &mut self.track_voices[track_index];
                    for clip in &track.audio_clips {
                        if clip.start_tick != self.tick_index {
                            continue;
                        }
                        let Some(buffer) = &clip.buffer else { continue };
                        tv.sample_voices[tv.next_sample_voice]
                            .trigger_with_gain(buffer.clone(), clip.gain);
                        tv.next_sample_voice = (tv.next_sample_voice + 1) % SAMPLE_VOICE_COUNT;
                    }
                }
                self.tick_index = (self.tick_index + 1) % arrangement_len_ticks;
                self.samples_until_next_tick += samples_per_tick;
            }
            self.samples_until_next_tick -= 1.0;

            for (track_index, tv) in self.track_voices.iter_mut().enumerate() {
                let mut mixed = 0.0f32;
                for voice in tv.voices.iter_mut() {
                    mixed += voice.next_sample();
                }
                let mod_slots = &snapshot.tracks[track_index].trine.mod_slots;
                for voice in tv.trine_voices.iter_mut() {
                    mixed += voice.next_sample(mod_slots);
                }
                let wave_mod_slots = &snapshot.tracks[track_index].wave.mod_slots;
                for voice in tv.wave_voices.iter_mut() {
                    mixed += voice.next_sample(wave_mod_slots);
                }
                for voice in tv.sample_voices.iter_mut() {
                    mixed += voice.next_sample();
                }
                track_out[track_index][sample_index] = mixed;
            }
            metronome_out[sample_index] = self.next_metronome_click_sample();
        }
    }
}

fn build_playback_stream<T>(
    device: &cpal::Device,
    config: &StreamConfig,
    song: Arc<Mutex<Song>>,
    transport: Transport,
    master_effect: MasterEffectSlot,
    track_effects: TrackEffectSlots,
    max_frames: usize,
) -> Result<Stream>
where
    T: SizedSample + FromSample<f32>,
{
    let sample_rate = config.sample_rate as f32;
    let channels = (config.channels as usize).max(1);

    let mut sequencer = Sequencer::new(sample_rate);
    let mut scratch: Vec<f32> = Vec::with_capacity(max_frames);
    let mut track_dry: Vec<Vec<f32>> = Vec::new();
    let mut metronome_dry: Vec<f32> = Vec::with_capacity(max_frames);

    // Per-track CLAP insert-effect-chain scratch (one `Vec<EffectScratch>` per track index, grown
    // lazily to match that track's chain length) and a pair of reusable stereo buffers plus a mono
    // downmix buffer for whichever track is currently being processed.
    let mut track_scratch: Vec<Vec<plugin_host::EffectScratch>> = Vec::new();
    let mut track_effect_out_l: Vec<f32> = Vec::with_capacity(max_frames);
    let mut track_effect_out_r: Vec<f32> = Vec::with_capacity(max_frames);
    let mut track_chain_mono: Vec<f32> = Vec::with_capacity(max_frames);

    // Scratch for the master-bus CLAP effect. Allocated once and reused every callback.
    let mut master_scratch = plugin_host::EffectScratch::new();
    master_scratch.reserve(max_frames);
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
        track_dry.resize_with(snapshot.tracks.len(), || Vec::with_capacity(max_frames));
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

    let err_fn = |err| eprintln!("audio stream error: {err}");

    let stream = device.build_output_stream(
        *config,
        move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
            let frames = data.len() / channels;
            scratch.resize(frames, 0.0);
            scratch.iter_mut().for_each(|s| *s = 0.0);

            // Note: even when stopped, silence still runs through the master
            // effect below rather than short-circuiting straight to the
            // device — otherwise a delay/reverb tail would cut off instantly
            // on Stop instead of ringing out naturally, like in a real DAW.
            // (Per-track effects don't get this treatment: while stopped, no
            // track has anything playing through them, so there's no tail to
            // preserve there — only the master bus stays fed with silence.)
            if transport.is_playing() {
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
                sequencer.process(
                    snapshot,
                    frames,
                    &mut track_dry,
                    transport.is_metronome_enabled(),
                    &mut metronome_dry,
                );
                transport
                    .current_tick
                    .store(sequencer.current_tick(), Ordering::Relaxed);

                // Run each track's dry mix through its own CLAP insert effect chain (if any
                // plugins are loaded there), apply that track's volume, then sum every track
                // (post-effect, post-volume) into the master bus.
                track_effect_out_l.resize(frames, 0.0);
                track_effect_out_r.resize(frames, 0.0);
                if let Ok(mut chains) = track_effects.lock() {
                    while track_scratch.len() < track_dry.len() {
                        track_scratch.push(Vec::new());
                    }
                    for (track_index, dry) in track_dry.iter().enumerate() {
                        let volume = snapshot.tracks.get(track_index).map_or(1.0, |t| t.volume);
                        let chain = chains
                            .get_mut(track_index)
                            .map_or(&mut [][..], Vec::as_mut_slice);
                        let stage_scratch = &mut track_scratch[track_index];
                        while stage_scratch.len() < chain.len() {
                            stage_scratch.push(plugin_host::EffectScratch::new());
                        }
                        let used = plugin_host::process_effect_chain(
                            chain,
                            dry,
                            &mut track_effect_out_l,
                            &mut track_effect_out_r,
                            stage_scratch,
                            &mut track_chain_mono,
                        );
                        if used {
                            for i in 0..frames {
                                scratch[i] +=
                                    volume * 0.5 * (track_effect_out_l[i] + track_effect_out_r[i]);
                            }
                        } else {
                            for (out, s) in scratch.iter_mut().zip(dry) {
                                *out += volume * *s;
                            }
                        }
                    }
                } else {
                    for (track_index, dry) in track_dry.iter().enumerate() {
                        let volume = snapshot.tracks.get(track_index).map_or(1.0, |t| t.volume);
                        for (out, s) in scratch.iter_mut().zip(dry) {
                            *out += volume * *s;
                        }
                    }
                }

                for (out, click) in scratch.iter_mut().zip(&metronome_dry) {
                    *out += *click;
                }

                for s in scratch.iter_mut() {
                    *s = (*s * MASTER_GAIN).tanh();
                }
            } else {
                sequencer.reset_position();
                transport.current_tick.store(0, Ordering::Relaxed);
            }

            // Run the mix through the master-bus CLAP effect, if one is loaded.
            // Falls back to the dry mono mix (duplicated to L/R) on any failure.
            // Channel counts come from what the plugin actually declared via
            // the `audio-ports` extension (see `plugin_host::load_and_activate`)
            // — assuming every effect is 2-in/2-out caused real plugins (e.g.
            // ZamDelay, which is mono-in) to read past their declared buffers.
            let mut used_plugin = false;
            if let Ok(mut guard) = master_effect.lock() {
                if let Some(effect) = guard.as_mut() {
                    plugin_out_l.resize(frames, 0.0);
                    plugin_out_r.resize(frames, 0.0);
                    used_plugin = plugin_host::process_effect(
                        effect,
                        &scratch,
                        &mut plugin_out_l,
                        &mut plugin_out_r,
                        &mut master_scratch,
                    );
                }
            }

            let (left, right): (&[f32], &[f32]) = if used_plugin {
                (&plugin_out_l, &plugin_out_r)
            } else {
                (&scratch, &scratch)
            };

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

/// Renders `loops` repetitions of the song's pattern content to a mono
/// 16-bit WAV file, using the exact same synthesis path as real-time
/// playback (via `Sequencer`), so the bounce sounds like what you hear.
pub fn render_song_to_wav(
    song: &Song,
    sample_rate: u32,
    loops: u32,
    path: &std::path::Path,
) -> Result<()> {
    let arrangement_len_ticks = arrangement_length_ticks(song);
    let samples_per_tick = (sample_rate as f64 * 60.0
        / (song.bpm.max(1.0) as f64)
        / STEPS_PER_BEAT
        / TICKS_PER_STEP as f64)
        .max(1.0);
    let total_samples =
        (arrangement_len_ticks as f64 * samples_per_tick * (loops.max(1) as f64)).round() as usize;

    // Dry-only, matching live playback's scope cut: CLAP effects (master or per-track) don't
    // route into the bounce.
    let mut sequencer = Sequencer::new(sample_rate as f32);
    let mut track_dry: Vec<Vec<f32>> = Vec::new();
    let mut metronome_dry: Vec<f32> = Vec::new();
    // The metronome is a monitoring aid, not part of the song — bounces never include it.
    sequencer.process(
        song,
        total_samples,
        &mut track_dry,
        false,
        &mut metronome_dry,
    );

    let mut buffer = vec![0.0f32; total_samples];
    for (track_buf, track) in track_dry.iter().zip(&song.tracks) {
        for (out, s) in buffer.iter_mut().zip(track_buf) {
            *out += track.volume * *s;
        }
    }
    for s in buffer.iter_mut() {
        *s = (*s * MASTER_GAIN).tanh();
    }

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .with_context(|| format!("failed to create wav file: {}", path.display()))?;
    for sample in buffer {
        let value = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        writer
            .write_sample(value)
            .context("failed to write wav sample")?;
    }
    writer.finalize().context("failed to finalize wav file")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pitch_to_freq_matches_concert_a() {
        assert!((pitch_to_freq(69) - 440.0).abs() < 0.001);
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
            early_peak = early_peak.max(voice.next_sample().abs());
        }
        assert!(
            early_peak > 0.05,
            "voice should be clearly audible right after trigger"
        );

        for _ in 0..(sample_rate as usize) {
            voice.next_sample();
        }
        assert!(
            !voice.active,
            "voice should have decayed to silence within 1 second"
        );
        assert_eq!(
            voice.next_sample(),
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
            short.next_sample();
            long.next_sample();
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
        let first = voice.next_sample().abs();
        assert!(
            first < 0.1,
            "voice should start near silent during its attack ramp, got {first}"
        );

        // After the attack window elapses the voice should be in full swing.
        for _ in 0..(sample_rate * 0.05) as usize {
            voice.next_sample();
        }
        let mut peak = 0.0f32;
        for _ in 0..50 {
            peak = peak.max(voice.next_sample().abs());
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
            voice.next_sample();
        }
        let mut peak = 0.0f32;
        for _ in 0..50 {
            peak = peak.max(voice.next_sample().abs());
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
                voice.next_sample();
            }
            for _ in 0..200 {
                let s = voice.next_sample();
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
            voice.next_sample();
        }
        let mut sustain_peak = 0.0f32;
        for _ in 0..50 {
            sustain_peak = sustain_peak.max(voice.next_sample().abs());
        }
        assert!(
            sustain_peak > 0.5,
            "should be holding near the sustain level, got {sustain_peak}"
        );
        assert!(voice.active, "voice should still be active while gated");

        // Run past the gate close (0.2s) plus the release tail.
        for _ in 0..(sample_rate * 0.2) as usize {
            voice.next_sample();
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
            let s = voice.next_sample();
            assert!(
                (-1.01..=1.01).contains(&s),
                "unison output out of range: {s}"
            );
        }
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
                voice.next_sample(); // let the filter settle
            }
            let mut peak = 0.0f32;
            for _ in 0..200 {
                peak = peak.max(voice.next_sample().abs());
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
                voice.next_sample(); // let the filter settle
            }
            for _ in 0..200 {
                let s = voice.next_sample();
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
                voice.next_sample(); // let the filter settle
            }
            let n = 400;
            let sum_sq: f32 = (0..n)
                .map(|_| {
                    let s = voice.next_sample();
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
                voice.next_sample(); // settle past attack/decay
            }
            let window = (sample_rate / 40.0) as usize; // 25ms, 4 windows per LFO cycle
            (0..8)
                .map(|_| {
                    let mut peak = 0.0f32;
                    for _ in 0..window {
                        peak = peak.max(voice.next_sample().abs());
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
                voice.next_sample();
            }
            (0..400).map(|_| voice.next_sample()).collect()
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
                voice.next_sample();
            }
            (0..400).map(|_| voice.next_sample()).collect()
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
                voice.next_sample();
            }
            let n = 400;
            let sum_sq: f32 = (0..n)
                .map(|_| {
                    let s = voice.next_sample();
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
        let early: Vec<f32> = (0..window).map(|_| voice.next_sample()).collect();
        let early_freq = rising_zero_crossing_freq(&early, sample_rate);
        assert!(
            (early_freq - start_freq).abs() < (target_freq - start_freq) * 0.5,
            "shortly after retriggering, pitch should still be close to the start frequency, not the target: \
             early_freq={early_freq} start={start_freq} target={target_freq}"
        );

        // Run out the rest of the glide plus a settling margin, then sample a late window.
        let glide_samples = (glide_seconds * sample_rate) as usize;
        for _ in 0..(glide_samples - window + 500) {
            voice.next_sample();
        }
        let late: Vec<f32> = (0..window).map(|_| voice.next_sample()).collect();
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
    fn render_song_to_wav_produces_expected_length_and_nonsilent_audio() {
        let song = crate::model::Song::demo();
        let sample_rate = 48_000u32;
        let path = std::env::temp_dir().join(format!("simple_daw_test_{}.wav", std::process::id()));

        render_song_to_wav(&song, sample_rate, 1, &path).expect("export should succeed");

        let mut reader = hound::WavReader::open(&path).expect("exported wav should be readable");
        let spec = reader.spec();
        assert_eq!(spec.channels, 1);
        assert_eq!(spec.sample_rate, sample_rate);

        let samples: Vec<i16> = reader
            .samples::<i16>()
            .collect::<std::result::Result<_, _>>()
            .unwrap();
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

    /// A one-track piano-roll song with explicit `regions` on that track, for testing
    /// `Sequencer::process`'s region-lookup logic directly (as opposed to `Song::demo()`'s
    /// implicit single-region shape, which every track always has content for).
    fn song_with_regions(regions: Vec<crate::model::Region>) -> crate::model::Song {
        let mut track = crate::model::Track::new_piano_roll("Lead", 1);
        track.regions = regions;
        crate::model::Song {
            name: "region test".to_string(),
            bpm: 120.0,
            tracks: vec![track],
            next_note_id: 0,
            master_effect_path: String::new(),
            master_effect_params: Vec::new(),
            plugins: Vec::new(),
            synth_presets: Vec::new(),
            time_signature_numerator: 4,
            time_signature_denominator: 4,
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
        }
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
        let samples: Vec<i16> = reader
            .samples::<i16>()
            .collect::<std::result::Result<_, _>>()
            .unwrap();
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
        let samples: Vec<i16> = reader
            .samples::<i16>()
            .collect::<std::result::Result<_, _>>()
            .unwrap();
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
            tracks: vec![track_a, track_b],
            next_note_id: 0,
            master_effect_path: String::new(),
            master_effect_params: Vec::new(),
            plugins: Vec::new(),
            synth_presets: Vec::new(),
            time_signature_numerator: 4,
            time_signature_denominator: 4,
        };

        let mut sequencer = Sequencer::new(48_000.0);
        let mut track_out = Vec::new();
        let mut metronome_out = Vec::new();
        // A few tick's worth of frames, enough for each track's note to trigger and decay somewhat.
        sequencer.process(&song, 4096, &mut track_out, false, &mut metronome_out);

        assert!(
            track_out[0].iter().any(|&s| s != 0.0),
            "track A's own region should sound"
        );
        assert!(
            track_out[1].iter().any(|&s| s != 0.0),
            "track B's own region should sound"
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
            tracks: vec![track],
            next_note_id: 0,
            master_effect_path: String::new(),
            master_effect_params: Vec::new(),
            plugins: Vec::new(),
            synth_presets: Vec::new(),
            time_signature_numerator: 4,
            time_signature_denominator: 4,
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
        let samples: Vec<i16> = reader
            .samples::<i16>()
            .collect::<std::result::Result<_, _>>()
            .unwrap();
        std::fs::remove_file(&path).ok();

        assert!(
            samples[0..1000].iter().any(|&s| s != 0),
            "the clip should be audible right at its start tick"
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
            early_peak = early_peak.max(voice.next_sample(&[]).abs());
        }
        assert!(
            early_peak > 0.05,
            "voice should be clearly audible right after trigger"
        );

        for _ in 0..(sample_rate as usize) {
            voice.next_sample(&[]);
        }
        assert!(
            !voice.active,
            "voice should have decayed to silence within 1 second"
        );
        assert_eq!(
            voice.next_sample(&[]),
            0.0,
            "an inactive voice must output silence"
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
            peak = peak.max(voice.next_sample(&[]).abs());
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
                    let s = voice.next_sample(&[]);
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
            let s = voice.next_sample(&[]);
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
            (0..2000).map(|_| voice.next_sample(mod_slots)).collect()
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
            early_peak = early_peak.max(voice.next_sample(&[]).abs());
        }
        assert!(
            early_peak > 0.05,
            "voice should be clearly audible right after trigger"
        );

        for _ in 0..(sample_rate as usize) {
            voice.next_sample(&[]);
        }
        assert!(
            !voice.active,
            "voice should have decayed to silence within 1 second"
        );
        assert_eq!(
            voice.next_sample(&[]),
            0.0,
            "an inactive voice must output silence"
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
            peak = peak.max(voice.next_sample(&[]).abs());
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
                    let s = voice.next_sample(&[]);
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
                    let s = voice.next_sample(&[]);
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
            (0..2000).map(|_| voice.next_sample(mod_slots)).collect()
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
