use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{BufferSize, FromSample, SampleFormat, SizedSample, Stream, StreamConfig};

use crate::metering::{LoudnessMeter, MeterHandles};
use crate::model::{
    AutomationLane, AutomationTarget, EffectParamKey, FilterRouting, FilterSlope, FilterType, Lane, LfoTarget,
    ModSlot, ModSource, ModTarget, RegionContent, Song, StepData, SynthEngine, SynthParams,
    SynthWaveform, TICKS_PER_STEP, Track, TrackKind, TrackOutput, TrineParams, WaveModSlot,
    WaveModSource, WaveModTarget, WaveParams,
};
use crate::plugin_host::{
    self, MasterEffectSlots, SendEffectSlots, SubmixEffectSlots, TrackEffectSlots,
};
use crate::sample::SampleBuffer;
use crate::wavetable::{self, WaveWarpMode, WavetableId};

/// 16th-note grid: 4 steps per beat.
const STEPS_PER_BEAT: f64 = 4.0;
const VOICE_COUNT: usize = 32;
/// Fixed fade-in/out applied at every `TakeFolder` comp-segment boundary (see `Sequencer::process`'s
/// take-folder trigger loop) so switching which take is heard mid-folder doesn't click — short
/// enough to be inaudible as a fade, long enough to smooth a hard amplitude discontinuity.
const TAKE_FOLDER_CROSSFADE_SECONDS: f32 = 0.005;
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
        master_effects: MasterEffectSlots,
        track_effects: TrackEffectSlots,
        send_effects: SendEffectSlots,
        submix_effects: SubmixEffectSlots,
        track_meters: MeterHandles,
        master_meter: MeterHandles,
        submix_meters: MeterHandles,
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
    /// How far unison voices spread across L/R (0.0..1.0) — see `SynthParams::unison_width`.
    unison_width: f32,
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
    /// bypass and must not blow up at typical device sample rates. Duplicated per channel (`_l`/
    /// `_r`) rather than shared: once `unison_width` spreads osc1's unison voices apart, left and
    /// right feed the filter a genuinely different pre-filter signal, so each channel needs its
    /// own integrator state — sharing one would silently collapse the spread back to mono right
    /// at the filter. At `unison_width == 0.0` both channels' input is identical, so both filter
    /// instances stay identical too (see `next_sample`'s doc comment).
    filter_ic1eq_l: f32,
    filter_ic2eq_l: f32,
    filter_ic1eq_r: f32,
    filter_ic2eq_r: f32,

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
            unison_width: 0.0,
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
            filter_ic1eq_l: 0.0,
            filter_ic2eq_l: 0.0,
            filter_ic1eq_r: 0.0,
            filter_ic2eq_r: 0.0,
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
        self.unison_width = synth.unison_width.clamp(0.0, 1.0);
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
        self.filter_ic1eq_l = 0.0;
        self.filter_ic2eq_l = 0.0;
        self.filter_ic1eq_r = 0.0;
        self.filter_ic2eq_r = 0.0;
    }

    /// Returns `(left, right)`. Osc1's unison voices (if `unison_count > 1`) spread across the
    /// stereo field by `unison_width`; osc2 and the sub oscillator stay centered (no unison
    /// stacking of their own, so nothing to spread). At `unison_width == 0.0` (the default) every
    /// unison voice's pan gain is `(1.0, 1.0)` — both channels get the exact same pre-filter
    /// signal, and since the filter below has no random/audio-independent state, both channels'
    /// filtered output ends up numerically identical too: byte-for-byte the same as this engine's
    /// pre-stereo mono output.
    fn next_sample(&mut self) -> (f32, f32) {
        if !self.active {
            return (0.0, 0.0);
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

        let mut osc1_l = 0.0;
        let mut osc1_r = 0.0;
        let mut osc1_master_wrapped = false;
        for i in 0..self.unison_count {
            self.phase_incs[i] = base_freq * self.unison_ratios[i] / self.sample_rate;
            let sample_i = waveform_sample(self.waveform, self.phases[i], self.pulse_width);
            // Voice 0 sits hard left, the last voice hard right, any middle voice (3-voice
            // unison) stays centered — same layout as `unison_ratios`' detune-down/center/
            // detune-up spread. Scaled by `unison_width` so 0.0 collapses every voice's position
            // to dead center, matching `unison_pan_gains`'s "both channels get gain 1.0" case.
            let position = if self.unison_count <= 1 {
                0.0
            } else {
                (i as f32 / (self.unison_count - 1) as f32) * 2.0 - 1.0
            };
            let (gain_l, gain_r) = unison_pan_gains(position * self.unison_width);
            osc1_l += sample_i * gain_l;
            osc1_r += sample_i * gain_r;
            self.phases[i] += self.phase_incs[i];
            if self.phases[i] >= 1.0 {
                self.phases[i] -= 1.0;
                if i == 0 {
                    osc1_master_wrapped = true;
                }
            }
        }
        osc1_l /= self.unison_count as f32;
        osc1_r /= self.unison_count as f32;

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

        // osc2/sub have no unison stacking of their own (see their struct docs), so they stay
        // centered — identical contribution to both channels.
        let osc2_and_sub = osc2 * self.osc2_mix + sub * self.sub_mix;
        let osc_l = osc1_l * (1.0 - self.osc2_mix) + osc2_and_sub;
        let osc_r = osc1_r * (1.0 - self.osc2_mix) + osc2_and_sub;

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
        let enveloped_l = osc_l * self.amp * amp_lfo;
        let enveloped_r = osc_r * self.amp * amp_lfo;

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
        // Two independent filter instances (own integrator state each, see `filter_ic1eq_l`'s doc
        // comment), sharing the `svf_stage` helper `TrineVoice`/`WaveVoice` already use for their
        // own cascaded filters — reusing it here (rather than hand-duplicating this engine's
        // previously-inlined filter math a second time) is exactly the "duplication introduces
        // higher risk than explicit shared logic" case the project's duplication rule carves out.
        let out_l = svf_stage(
            enveloped_l,
            cutoff,
            self.filter_resonance,
            self.filter_type,
            self.sample_rate,
            &mut self.filter_ic1eq_l,
            &mut self.filter_ic2eq_l,
        );
        let out_r = svf_stage(
            enveloped_r,
            cutoff,
            self.filter_resonance,
            self.filter_type,
            self.sample_rate,
            &mut self.filter_ic1eq_r,
            &mut self.filter_ic2eq_r,
        );
        (out_l, out_r)
    }
}

fn pitch_to_freq(pitch: u8) -> f32 {
    440.0 * 2f32.powf((pitch as f32 - 69.0) / 12.0)
}

/// Per-channel gain for one unison voice at stereo `position` (already scaled by
/// `SynthParams::unison_width`/`WaveParams::unison_width`, so callers pass `position * width`, in
/// -1.0..1.0 — negative pans the voice toward left, positive toward right, 0.0 is dead center).
/// Deliberately not the equal-power law `Track::pan`/`equal_power_pan_gains` uses: that law
/// assumes a single mono source being positioned somewhere in an already-stereo field, and
/// attenuates it (~-3dB) even when centered, to compensate for both ears hearing it. Here, a
/// centered unison voice isn't "positioned in the middle of a stereo signal" — it's simply present
/// in both channels at full strength, the way it always has been before this feature (`spread ==
/// 0.0` must return exactly `(1.0, 1.0)`, not a duller `(0.707, 0.707)`, or every existing preset's
/// loudness would silently change the moment this shipped). As `spread` moves toward +/-1.0, the
/// voice simply fades out of the opposite channel instead of both attenuating symmetrically.
fn unison_pan_gains(spread: f32) -> (f32, f32) {
    let gain_l = (1.0 - spread).clamp(0.0, 1.0);
    let gain_r = (1.0 + spread).clamp(0.0, 1.0);
    (gain_l, gain_r)
}

/// One TPT state-variable filter stage (Zavalishin's "Art of VA Filter Design") — shared by
/// `Voice` (one instance per channel once unison spread diverges L/R), and by `TrineVoice`/
/// `WaveVoice`, which each cascade it once (12dB/octave) or twice (24dB/octave) per `FilterSlope`.
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

/// Runs `WaveVoice`/`TrineVoice`'s shared dual-filter shape — filter1 alone, or filter1 feeding
/// filter2 in series, or filter1/filter2 summed in parallel, per `routing` — for one channel's own
/// integrator state. Factored out as a free function (not a method) so callers can pass `&mut`
/// borrows of individual struct fields (e.g. `filter1_ic1eq_l`) alongside plain copies of the
/// filter parameters, without the whole-`self` borrow a `&self` method receiver would require.
#[allow(clippy::too_many_arguments)]
fn run_dual_filter_stage(
    input: f32,
    routing: FilterRouting,
    filter1_cutoff: f32,
    filter1_resonance: f32,
    filter1_type: FilterType,
    filter1_slope: FilterSlope,
    filter1_ic1eq: &mut [f32; 2],
    filter1_ic2eq: &mut [f32; 2],
    filter2_cutoff: f32,
    filter2_resonance: f32,
    filter2_type: FilterType,
    filter2_slope: FilterSlope,
    filter2_ic1eq: &mut [f32; 2],
    filter2_ic2eq: &mut [f32; 2],
    sample_rate: f32,
) -> f32 {
    let filter1_out = run_filter_stage(
        input,
        filter1_cutoff,
        filter1_resonance,
        filter1_type,
        filter1_slope,
        sample_rate,
        filter1_ic1eq,
        filter1_ic2eq,
    );
    match routing {
        FilterRouting::Off => filter1_out,
        FilterRouting::Series => run_filter_stage(
            filter1_out,
            filter2_cutoff,
            filter2_resonance,
            filter2_type,
            filter2_slope,
            sample_rate,
            filter2_ic1eq,
            filter2_ic2eq,
        ),
        FilterRouting::Parallel => {
            let filter2_out = run_filter_stage(
                input,
                filter2_cutoff,
                filter2_resonance,
                filter2_type,
                filter2_slope,
                sample_rate,
                filter2_ic1eq,
                filter2_ic2eq,
            );
            filter1_out + filter2_out
        }
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

/// Per-channel state for `TrineVoice`'s drift-decorrelated stereo: everything that either carries
/// audio (filter integrators) or depends on this channel's own drift-modulated frequency (the
/// three oscillator phases — once `drift_seed` diverges, `freq` diverges, so each channel's phases
/// accumulate differently over time and can no longer be shared). `drift_seed` starts different
/// per channel (see `TrineVoice::trigger`); everything else starts identical, so at
/// `analog_drift == 0.0` (the default) `drift_lp` never contributes to `freq` and both channels
/// stay numerically identical forever, same as this engine's pre-stereo mono output.
#[derive(Clone, Copy, Default)]
struct TrineVoiceChannel {
    osc1_phase: f32,
    osc2_phase: f32,
    osc3_phase: f32,
    drift_lp: f32,
    drift_seed: u32,
    filter1_ic1eq: [f32; 2],
    filter1_ic2eq: [f32; 2],
    filter2_ic1eq: [f32; 2],
    filter2_ic2eq: [f32; 2],
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
    osc1_level: f32,
    pulse_width: f32,

    osc2_waveform: SynthWaveform,
    osc2_ratio: f32,
    osc2_level: f32,
    osc2_sync: bool,

    osc3_waveform: SynthWaveform,
    osc3_ratio: f32,
    osc3_level: f32,
    osc3_sync: bool,

    fm_amount: f32,
    ring_mod_mix: f32,
    analog_drift: f32,

    filter1_cutoff_hz: f32,
    filter1_resonance: f32,
    filter1_type: FilterType,
    filter1_slope: FilterSlope,
    filter2_cutoff_hz: f32,
    filter2_resonance: f32,
    filter2_type: FilterType,
    filter2_slope: FilterSlope,
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

    left: TrineVoiceChannel,
    right: TrineVoiceChannel,
}

impl Default for TrineVoice {
    fn default() -> Self {
        Self {
            active: false,
            sample_rate: 48_000.0,
            base_freq: 0.0,
            osc1_waveform: SynthWaveform::default(),
            osc1_level: 1.0,
            pulse_width: 0.5,
            osc2_waveform: SynthWaveform::default(),
            osc2_ratio: 1.0,
            osc2_level: 0.0,
            osc2_sync: false,
            osc3_waveform: SynthWaveform::default(),
            osc3_ratio: 1.0,
            osc3_level: 0.0,
            osc3_sync: false,
            fm_amount: 0.0,
            ring_mod_mix: 0.0,
            analog_drift: 0.0,
            filter1_cutoff_hz: 20_000.0,
            filter1_resonance: 0.707,
            filter1_type: FilterType::default(),
            filter1_slope: FilterSlope::default(),
            filter2_cutoff_hz: 20_000.0,
            filter2_resonance: 0.707,
            filter2_type: FilterType::default(),
            filter2_slope: FilterSlope::default(),
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
            left: TrineVoiceChannel::default(),
            right: TrineVoiceChannel::default(),
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
        self.osc1_level = trine.osc1_level.clamp(0.0, 1.0);
        self.pulse_width = trine.pulse_width.clamp(0.02, 0.98);

        self.osc2_waveform = trine.osc2_waveform;
        self.osc2_ratio = 2f32.powf(trine.osc2_semitones as f32 / 12.0)
            * 2f32.powf(trine.osc2_detune_cents / 1200.0);
        self.osc2_level = trine.osc2_level.clamp(0.0, 1.0);
        self.osc2_sync = trine.osc2_sync;

        self.osc3_waveform = trine.osc3_waveform;
        self.osc3_ratio = 2f32.powf(trine.osc3_semitones as f32 / 12.0)
            * 2f32.powf(trine.osc3_detune_cents / 1200.0);
        self.osc3_level = trine.osc3_level.clamp(0.0, 1.0);
        self.osc3_sync = trine.osc3_sync;

        self.fm_amount = trine.fm_amount.max(0.0);
        self.ring_mod_mix = trine.ring_mod_mix.clamp(0.0, 1.0);
        self.analog_drift = trine.analog_drift.max(0.0);
        // Both channels start from the same phases/filter state, diverging only in their drift
        // seed — the two XOR constants are arbitrary but distinct, so left and right decorrelate
        // once `analog_drift > 0.0` (see `TrineVoiceChannel`'s doc comment).
        self.left = TrineVoiceChannel {
            drift_seed: freq.to_bits() ^ 0xA5A5_5A5A,
            ..TrineVoiceChannel::default()
        };
        self.right = TrineVoiceChannel {
            drift_seed: freq.to_bits() ^ 0x5A5A_A5A5,
            ..TrineVoiceChannel::default()
        };

        self.filter1_cutoff_hz = trine.filter1_cutoff_hz.max(20.0);
        self.filter1_resonance = trine.filter1_resonance.max(0.05);
        self.filter1_type = trine.filter1_type;
        self.filter1_slope = trine.filter1_slope;
        self.filter2_cutoff_hz = trine.filter2_cutoff_hz.max(20.0);
        self.filter2_resonance = trine.filter2_resonance.max(0.05);
        self.filter2_type = trine.filter2_type;
        self.filter2_slope = trine.filter2_slope;
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
    /// for why it's a parameter here instead of a field copied in at `trigger` time. Returns
    /// `(left, right)` — identical at `analog_drift == 0.0` (see `TrineVoiceChannel`'s doc
    /// comment), otherwise the two channels' drift decorrelates their frequency (hence their
    /// oscillator phases and everything downstream) over time, run independently via
    /// `trine_channel_sample`.
    fn next_sample(&mut self, mod_slots: &[ModSlot]) -> (f32, f32) {
        if !self.active {
            return (0.0, 0.0);
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

        let pitch_ratio = 2f32.powf(pitch_semitones / 12.0);
        let env3_value = self.env3.advance();

        let shared = TrineSharedInputs {
            base_freq: self.base_freq,
            pitch_ratio,
            analog_drift: self.analog_drift,
            sample_rate: self.sample_rate,
            osc1_waveform: self.osc1_waveform,
            osc2_waveform: self.osc2_waveform,
            osc3_waveform: self.osc3_waveform,
            osc2_ratio: self.osc2_ratio,
            osc3_ratio: self.osc3_ratio,
            osc2_sync: self.osc2_sync,
            osc3_sync: self.osc3_sync,
            pulse_width: (self.pulse_width + pulse_width_delta).clamp(0.02, 0.98),
            osc1_level: (self.osc1_level + osc1_level_delta).clamp(0.0, 1.0),
            osc2_level: (self.osc2_level + osc2_level_delta).clamp(0.0, 1.0),
            osc3_level: (self.osc3_level + osc3_level_delta).clamp(0.0, 1.0),
            fm_amount: (self.fm_amount + fm_amount_delta).max(0.0),
            ring_mod_mix: (self.ring_mod_mix + ring_mod_delta).clamp(0.0, 1.0),
            filter_drive: self.filter_drive,
            env3_value,
            filter1_cutoff_hz: self.filter1_cutoff_hz,
            filter1_cutoff_delta,
            filter_fm_amount: self.filter_fm_amount,
            filter1_resonance: (self.filter1_resonance + filter1_resonance_delta).clamp(0.05, 20.0),
            filter1_type: self.filter1_type,
            filter1_slope: self.filter1_slope,
            filter2_cutoff: (self.filter2_cutoff_hz + filter2_cutoff_delta)
                .clamp(20.0, self.sample_rate * 0.49),
            filter2_resonance: self.filter2_resonance,
            filter2_type: self.filter2_type,
            filter2_slope: self.filter2_slope,
            filter_routing: self.filter_routing,
        };

        let out_l = trine_channel_sample(&mut self.left, &shared);
        let out_r = trine_channel_sample(&mut self.right, &shared);

        // Lifecycle mirrors `Voice`'s: once env3 (the amp envelope) has entered Release and decayed
        // below the floor, this voice is done.
        if self.env3.stage == EnvelopeStage::Release && self.env3.value < ENVELOPE_FLOOR {
            self.active = false;
        }

        (out_l, out_r)
    }
}

/// Read-only, per-sample inputs shared by both of `TrineVoice::next_sample`'s channel runs —
/// everything already resolved (mod-slot deltas applied, clamped) before either channel's own
/// drift/phase/filter state comes into play. Bundled into one struct instead of a long parameter
/// list purely for readability at the two `trine_channel_sample` call sites.
struct TrineSharedInputs {
    base_freq: f32,
    pitch_ratio: f32,
    analog_drift: f32,
    sample_rate: f32,
    osc1_waveform: SynthWaveform,
    osc2_waveform: SynthWaveform,
    osc3_waveform: SynthWaveform,
    osc2_ratio: f32,
    osc3_ratio: f32,
    osc2_sync: bool,
    osc3_sync: bool,
    pulse_width: f32,
    osc1_level: f32,
    osc2_level: f32,
    osc3_level: f32,
    fm_amount: f32,
    ring_mod_mix: f32,
    filter_drive: f32,
    env3_value: f32,
    filter1_cutoff_hz: f32,
    filter1_cutoff_delta: f32,
    filter_fm_amount: f32,
    filter1_resonance: f32,
    filter1_type: FilterType,
    filter1_slope: FilterSlope,
    filter2_cutoff: f32,
    filter2_resonance: f32,
    filter2_type: FilterType,
    filter2_slope: FilterSlope,
    filter_routing: FilterRouting,
}

/// Runs one channel's worth of `TrineVoice::next_sample` — drift, the three oscillators (with FM/
/// ring mod/hard sync), and the dual filter — entirely against `channel`'s own state, using
/// `shared`'s already-resolved parameters. A free function (not a `TrineVoice` method) for the
/// same reason `run_dual_filter_stage` is: it needs `&mut` access to one field (`channel`) of a
/// struct it's called twice on, which a `&self`/`&mut self` method receiver can't express.
fn trine_channel_sample(channel: &mut TrineVoiceChannel, shared: &TrineSharedInputs) -> f32 {
    channel.drift_seed = channel.drift_seed.wrapping_add(0x9E37_79B9);
    let drift_noise = hash_to_bipolar(channel.drift_seed);
    channel.drift_lp += (drift_noise - channel.drift_lp) * TRINE_ANALOG_DRIFT_SMOOTHING;
    let drift_ratio = 2f32.powf(
        shared.analog_drift * channel.drift_lp * TRINE_ANALOG_DRIFT_MAX_CENTS / 1200.0,
    );
    let freq = shared.base_freq * drift_ratio * shared.pitch_ratio;

    let osc1_inc = freq / shared.sample_rate;
    let osc2_inc = freq * shared.osc2_ratio / shared.sample_rate;
    let osc3_inc = freq * shared.osc3_ratio / shared.sample_rate;

    let osc2_raw = waveform_sample(shared.osc2_waveform, channel.osc2_phase, shared.pulse_width);
    let osc1_raw = waveform_sample(shared.osc1_waveform, channel.osc1_phase, shared.pulse_width);

    // FM: osc2's raw sample perturbs osc1's phase increment for this sample only.
    channel.osc1_phase += osc1_inc * (1.0 + shared.fm_amount * osc2_raw);
    let osc1_wrapped = !(0.0..1.0).contains(&channel.osc1_phase);
    if channel.osc1_phase >= 1.0 {
        channel.osc1_phase -= 1.0;
    } else if channel.osc1_phase < 0.0 {
        channel.osc1_phase += 1.0;
    }

    if shared.osc2_sync && osc1_wrapped {
        channel.osc2_phase = 0.0;
    } else {
        channel.osc2_phase += osc2_inc;
        if channel.osc2_phase >= 1.0 {
            channel.osc2_phase -= 1.0;
        }
    }

    let osc3_raw = waveform_sample(shared.osc3_waveform, channel.osc3_phase, shared.pulse_width);
    if shared.osc3_sync && osc1_wrapped {
        channel.osc3_phase = 0.0;
    } else {
        channel.osc3_phase += osc3_inc;
        if channel.osc3_phase >= 1.0 {
            channel.osc3_phase -= 1.0;
        }
    }

    let ring = osc1_raw * osc2_raw * shared.ring_mod_mix;
    let osc_sum =
        osc1_raw * shared.osc1_level + osc2_raw * shared.osc2_level + osc3_raw * shared.osc3_level + ring;

    let driven = if shared.filter_drive > 0.0 {
        (osc_sum * (1.0 + shared.filter_drive * 4.0)).tanh()
    } else {
        osc_sum
    };

    let enveloped = driven * shared.env3_value;

    let filter1_cutoff = (shared.filter1_cutoff_hz
        + shared.filter1_cutoff_delta
        + shared.filter_fm_amount * TRINE_FILTER_FM_RANGE_HZ * osc2_raw)
        .clamp(20.0, shared.sample_rate * 0.49);

    run_dual_filter_stage(
        enveloped,
        shared.filter_routing,
        filter1_cutoff,
        shared.filter1_resonance,
        shared.filter1_type,
        shared.filter1_slope,
        &mut channel.filter1_ic1eq,
        &mut channel.filter1_ic2eq,
        shared.filter2_cutoff,
        shared.filter2_resonance,
        shared.filter2_type,
        shared.filter2_slope,
        &mut channel.filter2_ic1eq,
        &mut channel.filter2_ic2eq,
        shared.sample_rate,
    )
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
    /// How far unison voices spread across L/R (0.0..1.0) — see `WaveParams::unison_width`.
    unison_width: f32,

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
    /// Duplicated per channel (`_l`/`_r`) — see `Voice::filter_ic1eq_l`'s doc comment for why.
    filter1_ic1eq_l: [f32; 2],
    filter1_ic2eq_l: [f32; 2],
    filter1_ic1eq_r: [f32; 2],
    filter1_ic2eq_r: [f32; 2],
    filter2_cutoff_hz: f32,
    filter2_resonance: f32,
    filter2_type: FilterType,
    filter2_slope: FilterSlope,
    filter2_ic1eq_l: [f32; 2],
    filter2_ic2eq_l: [f32; 2],
    filter2_ic1eq_r: [f32; 2],
    filter2_ic2eq_r: [f32; 2],
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
            unison_width: 0.0,
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
            filter1_ic1eq_l: [0.0; 2],
            filter1_ic2eq_l: [0.0; 2],
            filter1_ic1eq_r: [0.0; 2],
            filter1_ic2eq_r: [0.0; 2],
            filter2_cutoff_hz: 20_000.0,
            filter2_resonance: 0.707,
            filter2_type: FilterType::default(),
            filter2_slope: FilterSlope::default(),
            filter2_ic1eq_l: [0.0; 2],
            filter2_ic2eq_l: [0.0; 2],
            filter2_ic1eq_r: [0.0; 2],
            filter2_ic2eq_r: [0.0; 2],
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
        self.unison_width = wave.unison_width.clamp(0.0, 1.0);
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
        self.filter1_ic1eq_l = [0.0; 2];
        self.filter1_ic2eq_l = [0.0; 2];
        self.filter1_ic1eq_r = [0.0; 2];
        self.filter1_ic2eq_r = [0.0; 2];
        self.filter2_cutoff_hz = wave.filter2_cutoff_hz.max(20.0);
        self.filter2_resonance = wave.filter2_resonance.max(0.05);
        self.filter2_type = wave.filter2_type;
        self.filter2_slope = wave.filter2_slope;
        self.filter2_ic1eq_l = [0.0; 2];
        self.filter2_ic2eq_l = [0.0; 2];
        self.filter2_ic1eq_r = [0.0; 2];
        self.filter2_ic2eq_r = [0.0; 2];
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
    /// Returns `(left, right)`, currently identical on both channels — this engine's own stereo
    /// width (unison spread) lands in a follow-up change; for now this signature change is purely
    /// plumbing so the mixing loop and every downstream buffer can already carry real stereo.
    fn next_sample(&mut self, mod_slots: &[WaveModSlot]) -> (f32, f32) {
        if !self.active {
            return (0.0, 0.0);
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

        // Spread osc1's unison voices across L/R the same way `Voice::next_sample` does — see
        // `unison_pan_gains`'s doc comment. osc2/sub/noise have no unison stacking of their own,
        // so they stay centered below.
        let mut osc1_l = 0.0f32;
        let mut osc1_r = 0.0f32;
        for i in 0..self.unison {
            let phase_inc = freq * self.unison_ratios[i] / self.sample_rate;
            let warped =
                wavetable::warp_phase(self.unison_phases[i], self.osc1_warp_mode, osc1_warp_amount);
            let sample_i = wavetable::sample(self.osc1_table, osc1_position, warped, self.osc1_mip);
            let position = if self.unison <= 1 {
                0.0
            } else {
                (i as f32 / (self.unison - 1) as f32) * 2.0 - 1.0
            };
            let (gain_l, gain_r) = unison_pan_gains(position * self.unison_width);
            osc1_l += sample_i * gain_l;
            osc1_r += sample_i * gain_r;
            self.unison_phases[i] += phase_inc;
            if self.unison_phases[i] >= 1.0 {
                self.unison_phases[i] -= 1.0;
            }
        }
        osc1_l /= self.unison as f32;
        osc1_r /= self.unison as f32;

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

        // osc2/sub/noise stay centered (no unison of their own), so their contribution to the mix
        // is identical on both channels — only osc1's already-panned `osc1_l`/`osc1_r` differ.
        let osc2_and_rest =
            osc2_raw * self.osc2_level + sub_raw * self.sub_level + noise_raw * self.noise_level;
        let osc_sum_l = osc1_l * self.osc1_level + osc2_and_rest;
        let osc_sum_r = osc1_r * self.osc1_level + osc2_and_rest;

        let drive = |osc_sum: f32| {
            if self.filter_drive > 0.0 {
                (osc_sum * (1.0 + self.filter_drive * 4.0)).tanh()
            } else {
                osc_sum
            }
        };
        let driven_l = drive(osc_sum_l);
        let driven_r = drive(osc_sum_r);

        let amp_value = self.amp_env.advance();
        let enveloped_l = driven_l * amp_value;
        let enveloped_r = driven_r * amp_value;

        let filter1_cutoff =
            (self.filter1_cutoff_hz + filter1_cutoff_delta).clamp(20.0, self.sample_rate * 0.49);
        let filter1_resonance =
            (self.filter1_resonance + filter1_resonance_delta).clamp(0.05, 20.0);
        let filter2_cutoff =
            (self.filter2_cutoff_hz + filter2_cutoff_delta).clamp(20.0, self.sample_rate * 0.49);

        // Two independent dual-filter chains (own integrator state each, see
        // `filter1_ic1eq_l`'s doc comment) sharing `run_dual_filter_stage` — see that function's
        // doc comment for why it's a free function rather than a `&self` method.
        let output_l = run_dual_filter_stage(
            enveloped_l,
            self.filter_routing,
            filter1_cutoff,
            filter1_resonance,
            self.filter1_type,
            self.filter1_slope,
            &mut self.filter1_ic1eq_l,
            &mut self.filter1_ic2eq_l,
            filter2_cutoff,
            self.filter2_resonance,
            self.filter2_type,
            self.filter2_slope,
            &mut self.filter2_ic1eq_l,
            &mut self.filter2_ic2eq_l,
            self.sample_rate,
        );
        let output_r = run_dual_filter_stage(
            enveloped_r,
            self.filter_routing,
            filter1_cutoff,
            filter1_resonance,
            self.filter1_type,
            self.filter1_slope,
            &mut self.filter1_ic1eq_r,
            &mut self.filter1_ic2eq_r,
            filter2_cutoff,
            self.filter2_resonance,
            self.filter2_type,
            self.filter2_slope,
            &mut self.filter2_ic1eq_r,
            &mut self.filter2_ic2eq_r,
            self.sample_rate,
        );

        // Lifecycle mirrors `TrineVoice`'s: once the amp envelope has entered Release and decayed
        // below the floor, this voice is done.
        if self.amp_env.stage == EnvelopeStage::Release && self.amp_env.value < ENVELOPE_FLOOR {
            self.active = false;
        }

        (output_l, output_r)
    }
}

/// Plays back a pre-resampled one-shot sample from `start_position` to `end_position` (exclusive),
/// with optional linear fade-in/fade-out ramps at the edges — the frame-domain counterpart of
/// `Region::fade_gain_at`, but evaluated per sample rather than per tick since a clip's playback
/// position isn't tick-quantized.
#[derive(Clone, Default)]
struct SampleVoice {
    buffer: Option<Arc<SampleBuffer>>,
    position: usize,
    start_position: usize,
    end_position: usize,
    gain: f32,
    fade_in_frames: usize,
    fade_out_frames: usize,
}

impl SampleVoice {
    /// Plays `buffer` in full, from its own start to its own end, with no fades — used for
    /// velocity-triggered one-shot samples (drum-lane steps), not `AudioClip` playback (see
    /// `trigger_clip`).
    fn trigger(&mut self, buffer: Arc<SampleBuffer>, velocity: u8) {
        let gain = (velocity as f32 / 127.0).clamp(0.0, 1.0);
        self.trigger_clip(buffer, gain, 0, usize::MAX, 0, 0);
    }

    /// Same as `trigger`, but for `AudioClip` playback (see `model::AudioClip`): a continuous gain
    /// instead of a 0..127 velocity byte, plus a trim window (`start_frame..end_frame`, clamped to
    /// the buffer's own length) and fade-in/out ramp lengths in frames — both converted from the
    /// clip's tick-domain fields by the caller (`Sequencer::process`), since only that call site
    /// knows the tempo in effect at the clip's start tick.
    fn trigger_clip(
        &mut self,
        buffer: Arc<SampleBuffer>,
        gain: f32,
        start_frame: usize,
        end_frame: usize,
        fade_in_frames: usize,
        fade_out_frames: usize,
    ) {
        let len = buffer.mono.len();
        self.start_position = start_frame.min(len);
        self.position = self.start_position;
        self.end_position = end_frame.min(len);
        self.buffer = Some(buffer);
        self.gain = gain;
        self.fade_in_frames = fade_in_frames;
        self.fade_out_frames = fade_out_frames;
    }

    fn next_sample(&mut self) -> f32 {
        if self.position >= self.end_position {
            self.buffer = None;
            return 0.0;
        }
        let Some(buffer) = &self.buffer else {
            return 0.0;
        };
        let Some(&s) = buffer.mono.get(self.position) else {
            self.buffer = None;
            return 0.0;
        };
        let elapsed = self.position - self.start_position;
        let remaining = self.end_position - self.position;
        let mut fade = 1.0f32;
        if self.fade_in_frames > 0 {
            fade = fade.min((elapsed as f32 / self.fade_in_frames as f32).clamp(0.0, 1.0));
        }
        if self.fade_out_frames > 0 {
            fade = fade.min((remaining as f32 / self.fade_out_frames as f32).clamp(0.0, 1.0));
        }
        self.position += 1;
        s * self.gain * fade
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

struct Sequencer {
    sample_rate: f32,
    track_voices: Vec<TrackVoices>,
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
            track_fade_gain: Vec::new(),
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

    /// Renders `frames` samples, writing one dry stereo mix per track into `track_out_l[i]`/
    /// `track_out_r[i]` (both resized to match `snapshot.tracks`). Track count can change between
    /// calls (e.g. after loading a different song) — `track_voices` is resized to match,
    /// discarding in-flight voices for any removed track.
    fn process(
        &mut self,
        snapshot: &Song,
        frames: usize,
        track_out_l: &mut Vec<Vec<f32>>,
        track_out_r: &mut Vec<Vec<f32>>,
        metronome_enabled: bool,
        metronome_out: &mut Vec<f32>,
    ) {
        while self.track_voices.len() < snapshot.tracks.len() {
            self.track_voices.push(TrackVoices::new());
        }
        self.track_voices.truncate(snapshot.tracks.len());
        self.track_fade_gain.resize(snapshot.tracks.len(), 1.0);

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
                                        let velocity = &step.velocity;
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
                                            let (engine, synth, trine, wave) = if lane.synth_override
                                            {
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
                                                    let gate_seconds = trine.env3_attack_seconds
                                                        + trine.env3_decay_seconds;
                                                    tv.trine_voices[tv.next_trine_voice].trigger(
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
                                                    let gate_seconds = wave.amp_attack_seconds
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
                            );
                            tv.next_sample_voice = (tv.next_sample_voice + 1) % SAMPLE_VOICE_COUNT;
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
}

/// Samples per sequencer tick at `sample_rate`/`bpm` — the shared clock-rate formula every
/// tick-position calculation in this file (`Sequencer::process`, `render_song_to_wav`, and
/// `build_playback_stream`'s per-sample automation lookups) must agree on exactly, or automation/
/// fades would drift out of sync with what's actually playing.
fn samples_per_tick_at(sample_rate: f64, bpm: f32) -> f64 {
    (sample_rate * 60.0 / (bpm.max(1.0) as f64) / STEPS_PER_BEAT / TICKS_PER_STEP as f64).max(1.0)
}

/// Total sample count spanned by ticks `0..span_ticks` at `song`'s tempo — `song.bpm` alone if
/// `song.tempo_map` is empty (or has no points inside the span), otherwise the sum of each
/// constant-tempo segment's own duration. `render_song_to_wav` uses this instead of one flat
/// `samples_per_tick_at(sample_rate, song.bpm) * span_ticks` so a bounce comes out the right total
/// length even when the tempo changes partway through.
fn samples_for_tick_span(song: &Song, sample_rate: f64, span_ticks: usize) -> f64 {
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
fn step_triggering_at(lane: &Lane, region_local_tick: usize) -> Option<StepData> {
    let floor_step = region_local_tick / TICKS_PER_STEP;
    [floor_step, floor_step + 1].into_iter().find_map(|step_index| {
        let step = (*lane.steps.get(step_index)?)?;
        let target_tick = (step_index * TICKS_PER_STEP) as i64 + step.timing_offset_ticks as i64;
        (target_tick == region_local_tick as i64).then_some(step)
    })
}

/// Equal-power left/right gains for a `Track::pan` value (-1.0 hard left, 0.0 center, 1.0 hard
/// right) — the standard constant-power law (`cos`/`sin` of a quarter-turn sweep) so a centered
/// track doesn't get louder or quieter than a hard-panned one when summed to mono.
fn equal_power_pan_gains(pan: f32) -> (f32, f32) {
    let theta = (pan.clamp(-1.0, 1.0) + 1.0) * std::f32::consts::FRAC_PI_4;
    (theta.cos(), theta.sin())
}

/// One automation lane, paired with the region-local tick offset needed to evaluate it at a given
/// output sample — `AutomationPoint::tick` is relative to the *lane's own* region's `start_tick`
/// (see `AutomationLane`'s doc comment), so a lane collected from `collect_automation` carries its
/// own `base_offset` rather than sharing one with whichever bucket it lands in: an
/// `OtherTrackVolume` lane targeting this track may come from a *different* track's region than
/// this track's own `Volume` lane, each with its own `start_tick`.
#[derive(Clone, Copy)]
struct LaneRef<'a> {
    /// Region-local tick at this buffer's first sample (`buffer_start_tick - region.start_tick`).
    base_offset: f64,
    lane: &'a AutomationLane,
}

impl<'a> LaneRef<'a> {
    /// This lane's value at output sample `sample_index` of the current buffer, sample-accurate
    /// via `AutomationLane::value_at_fractional` — see `TrackAutomationOverride`'s doc comment.
    fn value_at(&self, sample_index: usize, samples_per_tick: f64) -> f32 {
        let tick = self.base_offset + sample_index as f64 / samples_per_tick;
        self.lane
            .value_at_fractional(tick)
            .expect("collect_automation only stores lanes with at least one point")
    }
}

/// One track's automated lanes (if any) for this buffer — not necessarily all *from* that same
/// track's own region: `AutomationTarget::OtherTrack*` lets a lane on one track's region ride a
/// different track's fader/pan/send-level, so this is populated by `collect_automation` scanning
/// every track's active region and bucketing by *target*, not by source. Holds lane references
/// rather than pre-evaluated values so `build_playback_stream` can call `LaneRef::value_at` per
/// output sample instead of holding one value for the whole buffer — the same tick-to-sample-
/// accurate upgrade `Sequencer`'s per-tick `track_fade_gain` already gets, just computed downstream
/// here instead of inside `Sequencer` since these targets (unlike a fade) aren't about already-
/// triggered, freely ringing voices.
#[derive(Default)]
struct TrackAutomationOverride<'a> {
    volume: Option<LaneRef<'a>>,
    pan: Option<LaneRef<'a>>,
    /// (send_index, lane) pairs — only sends this track actually has a `SendLevel`/
    /// `OtherTrackSendLevel` lane for.
    send_levels: Vec<(usize, LaneRef<'a>)>,
    /// (chain slot_index, param key, lane) triples for this track's own effect chain.
    effect_params: Vec<(usize, &'a EffectParamKey, LaneRef<'a>)>,
}

/// The master bus's automated effect-chain params for this buffer — see `TrackAutomationOverride`.
#[derive(Default)]
struct MasterAutomationOverride<'a> {
    /// (chain slot_index, param key, lane) triples for `Song::master_effects`.
    effect_params: Vec<(usize, &'a EffectParamKey, LaneRef<'a>)>,
}

/// One send bus's automated effect-chain params for this buffer — see `TrackAutomationOverride`.
#[derive(Default)]
struct SendAutomationOverride<'a> {
    /// (chain slot_index, param key, lane) triples for this send's own `SendBus::effects`.
    effect_params: Vec<(usize, &'a EffectParamKey, LaneRef<'a>)>,
}

/// The region (if any) on `track` whose on-timeline span currently contains `tick` — the same
/// active-region rule `Sequencer::process`'s trigger loop and `track_fade_gain` use. A region
/// shorter than its own automation lanes' furthest point simply never reaches those points; lanes
/// don't extend a region's span the way `fade_out_ticks` doesn't either.
fn active_region_at(track: &Track, tick: usize) -> Option<&crate::model::Region> {
    track.regions.iter().find(|region| {
        tick >= region.start_tick && tick < region.start_tick + region.loop_length_steps * TICKS_PER_STEP
    })
}

/// Evaluates one automation lane owned (in the source sense — see `TrackAutomationOverride`'s doc
/// comment) by `own_index`, at `base_offset`, into whichever bucket its `AutomationTarget` actually
/// resolves to — most land back on `tracks[own_index]` (the common case), but `OtherTrack*`/
/// `SendEffectParam`/`MasterEffectParam` redirect into a different track's, a send bus's, or the
/// master bus's own bucket instead (see `AutomationTarget`'s doc comment). An out-of-range
/// `track_index`/`send_index` on a redirecting target is silently ignored, same tolerance already
/// extended to overlapping regions elsewhere in this file. Shared body behind `collect_automation`'s
/// two passes (a track's own track-wide lanes, then its active region's lanes) over the same
/// buckets, so a region's lane naturally overrides a track-wide one on the same target via the
/// "later one wins" rule already documented on `collect_automation`.
fn apply_automation_lane<'a>(
    lane: &'a AutomationLane,
    base_offset: f64,
    own_index: usize,
    tracks: &mut [TrackAutomationOverride<'a>],
    master: &mut MasterAutomationOverride<'a>,
    sends: &mut [SendAutomationOverride<'a>],
) {
    if lane.points.is_empty() {
        return;
    }
    let lane_ref = LaneRef { base_offset, lane };
    match &lane.target {
        AutomationTarget::Volume => {
            tracks[own_index].volume = Some(lane_ref);
        }
        AutomationTarget::Pan => {
            tracks[own_index].pan = Some(lane_ref);
        }
        AutomationTarget::SendLevel { send_index } => {
            tracks[own_index].send_levels.push((*send_index, lane_ref));
        }
        AutomationTarget::EffectParam { slot_index, key } => {
            tracks[own_index].effect_params.push((*slot_index, key, lane_ref));
        }
        AutomationTarget::OtherTrackVolume { track_index } => {
            if let Some(t) = tracks.get_mut(*track_index) {
                t.volume = Some(lane_ref);
            }
        }
        AutomationTarget::OtherTrackPan { track_index } => {
            if let Some(t) = tracks.get_mut(*track_index) {
                t.pan = Some(lane_ref);
            }
        }
        AutomationTarget::OtherTrackSendLevel { track_index, send_index } => {
            if let Some(t) = tracks.get_mut(*track_index) {
                t.send_levels.push((*send_index, lane_ref));
            }
        }
        AutomationTarget::OtherTrackEffectParam { track_index, slot_index, key } => {
            if let Some(t) = tracks.get_mut(*track_index) {
                t.effect_params.push((*slot_index, key, lane_ref));
            }
        }
        AutomationTarget::SendEffectParam { send_index, slot_index, key } => {
            if let Some(s) = sends.get_mut(*send_index) {
                s.effect_params.push((*slot_index, key, lane_ref));
            }
        }
        AutomationTarget::MasterEffectParam { slot_index, key } => {
            master.effect_params.push((*slot_index, key, lane_ref));
        }
    }
}

/// Scans every track's own track-wide automation (`Track::automation`, evaluated at the *absolute*
/// `tick`) and then its currently-active region's automation (if any, evaluated region-locally) at
/// `tick` (this buffer's first sample), bucketing every lane by target owner via
/// `apply_automation_lane`. Track-wide lanes are applied first so an active region's own lane on
/// the same target overrides it, matching `Track::automation`'s doc comment — a region is the more
/// specific "clip automation" layer, the track-wide lane is the underlying "track automation"
/// layer it can locally override. A lane with no points yet is skipped entirely, so adding an
/// automation lane in the UI before placing any points doesn't silently zero out that parameter.
fn collect_automation(
    snapshot: &Song,
    tick: usize,
) -> (Vec<TrackAutomationOverride<'_>>, MasterAutomationOverride<'_>, Vec<SendAutomationOverride<'_>>) {
    let mut tracks: Vec<TrackAutomationOverride> =
        (0..snapshot.tracks.len()).map(|_| TrackAutomationOverride::default()).collect();
    let mut master = MasterAutomationOverride::default();
    let mut sends: Vec<SendAutomationOverride> =
        (0..snapshot.sends.len()).map(|_| SendAutomationOverride::default()).collect();

    for (own_index, track) in snapshot.tracks.iter().enumerate() {
        for lane in &track.automation {
            apply_automation_lane(lane, tick as f64, own_index, &mut tracks, &mut master, &mut sends);
        }
        if let Some(region) = active_region_at(track, tick) {
            let base_offset = (tick - region.start_tick) as f64;
            for lane in &region.automation {
                apply_automation_lane(lane, base_offset, own_index, &mut tracks, &mut master, &mut sends);
            }
        }
    }
    (tracks, master, sends)
}

/// Runs `chain` over `dry_l`/`dry_r`, sub-chunking this buffer at every point in `effect_params`
/// (from any of that lane's own points) that falls inside it — a single whole-buffer chunk when
/// nothing here is automated, the common case and the same one whole-buffer call this used to
/// always be before automated effect params existed. Re-applying each chunk's interpolated values
/// before processing it gives CLAP and built-in effects alike a breakpoint-rate approximation of
/// sample-accurate automation without either a plugin-event-timing path or per-effect DSP changes.
/// Shared body behind a track's, a send's, and the master bus's own per-buffer chain processing.
#[allow(clippy::too_many_arguments)]
fn process_chain_with_automation(
    chain: &mut [Option<plugin_host::EffectInstance>],
    effect_params: &[(usize, &EffectParamKey, LaneRef)],
    samples_per_tick: f64,
    dry_l: &[f32],
    dry_r: &[f32],
    out_l: &mut [f32],
    out_r: &mut [f32],
    scratch: &mut [plugin_host::EffectScratch],
    run_l: &mut Vec<f32>,
    run_r: &mut Vec<f32>,
) -> bool {
    let frames = dry_l.len();
    let mut boundaries = vec![0usize];
    for (_, _, lane_ref) in effect_params {
        for point in &lane_ref.lane.points {
            let offset = (point.tick as f64 - lane_ref.base_offset) * samples_per_tick;
            if offset > 0.0 && (offset as usize) < frames {
                boundaries.push(offset as usize);
            }
        }
    }
    boundaries.sort_unstable();
    boundaries.dedup();

    let mut used = false;
    for (chunk_index, &start) in boundaries.iter().enumerate() {
        let end = boundaries.get(chunk_index + 1).copied().unwrap_or(frames);
        if start >= end {
            continue;
        }
        for (slot_index, key, lane_ref) in effect_params {
            let value = lane_ref.value_at(start, samples_per_tick);
            let Some(Some(instance)) = chain.get_mut(*slot_index) else {
                continue;
            };
            match (instance, key) {
                (plugin_host::EffectInstance::Clap(effect), EffectParamKey::Clap { param_id }) => {
                    effect.set_param_by_id(*param_id, value as f64)
                }
                (
                    plugin_host::EffectInstance::BuiltIn(effect),
                    EffectParamKey::BuiltIn { param_name },
                ) => effect.set_automatable_param(param_name, value),
                _ => {}
            }
        }
        used |= plugin_host::process_effect_chain(
            chain,
            &dry_l[start..end],
            &dry_r[start..end],
            &mut out_l[start..end],
            &mut out_r[start..end],
            scratch,
            run_l,
            run_r,
        );
    }
    used
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
                );
                transport
                    .current_tick
                    .store(sequencer.current_tick(), Ordering::Relaxed);

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
                let (track_automation, master_override, send_automation) =
                    collect_automation(snapshot, buffer_start_tick);
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
                        let used = process_chain_with_automation(
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
                        );
                        let source_l = if used { &track_effect_out_l } else { dry_l };
                        let source_r = if used { &track_effect_out_r } else { dry_r };
                        for i in 0..frames {
                            let (pan_l, pan_r) = equal_power_pan_gains(pan_at(i));
                            track_meter_l[i] = volume_at(i) * pan_l * source_l[i];
                            track_meter_r[i] = volume_at(i) * pan_r * source_r[i];
                        }
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
                        );
                        if used {
                            for i in 0..frames {
                                submix_chain_out_l[i] *= volume;
                                submix_chain_out_r[i] *= volume;
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
                        submix_chain_out_l[i] = mix_l[i] * volume;
                        submix_chain_out_r[i] = mix_r[i] * volume;
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

/// Chunk size the offline bounce's mixdown (below) processes wet effect chains in. An offline
/// render has no cpal callback size to inherit, so this is picked directly — large enough to keep
/// CLAP `process()` call overhead low, small enough that automation still updates several times a
/// second even at slow tempos. Synthesis itself (`Sequencer::process`) isn't chunked; only the wet
/// mixdown below is, since that's the part that needs a `chunk_start_tick` to evaluate automation
/// against.
const OFFLINE_CHUNK_FRAMES: usize = 2048;

/// Renders `loops` repetitions of the song's pattern content to a stereo 16-bit WAV file. Shares
/// `Sequencer::process` with real-time playback for synthesis, so a bounce's notes/steps sound
/// like what you'd hear live — but the wet mixdown below (effect chains, automation, sends,
/// submixes) is its own separate, self-contained implementation, deliberately *not* shared with
/// `build_playback_stream`'s live mixdown, so nothing here can regress the real-time audio path;
/// some chunking/effect-application logic is duplicated between the two as a result. Every CLAP
/// plugin is loaded fresh for the duration of this call (`plugin_host::OfflineEffectChain`,
/// distinct from the live, UI-loaded `TrackEffectSlots`) at this bounce's own `sample_rate`, which
/// may differ from any live session's.
///
/// Track/submix mute and solo are not consulted — every track always renders. That's this
/// function's pre-existing behavior (from before this wet mixdown existed), preserved rather than
/// changed as part of unrelated automation/effects work; if "bounces should respect mute/solo"
/// turns out to be wanted, it's a separate, focused change.
pub fn render_song_to_wav(
    song: &Song,
    sample_rate: u32,
    loops: u32,
    path: &std::path::Path,
) -> Result<()> {
    let arrangement_len_ticks = arrangement_length_ticks(song);
    let samples_per_cycle = samples_for_tick_span(song, sample_rate as f64, arrangement_len_ticks);
    let total_samples = (samples_per_cycle * (loops.max(1) as f64)).round() as usize;

    let mut sequencer = Sequencer::new(sample_rate as f32);
    let mut track_dry_l: Vec<Vec<f32>> = Vec::new();
    let mut track_dry_r: Vec<Vec<f32>> = Vec::new();
    let mut metronome_dry: Vec<f32> = Vec::new();
    // The metronome is a monitoring aid, not part of the song — bounces never include it.
    sequencer.process(
        song,
        total_samples,
        &mut track_dry_l,
        &mut track_dry_r,
        false,
        &mut metronome_dry,
    );

    let buffer_l = vec![0.0f32; total_samples];
    let buffer_r = vec![0.0f32; total_samples];
    let (buffer_l, buffer_r) = mix_song_to_wav_buffer(
        song,
        sample_rate,
        &track_dry_l,
        &track_dry_r,
        buffer_l,
        buffer_r,
    );

    let spec = hound::WavSpec {
        channels: 2,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .with_context(|| format!("failed to create wav file: {}", path.display()))?;
    for (l, r) in buffer_l.into_iter().zip(buffer_r) {
        let l = (l.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        let r = (r.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        writer
            .write_sample(l)
            .context("failed to write wav sample")?;
        writer
            .write_sample(r)
            .context("failed to write wav sample")?;
    }
    writer.finalize().context("failed to finalize wav file")?;
    Ok(())
}

/// The wet mixdown behind `render_song_to_wav` — loads a fresh `OfflineEffectChain` per track/
/// send/submix/master, then walks `track_dry_l/r` in `OFFLINE_CHUNK_FRAMES`-sized chunks, each
/// time collecting automation at that chunk's own start tick (`collect_automation`) and running
/// every bus through its chain (`process_chain_with_automation` for the three targets automation
/// can reach — track, send, master; submixes have no automation target defined at all, so a plain
/// `plugin_host::process_effect_chain` there), applying volume/pan/send-level/submix-volume the
/// same way `build_playback_stream`'s live mixdown does. Takes and returns the output buffers by
/// value (rather than `&mut`) purely so `render_song_to_wav` can hand off already-zeroed `Vec`s
/// without an extra explicit zero-fill call.
fn mix_song_to_wav_buffer(
    song: &Song,
    sample_rate: u32,
    track_dry_l: &[Vec<f32>],
    track_dry_r: &[Vec<f32>],
    mut buffer_l: Vec<f32>,
    mut buffer_r: Vec<f32>,
) -> (Vec<f32>, Vec<f32>) {
    let total_samples = buffer_l.len();
    let block = OFFLINE_CHUNK_FRAMES as u32;
    let mut track_chains: Vec<plugin_host::OfflineEffectChain> = song
        .tracks
        .iter()
        .map(|t| plugin_host::load_offline_chain(&t.effects, sample_rate as f64, block))
        .collect();
    let mut send_chains: Vec<plugin_host::OfflineEffectChain> = song
        .sends
        .iter()
        .map(|s| plugin_host::load_offline_chain(&s.effects, sample_rate as f64, block))
        .collect();
    let mut submix_chains: Vec<plugin_host::OfflineEffectChain> = song
        .submixes
        .iter()
        .map(|s| plugin_host::load_offline_chain(&s.effects, sample_rate as f64, block))
        .collect();
    let mut master_chain =
        plugin_host::load_offline_chain(&song.master_effects, sample_rate as f64, block);

    let mut track_scratch: Vec<Vec<plugin_host::EffectScratch>> = track_chains
        .iter()
        .map(|c| c.chain.iter().map(|_| plugin_host::EffectScratch::new()).collect())
        .collect();
    let mut send_scratch: Vec<Vec<plugin_host::EffectScratch>> = send_chains
        .iter()
        .map(|c| c.chain.iter().map(|_| plugin_host::EffectScratch::new()).collect())
        .collect();
    let mut submix_scratch: Vec<Vec<plugin_host::EffectScratch>> = submix_chains
        .iter()
        .map(|c| c.chain.iter().map(|_| plugin_host::EffectScratch::new()).collect())
        .collect();
    let mut master_scratch: Vec<plugin_host::EffectScratch> =
        master_chain.chain.iter().map(|_| plugin_host::EffectScratch::new()).collect();

    // Reused per-chunk scratch, sized to one chunk rather than the whole render, so memory stays
    // bounded regardless of how long the bounce is.
    let mut track_out_l = vec![0.0f32; OFFLINE_CHUNK_FRAMES];
    let mut track_out_r = vec![0.0f32; OFFLINE_CHUNK_FRAMES];
    let mut track_run_l = Vec::with_capacity(OFFLINE_CHUNK_FRAMES);
    let mut track_run_r = Vec::with_capacity(OFFLINE_CHUNK_FRAMES);
    let mut track_meter_l = vec![0.0f32; OFFLINE_CHUNK_FRAMES];
    let mut track_meter_r = vec![0.0f32; OFFLINE_CHUNK_FRAMES];
    let mut send_mix_l: Vec<Vec<f32>> = vec![vec![0.0; OFFLINE_CHUNK_FRAMES]; song.sends.len()];
    let mut send_mix_r: Vec<Vec<f32>> = vec![vec![0.0; OFFLINE_CHUNK_FRAMES]; song.sends.len()];
    let mut send_out_l = vec![0.0f32; OFFLINE_CHUNK_FRAMES];
    let mut send_out_r = vec![0.0f32; OFFLINE_CHUNK_FRAMES];
    let mut send_run_l = Vec::with_capacity(OFFLINE_CHUNK_FRAMES);
    let mut send_run_r = Vec::with_capacity(OFFLINE_CHUNK_FRAMES);
    let mut submix_mix_l: Vec<Vec<f32>> = vec![vec![0.0; OFFLINE_CHUNK_FRAMES]; song.submixes.len()];
    let mut submix_mix_r: Vec<Vec<f32>> = vec![vec![0.0; OFFLINE_CHUNK_FRAMES]; song.submixes.len()];
    let mut submix_out_l = vec![0.0f32; OFFLINE_CHUNK_FRAMES];
    let mut submix_out_r = vec![0.0f32; OFFLINE_CHUNK_FRAMES];
    let mut submix_run_l = Vec::with_capacity(OFFLINE_CHUNK_FRAMES);
    let mut submix_run_r = Vec::with_capacity(OFFLINE_CHUNK_FRAMES);
    let mut master_mix_l = vec![0.0f32; OFFLINE_CHUNK_FRAMES];
    let mut master_mix_r = vec![0.0f32; OFFLINE_CHUNK_FRAMES];
    let mut master_out_l = vec![0.0f32; OFFLINE_CHUNK_FRAMES];
    let mut master_out_r = vec![0.0f32; OFFLINE_CHUNK_FRAMES];
    let mut master_run_l = Vec::with_capacity(OFFLINE_CHUNK_FRAMES);
    let mut master_run_r = Vec::with_capacity(OFFLINE_CHUNK_FRAMES);
    let empty_effect_params = Vec::new();

    let mut chunk_start = 0;
    // Running tick position, advanced below by each chunk's own duration (`chunk_len /
    // samples_per_tick`) rather than derived by dividing `chunk_start` by one fixed rate — the
    // latter would be wrong as soon as a `Song::tempo_map` change makes the tick rate vary across
    // the render, the same reason `Sequencer::process`'s live clock tracks a running tick position
    // instead of computing it from elapsed sample count directly.
    let mut tick_cursor = 0.0f64;
    while chunk_start < total_samples {
        let chunk_len = OFFLINE_CHUNK_FRAMES.min(total_samples - chunk_start);
        let chunk_start_tick = tick_cursor.round() as usize;
        // Resolved once per chunk (like `build_playback_stream`'s buffer-granularity precision,
        // not `Sequencer::process`'s per-tick precision) — held constant through this chunk's
        // `LaneRef::value_at`/`process_chain_with_automation` calls below, which assume one rate
        // per call.
        let samples_per_tick =
            samples_per_tick_at(sample_rate as f64, song.bpm_at(chunk_start_tick));
        let (track_automation, master_automation, send_automation) =
            collect_automation(song, chunk_start_tick);

        for buf in send_mix_l.iter_mut().chain(send_mix_r.iter_mut()) {
            buf[..chunk_len].fill(0.0);
        }
        for buf in submix_mix_l.iter_mut().chain(submix_mix_r.iter_mut()) {
            buf[..chunk_len].fill(0.0);
        }
        master_mix_l[..chunk_len].fill(0.0);
        master_mix_r[..chunk_len].fill(0.0);

        for (track_index, track) in song.tracks.iter().enumerate() {
            let dry_l = &track_dry_l[track_index][chunk_start..chunk_start + chunk_len];
            let dry_r = &track_dry_r[track_index][chunk_start..chunk_start + chunk_len];
            let automation = track_automation.get(track_index);
            let volume_at = |i: usize| {
                automation
                    .and_then(|a| a.volume.map(|lane| lane.value_at(i, samples_per_tick)))
                    .unwrap_or(track.volume)
            };
            let pan_at = |i: usize| {
                automation
                    .and_then(|a| a.pan.map(|lane| lane.value_at(i, samples_per_tick)))
                    .unwrap_or(track.pan)
            };
            let effect_params = automation.map_or(&empty_effect_params, |a| &a.effect_params);
            let used = process_chain_with_automation(
                &mut track_chains[track_index].chain,
                effect_params,
                samples_per_tick,
                dry_l,
                dry_r,
                &mut track_out_l[..chunk_len],
                &mut track_out_r[..chunk_len],
                &mut track_scratch[track_index],
                &mut track_run_l,
                &mut track_run_r,
            );
            let source_l = if used { &track_out_l[..chunk_len] } else { dry_l };
            let source_r = if used { &track_out_r[..chunk_len] } else { dry_r };
            for i in 0..chunk_len {
                let (pan_l, pan_r) = equal_power_pan_gains(pan_at(i));
                track_meter_l[i] = volume_at(i) * pan_l * source_l[i];
                track_meter_r[i] = volume_at(i) * pan_r * source_r[i];
            }
            match track.output {
                TrackOutput::Master => {
                    for i in 0..chunk_len {
                        master_mix_l[i] += track_meter_l[i];
                        master_mix_r[i] += track_meter_r[i];
                    }
                }
                TrackOutput::Submix(index) => {
                    if let (Some(mix_l), Some(mix_r)) =
                        (submix_mix_l.get_mut(index), submix_mix_r.get_mut(index))
                    {
                        for i in 0..chunk_len {
                            mix_l[i] += track_meter_l[i];
                            mix_r[i] += track_meter_r[i];
                        }
                    }
                }
            }
            for (send_index, &static_level) in track.send_levels.iter().enumerate() {
                let lane = automation.and_then(|a| {
                    a.send_levels.iter().find(|(i, _)| *i == send_index).map(|(_, lane)| *lane)
                });
                let Some((mix_l, mix_r)) =
                    send_mix_l.get_mut(send_index).zip(send_mix_r.get_mut(send_index))
                else {
                    continue;
                };
                match lane {
                    Some(lane) => {
                        for i in 0..chunk_len {
                            let level = lane.value_at(i, samples_per_tick);
                            mix_l[i] += track_meter_l[i] * level;
                            mix_r[i] += track_meter_r[i] * level;
                        }
                    }
                    None => {
                        if static_level != 0.0 {
                            for i in 0..chunk_len {
                                mix_l[i] += track_meter_l[i] * static_level;
                                mix_r[i] += track_meter_r[i] * static_level;
                            }
                        }
                    }
                }
            }
        }

        for (send_index, chain) in send_chains.iter_mut().enumerate() {
            let effect_params =
                send_automation.get(send_index).map_or(&empty_effect_params, |s| &s.effect_params);
            let used = process_chain_with_automation(
                &mut chain.chain,
                effect_params,
                samples_per_tick,
                &send_mix_l[send_index][..chunk_len],
                &send_mix_r[send_index][..chunk_len],
                &mut send_out_l[..chunk_len],
                &mut send_out_r[..chunk_len],
                &mut send_scratch[send_index],
                &mut send_run_l,
                &mut send_run_r,
            );
            let (source_l, source_r) = if used {
                (&send_out_l[..chunk_len], &send_out_r[..chunk_len])
            } else {
                (&send_mix_l[send_index][..chunk_len], &send_mix_r[send_index][..chunk_len])
            };
            for i in 0..chunk_len {
                master_mix_l[i] += source_l[i];
                master_mix_r[i] += source_r[i];
            }
        }

        for (submix_index, chain) in submix_chains.iter_mut().enumerate() {
            let volume = song.submixes.get(submix_index).map_or(1.0, |s| s.volume);
            let used = plugin_host::process_effect_chain(
                &mut chain.chain,
                &submix_mix_l[submix_index][..chunk_len],
                &submix_mix_r[submix_index][..chunk_len],
                &mut submix_out_l[..chunk_len],
                &mut submix_out_r[..chunk_len],
                &mut submix_scratch[submix_index],
                &mut submix_run_l,
                &mut submix_run_r,
            );
            let (source_l, source_r) = if used {
                (&submix_out_l[..chunk_len], &submix_out_r[..chunk_len])
            } else {
                (&submix_mix_l[submix_index][..chunk_len], &submix_mix_r[submix_index][..chunk_len])
            };
            for i in 0..chunk_len {
                master_mix_l[i] += source_l[i] * volume;
                master_mix_r[i] += source_r[i] * volume;
            }
        }

        let used = process_chain_with_automation(
            &mut master_chain.chain,
            &master_automation.effect_params,
            samples_per_tick,
            &master_mix_l[..chunk_len],
            &master_mix_r[..chunk_len],
            &mut master_out_l[..chunk_len],
            &mut master_out_r[..chunk_len],
            &mut master_scratch,
            &mut master_run_l,
            &mut master_run_r,
        );
        let (source_l, source_r) = if used {
            (&master_out_l[..chunk_len], &master_out_r[..chunk_len])
        } else {
            (&master_mix_l[..chunk_len], &master_mix_r[..chunk_len])
        };
        for i in 0..chunk_len {
            buffer_l[chunk_start + i] = (source_l[i] * MASTER_GAIN).tanh();
            buffer_r[chunk_start + i] = (source_r[i] * MASTER_GAIN).tanh();
        }

        chunk_start += chunk_len;
        tick_cursor += chunk_len as f64 / samples_per_tick;
    }

    (buffer_l, buffer_r)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pitch_to_freq_matches_concert_a() {
        assert!((pitch_to_freq(69) - 440.0).abs() < 0.001);
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
        voice.trigger_clip(buffer, 1.0, 1, 4, 0, 0);

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
    fn trigger_clip_ramps_fade_in_and_fade_out_linearly() {
        let buffer = Arc::new(SampleBuffer {
            sample_rate: 48_000,
            mono: vec![1.0; 8],
        });
        let mut voice = SampleVoice::default();
        // 8-frame clip, 2-frame fade-in, 2-frame fade-out.
        voice.trigger_clip(buffer, 1.0, 0, 8, 2, 2);

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
                crate::model::AutomationPoint { tick: 0, value: 0.0 },
                crate::model::AutomationPoint { tick: span_ticks, value: 1.0 },
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
        let (tracks, master, sends) = collect_automation(&song, 0);
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
        let (tracks, _, _) = collect_automation(&song, 10);
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
                        AutomationPoint { tick: 0, value: 0.0 },
                        AutomationPoint { tick: 96, value: 1.0 },
                    ],
                },
                AutomationLane {
                    target: AutomationTarget::Pan,
                    points: vec![AutomationPoint { tick: 0, value: -1.0 }],
                },
            ],
        )]);
        let (tracks, _, _) = collect_automation(&song, 48);
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
                points: vec![AutomationPoint { tick: 0, value: 0.25 }],
            }],
        )]);
        // Past the region's on-timeline span (4 steps * TICKS_PER_STEP).
        let (tracks, _, _) = collect_automation(&song, 4 * TICKS_PER_STEP + 1);
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
                    points: vec![AutomationPoint { tick: 0, value: 0.6 }],
                },
                AutomationLane {
                    target: AutomationTarget::EffectParam {
                        slot_index: 0,
                        key: EffectParamKey::BuiltIn { param_name: "Mix".to_string() },
                    },
                    points: vec![AutomationPoint { tick: 0, value: 0.3 }],
                },
            ],
        )]);
        let (tracks, _, _) = collect_automation(&song, 0);
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
                    points: vec![AutomationPoint { tick: 0, value: 0.4 }],
                },
                AutomationLane {
                    target: AutomationTarget::MasterEffectParam {
                        slot_index: 0,
                        key: EffectParamKey::BuiltIn { param_name: "Mix".to_string() },
                    },
                    points: vec![AutomationPoint { tick: 0, value: 0.8 }],
                },
            ],
        )]);
        // A second track, with no automation of its own, to be the redirect target.
        song.tracks.push(crate::model::Track::new_piano_roll("Other", 1));

        let (tracks, master, _) = collect_automation(&song, 0);
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
                AutomationPoint { tick: 0, value: 0.2 },
                AutomationPoint { tick: 100, value: 1.0 },
            ],
        });
        // No region at all, let alone one active at tick 50 — the track-wide lane still applies,
        // evaluated at the absolute tick (unlike a region lane, not offset by any region start).
        let (tracks, _, _) = collect_automation(&song, 50);
        assert_eq!(tracks[0].volume.unwrap().value_at(0, 1.0), 0.6);
    }

    #[test]
    fn collect_automation_lets_an_active_regions_lane_override_a_track_wide_lane() {
        use crate::model::{AutomationLane, AutomationPoint, AutomationTarget};
        let mut song = song_with_regions(vec![region_with_automation(
            4,
            vec![AutomationLane {
                target: AutomationTarget::Volume,
                points: vec![AutomationPoint { tick: 0, value: 0.9 }],
            }],
        )]);
        song.tracks[0].automation.push(AutomationLane {
            target: AutomationTarget::Volume,
            points: vec![AutomationPoint { tick: 0, value: 0.1 }],
        });
        // Tick 0 is inside the region's on-timeline span — its lane should win over the
        // track-wide one on the same target (Volume), per `Track::automation`'s doc comment.
        let (tracks, _, _) = collect_automation(&song, 0);
        assert_eq!(tracks[0].volume.unwrap().value_at(0, 1.0), 0.9);
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
            muted: true,
            solo: false,
        }];

        let mut sequencer = Sequencer::new(48_000.0);
        let mut track_out_l = Vec::new();
        let mut track_out_r = Vec::new();
        let mut metronome_out = Vec::new();
        sequencer.process(&song, 4096, &mut track_out_l, &mut track_out_r, false, &mut metronome_out);

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
            muted: false,
            solo: true,
        }];

        let mut sequencer = Sequencer::new(48_000.0);
        let mut track_out_l = Vec::new();
        let mut track_out_r = Vec::new();
        let mut metronome_out = Vec::new();
        sequencer.process(&song, 4096, &mut track_out_l, &mut track_out_r, false, &mut metronome_out);

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
