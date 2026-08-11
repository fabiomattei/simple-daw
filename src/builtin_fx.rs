//! Native DSP effects that need no external plugin file, as an alternative to CLAP hosting
//! (`plugin_host.rs`) in a track's insert chain. Each effect is a plain Rust struct with public
//! parameter fields (so the UI can bind sliders straight to them, the same way `LoadedEffect`'s
//! CLAP params are edited) plus whatever private DSP state it needs, and a `process` method that
//! runs it in place over one mono audio block.

use crate::model::{FilterMode, TrackEffectConfig};

/// Longest delay time the UI allows (see `main.rs`'s slider range) — the ring buffer is sized for
/// this once at creation so changing `time_ms` at runtime never needs a reallocation.
const MAX_DELAY_SECONDS: f32 = 2.0;

pub struct DelayEffect {
    pub time_ms: f32,
    pub feedback: f32,
    pub mix: f32,
    buffer: Vec<f32>,
    write_pos: usize,
    sample_rate: f32,
}

impl DelayEffect {
    fn new(time_ms: f32, feedback: f32, mix: f32, sample_rate: f32) -> Self {
        let buffer_len = ((MAX_DELAY_SECONDS * sample_rate) as usize).max(1);
        Self {
            time_ms,
            feedback,
            mix,
            buffer: vec![0.0; buffer_len],
            write_pos: 0,
            sample_rate,
        }
    }

    fn process(&mut self, buf: &mut [f32]) {
        let buffer_len = self.buffer.len();
        let delay_samples = ((self.time_ms.max(0.0) / 1000.0) * self.sample_rate) as usize;
        let delay_samples = delay_samples.clamp(1, buffer_len - 1);
        let feedback = self.feedback.clamp(0.0, 0.98);
        let mix = self.mix.clamp(0.0, 1.0);
        for sample in buf.iter_mut() {
            let read_pos = (self.write_pos + buffer_len - delay_samples) % buffer_len;
            let delayed = self.buffer[read_pos];
            let input = *sample;
            self.buffer[self.write_pos] = input + delayed * feedback;
            self.write_pos = (self.write_pos + 1) % buffer_len;
            *sample = input * (1.0 - mix) + delayed * mix;
        }
    }
}

pub struct BitcrusherEffect {
    pub bit_depth: f32,
    pub rate_divisor: u32,
    pub mix: f32,
    hold_value: f32,
    hold_counter: u32,
}

impl BitcrusherEffect {
    fn new(bit_depth: f32, rate_divisor: u32, mix: f32) -> Self {
        Self {
            bit_depth,
            rate_divisor,
            mix,
            hold_value: 0.0,
            hold_counter: 0,
        }
    }

    fn process(&mut self, buf: &mut [f32]) {
        let divisor = self.rate_divisor.max(1);
        let levels = 2f32.powf(self.bit_depth.clamp(1.0, 16.0));
        let mix = self.mix.clamp(0.0, 1.0);
        for sample in buf.iter_mut() {
            if self.hold_counter == 0 {
                self.hold_value = *sample;
                self.hold_counter = divisor;
            }
            self.hold_counter -= 1;
            let crushed = (self.hold_value * levels).round() / levels;
            *sample = *sample * (1.0 - mix) + crushed * mix;
        }
    }
}

pub struct DistortionEffect {
    pub drive: f32,
    pub mix: f32,
}

impl DistortionEffect {
    fn new(drive: f32, mix: f32) -> Self {
        Self { drive, mix }
    }

    fn process(&mut self, buf: &mut [f32]) {
        let drive = self.drive.max(1.0);
        let mix = self.mix.clamp(0.0, 1.0);
        for sample in buf.iter_mut() {
            let wet = (*sample * drive).tanh();
            *sample = *sample * (1.0 - mix) + wet * mix;
        }
    }
}

/// One tap of a Schroeder comb filter: a delay line with feedback plus a one-pole lowpass in the
/// feedback path (the standard Freeverb-style trick for damping high frequencies as the tail
/// decays, instead of ringing forever at a flat frequency response).
struct CombFilter {
    buffer: Vec<f32>,
    pos: usize,
    filter_store: f32,
}

impl CombFilter {
    fn new(len_samples: usize) -> Self {
        Self {
            buffer: vec![0.0; len_samples.max(1)],
            pos: 0,
            filter_store: 0.0,
        }
    }

    fn process(&mut self, input: f32, feedback: f32, damping: f32) -> f32 {
        let output = self.buffer[self.pos];
        self.filter_store = output * (1.0 - damping) + self.filter_store * damping;
        self.buffer[self.pos] = input + self.filter_store * feedback;
        self.pos = (self.pos + 1) % self.buffer.len();
        output
    }
}

/// A Schroeder allpass filter: diffuses the comb filters' output into a denser, less metallic
/// tail without adding its own coloration (flat frequency response by construction).
struct AllpassFilter {
    buffer: Vec<f32>,
    pos: usize,
}

impl AllpassFilter {
    fn new(len_samples: usize) -> Self {
        Self {
            buffer: vec![0.0; len_samples.max(1)],
            pos: 0,
        }
    }

    fn process(&mut self, input: f32) -> f32 {
        const FEEDBACK: f32 = 0.5;
        let buffered = self.buffer[self.pos];
        let output = buffered - input;
        self.buffer[self.pos] = input + buffered * FEEDBACK;
        self.pos = (self.pos + 1) % self.buffer.len();
        output
    }
}

/// Comb filter delay lengths in milliseconds, tuned (mutually prime-ish, spread out) the way
/// Freeverb's classic 44.1kHz sample counts are — scaled to the actual sample rate at creation
/// time rather than hardcoded as sample counts, so the same character holds at any device rate.
const COMB_TUNINGS_MS: [f32; 4] = [25.31, 26.94, 28.96, 30.75];
const ALLPASS_TUNINGS_MS: [f32; 2] = [12.61, 9.68];

pub struct ReverbEffect {
    pub room_size: f32,
    pub damping: f32,
    pub mix: f32,
    combs: Vec<CombFilter>,
    allpasses: Vec<AllpassFilter>,
}

impl ReverbEffect {
    fn new(room_size: f32, damping: f32, mix: f32, sample_rate: f32) -> Self {
        let combs = COMB_TUNINGS_MS
            .iter()
            .map(|ms| CombFilter::new(((ms / 1000.0) * sample_rate) as usize))
            .collect();
        let allpasses = ALLPASS_TUNINGS_MS
            .iter()
            .map(|ms| AllpassFilter::new(((ms / 1000.0) * sample_rate) as usize))
            .collect();
        Self {
            room_size,
            damping,
            mix,
            combs,
            allpasses,
        }
    }

    fn process(&mut self, buf: &mut [f32]) {
        // room_size maps to comb feedback: higher feedback means a longer-ringing tail.
        let feedback = 0.7 + self.room_size.clamp(0.0, 1.0) * 0.28;
        let damping = self.damping.clamp(0.0, 1.0);
        let mix = self.mix.clamp(0.0, 1.0);
        for sample in buf.iter_mut() {
            let input = *sample;
            let mut wet = 0.0;
            for comb in self.combs.iter_mut() {
                wet += comb.process(input, feedback, damping);
            }
            wet /= self.combs.len() as f32;
            for allpass in self.allpasses.iter_mut() {
                wet = allpass.process(wet);
            }
            *sample = input * (1.0 - mix) + wet * mix;
        }
    }
}

/// Longest modulation excursion the UI allows for chorus depth — the delay line is sized for
/// this once at creation, same rationale as `DelayEffect`'s `MAX_DELAY_SECONDS`.
const MAX_CHORUS_DELAY_MS: f32 = 30.0;

pub struct ChorusEffect {
    pub rate_hz: f32,
    pub depth_ms: f32,
    pub mix: f32,
    buffer: Vec<f32>,
    write_pos: usize,
    lfo_phase: f32,
    sample_rate: f32,
}

impl ChorusEffect {
    fn new(rate_hz: f32, depth_ms: f32, mix: f32, sample_rate: f32) -> Self {
        let buffer_len = ((MAX_CHORUS_DELAY_MS / 1000.0 * sample_rate) as usize).max(4);
        Self {
            rate_hz,
            depth_ms,
            mix,
            buffer: vec![0.0; buffer_len],
            write_pos: 0,
            lfo_phase: 0.0,
            sample_rate,
        }
    }

    fn process(&mut self, buf: &mut [f32]) {
        let buffer_len = self.buffer.len();
        let depth_samples =
            (self.depth_ms.max(0.0) / 1000.0 * self.sample_rate).min((buffer_len - 2) as f32);
        let mix = self.mix.clamp(0.0, 1.0);
        let phase_inc = 2.0 * std::f32::consts::PI * self.rate_hz.max(0.0) / self.sample_rate;
        for sample in buf.iter_mut() {
            let input = *sample;
            self.buffer[self.write_pos] = input;
            // The LFO swings the read head between 0 and `depth_samples` behind the write head, so
            // the delayed copy's pitch continuously drifts (the classic chorus detune character).
            let lfo = 0.5 * (1.0 + self.lfo_phase.sin());
            let delay_samples = (depth_samples * lfo).max(1.0);
            let mut read_pos = self.write_pos as f32 - delay_samples;
            if read_pos < 0.0 {
                read_pos += buffer_len as f32;
            }
            let idx0 = read_pos as usize % buffer_len;
            let idx1 = (idx0 + 1) % buffer_len;
            let frac = read_pos.fract();
            let delayed = self.buffer[idx0] * (1.0 - frac) + self.buffer[idx1] * frac;
            self.write_pos = (self.write_pos + 1) % buffer_len;
            self.lfo_phase += phase_inc;
            if self.lfo_phase > 2.0 * std::f32::consts::PI {
                self.lfo_phase -= 2.0 * std::f32::consts::PI;
            }
            *sample = input * (1.0 - mix) + delayed * mix;
        }
    }
}

/// A Chamberlin state-variable filter run in low-pass or high-pass mode. Cheap (no biquad
/// coefficient recompute needed per block) and stable enough for the cutoff/resonance ranges the
/// UI exposes.
pub struct FilterEffect {
    pub cutoff_hz: f32,
    pub resonance: f32,
    pub mode: FilterMode,
    pub mix: f32,
    low: f32,
    band: f32,
    sample_rate: f32,
}

impl FilterEffect {
    fn new(cutoff_hz: f32, resonance: f32, mode: FilterMode, mix: f32, sample_rate: f32) -> Self {
        Self {
            cutoff_hz,
            resonance,
            mode,
            mix,
            low: 0.0,
            band: 0.0,
            sample_rate,
        }
    }

    fn process(&mut self, buf: &mut [f32]) {
        let nyquist = self.sample_rate * 0.5;
        let cutoff = self.cutoff_hz.clamp(20.0, nyquist - 100.0);
        let f = 2.0 * (std::f32::consts::PI * cutoff / self.sample_rate).sin();
        // resonance in 0..1 from the UI maps to the SVF's damping (q): higher resonance means
        // lower damping, i.e. more feedback ringing at the cutoff frequency.
        let q = (1.0 - self.resonance.clamp(0.0, 0.99)).max(0.01) * 2.0;
        let mix = self.mix.clamp(0.0, 1.0);
        for sample in buf.iter_mut() {
            let input = *sample;
            let high = input - self.low - q * self.band;
            self.band += f * high;
            self.low += f * self.band;
            let output = match self.mode {
                FilterMode::LowPass => self.low,
                FilterMode::HighPass => high,
            };
            *sample = input * (1.0 - mix) + output * mix;
        }
    }
}

pub struct TremoloEffect {
    pub rate_hz: f32,
    pub depth: f32,
    phase: f32,
    sample_rate: f32,
}

impl TremoloEffect {
    fn new(rate_hz: f32, depth: f32, sample_rate: f32) -> Self {
        Self {
            rate_hz,
            depth,
            phase: 0.0,
            sample_rate,
        }
    }

    fn process(&mut self, buf: &mut [f32]) {
        let depth = self.depth.clamp(0.0, 1.0);
        let phase_inc = 2.0 * std::f32::consts::PI * self.rate_hz.max(0.0) / self.sample_rate;
        for sample in buf.iter_mut() {
            let lfo = 0.5 * (1.0 + self.phase.sin());
            let gain = 1.0 - depth * (1.0 - lfo);
            *sample *= gain;
            self.phase += phase_inc;
            if self.phase > 2.0 * std::f32::consts::PI {
                self.phase -= 2.0 * std::f32::consts::PI;
            }
        }
    }
}

/// Feedforward dynamics compressor: a one-pole peak envelope follower (separate attack/release
/// time constants) feeding a dB-domain gain computer, plus a static makeup gain.
pub struct CompressorEffect {
    pub threshold_db: f32,
    pub ratio: f32,
    pub attack_ms: f32,
    pub release_ms: f32,
    pub makeup_db: f32,
    envelope: f32,
    sample_rate: f32,
}

impl CompressorEffect {
    fn new(
        threshold_db: f32,
        ratio: f32,
        attack_ms: f32,
        release_ms: f32,
        makeup_db: f32,
        sample_rate: f32,
    ) -> Self {
        Self {
            threshold_db,
            ratio,
            attack_ms,
            release_ms,
            makeup_db,
            envelope: 0.0,
            sample_rate,
        }
    }

    fn time_coeff(time_ms: f32, sample_rate: f32) -> f32 {
        (-1.0 / (time_ms.max(0.1) / 1000.0 * sample_rate)).exp()
    }

    fn process(&mut self, buf: &mut [f32]) {
        let ratio = self.ratio.max(1.0);
        let threshold = self.threshold_db;
        let makeup = 10f32.powf(self.makeup_db / 20.0);
        let attack_coeff = Self::time_coeff(self.attack_ms, self.sample_rate);
        let release_coeff = Self::time_coeff(self.release_ms, self.sample_rate);
        for sample in buf.iter_mut() {
            let input = *sample;
            let rectified = input.abs();
            let coeff = if rectified > self.envelope {
                attack_coeff
            } else {
                release_coeff
            };
            self.envelope = coeff * self.envelope + (1.0 - coeff) * rectified;
            let env_db = 20.0 * self.envelope.max(1e-6).log10();
            let gain_db = if env_db > threshold {
                (threshold + (env_db - threshold) / ratio) - env_db
            } else {
                0.0
            };
            let gain = 10f32.powf(gain_db / 20.0) * makeup;
            *sample = input * gain;
        }
    }
}

/// Longest delay range a flanger's LFO sweeps — much shorter than chorus's, which combined with
/// the feedback path below is what gives a flanger its more resonant, metallic sweep.
const MAX_FLANGER_DELAY_MS: f32 = 10.0;

pub struct FlangerEffect {
    pub rate_hz: f32,
    pub depth_ms: f32,
    pub feedback: f32,
    pub mix: f32,
    buffer: Vec<f32>,
    write_pos: usize,
    lfo_phase: f32,
    sample_rate: f32,
}

impl FlangerEffect {
    fn new(rate_hz: f32, depth_ms: f32, feedback: f32, mix: f32, sample_rate: f32) -> Self {
        let buffer_len = ((MAX_FLANGER_DELAY_MS / 1000.0 * sample_rate) as usize).max(4);
        Self {
            rate_hz,
            depth_ms,
            feedback,
            mix,
            buffer: vec![0.0; buffer_len],
            write_pos: 0,
            lfo_phase: 0.0,
            sample_rate,
        }
    }

    fn process(&mut self, buf: &mut [f32]) {
        let buffer_len = self.buffer.len();
        let depth_samples =
            (self.depth_ms.max(0.0) / 1000.0 * self.sample_rate).min((buffer_len - 2) as f32);
        let feedback = self.feedback.clamp(0.0, 0.95);
        let mix = self.mix.clamp(0.0, 1.0);
        let phase_inc = 2.0 * std::f32::consts::PI * self.rate_hz.max(0.0) / self.sample_rate;
        for sample in buf.iter_mut() {
            let input = *sample;
            let lfo = 0.5 * (1.0 + self.lfo_phase.sin());
            let delay_samples = (depth_samples * lfo).max(1.0);
            let mut read_pos = self.write_pos as f32 - delay_samples;
            if read_pos < 0.0 {
                read_pos += buffer_len as f32;
            }
            let idx0 = read_pos as usize % buffer_len;
            let idx1 = (idx0 + 1) % buffer_len;
            let frac = read_pos.fract();
            let delayed = self.buffer[idx0] * (1.0 - frac) + self.buffer[idx1] * frac;
            // Unlike chorus, the delayed (wet) signal feeds back into the buffer, not just the
            // dry input — this is what gives a flanger its resonant comb-filter character.
            self.buffer[self.write_pos] = input + delayed * feedback;
            self.write_pos = (self.write_pos + 1) % buffer_len;
            self.lfo_phase += phase_inc;
            if self.lfo_phase > 2.0 * std::f32::consts::PI {
                self.lfo_phase -= 2.0 * std::f32::consts::PI;
            }
            *sample = input * (1.0 - mix) + delayed * mix;
        }
    }
}

/// One first-order allpass stage of a phaser, with a time-varying coefficient driven by the
/// shared LFO in `PhaserEffect::process`.
struct PhaserAllpassStage {
    x1: f32,
    y1: f32,
}

impl PhaserAllpassStage {
    fn new() -> Self {
        Self { x1: 0.0, y1: 0.0 }
    }

    fn process(&mut self, input: f32, coeff: f32) -> f32 {
        let output = -coeff * input + self.x1 + coeff * self.y1;
        self.x1 = input;
        self.y1 = output;
        output
    }
}

const PHASER_STAGE_COUNT: usize = 4;

pub struct PhaserEffect {
    pub rate_hz: f32,
    pub depth: f32,
    pub feedback: f32,
    pub mix: f32,
    stages: Vec<PhaserAllpassStage>,
    lfo_phase: f32,
    feedback_sample: f32,
    sample_rate: f32,
}

impl PhaserEffect {
    fn new(rate_hz: f32, depth: f32, feedback: f32, mix: f32, sample_rate: f32) -> Self {
        Self {
            rate_hz,
            depth,
            feedback,
            mix,
            stages: (0..PHASER_STAGE_COUNT)
                .map(|_| PhaserAllpassStage::new())
                .collect(),
            lfo_phase: 0.0,
            feedback_sample: 0.0,
            sample_rate,
        }
    }

    fn process(&mut self, buf: &mut [f32]) {
        let depth = self.depth.clamp(0.0, 1.0);
        let feedback = self.feedback.clamp(0.0, 0.95);
        let mix = self.mix.clamp(0.0, 1.0);
        let phase_inc = 2.0 * std::f32::consts::PI * self.rate_hz.max(0.0) / self.sample_rate;
        for sample in buf.iter_mut() {
            let input = *sample;
            let lfo = 0.5 * (1.0 + self.lfo_phase.sin());
            // Keep the allpass coefficient well inside (-1, 1) at all times so every stage stays
            // stable across the full sweep.
            let coeff = -0.9 + depth * 1.8 * lfo;
            let mut signal = input + self.feedback_sample * feedback;
            for stage in self.stages.iter_mut() {
                signal = stage.process(signal, coeff);
            }
            self.feedback_sample = signal;
            self.lfo_phase += phase_inc;
            if self.lfo_phase > 2.0 * std::f32::consts::PI {
                self.lfo_phase -= 2.0 * std::f32::consts::PI;
            }
            *sample = input * (1.0 - mix) + signal * mix;
        }
    }
}

pub struct RingModulatorEffect {
    pub carrier_hz: f32,
    pub mix: f32,
    phase: f32,
    sample_rate: f32,
}

impl RingModulatorEffect {
    fn new(carrier_hz: f32, mix: f32, sample_rate: f32) -> Self {
        Self {
            carrier_hz,
            mix,
            phase: 0.0,
            sample_rate,
        }
    }

    fn process(&mut self, buf: &mut [f32]) {
        let mix = self.mix.clamp(0.0, 1.0);
        let phase_inc = 2.0 * std::f32::consts::PI * self.carrier_hz.max(0.0) / self.sample_rate;
        for sample in buf.iter_mut() {
            let input = *sample;
            let modulated = input * self.phase.sin();
            *sample = input * (1.0 - mix) + modulated * mix;
            self.phase += phase_inc;
            if self.phase > 2.0 * std::f32::consts::PI {
                self.phase -= 2.0 * std::f32::consts::PI;
            }
        }
    }
}

/// A noise gate: an envelope follower decides whether the signal is above `threshold_db`, and a
/// separately-smoothed gain (same attack/release time constants) is applied so it doesn't click
/// open/closed. `range_db` is how far down the gate attenuates when closed — a large negative
/// number (e.g. -60dB) rather than hard-muting to exact silence, matching how real noise gates
/// expose a "range" control instead of just on/off.
pub struct NoiseGateEffect {
    pub threshold_db: f32,
    pub attack_ms: f32,
    pub release_ms: f32,
    pub range_db: f32,
    envelope: f32,
    gain: f32,
    sample_rate: f32,
}

impl NoiseGateEffect {
    fn new(
        threshold_db: f32,
        attack_ms: f32,
        release_ms: f32,
        range_db: f32,
        sample_rate: f32,
    ) -> Self {
        Self {
            threshold_db,
            attack_ms,
            release_ms,
            range_db,
            envelope: 0.0,
            gain: 0.0,
            sample_rate,
        }
    }

    fn time_coeff(time_ms: f32, sample_rate: f32) -> f32 {
        (-1.0 / (time_ms.max(0.1) / 1000.0 * sample_rate)).exp()
    }

    fn process(&mut self, buf: &mut [f32]) {
        let attack_coeff = Self::time_coeff(self.attack_ms, self.sample_rate);
        let release_coeff = Self::time_coeff(self.release_ms, self.sample_rate);
        let closed_gain = 10f32.powf(self.range_db / 20.0);
        for sample in buf.iter_mut() {
            let input = *sample;
            let rectified = input.abs();
            let env_coeff = if rectified > self.envelope {
                attack_coeff
            } else {
                release_coeff
            };
            self.envelope = env_coeff * self.envelope + (1.0 - env_coeff) * rectified;
            let env_db = 20.0 * self.envelope.max(1e-6).log10();
            let target_gain = if env_db > self.threshold_db {
                1.0
            } else {
                closed_gain
            };
            let gain_coeff = if target_gain > self.gain {
                attack_coeff
            } else {
                release_coeff
            };
            self.gain = gain_coeff * self.gain + (1.0 - gain_coeff) * target_gain;
            *sample = input * self.gain;
        }
    }
}

/// A live, running built-in effect, owning whatever DSP state it needs (ring buffers, filter
/// state) across successive `process` calls. Mirrors `plugin_host::LoadedEffect`'s role for CLAP
/// effects, but construction is infallible (given a sample rate) since there's no external file
/// or plugin activation involved.
pub enum BuiltInEffect {
    Delay(DelayEffect),
    Bitcrusher(BitcrusherEffect),
    Distortion(DistortionEffect),
    Reverb(ReverbEffect),
    Chorus(ChorusEffect),
    Filter(FilterEffect),
    Tremolo(TremoloEffect),
    Compressor(CompressorEffect),
    Flanger(FlangerEffect),
    Phaser(PhaserEffect),
    RingModulator(RingModulatorEffect),
    NoiseGate(NoiseGateEffect),
}

impl BuiltInEffect {
    /// Builds a live effect from its saved/edited parameters. `None` only if `config` is actually
    /// `TrackEffectConfig::Clap` — callers are expected to route CLAP configs through
    /// `plugin_host::load_and_activate` instead; this is a defensive fallback, not a normal path.
    pub fn from_config(config: &TrackEffectConfig, sample_rate: f32) -> Option<Self> {
        Some(match config {
            TrackEffectConfig::Delay {
                time_ms,
                feedback,
                mix,
            } => BuiltInEffect::Delay(DelayEffect::new(*time_ms, *feedback, *mix, sample_rate)),
            TrackEffectConfig::Bitcrusher {
                bit_depth,
                rate_divisor,
                mix,
            } => BuiltInEffect::Bitcrusher(BitcrusherEffect::new(*bit_depth, *rate_divisor, *mix)),
            TrackEffectConfig::Distortion { drive, mix } => {
                BuiltInEffect::Distortion(DistortionEffect::new(*drive, *mix))
            }
            TrackEffectConfig::Reverb {
                room_size,
                damping,
                mix,
            } => BuiltInEffect::Reverb(ReverbEffect::new(*room_size, *damping, *mix, sample_rate)),
            TrackEffectConfig::Chorus {
                rate_hz,
                depth_ms,
                mix,
            } => BuiltInEffect::Chorus(ChorusEffect::new(*rate_hz, *depth_ms, *mix, sample_rate)),
            TrackEffectConfig::Filter {
                cutoff_hz,
                resonance,
                mode,
                mix,
            } => BuiltInEffect::Filter(FilterEffect::new(
                *cutoff_hz,
                *resonance,
                *mode,
                *mix,
                sample_rate,
            )),
            TrackEffectConfig::Tremolo { rate_hz, depth } => {
                BuiltInEffect::Tremolo(TremoloEffect::new(*rate_hz, *depth, sample_rate))
            }
            TrackEffectConfig::Compressor {
                threshold_db,
                ratio,
                attack_ms,
                release_ms,
                makeup_db,
            } => BuiltInEffect::Compressor(CompressorEffect::new(
                *threshold_db,
                *ratio,
                *attack_ms,
                *release_ms,
                *makeup_db,
                sample_rate,
            )),
            TrackEffectConfig::Flanger {
                rate_hz,
                depth_ms,
                feedback,
                mix,
            } => BuiltInEffect::Flanger(FlangerEffect::new(
                *rate_hz,
                *depth_ms,
                *feedback,
                *mix,
                sample_rate,
            )),
            TrackEffectConfig::Phaser {
                rate_hz,
                depth,
                feedback,
                mix,
            } => BuiltInEffect::Phaser(PhaserEffect::new(
                *rate_hz,
                *depth,
                *feedback,
                *mix,
                sample_rate,
            )),
            TrackEffectConfig::RingModulator { carrier_hz, mix } => BuiltInEffect::RingModulator(
                RingModulatorEffect::new(*carrier_hz, *mix, sample_rate),
            ),
            TrackEffectConfig::NoiseGate {
                threshold_db,
                attack_ms,
                release_ms,
                range_db,
            } => BuiltInEffect::NoiseGate(NoiseGateEffect::new(
                *threshold_db,
                *attack_ms,
                *release_ms,
                *range_db,
                sample_rate,
            )),
            TrackEffectConfig::Clap { .. } => return None,
        })
    }

    /// Snapshots this effect's current parameter values for persisting to a song file (see
    /// `main.rs`'s `sync_song_effects`) — the counterpart of `LoadedEffect::param_snapshot` for
    /// CLAP effects.
    pub fn to_config(&self) -> TrackEffectConfig {
        match self {
            BuiltInEffect::Delay(e) => TrackEffectConfig::Delay {
                time_ms: e.time_ms,
                feedback: e.feedback,
                mix: e.mix,
            },
            BuiltInEffect::Bitcrusher(e) => TrackEffectConfig::Bitcrusher {
                bit_depth: e.bit_depth,
                rate_divisor: e.rate_divisor,
                mix: e.mix,
            },
            BuiltInEffect::Distortion(e) => TrackEffectConfig::Distortion {
                drive: e.drive,
                mix: e.mix,
            },
            BuiltInEffect::Reverb(e) => TrackEffectConfig::Reverb {
                room_size: e.room_size,
                damping: e.damping,
                mix: e.mix,
            },
            BuiltInEffect::Chorus(e) => TrackEffectConfig::Chorus {
                rate_hz: e.rate_hz,
                depth_ms: e.depth_ms,
                mix: e.mix,
            },
            BuiltInEffect::Filter(e) => TrackEffectConfig::Filter {
                cutoff_hz: e.cutoff_hz,
                resonance: e.resonance,
                mode: e.mode,
                mix: e.mix,
            },
            BuiltInEffect::Tremolo(e) => TrackEffectConfig::Tremolo {
                rate_hz: e.rate_hz,
                depth: e.depth,
            },
            BuiltInEffect::Compressor(e) => TrackEffectConfig::Compressor {
                threshold_db: e.threshold_db,
                ratio: e.ratio,
                attack_ms: e.attack_ms,
                release_ms: e.release_ms,
                makeup_db: e.makeup_db,
            },
            BuiltInEffect::Flanger(e) => TrackEffectConfig::Flanger {
                rate_hz: e.rate_hz,
                depth_ms: e.depth_ms,
                feedback: e.feedback,
                mix: e.mix,
            },
            BuiltInEffect::Phaser(e) => TrackEffectConfig::Phaser {
                rate_hz: e.rate_hz,
                depth: e.depth,
                feedback: e.feedback,
                mix: e.mix,
            },
            BuiltInEffect::RingModulator(e) => TrackEffectConfig::RingModulator {
                carrier_hz: e.carrier_hz,
                mix: e.mix,
            },
            BuiltInEffect::NoiseGate(e) => TrackEffectConfig::NoiseGate {
                threshold_db: e.threshold_db,
                attack_ms: e.attack_ms,
                release_ms: e.release_ms,
                range_db: e.range_db,
            },
        }
    }

    /// Short label for the FX chain row UI (see `main.rs`'s `track_ui`).
    pub fn label(&self) -> &'static str {
        match self {
            BuiltInEffect::Delay(_) => "Delay",
            BuiltInEffect::Bitcrusher(_) => "Bitcrusher",
            BuiltInEffect::Distortion(_) => "Distortion",
            BuiltInEffect::Reverb(_) => "Reverb",
            BuiltInEffect::Chorus(_) => "Chorus",
            BuiltInEffect::Filter(_) => "Filter",
            BuiltInEffect::Tremolo(_) => "Tremolo",
            BuiltInEffect::Compressor(_) => "Compressor",
            BuiltInEffect::Flanger(_) => "Flanger",
            BuiltInEffect::Phaser(_) => "Phaser",
            BuiltInEffect::RingModulator(_) => "Ring Mod",
            BuiltInEffect::NoiseGate(_) => "Noise Gate",
        }
    }

    /// Runs this effect over one mono audio block, in place.
    pub fn process(&mut self, buf: &mut [f32]) {
        match self {
            BuiltInEffect::Delay(e) => e.process(buf),
            BuiltInEffect::Bitcrusher(e) => e.process(buf),
            BuiltInEffect::Distortion(e) => e.process(buf),
            BuiltInEffect::Reverb(e) => e.process(buf),
            BuiltInEffect::Chorus(e) => e.process(buf),
            BuiltInEffect::Filter(e) => e.process(buf),
            BuiltInEffect::Tremolo(e) => e.process(buf),
            BuiltInEffect::Compressor(e) => e.process(buf),
            BuiltInEffect::Flanger(e) => e.process(buf),
            BuiltInEffect::Phaser(e) => e.process(buf),
            BuiltInEffect::RingModulator(e) => e.process(buf),
            BuiltInEffect::NoiseGate(e) => e.process(buf),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delay_echoes_an_impulse_after_the_delay_time() {
        // 1kHz sample rate makes "10ms" a round 10 samples, easy to check exactly.
        let mut delay = DelayEffect::new(10.0, 0.5, 1.0, 1000.0);
        let mut buf = vec![0.0f32; 30];
        buf[0] = 1.0;
        delay.process(&mut buf);
        // Fully wet (mix = 1.0): the dry impulse itself is replaced by silence at t=0, but its
        // delayed copy appears exactly one delay-time later.
        assert_eq!(buf[0], 0.0);
        assert!(
            buf[10] > 0.0,
            "expected an echo at the delay time, got {}",
            buf[10]
        );
    }

    #[test]
    fn bitcrusher_holds_each_sample_for_rate_divisor_steps() {
        let mut crusher = BitcrusherEffect::new(16.0, 4, 1.0);
        let mut buf: Vec<f32> = (0..8).map(|i| i as f32 / 8.0).collect();
        crusher.process(&mut buf);
        // rate_divisor = 4: the first 4 output samples all hold the first input's (quantized)
        // value, then the 5th sample resamples and holds a new value.
        assert_eq!(buf[0], buf[1]);
        assert_eq!(buf[1], buf[2]);
        assert_eq!(buf[2], buf[3]);
        assert_ne!(buf[3], buf[4]);
    }

    #[test]
    fn distortion_keeps_bounded_input_bounded() {
        let mut distortion = DistortionEffect::new(20.0, 1.0);
        let mut buf: Vec<f32> = (0..64).map(|i| (i as f32 * 0.3).sin()).collect();
        distortion.process(&mut buf);
        for sample in buf {
            assert!(
                sample.abs() <= 1.0,
                "tanh drive should saturate, not blow up: {sample}"
            );
        }
    }

    #[test]
    fn reverb_produces_a_tail_after_the_input_stops_and_never_produces_nan() {
        let mut reverb = ReverbEffect::new(0.8, 0.3, 1.0, 44100.0);
        let mut buf = vec![0.0f32; 4000];
        buf[0] = 1.0;
        reverb.process(&mut buf);
        let tail_energy: f32 = buf[100..4000].iter().map(|s| s.abs()).sum();
        assert!(
            tail_energy > 0.0,
            "expected reverb tail energy after the impulse"
        );
        assert!(
            buf.iter().all(|s| s.is_finite()),
            "reverb output should never be NaN/infinite"
        );
    }

    #[test]
    fn chorus_modulates_a_sustained_tone_away_from_the_dry_signal() {
        // 1kHz sample rate keeps the LFO period small relative to the buffer so the modulation
        // sweeps through multiple delay depths within the test.
        let mut chorus = ChorusEffect::new(5.0, 10.0, 1.0, 1000.0);
        let dry: Vec<f32> = (0..500).map(|i| (i as f32 * 0.05).sin()).collect();
        let mut buf = dry.clone();
        chorus.process(&mut buf);
        let differs = buf
            .iter()
            .zip(dry.iter())
            .any(|(wet, dry)| (wet - dry).abs() > 1e-4);
        assert!(
            differs,
            "fully-wet chorus output should differ from the dry input"
        );
    }

    #[test]
    fn filter_lowpass_attenuates_a_high_frequency_tone() {
        let sample_rate = 44100.0;
        let mut filter = FilterEffect::new(200.0, 0.2, FilterMode::LowPass, 1.0, sample_rate);
        // A tone well above the 200Hz cutoff should be strongly attenuated by a low-pass.
        let mut buf: Vec<f32> = (0..2000)
            .map(|i| (2.0 * std::f32::consts::PI * 5000.0 * i as f32 / sample_rate).sin())
            .collect();
        let input_rms = (buf.iter().map(|s| s * s).sum::<f32>() / buf.len() as f32).sqrt();
        filter.process(&mut buf);
        let output_rms = (buf.iter().map(|s| s * s).sum::<f32>() / buf.len() as f32).sqrt();
        assert!(
            output_rms < input_rms * 0.5,
            "expected low-pass to attenuate a 5kHz tone well below a 200Hz cutoff, input_rms={input_rms}, output_rms={output_rms}"
        );
    }

    #[test]
    fn tremolo_modulates_amplitude_between_one_and_one_minus_depth() {
        let sample_rate = 1000.0;
        let mut tremolo = TremoloEffect::new(2.0, 0.6, sample_rate);
        let mut buf = vec![1.0f32; 1000];
        tremolo.process(&mut buf);
        let max = buf.iter().cloned().fold(f32::MIN, f32::max);
        let min = buf.iter().cloned().fold(f32::MAX, f32::min);
        assert!(
            max > 0.95,
            "expected the LFO peak to leave amplitude near unity, got {max}"
        );
        assert!(
            min < 0.45,
            "expected the LFO trough to cut amplitude by ~depth, got {min}"
        );
    }

    #[test]
    fn compressor_reduces_gain_once_signal_exceeds_threshold() {
        let sample_rate = 44100.0;
        let mut compressor = CompressorEffect::new(-18.0, 4.0, 1.0, 50.0, 0.0, sample_rate);
        // A loud, sustained signal well above the -18dB threshold; run long enough for the
        // envelope follower to settle so the measured gain reduction is steady-state.
        let mut buf = vec![0.5f32; 4000];
        compressor.process(&mut buf);
        let settled = buf[3000];
        assert!(
            settled < 0.5,
            "expected the compressor to reduce gain above threshold, got {settled} from input 0.5"
        );
    }

    #[test]
    fn flanger_modulates_a_sustained_tone_away_from_the_dry_signal() {
        let mut flanger = FlangerEffect::new(3.0, 4.0, 0.7, 1.0, 1000.0);
        let dry: Vec<f32> = (0..500).map(|i| (i as f32 * 0.05).sin()).collect();
        let mut buf = dry.clone();
        flanger.process(&mut buf);
        let differs = buf
            .iter()
            .zip(dry.iter())
            .any(|(wet, dry)| (wet - dry).abs() > 1e-4);
        assert!(
            differs,
            "fully-wet flanger output should differ from the dry input"
        );
        assert!(
            buf.iter().all(|s| s.is_finite()),
            "flanger feedback should never blow up to NaN/infinite"
        );
    }

    #[test]
    fn phaser_sweeps_without_blowing_up() {
        let mut phaser = PhaserEffect::new(0.5, 1.0, 0.5, 0.5, 44100.0);
        let mut buf: Vec<f32> = (0..4000).map(|i| (i as f32 * 0.1).sin()).collect();
        phaser.process(&mut buf);
        assert!(
            buf.iter().all(|s| s.is_finite()),
            "phaser allpass cascade should stay stable"
        );
        assert!(
            buf.iter().any(|s| s.abs() > 1e-4),
            "expected the phaser to still produce a non-silent signal"
        );
    }

    #[test]
    fn ring_modulator_produces_sum_and_difference_frequencies() {
        // A carrier well below the input tone's frequency should audibly change the waveform
        // shape (amplitude modulation), not just scale it uniformly.
        let mut ring_mod = RingModulatorEffect::new(30.0, 1.0, 1000.0);
        let dry: Vec<f32> = (0..500)
            .map(|i| (2.0 * std::f32::consts::PI * 100.0 * i as f32 / 1000.0).sin())
            .collect();
        let mut buf = dry.clone();
        ring_mod.process(&mut buf);
        let differs = buf
            .iter()
            .zip(dry.iter())
            .any(|(wet, dry)| (wet - dry).abs() > 1e-4);
        assert!(
            differs,
            "fully-wet ring modulation should differ from the dry input"
        );
    }

    #[test]
    fn noise_gate_attenuates_quiet_signal_but_passes_loud_signal() {
        let sample_rate = 44100.0;
        let mut quiet_gate = NoiseGateEffect::new(-20.0, 1.0, 20.0, -60.0, sample_rate);
        let mut quiet_buf = vec![0.01f32; 4000]; // well below -20dB threshold
        quiet_gate.process(&mut quiet_buf);
        assert!(
            quiet_buf[3000].abs() < 0.01,
            "expected the gate to attenuate a signal below threshold, got {}",
            quiet_buf[3000]
        );

        let mut loud_gate = NoiseGateEffect::new(-20.0, 1.0, 20.0, -60.0, sample_rate);
        let mut loud_buf = vec![0.5f32; 4000]; // well above -20dB threshold
        loud_gate.process(&mut loud_buf);
        assert!(
            loud_buf[3000] > 0.45,
            "expected the gate to pass a signal above threshold mostly unchanged, got {}",
            loud_buf[3000]
        );
    }
}
