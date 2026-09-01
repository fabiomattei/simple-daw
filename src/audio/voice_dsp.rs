//! Shared DSP primitives used by two or more of the four voice types (`simple_voice`,
//! `trine_voice`, `wave_voice`, `sample_voice`): raw oscillator waveform generation
//! (`waveform_sample`), a fast noise hash (`hash_to_bipolar`), pitch/pan conversions
//! (`pitch_to_freq`, `unison_pan_gains`), the state-variable filter stage shared by every engine's
//! filter (`svf_stage`, `run_filter_stage`, `run_dual_filter_stage`), and the reusable ADSR
//! generator `Trine`/`Wave` voices instantiate multiple copies of (`EnvGen`) — `Simple` has its
//! own inline envelope instead (see `simple_voice::Voice`), sharing only `EnvelopeStage`.

use crate::model::{FilterRouting, FilterSlope, FilterType, SynthWaveform};

use super::ENVELOPE_FLOOR;

/// Raw oscillator output in `[-1, 1]` for one cycle, `phase` running `0..1`. `pulse_width` is the
/// Square wave's duty cycle (0.5 = classic 50/50 square); ignored by the other waveforms.
pub(crate) fn waveform_sample(waveform: SynthWaveform, phase: f32, pulse_width: f32) -> f32 {
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
pub(crate) fn hash_to_bipolar(x: u32) -> f32 {
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
pub(crate) enum EnvelopeStage {
    #[default]
    Attack,
    Decay,
    Sustain,
    Release,
}

pub(crate) fn pitch_to_freq(pitch: u8) -> f32 {
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
pub(crate) fn unison_pan_gains(spread: f32) -> (f32, f32) {
    let gain_l = (1.0 - spread).clamp(0.0, 1.0);
    let gain_r = (1.0 + spread).clamp(0.0, 1.0);
    (gain_l, gain_r)
}

/// One TPT state-variable filter stage (Zavalishin's "Art of VA Filter Design") — shared by
/// `Voice` (one instance per channel once unison spread diverges L/R), and by `TrineVoice`/
/// `WaveVoice`, which each cascade it once (12dB/octave) or twice (24dB/octave) per `FilterSlope`.
pub(crate) fn svf_stage(
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
pub(crate) fn run_filter_stage(
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
pub(crate) fn run_dual_filter_stage(
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
pub(crate) struct EnvGen {
    pub(crate) stage: EnvelopeStage,
    elapsed_samples: u64,
    gate_samples: u64,
    peak: f32,
    sustain: f32,
    pub(crate) value: f32,
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
    pub(crate) fn trigger(
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

    pub(crate) fn advance(&mut self) -> f32 {
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
