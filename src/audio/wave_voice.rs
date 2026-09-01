//! The `Wave` synth engine's real-time voice — see `model::WaveParams`. Two wavetable oscillators
//! (via `crate::wavetable`, with optional phase-warp) run through a dual filter/mod-matrix
//! (`voice_dsp::run_dual_filter_stage`, shared with `trine_voice::TrineVoice`), driven by up to
//! five modulation sources (2 LFOs, 2 free `voice_dsp::EnvGen`s, velocity) plus a sub-oscillator
//! and noise oscillator mixed in additively.

use crate::model::{
    FilterRouting, FilterSlope, FilterType, SynthWaveform, WaveModSlot, WaveModSource, WaveModTarget, WaveParams,
};
use crate::wavetable::{self, WaveWarpMode, WavetableId};

use super::voice_dsp::{EnvGen, EnvelopeStage, hash_to_bipolar, run_dual_filter_stage, unison_pan_gains, waveform_sample};
use super::{ENVELOPE_FLOOR, MAX_UNISON_VOICES};

/// Max pitch swing (semitones) at full modulation depth when a `WaveVoice` matrix slot targets
/// `WaveModTarget::Pitch` — see `TRINE_PITCH_MOD_MAX_SEMITONES`, the equivalent for `TrineVoice`.
const WAVE_PITCH_MOD_MAX_SEMITONES: f32 = 12.0;
/// Hz swept at full modulation depth when a matrix slot targets `WaveModTarget::FilterCutoff` or
/// `Filter2Cutoff` — see `TRINE_FILTER_MOD_RANGE_HZ`.
const WAVE_FILTER_MOD_RANGE_HZ: f32 = 8000.0;
/// Resonance swept at full modulation depth when a matrix slot targets `WaveModTarget::FilterResonance`.
const WAVE_RESONANCE_MOD_RANGE: f32 = 5.0;

/// One Wave-engine voice — see `WaveParams`. Two wavetable oscillators (each scanning its table's
/// frames via `osc*_position`, with an optional phase-warp — see `wavetable::warp_phase`) run
/// into a dual filter (series/parallel/off routing, switchable slope, drive) shared with
/// `TrineVoice`'s implementation, while up to five modulation sources (2 LFOs, 2 free envelopes,
/// velocity) are evaluated each sample and routed through the track's `mod_slots`. A sub-
/// oscillator and a noise oscillator mix in additively alongside the two wavetable oscillators.
/// `mod_slots` is passed into `next_sample` rather than copied in at `trigger` time, for the same
/// reason as `TrineVoice`'s.
#[derive(Clone, Copy)]
pub(crate) struct WaveVoice {
    pub(crate) active: bool,
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
    pub(crate) fn trigger(
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
    pub(crate) fn next_sample(&mut self, mod_slots: &[WaveModSlot]) -> (f32, f32) {
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
