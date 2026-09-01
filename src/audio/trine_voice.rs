//! The `Trine` synth engine's real-time voice — see `model::TrineParams`. Three oscillators run
//! through FM/ring-mod into a dual filter/mod-matrix (`voice_dsp::run_dual_filter_stage`, shared
//! with `wave_voice::WaveVoice`), driven by up to three `voice_dsp::EnvGen` instances (Env1/Env2/
//! Env3) and two LFOs. Stereo width comes from running the whole signal chain twice with
//! independently-seeded analog drift (`TrineVoiceChannel`) rather than a pan/unison spread — see
//! `TrineVoiceChannel`'s own doc comment.

use crate::model::{FilterRouting, FilterSlope, FilterType, ModSlot, ModSource, ModTarget, SynthWaveform, TrineParams};

use super::ENVELOPE_FLOOR;
use super::voice_dsp::{EnvGen, EnvelopeStage, hash_to_bipolar, run_dual_filter_stage, waveform_sample};

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
pub(crate) struct TrineVoice {
    pub(crate) active: bool,
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
    pub(crate) fn trigger(
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
    pub(crate) fn next_sample(&mut self, mod_slots: &[ModSlot]) -> (f32, f32) {
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
