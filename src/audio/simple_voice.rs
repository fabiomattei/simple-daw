//! The `Simple` synth engine's real-time voice — see `model::SynthParams`. A hand-inlined
//! attack/decay/sustain/release envelope (not `voice_dsp::EnvGen`, which `Trine`/`Wave` use
//! instead — `Simple` only ever needs one envelope per voice, so there's nothing to share) drives
//! up to `MAX_UNISON_VOICES` detuned oscillator copies, an optional second oscillator, sub-osc,
//! LFO, and a single resonant filter with its own decay-only envelope.

use crate::model::{FilterType, LfoTarget, SynthParams, SynthWaveform};

use super::voice_dsp::{EnvelopeStage, svf_stage, unison_pan_gains, waveform_sample};
use super::{ENVELOPE_FLOOR, MAX_UNISON_VOICES};

/// Hz swept by the LFO at full depth when `LfoTarget::FilterCutoff` is selected — deliberately
/// narrower than `filter_env_amount_hz`'s +/-10kHz range since an LFO sweep is felt continuously
/// rather than once per note.
const FILTER_LFO_RANGE_HZ: f32 = 4000.0;
/// Max pitch swing (cents) at full LFO depth when `LfoTarget::Pitch` is selected — one semitone,
/// a standard vibrato range.
const PITCH_LFO_MAX_CENTS: f32 = 100.0;

/// A single synthesized voice: up to `MAX_UNISON_VOICES` detuned copies of an oscillator (see
/// `SynthWaveform`), plus an optional second oscillator (crossfaded) and sub-oscillator (mixed
/// additively), summed and run through a real attack/decay/sustain/release envelope, an optional
/// LFO (pitch/amplitude/filter-cutoff), and a resonant filter (switchable type) with its own
/// decay-only envelope. Standing in for a real instrument until phase 4 (which added samples for
/// drums; this remains the melodic/synth path).
#[derive(Clone, Copy)]
pub(crate) struct Voice {
    pub(crate) active: bool,

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
    pub(crate) fn trigger(
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
    pub(crate) fn next_sample(&mut self) -> (f32, f32) {
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
