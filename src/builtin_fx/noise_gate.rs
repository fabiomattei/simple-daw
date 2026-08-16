//! Stereo noise gate, dual-mono: each channel runs its own envelope follower and gain smoother
//! (see `builtin_fx`'s module doc, and `compressor.rs`'s note on why L/R aren't linked).

struct NoiseGateChannel {
    envelope: f32,
    gain: f32,
}

impl NoiseGateChannel {
    fn new() -> Self {
        Self {
            envelope: 0.0,
            gain: 0.0,
        }
    }

    fn process(
        &mut self,
        buf: &mut [f32],
        key: Option<&[f32]>,
        threshold_db: f32,
        closed_gain: f32,
        attack_coeff: f32,
        release_coeff: f32,
    ) {
        for (i, sample) in buf.iter_mut().enumerate() {
            let input = *sample;
            let rectified = key.map_or(input.abs(), |key| key[i].abs());
            let env_coeff = if rectified > self.envelope {
                attack_coeff
            } else {
                release_coeff
            };
            self.envelope = env_coeff * self.envelope + (1.0 - env_coeff) * rectified;
            let env_db = 20.0 * self.envelope.max(1e-6).log10();
            let target_gain = if env_db > threshold_db {
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

/// A noise gate: an envelope follower decides whether the signal is above `threshold_db`, and a
/// separately-smoothed gain (same attack/release time constants) is applied so it doesn't click
/// open/closed. `range_db` is how far down the gate attenuates when closed — a large negative
/// number (e.g. -60dB) rather than hard-muting to exact silence, matching how real noise gates
/// expose a "range" control instead of just on/off.
pub(crate) struct NoiseGateEffect {
    pub threshold_db: f32,
    pub attack_ms: f32,
    pub release_ms: f32,
    pub range_db: f32,
    /// See `TrackEffectConfig::NoiseGate`'s `sidechain_source` doc. Set from that field by
    /// `BuiltInEffect::from_config`, edited live via the FX chain UI's sidechain picker (same as
    /// this struct's other `pub` fields), and read back by `BuiltInEffect::to_config`.
    pub sidechain_source: Option<usize>,
    left: NoiseGateChannel,
    right: NoiseGateChannel,
    sample_rate: f32,
}

impl NoiseGateEffect {
    pub(super) fn new(
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
            sidechain_source: None,
            left: NoiseGateChannel::new(),
            right: NoiseGateChannel::new(),
            sample_rate,
        }
    }

    fn time_coeff(time_ms: f32, sample_rate: f32) -> f32 {
        (-1.0 / (time_ms.max(0.1) / 1000.0 * sample_rate)).exp()
    }

    /// The gate opens/closes off `sidechain`'s envelope (a key input) rather than `l`/`r`'s own
    /// — e.g. gating a pad open only while a rhythm track is playing. `None` follows the gate's
    /// own input, as before sidechain existed.
    pub(super) fn process_with_sidechain(
        &mut self,
        l: &mut [f32],
        r: &mut [f32],
        sidechain: Option<(&[f32], &[f32])>,
    ) {
        let attack_coeff = Self::time_coeff(self.attack_ms, self.sample_rate);
        let release_coeff = Self::time_coeff(self.release_ms, self.sample_rate);
        let closed_gain = 10f32.powf(self.range_db / 20.0);
        let (key_l, key_r) = match sidechain {
            Some((key_l, key_r)) => (Some(key_l), Some(key_r)),
            None => (None, None),
        };
        self.left.process(
            l,
            key_l,
            self.threshold_db,
            closed_gain,
            attack_coeff,
            release_coeff,
        );
        self.right.process(
            r,
            key_r,
            self.threshold_db,
            closed_gain,
            attack_coeff,
            release_coeff,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_gate_attenuates_quiet_signal_but_passes_loud_signal() {
        let sample_rate = 44100.0;
        let mut quiet_gate = NoiseGateEffect::new(-20.0, 1.0, 20.0, -60.0, sample_rate);
        let mut quiet_l = vec![0.01f32; 4000]; // well below -20dB threshold
        let mut quiet_r = quiet_l.clone();
        quiet_gate.process_with_sidechain(&mut quiet_l, &mut quiet_r, None);
        assert!(
            quiet_l[3000].abs() < 0.01,
            "expected the gate to attenuate a signal below threshold, got {}",
            quiet_l[3000]
        );

        let mut loud_gate = NoiseGateEffect::new(-20.0, 1.0, 20.0, -60.0, sample_rate);
        let mut loud_l = vec![0.5f32; 4000]; // well above -20dB threshold
        let mut loud_r = loud_l.clone();
        loud_gate.process_with_sidechain(&mut loud_l, &mut loud_r, None);
        assert!(
            loud_l[3000] > 0.45,
            "expected the gate to pass a signal above threshold mostly unchanged, got {}",
            loud_l[3000]
        );
    }

    #[test]
    fn sidechain_key_opens_the_gate_for_a_quiet_main_signal() {
        let sample_rate = 44100.0;
        let mut gate = NoiseGateEffect::new(-20.0, 1.0, 20.0, -60.0, sample_rate);
        // Below threshold on its own (mirrors the `quiet_gate` case above), but a loud key signal
        // should hold the gate open anyway, proving it follows the key, not `l`/`r`.
        let mut l = vec![0.01f32; 4000];
        let mut r = l.clone();
        let key = vec![0.5f32; 4000];
        gate.process_with_sidechain(&mut l, &mut r, Some((&key, &key)));
        assert!(
            l[3000] > 0.009,
            "expected a loud sidechain key to hold the gate open for a quiet main signal, got {}",
            l[3000]
        );
    }
}
