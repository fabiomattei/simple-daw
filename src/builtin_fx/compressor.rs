//! Stereo compressor, dual-mono: each channel runs its own independent envelope follower and gain
//! computer (see `builtin_fx`'s module doc) — L/R aren't linked, so a loud transient on one
//! channel doesn't duck the other the way a stereo-linked compressor would. That's a deliberate
//! simplification, not an oversight: linking would need its own explicit design (which signal
//! drives the shared envelope) rather than falling out of the dual-mono pattern for free.

struct CompressorChannel {
    envelope: f32,
}

impl CompressorChannel {
    fn new() -> Self {
        Self { envelope: 0.0 }
    }

    fn process(
        &mut self,
        buf: &mut [f32],
        ratio: f32,
        threshold: f32,
        makeup: f32,
        attack_coeff: f32,
        release_coeff: f32,
    ) {
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

/// Feedforward dynamics compressor: a one-pole peak envelope follower (separate attack/release
/// time constants) feeding a dB-domain gain computer, plus a static makeup gain.
pub(crate) struct CompressorEffect {
    pub threshold_db: f32,
    pub ratio: f32,
    pub attack_ms: f32,
    pub release_ms: f32,
    pub makeup_db: f32,
    left: CompressorChannel,
    right: CompressorChannel,
    sample_rate: f32,
}

impl CompressorEffect {
    pub(super) fn new(
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
            left: CompressorChannel::new(),
            right: CompressorChannel::new(),
            sample_rate,
        }
    }

    fn time_coeff(time_ms: f32, sample_rate: f32) -> f32 {
        (-1.0 / (time_ms.max(0.1) / 1000.0 * sample_rate)).exp()
    }

    pub(super) fn process(&mut self, l: &mut [f32], r: &mut [f32]) {
        let ratio = self.ratio.max(1.0);
        let threshold = self.threshold_db;
        let makeup = 10f32.powf(self.makeup_db / 20.0);
        let attack_coeff = Self::time_coeff(self.attack_ms, self.sample_rate);
        let release_coeff = Self::time_coeff(self.release_ms, self.sample_rate);
        self.left
            .process(l, ratio, threshold, makeup, attack_coeff, release_coeff);
        self.right
            .process(r, ratio, threshold, makeup, attack_coeff, release_coeff);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compressor_reduces_gain_once_signal_exceeds_threshold() {
        let sample_rate = 44100.0;
        let mut compressor = CompressorEffect::new(-18.0, 4.0, 1.0, 50.0, 0.0, sample_rate);
        // A loud, sustained signal well above the -18dB threshold; run long enough for the
        // envelope follower to settle so the measured gain reduction is steady-state.
        let mut l = vec![0.5f32; 4000];
        let mut r = vec![0.5f32; 4000];
        compressor.process(&mut l, &mut r);
        let settled = l[3000];
        assert!(
            settled < 0.5,
            "expected the compressor to reduce gain above threshold, got {settled} from input 0.5"
        );
        assert!(r[3000] < 0.5);
    }
}
