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
        key: Option<&[f32]>,
        ratio: f32,
        threshold: f32,
        makeup: f32,
        attack_coeff: f32,
        release_coeff: f32,
    ) {
        for (i, sample) in buf.iter_mut().enumerate() {
            let input = *sample;
            let rectified = key.map_or(input.abs(), |key| key[i].abs());
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
    /// See `TrackEffectConfig::Compressor`'s `sidechain_source` doc. Set from that field by
    /// `BuiltInEffect::from_config`, edited live via the FX chain UI's sidechain picker (same as
    /// this struct's other `pub` fields), and read back by `BuiltInEffect::to_config`.
    pub sidechain_source: Option<usize>,
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
            sidechain_source: None,
            left: CompressorChannel::new(),
            right: CompressorChannel::new(),
            sample_rate,
        }
    }

    fn time_coeff(time_ms: f32, sample_rate: f32) -> f32 {
        (-1.0 / (time_ms.max(0.1) / 1000.0 * sample_rate)).exp()
    }

    pub(super) fn process(&mut self, l: &mut [f32], r: &mut [f32]) {
        self.process_with_sidechain(l, r, None);
    }

    /// Same as `process`, but drives the envelope follower from `sidechain`'s rectified signal
    /// (a key input, e.g. another track routed in for ducking) instead of `l`/`r`'s own — the
    /// gain computed off that key is still applied to `l`/`r` themselves. `None` behaves exactly
    /// like `process` (envelope follows the compressor's own input, as before sidechain existed).
    pub(super) fn process_with_sidechain(
        &mut self,
        l: &mut [f32],
        r: &mut [f32],
        sidechain: Option<(&[f32], &[f32])>,
    ) {
        let ratio = self.ratio.max(1.0);
        let threshold = self.threshold_db;
        let makeup = 10f32.powf(self.makeup_db / 20.0);
        let attack_coeff = Self::time_coeff(self.attack_ms, self.sample_rate);
        let release_coeff = Self::time_coeff(self.release_ms, self.sample_rate);
        let (key_l, key_r) = match sidechain {
            Some((key_l, key_r)) => (Some(key_l), Some(key_r)),
            None => (None, None),
        };
        self.left
            .process(l, key_l, ratio, threshold, makeup, attack_coeff, release_coeff);
        self.right
            .process(r, key_r, ratio, threshold, makeup, attack_coeff, release_coeff);
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

    #[test]
    fn sidechain_key_drives_the_envelope_instead_of_the_main_signal() {
        let sample_rate = 44100.0;
        let mut compressor = CompressorEffect::new(-18.0, 4.0, 1.0, 50.0, 0.0, sample_rate);
        // The main signal alone is a quiet, steady tone well below the -18dB threshold — with no
        // sidechain, it would pass through untouched (mirroring the settled-gain assertion above).
        // A loud key signal should still duck it, proving the envelope follows the key, not `l`/`r`.
        let mut l = vec![0.05f32; 4000];
        let mut r = vec![0.05f32; 4000];
        let key_l = vec![0.9f32; 4000];
        let key_r = vec![0.9f32; 4000];
        compressor.process_with_sidechain(&mut l, &mut r, Some((&key_l, &key_r)));
        let settled = l[3000];
        assert!(
            settled < 0.05,
            "expected a loud sidechain key to duck a quiet main signal, got {settled} from input 0.05"
        );
        assert!(r[3000] < 0.05);
    }
}
