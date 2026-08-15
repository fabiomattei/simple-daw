//! Stereo state-variable filter, dual-mono (see `builtin_fx`'s module doc).

use crate::model::FilterMode;

struct FilterChannel {
    low: f32,
    band: f32,
}

impl FilterChannel {
    fn new() -> Self {
        Self {
            low: 0.0,
            band: 0.0,
        }
    }

    fn process(&mut self, buf: &mut [f32], f: f32, q: f32, mode: FilterMode, mix: f32) {
        for sample in buf.iter_mut() {
            let input = *sample;
            let high = input - self.low - q * self.band;
            self.band += f * high;
            self.low += f * self.band;
            let output = match mode {
                FilterMode::LowPass => self.low,
                FilterMode::HighPass => high,
            };
            *sample = input * (1.0 - mix) + output * mix;
        }
    }
}

/// A Chamberlin state-variable filter run in low-pass or high-pass mode. Cheap (no biquad
/// coefficient recompute needed per block) and stable enough for the cutoff/resonance ranges the
/// UI exposes.
pub(crate) struct FilterEffect {
    pub cutoff_hz: f32,
    pub resonance: f32,
    pub mode: FilterMode,
    pub mix: f32,
    left: FilterChannel,
    right: FilterChannel,
    sample_rate: f32,
}

impl FilterEffect {
    pub(super) fn new(cutoff_hz: f32, resonance: f32, mode: FilterMode, mix: f32, sample_rate: f32) -> Self {
        Self {
            cutoff_hz,
            resonance,
            mode,
            mix,
            left: FilterChannel::new(),
            right: FilterChannel::new(),
            sample_rate,
        }
    }

    pub(super) fn process(&mut self, l: &mut [f32], r: &mut [f32]) {
        let nyquist = self.sample_rate * 0.5;
        let cutoff = self.cutoff_hz.clamp(20.0, nyquist - 100.0);
        let f = 2.0 * (std::f32::consts::PI * cutoff / self.sample_rate).sin();
        // resonance in 0..1 from the UI maps to the SVF's damping (q): higher resonance means
        // lower damping, i.e. more feedback ringing at the cutoff frequency.
        let q = (1.0 - self.resonance.clamp(0.0, 0.99)).max(0.01) * 2.0;
        let mix = self.mix.clamp(0.0, 1.0);
        self.left.process(l, f, q, self.mode, mix);
        self.right.process(r, f, q, self.mode, mix);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_lowpass_attenuates_a_high_frequency_tone() {
        let sample_rate = 44100.0;
        let mut filter = FilterEffect::new(200.0, 0.2, FilterMode::LowPass, 1.0, sample_rate);
        // A tone well above the 200Hz cutoff should be strongly attenuated by a low-pass.
        let dry: Vec<f32> = (0..2000)
            .map(|i| (2.0 * std::f32::consts::PI * 5000.0 * i as f32 / sample_rate).sin())
            .collect();
        let mut l = dry.clone();
        let mut r = dry.clone();
        let input_rms = (dry.iter().map(|s| s * s).sum::<f32>() / dry.len() as f32).sqrt();
        filter.process(&mut l, &mut r);
        let output_rms = (l.iter().map(|s| s * s).sum::<f32>() / l.len() as f32).sqrt();
        assert!(
            output_rms < input_rms * 0.5,
            "expected low-pass to attenuate a 5kHz tone well below a 200Hz cutoff, input_rms={input_rms}, output_rms={output_rms}"
        );
        assert_eq!(l, r, "identical input on both channels should produce identical output");
    }
}
