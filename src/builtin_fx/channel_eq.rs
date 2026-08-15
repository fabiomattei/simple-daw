//! Parametric multiband EQ, dual-mono (see `builtin_fx`'s module doc). Each `EqBand` becomes one
//! RBJ Audio EQ Cookbook biquad stage; bands run in series in the order they're stored.

use crate::model::{EqBand, EqBandType};

#[derive(Clone, Copy)]
struct BiquadCoeffs {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
}

impl BiquadCoeffs {
    /// RBJ Audio EQ Cookbook formulas, normalized so `a0 == 1`. `None` for a disabled band —
    /// callers skip biquad processing entirely rather than running an identity stage.
    fn compute(band: &EqBand, sample_rate: f32) -> Option<Self> {
        if !band.enabled {
            return None;
        }
        let nyquist = sample_rate * 0.5;
        let freq = band.freq_hz.clamp(20.0, nyquist - 100.0);
        let q = band.q.clamp(0.1, 10.0);
        let w0 = 2.0 * std::f32::consts::PI * freq / sample_rate;
        let (sin_w0, cos_w0) = w0.sin_cos();
        let alpha = sin_w0 / (2.0 * q);

        let (b0, b1, b2, a0, a1, a2) = match band.band_type {
            EqBandType::HighPass => (
                (1.0 + cos_w0) / 2.0,
                -(1.0 + cos_w0),
                (1.0 + cos_w0) / 2.0,
                1.0 + alpha,
                -2.0 * cos_w0,
                1.0 - alpha,
            ),
            EqBandType::LowPass => (
                (1.0 - cos_w0) / 2.0,
                1.0 - cos_w0,
                (1.0 - cos_w0) / 2.0,
                1.0 + alpha,
                -2.0 * cos_w0,
                1.0 - alpha,
            ),
            EqBandType::Peak => {
                let a = 10f32.powf(band.gain_db / 40.0);
                (
                    1.0 + alpha * a,
                    -2.0 * cos_w0,
                    1.0 - alpha * a,
                    1.0 + alpha / a,
                    -2.0 * cos_w0,
                    1.0 - alpha / a,
                )
            }
            EqBandType::LowShelf => {
                let a = 10f32.powf(band.gain_db / 40.0);
                let sqrt_a = a.sqrt();
                let shelf_alpha = sin_w0 / 2.0 * ((a + 1.0 / a) * (1.0 / q - 1.0) + 2.0).sqrt();
                (
                    a * ((a + 1.0) - (a - 1.0) * cos_w0 + 2.0 * sqrt_a * shelf_alpha),
                    2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0),
                    a * ((a + 1.0) - (a - 1.0) * cos_w0 - 2.0 * sqrt_a * shelf_alpha),
                    (a + 1.0) + (a - 1.0) * cos_w0 + 2.0 * sqrt_a * shelf_alpha,
                    -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0),
                    (a + 1.0) + (a - 1.0) * cos_w0 - 2.0 * sqrt_a * shelf_alpha,
                )
            }
            EqBandType::HighShelf => {
                let a = 10f32.powf(band.gain_db / 40.0);
                let sqrt_a = a.sqrt();
                let shelf_alpha = sin_w0 / 2.0 * ((a + 1.0 / a) * (1.0 / q - 1.0) + 2.0).sqrt();
                (
                    a * ((a + 1.0) + (a - 1.0) * cos_w0 + 2.0 * sqrt_a * shelf_alpha),
                    -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0),
                    a * ((a + 1.0) + (a - 1.0) * cos_w0 - 2.0 * sqrt_a * shelf_alpha),
                    (a + 1.0) - (a - 1.0) * cos_w0 + 2.0 * sqrt_a * shelf_alpha,
                    2.0 * ((a - 1.0) - (a + 1.0) * cos_w0),
                    (a + 1.0) - (a - 1.0) * cos_w0 - 2.0 * sqrt_a * shelf_alpha,
                )
            }
        };

        Some(Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        })
    }
}

/// Direct Form II Transposed biquad state for one band on one channel.
#[derive(Default)]
struct BiquadState {
    z1: f32,
    z2: f32,
}

impl BiquadState {
    fn process(&mut self, c: &BiquadCoeffs, input: f32) -> f32 {
        let output = c.b0 * input + self.z1;
        self.z1 = c.b1 * input - c.a1 * output + self.z2;
        self.z2 = c.b2 * input - c.a2 * output;
        output
    }
}

pub(crate) struct ChannelEqEffect {
    pub bands: Vec<EqBand>,
    left: Vec<BiquadState>,
    right: Vec<BiquadState>,
    sample_rate: f32,
}

impl ChannelEqEffect {
    pub(super) fn new(bands: Vec<EqBand>, sample_rate: f32) -> Self {
        let left = bands.iter().map(|_| BiquadState::default()).collect();
        let right = bands.iter().map(|_| BiquadState::default()).collect();
        Self {
            bands,
            left,
            right,
            sample_rate,
        }
    }

    pub(super) fn process(&mut self, l: &mut [f32], r: &mut [f32]) {
        for (i, band) in self.bands.iter().enumerate() {
            let Some(coeffs) = BiquadCoeffs::compute(band, self.sample_rate) else {
                continue;
            };
            for sample in l.iter_mut() {
                *sample = self.left[i].process(&coeffs, *sample);
            }
            for sample in r.iter_mut() {
                *sample = self.right[i].process(&coeffs, *sample);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(freq: f32, sample_rate: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate).sin())
            .collect()
    }

    fn rms(buf: &[f32]) -> f32 {
        (buf.iter().map(|s| s * s).sum::<f32>() / buf.len() as f32).sqrt()
    }

    fn single_band_eq(band: EqBand, sample_rate: f32) -> ChannelEqEffect {
        ChannelEqEffect::new(vec![band], sample_rate)
    }

    #[test]
    fn disabled_band_is_a_no_op() {
        let sample_rate = 44100.0;
        let mut eq = single_band_eq(
            EqBand {
                band_type: EqBandType::Peak,
                freq_hz: 1000.0,
                gain_db: 12.0,
                q: 1.0,
                enabled: false,
            },
            sample_rate,
        );
        let dry = sine(1000.0, sample_rate, 512);
        let mut l = dry.clone();
        let mut r = dry.clone();
        eq.process(&mut l, &mut r);
        assert_eq!(l, dry);
        assert_eq!(r, dry);
    }

    #[test]
    fn peak_boost_raises_energy_at_center_frequency() {
        let sample_rate = 44100.0;
        let mut eq = single_band_eq(
            EqBand {
                band_type: EqBandType::Peak,
                freq_hz: 1000.0,
                gain_db: 12.0,
                q: 1.0,
                enabled: true,
            },
            sample_rate,
        );
        let dry = sine(1000.0, sample_rate, 4096);
        let input_rms = rms(&dry);
        let mut l = dry.clone();
        let mut r = dry.clone();
        eq.process(&mut l, &mut r);
        assert!(
            rms(&l) > input_rms * 1.5,
            "expected a +12dB peak at 1kHz to noticeably boost a 1kHz tone"
        );
        assert_eq!(l, r, "identical input on both channels should produce identical output");
    }

    #[test]
    fn low_shelf_boosts_low_frequency_more_than_high() {
        let sample_rate = 44100.0;
        let band = EqBand {
            band_type: EqBandType::LowShelf,
            freq_hz: 200.0,
            gain_db: 12.0,
            q: 0.7,
            enabled: true,
        };
        let low = sine(60.0, sample_rate, 4096);
        let high = sine(8000.0, sample_rate, 4096);

        let mut eq_low = single_band_eq(band, sample_rate);
        let mut low_l = low.clone();
        let mut low_r = low.clone();
        eq_low.process(&mut low_l, &mut low_r);

        let mut eq_high = single_band_eq(band, sample_rate);
        let mut high_l = high.clone();
        let mut high_r = high.clone();
        eq_high.process(&mut high_l, &mut high_r);

        assert!(
            rms(&low_l) / rms(&low) > rms(&high_l) / rms(&high),
            "a low shelf should boost a 60Hz tone more than an 8kHz tone"
        );
    }

    #[test]
    fn high_pass_attenuates_a_tone_below_cutoff() {
        let sample_rate = 44100.0;
        let mut eq = single_band_eq(
            EqBand {
                band_type: EqBandType::HighPass,
                freq_hz: 1000.0,
                gain_db: 0.0,
                q: 0.7,
                enabled: true,
            },
            sample_rate,
        );
        let dry = sine(100.0, sample_rate, 4096);
        let input_rms = rms(&dry);
        let mut l = dry.clone();
        let mut r = dry.clone();
        eq.process(&mut l, &mut r);
        assert!(
            rms(&l) < input_rms * 0.5,
            "expected a 1kHz high-pass to strongly attenuate a 100Hz tone"
        );
    }
}
