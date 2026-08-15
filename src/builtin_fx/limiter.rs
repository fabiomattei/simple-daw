//! Stereo brickwall/peak limiter, dual-mono: each channel runs its own look-ahead peak detector
//! and gain smoother (see `builtin_fx`'s module doc, and `compressor.rs`'s note on why L/R aren't
//! linked). Unlike `CompressorEffect` (a same-instant envelope follower, so a fast transient can
//! still poke above the gain computer's threshold before the envelope catches up), this effect
//! delays the audio by a short, fixed look-ahead window and scans that window for its true peak
//! before letting the delayed sample out — so gain reduction is already in place by the time the
//! peak itself reaches the output. That's what makes `ceiling_db` a hard ceiling rather than a
//! statistical target, at the cost of a fixed ~5ms output latency.

/// Fixed look-ahead window. Not user-exposed — long enough to catch any transient a look-ahead
/// limiter needs to see coming, short enough that the induced latency stays inaudible/negligible
/// for this app's use (there's no cross-effect latency compensation elsewhere in the chain).
const LOOKAHEAD_MS: f32 = 5.0;

struct LimiterChannel {
    /// Ring buffer of the last `buffer.len()` input samples, doubling as the delay line: the
    /// slot about to be overwritten always holds the oldest buffered sample, i.e. the one due out
    /// next.
    buffer: Vec<f32>,
    write_pos: usize,
    gain: f32,
}

impl LimiterChannel {
    fn new(lookahead_samples: usize) -> Self {
        Self {
            buffer: vec![0.0; lookahead_samples.max(1)],
            write_pos: 0,
            gain: 1.0,
        }
    }

    fn process(&mut self, buf: &mut [f32], input_gain: f32, ceiling: f32, release_coeff: f32) {
        for sample in buf.iter_mut() {
            let x = *sample * input_gain;
            let delayed = self.buffer[self.write_pos];
            self.buffer[self.write_pos] = x;
            self.write_pos = (self.write_pos + 1) % self.buffer.len();

            // The peak over the window [delayed sample .. current sample]: `delayed` plus every
            // sample now held in the ring buffer (which, post-write, spans everything newer than
            // `delayed` up to and including `x`).
            let mut peak = delayed.abs();
            for &s in &self.buffer {
                let a = s.abs();
                if a > peak {
                    peak = a;
                }
            }

            let target_gain = if peak > ceiling { ceiling / peak } else { 1.0 };
            // Reduction is applied the instant the look-ahead window reveals it's needed (no
            // attack smoothing required — that's the point of scanning ahead); only the return
            // toward unity gain once the loud passage ends is smoothed, so gain recovery doesn't
            // pump audibly.
            self.gain = if target_gain < self.gain {
                target_gain
            } else {
                release_coeff * self.gain + (1.0 - release_coeff) * target_gain
            };
            *sample = delayed * self.gain;
        }
    }
}

/// Look-ahead peak limiter: `input_gain_db` drives the signal into the limiter, `ceiling_db` is
/// the hard output ceiling it never exceeds, `release_ms` sets how fast gain recovers back toward
/// unity after a loud passage.
pub(crate) struct LimiterEffect {
    pub input_gain_db: f32,
    pub ceiling_db: f32,
    pub release_ms: f32,
    left: LimiterChannel,
    right: LimiterChannel,
    sample_rate: f32,
}

impl LimiterEffect {
    pub(super) fn new(
        input_gain_db: f32,
        ceiling_db: f32,
        release_ms: f32,
        sample_rate: f32,
    ) -> Self {
        let lookahead_samples = ((LOOKAHEAD_MS / 1000.0) * sample_rate).round() as usize;
        Self {
            input_gain_db,
            ceiling_db,
            release_ms,
            left: LimiterChannel::new(lookahead_samples),
            right: LimiterChannel::new(lookahead_samples),
            sample_rate,
        }
    }

    fn time_coeff(time_ms: f32, sample_rate: f32) -> f32 {
        (-1.0 / (time_ms.max(0.1) / 1000.0 * sample_rate)).exp()
    }

    pub(super) fn process(&mut self, l: &mut [f32], r: &mut [f32]) {
        let input_gain = 10f32.powf(self.input_gain_db / 20.0);
        let ceiling = 10f32.powf(self.ceiling_db / 20.0);
        let release_coeff = Self::time_coeff(self.release_ms, self.sample_rate);
        self.left.process(l, input_gain, ceiling, release_coeff);
        self.right.process(r, input_gain, ceiling, release_coeff);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limiter_never_lets_a_transient_exceed_the_ceiling() {
        let sample_rate = 44100.0;
        let ceiling_db = -1.0;
        let mut limiter = LimiterEffect::new(0.0, ceiling_db, 50.0, sample_rate);
        let ceiling_linear = 10f32.powf(ceiling_db / 20.0);

        // A quiet bed with a sharp, short spike well above the ceiling in the middle.
        let mut l = vec![0.1f32; 2000];
        for s in l.iter_mut().skip(900).take(20) {
            *s = 2.0;
        }
        let mut r = l.clone();
        limiter.process(&mut l, &mut r);

        let peak = l.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
        assert!(
            peak <= ceiling_linear + 1e-4,
            "expected no output sample to exceed the ceiling ({ceiling_linear}), got peak {peak}"
        );
    }

    #[test]
    fn limiter_passes_quiet_signal_through_near_unity() {
        let sample_rate = 44100.0;
        let mut limiter = LimiterEffect::new(0.0, -1.0, 50.0, sample_rate);
        let mut l = vec![0.05f32; 2000];
        let mut r = l.clone();
        limiter.process(&mut l, &mut r);
        assert!(
            (l[1500] - 0.05).abs() < 0.001,
            "expected a signal well under the ceiling to pass through mostly unchanged, got {}",
            l[1500]
        );
    }
}
