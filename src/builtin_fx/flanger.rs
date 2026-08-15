//! Stereo flanger, dual-mono (see `builtin_fx`'s module doc).

/// Longest delay range a flanger's LFO sweeps — much shorter than chorus's, which combined with
/// the feedback path below is what gives a flanger its more resonant, metallic sweep.
const MAX_FLANGER_DELAY_MS: f32 = 10.0;

struct FlangerChannel {
    buffer: Vec<f32>,
    write_pos: usize,
    lfo_phase: f32,
}

impl FlangerChannel {
    fn new(buffer_len: usize) -> Self {
        Self {
            buffer: vec![0.0; buffer_len],
            write_pos: 0,
            lfo_phase: 0.0,
        }
    }

    fn process(&mut self, buf: &mut [f32], depth_samples: f32, feedback: f32, mix: f32, phase_inc: f32) {
        let buffer_len = self.buffer.len();
        for sample in buf.iter_mut() {
            let input = *sample;
            let lfo = 0.5 * (1.0 + self.lfo_phase.sin());
            let delay_samples = (depth_samples * lfo).max(1.0);
            let mut read_pos = self.write_pos as f32 - delay_samples;
            if read_pos < 0.0 {
                read_pos += buffer_len as f32;
            }
            let idx0 = read_pos as usize % buffer_len;
            let idx1 = (idx0 + 1) % buffer_len;
            let frac = read_pos.fract();
            let delayed = self.buffer[idx0] * (1.0 - frac) + self.buffer[idx1] * frac;
            // Unlike chorus, the delayed (wet) signal feeds back into the buffer, not just the
            // dry input — this is what gives a flanger its resonant comb-filter character.
            self.buffer[self.write_pos] = input + delayed * feedback;
            self.write_pos = (self.write_pos + 1) % buffer_len;
            self.lfo_phase += phase_inc;
            if self.lfo_phase > 2.0 * std::f32::consts::PI {
                self.lfo_phase -= 2.0 * std::f32::consts::PI;
            }
            *sample = input * (1.0 - mix) + delayed * mix;
        }
    }
}

pub(crate) struct FlangerEffect {
    pub rate_hz: f32,
    pub depth_ms: f32,
    pub feedback: f32,
    pub mix: f32,
    left: FlangerChannel,
    right: FlangerChannel,
    sample_rate: f32,
}

impl FlangerEffect {
    pub(super) fn new(rate_hz: f32, depth_ms: f32, feedback: f32, mix: f32, sample_rate: f32) -> Self {
        let buffer_len = ((MAX_FLANGER_DELAY_MS / 1000.0 * sample_rate) as usize).max(4);
        Self {
            rate_hz,
            depth_ms,
            feedback,
            mix,
            left: FlangerChannel::new(buffer_len),
            right: FlangerChannel::new(buffer_len),
            sample_rate,
        }
    }

    pub(super) fn process(&mut self, l: &mut [f32], r: &mut [f32]) {
        let buffer_len = self.left.buffer.len();
        let depth_samples =
            (self.depth_ms.max(0.0) / 1000.0 * self.sample_rate).min((buffer_len - 2) as f32);
        let feedback = self.feedback.clamp(0.0, 0.95);
        let mix = self.mix.clamp(0.0, 1.0);
        let phase_inc = 2.0 * std::f32::consts::PI * self.rate_hz.max(0.0) / self.sample_rate;
        self.left.process(l, depth_samples, feedback, mix, phase_inc);
        self.right.process(r, depth_samples, feedback, mix, phase_inc);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flanger_modulates_a_sustained_tone_away_from_the_dry_signal() {
        let mut flanger = FlangerEffect::new(3.0, 4.0, 0.7, 1.0, 1000.0);
        let dry: Vec<f32> = (0..500).map(|i| (i as f32 * 0.05).sin()).collect();
        let mut l = dry.clone();
        let mut r = dry.clone();
        flanger.process(&mut l, &mut r);
        let differs = l
            .iter()
            .zip(dry.iter())
            .any(|(wet, dry)| (wet - dry).abs() > 1e-4);
        assert!(
            differs,
            "fully-wet flanger output should differ from the dry input"
        );
        assert!(
            l.iter().chain(r.iter()).all(|s| s.is_finite()),
            "flanger feedback should never blow up to NaN/infinite"
        );
    }
}
