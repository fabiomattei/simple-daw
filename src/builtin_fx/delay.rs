//! Stereo delay/echo, dual-mono: each channel gets its own tap into its own delay line, sharing
//! only the time/feedback/mix parameters (see `builtin_fx`'s module doc for the dual-mono pattern
//! this and its sibling effects follow).

/// Longest delay time the UI allows (see `main.rs`'s slider range) — the ring buffer is sized for
/// this once at creation so changing `time_ms` at runtime never needs a reallocation.
const MAX_DELAY_SECONDS: f32 = 2.0;

struct DelayChannel {
    buffer: Vec<f32>,
    write_pos: usize,
}

impl DelayChannel {
    fn new(buffer_len: usize) -> Self {
        Self {
            buffer: vec![0.0; buffer_len],
            write_pos: 0,
        }
    }

    fn process(&mut self, buf: &mut [f32], delay_samples: usize, feedback: f32, mix: f32) {
        let buffer_len = self.buffer.len();
        for sample in buf.iter_mut() {
            let read_pos = (self.write_pos + buffer_len - delay_samples) % buffer_len;
            let delayed = self.buffer[read_pos];
            let input = *sample;
            self.buffer[self.write_pos] = input + delayed * feedback;
            self.write_pos = (self.write_pos + 1) % buffer_len;
            *sample = input * (1.0 - mix) + delayed * mix;
        }
    }
}

pub(crate) struct DelayEffect {
    pub time_ms: f32,
    pub feedback: f32,
    pub mix: f32,
    left: DelayChannel,
    right: DelayChannel,
    sample_rate: f32,
}

impl DelayEffect {
    pub(super) fn new(time_ms: f32, feedback: f32, mix: f32, sample_rate: f32) -> Self {
        let buffer_len = ((MAX_DELAY_SECONDS * sample_rate) as usize).max(1);
        Self {
            time_ms,
            feedback,
            mix,
            left: DelayChannel::new(buffer_len),
            right: DelayChannel::new(buffer_len),
            sample_rate,
        }
    }

    pub(super) fn process(&mut self, l: &mut [f32], r: &mut [f32]) {
        let buffer_len = self.left.buffer.len();
        let delay_samples = ((self.time_ms.max(0.0) / 1000.0) * self.sample_rate) as usize;
        let delay_samples = delay_samples.clamp(1, buffer_len - 1);
        let feedback = self.feedback.clamp(0.0, 0.98);
        let mix = self.mix.clamp(0.0, 1.0);
        self.left.process(l, delay_samples, feedback, mix);
        self.right.process(r, delay_samples, feedback, mix);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delay_echoes_an_impulse_after_the_delay_time() {
        // 1kHz sample rate makes "10ms" a round 10 samples, easy to check exactly.
        let mut delay = DelayEffect::new(10.0, 0.5, 1.0, 1000.0);
        let mut l = vec![0.0f32; 30];
        let mut r = vec![0.0f32; 30];
        l[0] = 1.0;
        r[0] = 1.0;
        delay.process(&mut l, &mut r);
        // Fully wet (mix = 1.0): the dry impulse itself is replaced by silence at t=0, but its
        // delayed copy appears exactly one delay-time later, independently on each channel.
        assert_eq!(l[0], 0.0);
        assert!(l[10] > 0.0, "expected an echo at the delay time, got {}", l[10]);
        assert_eq!(r[0], 0.0);
        assert!(r[10] > 0.0, "expected an echo at the delay time, got {}", r[10]);
    }
}
