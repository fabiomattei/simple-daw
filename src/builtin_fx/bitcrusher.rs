//! Stereo bit/sample-rate crusher, dual-mono (see `builtin_fx`'s module doc).

struct BitcrusherChannel {
    hold_value: f32,
    hold_counter: u32,
}

impl BitcrusherChannel {
    fn new() -> Self {
        Self {
            hold_value: 0.0,
            hold_counter: 0,
        }
    }

    fn process(&mut self, buf: &mut [f32], divisor: u32, levels: f32, mix: f32) {
        for sample in buf.iter_mut() {
            if self.hold_counter == 0 {
                self.hold_value = *sample;
                self.hold_counter = divisor;
            }
            self.hold_counter -= 1;
            let crushed = (self.hold_value * levels).round() / levels;
            *sample = *sample * (1.0 - mix) + crushed * mix;
        }
    }
}

pub(crate) struct BitcrusherEffect {
    pub bit_depth: f32,
    pub rate_divisor: u32,
    pub mix: f32,
    left: BitcrusherChannel,
    right: BitcrusherChannel,
}

impl BitcrusherEffect {
    pub(super) fn new(bit_depth: f32, rate_divisor: u32, mix: f32) -> Self {
        Self {
            bit_depth,
            rate_divisor,
            mix,
            left: BitcrusherChannel::new(),
            right: BitcrusherChannel::new(),
        }
    }

    pub(super) fn process(&mut self, l: &mut [f32], r: &mut [f32]) {
        let divisor = self.rate_divisor.max(1);
        let levels = 2f32.powf(self.bit_depth.clamp(1.0, 16.0));
        let mix = self.mix.clamp(0.0, 1.0);
        self.left.process(l, divisor, levels, mix);
        self.right.process(r, divisor, levels, mix);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitcrusher_holds_each_sample_for_rate_divisor_steps() {
        let mut crusher = BitcrusherEffect::new(16.0, 4, 1.0);
        let mut l: Vec<f32> = (0..8).map(|i| i as f32 / 8.0).collect();
        let mut r = l.clone();
        crusher.process(&mut l, &mut r);
        // rate_divisor = 4: the first 4 output samples all hold the first input's (quantized)
        // value, then the 5th sample resamples and holds a new value.
        assert_eq!(l[0], l[1]);
        assert_eq!(l[1], l[2]);
        assert_eq!(l[2], l[3]);
        assert_ne!(l[3], l[4]);
        assert_eq!(r[0], r[1]);
        assert_ne!(r[3], r[4]);
    }
}
