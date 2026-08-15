//! Stereo saturation/distortion. Stateless per sample, so unlike its sibling effects there's no
//! per-channel state to duplicate — each channel just runs the same memoryless transform.

pub(crate) struct DistortionEffect {
    pub drive: f32,
    pub mix: f32,
}

impl DistortionEffect {
    pub(super) fn new(drive: f32, mix: f32) -> Self {
        Self { drive, mix }
    }

    fn process_channel(&self, buf: &mut [f32]) {
        let drive = self.drive.max(1.0);
        let mix = self.mix.clamp(0.0, 1.0);
        for sample in buf.iter_mut() {
            let wet = (*sample * drive).tanh();
            *sample = *sample * (1.0 - mix) + wet * mix;
        }
    }

    pub(super) fn process(&mut self, l: &mut [f32], r: &mut [f32]) {
        self.process_channel(l);
        self.process_channel(r);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distortion_keeps_bounded_input_bounded() {
        let mut distortion = DistortionEffect::new(20.0, 1.0);
        let mut l: Vec<f32> = (0..64).map(|i| (i as f32 * 0.3).sin()).collect();
        let mut r = l.clone();
        distortion.process(&mut l, &mut r);
        for sample in l.into_iter().chain(r) {
            assert!(
                sample.abs() <= 1.0,
                "tanh drive should saturate, not blow up: {sample}"
            );
        }
    }
}
