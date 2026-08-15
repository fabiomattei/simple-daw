//! Stereo tremolo, dual-mono (see `builtin_fx`'s module doc). No delay buffer to duplicate here —
//! just an LFO phase per channel, kept as two plain fields rather than a wrapper struct since
//! there's nothing else to group them with.

pub(crate) struct TremoloEffect {
    pub rate_hz: f32,
    pub depth: f32,
    left_phase: f32,
    right_phase: f32,
    sample_rate: f32,
}

impl TremoloEffect {
    pub(super) fn new(rate_hz: f32, depth: f32, sample_rate: f32) -> Self {
        Self {
            rate_hz,
            depth,
            left_phase: 0.0,
            right_phase: 0.0,
            sample_rate,
        }
    }

    fn process_channel(buf: &mut [f32], phase: &mut f32, depth: f32, phase_inc: f32) {
        for sample in buf.iter_mut() {
            let lfo = 0.5 * (1.0 + phase.sin());
            let gain = 1.0 - depth * (1.0 - lfo);
            *sample *= gain;
            *phase += phase_inc;
            if *phase > 2.0 * std::f32::consts::PI {
                *phase -= 2.0 * std::f32::consts::PI;
            }
        }
    }

    pub(super) fn process(&mut self, l: &mut [f32], r: &mut [f32]) {
        let depth = self.depth.clamp(0.0, 1.0);
        let phase_inc = 2.0 * std::f32::consts::PI * self.rate_hz.max(0.0) / self.sample_rate;
        Self::process_channel(l, &mut self.left_phase, depth, phase_inc);
        Self::process_channel(r, &mut self.right_phase, depth, phase_inc);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tremolo_modulates_amplitude_between_one_and_one_minus_depth() {
        let sample_rate = 1000.0;
        let mut tremolo = TremoloEffect::new(2.0, 0.6, sample_rate);
        let mut l = vec![1.0f32; 1000];
        let mut r = vec![1.0f32; 1000];
        tremolo.process(&mut l, &mut r);
        let max = l.iter().cloned().fold(f32::MIN, f32::max);
        let min = l.iter().cloned().fold(f32::MAX, f32::min);
        assert!(
            max > 0.95,
            "expected the LFO peak to leave amplitude near unity, got {max}"
        );
        assert!(
            min < 0.45,
            "expected the LFO trough to cut amplitude by ~depth, got {min}"
        );
        assert_eq!(l, r, "identical input on both channels should stay in phase-lock");
    }
}
