//! Stereo phaser, dual-mono: each channel gets its own allpass cascade and feedback tap (see
//! `builtin_fx`'s module doc).

/// One first-order allpass stage of a phaser, with a time-varying coefficient driven by the
/// shared LFO in `PhaserChannel::process`.
struct PhaserAllpassStage {
    x1: f32,
    y1: f32,
}

impl PhaserAllpassStage {
    fn new() -> Self {
        Self { x1: 0.0, y1: 0.0 }
    }

    fn process(&mut self, input: f32, coeff: f32) -> f32 {
        let output = -coeff * input + self.x1 + coeff * self.y1;
        self.x1 = input;
        self.y1 = output;
        output
    }
}

const PHASER_STAGE_COUNT: usize = 4;

struct PhaserChannel {
    stages: Vec<PhaserAllpassStage>,
    lfo_phase: f32,
    feedback_sample: f32,
}

impl PhaserChannel {
    fn new() -> Self {
        Self {
            stages: (0..PHASER_STAGE_COUNT)
                .map(|_| PhaserAllpassStage::new())
                .collect(),
            lfo_phase: 0.0,
            feedback_sample: 0.0,
        }
    }

    fn process(&mut self, buf: &mut [f32], depth: f32, feedback: f32, mix: f32, phase_inc: f32) {
        for sample in buf.iter_mut() {
            let input = *sample;
            let lfo = 0.5 * (1.0 + self.lfo_phase.sin());
            // Keep the allpass coefficient well inside (-1, 1) at all times so every stage stays
            // stable across the full sweep.
            let coeff = -0.9 + depth * 1.8 * lfo;
            let mut signal = input + self.feedback_sample * feedback;
            for stage in self.stages.iter_mut() {
                signal = stage.process(signal, coeff);
            }
            self.feedback_sample = signal;
            self.lfo_phase += phase_inc;
            if self.lfo_phase > 2.0 * std::f32::consts::PI {
                self.lfo_phase -= 2.0 * std::f32::consts::PI;
            }
            *sample = input * (1.0 - mix) + signal * mix;
        }
    }
}

pub(crate) struct PhaserEffect {
    pub rate_hz: f32,
    pub depth: f32,
    pub feedback: f32,
    pub mix: f32,
    left: PhaserChannel,
    right: PhaserChannel,
    sample_rate: f32,
}

impl PhaserEffect {
    pub(super) fn new(rate_hz: f32, depth: f32, feedback: f32, mix: f32, sample_rate: f32) -> Self {
        Self {
            rate_hz,
            depth,
            feedback,
            mix,
            left: PhaserChannel::new(),
            right: PhaserChannel::new(),
            sample_rate,
        }
    }

    pub(super) fn process(&mut self, l: &mut [f32], r: &mut [f32]) {
        let depth = self.depth.clamp(0.0, 1.0);
        let feedback = self.feedback.clamp(0.0, 0.95);
        let mix = self.mix.clamp(0.0, 1.0);
        let phase_inc = 2.0 * std::f32::consts::PI * self.rate_hz.max(0.0) / self.sample_rate;
        self.left.process(l, depth, feedback, mix, phase_inc);
        self.right.process(r, depth, feedback, mix, phase_inc);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phaser_sweeps_without_blowing_up() {
        let mut phaser = PhaserEffect::new(0.5, 1.0, 0.5, 0.5, 44100.0);
        let dry: Vec<f32> = (0..4000).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut l = dry.clone();
        let mut r = dry;
        phaser.process(&mut l, &mut r);
        assert!(
            l.iter().chain(r.iter()).all(|s| s.is_finite()),
            "phaser allpass cascade should stay stable"
        );
        assert!(
            l.iter().any(|s| s.abs() > 1e-4),
            "expected the phaser to still produce a non-silent signal"
        );
    }
}
