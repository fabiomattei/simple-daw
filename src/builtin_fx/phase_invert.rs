//! Polarity (phase) inversion, independently switchable per channel — used for stereo
//! phase-troubleshooting (e.g. correcting an out-of-phase mic pair) rather than as a tone-shaping
//! effect, so it carries no dry/wet mix: it's either inverted or it isn't.

pub(crate) struct PhaseInvertEffect {
    pub invert_left: bool,
    pub invert_right: bool,
}

impl PhaseInvertEffect {
    pub(super) fn new(invert_left: bool, invert_right: bool) -> Self {
        Self {
            invert_left,
            invert_right,
        }
    }

    pub(super) fn process(&mut self, l: &mut [f32], r: &mut [f32]) {
        if self.invert_left {
            for sample in l.iter_mut() {
                *sample = -*sample;
            }
        }
        if self.invert_right {
            for sample in r.iter_mut() {
                *sample = -*sample;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverts_only_the_channels_flagged() {
        let mut effect = PhaseInvertEffect::new(true, false);
        let mut l = vec![0.5, -0.25, 1.0];
        let mut r = vec![0.5, -0.25, 1.0];
        effect.process(&mut l, &mut r);
        assert_eq!(l, vec![-0.5, 0.25, -1.0]);
        assert_eq!(r, vec![0.5, -0.25, 1.0]);
    }
}
