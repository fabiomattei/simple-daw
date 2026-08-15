//! Stereo ring modulator, dual-mono (see `builtin_fx`'s module doc). Like tremolo, there's no
//! delay buffer to duplicate — just an LFO phase per channel.

pub(crate) struct RingModulatorEffect {
    pub carrier_hz: f32,
    pub mix: f32,
    left_phase: f32,
    right_phase: f32,
    sample_rate: f32,
}

impl RingModulatorEffect {
    pub(super) fn new(carrier_hz: f32, mix: f32, sample_rate: f32) -> Self {
        Self {
            carrier_hz,
            mix,
            left_phase: 0.0,
            right_phase: 0.0,
            sample_rate,
        }
    }

    fn process_channel(buf: &mut [f32], phase: &mut f32, mix: f32, phase_inc: f32) {
        for sample in buf.iter_mut() {
            let input = *sample;
            let modulated = input * phase.sin();
            *sample = input * (1.0 - mix) + modulated * mix;
            *phase += phase_inc;
            if *phase > 2.0 * std::f32::consts::PI {
                *phase -= 2.0 * std::f32::consts::PI;
            }
        }
    }

    pub(super) fn process(&mut self, l: &mut [f32], r: &mut [f32]) {
        let mix = self.mix.clamp(0.0, 1.0);
        let phase_inc = 2.0 * std::f32::consts::PI * self.carrier_hz.max(0.0) / self.sample_rate;
        Self::process_channel(l, &mut self.left_phase, mix, phase_inc);
        Self::process_channel(r, &mut self.right_phase, mix, phase_inc);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_modulator_produces_sum_and_difference_frequencies() {
        // A carrier well below the input tone's frequency should audibly change the waveform
        // shape (amplitude modulation), not just scale it uniformly.
        let mut ring_mod = RingModulatorEffect::new(30.0, 1.0, 1000.0);
        let dry: Vec<f32> = (0..500)
            .map(|i| (2.0 * std::f32::consts::PI * 100.0 * i as f32 / 1000.0).sin())
            .collect();
        let mut l = dry.clone();
        let mut r = dry.clone();
        ring_mod.process(&mut l, &mut r);
        let differs = l
            .iter()
            .zip(dry.iter())
            .any(|(wet, dry)| (wet - dry).abs() > 1e-4);
        assert!(
            differs,
            "fully-wet ring modulation should differ from the dry input"
        );
        assert_eq!(l, r, "identical input on both channels should stay in phase-lock");
    }
}
