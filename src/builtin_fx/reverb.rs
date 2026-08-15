//! Stereo Schroeder reverb, dual-mono: each channel gets its own bank of comb/allpass filters
//! (see `builtin_fx`'s module doc), so the tail isn't perfectly correlated the instant its input
//! channels diverge (e.g. once real per-track panning lands upstream), but no stereo-decorrelated
//! tuning is applied beyond that — both channels start from the same comb/allpass lengths.

/// One tap of a Schroeder comb filter: a delay line with feedback plus a one-pole lowpass in the
/// feedback path (the standard Freeverb-style trick for damping high frequencies as the tail
/// decays, instead of ringing forever at a flat frequency response).
struct CombFilter {
    buffer: Vec<f32>,
    pos: usize,
    filter_store: f32,
}

impl CombFilter {
    fn new(len_samples: usize) -> Self {
        Self {
            buffer: vec![0.0; len_samples.max(1)],
            pos: 0,
            filter_store: 0.0,
        }
    }

    fn process(&mut self, input: f32, feedback: f32, damping: f32) -> f32 {
        let output = self.buffer[self.pos];
        self.filter_store = output * (1.0 - damping) + self.filter_store * damping;
        self.buffer[self.pos] = input + self.filter_store * feedback;
        self.pos = (self.pos + 1) % self.buffer.len();
        output
    }
}

/// A Schroeder allpass filter: diffuses the comb filters' output into a denser, less metallic
/// tail without adding its own coloration (flat frequency response by construction).
struct AllpassFilter {
    buffer: Vec<f32>,
    pos: usize,
}

impl AllpassFilter {
    fn new(len_samples: usize) -> Self {
        Self {
            buffer: vec![0.0; len_samples.max(1)],
            pos: 0,
        }
    }

    fn process(&mut self, input: f32) -> f32 {
        const FEEDBACK: f32 = 0.5;
        let buffered = self.buffer[self.pos];
        let output = buffered - input;
        self.buffer[self.pos] = input + buffered * FEEDBACK;
        self.pos = (self.pos + 1) % self.buffer.len();
        output
    }
}

/// Comb filter delay lengths in milliseconds, tuned (mutually prime-ish, spread out) the way
/// Freeverb's classic 44.1kHz sample counts are — scaled to the actual sample rate at creation
/// time rather than hardcoded as sample counts, so the same character holds at any device rate.
const COMB_TUNINGS_MS: [f32; 4] = [25.31, 26.94, 28.96, 30.75];
const ALLPASS_TUNINGS_MS: [f32; 2] = [12.61, 9.68];

struct ReverbChannel {
    combs: Vec<CombFilter>,
    allpasses: Vec<AllpassFilter>,
}

impl ReverbChannel {
    fn new(sample_rate: f32) -> Self {
        let combs = COMB_TUNINGS_MS
            .iter()
            .map(|ms| CombFilter::new(((ms / 1000.0) * sample_rate) as usize))
            .collect();
        let allpasses = ALLPASS_TUNINGS_MS
            .iter()
            .map(|ms| AllpassFilter::new(((ms / 1000.0) * sample_rate) as usize))
            .collect();
        Self { combs, allpasses }
    }

    fn process(&mut self, buf: &mut [f32], feedback: f32, damping: f32, mix: f32) {
        for sample in buf.iter_mut() {
            let input = *sample;
            let mut wet = 0.0;
            for comb in self.combs.iter_mut() {
                wet += comb.process(input, feedback, damping);
            }
            wet /= self.combs.len() as f32;
            for allpass in self.allpasses.iter_mut() {
                wet = allpass.process(wet);
            }
            *sample = input * (1.0 - mix) + wet * mix;
        }
    }
}

pub(crate) struct ReverbEffect {
    pub room_size: f32,
    pub damping: f32,
    pub mix: f32,
    left: ReverbChannel,
    right: ReverbChannel,
}

impl ReverbEffect {
    pub(super) fn new(room_size: f32, damping: f32, mix: f32, sample_rate: f32) -> Self {
        Self {
            room_size,
            damping,
            mix,
            left: ReverbChannel::new(sample_rate),
            right: ReverbChannel::new(sample_rate),
        }
    }

    pub(super) fn process(&mut self, l: &mut [f32], r: &mut [f32]) {
        // room_size maps to comb feedback: higher feedback means a longer-ringing tail.
        let feedback = 0.7 + self.room_size.clamp(0.0, 1.0) * 0.28;
        let damping = self.damping.clamp(0.0, 1.0);
        let mix = self.mix.clamp(0.0, 1.0);
        self.left.process(l, feedback, damping, mix);
        self.right.process(r, feedback, damping, mix);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reverb_produces_a_tail_after_the_input_stops_and_never_produces_nan() {
        let mut reverb = ReverbEffect::new(0.8, 0.3, 1.0, 44100.0);
        let mut l = vec![0.0f32; 4000];
        let mut r = vec![0.0f32; 4000];
        l[0] = 1.0;
        r[0] = 1.0;
        reverb.process(&mut l, &mut r);
        let tail_energy: f32 = l[100..4000].iter().map(|s| s.abs()).sum();
        assert!(
            tail_energy > 0.0,
            "expected reverb tail energy after the impulse"
        );
        assert!(
            l.iter().chain(r.iter()).all(|s| s.is_finite()),
            "reverb output should never be NaN/infinite"
        );
    }
}
