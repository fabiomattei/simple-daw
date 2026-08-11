//! Procedurally generated wavetables backing the Wave synth engine (see `model::WaveParams`).
//! There's no access to any commercial synth's actual factory tables, so instead of loading
//! files, every table here is synthesized once from a hand-picked harmonic spectrum per frame —
//! see the `*_table` functions at the bottom. Mirrors `sample.rs`'s role for `Lane`: this module
//! owns real signal-generation logic and hands `model.rs` just an id enum to store per-track,
//! exactly like `Lane` stores an `Arc<SampleBuffer>` rather than knowing how WAV decoding works.

use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

/// Samples in one single-cycle frame. 1024 is a common middle-ground wavetable frame length —
/// enough resolution for audio, cheap enough that generating a handful of tables at startup is
/// instant.
const FRAME_LEN: usize = 1024;
/// Morph frames per table — `WaveParams::osc1_position` (0..1) scans across these.
const FRAMES_PER_TABLE: usize = 8;
/// Harmonic-count ceilings for each precomputed anti-aliased mip, highest-fidelity first. A voice
/// picks the highest cap that still stays under Nyquist for its pitch (see `choose_mip_level`),
/// so high notes fall back to a duller (but non-aliased) copy of the same table instead of
/// foldover noise. Deliberately coarse (4 levels, not a cap per octave) — good enough for a
/// groovebox that also has to stay cheap to synthesize at startup.
const MIP_HARMONIC_CAPS: [usize; 4] = [96, 32, 12, 4];

/// Which built-in wavetable a `WaveParams` oscillator reads from — see the module doc comment for
/// why these are generated, not loaded. Pure data; the actual tables live in this module's
/// process-wide statics (`tables()`), generated lazily and shared by every voice/track.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum WavetableId {
    /// Sine -> triangle -> saw -> square, blended smoothly across the 8 frames.
    #[default]
    ClassicMorph,
    /// Same harmonic falloff shape (1/n^p) at every frame, with p sweeping from bright/buzzy to
    /// dark/mellow.
    HarmonicSweep,
    /// A saw-like falloff with a pair of resonant harmonic "bumps" (formants) that sweep upward
    /// across frames, vaguely vowel-like.
    Formant,
    /// Sweeps a pulse wave's duty cycle from a wide ~45% pulse down to a narrow ~5% pulse —
    /// mirroring the fixed duty-cycle steps classic 8-bit consoles' pulse channels offered, a
    /// deliberate nod to this app's video-game-music focus.
    Chip,
}

impl WavetableId {
    pub const ALL: [WavetableId; 4] = [
        WavetableId::ClassicMorph,
        WavetableId::HarmonicSweep,
        WavetableId::Formant,
        WavetableId::Chip,
    ];

    /// Display name for this table in the engine-selection UI.
    pub fn label(self) -> &'static str {
        match self {
            WavetableId::ClassicMorph => "Classic Morph",
            WavetableId::HarmonicSweep => "Harmonic Sweep",
            WavetableId::Formant => "Formant",
            WavetableId::Chip => "Chip",
        }
    }

    fn index(self) -> usize {
        match self {
            WavetableId::ClassicMorph => 0,
            WavetableId::HarmonicSweep => 1,
            WavetableId::Formant => 2,
            WavetableId::Chip => 3,
        }
    }
}

/// How a `WaveParams` oscillator's phase-warp works, applied before reading the wavetable — an
/// approximation of Serum's warp modes as simple phase-domain remaps, not a reproduction of its
/// actual DSP. `Off` reads the table at a plain linear phase, unchanged from a bare wavetable
/// oscillator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum WaveWarpMode {
    #[default]
    Off,
    /// Skews playback speed unevenly through the cycle (a power-curve phase remap): low amounts
    /// barely bend it, high amounts spend most of the cycle near phase 0 then rush through the
    /// rest.
    Bend,
    /// Restarts the wavetable read partway through the cycle at a rate scaled by `amount`, like a
    /// hard-sync applied to the table read instead of a second oscillator.
    Sync,
    /// Folds the second half of the cycle back over the first around a fold point that moves
    /// with `amount`.
    Mirror,
    /// Self-modulates the read position with a fast internal sine, like an FM index applied to
    /// the table read.
    Fm,
}

/// Applies `mode` to a raw `0..1` phase before it's used to read a wavetable — see `WaveWarpMode`.
/// `amount` is `0..1`. Factored out to a free function (rather than a `Wavetable` method) since it
/// only transforms the read position, never touches table data.
pub fn warp_phase(phase: f32, mode: WaveWarpMode, amount: f32) -> f32 {
    let phase = phase.rem_euclid(1.0);
    let amount = amount.clamp(0.0, 1.0);
    let warped = match mode {
        WaveWarpMode::Off => phase,
        WaveWarpMode::Bend => {
            let k = 1.0 + amount * 3.0;
            phase.powf(k)
        }
        WaveWarpMode::Sync => {
            let ratio = 1.0 + amount * 7.0;
            phase * ratio
        }
        WaveWarpMode::Mirror => {
            let fold = 0.5 + amount * 0.49;
            if phase < fold {
                phase / fold * 0.5
            } else {
                0.5 + (1.0 - (phase - fold) / (1.0 - fold)) * 0.5
            }
        }
        WaveWarpMode::Fm => {
            let index = amount * 3.0;
            phase + index * (phase * std::f32::consts::TAU).sin() * 0.15
        }
    };
    // Bend/Sync/Fm can land outside `[0, 1)` (an out-of-range exponent, a sync ratio > 1, or FM
    // overshoot) and Mirror's own last step can land exactly on 1.0 — wrap once here instead of
    // in every branch above.
    warped.rem_euclid(1.0)
}

/// One harmonic amplitude spectrum for a single wavetable frame; index 0 = fundamental.
type Spectrum = Vec<f32>;

fn normalize_peak(buf: &mut [f32]) {
    let peak = buf.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    if peak > 1e-6 {
        for v in buf.iter_mut() {
            *v /= peak;
        }
    }
}

/// Sums `spectrum`'s first `harmonic_cap` harmonics into one `FRAME_LEN`-sample cycle, then
/// peak-normalizes so every mip level of every frame reaches a consistent, full-scale level
/// regardless of how many harmonics went into it.
fn synthesize_from_spectrum(spectrum: &[f32], harmonic_cap: usize) -> Vec<f32> {
    let cap = harmonic_cap.min(spectrum.len());
    let mut out = vec![0.0f32; FRAME_LEN];
    for (h, &amp) in spectrum.iter().enumerate().take(cap) {
        if amp.abs() < 1e-9 {
            continue;
        }
        let harmonic = (h + 1) as f32;
        let w = std::f32::consts::TAU * harmonic / FRAME_LEN as f32;
        for (i, sample) in out.iter_mut().enumerate() {
            *sample += amp * (w * i as f32).sin();
        }
    }
    normalize_peak(&mut out);
    out
}

fn blend_spectra(a: &[f32], b: &[f32], t: f32) -> Spectrum {
    a.iter().zip(b).map(|(x, y)| x + (y - x) * t).collect()
}

/// Harmonic spectrum of a classic waveform shape: 0 = sine, 1 = triangle, 2 = saw, 3 = square.
/// Standard Fourier-series coefficients for each, truncated to `harmonics` terms.
fn classic_spectrum(kind: usize, harmonics: usize) -> Spectrum {
    let mut s = vec![0.0f32; harmonics];
    match kind {
        0 => s[0] = 1.0,
        1 => {
            for n in (1..=harmonics).step_by(2) {
                let sign = if ((n - 1) / 2) % 2 == 0 { 1.0 } else { -1.0 };
                s[n - 1] = sign / (n * n) as f32;
            }
        }
        2 => {
            for n in 1..=harmonics {
                let sign = if n % 2 == 0 { -1.0 } else { 1.0 };
                s[n - 1] = sign / n as f32;
            }
        }
        3 => {
            for n in (1..=harmonics).step_by(2) {
                s[n - 1] = 1.0 / n as f32;
            }
        }
        _ => unreachable!(),
    }
    s
}

/// A generated wavetable: one set of `FRAMES_PER_TABLE` frames per mip level (see
/// `MIP_HARMONIC_CAPS`), each frame `FRAME_LEN` samples long.
struct Wavetable {
    mips: Vec<Vec<Vec<f32>>>,
}

impl Wavetable {
    fn build(spectra: [Spectrum; FRAMES_PER_TABLE]) -> Self {
        let mips = MIP_HARMONIC_CAPS
            .iter()
            .map(|&cap| {
                spectra
                    .iter()
                    .map(|s| synthesize_from_spectrum(s, cap))
                    .collect()
            })
            .collect();
        Self { mips }
    }

    /// Linearly interpolates between the two nearest frames at `position` (0..1) and the two
    /// nearest samples at `phase` (0..1, wrapped), at the given `mip_level` (clamped in range).
    fn sample(&self, position: f32, phase: f32, mip_level: usize) -> f32 {
        let mip_level = mip_level.min(self.mips.len() - 1);
        let frames = &self.mips[mip_level];
        let position = position.clamp(0.0, 1.0);
        let frame_pos = position * (frames.len() - 1) as f32;
        let frame_index = frame_pos.floor() as usize;
        let frame_frac = frame_pos - frame_index as f32;
        let next_frame_index = (frame_index + 1).min(frames.len() - 1);

        let a = sample_frame(&frames[frame_index], phase);
        let b = sample_frame(&frames[next_frame_index], phase);
        a + (b - a) * frame_frac
    }
}

fn sample_frame(frame: &[f32], phase: f32) -> f32 {
    let phase = phase.rem_euclid(1.0);
    let pos = phase * frame.len() as f32;
    let i0 = pos.floor() as usize % frame.len();
    let i1 = (i0 + 1) % frame.len();
    let frac = pos - pos.floor();
    frame[i0] + (frame[i1] - frame[i0]) * frac
}

fn classic_morph_table() -> Wavetable {
    let cap = MIP_HARMONIC_CAPS[0];
    let shapes = [
        classic_spectrum(0, cap),
        classic_spectrum(1, cap),
        classic_spectrum(2, cap),
        classic_spectrum(3, cap),
    ];
    let spectra: [Spectrum; FRAMES_PER_TABLE] = std::array::from_fn(|i| {
        let t = i as f32 / (FRAMES_PER_TABLE - 1) as f32 * (shapes.len() - 1) as f32;
        let seg = (t.floor() as usize).min(shapes.len() - 2);
        let frac = t - seg as f32;
        blend_spectra(&shapes[seg], &shapes[seg + 1], frac)
    });
    Wavetable::build(spectra)
}

fn harmonic_sweep_table() -> Wavetable {
    let cap = MIP_HARMONIC_CAPS[0];
    let spectra: [Spectrum; FRAMES_PER_TABLE] = std::array::from_fn(|i| {
        let t = i as f32 / (FRAMES_PER_TABLE - 1) as f32;
        let p = 0.5 + t * 2.0;
        (1..=cap).map(|h| 1.0 / (h as f32).powf(p)).collect()
    });
    Wavetable::build(spectra)
}

fn formant_table() -> Wavetable {
    let cap = MIP_HARMONIC_CAPS[0];
    let spectra: [Spectrum; FRAMES_PER_TABLE] = std::array::from_fn(|i| {
        let t = i as f32 / (FRAMES_PER_TABLE - 1) as f32;
        let center1 = 3.0 + t * 10.0;
        let center2 = 9.0 + t * 20.0;
        (1..=cap)
            .map(|h| {
                let h = h as f32;
                let base = 1.0 / h;
                let bump1 = (-((h - center1).powi(2)) / (2.0 * 2.0f32.powi(2))).exp();
                let bump2 = (-((h - center2).powi(2)) / (2.0 * 3.0f32.powi(2))).exp();
                base * (0.25 + 0.75 * bump1.max(bump2))
            })
            .collect()
    });
    Wavetable::build(spectra)
}

fn chip_table() -> Wavetable {
    let cap = MIP_HARMONIC_CAPS[0];
    let duty_cycles = [0.45f32, 0.35, 0.25, 0.18, 0.125, 0.09, 0.065, 0.05];
    let spectra: [Spectrum; FRAMES_PER_TABLE] = std::array::from_fn(|i| {
        let d = duty_cycles[i];
        (1..=cap)
            .map(|h| {
                let h = h as f32;
                (2.0 / (std::f32::consts::PI * h)) * (std::f32::consts::PI * h * d).sin()
            })
            .collect()
    });
    Wavetable::build(spectra)
}

static TABLES: OnceLock<[Wavetable; 4]> = OnceLock::new();

fn tables() -> &'static [Wavetable; 4] {
    TABLES.get_or_init(|| {
        [
            classic_morph_table(),
            harmonic_sweep_table(),
            formant_table(),
            chip_table(),
        ]
    })
}

/// Forces the (one-time, ~instant but non-zero) table generation to happen now, on whatever
/// thread calls this — must run before the first `Wave` note can ever be triggered on the
/// real-time audio callback, since generating tables lazily there would risk an audible dropout.
/// See `AudioEngine::start`.
pub fn warm_up() {
    tables();
}

/// Reads table `id` at morph `position` (0..1) and cycle `phase` (0..1) from the precomputed
/// `mip_level` (see `choose_mip_level`), generating the process-wide table set on first use.
pub fn sample(id: WavetableId, position: f32, phase: f32, mip_level: usize) -> f32 {
    tables()[id.index()].sample(position, phase, mip_level)
}

/// Picks the highest-fidelity mip level that won't alias for a voice at `freq_hz` on
/// `sample_rate` — see `MIP_HARMONIC_CAPS`.
pub fn choose_mip_level(freq_hz: f32, sample_rate: f32) -> usize {
    let nyquist = sample_rate * 0.5;
    let max_harmonics = (nyquist / freq_hz.max(1.0)).floor().max(1.0) as usize;
    MIP_HARMONIC_CAPS
        .iter()
        .position(|&cap| cap <= max_harmonics)
        .unwrap_or(MIP_HARMONIC_CAPS.len() - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_table_produces_bounded_signal_at_every_mip_and_position() {
        for id in WavetableId::ALL {
            for mip in 0..MIP_HARMONIC_CAPS.len() {
                for i in 0..=10 {
                    let position = i as f32 / 10.0;
                    for j in 0..32 {
                        let phase = j as f32 / 32.0;
                        let s = sample(id, position, phase, mip);
                        assert!(
                            s.is_finite(),
                            "{id:?} mip {mip} pos {position} phase {phase} produced {s}"
                        );
                        assert!(
                            s.abs() <= 1.01,
                            "{id:?} mip {mip} pos {position} phase {phase} exceeded unity: {s}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn choose_mip_level_uses_fewer_harmonics_for_higher_pitch() {
        let low = choose_mip_level(80.0, 48_000.0);
        let high = choose_mip_level(8_000.0, 48_000.0);
        assert!(
            high > low,
            "a high note should fall back to a coarser (higher-index) mip than a low note"
        );
    }

    #[test]
    fn choose_mip_level_stays_in_bounds_for_extreme_pitch() {
        assert!(choose_mip_level(1.0, 48_000.0) < MIP_HARMONIC_CAPS.len());
        assert!(choose_mip_level(40_000.0, 48_000.0) < MIP_HARMONIC_CAPS.len());
    }

    #[test]
    fn warp_off_is_identity() {
        for i in 0..32 {
            let phase = i as f32 / 32.0;
            assert_eq!(warp_phase(phase, WaveWarpMode::Off, 0.7), phase);
        }
    }

    #[test]
    fn every_warp_mode_stays_in_unit_range() {
        for mode in [
            WaveWarpMode::Bend,
            WaveWarpMode::Sync,
            WaveWarpMode::Mirror,
            WaveWarpMode::Fm,
        ] {
            for i in 0..32 {
                let phase = i as f32 / 32.0;
                for amount in [0.0, 0.3, 0.7, 1.0] {
                    let warped = warp_phase(phase, mode, amount);
                    assert!(
                        (0.0..1.0).contains(&warped),
                        "{mode:?} amount {amount} phase {phase} -> {warped}"
                    );
                }
            }
        }
    }
}
