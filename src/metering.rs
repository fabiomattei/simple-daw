//! Peak/RMS/LUFS metering for the mixer UI. The real-time audio callback (`audio.rs`) owns one
//! `LoudnessMeter` per track plus one for the master bus, feeds post-fader/pan samples through
//! `LoudnessMeter::process` every buffer, and publishes the result into a `ChannelMeterAtomics` —
//! the same lock-free publish idiom `audio_input.rs`'s `peak_bits: Arc<AtomicU32>` uses for its
//! input-level meter, just with more fields. `MeterHandles` mirrors `plugin_host::TrackEffectSlots`
//! (`Arc<Mutex<Vec<Arc<ChannelMeterAtomics>>>>`, one entry per track) and its `MasterEffectSlots`
//! sibling (the same type, pinned to a single row) — meters and effect chains resize, lock, and get
//! indexed by the UI thread the exact same way.
//!
//! LUFS follows ITU-R BS.1770-4: a two-stage K-weighting pre-filter (high-shelf then high-pass,
//! designed here for the engine's actual sample rate rather than the standard's tabulated 48kHz
//! coefficients), 400ms gating blocks on a 100ms hop, and the two-stage absolute (-70 LUFS) +
//! relative (mean - 10 LU) gated average for integrated loudness. Integrated loudness keeps every
//! gating block's mean square since the last `reset()` (called on transport stop, so it measures
//! "since last playback start") and rescans that history each hop to fold in the moving relative
//! threshold — an O(n) rescan every 100ms, not a fully incremental algorithm. That's fine for this
//! app's realistic session lengths (short game-music loops, not hours of broadcast capture) and is
//! no less real-time-pragmatic than this same audio callback already blocking on `Mutex::lock` for
//! `track_effects`/`master_effects` and cloning the whole `Song` snapshot every buffer.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

/// BS.1770 absolute-gate threshold, in LUFS. Doubles as the floor value shown for a channel with
/// no signal yet (silence and "gated out" are the same displayable state: nothing loud enough to
/// register) — `pub` so the UI can tell "no reading yet" apart from a genuine quiet reading.
pub const SILENCE_LUFS: f32 = -70.0;

const GATING_HOP_MS: f32 = 100.0;
const MOMENTARY_BLOCKS: usize = 4; // 400ms / 100ms hop
const SHORT_TERM_BLOCKS: usize = 30; // 3000ms / 100ms hop
const PEAK_DECAY_MS: f32 = 500.0;
const RMS_INTEGRATION_MS: f32 = 300.0;

/// One channel strip's meter readings, published by the audio thread and polled by the UI thread
/// every frame. `peak_*`/`rms_*` are linear amplitude (0.0..~1.0+, not K-weighted); the `lufs_*`
/// fields are already in LUFS (dB-like, K-weighted, gated per BS.1770-4 for `lufs_integrated`).
#[derive(Clone, Copy, Debug)]
pub struct MeterReadings {
    pub peak_l: f32,
    pub peak_r: f32,
    pub rms_l: f32,
    pub rms_r: f32,
    pub lufs_momentary: f32,
    pub lufs_short_term: f32,
    pub lufs_integrated: f32,
}

impl MeterReadings {
    fn silent() -> Self {
        Self {
            peak_l: 0.0,
            peak_r: 0.0,
            rms_l: 0.0,
            rms_r: 0.0,
            lufs_momentary: SILENCE_LUFS,
            lufs_short_term: SILENCE_LUFS,
            lufs_integrated: SILENCE_LUFS,
        }
    }
}

impl Default for MeterReadings {
    /// The UI's fallback for a track/master strip whose meter hasn't published anything yet (e.g.
    /// a track added this frame, before the audio thread resizes to match).
    fn default() -> Self {
        Self::silent()
    }
}

/// Lock-free published form of `MeterReadings`, one per channel strip. See the module doc for the
/// `MeterHandles` collection this lives in.
pub struct ChannelMeterAtomics {
    peak_l: AtomicU32,
    peak_r: AtomicU32,
    rms_l: AtomicU32,
    rms_r: AtomicU32,
    lufs_momentary: AtomicU32,
    lufs_short_term: AtomicU32,
    lufs_integrated: AtomicU32,
}

impl ChannelMeterAtomics {
    pub fn new() -> Self {
        let silent = MeterReadings::silent();
        Self {
            peak_l: AtomicU32::new(silent.peak_l.to_bits()),
            peak_r: AtomicU32::new(silent.peak_r.to_bits()),
            rms_l: AtomicU32::new(silent.rms_l.to_bits()),
            rms_r: AtomicU32::new(silent.rms_r.to_bits()),
            lufs_momentary: AtomicU32::new(silent.lufs_momentary.to_bits()),
            lufs_short_term: AtomicU32::new(silent.lufs_short_term.to_bits()),
            lufs_integrated: AtomicU32::new(silent.lufs_integrated.to_bits()),
        }
    }

    /// Called from the audio thread once per buffer with that channel's freshly computed readings.
    pub fn publish(&self, readings: &MeterReadings) {
        self.peak_l.store(readings.peak_l.to_bits(), Ordering::Relaxed);
        self.peak_r.store(readings.peak_r.to_bits(), Ordering::Relaxed);
        self.rms_l.store(readings.rms_l.to_bits(), Ordering::Relaxed);
        self.rms_r.store(readings.rms_r.to_bits(), Ordering::Relaxed);
        self.lufs_momentary
            .store(readings.lufs_momentary.to_bits(), Ordering::Relaxed);
        self.lufs_short_term
            .store(readings.lufs_short_term.to_bits(), Ordering::Relaxed);
        self.lufs_integrated
            .store(readings.lufs_integrated.to_bits(), Ordering::Relaxed);
    }

    /// Called from the UI thread every frame — a handful of relaxed atomic loads, safe to poll.
    pub fn snapshot(&self) -> MeterReadings {
        MeterReadings {
            peak_l: f32::from_bits(self.peak_l.load(Ordering::Relaxed)),
            peak_r: f32::from_bits(self.peak_r.load(Ordering::Relaxed)),
            rms_l: f32::from_bits(self.rms_l.load(Ordering::Relaxed)),
            rms_r: f32::from_bits(self.rms_r.load(Ordering::Relaxed)),
            lufs_momentary: f32::from_bits(self.lufs_momentary.load(Ordering::Relaxed)),
            lufs_short_term: f32::from_bits(self.lufs_short_term.load(Ordering::Relaxed)),
            lufs_integrated: f32::from_bits(self.lufs_integrated.load(Ordering::Relaxed)),
        }
    }
}

impl Default for ChannelMeterAtomics {
    fn default() -> Self {
        Self::new()
    }
}

/// One meter slot per track, plus a second instance (pinned to a single row, same convention as
/// `plugin_host::MasterEffectSlots`) for the master bus.
pub type MeterHandles = Arc<Mutex<Vec<Arc<ChannelMeterAtomics>>>>;

/// An empty single-row `MeterHandles`, for the master bus.
pub fn new_master_meter_handles() -> MeterHandles {
    new_track_meter_handles(1)
}

/// A `MeterHandles` with `track_count` freshly published (silent) entries.
pub fn new_track_meter_handles(track_count: usize) -> MeterHandles {
    Arc::new(Mutex::new(
        (0..track_count).map(|_| Arc::new(ChannelMeterAtomics::new())).collect(),
    ))
}

/// A single first- or second-order IIR stage, direct form I. `process` is the only real-time-path
/// method; coefficients are fixed for the stage's lifetime (designed once in `KWeighting::new`).
#[derive(Clone, Copy)]
struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl Biquad {
    fn new(b0: f32, b1: f32, b2: f32, a1: f32, a2: f32) -> Self {
        Self { b0, b1, b2, a1, a2, x1: 0.0, x2: 0.0, y1: 0.0, y2: 0.0 }
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2 - self.a1 * self.y1 - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

/// The BS.1770-4 K-weighting pre-filter for one channel: a high-shelf stage (models the head's
/// acoustic effect at high frequencies) cascaded with a high-pass stage (the "RLB" weighting,
/// approximating equal-loudness contours at low frequencies). Designed for an arbitrary sample
/// rate via the standard's analog-prototype-plus-bilinear-transform formulas (Annex 1), rather
/// than only supporting the standard's tabulated 48kHz coefficients.
#[derive(Clone, Copy)]
struct KWeighting {
    shelf: Biquad,
    highpass: Biquad,
}

impl KWeighting {
    fn new(sample_rate: f32) -> Self {
        Self {
            shelf: Self::design_shelf(sample_rate),
            highpass: Self::design_highpass(sample_rate),
        }
    }

    fn design_shelf(sample_rate: f32) -> Biquad {
        let f0 = 1_681.974_450_955_532_f64;
        let gain_db = 3.999_843_853_97_f64;
        let q = 0.707_175_236_955_419_3_f64;
        let k = (std::f64::consts::PI * f0 / sample_rate as f64).tan();
        let vh = 10f64.powf(gain_db / 20.0);
        let vb = vh.powf(0.499_666_774_154_541_6);
        let a0 = 1.0 + k / q + k * k;
        let b0 = (vh + vb * k / q + k * k) / a0;
        let b1 = 2.0 * (k * k - vh) / a0;
        let b2 = (vh - vb * k / q + k * k) / a0;
        let a1 = 2.0 * (k * k - 1.0) / a0;
        let a2 = (1.0 - k / q + k * k) / a0;
        Biquad::new(b0 as f32, b1 as f32, b2 as f32, a1 as f32, a2 as f32)
    }

    fn design_highpass(sample_rate: f32) -> Biquad {
        let f0 = 38.135_470_876_024_44_f64;
        let q = 0.500_327_037_323_877_3_f64;
        let k = (std::f64::consts::PI * f0 / sample_rate as f64).tan();
        let a0 = 1.0 + k / q + k * k;
        // Unlike the shelf stage, BS.1770's tabulated high-pass coefficients leave b0/b1/b2 as the
        // plain double-zero-at-DC numerator (1, -2, 1) — only a1/a2 are normalized by a0.
        let b0 = 1.0;
        let b1 = -2.0;
        let b2 = 1.0;
        let a1 = 2.0 * (k * k - 1.0) / a0;
        let a2 = (1.0 - k / q + k * k) / a0;
        Biquad::new(b0 as f32, b1 as f32, b2 as f32, a1 as f32, a2 as f32)
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        self.highpass.process(self.shelf.process(x))
    }
}

fn one_pole_decay_coeff(time_ms: f32, sample_rate: f32) -> f32 {
    (-1.0 / (time_ms.max(0.1) / 1000.0 * sample_rate)).exp()
}

fn lufs_from_mean_square(mean_square: f64) -> f32 {
    if mean_square <= 1e-12 {
        SILENCE_LUFS
    } else {
        (-0.691 + 10.0 * mean_square.log10()) as f32
    }
}

/// Per-track (or master) running peak, RMS, and BS.1770 LUFS meter. Owned by the audio thread —
/// see the module doc for how its output reaches the UI. Not `Clone`/`Send`-shared: each track and
/// the master bus gets its own instance, live only inside the real-time callback's closure state.
pub struct LoudnessMeter {
    peak_decay: f32,
    rms_coeff: f32,
    peak_l: f32,
    peak_r: f32,
    rms_ms_l: f32,
    rms_ms_r: f32,

    k_left: KWeighting,
    k_right: KWeighting,
    hop_samples: usize,
    hop_count: usize,
    hop_sum_sq_l: f32,
    hop_sum_sq_r: f32,

    /// Mean-square value of each finished gating block, most recent last, capped at
    /// `SHORT_TERM_BLOCKS` — feeds momentary/short-term LUFS.
    recent_blocks: VecDeque<f64>,
    /// Every finished gating block's mean square since the last `reset()` — feeds integrated LUFS.
    /// Grows for the duration of playback; see the module doc for why that's an accepted trade-off.
    integrated_blocks: Vec<f64>,

    momentary_lufs: f32,
    short_term_lufs: f32,
    integrated_lufs: f32,
}

impl LoudnessMeter {
    pub fn new(sample_rate: f32) -> Self {
        let hop_samples = ((sample_rate * GATING_HOP_MS / 1000.0).round() as usize).max(1);
        Self {
            peak_decay: one_pole_decay_coeff(PEAK_DECAY_MS, sample_rate),
            rms_coeff: one_pole_decay_coeff(RMS_INTEGRATION_MS, sample_rate),
            peak_l: 0.0,
            peak_r: 0.0,
            rms_ms_l: 0.0,
            rms_ms_r: 0.0,
            k_left: KWeighting::new(sample_rate),
            k_right: KWeighting::new(sample_rate),
            hop_samples,
            hop_count: 0,
            hop_sum_sq_l: 0.0,
            hop_sum_sq_r: 0.0,
            recent_blocks: VecDeque::with_capacity(SHORT_TERM_BLOCKS),
            integrated_blocks: Vec::new(),
            momentary_lufs: SILENCE_LUFS,
            short_term_lufs: SILENCE_LUFS,
            integrated_lufs: SILENCE_LUFS,
        }
    }

    /// Clears integrated-loudness history (and every other running reading) — called on transport
    /// stop, so integrated LUFS measures "since the last time playback started," matching how a
    /// loudness-meter plugin is normally used (reset, play the section, read the number).
    pub fn reset(&mut self) {
        self.peak_l = 0.0;
        self.peak_r = 0.0;
        self.rms_ms_l = 0.0;
        self.rms_ms_r = 0.0;
        self.hop_count = 0;
        self.hop_sum_sq_l = 0.0;
        self.hop_sum_sq_r = 0.0;
        self.recent_blocks.clear();
        self.integrated_blocks.clear();
        self.momentary_lufs = SILENCE_LUFS;
        self.short_term_lufs = SILENCE_LUFS;
        self.integrated_lufs = SILENCE_LUFS;
    }

    /// Processes one buffer's worth of post-fader/pan (track) or post-master-FX (master) stereo
    /// samples and returns the freshly updated readings — the caller publishes these into a
    /// `ChannelMeterAtomics`. `l`/`r` must be equal length.
    pub fn process(&mut self, l: &[f32], r: &[f32]) -> MeterReadings {
        for (&xl, &xr) in l.iter().zip(r.iter()) {
            self.peak_l = xl.abs().max(self.peak_l * self.peak_decay);
            self.peak_r = xr.abs().max(self.peak_r * self.peak_decay);
            self.rms_ms_l = self.rms_coeff * self.rms_ms_l + (1.0 - self.rms_coeff) * xl * xl;
            self.rms_ms_r = self.rms_coeff * self.rms_ms_r + (1.0 - self.rms_coeff) * xr * xr;

            let kl = self.k_left.process(xl);
            let kr = self.k_right.process(xr);
            self.hop_sum_sq_l += kl * kl;
            self.hop_sum_sq_r += kr * kr;
            self.hop_count += 1;
            if self.hop_count >= self.hop_samples {
                self.finish_hop();
            }
        }

        MeterReadings {
            peak_l: self.peak_l,
            peak_r: self.peak_r,
            rms_l: self.rms_ms_l.max(0.0).sqrt(),
            rms_r: self.rms_ms_r.max(0.0).sqrt(),
            lufs_momentary: self.momentary_lufs,
            lufs_short_term: self.short_term_lufs,
            lufs_integrated: self.integrated_lufs,
        }
    }

    fn finish_hop(&mut self) {
        let n = self.hop_count as f64;
        let mean_sq_l = self.hop_sum_sq_l as f64 / n;
        let mean_sq_r = self.hop_sum_sq_r as f64 / n;
        // BS.1770 channel weighting G_i is 1.0 for both channels of a stereo pair.
        let block_mean_square = mean_sq_l + mean_sq_r;
        self.hop_sum_sq_l = 0.0;
        self.hop_sum_sq_r = 0.0;
        self.hop_count = 0;

        self.recent_blocks.push_back(block_mean_square);
        while self.recent_blocks.len() > SHORT_TERM_BLOCKS {
            self.recent_blocks.pop_front();
        }
        self.integrated_blocks.push(block_mean_square);

        self.momentary_lufs = lufs_from_mean_square(Self::mean_of_last(&self.recent_blocks, MOMENTARY_BLOCKS));
        self.short_term_lufs = lufs_from_mean_square(Self::mean_of_last(&self.recent_blocks, SHORT_TERM_BLOCKS));
        self.integrated_lufs = Self::compute_integrated(&self.integrated_blocks);
    }

    fn mean_of_last(blocks: &VecDeque<f64>, n: usize) -> f64 {
        let take = n.min(blocks.len());
        if take == 0 {
            return 0.0;
        }
        blocks.iter().rev().take(take).sum::<f64>() / take as f64
    }

    /// BS.1770-4's two-stage gated average: discard blocks quieter than the -70 LUFS absolute
    /// threshold, then discard blocks quieter than (mean of what's left - 10 LU), and average
    /// what remains.
    fn compute_integrated(blocks: &[f64]) -> f32 {
        let absolute_gate_ms = 10f64.powf((SILENCE_LUFS as f64 + 0.691) / 10.0);
        let passed_absolute: Vec<f64> = blocks.iter().copied().filter(|&z| z > absolute_gate_ms).collect();
        if passed_absolute.is_empty() {
            return SILENCE_LUFS;
        }
        let mean_absolute = passed_absolute.iter().sum::<f64>() / passed_absolute.len() as f64;
        let relative_threshold_lufs = -0.691 + 10.0 * mean_absolute.log10() - 10.0;
        let relative_gate_ms = 10f64.powf((relative_threshold_lufs + 0.691) / 10.0);
        let passed_relative: Vec<f64> = passed_absolute.iter().copied().filter(|&z| z > relative_gate_ms).collect();
        if passed_relative.is_empty() {
            return SILENCE_LUFS;
        }
        let mean_relative = passed_relative.iter().sum::<f64>() / passed_relative.len() as f64;
        lufs_from_mean_square(mean_relative)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The K-weighting formulas here are designed for an arbitrary sample rate; at exactly 48kHz
    /// they must reproduce the fixed coefficients tabulated in BS.1770-4 Annex 1 Table 1.
    #[test]
    fn k_weighting_matches_bs1770_reference_coefficients_at_48khz() {
        let shelf = KWeighting::design_shelf(48000.0);
        assert!((shelf.b0 - 1.535_124_9).abs() < 1e-5);
        assert!((shelf.b1 - (-2.691_696_2)).abs() < 1e-5);
        assert!((shelf.b2 - 1.198_392_8).abs() < 1e-5);
        assert!((shelf.a1 - (-1.690_659_3)).abs() < 1e-5);
        assert!((shelf.a2 - 0.732_480_77).abs() < 1e-5);

        let highpass = KWeighting::design_highpass(48000.0);
        assert!((highpass.b0 - 1.0).abs() < 1e-5);
        assert!((highpass.b1 - (-2.0)).abs() < 1e-5);
        assert!((highpass.b2 - 1.0).abs() < 1e-5);
        assert!((highpass.a1 - (-1.990_047_5)).abs() < 1e-5);
        assert!((highpass.a2 - 0.990_072_25).abs() < 1e-5);
    }

    /// A full-scale (0 dBFS) 1kHz sine on a single channel (the other silent, matching the
    /// standard's mono reference scenario) has mean square 0.5, so BS.1770's own formula
    /// (`-0.691 + 10*log10(sum_i G_i * z_i)`) gives `-0.691 + 10*log10(0.5)` ≈ -3.70 LUFS *if*
    /// K-weighting were perfectly flat at 1kHz. This checks the whole pipeline (filter, gating
    /// block accumulation, two-stage gated integration) lands near that self-derived figure — a
    /// generous tolerance absorbs the K-weighting stage's actual (small but nonzero) gain at
    /// 1kHz, so this is a plausibility check, not a bit-exact conformance test.
    #[test]
    fn integrated_lufs_of_full_scale_1khz_mono_sine_is_near_expected() {
        let sample_rate = 48000.0f32;
        let mut meter = LoudnessMeter::new(sample_rate);
        let freq = 1000.0f32;
        let seconds = 2.0f32;
        let total_samples = (sample_rate * seconds) as usize;
        let block = 512;
        let mut phase = 0.0f32;
        let phase_step = std::f32::consts::TAU * freq / sample_rate;
        let mut i = 0;
        while i < total_samples {
            let n = block.min(total_samples - i);
            let mut l = vec![0.0f32; n];
            let r = vec![0.0f32; n];
            for sample in l.iter_mut() {
                *sample = phase.sin();
                phase += phase_step;
            }
            meter.process(&l, &r);
            i += n;
        }
        let readings = meter.process(&[], &[]);
        let expected = -0.691 + 10.0 * 0.5f32.log10();
        assert!(
            (readings.lufs_integrated - expected).abs() < 1.0,
            "expected ~{expected} LUFS, got {}",
            readings.lufs_integrated
        );
    }

    #[test]
    fn reset_clears_integrated_history() {
        let mut meter = LoudnessMeter::new(48000.0);
        let l = vec![0.8f32; 48000];
        let r = vec![0.8f32; 48000];
        meter.process(&l, &r);
        assert!(meter.process(&[], &[]).lufs_integrated > SILENCE_LUFS);
        meter.reset();
        assert_eq!(meter.process(&[], &[]).lufs_integrated, SILENCE_LUFS);
    }
}
