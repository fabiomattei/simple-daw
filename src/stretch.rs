//! Pure time-domain, pitch-preserving time-stretch (WSOLA — Waveform Similarity Overlap-Add). No
//! audio, no UI: `warp_buffer` is `model::AudioClip::load`'s last decode step when
//! `AudioClip::warp_markers` is set (Flex Time), and `pitch::pitch_shift_segment` builds on
//! `wsola_stretch` too (see that module's doc comment for how pitch-shift reuses this). Baked into
//! the decoded buffer once at load time rather than done live in the real-time engine — the exact
//! same "print once, play back through the ordinary `SampleVoice` path" approach `main.rs`'s
//! "Strip Silence" action already uses for its own one-shot edit.
//!
//! WSOLA, not naive fixed-hop overlap-add: a naive OLA (paste fixed-size, evenly-spaced grains
//! back to back with a crossfade) produces audible phase-cancellation "warble" wherever two
//! overlapping grains' waveforms don't line up. WSOLA fixes this by *searching* a small window
//! around each grain's ideal source position for the offset that best cross-correlates with
//! what's already been written to the output, so consecutive grains line up in phase before
//! they're crossfaded — the standard fix, and the reason this isn't just "resample_linear with
//! extra steps" (`sample.rs`'s resampler changes pitch and duration together; this changes only
//! duration).

use crate::sample::SampleBuffer;

/// Length of each analysis/synthesis grain — long enough to contain a few cycles of a low male
/// voice's fundamental (~80Hz => ~12.5ms/cycle), short enough to keep transients from smearing.
const ANALYSIS_WINDOW_SECONDS: f64 = 0.030;
/// Fixed hop on the *output* side between grain placements — half the window, i.e. 50% overlap,
/// the standard OLA/WSOLA choice that keeps the Hann-windowed overlap-sum close to unity gain.
const SYNTHESIS_HOP_FRACTION: f64 = 0.5;
/// How far WSOLA is allowed to search around a grain's ideal source position for the
/// best-correlating offset — wide enough to realign a cycle or two, narrow enough that the search
/// doesn't wander into unrelated material.
const SEARCH_RADIUS_SECONDS: f64 = 0.010;

/// A symmetric Hann window of length `len` (1.0 for `len <= 1`, since a single-sample "window"
/// has nothing to taper).
fn hann_window(len: usize) -> Vec<f32> {
    if len <= 1 {
        return vec![1.0; len];
    }
    (0..len)
        .map(|n| {
            let phase = std::f64::consts::TAU * n as f64 / (len as f64 - 1.0);
            (0.5 - 0.5 * phase.cos()) as f32
        })
        .collect()
}

/// The source-buffer start offset, within `[ideal - radius, ideal + radius]` (clamped to valid
/// grain positions), whose first `overlap_len` samples best cross-correlate (plain dot product —
/// grains are already comparable amplitude, so normalization isn't needed to rank them) with
/// `output[synth_pos..synth_pos + overlap_len]`, the tail already written by the previous grain.
/// Falls back to the (clamped) ideal position when there's nothing to correlate against yet (the
/// very first grain) or the output range is out of bounds.
fn best_analysis_start(
    source: &[f32],
    ideal_start: i64,
    search_radius: usize,
    overlap_len: usize,
    output: &[f32],
    synth_pos: usize,
) -> usize {
    let max_start = source.len().saturating_sub(overlap_len.max(1));
    let clamped_ideal = ideal_start.clamp(0, max_start as i64) as usize;
    if overlap_len == 0 || synth_pos + overlap_len > output.len() {
        return clamped_ideal;
    }
    let lo = ideal_start.saturating_sub(search_radius as i64).max(0) as usize;
    let hi = ((ideal_start + search_radius as i64).max(0) as usize).min(max_start);
    if lo >= hi {
        return clamped_ideal;
    }
    let target = &output[synth_pos..synth_pos + overlap_len];
    let mut best_start = clamped_ideal;
    let mut best_score = f32::MIN;
    for candidate in lo..=hi {
        let window = &source[candidate..candidate + overlap_len];
        let score: f32 = target.iter().zip(window.iter()).map(|(a, b)| a * b).sum();
        if score > best_score {
            best_score = score;
            best_start = candidate;
        }
    }
    best_start
}

/// Time-stretches `source` (at `sample_rate`) to exactly `target_len` samples, preserving pitch —
/// the core WSOLA primitive. A no-op copy when `source` is empty, `target_len` is `0`, or the
/// lengths already match (avoiding any WSOLA-search artifacts when no stretch is actually needed).
pub fn wsola_stretch(source: &[f32], target_len: usize, sample_rate: u32) -> Vec<f32> {
    if source.is_empty() || target_len == 0 {
        return vec![0.0; target_len];
    }
    if source.len() == target_len {
        return source.to_vec();
    }

    let window_len = ((sample_rate as f64 * ANALYSIS_WINDOW_SECONDS) as usize)
        .max(4)
        .min(source.len());
    let synthesis_hop = ((window_len as f64 * SYNTHESIS_HOP_FRACTION) as usize).max(1);
    let overlap_len = window_len.saturating_sub(synthesis_hop);
    let search_radius = (sample_rate as f64 * SEARCH_RADIUS_SECONDS) as usize;
    // Ratio > 1 stretches longer (output slower than source); < 1 compresses (output faster).
    let ratio = target_len as f64 / source.len() as f64;
    let analysis_hop = (synthesis_hop as f64 / ratio).max(1.0);

    let mut output = vec![0.0f32; target_len];
    let mut gain = vec![0.0f32; target_len];
    let window = hann_window(window_len);

    let mut grain_index = 0usize;
    let mut prev_analysis_start: Option<i64> = None;
    loop {
        let synth_pos = grain_index * synthesis_hop;
        if synth_pos >= target_len {
            break;
        }
        let ideal_analysis_start = (grain_index as f64 * analysis_hop).round() as i64;
        let analysis_start = match prev_analysis_start {
            Some(_) => best_analysis_start(
                source,
                ideal_analysis_start,
                search_radius,
                overlap_len,
                &output,
                synth_pos,
            ),
            None => ideal_analysis_start.clamp(0, source.len().saturating_sub(window_len) as i64) as usize,
        };
        for k in 0..window_len {
            let (Some(&s), Some(o), Some(g)) = (
                source.get(analysis_start + k),
                output.get_mut(synth_pos + k),
                gain.get_mut(synth_pos + k),
            ) else {
                break;
            };
            *o += s * window[k];
            *g += window[k];
        }
        prev_analysis_start = Some(analysis_start as i64);
        grain_index += 1;
    }

    for (sample, g) in output.iter_mut().zip(gain.iter()) {
        if *g > 1e-6 {
            *sample /= g;
        }
    }
    output
}

/// A point in `AudioClip::warp_markers`' piecewise time-map: `source_frame` (fixed once placed —
/// "where this transient is in the recording") to `output_frame` (draggable — "where it should
/// land"). See `model::AudioClip`.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WarpMarker {
    pub source_frame: usize,
    pub output_frame: usize,
}

/// Rebuilds `source` by WSOLA-stretching each span between consecutive `markers` (sorted by
/// `source_frame`) to fit the span between their `output_frame`s — Flex Time's own entry point.
/// A no-op clone when fewer than 2 markers are given (the "not warped" sentinel — see
/// `model::AudioClip::warp_markers`'s doc comment): one marker alone doesn't define a span to
/// stretch, matching every other "empty means untouched" convention in this codebase.
pub fn warp_buffer(source: &SampleBuffer, markers: &[WarpMarker]) -> SampleBuffer {
    if markers.len() < 2 || source.mono.is_empty() {
        return source.clone();
    }
    let mut sorted = markers.to_vec();
    sorted.sort_by_key(|m| m.source_frame);

    let mut mono = Vec::new();
    for pair in sorted.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        let src_start = a.source_frame.min(source.mono.len());
        let src_end = b.source_frame.min(source.mono.len());
        if src_end <= src_start {
            continue;
        }
        let target_len = b.output_frame.saturating_sub(a.output_frame);
        mono.extend(wsola_stretch(
            &source.mono[src_start..src_end],
            target_len,
            source.sample_rate,
        ));
    }
    SampleBuffer {
        sample_rate: source.sample_rate,
        mono,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Counts rising zero-crossings per second — the same functional pitch check
    /// `audio.rs`'s glide test already uses, since it doesn't require actually listening to
    /// confirm WSOLA preserved pitch rather than just changing duration.
    fn rising_zero_crossing_freq(samples: &[f32], sample_rate: u32) -> f32 {
        let mut crossings = 0usize;
        for w in samples.windows(2) {
            if w[0] <= 0.0 && w[1] > 0.0 {
                crossings += 1;
            }
        }
        crossings as f32 * sample_rate as f32 / samples.len() as f32
    }

    fn sine(freq: f32, sample_rate: u32, duration_seconds: f32) -> Vec<f32> {
        let n = (sample_rate as f32 * duration_seconds) as usize;
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin())
            .collect()
    }

    #[test]
    fn wsola_stretch_produces_exactly_the_requested_length() {
        let source = sine(220.0, 48_000, 1.0);
        let longer = wsola_stretch(&source, 96_000, 48_000);
        let shorter = wsola_stretch(&source, 24_000, 48_000);
        assert_eq!(longer.len(), 96_000);
        assert_eq!(shorter.len(), 24_000);
    }

    #[test]
    fn wsola_stretch_is_a_noop_when_lengths_already_match() {
        let source = sine(220.0, 48_000, 0.5);
        let out = wsola_stretch(&source, source.len(), 48_000);
        assert_eq!(out, source);
    }

    #[test]
    fn wsola_stretch_returns_silence_for_silence() {
        let source = vec![0.0f32; 48_000];
        let out = wsola_stretch(&source, 96_000, 48_000);
        assert_eq!(out.len(), 96_000);
        assert!(out.iter().all(|&s| s.abs() < 1e-4));
    }

    #[test]
    fn wsola_stretch_preserves_pitch_when_doubling_duration() {
        let sample_rate = 48_000;
        let source = sine(220.0, sample_rate, 1.0);
        let stretched = wsola_stretch(&source, source.len() * 2, sample_rate);
        // Skip the first/last grain's settling region; check the steady middle.
        let mid = &stretched[sample_rate as usize / 2..stretched.len() - sample_rate as usize / 2];
        let freq = rising_zero_crossing_freq(mid, sample_rate);
        assert!((freq - 220.0).abs() < 10.0, "expected ~220Hz, got {freq}Hz");
    }

    #[test]
    fn wsola_stretch_preserves_pitch_when_halving_duration() {
        let sample_rate = 48_000;
        let source = sine(220.0, sample_rate, 1.0);
        let compressed = wsola_stretch(&source, source.len() / 2, sample_rate);
        let mid = &compressed[sample_rate as usize / 8..compressed.len() - sample_rate as usize / 8];
        let freq = rising_zero_crossing_freq(mid, sample_rate);
        assert!((freq - 220.0).abs() < 15.0, "expected ~220Hz, got {freq}Hz");
    }

    #[test]
    fn warp_buffer_is_a_noop_clone_with_fewer_than_two_markers() {
        let source = SampleBuffer {
            sample_rate: 48_000,
            mono: sine(220.0, 48_000, 0.2),
        };
        let out = warp_buffer(&source, &[]);
        assert_eq!(out.mono, source.mono);
        let out = warp_buffer(
            &source,
            &[WarpMarker { source_frame: 0, output_frame: 0 }],
        );
        assert_eq!(out.mono, source.mono);
    }

    #[test]
    fn warp_buffer_matches_the_markers_own_output_length() {
        let source = SampleBuffer {
            sample_rate: 48_000,
            mono: sine(220.0, 48_000, 1.0),
        };
        let markers = [
            WarpMarker { source_frame: 0, output_frame: 0 },
            WarpMarker { source_frame: 24_000, output_frame: 36_000 }, // stretch first half 1.5x
            WarpMarker { source_frame: 48_000, output_frame: 48_000 }, // compress second half back
        ];
        let out = warp_buffer(&source, &markers);
        assert_eq!(out.mono.len(), 48_000);
    }

    #[test]
    fn warp_buffer_sorts_out_of_order_markers_by_source_frame() {
        let source = SampleBuffer {
            sample_rate: 48_000,
            mono: sine(220.0, 48_000, 0.5),
        };
        let forward = warp_buffer(
            &source,
            &[
                WarpMarker { source_frame: 0, output_frame: 0 },
                WarpMarker { source_frame: 24_000, output_frame: 24_000 },
            ],
        );
        let backward = warp_buffer(
            &source,
            &[
                WarpMarker { source_frame: 24_000, output_frame: 24_000 },
                WarpMarker { source_frame: 0, output_frame: 0 },
            ],
        );
        assert_eq!(forward.mono.len(), backward.mono.len());
    }
}
