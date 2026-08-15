//! Estimates a WAV recording's tempo (BPM) from its audio content via simple energy-based onset
//! detection: a short-time energy envelope, its onset "novelty" (positive change), peak-picking
//! to find onset times, then a histogram vote over the intervals between consecutive onsets. No
//! FFT/spectral analysis — good enough for clearly rhythmic source material (a drum loop, a click
//! track, a percussive recording), not a general-purpose beat tracker for a dense mix. Called from
//! the "Detect Tempo" dialog (`main.rs`).

use std::collections::HashMap;

/// Width of each energy-envelope window — fine enough to resolve onsets a fraction of a beat
/// apart, coarse enough to average out single-sample noise.
const HOP_SECONDS: f64 = 0.01;
/// Onsets closer together than this are treated as the same hit (a decaying transient's tail
/// re-triggering the novelty curve), not two separate beats.
const MIN_ONSET_SPACING_SECONDS: f64 = 0.1;
/// Candidate BPMs are folded (halved/doubled) into this range before voting, so an onset pattern
/// that's actually every other beat (or every beat of a half-time feel) still lands on the same
/// histogram bucket as the "true" tempo it's an octave away from.
const PLAUSIBLE_BPM_MIN: f64 = 60.0;
const PLAUSIBLE_BPM_MAX: f64 = 180.0;

/// Estimates BPM from `mono` audio at `sample_rate`, or `None` if too few onsets are found to
/// derive an interval from (silence, a single hit, audio too short to contain a full beat).
pub fn detect_bpm(mono: &[f32], sample_rate: u32) -> Option<f32> {
    let onsets = onset_times_seconds(mono, sample_rate);
    if onsets.len() < 2 {
        return None;
    }
    let intervals: Vec<f64> = onsets.windows(2).map(|w| w[1] - w[0]).collect();
    bpm_from_intervals(&intervals)
}

/// Onset timestamps (seconds from the start of `mono`), found by peak-picking the energy
/// envelope's onset novelty curve above an adaptive (mean + 1 std-dev) threshold.
fn onset_times_seconds(mono: &[f32], sample_rate: u32) -> Vec<f64> {
    if mono.is_empty() || sample_rate == 0 {
        return Vec::new();
    }
    let hop = ((sample_rate as f64) * HOP_SECONDS).round().max(1.0) as usize;
    let envelope: Vec<f32> = mono
        .chunks(hop)
        .map(|chunk| {
            let sum_sq: f32 = chunk.iter().map(|s| s * s).sum();
            (sum_sq / chunk.len() as f32).sqrt()
        })
        .collect();
    if envelope.len() < 2 {
        return Vec::new();
    }

    // Onset novelty: how much louder this window got than the last one — half-wave rectified,
    // since only rising energy (an attack), not decay, marks a new onset.
    let novelty: Vec<f32> = std::iter::once(0.0)
        .chain(envelope.windows(2).map(|w| (w[1] - w[0]).max(0.0)))
        .collect();

    let mean = novelty.iter().sum::<f32>() / novelty.len() as f32;
    let variance = novelty.iter().map(|n| (n - mean).powi(2)).sum::<f32>() / novelty.len() as f32;
    let threshold = mean + variance.sqrt();

    let min_spacing_hops = (MIN_ONSET_SPACING_SECONDS / HOP_SECONDS).round().max(1.0) as usize;
    let mut onsets = Vec::new();
    let mut last_onset_hop: Option<usize> = None;
    for (i, &value) in novelty.iter().enumerate() {
        if value <= threshold {
            continue;
        }
        let is_local_peak = novelty.get(i.wrapping_sub(1)).is_none_or(|&p| value >= p)
            && novelty.get(i + 1).is_none_or(|&n| value >= n);
        if !is_local_peak {
            continue;
        }
        if last_onset_hop.is_some_and(|last| i - last < min_spacing_hops) {
            continue;
        }
        onsets.push(i as f64 * HOP_SECONDS);
        last_onset_hop = Some(i);
    }
    onsets
}

/// Folds each interval into a BPM candidate within `PLAUSIBLE_BPM_MIN..=PLAUSIBLE_BPM_MAX`, then
/// returns the average of whichever rounded-BPM bucket got the most votes — a simple histogram
/// mode, robust to the occasional missed or spurious onset that a single averaged interval
/// wouldn't be.
fn bpm_from_intervals(intervals: &[f64]) -> Option<f32> {
    let mut votes: HashMap<i32, Vec<f64>> = HashMap::new();
    for &interval in intervals {
        if interval <= 0.0 {
            continue;
        }
        let mut bpm = 60.0 / interval;
        while bpm < PLAUSIBLE_BPM_MIN {
            bpm *= 2.0;
        }
        while bpm > PLAUSIBLE_BPM_MAX {
            bpm /= 2.0;
        }
        votes.entry(bpm.round() as i32).or_default().push(bpm);
    }
    let (_, best_votes) = votes.into_iter().max_by_key(|(_, v)| v.len())?;
    Some((best_votes.iter().sum::<f64>() / best_votes.len() as f64) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A synthetic "click track": short decaying percussive pulses at a steady `bpm`, so
    /// `detect_bpm` has an unambiguous ground truth to check against.
    fn click_track(bpm: f32, sample_rate: u32, num_clicks: usize) -> Vec<f32> {
        let interval_samples = (sample_rate as f64 * 60.0 / bpm as f64).round() as usize;
        let click_len = 200;
        let mut buf = vec![0.0f32; interval_samples * num_clicks + click_len];
        for click in 0..num_clicks {
            let start = click * interval_samples;
            for i in 0..click_len {
                buf[start + i] = (-(i as f32) / 20.0).exp();
            }
        }
        buf
    }

    #[test]
    fn detects_bpm_of_a_steady_120_bpm_click_track() {
        let sample_rate = 44_100;
        let clicks = click_track(120.0, sample_rate, 16);
        let bpm = detect_bpm(&clicks, sample_rate).expect("should detect a tempo");
        assert!((bpm - 120.0).abs() < 2.0, "expected ~120 BPM, got {bpm}");
    }

    #[test]
    fn detects_bpm_of_a_different_steady_tempo() {
        let sample_rate = 44_100;
        let clicks = click_track(90.0, sample_rate, 16);
        let bpm = detect_bpm(&clicks, sample_rate).expect("should detect a tempo");
        assert!((bpm - 90.0).abs() < 2.0, "expected ~90 BPM, got {bpm}");
    }

    #[test]
    fn returns_none_for_silence() {
        let silence = vec![0.0f32; 44_100 * 2];
        assert_eq!(detect_bpm(&silence, 44_100), None);
    }

    #[test]
    fn returns_none_for_a_single_hit() {
        let mut buf = vec![0.0f32; 44_100];
        for i in 0..200 {
            buf[1000 + i] = (-(i as f32) / 20.0).exp();
        }
        assert_eq!(detect_bpm(&buf, 44_100), None);
    }

    #[test]
    fn returns_none_for_empty_input() {
        assert_eq!(detect_bpm(&[], 44_100), None);
    }
}
