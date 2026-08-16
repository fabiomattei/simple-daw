//! Pure signal-analysis helpers for the Playlist's audio-clip editing tools — no audio, no UI,
//! same separation `tempo_detection.rs` and `groove.rs` already keep. Two independent tools built
//! on the same short-time RMS envelope:
//! - `detect_transients`: attack positions, drawn as tick marks on a clip's waveform
//!   (`main.rs`'s `draw_audio_clip_transient_markers`). Shares its onset-novelty/adaptive-threshold/
//!   peak-picking shape with `tempo_detection::detect_bpm`'s onset detector, but reports each
//!   onset's own position instead of folding them into one BPM estimate.
//! - `detect_non_silent_segments`: the backend of the Playlist's "Strip Silence" action
//!   (`main.rs`'s `handle_audio_clip_interaction` context menu), a fixed-dB-threshold-with-hold
//!   gate rather than an adaptive one, since silence detection needs an absolute floor, not a
//!   relative-to-this-clip's-own-loudness one.

/// Width of each RMS-envelope window — fine enough to resolve onsets/silence boundaries a
/// fraction of a beat apart, coarse enough to average out single-sample noise.
const ENVELOPE_WINDOW_SECONDS: f32 = 0.01;

/// Onsets/silence boundaries closer together than this are treated as one event, not two.
const MIN_EVENT_SPACING_SECONDS: f32 = 0.05;

/// Short-time RMS envelope of `samples` at `sample_rate`, one value per
/// `ENVELOPE_WINDOW_SECONDS`-long window — the shared first step for both functions below.
fn rms_envelope(samples: &[f32], sample_rate: u32) -> (Vec<f32>, usize) {
    let window = ((sample_rate as f32 * ENVELOPE_WINDOW_SECONDS) as usize).max(1);
    let envelope = samples
        .chunks(window)
        .map(|chunk| {
            let sum_sq: f32 = chunk.iter().map(|&s| s * s).sum();
            (sum_sq / chunk.len() as f32).sqrt()
        })
        .collect();
    (envelope, window)
}

/// Frame positions of detected attacks in `samples` — a rising RMS-envelope peak above an
/// adaptive (mean + 1 std-dev) threshold, at least `MIN_EVENT_SPACING_SECONDS` apart. Mirrors
/// `tempo_detection::onset_times_seconds`'s peak-picking exactly, just reporting sample-frame
/// positions instead of folding intervals into a BPM.
pub fn detect_transients(samples: &[f32], sample_rate: u32) -> Vec<usize> {
    if samples.is_empty() || sample_rate == 0 {
        return Vec::new();
    }
    let (envelope, window) = rms_envelope(samples, sample_rate);
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

    let min_spacing_windows = (MIN_EVENT_SPACING_SECONDS / ENVELOPE_WINDOW_SECONDS).round().max(1.0) as usize;
    let mut markers = Vec::new();
    let mut last_marker_window: Option<usize> = None;
    for (i, &value) in novelty.iter().enumerate() {
        if value <= threshold {
            continue;
        }
        let is_local_peak = novelty.get(i.wrapping_sub(1)).is_none_or(|&p| value >= p)
            && novelty.get(i + 1).is_none_or(|&n| value >= n);
        if !is_local_peak {
            continue;
        }
        if last_marker_window.is_some_and(|last| i - last < min_spacing_windows) {
            continue;
        }
        markers.push(i * window);
        last_marker_window = Some(i);
    }
    markers
}

/// Frame ranges (`start_frame..end_frame`, sorted, non-overlapping) of `samples` whose RMS stays
/// at or above `threshold_db` (dBFS, e.g. `-40.0`) for at least `min_segment_seconds` — the
/// backend of the Playlist's "Strip Silence" action, which replaces one `AudioClip` with one clip
/// per returned range, each trimmed via `source_start_frame`/`length_ticks` (see
/// `model::AudioClip`) and re-anchored to the same absolute song tick the audio originally
/// occupied there — so, unlike closing the gap, sync with anything else on the timeline (other
/// tracks, a picked-up video reference) is preserved.
///
/// `min_silence_seconds` bridges brief dips below threshold (a consonant's dip mid-word, a
/// snare's natural decay) so they don't fracture one take into dozens of segments.
pub fn detect_non_silent_segments(
    samples: &[f32],
    sample_rate: u32,
    threshold_db: f32,
    min_silence_seconds: f32,
    min_segment_seconds: f32,
) -> Vec<(usize, usize)> {
    if samples.is_empty() || sample_rate == 0 {
        return Vec::new();
    }
    let (envelope, window) = rms_envelope(samples, sample_rate);
    let threshold_linear = 10f32.powf(threshold_db / 20.0);
    let min_silence_windows = (min_silence_seconds / ENVELOPE_WINDOW_SECONDS).round().max(1.0) as usize;

    let mut windowed_segments: Vec<(usize, usize)> = Vec::new();
    let mut current_start: Option<usize> = None;
    let mut silence_run = 0usize;
    for (i, &level) in envelope.iter().enumerate() {
        if level >= threshold_linear {
            current_start.get_or_insert(i);
            silence_run = 0;
        } else if let Some(start) = current_start {
            silence_run += 1;
            if silence_run >= min_silence_windows {
                windowed_segments.push((start, i - silence_run + 1));
                current_start = None;
                silence_run = 0;
            }
        }
    }
    if let Some(start) = current_start {
        windowed_segments.push((start, envelope.len()));
    }

    let min_segment_windows = (min_segment_seconds / ENVELOPE_WINDOW_SECONDS) as usize;
    windowed_segments
        .into_iter()
        .filter(|(start, end)| end.saturating_sub(*start) >= min_segment_windows)
        .map(|(start, end)| (start * window, (end * window).min(samples.len())))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn click_track(sample_rate: u32, click_at_seconds: &[f32], duration_seconds: f32) -> Vec<f32> {
        let total_samples = (sample_rate as f32 * duration_seconds) as usize;
        let mut samples = vec![0.0f32; total_samples];
        for &t in click_at_seconds {
            let start = (t * sample_rate as f32) as usize;
            for (i, s) in samples.iter_mut().enumerate().skip(start).take(200) {
                // A short decaying burst, not a single-sample impulse — closer to a real transient.
                let decay = 1.0 - (i - start) as f32 / 200.0;
                *s = decay;
            }
        }
        samples
    }

    #[test]
    fn detect_transients_returns_nothing_for_silence() {
        let samples = vec![0.0f32; 48_000];
        assert!(detect_transients(&samples, 48_000).is_empty());
    }

    #[test]
    fn detect_transients_finds_each_click_once() {
        let sample_rate = 48_000;
        let samples = click_track(sample_rate, &[0.1, 0.5, 0.9], 1.2);
        let markers = detect_transients(&samples, sample_rate);
        assert_eq!(markers.len(), 3, "markers: {markers:?}");
        for (marker, expected_seconds) in markers.iter().zip([0.1, 0.5, 0.9]) {
            let marker_seconds = *marker as f32 / sample_rate as f32;
            assert!(
                (marker_seconds - expected_seconds).abs() < 0.02,
                "marker at {marker_seconds}s, expected near {expected_seconds}s"
            );
        }
    }

    #[test]
    fn detect_non_silent_segments_finds_a_single_loud_region_in_silence() {
        let sample_rate = 48_000;
        let mut samples = vec![0.0f32; sample_rate as usize];
        for s in samples.iter_mut().skip(sample_rate as usize / 4).take(sample_rate as usize / 2) {
            *s = 0.8;
        }
        let segments = detect_non_silent_segments(&samples, sample_rate, -40.0, 0.1, 0.05);
        assert_eq!(segments.len(), 1, "segments: {segments:?}");
        let (start, end) = segments[0];
        let start_seconds = start as f32 / sample_rate as f32;
        let end_seconds = end as f32 / sample_rate as f32;
        assert!((start_seconds - 0.25).abs() < 0.02, "start: {start_seconds}s");
        assert!((end_seconds - 0.75).abs() < 0.02, "end: {end_seconds}s");
    }

    #[test]
    fn detect_non_silent_segments_bridges_brief_dips_below_threshold() {
        let sample_rate = 48_000;
        let mut samples = vec![0.8f32; sample_rate as usize];
        // A 20ms dip to silence in the middle — shorter than min_silence_seconds (0.1s).
        let dip_start = sample_rate as usize / 2;
        for s in samples.iter_mut().skip(dip_start).take(sample_rate as usize / 50) {
            *s = 0.0;
        }
        let segments = detect_non_silent_segments(&samples, sample_rate, -40.0, 0.1, 0.05);
        assert_eq!(
            segments.len(),
            1,
            "a dip shorter than min_silence_seconds should not split the segment: {segments:?}"
        );
    }

    #[test]
    fn detect_non_silent_segments_splits_on_a_long_enough_gap() {
        let sample_rate = 48_000;
        let mut samples = vec![0.8f32; sample_rate as usize];
        // A 200ms gap of silence in the middle — longer than min_silence_seconds (0.1s).
        let gap_start = sample_rate as usize / 2;
        for s in samples.iter_mut().skip(gap_start).take(sample_rate as usize / 5) {
            *s = 0.0;
        }
        let segments = detect_non_silent_segments(&samples, sample_rate, -40.0, 0.1, 0.05);
        assert_eq!(segments.len(), 2, "segments: {segments:?}");
    }

    #[test]
    fn detect_non_silent_segments_drops_segments_shorter_than_min_segment_seconds() {
        let sample_rate = 48_000;
        let mut samples = vec![0.0f32; sample_rate as usize];
        // A 20ms blip — shorter than min_segment_seconds (0.05s).
        for s in samples.iter_mut().skip(sample_rate as usize / 4).take(sample_rate as usize / 50) {
            *s = 0.8;
        }
        let segments = detect_non_silent_segments(&samples, sample_rate, -40.0, 0.1, 0.05);
        assert!(segments.is_empty(), "segments: {segments:?}");
    }

    #[test]
    fn detect_non_silent_segments_empty_input_returns_empty() {
        assert!(detect_non_silent_segments(&[], 48_000, -40.0, 0.1, 0.05).is_empty());
    }
}
