//! Pure pitch detection and pitch-shifting — no audio, no UI. `detect_notes` segments a clip's
//! buffer into monophonic "notes" (consistent detected pitch, above a voicing threshold) for the
//! Flex Pitch editor to display and let the user retarget; `apply_pitch_corrections` is
//! `model::AudioClip::load`'s decode step for `AudioClip::pitch_corrections`, the same
//! "bake once at load time" approach `stretch::warp_buffer` uses for Flex Time (see that module's
//! doc comment).
//!
//! Pitch-shifting here is *not* PSOLA — it's the classic two-step trick that reuses machinery
//! this codebase already has: resample the segment by the target ratio first (a plain rate change
//! ties pitch and duration together — this is the actual pitch-shifting step), then
//! `stretch::wsola_stretch` it back to the segment's original length (WSOLA preserves whatever
//! pitch is already present, so this undoes only the *duration* change from the resample). Lower
//! risk than a bespoke grain-scheduler: it's built entirely from `wsola_stretch`, already tested
//! on its own, plus a small linear resampler mirroring `sample.rs`'s.

use crate::sample::SampleBuffer;
use crate::stretch::wsola_stretch;

/// Analysis window for one pitch estimate — long enough to contain several cycles of a low male
/// voice's fundamental (~80Hz => ~12.5ms/cycle).
const PITCH_WINDOW_SECONDS: f32 = 0.04;
/// Hop between successive analysis windows — half the window, so pitch/voicing changes are
/// caught within one hop's latency without doubling the analysis cost of a much smaller hop.
const PITCH_HOP_SECONDS: f32 = 0.02;
/// Plausible fundamental range for a voice/melodic instrument — bounds the autocorrelation lag
/// search the same way `tempo_detection.rs`'s BPM folding bounds its own search range.
const MIN_FREQ_HZ: f32 = 70.0;
const MAX_FREQ_HZ: f32 = 1_000.0;
/// Minimum normalized-autocorrelation peak to call a frame "voiced" (has a clear pitch) rather
/// than noise/silence/an unpitched consonant.
const VOICING_THRESHOLD: f32 = 0.3;
/// How far two consecutive frames' detected pitch can drift and still count as the same note —
/// half a semitone either way, tight enough that a real note change reliably crosses it.
const NOTE_PITCH_TOLERANCE_CENTS: f32 = 50.0;

/// One detected monophonic note: `start_frame..end_frame` (frame indices into the source buffer)
/// were sung/played at roughly `frequency_hz` throughout. Purely a display/analysis result —
/// not persisted (`AudioClip::pitch_corrections` stores its own frame ranges independently, so a
/// saved correction survives even if this detector's constants change later).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DetectedNote {
    pub start_frame: usize,
    pub end_frame: usize,
    pub frequency_hz: f32,
}

/// Normalized-autocorrelation pitch estimate for one analysis `frame`, or `None` if it's silent
/// or the strongest periodicity found is too weak to call "voiced" (see `VOICING_THRESHOLD`).
fn autocorrelation_pitch(frame: &[f32], sample_rate: u32) -> Option<f32> {
    let min_lag = ((sample_rate as f32 / MAX_FREQ_HZ).floor() as usize).max(1);
    let max_lag = (sample_rate as f32 / MIN_FREQ_HZ).ceil() as usize;
    if frame.len() <= max_lag {
        return None;
    }
    let energy: f32 = frame.iter().map(|s| s * s).sum();
    if energy < 1e-6 {
        return None;
    }
    let mut best_lag = 0usize;
    let mut best_corr = 0.0f32;
    for lag in min_lag..=max_lag {
        let corr: f32 = (0..frame.len() - lag).map(|i| frame[i] * frame[i + lag]).sum();
        let normalized = corr / energy;
        if normalized > best_corr {
            best_corr = normalized;
            best_lag = lag;
        }
    }
    if best_lag == 0 || best_corr < VOICING_THRESHOLD {
        return None;
    }
    Some(sample_rate as f32 / best_lag as f32)
}

/// Segments `samples` into monophonic notes: consecutive analysis frames whose detected pitch
/// stays within `NOTE_PITCH_TOLERANCE_CENTS` of the note's own first frame are merged; an
/// unvoiced/silent frame ends whatever note was in progress. Empty for silence or audio too short
/// to fill one analysis window.
pub fn detect_notes(samples: &[f32], sample_rate: u32) -> Vec<DetectedNote> {
    if samples.is_empty() || sample_rate == 0 {
        return Vec::new();
    }
    let window = ((sample_rate as f32 * PITCH_WINDOW_SECONDS) as usize).max(4);
    let hop = ((sample_rate as f32 * PITCH_HOP_SECONDS) as usize).max(1);

    let mut notes = Vec::new();
    let mut current: Option<(usize, f32, usize)> = None; // (start_frame, reference_freq, end_frame)
    let mut pos = 0usize;
    while pos + window <= samples.len() {
        let pitch = autocorrelation_pitch(&samples[pos..pos + window], sample_rate);
        let frame_end = pos + window;
        match (pitch, current) {
            (Some(freq), Some((start, reference_freq, end))) => {
                let cents = 1200.0 * (freq / reference_freq).log2();
                if cents.abs() <= NOTE_PITCH_TOLERANCE_CENTS {
                    current = Some((start, reference_freq, frame_end));
                } else {
                    notes.push(DetectedNote {
                        start_frame: start,
                        end_frame: end,
                        frequency_hz: reference_freq,
                    });
                    current = Some((pos, freq, frame_end));
                }
            }
            (Some(freq), None) => {
                current = Some((pos, freq, frame_end));
            }
            (None, Some((start, reference_freq, end))) => {
                notes.push(DetectedNote {
                    start_frame: start,
                    end_frame: end,
                    frequency_hz: reference_freq,
                });
                current = None;
            }
            (None, None) => {}
        }
        pos += hop;
    }
    if let Some((start, reference_freq, end)) = current {
        notes.push(DetectedNote {
            start_frame: start,
            end_frame: end,
            frequency_hz: reference_freq,
        });
    }
    notes
}

/// Linear-interpolation resample of `input` to exactly `target_len` samples — mirrors
/// `sample::resample_linear`'s own formula, kept as its own small copy here (rather than reusing
/// that private, rate-pair-parametrized function) since this needs an exact target *length*, not
/// a sample-rate pair; duplicating a few lines beats threading a new parametrization through
/// `sample.rs`'s existing, already-tested function for one caller.
fn resample_to_length(input: &[f32], target_len: usize) -> Vec<f32> {
    if input.is_empty() || target_len == 0 {
        return vec![0.0; target_len];
    }
    let ratio = input.len() as f64 / target_len as f64;
    (0..target_len)
        .map(|i| {
            let src_pos = i as f64 * ratio;
            let idx = src_pos.floor() as usize;
            let frac = (src_pos - idx as f64) as f32;
            let a = input.get(idx).copied().unwrap_or(0.0);
            let b = input.get(idx + 1).copied().unwrap_or(a);
            a + (b - a) * frac
        })
        .collect()
}

/// Shifts `segment`'s pitch by `ratio` (`2^(semitones/12)`; `>1.0` raises pitch) while keeping its
/// length exactly unchanged — see this module's doc comment for the resample-then-restretch
/// technique. A no-op copy for an empty segment or a ratio indistinguishable from unity.
pub fn pitch_shift_segment(segment: &[f32], ratio: f32, sample_rate: u32) -> Vec<f32> {
    if segment.is_empty() || (ratio - 1.0).abs() < 1e-4 {
        return segment.to_vec();
    }
    let resampled_len = ((segment.len() as f64) / ratio as f64).round().max(1.0) as usize;
    let resampled = resample_to_length(segment, resampled_len);
    wsola_stretch(&resampled, segment.len(), sample_rate)
}

/// One retargeted note in `AudioClip::pitch_corrections`: `start_frame..end_frame` (the source
/// buffer's own frame indices — set from a `DetectedNote`'s span when the user drags it, but
/// stored independently so a saved correction doesn't depend on `detect_notes` finding the exact
/// same boundaries again later) should sound `target_semitones` higher (or lower, if negative)
/// than it was recorded. `0.0` is a no-op, matching this codebase's usual "zero means untouched"
/// sentinel convention.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PitchCorrection {
    pub start_frame: usize,
    pub end_frame: usize,
    pub target_semitones: f32,
}

/// Applies every correction in `corrections` to a copy of `source`, in place per segment — safe
/// because `pitch_shift_segment` always returns exactly as many samples as it was given, so
/// replacing `mono[start..end]` with its shifted version never shifts anything after it. A no-op
/// clone for an empty `corrections` list (see `PitchCorrection`'s doc comment on the zero
/// sentinel — an empty list is `AudioClip::pitch_corrections`' own equivalent).
pub fn apply_pitch_corrections(source: &SampleBuffer, corrections: &[PitchCorrection]) -> SampleBuffer {
    if corrections.is_empty() {
        return source.clone();
    }
    let mut mono = source.mono.clone();
    for correction in corrections {
        let start = correction.start_frame.min(mono.len());
        let end = correction.end_frame.min(mono.len());
        if end <= start || correction.target_semitones == 0.0 {
            continue;
        }
        let ratio = 2f32.powf(correction.target_semitones / 12.0);
        let shifted = pitch_shift_segment(&mono[start..end], ratio, source.sample_rate);
        let copy_len = (end - start).min(shifted.len());
        mono[start..start + copy_len].copy_from_slice(&shifted[..copy_len]);
    }
    SampleBuffer {
        sample_rate: source.sample_rate,
        mono,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(freq: f32, sample_rate: u32, duration_seconds: f32) -> Vec<f32> {
        let n = (sample_rate as f32 * duration_seconds) as usize;
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin())
            .collect()
    }

    fn rising_zero_crossing_freq(samples: &[f32], sample_rate: u32) -> f32 {
        let mut crossings = 0usize;
        for w in samples.windows(2) {
            if w[0] <= 0.0 && w[1] > 0.0 {
                crossings += 1;
            }
        }
        crossings as f32 * sample_rate as f32 / samples.len() as f32
    }

    #[test]
    fn detect_notes_returns_nothing_for_silence() {
        let samples = vec![0.0f32; 48_000];
        assert!(detect_notes(&samples, 48_000).is_empty());
    }

    #[test]
    fn detect_notes_finds_one_note_in_a_steady_tone() {
        let sample_rate = 48_000;
        let samples = sine(220.0, sample_rate, 1.0);
        let notes = detect_notes(&samples, sample_rate);
        assert_eq!(notes.len(), 1, "notes: {notes:?}");
        assert!(
            (notes[0].frequency_hz - 220.0).abs() < 5.0,
            "detected {}",
            notes[0].frequency_hz
        );
        assert!(notes[0].end_frame > notes[0].start_frame);
    }

    #[test]
    fn detect_notes_splits_on_a_pitch_change() {
        let sample_rate = 48_000;
        let mut samples = sine(220.0, sample_rate, 0.5);
        samples.extend(sine(440.0, sample_rate, 0.5));
        let notes = detect_notes(&samples, sample_rate);
        assert_eq!(notes.len(), 2, "notes: {notes:?}");
        assert!((notes[0].frequency_hz - 220.0).abs() < 5.0);
        assert!((notes[1].frequency_hz - 440.0).abs() < 10.0);
        assert!(notes[0].end_frame <= notes[1].start_frame + sample_rate as usize / 20);
    }

    #[test]
    fn detect_notes_splits_across_a_silent_gap() {
        let sample_rate = 48_000;
        let mut samples = sine(220.0, sample_rate, 0.3);
        samples.extend(vec![0.0; sample_rate as usize / 4]);
        samples.extend(sine(220.0, sample_rate, 0.3));
        let notes = detect_notes(&samples, sample_rate);
        assert_eq!(notes.len(), 2, "notes: {notes:?}");
    }

    #[test]
    fn pitch_shift_segment_keeps_the_same_length() {
        let sample_rate = 48_000;
        let segment = sine(220.0, sample_rate, 0.5);
        let shifted = pitch_shift_segment(&segment, 1.5, sample_rate);
        assert_eq!(shifted.len(), segment.len());
    }

    #[test]
    fn pitch_shift_segment_raises_frequency_by_the_given_ratio() {
        let sample_rate = 48_000;
        let segment = sine(220.0, sample_rate, 1.0);
        let shifted = pitch_shift_segment(&segment, 1.5, sample_rate); // up a perfect fifth
        let mid = &shifted[sample_rate as usize / 4..shifted.len() - sample_rate as usize / 4];
        let freq = rising_zero_crossing_freq(mid, sample_rate);
        assert!((freq - 330.0).abs() < 15.0, "expected ~330Hz, got {freq}Hz");
    }

    #[test]
    fn pitch_shift_segment_is_a_noop_for_unity_ratio() {
        let segment = sine(220.0, 48_000, 0.2);
        let shifted = pitch_shift_segment(&segment, 1.0, 48_000);
        assert_eq!(shifted, segment);
    }

    #[test]
    fn apply_pitch_corrections_is_a_noop_clone_when_empty() {
        let source = SampleBuffer {
            sample_rate: 48_000,
            mono: sine(220.0, 48_000, 0.2),
        };
        let out = apply_pitch_corrections(&source, &[]);
        assert_eq!(out.mono, source.mono);
    }

    #[test]
    fn apply_pitch_corrections_preserves_total_length() {
        let sample_rate = 48_000;
        let source = SampleBuffer {
            sample_rate,
            mono: sine(220.0, sample_rate, 1.0),
        };
        let corrections = [PitchCorrection {
            start_frame: 10_000,
            end_frame: 30_000,
            target_semitones: 3.0,
        }];
        let out = apply_pitch_corrections(&source, &corrections);
        assert_eq!(out.mono.len(), source.mono.len());
    }
}
