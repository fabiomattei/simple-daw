//! Pure tap-tempo calculation: turns a sequence of tap timestamps into a BPM estimate. No audio,
//! no UI — called from the transport LCD's Tap button (`transport_lcd_ui` in `main.rs`).

use std::time::{Duration, Instant};

/// How many recent taps to average over — recent enough to track a tempo change, long enough to
/// smooth out one mistimed tap.
const MAX_TAPS: usize = 8;

/// A tap arriving more than this long after the previous one starts a fresh sequence instead of
/// averaging with it — the user paused/restarted rather than continuing the same tempo.
const TAP_TIMEOUT: Duration = Duration::from_millis(2000);

/// Accumulates tap timestamps and derives a BPM estimate from their average interval. Pure
/// timing logic; owns no audio/UI state beyond the taps themselves — `SimpleDawApp` holds one of
/// these as transient (unsaved) state, separate from `Song::bpm`, which a caller writes the
/// estimate into.
#[derive(Default)]
pub struct TapTempo {
    taps: Vec<Instant>,
}

impl TapTempo {
    /// Records a tap at `now`, discarding any prior taps if more than `TAP_TIMEOUT` has passed
    /// since the last one. Returns the new BPM estimate once at least two taps are close enough
    /// together to average, else `None` — a single tap alone can't imply a tempo.
    pub fn tap(&mut self, now: Instant) -> Option<f32> {
        if let Some(&last) = self.taps.last()
            && now.duration_since(last) > TAP_TIMEOUT
        {
            self.taps.clear();
        }
        self.taps.push(now);
        if self.taps.len() > MAX_TAPS {
            self.taps.remove(0);
        }
        self.bpm()
    }

    /// The current BPM estimate from the average interval between recorded taps, or `None` with
    /// fewer than two taps.
    fn bpm(&self) -> Option<f32> {
        let (first, last) = (self.taps.first()?, self.taps.last()?);
        let intervals = self.taps.len() - 1;
        if intervals == 0 {
            return None;
        }
        let avg_interval_secs = last.duration_since(*first).as_secs_f64() / intervals as f64;
        (avg_interval_secs > 0.0).then(|| (60.0 / avg_interval_secs) as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_single_tap_implies_no_tempo_yet() {
        let mut tap_tempo = TapTempo::default();
        assert_eq!(tap_tempo.tap(Instant::now()), None);
    }

    #[test]
    fn two_taps_half_a_second_apart_imply_120_bpm() {
        let mut tap_tempo = TapTempo::default();
        let t0 = Instant::now();
        tap_tempo.tap(t0);
        let bpm = tap_tempo.tap(t0 + Duration::from_millis(500)).unwrap();
        assert!((bpm - 120.0).abs() < 0.01);
    }

    #[test]
    fn averages_over_multiple_steady_taps() {
        let mut tap_tempo = TapTempo::default();
        let t0 = Instant::now();
        tap_tempo.tap(t0);
        tap_tempo.tap(t0 + Duration::from_millis(500));
        let bpm = tap_tempo.tap(t0 + Duration::from_millis(1000)).unwrap();
        assert!((bpm - 120.0).abs() < 0.01);
    }

    #[test]
    fn a_long_pause_resets_the_sequence() {
        let mut tap_tempo = TapTempo::default();
        let t0 = Instant::now();
        tap_tempo.tap(t0);
        tap_tempo.tap(t0 + Duration::from_millis(500)); // 120 BPM so far
        let t1 = t0 + Duration::from_secs(5); // well past TAP_TIMEOUT
        assert_eq!(tap_tempo.tap(t1), None, "should start a fresh sequence, not average with the old one");
        let bpm = tap_tempo.tap(t1 + Duration::from_secs(1)).unwrap(); // 60 BPM
        assert!((bpm - 60.0).abs() < 0.01);
    }

    #[test]
    fn caps_history_length_and_still_tracks_recent_tempo() {
        let mut tap_tempo = TapTempo::default();
        let mut t = Instant::now();
        let mut last_bpm = None;
        for _ in 0..(MAX_TAPS + 4) {
            last_bpm = tap_tempo.tap(t);
            t += Duration::from_millis(500);
        }
        assert!((last_bpm.unwrap() - 120.0).abs() < 0.01);
    }
}
