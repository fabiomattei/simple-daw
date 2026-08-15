//! Pure quantize/humanize/groove-template transforms for piano-roll `Note`s and step-grid
//! `Lane` steps. No audio, no UI — these operate on song data the same way
//! `model::add_note`/`clear_overlaps` do, and are called from the Piano Roll's toolbar
//! (`piano_roll_quantize_humanize_groove_ui` in `main.rs`) and the Beats window's per-lane
//! groove menu (`step_grid_lanes_ui` in `main.rs`).

use std::collections::HashSet;

use crate::model::{MAX_STEP_TIMING_OFFSET_TICKS, Note, StepData};

/// A named pattern of per-position timing/velocity nudges, applied cyclically to notes after
/// they're snapped to a grid. A note's position in the cycle is its grid index modulo
/// `timing_offsets.len()` (kept equal in length to `velocity_deltas` by every entry in
/// `GROOVE_TEMPLATES`).
pub struct GrooveTemplate {
    pub name: &'static str,
    /// Timing nudge per cycle position, as a fraction of one grid cell (typically -0.5..0.5).
    pub timing_offsets: &'static [f32],
    /// Velocity delta per cycle position, added to (and clamped back into 0..=127 on) the note's
    /// velocity.
    pub velocity_deltas: &'static [i8],
}

pub const GROOVE_TEMPLATES: &[GrooveTemplate] = &[
    GrooveTemplate { name: "Straight", timing_offsets: &[0.0], velocity_deltas: &[0] },
    GrooveTemplate {
        name: "Swing 8th (Light)",
        timing_offsets: &[0.0, 0.17],
        velocity_deltas: &[0, -8],
    },
    GrooveTemplate {
        name: "Swing 8th (Heavy)",
        timing_offsets: &[0.0, 0.33],
        velocity_deltas: &[0, -16],
    },
    GrooveTemplate {
        name: "Swing 16th (Light)",
        timing_offsets: &[0.0, 0.0, 0.0, 0.17],
        velocity_deltas: &[0, 0, 0, -8],
    },
    GrooveTemplate {
        name: "MPC Push",
        timing_offsets: &[0.08, 0.0, 0.08, 0.0],
        velocity_deltas: &[4, 0, 4, 0],
    },
    GrooveTemplate {
        name: "MPC Lay-back",
        timing_offsets: &[-0.08, 0.0, -0.08, 0.0],
        velocity_deltas: &[-4, 0, -4, 0],
    },
];

fn note_is_targeted(note: &Note, selected: Option<&HashSet<u64>>) -> bool {
    selected.is_none_or(|sel| sel.contains(&note.id))
}

/// Moves each targeted note's `start_tick` toward the nearest multiple of `grid_ticks`, by
/// `strength` (0.0 = no change, 1.0 = fully snapped). `selected` restricts which notes are
/// touched; `None` targets every note in `notes`.
pub fn quantize_notes(
    notes: &mut [Note],
    selected: Option<&HashSet<u64>>,
    grid_ticks: usize,
    strength: f32,
) {
    let grid_ticks = grid_ticks.max(1) as i64;
    let strength = strength.clamp(0.0, 1.0) as f64;
    for note in notes.iter_mut() {
        if !note_is_targeted(note, selected) {
            continue;
        }
        let start = note.start_tick as i64;
        let nearest_grid = (start as f64 / grid_ticks as f64).round() as i64 * grid_ticks;
        let nudged = start + ((nearest_grid - start) as f64 * strength).round() as i64;
        note.start_tick = nudged.max(0) as usize;
    }
}

/// Nudges each targeted note's timing and/or velocity by a pseudo-random amount up to
/// `timing_amount_ticks`/`velocity_amount`, deterministic from `seed` (re-running with the same
/// seed reproduces the same result — pass a fresh seed, e.g. from the system clock, for a
/// different result each click).
pub fn humanize_notes(
    notes: &mut [Note],
    selected: Option<&HashSet<u64>>,
    timing_amount_ticks: usize,
    velocity_amount: u8,
    seed: u64,
) {
    for (index, note) in notes.iter_mut().enumerate() {
        if !note_is_targeted(note, selected) {
            continue;
        }
        if timing_amount_ticks > 0 {
            let jitter = signed_jitter(seed, index as u64 * 2, timing_amount_ticks as i64);
            note.start_tick = (note.start_tick as i64 + jitter).max(0) as usize;
        }
        if velocity_amount > 0 {
            let jitter = signed_jitter(seed, index as u64 * 2 + 1, velocity_amount as i64);
            note.velocity = (note.velocity as i64 + jitter).clamp(0, 127) as u8;
        }
    }
}

/// Snaps each targeted note to `grid_ticks` (like `quantize_notes` at full strength), then applies
/// `template`'s per-position timing/velocity nudge based on the note's resulting grid index.
pub fn apply_groove_template(
    notes: &mut [Note],
    selected: Option<&HashSet<u64>>,
    grid_ticks: usize,
    template: &GrooveTemplate,
) {
    let grid_ticks_i = grid_ticks.max(1) as i64;
    let cycle_len = template.timing_offsets.len().max(1);
    for note in notes.iter_mut() {
        if !note_is_targeted(note, selected) {
            continue;
        }
        let grid_index = (note.start_tick as i64).div_euclid(grid_ticks_i);
        let snapped = grid_index * grid_ticks_i;
        let position = grid_index.rem_euclid(cycle_len as i64) as usize;
        let timing_offset = template.timing_offsets.get(position).copied().unwrap_or(0.0);
        let velocity_delta = template.velocity_deltas.get(position).copied().unwrap_or(0);
        let nudged = snapped + (timing_offset as f64 * grid_ticks_i as f64).round() as i64;
        note.start_tick = nudged.max(0) as usize;
        note.velocity = (note.velocity as i16 + velocity_delta as i16).clamp(0, 127) as u8;
    }
}

/// Applies `template`'s per-position timing/velocity nudge to every active step in `steps`,
/// cycling by step index (position = step index modulo the template's cycle length) — the
/// step-grid counterpart of `apply_groove_template`. Steps are already grid-aligned (unlike a
/// `Note`'s free-form tick), so there's no separate snap pass first; `timing_offset_ticks` is
/// clamped to `+/-MAX_STEP_TIMING_OFFSET_TICKS` (see `StepData`'s doc comment).
pub fn apply_groove_template_to_steps(steps: &mut [Option<StepData>], template: &GrooveTemplate) {
    let cycle_len = template.timing_offsets.len().max(1);
    for (index, slot) in steps.iter_mut().enumerate() {
        let Some(step) = slot else { continue };
        let position = index % cycle_len;
        let timing_offset = template.timing_offsets.get(position).copied().unwrap_or(0.0);
        let velocity_delta = template.velocity_deltas.get(position).copied().unwrap_or(0);
        let ticks = (timing_offset as f64 * TICKS_PER_STEP_F64).round() as i64;
        step.timing_offset_ticks = clamp_step_offset(ticks);
        step.velocity = (step.velocity as i16 + velocity_delta as i16).clamp(0, 127) as u8;
    }
}

/// Nudges every active step's timing and/or velocity by a pseudo-random amount up to
/// `timing_amount_ticks` (clamped to `+/-MAX_STEP_TIMING_OFFSET_TICKS`)/`velocity_amount`,
/// deterministic from `seed` — the step-grid counterpart of `humanize_notes`.
pub fn humanize_steps(
    steps: &mut [Option<StepData>],
    timing_amount_ticks: u8,
    velocity_amount: u8,
    seed: u64,
) {
    let timing_amount_ticks = (timing_amount_ticks as i64).min(MAX_STEP_TIMING_OFFSET_TICKS as i64);
    for (index, slot) in steps.iter_mut().enumerate() {
        let Some(step) = slot else { continue };
        if timing_amount_ticks > 0 {
            let jitter = signed_jitter(seed, index as u64 * 2, timing_amount_ticks);
            step.timing_offset_ticks =
                clamp_step_offset(step.timing_offset_ticks as i64 + jitter);
        }
        if velocity_amount > 0 {
            let jitter = signed_jitter(seed, index as u64 * 2 + 1, velocity_amount as i64);
            step.velocity = (step.velocity as i64 + jitter).clamp(0, 127) as u8;
        }
    }
}

const TICKS_PER_STEP_F64: f64 = crate::model::TICKS_PER_STEP as f64;

fn clamp_step_offset(ticks: i64) -> i8 {
    ticks.clamp(-(MAX_STEP_TIMING_OFFSET_TICKS as i64), MAX_STEP_TIMING_OFFSET_TICKS as i64) as i8
}

/// Cheap, dependency-free deterministic pseudo-random offset in `-max..=max`, mixing `seed` and
/// `index` with a Murmur3-style finalizer (same family of bit-mixing `audio::hash_to_bipolar`
/// uses for oscillator noise) — no `rand` dependency needed for a one-shot UI action like this.
fn signed_jitter(seed: u64, index: u64, max: i64) -> i64 {
    if max <= 0 {
        return 0;
    }
    let mut h = seed ^ index.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    h ^= h >> 33;
    let unit = h as f64 / u64::MAX as f64; // 0.0..1.0
    ((unit * 2.0 - 1.0) * max as f64).round() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn note(id: u64, start_tick: usize, velocity: u8) -> Note {
        Note { id, pitch: 60, start_tick, length_ticks: 24, velocity }
    }

    #[test]
    fn quantize_snaps_fully_at_strength_one() {
        let mut notes = vec![note(1, 10, 100), note(2, 37, 100)];
        quantize_notes(&mut notes, None, 24, 1.0);
        assert_eq!(notes[0].start_tick, 0);
        assert_eq!(notes[1].start_tick, 48);
    }

    #[test]
    fn quantize_partial_strength_moves_halfway() {
        let mut notes = vec![note(1, 12, 100)]; // nearest grid line (24) is 12 ticks away
        quantize_notes(&mut notes, None, 24, 0.5);
        assert_eq!(notes[0].start_tick, 18);
    }

    #[test]
    fn quantize_respects_selection() {
        let mut notes = vec![note(1, 10, 100), note(2, 10, 100)];
        let mut selected = HashSet::new();
        selected.insert(1);
        quantize_notes(&mut notes, Some(&selected), 24, 1.0);
        assert_eq!(notes[0].start_tick, 0);
        assert_eq!(notes[1].start_tick, 10);
    }

    #[test]
    fn humanize_stays_within_bounds_and_is_deterministic() {
        let mut a = vec![note(1, 100, 64), note(2, 200, 64), note(3, 300, 64)];
        let mut b = vec![note(1, 100, 64), note(2, 200, 64), note(3, 300, 64)];
        humanize_notes(&mut a, None, 12, 20, 42);
        humanize_notes(&mut b, None, 12, 20, 42);
        for (na, nb) in a.iter().zip(b.iter()) {
            assert_eq!(na.start_tick, nb.start_tick, "same seed should reproduce the same result");
            assert_eq!(na.velocity, nb.velocity);
        }
        for (orig_tick, n) in [100i64, 200, 300].into_iter().zip(a.iter()) {
            assert!((n.start_tick as i64 - orig_tick).abs() <= 12);
            assert!(n.velocity >= 44 && n.velocity <= 84);
        }
    }

    #[test]
    fn humanize_zero_amount_is_a_no_op() {
        let mut notes = vec![note(1, 100, 64)];
        humanize_notes(&mut notes, None, 0, 0, 42);
        assert_eq!(notes[0].start_tick, 100);
        assert_eq!(notes[0].velocity, 64);
    }

    #[test]
    fn groove_template_straight_is_identity_on_grid() {
        let mut notes = vec![note(1, 48, 100)];
        apply_groove_template(&mut notes, None, 24, &GROOVE_TEMPLATES[0]);
        assert_eq!(notes[0].start_tick, 48);
        assert_eq!(notes[0].velocity, 100);
    }

    #[test]
    fn groove_template_swing_delays_odd_positions() {
        let swing = &GROOVE_TEMPLATES[1]; // Swing 8th (Light)
        let mut notes = vec![note(1, 24, 100)]; // grid index 1 -> odd (swung) position
        apply_groove_template(&mut notes, None, 24, swing);
        assert!(notes[0].start_tick > 24);
        assert!(notes[0].velocity < 100);
    }

    #[test]
    fn groove_template_even_positions_are_untouched() {
        let swing = &GROOVE_TEMPLATES[1]; // Swing 8th (Light)
        let mut notes = vec![note(1, 0, 100)]; // grid index 0 -> even (straight) position
        apply_groove_template(&mut notes, None, 24, swing);
        assert_eq!(notes[0].start_tick, 0);
        assert_eq!(notes[0].velocity, 100);
    }

    fn step(velocity: u8) -> Option<StepData> {
        Some(StepData { velocity, timing_offset_ticks: 0 })
    }

    #[test]
    fn humanize_steps_leaves_empty_slots_alone() {
        let mut steps = vec![None, step(100), None];
        humanize_steps(&mut steps, 10, 20, 42);
        assert!(steps[0].is_none());
        assert!(steps[2].is_none());
    }

    #[test]
    fn humanize_steps_stays_within_clamped_bounds() {
        let mut steps = vec![step(64); 8];
        // 100 exceeds MAX_STEP_TIMING_OFFSET_TICKS — should clamp, not overshoot into a
        // neighboring step's own territory.
        humanize_steps(&mut steps, 100, 30, 7);
        for slot in &steps {
            let s = slot.unwrap();
            assert!(s.timing_offset_ticks.abs() <= MAX_STEP_TIMING_OFFSET_TICKS);
            assert!(s.velocity >= 34 && s.velocity <= 94);
        }
    }

    #[test]
    fn humanize_steps_zero_amount_is_a_no_op() {
        let mut steps = vec![step(64)];
        humanize_steps(&mut steps, 0, 0, 42);
        let s = steps[0].unwrap();
        assert_eq!(s.timing_offset_ticks, 0);
        assert_eq!(s.velocity, 64);
    }

    #[test]
    fn groove_template_to_steps_skips_empty_slots() {
        let mut steps = vec![None, None];
        apply_groove_template_to_steps(&mut steps, &GROOVE_TEMPLATES[1]);
        assert!(steps.iter().all(|s| s.is_none()));
    }

    #[test]
    fn groove_template_to_steps_swings_odd_step_indices() {
        let mut steps = vec![step(100), step(100)];
        apply_groove_template_to_steps(&mut steps, &GROOVE_TEMPLATES[1]); // Swing 8th (Light)
        assert_eq!(steps[0].unwrap().timing_offset_ticks, 0);
        assert!(steps[1].unwrap().timing_offset_ticks > 0);
        assert!(steps[1].unwrap().velocity < 100);
    }

    #[test]
    fn groove_template_to_steps_clamps_offset_within_range() {
        // A template whose fractional offset would exceed half a step must still clamp.
        let extreme = GrooveTemplate {
            name: "test-extreme",
            timing_offsets: &[0.9],
            velocity_deltas: &[0],
        };
        let mut steps = vec![step(100)];
        apply_groove_template_to_steps(&mut steps, &extreme);
        assert_eq!(steps[0].unwrap().timing_offset_ticks, MAX_STEP_TIMING_OFFSET_TICKS);
    }
}
