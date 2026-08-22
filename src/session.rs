//! Pure Session View clip-slot state machine and launch-quantization math — no audio, no UI, the
//! same shape as `groove.rs`/`transient_detection.rs`. Operates on `model::LaunchIntent`/
//! `model::FollowAction`/`model::FollowActionConfig` and plain tick counts; called from
//! `audio::Sequencer::process`'s session-mode trigger loop and unit-tested here in isolation.

use crate::model::{FollowAction, FollowActionConfig, LaunchIntent, LaunchMode};

/// One Session View slot's playback state, owned by `audio::Sequencer` (not `Song` — see
/// `model::SessionLaunchRequest`'s doc comment on why click intent and playback state live on
/// opposite sides of the UI/audio-thread snapshot boundary).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SlotState {
    #[default]
    Stopped,
    /// Waiting for `launch_tick` to arrive before starting playback from the top of its loop.
    Queued { launch_tick: usize },
    /// Actively looping; `local_tick` is this slot's own playhead, wrapped modulo the clip's
    /// loop length by `advance_slot` — the session-playback counterpart of a `Region`'s
    /// `region_local_tick` in `Sequencer::process`, just not tied to any absolute song tick.
    /// `loop_count` counts completed passes through the loop (incremented exactly when
    /// `local_tick` wraps back to `0`) — what `resolve_follow_action` measures against
    /// `FollowActionConfig::times`. Resets to `0` on every fresh launch (a plain start, a legato
    /// handoff, or a follow-action restart), so "play N times" always measures from that launch.
    Playing { local_tick: usize, loop_count: u32 },
    /// Still playing (and still advancing `local_tick`) until `stop_tick` arrives. No
    /// `loop_count`/follow-action bookkeeping here — the user (or a prior follow action) already
    /// decided this slot is on its way out, so a loop completing while winding down doesn't fire
    /// another follow action.
    QueuedStop { local_tick: usize, stop_tick: usize },
}

/// Rounds `current_tick` up to the next multiple of `quantize_ticks` (the next bar/beat/etc.
/// boundary a launch/stop should snap to). `quantize_ticks == 0` means "no quantization" — acts
/// immediately, at `current_tick` itself.
pub fn next_quantize_boundary(current_tick: usize, quantize_ticks: usize) -> usize {
    if quantize_ticks == 0 {
        return current_tick;
    }
    current_tick.div_ceil(quantize_ticks) * quantize_ticks
}

/// The click handler's pure core for `LaunchMode::Toggle`/`Trigger` (see `advance_held_slot` for
/// `Gate`/`Repeat`, which are driven by a continuous held signal instead of discrete clicks like
/// this). Folds a new `LaunchIntent` into `state`, queuing a start/stop at the next quantize
/// boundary rather than acting instantly (unless `quantize_ticks == 0`, in which case the
/// boundary is `tick_now` and the transition below still applies on the very next `advance_slot`
/// call). A `Stop` click while already `Stopped` is a no-op. A `Stop` click on a `Queued` slot
/// cancels the queued launch outright (it never started, so there's nothing to fade out of); a
/// `Play` click on a `QueuedStop` slot un-queues the stop and keeps playing from where it is.
///
/// `mode` only changes the `Playing`+`Play` case: under `Toggle` it's a no-op (already playing,
/// nothing to do — the UI instead sends `Stop` for a second click in `Toggle` mode); under
/// `Trigger`/`Repeat` it retriggers the clip from the top at the next boundary, matching
/// Ableton's own Trigger-mode "click again to restart" behavior rather than requiring a separate
/// stop first. `Gate` never reaches this function (see `advance_held_slot`), but is accepted here
/// too, behaving like `Trigger`, so a stray `Play` click on a Gate-mode clip (e.g. from a MIDI
/// controller that only sends discrete clicks) still does something sensible.
pub fn apply_launch_request(
    state: SlotState,
    intent: LaunchIntent,
    mode: LaunchMode,
    tick_now: usize,
    quantize_ticks: usize,
) -> SlotState {
    let boundary = next_quantize_boundary(tick_now, quantize_ticks);
    match (state, intent) {
        (SlotState::Stopped, LaunchIntent::Play) => SlotState::Queued { launch_tick: boundary },
        (SlotState::Playing { .. }, LaunchIntent::Play)
            if matches!(mode, LaunchMode::Trigger | LaunchMode::Repeat | LaunchMode::Gate) =>
        {
            SlotState::Queued { launch_tick: boundary }
        }
        (SlotState::Playing { local_tick, .. }, LaunchIntent::Stop) => {
            SlotState::QueuedStop { local_tick, stop_tick: boundary }
        }
        (SlotState::Queued { .. }, LaunchIntent::Stop) => SlotState::Stopped,
        (SlotState::QueuedStop { local_tick, .. }, LaunchIntent::Play) => {
            SlotState::Playing { local_tick, loop_count: 0 }
        }
        (unchanged, _) => unchanged,
    }
}

/// The continuous-hold counterpart of `apply_launch_request`, driving `LaunchMode::Gate`/
/// `Repeat` from `SessionLaunchRequest::held` rather than a discrete click — read fresh every
/// tick from whatever the latest snapshot carries (see that field's doc comment). Returns the new
/// state plus whether this call should force a fresh content trigger (a mid-hold `Repeat`
/// retrigger doesn't go through `Stopped`/`Queued` on its way back to `Playing`, so
/// `just_started_playing` alone can't detect it the way an ordinary launch does).
///
/// Releasing (`held == false`) stops immediately, bypassing quantization entirely — a physical
/// gate/key doesn't wait for the next bar to let go. Holding while `Stopped` queues a start at
/// the next boundary like any other launch. `Repeat` additionally retriggers from the top every
/// time `tick_now` lands exactly on a quantize boundary while still `Playing` and held —
/// `quantize_ticks == 0` disables this (no boundary to land on), leaving `Repeat` behaving like
/// plain `Gate` in that case.
pub fn advance_held_slot(
    state: SlotState,
    held: bool,
    mode: LaunchMode,
    tick_now: usize,
    quantize_ticks: usize,
) -> (SlotState, bool) {
    if !held {
        return (SlotState::Stopped, false);
    }
    match state {
        SlotState::Stopped => {
            (SlotState::Queued { launch_tick: next_quantize_boundary(tick_now, quantize_ticks) }, false)
        }
        SlotState::Playing { .. }
            if mode == LaunchMode::Repeat && quantize_ticks > 0 && tick_now.is_multiple_of(quantize_ticks) =>
        {
            (SlotState::Playing { local_tick: 0, loop_count: 0 }, true)
        }
        other => (other, false),
    }
}

/// Advances `state` by one tick: resolves a `Queued`/`QueuedStop` boundary once `tick_now`
/// reaches it, and wraps `local_tick` modulo `loop_length_ticks` while playing (incrementing
/// `loop_count` on each wrap). Called once per tick for every slot that holds content, regardless
/// of its current state (cheap no-op for `Stopped`), the same "always advance, branch on state"
/// shape `Sequencer::process` already uses for its tick clock. `loop_length_ticks == 0` (an
/// unset/failed-to-load clip) is treated as "never wrap" to avoid a division by zero —
/// `local_tick` just stays at `0`, and `loop_count` never advances either.
pub fn advance_slot(state: SlotState, tick_now: usize, loop_length_ticks: usize) -> SlotState {
    match state {
        SlotState::Stopped => SlotState::Stopped,
        SlotState::Queued { launch_tick } => {
            if tick_now >= launch_tick {
                SlotState::Playing { local_tick: 0, loop_count: 0 }
            } else {
                state
            }
        }
        SlotState::Playing { local_tick, loop_count } => {
            if loop_length_ticks == 0 {
                SlotState::Playing { local_tick: 0, loop_count }
            } else if local_tick + 1 >= loop_length_ticks {
                SlotState::Playing { local_tick: 0, loop_count: loop_count + 1 }
            } else {
                SlotState::Playing { local_tick: local_tick + 1, loop_count }
            }
        }
        SlotState::QueuedStop { local_tick, stop_tick } => {
            if tick_now >= stop_tick {
                SlotState::Stopped
            } else {
                SlotState::QueuedStop { local_tick: next_local_tick(local_tick, loop_length_ticks), stop_tick }
            }
        }
    }
}

fn next_local_tick(local_tick: usize, loop_length_ticks: usize) -> usize {
    if loop_length_ticks == 0 {
        0
    } else {
        (local_tick + 1) % loop_length_ticks
    }
}

/// True exactly on the tick a slot transitions into `Playing` from `Stopped`/`Queued` — the
/// moment one-shot session content (an audio clip) should be freshly triggered
/// (`SampleVoice::trigger_clip`), as opposed to step-grid/piano-roll content, which is triggered
/// by matching `local_tick` against grid/note positions every tick regardless of this signal.
pub fn just_started_playing(before: SlotState, after: SlotState) -> bool {
    !matches!(before, SlotState::Playing { .. }) && matches!(after, SlotState::Playing { .. })
}

/// True exactly on the tick a `Playing` slot's `loop_count` just increased — it completed
/// another full pass through its loop. Distinct from `just_started_playing`, which only fires
/// once, at the very first start; this fires on every subsequent loop-back too, and is what
/// gates a `resolve_follow_action` check.
pub fn just_completed_a_loop(before: SlotState, after: SlotState) -> bool {
    matches!(
        (before, after),
        (
            SlotState::Playing { loop_count: before_count, .. },
            SlotState::Playing { loop_count: after_count, .. }
        ) if after_count > before_count
    )
}

/// Resolves whether a completed loop (see `just_completed_a_loop`) should trigger a follow
/// action: `None` if the clip hasn't yet played `config.times` times through, else the winning
/// action — weighted-random between `action_a`/`action_b` by `chance_a`/`chance_b` (matching
/// Ableton: the choice is made fresh every time the follow action fires, not fixed at clip-create
/// time). Non-positive/zero total weight resolves to `FollowAction::None` (nothing to do) rather
/// than panicking on a degenerate weight split. `seed` should vary per call (e.g. mixing the
/// current tick with the track/slot index) — deterministic given the same seed, which tests rely
/// on, the same dependency-free-hash approach `groove.rs`'s `signed_jitter` uses instead of
/// pulling in `rand` for a single one-shot pick like this.
pub fn resolve_follow_action(loop_count: u32, config: &FollowActionConfig, seed: u64) -> Option<FollowAction> {
    if loop_count < config.times.max(1) {
        return None;
    }
    let chance_a = config.chance_a.max(0.0);
    let chance_b = config.chance_b.max(0.0);
    let total = chance_a + chance_b;
    if total <= 0.0 {
        return Some(FollowAction::None);
    }
    let roll = unit_hash(seed) * total as f64;
    Some(if roll < chance_a as f64 { config.action_a } else { config.action_b })
}

/// Resolves a `FollowAction` (relative to `slot_index` within a track that has `slot_count`
/// slots) into a concrete target row index. `None`/`Stop`/an out-of-range `Other` all resolve to
/// `None` — nothing to trigger, so the track goes silent. `Again` targets `slot_index` itself
/// (the caller is responsible for treating "target == source" as a fresh restart, not a no-op —
/// see `Sequencer::trigger_session_clips`). `rng_seed` is only consulted for `Any`.
pub fn follow_action_target(
    action: FollowAction,
    slot_index: usize,
    slot_count: usize,
    rng_seed: u64,
) -> Option<usize> {
    if slot_count == 0 {
        return None;
    }
    match action {
        FollowAction::None | FollowAction::Stop => None,
        FollowAction::Again => Some(slot_index),
        FollowAction::Previous => Some((slot_index + slot_count - 1) % slot_count),
        FollowAction::Next => Some((slot_index + 1) % slot_count),
        FollowAction::First => Some(0),
        FollowAction::Last => Some(slot_count - 1),
        FollowAction::Any => Some((unit_hash(rng_seed) * slot_count as f64) as usize % slot_count),
        FollowAction::Other(index) => (index < slot_count).then_some(index),
    }
}

/// Cheap, dependency-free deterministic pseudo-random value in `0.0..1.0`, mixing `seed` with a
/// Murmur3-style finalizer — the same family of bit-mixing `groove.rs`'s `signed_jitter` uses, a
/// fresh local copy here rather than a cross-module import so this file stays as self-contained
/// as `groove.rs` itself is.
fn unit_hash(seed: u64) -> f64 {
    let mut h = seed;
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    h ^= h >> 33;
    h as f64 / u64::MAX as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_quantize_boundary_rounds_up_to_the_next_multiple() {
        assert_eq!(next_quantize_boundary(0, 96), 0);
        assert_eq!(next_quantize_boundary(1, 96), 96);
        assert_eq!(next_quantize_boundary(96, 96), 96);
        assert_eq!(next_quantize_boundary(97, 96), 192);
    }

    #[test]
    fn next_quantize_boundary_zero_means_immediate() {
        assert_eq!(next_quantize_boundary(5, 0), 5);
        assert_eq!(next_quantize_boundary(0, 0), 0);
    }

    #[test]
    fn play_click_on_stopped_slot_queues_at_next_boundary() {
        let state = apply_launch_request(SlotState::Stopped, LaunchIntent::Play, LaunchMode::Toggle, 10, 96);
        assert_eq!(state, SlotState::Queued { launch_tick: 96 });
    }

    #[test]
    fn stop_click_on_playing_slot_queues_a_stop_at_next_boundary() {
        let playing = SlotState::Playing { local_tick: 40, loop_count: 3 };
        let state = apply_launch_request(playing, LaunchIntent::Stop, LaunchMode::Toggle, 10, 96);
        assert_eq!(state, SlotState::QueuedStop { local_tick: 40, stop_tick: 96 });
    }

    #[test]
    fn stop_click_on_queued_slot_cancels_the_launch() {
        let queued = SlotState::Queued { launch_tick: 96 };
        let state = apply_launch_request(queued, LaunchIntent::Stop, LaunchMode::Toggle, 10, 96);
        assert_eq!(state, SlotState::Stopped);
    }

    #[test]
    fn play_click_on_queued_stop_slot_resumes_without_stopping() {
        let queued_stop = SlotState::QueuedStop { local_tick: 40, stop_tick: 96 };
        let state = apply_launch_request(queued_stop, LaunchIntent::Play, LaunchMode::Toggle, 10, 96);
        assert_eq!(state, SlotState::Playing { local_tick: 40, loop_count: 0 });
    }

    #[test]
    fn redundant_clicks_are_no_ops_in_toggle_mode() {
        let playing = SlotState::Playing { local_tick: 5, loop_count: 1 };
        assert_eq!(
            apply_launch_request(playing, LaunchIntent::Play, LaunchMode::Toggle, 10, 96),
            playing
        );
        assert_eq!(
            apply_launch_request(SlotState::Stopped, LaunchIntent::Stop, LaunchMode::Toggle, 10, 96),
            SlotState::Stopped
        );
    }

    #[test]
    fn trigger_mode_play_click_while_playing_retriggers_at_next_boundary() {
        let playing = SlotState::Playing { local_tick: 40, loop_count: 3 };
        let state = apply_launch_request(playing, LaunchIntent::Play, LaunchMode::Trigger, 10, 96);
        assert_eq!(state, SlotState::Queued { launch_tick: 96 });
    }

    #[test]
    fn gate_mode_holding_from_stopped_queues_a_start() {
        let (state, forced) = advance_held_slot(SlotState::Stopped, true, LaunchMode::Gate, 10, 96);
        assert_eq!(state, SlotState::Queued { launch_tick: 96 });
        assert!(!forced);
    }

    #[test]
    fn releasing_a_held_slot_stops_immediately_ignoring_quantization() {
        let playing = SlotState::Playing { local_tick: 40, loop_count: 3 };
        let (state, forced) = advance_held_slot(playing, false, LaunchMode::Gate, 10, 96);
        assert_eq!(state, SlotState::Stopped);
        assert!(!forced);

        let queued = SlotState::Queued { launch_tick: 96 };
        let (state, _) = advance_held_slot(queued, false, LaunchMode::Repeat, 10, 96);
        assert_eq!(state, SlotState::Stopped);
    }

    #[test]
    fn repeat_mode_retriggers_exactly_on_quantize_boundaries_while_held() {
        let playing = SlotState::Playing { local_tick: 40, loop_count: 3 };
        let (state, forced) = advance_held_slot(playing, true, LaunchMode::Repeat, 96, 96);
        assert_eq!(state, SlotState::Playing { local_tick: 0, loop_count: 0 });
        assert!(forced);

        // Off a boundary, Repeat behaves like plain Gate: no retrigger.
        let (state, forced) = advance_held_slot(playing, true, LaunchMode::Repeat, 100, 96);
        assert_eq!(state, playing);
        assert!(!forced);
    }

    #[test]
    fn gate_mode_never_retriggers_mid_hold() {
        let playing = SlotState::Playing { local_tick: 40, loop_count: 3 };
        let (state, forced) = advance_held_slot(playing, true, LaunchMode::Gate, 96, 96);
        assert_eq!(state, playing);
        assert!(!forced);
    }

    #[test]
    fn repeat_mode_with_zero_quantize_never_retriggers() {
        let playing = SlotState::Playing { local_tick: 40, loop_count: 3 };
        let (state, forced) = advance_held_slot(playing, true, LaunchMode::Repeat, 96, 0);
        assert_eq!(state, playing);
        assert!(!forced);
    }

    #[test]
    fn advance_slot_leaves_stopped_alone() {
        assert_eq!(advance_slot(SlotState::Stopped, 5, 96), SlotState::Stopped);
    }

    #[test]
    fn advance_slot_waits_for_the_launch_boundary() {
        let queued = SlotState::Queued { launch_tick: 96 };
        assert_eq!(advance_slot(queued, 95, 384), queued);
        assert_eq!(
            advance_slot(queued, 96, 384),
            SlotState::Playing { local_tick: 0, loop_count: 0 }
        );
    }

    #[test]
    fn advance_slot_wraps_local_tick_and_counts_the_completed_loop() {
        let playing = SlotState::Playing { local_tick: 95, loop_count: 2 };
        assert_eq!(
            advance_slot(playing, 0, 96),
            SlotState::Playing { local_tick: 0, loop_count: 3 }
        );
    }

    #[test]
    fn advance_slot_mid_loop_leaves_loop_count_untouched() {
        let playing = SlotState::Playing { local_tick: 10, loop_count: 2 };
        assert_eq!(
            advance_slot(playing, 0, 96),
            SlotState::Playing { local_tick: 11, loop_count: 2 }
        );
    }

    #[test]
    fn advance_slot_stops_at_the_queued_stop_boundary() {
        let queued_stop = SlotState::QueuedStop { local_tick: 10, stop_tick: 96 };
        assert_eq!(
            advance_slot(queued_stop, 95, 384),
            SlotState::QueuedStop { local_tick: 11, stop_tick: 96 }
        );
        assert_eq!(advance_slot(queued_stop, 96, 384), SlotState::Stopped);
    }

    #[test]
    fn advance_slot_with_zero_loop_length_never_panics_or_wraps() {
        let playing = SlotState::Playing { local_tick: 0, loop_count: 0 };
        assert_eq!(
            advance_slot(playing, 0, 0),
            SlotState::Playing { local_tick: 0, loop_count: 0 }
        );
    }

    #[test]
    fn just_started_playing_is_true_only_on_the_stopped_to_playing_transition() {
        assert!(just_started_playing(
            SlotState::Queued { launch_tick: 0 },
            SlotState::Playing { local_tick: 0, loop_count: 0 }
        ));
        assert!(!just_started_playing(
            SlotState::Playing { local_tick: 0, loop_count: 0 },
            SlotState::Playing { local_tick: 1, loop_count: 0 }
        ));
        assert!(!just_started_playing(SlotState::Stopped, SlotState::Stopped));
    }

    #[test]
    fn just_completed_a_loop_is_true_only_when_loop_count_increases() {
        assert!(just_completed_a_loop(
            SlotState::Playing { local_tick: 95, loop_count: 0 },
            SlotState::Playing { local_tick: 0, loop_count: 1 }
        ));
        assert!(!just_completed_a_loop(
            SlotState::Playing { local_tick: 10, loop_count: 0 },
            SlotState::Playing { local_tick: 11, loop_count: 0 }
        ));
        assert!(!just_completed_a_loop(
            SlotState::Queued { launch_tick: 0 },
            SlotState::Playing { local_tick: 0, loop_count: 0 }
        ));
    }

    fn no_follow_action() -> FollowActionConfig {
        FollowActionConfig { times: 1, action_a: FollowAction::None, chance_a: 1.0, action_b: FollowAction::None, chance_b: 0.0 }
    }

    #[test]
    fn resolve_follow_action_waits_for_times_loops() {
        let config = FollowActionConfig { times: 3, ..no_follow_action() };
        assert_eq!(resolve_follow_action(1, &config, 42), None);
        assert_eq!(resolve_follow_action(2, &config, 42), None);
        assert!(resolve_follow_action(3, &config, 42).is_some());
    }

    #[test]
    fn resolve_follow_action_picks_a_with_full_weight() {
        let config = FollowActionConfig {
            times: 1,
            action_a: FollowAction::Next,
            chance_a: 1.0,
            action_b: FollowAction::Previous,
            chance_b: 0.0,
        };
        for seed in 0..20u64 {
            assert_eq!(resolve_follow_action(1, &config, seed), Some(FollowAction::Next));
        }
    }

    #[test]
    fn resolve_follow_action_picks_b_with_full_weight() {
        let config = FollowActionConfig {
            times: 1,
            action_a: FollowAction::Next,
            chance_a: 0.0,
            action_b: FollowAction::Previous,
            chance_b: 1.0,
        };
        for seed in 0..20u64 {
            assert_eq!(resolve_follow_action(1, &config, seed), Some(FollowAction::Previous));
        }
    }

    #[test]
    fn resolve_follow_action_picks_both_with_even_weights_over_many_seeds() {
        let config = FollowActionConfig {
            times: 1,
            action_a: FollowAction::Next,
            chance_a: 0.5,
            action_b: FollowAction::Previous,
            chance_b: 0.5,
        };
        let mut a_count = 0;
        let mut b_count = 0;
        for seed in 0..200u64 {
            match resolve_follow_action(1, &config, seed) {
                Some(FollowAction::Next) => a_count += 1,
                Some(FollowAction::Previous) => b_count += 1,
                other => panic!("unexpected result: {other:?}"),
            }
        }
        assert!(a_count > 50 && b_count > 50, "expected a reasonable split, got {a_count}/{b_count}");
    }

    #[test]
    fn resolve_follow_action_zero_total_weight_resolves_to_none() {
        let config = FollowActionConfig { times: 1, chance_a: 0.0, chance_b: 0.0, ..no_follow_action() };
        assert_eq!(resolve_follow_action(1, &config, 7), Some(FollowAction::None));
    }

    #[test]
    fn follow_action_target_resolves_relative_directions_with_wraparound() {
        assert_eq!(follow_action_target(FollowAction::Next, 2, 4, 0), Some(3));
        assert_eq!(follow_action_target(FollowAction::Next, 3, 4, 0), Some(0));
        assert_eq!(follow_action_target(FollowAction::Previous, 0, 4, 0), Some(3));
        assert_eq!(follow_action_target(FollowAction::First, 3, 4, 0), Some(0));
        assert_eq!(follow_action_target(FollowAction::Last, 0, 4, 0), Some(3));
        assert_eq!(follow_action_target(FollowAction::Again, 2, 4, 0), Some(2));
    }

    #[test]
    fn follow_action_target_none_and_stop_and_out_of_range_other_resolve_to_none() {
        assert_eq!(follow_action_target(FollowAction::None, 0, 4, 0), None);
        assert_eq!(follow_action_target(FollowAction::Stop, 0, 4, 0), None);
        assert_eq!(follow_action_target(FollowAction::Other(9), 0, 4, 0), None);
        assert_eq!(follow_action_target(FollowAction::Other(2), 0, 4, 0), Some(2));
    }

    #[test]
    fn follow_action_target_any_stays_in_bounds_over_many_seeds() {
        for seed in 0..100u64 {
            let target = follow_action_target(FollowAction::Any, 1, 5, seed).unwrap();
            assert!(target < 5);
        }
    }
}
