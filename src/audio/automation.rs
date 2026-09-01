//! Sample-accurate per-buffer automation: `collect_automation` scans every track's track-wide and
//! active-region-or-session-clip lanes into per-target buckets (`TrackAutomationOverride`/
//! `MasterAutomationOverride`/`SendAutomationOverride`, each holding `LaneRef`s rather than
//! pre-evaluated values so `build_playback_stream` can call `LaneRef::value_at` per output sample);
//! `process_chain_with_automation` sub-chunks a buffer at every automation breakpoint inside it for
//! `EffectParam` targets. `DelayLine`/`pdc_delay_samples_per_track` (automatic plugin delay
//! compensation) live here too since they're part of the same per-track mixdown-adjacent layer.

use crate::model::{AutomationLane, AutomationTarget, EffectParamKey, Song, TICKS_PER_STEP, Track, TrackOutput};
use crate::plugin_host;
use crate::session::SlotState;

/// A per-channel sample delay, used for automatic plugin delay compensation (PDC) — see
/// `pdc_delay_samples_per_track`. Grows/shrinks its internal buffer to match `delay_samples` on
/// each `process` call; a change in delay amount (expected only when a chain's plugin composition
/// actually changes, not continuously) may produce one short glitch, the same "brief
/// recalculation blip" PDC causes in other hosts too.
pub(crate) struct DelayLine {
    buffer_l: std::collections::VecDeque<f32>,
    buffer_r: std::collections::VecDeque<f32>,
}

impl DelayLine {
    pub(crate) fn new() -> Self {
        Self { buffer_l: std::collections::VecDeque::new(), buffer_r: std::collections::VecDeque::new() }
    }

    pub(crate) fn process(&mut self, l: &mut [f32], r: &mut [f32], delay_samples: usize) {
        while self.buffer_l.len() < delay_samples {
            self.buffer_l.push_back(0.0);
            self.buffer_r.push_back(0.0);
        }
        while self.buffer_l.len() > delay_samples {
            self.buffer_l.pop_front();
            self.buffer_r.pop_front();
        }
        for i in 0..l.len() {
            self.buffer_l.push_back(l[i]);
            self.buffer_r.push_back(r[i]);
            l[i] = self.buffer_l.pop_front().unwrap_or(0.0);
            r[i] = self.buffer_r.pop_front().unwrap_or(0.0);
        }
    }
}

/// Computes each track's automatic plugin delay compensation amount, in samples: how much pure
/// delay to insert into that track's own post-chain signal so every track's total path latency to
/// master — `track_latency[i]` (that track's own chain), plus `submix_latency[Track::output]`'s
/// chain if it's routed through one — lines up with the slowest such path, the same way a DAW
/// aligns tracks carrying different plugin-induced delays before they'd otherwise phase-cancel or
/// comb-filter when summed. Takes already-summed-per-chain latencies (`plugin_host::
/// chain_latency_samples`) rather than the chains themselves, since callers hold them in different
/// shapes (a live `MutexGuard<Vec<Vec<Option<EffectInstance>>>>` vs. offline's
/// `Vec<plugin_host::OfflineEffectChain>`). Scope cut: a send bus's own path (a parallel tap, not
/// this track's direct contribution) and the master bus's own chain (which delays everything
/// equally, needing no per-path compensation) are *not* covered — see this feature's own design
/// note for why.
pub(crate) fn pdc_delay_samples_per_track(
    tracks: &[Track],
    track_latency: &[u32],
    submix_latency: &[u32],
) -> Vec<u32> {
    let effective_latency: Vec<u32> = tracks
        .iter()
        .enumerate()
        .map(|(i, track)| {
            let submix = match track.output {
                TrackOutput::Submix(index) => submix_latency.get(index).copied().unwrap_or(0),
                TrackOutput::Master => 0,
            };
            track_latency.get(i).copied().unwrap_or(0) + submix
        })
        .collect();
    let max_latency = effective_latency.iter().copied().max().unwrap_or(0);
    effective_latency.iter().map(|&latency| max_latency - latency).collect()
}

/// One automation lane, paired with the region-local tick offset needed to evaluate it at a given
/// output sample — `AutomationPoint::tick` is relative to the *lane's own* region's `start_tick`
/// (see `AutomationLane`'s doc comment), so a lane collected from `collect_automation` carries its
/// own `base_offset` rather than sharing one with whichever bucket it lands in: an
/// `OtherTrackVolume` lane targeting this track may come from a *different* track's region than
/// this track's own `Volume` lane, each with its own `start_tick`.
#[derive(Clone, Copy)]
pub(crate) struct LaneRef<'a> {
    /// Region-local tick at this buffer's first sample (`buffer_start_tick - region.start_tick`).
    base_offset: f64,
    lane: &'a AutomationLane,
}

impl<'a> LaneRef<'a> {
    /// This lane's value at output sample `sample_index` of the current buffer, sample-accurate
    /// via `AutomationLane::value_at_fractional` — see `TrackAutomationOverride`'s doc comment.
    pub(crate) fn value_at(&self, sample_index: usize, samples_per_tick: f64) -> f32 {
        let tick = self.base_offset + sample_index as f64 / samples_per_tick;
        self.lane
            .value_at_fractional(tick)
            .expect("collect_automation only stores lanes with at least one point")
    }
}

/// One track's automated lanes (if any) for this buffer — not necessarily all *from* that same
/// track's own region: `AutomationTarget::OtherTrack*` lets a lane on one track's region ride a
/// different track's fader/pan/send-level, so this is populated by `collect_automation` scanning
/// every track's active region and bucketing by *target*, not by source. Holds lane references
/// rather than pre-evaluated values so `build_playback_stream` can call `LaneRef::value_at` per
/// output sample instead of holding one value for the whole buffer — the same tick-to-sample-
/// accurate upgrade `Sequencer`'s per-tick `track_fade_gain` already gets, just computed downstream
/// here instead of inside `Sequencer` since these targets (unlike a fade) aren't about already-
/// triggered, freely ringing voices.
#[derive(Default)]
pub(crate) struct TrackAutomationOverride<'a> {
    pub(crate) volume: Option<LaneRef<'a>>,
    pub(crate) pan: Option<LaneRef<'a>>,
    /// (send_index, lane) pairs — only sends this track actually has a `SendLevel`/
    /// `OtherTrackSendLevel` lane for.
    pub(crate) send_levels: Vec<(usize, LaneRef<'a>)>,
    /// (chain slot_index, param key, lane) triples for this track's own effect chain.
    pub(crate) effect_params: Vec<(usize, &'a EffectParamKey, LaneRef<'a>)>,
}

/// The master bus's automated effect-chain params for this buffer — see `TrackAutomationOverride`.
#[derive(Default)]
pub(crate) struct MasterAutomationOverride<'a> {
    /// (chain slot_index, param key, lane) triples for `Song::master_effects`.
    pub(crate) effect_params: Vec<(usize, &'a EffectParamKey, LaneRef<'a>)>,
}

/// One send bus's automated effect-chain params for this buffer — see `TrackAutomationOverride`.
#[derive(Default)]
pub(crate) struct SendAutomationOverride<'a> {
    /// (chain slot_index, param key, lane) triples for this send's own `SendBus::effects`.
    pub(crate) effect_params: Vec<(usize, &'a EffectParamKey, LaneRef<'a>)>,
}

/// The region (if any) on `track` whose on-timeline span currently contains `tick` — the same
/// active-region rule `Sequencer::process`'s trigger loop and `track_fade_gain` use. A region
/// shorter than its own automation lanes' furthest point simply never reaches those points; lanes
/// don't extend a region's span the way `fade_out_ticks` doesn't either.
fn active_region_at(track: &Track, tick: usize) -> Option<&crate::model::Region> {
    track.regions.iter().find(|region| {
        tick >= region.start_tick && tick < region.start_tick + region.loop_length_steps * TICKS_PER_STEP
    })
}

/// `active_region_at`'s Session View counterpart: the one slot (if any) on `track_index` that's
/// currently sounding — `Playing` or winding down via `QueuedStop` — plus its own `local_tick`, the
/// session-playback analog of `active_region_at`'s region-local offset. Session View plays at most
/// one clip per track at a time (`Sequencer::stop_other_session_slots`), so there's at most one
/// match. `session_slots` is `Sequencer::session_slots` as published for this buffer — index-aligned
/// with `track.session_clips`, same convention as `SessionSlotHandles`.
fn active_session_clip_at<'a>(
    track: &'a Track,
    track_index: usize,
    session_slots: &[Vec<SlotState>],
) -> Option<(&'a crate::model::SessionClip, usize)> {
    let slots = session_slots.get(track_index)?;
    slots.iter().enumerate().find_map(|(slot_index, state)| {
        let local_tick = match state {
            SlotState::Playing { local_tick, .. } | SlotState::QueuedStop { local_tick, .. } => {
                *local_tick
            }
            SlotState::Stopped | SlotState::Queued { .. } => return None,
        };
        track.session_clips.get(slot_index)?.as_ref().map(|clip| (clip, local_tick))
    })
}

/// Evaluates one automation lane owned (in the source sense — see `TrackAutomationOverride`'s doc
/// comment) by `own_index`, at `base_offset`, into whichever bucket its `AutomationTarget` actually
/// resolves to — most land back on `tracks[own_index]` (the common case), but `OtherTrack*`/
/// `SendEffectParam`/`MasterEffectParam` redirect into a different track's, a send bus's, or the
/// master bus's own bucket instead (see `AutomationTarget`'s doc comment). An out-of-range
/// `track_index`/`send_index` on a redirecting target is silently ignored, same tolerance already
/// extended to overlapping regions elsewhere in this file. Shared body behind `collect_automation`'s
/// two passes (a track's own track-wide lanes, then its active region's lanes) over the same
/// buckets, so a region's lane naturally overrides a track-wide one on the same target via the
/// "later one wins" rule already documented on `collect_automation`.
fn apply_automation_lane<'a>(
    lane: &'a AutomationLane,
    base_offset: f64,
    own_index: usize,
    tracks: &mut [TrackAutomationOverride<'a>],
    master: &mut MasterAutomationOverride<'a>,
    sends: &mut [SendAutomationOverride<'a>],
) {
    if lane.points.is_empty() {
        return;
    }
    let lane_ref = LaneRef { base_offset, lane };
    match &lane.target {
        AutomationTarget::Volume => {
            tracks[own_index].volume = Some(lane_ref);
        }
        AutomationTarget::Pan => {
            tracks[own_index].pan = Some(lane_ref);
        }
        AutomationTarget::SendLevel { send_index } => {
            tracks[own_index].send_levels.push((*send_index, lane_ref));
        }
        AutomationTarget::EffectParam { slot_index, key } => {
            tracks[own_index].effect_params.push((*slot_index, key, lane_ref));
        }
        AutomationTarget::OtherTrackVolume { track_index } => {
            if let Some(t) = tracks.get_mut(*track_index) {
                t.volume = Some(lane_ref);
            }
        }
        AutomationTarget::OtherTrackPan { track_index } => {
            if let Some(t) = tracks.get_mut(*track_index) {
                t.pan = Some(lane_ref);
            }
        }
        AutomationTarget::OtherTrackSendLevel { track_index, send_index } => {
            if let Some(t) = tracks.get_mut(*track_index) {
                t.send_levels.push((*send_index, lane_ref));
            }
        }
        AutomationTarget::OtherTrackEffectParam { track_index, slot_index, key } => {
            if let Some(t) = tracks.get_mut(*track_index) {
                t.effect_params.push((*slot_index, key, lane_ref));
            }
        }
        AutomationTarget::SendEffectParam { send_index, slot_index, key } => {
            if let Some(s) = sends.get_mut(*send_index) {
                s.effect_params.push((*slot_index, key, lane_ref));
            }
        }
        AutomationTarget::MasterEffectParam { slot_index, key } => {
            master.effect_params.push((*slot_index, key, lane_ref));
        }
    }
}

/// Scans every track's own track-wide automation (`Track::automation`, evaluated at the *absolute*
/// `tick`) and then its currently-active clip automation at `tick` (this buffer's first sample),
/// bucketing every lane by target owner via `apply_automation_lane`. Which "active clip" means
/// depends on `session_mode` — the same mode switch `Sequencer::process`/`trigger_session_clips`
/// use (see `Transport::session_mode`'s doc comment): outside Session View it's a Playlist
/// `Region`'s automation, evaluated region-locally (`active_region_at`); in Session View it's
/// instead the one currently-sounding `SessionClip`'s own automation, evaluated against its
/// `local_tick` (`active_session_clip_at`) — never both, matching every other session-mode branch
/// in this file. Track-wide lanes are applied first so the active clip's own lane on the same
/// target overrides it, matching `Track::automation`'s doc comment — a clip is the more specific
/// "clip automation" layer, the track-wide lane is the underlying "track automation" layer it can
/// locally override. A lane with no points yet is skipped entirely, so adding an automation lane in
/// the UI before placing any points doesn't silently zero out that parameter.
pub(crate) fn collect_automation<'a>(
    snapshot: &'a Song,
    tick: usize,
    session_mode: bool,
    session_slots: &[Vec<SlotState>],
) -> (Vec<TrackAutomationOverride<'a>>, MasterAutomationOverride<'a>, Vec<SendAutomationOverride<'a>>) {
    let mut tracks: Vec<TrackAutomationOverride> =
        (0..snapshot.tracks.len()).map(|_| TrackAutomationOverride::default()).collect();
    let mut master = MasterAutomationOverride::default();
    let mut sends: Vec<SendAutomationOverride> =
        (0..snapshot.sends.len()).map(|_| SendAutomationOverride::default()).collect();

    for (own_index, track) in snapshot.tracks.iter().enumerate() {
        for lane in &track.automation {
            apply_automation_lane(lane, tick as f64, own_index, &mut tracks, &mut master, &mut sends);
        }
        if session_mode {
            if let Some((clip, local_tick)) = active_session_clip_at(track, own_index, session_slots) {
                for lane in &clip.automation {
                    apply_automation_lane(
                        lane,
                        local_tick as f64,
                        own_index,
                        &mut tracks,
                        &mut master,
                        &mut sends,
                    );
                }
            }
        } else if let Some(region) = active_region_at(track, tick) {
            let base_offset = (tick - region.start_tick) as f64;
            for lane in &region.automation {
                apply_automation_lane(lane, base_offset, own_index, &mut tracks, &mut master, &mut sends);
            }
        }
    }
    (tracks, master, sends)
}

/// Runs `chain` over `dry_l`/`dry_r`, sub-chunking this buffer at every point in `effect_params`
/// (from any of that lane's own points) that falls inside it — a single whole-buffer chunk when
/// nothing here is automated, the common case and the same one whole-buffer call this used to
/// always be before automated effect params existed. Re-applying each chunk's interpolated values
/// before processing it gives CLAP and built-in effects alike a breakpoint-rate approximation of
/// sample-accurate automation without either a plugin-event-timing path or per-effect DSP changes.
/// Shared body behind a track's, a send's, and the master bus's own per-buffer chain processing.
///
/// `all_track_dry` — see `plugin_host::process_effect_chain`'s doc — is passed straight through to
/// every sub-chunk's `process_effect_chain` call, re-sliced to that sub-chunk's own `[start..end]`
/// range so a resolved sidechain key always lines up sample-for-sample with `dry_l`/`dry_r`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn process_chain_with_automation(
    chain: &mut [Option<plugin_host::EffectInstance>],
    effect_params: &[(usize, &EffectParamKey, LaneRef)],
    samples_per_tick: f64,
    dry_l: &[f32],
    dry_r: &[f32],
    out_l: &mut [f32],
    out_r: &mut [f32],
    scratch: &mut [plugin_host::EffectScratch],
    run_l: &mut Vec<f32>,
    run_r: &mut Vec<f32>,
    all_track_dry: &[(&[f32], &[f32])],
) -> bool {
    let frames = dry_l.len();
    let mut boundaries = vec![0usize];
    for (_, _, lane_ref) in effect_params {
        for point in &lane_ref.lane.points {
            let offset = (point.tick as f64 - lane_ref.base_offset) * samples_per_tick;
            if offset > 0.0 && (offset as usize) < frames {
                boundaries.push(offset as usize);
            }
        }
    }
    boundaries.sort_unstable();
    boundaries.dedup();

    let mut used = false;
    for (chunk_index, &start) in boundaries.iter().enumerate() {
        let end = boundaries.get(chunk_index + 1).copied().unwrap_or(frames);
        if start >= end {
            continue;
        }
        for (slot_index, key, lane_ref) in effect_params {
            let value = lane_ref.value_at(start, samples_per_tick);
            let Some(Some(instance)) = chain.get_mut(*slot_index) else {
                continue;
            };
            match (instance, key) {
                (plugin_host::EffectInstance::Clap(effect), EffectParamKey::Clap { param_id }) => {
                    effect.set_param_by_id(*param_id, value as f64)
                }
                (
                    plugin_host::EffectInstance::BuiltIn(effect),
                    EffectParamKey::BuiltIn { param_name },
                ) => effect.set_automatable_param(param_name, value),
                _ => {}
            }
        }
        let chunk_track_dry: Vec<(&[f32], &[f32])> = all_track_dry
            .iter()
            .map(|(l, r)| (&l[start..end], &r[start..end]))
            .collect();
        used |= plugin_host::process_effect_chain(
            chain,
            &dry_l[start..end],
            &dry_r[start..end],
            &mut out_l[start..end],
            &mut out_r[start..end],
            scratch,
            run_l,
            run_r,
            &chunk_track_dry,
        );
    }
    used
}
