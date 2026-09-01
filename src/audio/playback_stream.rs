//! The real-time `cpal` playback callback: `build_playback_stream` synthesizes each buffer via
//! `Sequencer::process` (the exact same synthesis path `render_song_to_wav`/
//! `render_track_to_buffer` share for the offline bounce), then runs its own wet mixdown — per-
//! track effect chains, automation, sends, submixes, plugin delay compensation, metering — before
//! summing to master and handing the result to `cpal`. Deliberately not shared with the offline
//! mixdown (`offline_render::mix_song_to_wav_buffer`) so nothing here can regress that path or
//! vice versa.

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use cpal::traits::DeviceTrait;
use cpal::{FromSample, SizedSample, Stream, StreamConfig};

use crate::metering::{LoudnessMeter, MeterHandles};
use crate::model::{Song, TrackOutput};
use crate::plugin_host::{self, MasterEffectSlots, SendEffectSlots, SubmixEffectSlots, TrackEffectSlots};

use super::automation::{DelayLine, MasterAutomationOverride, collect_automation, pdc_delay_samples_per_track, process_chain_with_automation};
use super::sample_voice::TrackVoices;
use super::sequencer::{CaptureLogHandle, SessionSlotHandles, Sequencer, equal_power_pan_gains, samples_per_tick_at};
use super::{MASTER_GAIN, Transport};

pub(crate) fn build_playback_stream<T>(
    device: &cpal::Device,
    config: &StreamConfig,
    song: Arc<Mutex<Song>>,
    transport: Transport,
    master_effects: MasterEffectSlots,
    track_effects: TrackEffectSlots,
    send_effects: SendEffectSlots,
    submix_effects: SubmixEffectSlots,
    track_meters: MeterHandles,
    master_meter: MeterHandles,
    submix_meters: MeterHandles,
    session_slots: SessionSlotHandles,
    capture_log: CaptureLogHandle,
    max_frames: usize,
) -> Result<Stream>
where
    T: SizedSample + FromSample<f32>,
{
    let sample_rate = config.sample_rate as f32;
    let channels = (config.channels as usize).max(1);

    let mut sequencer = Sequencer::new(sample_rate);
    let mut scratch_l: Vec<f32> = Vec::with_capacity(max_frames);
    let mut scratch_r: Vec<f32> = Vec::with_capacity(max_frames);
    let mut track_dry_l: Vec<Vec<f32>> = Vec::new();
    let mut track_dry_r: Vec<Vec<f32>> = Vec::new();
    let mut metronome_dry: Vec<f32> = Vec::with_capacity(max_frames);

    // One `LoudnessMeter` per track (post-fader/pan, resized in lockstep with `track_dry_l`
    // below) plus one for the master bus (post-master-FX) — audio-thread-owned real-time state,
    // published each buffer into `track_meters`/`master_meter` for the UI thread to poll (see
    // `metering`'s module doc).
    let mut track_loudness: Vec<LoudnessMeter> = Vec::new();
    let mut master_loudness = LoudnessMeter::new(sample_rate);
    let mut was_playing = false;
    // One PDC compensation `DelayLine` per track (resized in lockstep with `track_dry_l`, same as
    // `track_loudness` above) — see `pdc_delay_samples_per_track`.
    let mut track_pdc_delay: Vec<DelayLine> = Vec::new();

    // Per-track CLAP insert-effect-chain scratch (one `Vec<EffectScratch>` per track index, grown
    // lazily to match that track's chain length) and a pair of reusable stereo buffers plus the
    // chain's own in-flight stereo scratch for whichever track is currently being processed.
    let mut track_scratch: Vec<Vec<plugin_host::EffectScratch>> = Vec::new();
    let mut track_effect_out_l: Vec<f32> = Vec::with_capacity(max_frames);
    let mut track_effect_out_r: Vec<f32> = Vec::with_capacity(max_frames);
    let mut track_chain_run_l: Vec<f32> = Vec::with_capacity(max_frames);
    let mut track_chain_run_r: Vec<f32> = Vec::with_capacity(max_frames);
    // Reused scratch for whichever track's post-fader/pan signal is currently being fed to its
    // `LoudnessMeter` (see below) — not summed anywhere itself, just a metering tap.
    let mut track_meter_l: Vec<f32> = Vec::with_capacity(max_frames);
    let mut track_meter_r: Vec<f32> = Vec::with_capacity(max_frames);

    // One accumulation buffer per send bus (resized in lockstep with `Song::sends`), fed by every
    // track's post-fader/pan signal scaled by that track's `Track::send_levels` entry for this
    // send — the same tap point `track_meter_l/r` reads, just scaled per-send instead of summed
    // straight into the master mix. Plus per-send CLAP/built-in effect-chain scratch, mirroring
    // `track_scratch`'s per-track shape, and a pair of reusable output/run buffers (sends are
    // processed one at a time, so one reusable pair covers all of them per callback).
    let mut send_mix_l: Vec<Vec<f32>> = Vec::new();
    let mut send_mix_r: Vec<Vec<f32>> = Vec::new();
    let mut send_scratch: Vec<Vec<plugin_host::EffectScratch>> = Vec::new();
    let mut send_chain_out_l: Vec<f32> = Vec::with_capacity(max_frames);
    let mut send_chain_out_r: Vec<f32> = Vec::with_capacity(max_frames);
    let mut send_chain_run_l: Vec<f32> = Vec::with_capacity(max_frames);
    let mut send_chain_run_r: Vec<f32> = Vec::with_capacity(max_frames);

    // One accumulation buffer per submix bus (resized in lockstep with `Song::submixes`), fed
    // *instead of* `scratch_l/r` by every track whose `Track::output` targets that submix — unlike
    // `send_mix_l/r` above, this replaces a track's direct contribution to the master mix rather
    // than tapping it in parallel. Same per-submix effect-chain scratch/output/run-buffer shape as
    // sends, plus one `LoudnessMeter` per submix (mirroring `track_loudness`) since a submix has
    // its own fader and deserves its own meter in the Mixer.
    let mut submix_mix_l: Vec<Vec<f32>> = Vec::new();
    let mut submix_mix_r: Vec<Vec<f32>> = Vec::new();
    let mut submix_scratch: Vec<Vec<plugin_host::EffectScratch>> = Vec::new();
    let mut submix_chain_out_l: Vec<f32> = Vec::with_capacity(max_frames);
    let mut submix_chain_out_r: Vec<f32> = Vec::with_capacity(max_frames);
    let mut submix_chain_run_l: Vec<f32> = Vec::with_capacity(max_frames);
    let mut submix_chain_run_r: Vec<f32> = Vec::with_capacity(max_frames);
    let mut submix_loudness: Vec<LoudnessMeter> = Vec::new();

    // Scratch for the master bus's own effect chain — same shape as `track_scratch`'s per-track
    // entries (one `EffectScratch` per chain slot), since the master chain runs through the exact
    // same `process_effect_chain` call a track's chain does.
    let mut master_scratch: Vec<plugin_host::EffectScratch> = Vec::new();
    let mut master_chain_run_l: Vec<f32> = Vec::with_capacity(max_frames);
    let mut master_chain_run_r: Vec<f32> = Vec::with_capacity(max_frames);
    let mut plugin_out_l: Vec<f32> = Vec::with_capacity(max_frames);
    let mut plugin_out_r: Vec<f32> = Vec::with_capacity(max_frames);

    // Pre-warm every per-track buffer to the song's shape *right now* (rather than letting the
    // first real-time callback discover it needs to grow `Vec`s and allocate 32+32 `Voice`/
    // `SampleVoice` structs per track) — a debug build doing that heap work inside the very first
    // callback was slow enough to blow the backend's deadline and log a startup underrun.
    //
    // Also captures the first snapshot for `last_snapshot` below — blocking here at stream setup
    // is fine since it's not on the real-time path yet.
    let mut last_snapshot: Option<Song> = None;
    if let Ok(snapshot) = song.lock() {
        for _ in 0..snapshot.tracks.len() {
            sequencer.track_voices.push(TrackVoices::new());
        }
        track_dry_l.resize_with(snapshot.tracks.len(), || Vec::with_capacity(max_frames));
        track_dry_r.resize_with(snapshot.tracks.len(), || Vec::with_capacity(max_frames));
        last_snapshot = Some(snapshot.clone());
    }
    if let Ok(chains) = track_effects.lock() {
        for chain in chains.iter() {
            let mut stage_scratch = Vec::with_capacity(chain.len());
            for _ in chain {
                let mut s = plugin_host::EffectScratch::new();
                s.reserve(max_frames);
                stage_scratch.push(s);
            }
            track_scratch.push(stage_scratch);
        }
    }
    if let Ok(chains) = master_effects.lock()
        && let Some(chain) = chains.first()
    {
        for _ in chain {
            let mut s = plugin_host::EffectScratch::new();
            s.reserve(max_frames);
            master_scratch.push(s);
        }
    }
    if let Ok(chains) = send_effects.lock() {
        for chain in chains.iter() {
            let mut stage_scratch = Vec::with_capacity(chain.len());
            for _ in chain {
                let mut s = plugin_host::EffectScratch::new();
                s.reserve(max_frames);
                stage_scratch.push(s);
            }
            send_scratch.push(stage_scratch);
        }
    }
    if let Ok(chains) = submix_effects.lock() {
        for chain in chains.iter() {
            let mut stage_scratch = Vec::with_capacity(chain.len());
            for _ in chain {
                let mut s = plugin_host::EffectScratch::new();
                s.reserve(max_frames);
                stage_scratch.push(s);
            }
            submix_scratch.push(stage_scratch);
        }
    }

    let err_fn = |err| eprintln!("audio stream error: {err}");

    let stream = device.build_output_stream(
        *config,
        move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
            let frames = data.len() / channels;
            scratch_l.resize(frames, 0.0);
            scratch_r.resize(frames, 0.0);
            scratch_l.iter_mut().for_each(|s| *s = 0.0);
            scratch_r.iter_mut().for_each(|s| *s = 0.0);

            // Integrated LUFS measures "since the last time playback started" (see
            // `LoudnessMeter::reset`'s doc comment) — reset every meter exactly on the
            // playing-to-stopped edge, the same transition that resets the sequencer's position
            // below, rather than continuing to accumulate through a stop.
            let is_playing = transport.is_playing();
            if was_playing && !is_playing {
                master_loudness.reset();
                for meter in track_loudness.iter_mut() {
                    meter.reset();
                }
            }
            was_playing = is_playing;

            // Note: even when stopped, silence still runs through the master
            // effect below rather than short-circuiting straight to the
            // device — otherwise a delay/reverb tail would cut off instantly
            // on Stop instead of ringing out naturally, like in a real DAW.
            // (Per-track effects don't get this treatment: while stopped, no
            // track has anything playing through them, so there's no tail to
            // preserve there — only the master bus stays fed with silence.)
            //
            // Declared outside the `is_playing` branch (default/empty when not playing) since the
            // master chain runs unconditionally, below, after this `if`/`else` — see there.
            let mut master_automation = MasterAutomationOverride::default();
            let mut master_samples_per_tick = 1.0f64;
            // Same "declared outside, default/empty when not playing" reasoning as
            // `master_automation` above — the master chain (which can itself have a sidechain
            // source) runs unconditionally below, after this `if`/`else`.
            let mut all_track_dry: Vec<(&[f32], &[f32])> = Vec::new();
            if is_playing {
                // Snapshot the song once per callback (not per sample) so the real-time thread
                // only briefly touches the shared lock. Uses `try_lock`, not `lock`, and falls
                // back to the previous snapshot on contention: the UI thread holds this same
                // mutex for its whole paint pass (`SimpleDawApp::ui`), and painting a large song
                // (many tracks/notes, e.g. after a MIDI import) can take long enough that blocking
                // here would miss this callback's deadline and log a real buffer underrun — reusing
                // a slightly stale snapshot for one buffer is inaudible, a dropout isn't.
                if let Ok(guard) = song.try_lock() {
                    last_snapshot = Some(guard.clone());
                }
                let Some(snapshot) = last_snapshot.as_ref() else {
                    return;
                };
                // The tick in effect as of this buffer's very first sample — whatever tick most
                // recently triggered as of the end of the *previous* callback, since a tick's
                // state holds until the next one fires (same reasoning as `track_fade_gain`).
                // Captured before `process()` advances it, so per-sample automation lookups below
                // can walk forward from a correct starting point instead of the buffer's *last*
                // triggered tick (which `sequencer.current_tick()` would give after `process()`).
                let buffer_start_tick = sequencer.current_tick();
                sequencer.process(
                    snapshot,
                    frames,
                    &mut track_dry_l,
                    &mut track_dry_r,
                    transport.is_metronome_enabled(),
                    &mut metronome_dry,
                    transport.is_session_mode(),
                    transport.session_quantize_ticks(),
                    transport.is_capturing(),
                );
                transport
                    .current_tick
                    .store(sequencer.current_tick(), Ordering::Relaxed);
                // See `SessionSlotHandles`'s doc comment — published every buffer so the UI thread
                // can show live queued/playing/stopped state, the same "audio thread publishes, UI
                // thread reads a cheap clone" split `track_meters` already uses just below.
                if let Ok(mut published_session_slots) = session_slots.lock() {
                    published_session_slots.clone_from(&sequencer.session_slots);
                }
                // See `CaptureLogHandle`'s doc comment — published every buffer so `main.rs` can
                // read the latest log the moment it turns `Transport::capturing` back off.
                if let Ok(mut published_capture_log) = capture_log.lock() {
                    published_capture_log.0.clone_from(&sequencer.capture_log);
                    published_capture_log.1 = sequencer.capture_tick;
                }

                // Resolved once per buffer from the tempo at its first tick — unlike
                // `Sequencer::process`'s own per-tick-boundary resolution above, a `tempo_map`
                // change landing mid-buffer only takes effect for automation/fades starting next
                // buffer (at most a few milliseconds' latency), so `LaneRef::value_at`'s per-sample
                // tick conversion below can keep assuming one constant rate for the whole buffer.
                let samples_per_tick =
                    samples_per_tick_at(sample_rate as f64, snapshot.bpm_at(buffer_start_tick));
                master_samples_per_tick = samples_per_tick;
                // One automated-lane snapshot per track/send/master for this whole buffer,
                // evaluated per output sample below via `LaneRef::value_at` — see
                // `TrackAutomationOverride`'s doc comment.
                let (track_automation, master_override, send_automation) = collect_automation(
                    snapshot,
                    buffer_start_tick,
                    transport.is_session_mode(),
                    &sequencer.session_slots,
                );
                master_automation = master_override;

                // Track count can change between callbacks (tracks added/removed) — resize in
                // lockstep with `track_dry_l`, same as `sequencer.track_voices` above.
                while track_loudness.len() < track_dry_l.len() {
                    track_loudness.push(LoudnessMeter::new(sample_rate));
                }
                track_loudness.truncate(track_dry_l.len());
                let published_track_meters = track_meters.lock().ok();

                // Run each track's dry mix through its own CLAP/built-in insert effect chain (if
                // any are loaded there — the chain now carries real stereo between stages, see
                // `plugin_host::process_effect_chain`), apply that track's volume and pan (as an
                // equal-power gain split, the same point a channel strip's pan pot sits after its
                // inserts), then sum every track into the master bus. The same post-fader/pan
                // samples feed that track's `LoudnessMeter` — the natural tap point for a channel
                // strip's meter, distinct from both the raw synthesis (`track_dry_l/r`) and the
                // final master mix.
                track_effect_out_l.resize(frames, 0.0);
                track_effect_out_r.resize(frames, 0.0);
                track_meter_l.resize(frames, 0.0);
                track_meter_r.resize(frames, 0.0);

                // Send bus count can change between callbacks (buses added/removed) — resize in
                // lockstep with `Song::sends`, same as `track_loudness` above.
                send_mix_l.resize_with(snapshot.sends.len(), Vec::new);
                send_mix_r.resize_with(snapshot.sends.len(), Vec::new);
                for buf in send_mix_l.iter_mut().chain(send_mix_r.iter_mut()) {
                    buf.clear();
                    buf.resize(frames, 0.0);
                }
                send_scratch.resize_with(snapshot.sends.len(), Vec::new);

                // Submix bus count can change between callbacks (buses added/removed) — resize in
                // lockstep with `Song::submixes`, same as the send buffers above.
                submix_mix_l.resize_with(snapshot.submixes.len(), Vec::new);
                submix_mix_r.resize_with(snapshot.submixes.len(), Vec::new);
                for buf in submix_mix_l.iter_mut().chain(submix_mix_r.iter_mut()) {
                    buf.clear();
                    buf.resize(frames, 0.0);
                }
                submix_scratch.resize_with(snapshot.submixes.len(), Vec::new);
                while submix_loudness.len() < snapshot.submixes.len() {
                    submix_loudness.push(LoudnessMeter::new(sample_rate));
                }
                submix_loudness.truncate(snapshot.submixes.len());
                let published_submix_meters = submix_meters.lock().ok();

                // Every track's own pre-effects, pre-volume/pan signal for this buffer, indexed by
                // `Song::tracks` index — the sidechain key source pool any chain (a track's, a
                // send's, a submix's, or the master's) in this callback can route from. Built once
                // here since it doesn't depend on which chain is being processed; see
                // `plugin_host::process_effect_chain`'s doc.
                all_track_dry = track_dry_l
                    .iter()
                    .zip(track_dry_r.iter())
                    .map(|(l, r)| (l.as_slice(), r.as_slice()))
                    .collect();

                // Automatic plugin delay compensation: computed fresh each buffer from whichever
                // chains are currently loaded (see `pdc_delay_samples_per_track`), applied to each
                // track's own post-chain signal below via `track_pdc_delay`'s persistent state.
                let track_pdc_delay_samples: Vec<u32> =
                    match (track_effects.lock(), submix_effects.lock()) {
                        (Ok(track_chains), Ok(submix_chains)) => {
                            let track_latency: Vec<u32> = track_chains
                                .iter()
                                .map(|chain| plugin_host::chain_latency_samples(chain))
                                .collect();
                            let submix_latency: Vec<u32> = submix_chains
                                .iter()
                                .map(|chain| plugin_host::chain_latency_samples(chain))
                                .collect();
                            pdc_delay_samples_per_track(&snapshot.tracks, &track_latency, &submix_latency)
                        }
                        _ => vec![0; snapshot.tracks.len()],
                    };
                while track_pdc_delay.len() < snapshot.tracks.len() {
                    track_pdc_delay.push(DelayLine::new());
                }
                track_pdc_delay.truncate(snapshot.tracks.len());

                if let Ok(mut chains) = track_effects.lock() {
                    while track_scratch.len() < track_dry_l.len() {
                        track_scratch.push(Vec::new());
                    }
                    for (track_index, (dry_l, dry_r)) in
                        track_dry_l.iter().zip(track_dry_r.iter()).enumerate()
                    {
                        let track = snapshot.tracks.get(track_index);
                        let automation = track_automation.get(track_index);
                        let static_volume = track.map_or(1.0, |t| t.volume);
                        let static_pan = track.map_or(0.0, |t| t.pan);
                        // Per-output-sample volume/pan, sample-accurate when a lane is present
                        // (`LaneRef::value_at` at this sample's exact tick position) rather than
                        // one value held for the whole buffer.
                        let volume_at = |i: usize| {
                            automation
                                .and_then(|a| a.volume.map(|lane| lane.value_at(i, samples_per_tick)))
                                .unwrap_or(static_volume)
                        };
                        let pan_at = |i: usize| {
                            automation
                                .and_then(|a| a.pan.map(|lane| lane.value_at(i, samples_per_tick)))
                                .unwrap_or(static_pan)
                        };
                        let chain = chains
                            .get_mut(track_index)
                            .map_or(&mut [][..], Vec::as_mut_slice);
                        let stage_scratch = &mut track_scratch[track_index];
                        while stage_scratch.len() < chain.len() {
                            stage_scratch.push(plugin_host::EffectScratch::new());
                        }
                        track_effect_out_l.resize(frames, 0.0);
                        track_effect_out_r.resize(frames, 0.0);

                        let empty_effect_params = Vec::new();
                        let effect_params =
                            automation.map_or(&empty_effect_params, |a| &a.effect_params);
                        // A frozen track's own chain is skipped: `dry_l/r` already came from
                        // `frozen_clip`, which was baked *through* this same chain (see
                        // `render_track_to_buffer`) — running it again here would double-process.
                        let used = !track.is_some_and(|t| t.frozen)
                            && process_chain_with_automation(
                                chain,
                                effect_params,
                                samples_per_tick,
                                dry_l,
                                dry_r,
                                &mut track_effect_out_l,
                                &mut track_effect_out_r,
                                stage_scratch,
                                &mut track_chain_run_l,
                                &mut track_chain_run_r,
                                &all_track_dry,
                            );
                        let source_l = if used { &track_effect_out_l } else { dry_l };
                        let source_r = if used { &track_effect_out_r } else { dry_r };
                        for i in 0..frames {
                            let (pan_l, pan_r) = equal_power_pan_gains(pan_at(i));
                            track_meter_l[i] = volume_at(i) * pan_l * source_l[i];
                            track_meter_r[i] = volume_at(i) * pan_r * source_r[i];
                        }
                        track_pdc_delay[track_index].process(
                            &mut track_meter_l[..frames],
                            &mut track_meter_r[..frames],
                            track_pdc_delay_samples[track_index] as usize,
                        );
                        // This track's post-fader/pan signal sums into its `TrackOutput`
                        // target — straight to the master accumulator, or exclusively into its
                        // submix bus's own accumulator instead (see `SubmixBus`'s doc comment).
                        match track.map_or(TrackOutput::Master, |t| t.output) {
                            TrackOutput::Master => {
                                for i in 0..frames {
                                    scratch_l[i] += track_meter_l[i];
                                    scratch_r[i] += track_meter_r[i];
                                }
                            }
                            TrackOutput::Submix(index) => {
                                if let (Some(mix_l), Some(mix_r)) =
                                    (submix_mix_l.get_mut(index), submix_mix_r.get_mut(index))
                                {
                                    for i in 0..frames {
                                        mix_l[i] += track_meter_l[i];
                                        mix_r[i] += track_meter_r[i];
                                    }
                                }
                            }
                        }
                        let readings = track_loudness[track_index].process(&track_meter_l, &track_meter_r);
                        if let Some(handle) =
                            published_track_meters.as_ref().and_then(|h| h.get(track_index))
                        {
                            handle.publish(&readings);
                        }

                        // Feed this track's post-fader/pan signal (`track_meter_l/r`, just
                        // published to the meter above) into every send bus it has a nonzero
                        // level for — the same tap point a channel strip's send knob reads from.
                        // A `SendLevel` automation lane on that send overrides the static level,
                        // sample-accurately when present, same as volume/pan above.
                        if let Some(send_levels) = track.map(|t| t.send_levels.as_slice()) {
                            for (send_index, &static_level) in send_levels.iter().enumerate() {
                                let lane = automation.and_then(|a| {
                                    a.send_levels
                                        .iter()
                                        .find(|(i, _)| *i == send_index)
                                        .map(|(_, lane)| *lane)
                                });
                                let Some((mix_l, mix_r)) = send_mix_l
                                    .get_mut(send_index)
                                    .zip(send_mix_r.get_mut(send_index))
                                else {
                                    continue;
                                };
                                match lane {
                                    Some(lane) => {
                                        for i in 0..frames {
                                            let level = lane.value_at(i, samples_per_tick);
                                            mix_l[i] += track_meter_l[i] * level;
                                            mix_r[i] += track_meter_r[i] * level;
                                        }
                                    }
                                    None => {
                                        if static_level != 0.0 {
                                            for i in 0..frames {
                                                mix_l[i] += track_meter_l[i] * static_level;
                                                mix_r[i] += track_meter_r[i] * static_level;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    for (track_index, (dry_l, dry_r)) in
                        track_dry_l.iter().zip(track_dry_r.iter()).enumerate()
                    {
                        let track = snapshot.tracks.get(track_index);
                        let automation = track_automation.get(track_index);
                        let static_volume = track.map_or(1.0, |t| t.volume);
                        let static_pan = track.map_or(0.0, |t| t.pan);
                        for i in 0..frames {
                            let volume = automation
                                .and_then(|a| a.volume.map(|lane| lane.value_at(i, samples_per_tick)))
                                .unwrap_or(static_volume);
                            let pan = automation
                                .and_then(|a| a.pan.map(|lane| lane.value_at(i, samples_per_tick)))
                                .unwrap_or(static_pan);
                            let (pan_l, pan_r) = equal_power_pan_gains(pan);
                            track_meter_l[i] = volume * pan_l * dry_l[i];
                            track_meter_r[i] = volume * pan_r * dry_r[i];
                        }
                        track_pdc_delay[track_index].process(
                            &mut track_meter_l[..frames],
                            &mut track_meter_r[..frames],
                            track_pdc_delay_samples[track_index] as usize,
                        );
                        // Output-routing: see the matching `match` in the `Ok(mut chains)` branch above.
                        match track.map_or(TrackOutput::Master, |t| t.output) {
                            TrackOutput::Master => {
                                for i in 0..frames {
                                    scratch_l[i] += track_meter_l[i];
                                    scratch_r[i] += track_meter_r[i];
                                }
                            }
                            TrackOutput::Submix(index) => {
                                if let (Some(mix_l), Some(mix_r)) =
                                    (submix_mix_l.get_mut(index), submix_mix_r.get_mut(index))
                                {
                                    for i in 0..frames {
                                        mix_l[i] += track_meter_l[i];
                                        mix_r[i] += track_meter_r[i];
                                    }
                                }
                            }
                        }
                        let readings = track_loudness[track_index].process(&track_meter_l, &track_meter_r);
                        if let Some(handle) =
                            published_track_meters.as_ref().and_then(|h| h.get(track_index))
                        {
                            handle.publish(&readings);
                        }

                        // See the matching comment in the `Ok(mut chains)` branch above.
                        if let Some(send_levels) = track.map(|t| t.send_levels.as_slice()) {
                            for (send_index, &static_level) in send_levels.iter().enumerate() {
                                let lane = automation.and_then(|a| {
                                    a.send_levels
                                        .iter()
                                        .find(|(i, _)| *i == send_index)
                                        .map(|(_, lane)| *lane)
                                });
                                let Some((mix_l, mix_r)) = send_mix_l
                                    .get_mut(send_index)
                                    .zip(send_mix_r.get_mut(send_index))
                                else {
                                    continue;
                                };
                                match lane {
                                    Some(lane) => {
                                        for i in 0..frames {
                                            let level = lane.value_at(i, samples_per_tick);
                                            mix_l[i] += track_meter_l[i] * level;
                                            mix_r[i] += track_meter_r[i] * level;
                                        }
                                    }
                                    None => {
                                        if static_level != 0.0 {
                                            for i in 0..frames {
                                                mix_l[i] += track_meter_l[i] * static_level;
                                                mix_r[i] += track_meter_r[i] * static_level;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Run each send bus's own effect chain (same `process_effect_chain` machinery a
                // track's or the master's chain uses) over its accumulated `send_mix_l/r`, then
                // sum the result straight into the master mix — a send bus has no fader of its
                // own in this minimal model, just its chain and whatever level each track sent it.
                // A `SendEffectParam` automation lane (from any track's region) overrides that
                // send's own chain params, same breakpoint-chunked approximation a track's chain
                // gets — see `process_chain_with_automation`.
                send_chain_out_l.resize(frames, 0.0);
                send_chain_out_r.resize(frames, 0.0);
                let empty_send_effect_params = Vec::new();
                for (send_index, (mix_l, mix_r)) in
                    send_mix_l.iter().zip(send_mix_r.iter()).enumerate()
                {
                    if let Ok(mut chains) = send_effects.lock() {
                        let chain = chains
                            .get_mut(send_index)
                            .map_or(&mut [][..], Vec::as_mut_slice);
                        let stage_scratch = &mut send_scratch[send_index];
                        while stage_scratch.len() < chain.len() {
                            stage_scratch.push(plugin_host::EffectScratch::new());
                        }
                        let effect_params = send_automation
                            .get(send_index)
                            .map_or(&empty_send_effect_params, |s| &s.effect_params);
                        let used = process_chain_with_automation(
                            chain,
                            effect_params,
                            samples_per_tick,
                            mix_l,
                            mix_r,
                            &mut send_chain_out_l,
                            &mut send_chain_out_r,
                            stage_scratch,
                            &mut send_chain_run_l,
                            &mut send_chain_run_r,
                            &all_track_dry,
                        );
                        if used {
                            for i in 0..frames {
                                scratch_l[i] += send_chain_out_l[i];
                                scratch_r[i] += send_chain_out_r[i];
                            }
                            continue;
                        }
                    }
                    for i in 0..frames {
                        scratch_l[i] += mix_l[i];
                        scratch_r[i] += mix_r[i];
                    }
                }

                // Run each submix bus's own effect chain (same `process_effect_chain` machinery a
                // track's/send's chain uses) over its accumulated `submix_mix_l/r`, apply that
                // submix's `volume` fader (unlike a send bus, which has none — a submix stands in
                // for its member tracks' direct contribution to the mix), publish its own
                // `LoudnessMeter` reading (the post-chain, post-fader signal — the same tap point
                // a track's own meter reads), then sum into the master mix.
                submix_chain_out_l.resize(frames, 0.0);
                submix_chain_out_r.resize(frames, 0.0);
                for (submix_index, (mix_l, mix_r)) in
                    submix_mix_l.iter().zip(submix_mix_r.iter()).enumerate()
                {
                    let volume = snapshot.submixes.get(submix_index).map_or(1.0, |s| s.volume);
                    let (pan_l, pan_r) = equal_power_pan_gains(
                        snapshot.submixes.get(submix_index).map_or(0.0, |s| s.pan),
                    );
                    if let Ok(mut chains) = submix_effects.lock() {
                        let chain = chains
                            .get_mut(submix_index)
                            .map_or(&mut [][..], Vec::as_mut_slice);
                        let stage_scratch = &mut submix_scratch[submix_index];
                        while stage_scratch.len() < chain.len() {
                            stage_scratch.push(plugin_host::EffectScratch::new());
                        }
                        let used = plugin_host::process_effect_chain(
                            chain,
                            mix_l,
                            mix_r,
                            &mut submix_chain_out_l,
                            &mut submix_chain_out_r,
                            stage_scratch,
                            &mut submix_chain_run_l,
                            &mut submix_chain_run_r,
                            &all_track_dry,
                        );
                        if used {
                            for i in 0..frames {
                                submix_chain_out_l[i] *= volume * pan_l;
                                submix_chain_out_r[i] *= volume * pan_r;
                            }
                            let readings = submix_loudness[submix_index]
                                .process(&submix_chain_out_l, &submix_chain_out_r);
                            if let Some(handle) = published_submix_meters
                                .as_ref()
                                .and_then(|h| h.get(submix_index))
                            {
                                handle.publish(&readings);
                            }
                            for i in 0..frames {
                                scratch_l[i] += submix_chain_out_l[i];
                                scratch_r[i] += submix_chain_out_r[i];
                            }
                            continue;
                        }
                    }
                    for i in 0..frames {
                        submix_chain_out_l[i] = mix_l[i] * volume * pan_l;
                        submix_chain_out_r[i] = mix_r[i] * volume * pan_r;
                    }
                    let readings = submix_loudness[submix_index]
                        .process(&submix_chain_out_l, &submix_chain_out_r);
                    if let Some(handle) =
                        published_submix_meters.as_ref().and_then(|h| h.get(submix_index))
                    {
                        handle.publish(&readings);
                    }
                    for i in 0..frames {
                        scratch_l[i] += submix_chain_out_l[i];
                        scratch_r[i] += submix_chain_out_r[i];
                    }
                }

                for i in 0..frames {
                    scratch_l[i] += metronome_dry[i];
                    scratch_r[i] += metronome_dry[i];
                }

                for s in scratch_l.iter_mut() {
                    *s = (*s * MASTER_GAIN).tanh();
                }
                for s in scratch_r.iter_mut() {
                    *s = (*s * MASTER_GAIN).tanh();
                }
            } else {
                sequencer.reset_position();
                transport.current_tick.store(0, Ordering::Relaxed);
            }

            // Run the mix through the master bus's effect chain (CLAP and/or built-in stages, same
            // machinery a track's own chain uses — see `plugin_host::process_effect_chain`), if
            // any effects are loaded there. Falls back to the dry stereo mix if the chain is empty
            // or nothing in it actually processed. Channel counts for a CLAP stage come from what
            // the plugin actually declared via the `audio-ports` extension (see
            // `plugin_host::load_and_activate`) — assuming every effect is 2-in/2-out caused real
            // plugins (e.g. ZamDelay, which is mono-in) to read past their declared buffers.
            //
            // A `MasterEffectParam` automation lane (from any track's region, see
            // `AutomationTarget`) overrides the master chain's own params here, same
            // breakpoint-chunked approximation a track's/send's chain gets.
            let mut used_master_chain = false;
            if let Ok(mut chains) = master_effects.lock() {
                let chain = chains.get_mut(0).map_or(&mut [][..], Vec::as_mut_slice);
                while master_scratch.len() < chain.len() {
                    master_scratch.push(plugin_host::EffectScratch::new());
                }
                plugin_out_l.resize(frames, 0.0);
                plugin_out_r.resize(frames, 0.0);
                used_master_chain = process_chain_with_automation(
                    chain,
                    &master_automation.effect_params,
                    master_samples_per_tick,
                    &scratch_l,
                    &scratch_r,
                    &mut plugin_out_l,
                    &mut plugin_out_r,
                    &mut master_scratch,
                    &mut master_chain_run_l,
                    &mut master_chain_run_r,
                    &all_track_dry,
                );
            }

            let (left, right): (&[f32], &[f32]) = if used_master_chain {
                (&plugin_out_l, &plugin_out_r)
            } else {
                (&scratch_l, &scratch_r)
            };

            let master_readings = master_loudness.process(left, right);
            if let Ok(handles) = master_meter.lock()
                && let Some(handle) = handles.first()
            {
                handle.publish(&master_readings);
            }

            for (i, frame) in data.chunks_mut(channels).enumerate() {
                frame[0] = T::from_sample(left[i]);
                if channels > 1 {
                    let r = T::from_sample(right[i]);
                    for sample in &mut frame[1..] {
                        *sample = r;
                    }
                }
            }
        },
        err_fn,
        None,
    )?;
    Ok(stream)
}
