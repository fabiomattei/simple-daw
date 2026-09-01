//! The offline WAV bounce path: `render_song_to_wav` (File > Export and the MCP `export_wav`
//! tool) and `render_track_to_buffer` (the shared primitive behind track freeze and destructive
//! bounce-in-place). Both share `Sequencer::process` with real-time playback for synthesis, but
//! `mix_song_to_wav_buffer`'s wet mixdown (effect chains, automation, sends, submixes) is its own
//! separate, self-contained implementation — deliberately *not* shared with
//! `build_playback_stream`'s live mixdown, so nothing here can regress the real-time audio path.

use anyhow::{Context, Result};

use crate::model::{Song, TrackOutput};
use crate::plugin_host;

use super::{
    DelayLine, MASTER_GAIN, Sequencer, arrangement_length_ticks, collect_automation, equal_power_pan_gains,
    pdc_delay_samples_per_track, process_chain_with_automation, samples_for_tick_span, samples_per_tick_at,
};

/// Chunk size the offline bounce's mixdown (below) processes wet effect chains in. An offline
/// render has no cpal callback size to inherit, so this is picked directly — large enough to keep
/// CLAP `process()` call overhead low, small enough that automation still updates several times a
/// second even at slow tempos. Synthesis itself (`Sequencer::process`) isn't chunked; only the wet
/// mixdown below is, since that's the part that needs a `chunk_start_tick` to evaluate automation
/// against.
const OFFLINE_CHUNK_FRAMES: usize = 2048;

/// to a pair of L/R buffers spanning the arrangement once, at `sample_rate`. Used by both track
/// freeze (`Track::frozen`/`frozen_clip`) and destructive bounce-in-place: the shared "bake this
/// track down to audio" primitive behind both, matching `render_song_to_wav`'s own "share
/// `Sequencer::process` for synthesis, keep the wet mixdown separate" split — but for one track's
/// chain, not the whole song's master/send/submix mix. `None` only if `track_index` is out of
/// range or the arrangement is empty (nothing to render).
///
/// Deliberately does **not** consult the track's own `frozen`/`frozen_clip` — a track can be
/// re-frozen (or bounced) while already frozen, and doing so should re-render its live content
/// fresh, not the previous bake. Callers already frozen must clear `frozen` before calling this if
/// they want the track's *original* notes/steps back in the render.
pub fn render_track_to_buffer(
    song: &Song,
    track_index: usize,
    sample_rate: u32,
) -> Option<(Vec<f32>, Vec<f32>)> {
    let track = song.tracks.get(track_index)?;
    let arrangement_len_ticks = arrangement_length_ticks(song);
    let total_samples =
        samples_for_tick_span(song, sample_rate as f64, arrangement_len_ticks).round() as usize;
    if total_samples == 0 {
        return None;
    }

    let mut sequencer = Sequencer::new(sample_rate as f32);
    let mut track_dry_l: Vec<Vec<f32>> = Vec::new();
    let mut track_dry_r: Vec<Vec<f32>> = Vec::new();
    let mut metronome_dry: Vec<f32> = Vec::new();
    sequencer.process(
        song,
        total_samples,
        &mut track_dry_l,
        &mut track_dry_r,
        false,
        &mut metronome_dry,
        // The offline bounce always renders the Playlist arrangement, never a live Session View
        // performance (which has no persisted timeline to render) — see `render_song_to_wav`'s
        // doc comment.
        false,
        0,
        false, // capturing — the offline bounce never logs a Session View performance
    );

    let mut chain = plugin_host::load_offline_chain(
        &track.effects,
        sample_rate as f64,
        OFFLINE_CHUNK_FRAMES as u32,
    );
    let mut out_l = vec![0.0f32; total_samples];
    let mut out_r = vec![0.0f32; total_samples];
    let mut scratch: Vec<plugin_host::EffectScratch> =
        (0..chain.chain.len()).map(|_| plugin_host::EffectScratch::new()).collect();
    let mut run_l = Vec::with_capacity(OFFLINE_CHUNK_FRAMES);
    let mut run_r = Vec::with_capacity(OFFLINE_CHUNK_FRAMES);
    let all_track_dry: Vec<(&[f32], &[f32])> = track_dry_l
        .iter()
        .zip(track_dry_r.iter())
        .map(|(l, r)| (l.as_slice(), r.as_slice()))
        .collect();

    let mut chunk_start = 0;
    let mut tick_cursor = 0.0f64;
    while chunk_start < total_samples {
        let chunk_len = OFFLINE_CHUNK_FRAMES.min(total_samples - chunk_start);
        let chunk_start_tick = tick_cursor.round() as usize;
        let samples_per_tick =
            samples_per_tick_at(sample_rate as f64, song.bpm_at(chunk_start_tick));
        // The offline bounce always renders the Playlist arrangement, never a live Session View
        // performance (which has no persisted timeline to render) — see this loop's own doc
        // comment above and `render_song_to_wav`'s.
        let (track_automation, _master_automation, _send_automation) =
            collect_automation(song, chunk_start_tick, false, &[]);
        let automation = track_automation.get(track_index);
        let empty_effect_params = Vec::new();
        let effect_params = automation.map_or(&empty_effect_params, |a| &a.effect_params);
        let dry_l = &track_dry_l[track_index][chunk_start..chunk_start + chunk_len];
        let dry_r = &track_dry_r[track_index][chunk_start..chunk_start + chunk_len];
        process_chain_with_automation(
            &mut chain.chain,
            effect_params,
            samples_per_tick,
            dry_l,
            dry_r,
            &mut out_l[chunk_start..chunk_start + chunk_len],
            &mut out_r[chunk_start..chunk_start + chunk_len],
            &mut scratch,
            &mut run_l,
            &mut run_r,
            &all_track_dry,
        );
        chunk_start += chunk_len;
        tick_cursor += chunk_len as f64 / samples_per_tick;
    }

    Some((out_l, out_r))
}

/// Renders `loops` repetitions of the song's pattern content to a stereo 16-bit WAV file. Shares
/// `Sequencer::process` with real-time playback for synthesis, so a bounce's notes/steps sound
/// like what you'd hear live — but the wet mixdown below (effect chains, automation, sends,
/// submixes) is its own separate, self-contained implementation, deliberately *not* shared with
/// `build_playback_stream`'s live mixdown, so nothing here can regress the real-time audio path;
/// some chunking/effect-application logic is duplicated between the two as a result. Every CLAP
/// plugin is loaded fresh for the duration of this call (`plugin_host::OfflineEffectChain`,
/// distinct from the live, UI-loaded `TrackEffectSlots`) at this bounce's own `sample_rate`, which
/// may differ from any live session's.
///
/// Track/submix mute and solo are not consulted — every track always renders. That's this
/// function's pre-existing behavior (from before this wet mixdown existed), preserved rather than
/// changed as part of unrelated automation/effects work; if "bounces should respect mute/solo"
/// turns out to be wanted, it's a separate, focused change.
pub fn render_song_to_wav(
    song: &Song,
    sample_rate: u32,
    loops: u32,
    path: &std::path::Path,
) -> Result<()> {
    let arrangement_len_ticks = arrangement_length_ticks(song);
    let samples_per_cycle = samples_for_tick_span(song, sample_rate as f64, arrangement_len_ticks);
    let total_samples = (samples_per_cycle * (loops.max(1) as f64)).round() as usize;

    let mut sequencer = Sequencer::new(sample_rate as f32);
    let mut track_dry_l: Vec<Vec<f32>> = Vec::new();
    let mut track_dry_r: Vec<Vec<f32>> = Vec::new();
    let mut metronome_dry: Vec<f32> = Vec::new();
    // The metronome is a monitoring aid, not part of the song — bounces never include it.
    sequencer.process(
        song,
        total_samples,
        &mut track_dry_l,
        &mut track_dry_r,
        false,
        &mut metronome_dry,
        // The offline bounce always renders the Playlist arrangement, never a live Session View
        // performance (which has no persisted timeline to render) — see `render_song_to_wav`'s
        // doc comment.
        false,
        0,
        false, // capturing — the offline bounce never logs a Session View performance
    );

    let buffer_l = vec![0.0f32; total_samples];
    let buffer_r = vec![0.0f32; total_samples];
    let (buffer_l, buffer_r) = mix_song_to_wav_buffer(
        song,
        sample_rate,
        &track_dry_l,
        &track_dry_r,
        buffer_l,
        buffer_r,
    );

    let spec = hound::WavSpec {
        channels: 2,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .with_context(|| format!("failed to create wav file: {}", path.display()))?;
    for (l, r) in buffer_l.into_iter().zip(buffer_r) {
        let l = (l.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        let r = (r.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        writer
            .write_sample(l)
            .context("failed to write wav sample")?;
        writer
            .write_sample(r)
            .context("failed to write wav sample")?;
    }
    writer.finalize().context("failed to finalize wav file")?;
    Ok(())
}

/// The wet mixdown behind `render_song_to_wav` — loads a fresh `OfflineEffectChain` per track/
/// send/submix/master, then walks `track_dry_l/r` in `OFFLINE_CHUNK_FRAMES`-sized chunks, each
/// time collecting automation at that chunk's own start tick (`collect_automation`) and running
/// every bus through its chain (`process_chain_with_automation` for the three targets automation
/// can reach — track, send, master; submixes have no automation target defined at all, so a plain
/// `plugin_host::process_effect_chain` there), applying volume/pan/send-level/submix-volume the
/// same way `build_playback_stream`'s live mixdown does. Takes and returns the output buffers by
/// value (rather than `&mut`) purely so `render_song_to_wav` can hand off already-zeroed `Vec`s
/// without an extra explicit zero-fill call.
fn mix_song_to_wav_buffer(
    song: &Song,
    sample_rate: u32,
    track_dry_l: &[Vec<f32>],
    track_dry_r: &[Vec<f32>],
    mut buffer_l: Vec<f32>,
    mut buffer_r: Vec<f32>,
) -> (Vec<f32>, Vec<f32>) {
    let total_samples = buffer_l.len();
    let block = OFFLINE_CHUNK_FRAMES as u32;
    let mut track_chains: Vec<plugin_host::OfflineEffectChain> = song
        .tracks
        .iter()
        .map(|t| plugin_host::load_offline_chain(&t.effects, sample_rate as f64, block))
        .collect();
    let mut send_chains: Vec<plugin_host::OfflineEffectChain> = song
        .sends
        .iter()
        .map(|s| plugin_host::load_offline_chain(&s.effects, sample_rate as f64, block))
        .collect();
    let mut submix_chains: Vec<plugin_host::OfflineEffectChain> = song
        .submixes
        .iter()
        .map(|s| plugin_host::load_offline_chain(&s.effects, sample_rate as f64, block))
        .collect();
    let mut master_chain =
        plugin_host::load_offline_chain(&song.master_effects, sample_rate as f64, block);

    let mut track_scratch: Vec<Vec<plugin_host::EffectScratch>> = track_chains
        .iter()
        .map(|c| c.chain.iter().map(|_| plugin_host::EffectScratch::new()).collect())
        .collect();
    let mut send_scratch: Vec<Vec<plugin_host::EffectScratch>> = send_chains
        .iter()
        .map(|c| c.chain.iter().map(|_| plugin_host::EffectScratch::new()).collect())
        .collect();
    let mut submix_scratch: Vec<Vec<plugin_host::EffectScratch>> = submix_chains
        .iter()
        .map(|c| c.chain.iter().map(|_| plugin_host::EffectScratch::new()).collect())
        .collect();
    let mut master_scratch: Vec<plugin_host::EffectScratch> =
        master_chain.chain.iter().map(|_| plugin_host::EffectScratch::new()).collect();

    // Automatic plugin delay compensation: unlike the live mixdown (which re-derives this every
    // buffer, since a plugin can load/unload mid-session), every offline chain here is loaded once
    // for the whole render and never changes — so this is computed once too, not per chunk.
    let track_latency: Vec<u32> =
        track_chains.iter().map(|c| plugin_host::chain_latency_samples(&c.chain)).collect();
    let submix_latency: Vec<u32> =
        submix_chains.iter().map(|c| plugin_host::chain_latency_samples(&c.chain)).collect();
    let track_pdc_delay_samples = pdc_delay_samples_per_track(&song.tracks, &track_latency, &submix_latency);
    let mut track_pdc_delay: Vec<DelayLine> = (0..song.tracks.len()).map(|_| DelayLine::new()).collect();

    // Reused per-chunk scratch, sized to one chunk rather than the whole render, so memory stays
    // bounded regardless of how long the bounce is.
    let mut track_out_l = vec![0.0f32; OFFLINE_CHUNK_FRAMES];
    let mut track_out_r = vec![0.0f32; OFFLINE_CHUNK_FRAMES];
    let mut track_run_l = Vec::with_capacity(OFFLINE_CHUNK_FRAMES);
    let mut track_run_r = Vec::with_capacity(OFFLINE_CHUNK_FRAMES);
    let mut track_meter_l = vec![0.0f32; OFFLINE_CHUNK_FRAMES];
    let mut track_meter_r = vec![0.0f32; OFFLINE_CHUNK_FRAMES];
    let mut send_mix_l: Vec<Vec<f32>> = vec![vec![0.0; OFFLINE_CHUNK_FRAMES]; song.sends.len()];
    let mut send_mix_r: Vec<Vec<f32>> = vec![vec![0.0; OFFLINE_CHUNK_FRAMES]; song.sends.len()];
    let mut send_out_l = vec![0.0f32; OFFLINE_CHUNK_FRAMES];
    let mut send_out_r = vec![0.0f32; OFFLINE_CHUNK_FRAMES];
    let mut send_run_l = Vec::with_capacity(OFFLINE_CHUNK_FRAMES);
    let mut send_run_r = Vec::with_capacity(OFFLINE_CHUNK_FRAMES);
    let mut submix_mix_l: Vec<Vec<f32>> = vec![vec![0.0; OFFLINE_CHUNK_FRAMES]; song.submixes.len()];
    let mut submix_mix_r: Vec<Vec<f32>> = vec![vec![0.0; OFFLINE_CHUNK_FRAMES]; song.submixes.len()];
    let mut submix_out_l = vec![0.0f32; OFFLINE_CHUNK_FRAMES];
    let mut submix_out_r = vec![0.0f32; OFFLINE_CHUNK_FRAMES];
    let mut submix_run_l = Vec::with_capacity(OFFLINE_CHUNK_FRAMES);
    let mut submix_run_r = Vec::with_capacity(OFFLINE_CHUNK_FRAMES);
    let mut master_mix_l = vec![0.0f32; OFFLINE_CHUNK_FRAMES];
    let mut master_mix_r = vec![0.0f32; OFFLINE_CHUNK_FRAMES];
    let mut master_out_l = vec![0.0f32; OFFLINE_CHUNK_FRAMES];
    let mut master_out_r = vec![0.0f32; OFFLINE_CHUNK_FRAMES];
    let mut master_run_l = Vec::with_capacity(OFFLINE_CHUNK_FRAMES);
    let mut master_run_r = Vec::with_capacity(OFFLINE_CHUNK_FRAMES);
    let empty_effect_params = Vec::new();

    let mut chunk_start = 0;
    // Running tick position, advanced below by each chunk's own duration (`chunk_len /
    // samples_per_tick`) rather than derived by dividing `chunk_start` by one fixed rate — the
    // latter would be wrong as soon as a `Song::tempo_map` change makes the tick rate vary across
    // the render, the same reason `Sequencer::process`'s live clock tracks a running tick position
    // instead of computing it from elapsed sample count directly.
    let mut tick_cursor = 0.0f64;
    while chunk_start < total_samples {
        let chunk_len = OFFLINE_CHUNK_FRAMES.min(total_samples - chunk_start);
        let chunk_start_tick = tick_cursor.round() as usize;
        // Resolved once per chunk (like `build_playback_stream`'s buffer-granularity precision,
        // not `Sequencer::process`'s per-tick precision) — held constant through this chunk's
        // `LaneRef::value_at`/`process_chain_with_automation` calls below, which assume one rate
        // per call.
        let samples_per_tick =
            samples_per_tick_at(sample_rate as f64, song.bpm_at(chunk_start_tick));
        // Same "always the Playlist arrangement, never Session View" reasoning as
        // `render_song_to_wav`'s own `collect_automation` call above.
        let (track_automation, master_automation, send_automation) =
            collect_automation(song, chunk_start_tick, false, &[]);

        for buf in send_mix_l.iter_mut().chain(send_mix_r.iter_mut()) {
            buf[..chunk_len].fill(0.0);
        }
        for buf in submix_mix_l.iter_mut().chain(submix_mix_r.iter_mut()) {
            buf[..chunk_len].fill(0.0);
        }
        master_mix_l[..chunk_len].fill(0.0);
        master_mix_r[..chunk_len].fill(0.0);

        // See `build_playback_stream`'s own `all_track_dry` — same idea, sliced to this chunk.
        let all_track_dry: Vec<(&[f32], &[f32])> = track_dry_l
            .iter()
            .map(|buf| &buf[chunk_start..chunk_start + chunk_len])
            .zip(track_dry_r.iter().map(|buf| &buf[chunk_start..chunk_start + chunk_len]))
            .collect();

        for (track_index, track) in song.tracks.iter().enumerate() {
            let dry_l = &track_dry_l[track_index][chunk_start..chunk_start + chunk_len];
            let dry_r = &track_dry_r[track_index][chunk_start..chunk_start + chunk_len];
            let automation = track_automation.get(track_index);
            let volume_at = |i: usize| {
                automation
                    .and_then(|a| a.volume.map(|lane| lane.value_at(i, samples_per_tick)))
                    .unwrap_or(track.volume)
            };
            let pan_at = |i: usize| {
                automation
                    .and_then(|a| a.pan.map(|lane| lane.value_at(i, samples_per_tick)))
                    .unwrap_or(track.pan)
            };
            let effect_params = automation.map_or(&empty_effect_params, |a| &a.effect_params);
            // See the matching guard in `build_playback_stream`: a frozen track's own chain is
            // skipped since `dry_l/r` is already the baked, post-chain `frozen_clip` signal.
            let used = !track.frozen
                && process_chain_with_automation(
                    &mut track_chains[track_index].chain,
                    effect_params,
                    samples_per_tick,
                    dry_l,
                    dry_r,
                    &mut track_out_l[..chunk_len],
                    &mut track_out_r[..chunk_len],
                    &mut track_scratch[track_index],
                    &mut track_run_l,
                    &mut track_run_r,
                    &all_track_dry,
                );
            let source_l = if used { &track_out_l[..chunk_len] } else { dry_l };
            let source_r = if used { &track_out_r[..chunk_len] } else { dry_r };
            for i in 0..chunk_len {
                let (pan_l, pan_r) = equal_power_pan_gains(pan_at(i));
                track_meter_l[i] = volume_at(i) * pan_l * source_l[i];
                track_meter_r[i] = volume_at(i) * pan_r * source_r[i];
            }
            track_pdc_delay[track_index].process(
                &mut track_meter_l[..chunk_len],
                &mut track_meter_r[..chunk_len],
                track_pdc_delay_samples[track_index] as usize,
            );
            match track.output {
                TrackOutput::Master => {
                    for i in 0..chunk_len {
                        master_mix_l[i] += track_meter_l[i];
                        master_mix_r[i] += track_meter_r[i];
                    }
                }
                TrackOutput::Submix(index) => {
                    if let (Some(mix_l), Some(mix_r)) =
                        (submix_mix_l.get_mut(index), submix_mix_r.get_mut(index))
                    {
                        for i in 0..chunk_len {
                            mix_l[i] += track_meter_l[i];
                            mix_r[i] += track_meter_r[i];
                        }
                    }
                }
            }
            for (send_index, &static_level) in track.send_levels.iter().enumerate() {
                let lane = automation.and_then(|a| {
                    a.send_levels.iter().find(|(i, _)| *i == send_index).map(|(_, lane)| *lane)
                });
                let Some((mix_l, mix_r)) =
                    send_mix_l.get_mut(send_index).zip(send_mix_r.get_mut(send_index))
                else {
                    continue;
                };
                match lane {
                    Some(lane) => {
                        for i in 0..chunk_len {
                            let level = lane.value_at(i, samples_per_tick);
                            mix_l[i] += track_meter_l[i] * level;
                            mix_r[i] += track_meter_r[i] * level;
                        }
                    }
                    None => {
                        if static_level != 0.0 {
                            for i in 0..chunk_len {
                                mix_l[i] += track_meter_l[i] * static_level;
                                mix_r[i] += track_meter_r[i] * static_level;
                            }
                        }
                    }
                }
            }
        }

        for (send_index, chain) in send_chains.iter_mut().enumerate() {
            let effect_params =
                send_automation.get(send_index).map_or(&empty_effect_params, |s| &s.effect_params);
            let used = process_chain_with_automation(
                &mut chain.chain,
                effect_params,
                samples_per_tick,
                &send_mix_l[send_index][..chunk_len],
                &send_mix_r[send_index][..chunk_len],
                &mut send_out_l[..chunk_len],
                &mut send_out_r[..chunk_len],
                &mut send_scratch[send_index],
                &mut send_run_l,
                &mut send_run_r,
                &all_track_dry,
            );
            let (source_l, source_r) = if used {
                (&send_out_l[..chunk_len], &send_out_r[..chunk_len])
            } else {
                (&send_mix_l[send_index][..chunk_len], &send_mix_r[send_index][..chunk_len])
            };
            for i in 0..chunk_len {
                master_mix_l[i] += source_l[i];
                master_mix_r[i] += source_r[i];
            }
        }

        for (submix_index, chain) in submix_chains.iter_mut().enumerate() {
            let volume = song.submixes.get(submix_index).map_or(1.0, |s| s.volume);
            let (pan_l, pan_r) =
                equal_power_pan_gains(song.submixes.get(submix_index).map_or(0.0, |s| s.pan));
            let used = plugin_host::process_effect_chain(
                &mut chain.chain,
                &submix_mix_l[submix_index][..chunk_len],
                &submix_mix_r[submix_index][..chunk_len],
                &mut submix_out_l[..chunk_len],
                &mut submix_out_r[..chunk_len],
                &mut submix_scratch[submix_index],
                &mut submix_run_l,
                &mut submix_run_r,
                &all_track_dry,
            );
            let (source_l, source_r) = if used {
                (&submix_out_l[..chunk_len], &submix_out_r[..chunk_len])
            } else {
                (&submix_mix_l[submix_index][..chunk_len], &submix_mix_r[submix_index][..chunk_len])
            };
            for i in 0..chunk_len {
                master_mix_l[i] += source_l[i] * volume * pan_l;
                master_mix_r[i] += source_r[i] * volume * pan_r;
            }
        }

        let used = process_chain_with_automation(
            &mut master_chain.chain,
            &master_automation.effect_params,
            samples_per_tick,
            &master_mix_l[..chunk_len],
            &master_mix_r[..chunk_len],
            &mut master_out_l[..chunk_len],
            &mut master_out_r[..chunk_len],
            &mut master_scratch,
            &mut master_run_l,
            &mut master_run_r,
            &all_track_dry,
        );
        let (source_l, source_r) = if used {
            (&master_out_l[..chunk_len], &master_out_r[..chunk_len])
        } else {
            (&master_mix_l[..chunk_len], &master_mix_r[..chunk_len])
        };
        for i in 0..chunk_len {
            buffer_l[chunk_start + i] = (source_l[i] * MASTER_GAIN).tanh();
            buffer_r[chunk_start + i] = (source_r[i] * MASTER_GAIN).tanh();
        }

        chunk_start += chunk_len;
        tick_cursor += chunk_len as f64 / samples_per_tick;
    }

    (buffer_l, buffer_r)
}
