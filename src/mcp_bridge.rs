//! MCP command dispatch: applies queued commands from the `simple-daw-mcp` companion binary
//! (see `mcp_control`) to the live `Song`, driving it through the same model methods/helpers a
//! UI action would use. `apply_mcp_command` is called from `SimpleDawApp::update`, once per
//! queued request, before anything else that frame reads `song`.

use crate::audio::{self, Transport};
use crate::metering::MeterHandles;
use crate::model::{self, RegionContent, Song, SynthEngine, TICKS_PER_STEP, Track, TrackEffectConfig, TrackKind, add_note};
use crate::plugin_host::{DawHost, MasterEffectSlots, PluginGuiHandle, SendEffectSlots, SubmixEffectSlots, TrackEffectSlots};
use crate::factory_presets::factory_presets;
use crate::file_ops::{apply_loaded_effects, perform_load, perform_save, sync_song_effects};
use crate::{remove_track_effects, remove_track_meter, resize_track_effects, resize_track_meters};
use clack_host::prelude::PluginInstance;

/// Bundles the pieces of `SimpleDawApp` an MCP command handler needs — mirrors `main.rs`'s
/// `ChannelRackUi`'s "disjoint field borrows" pattern: `song` in `apply_mcp_command` is borrowed
/// straight from `self.song.lock()`, not through `self`, so a real `&mut self` method can't be
/// called alongside it; constructing this struct from individual `self.field` borrows can.
pub(crate) struct McpContext<'a> {
    pub(crate) transport: &'a Transport,
    pub(crate) sample_rate: Option<u32>,
    pub(crate) engine_config: Option<(f64, u32, u32)>,
    pub(crate) song_path: &'a mut String,
    pub(crate) master_effect_paths: &'a mut Vec<String>,
    pub(crate) master_effect_slots: &'a MasterEffectSlots,
    pub(crate) master_effect_instances: &'a mut Vec<Option<PluginInstance<DawHost>>>,
    pub(crate) master_effect_guis: &'a mut Vec<Option<PluginGuiHandle>>,
    pub(crate) master_effect_messages: &'a mut Vec<Option<(bool, String)>>,
    pub(crate) track_effect_slots: &'a TrackEffectSlots,
    pub(crate) track_effect_instances: &'a mut Vec<Vec<Option<PluginInstance<DawHost>>>>,
    pub(crate) track_effect_guis: &'a mut Vec<Vec<Option<PluginGuiHandle>>>,
    pub(crate) track_effect_paths: &'a mut Vec<Vec<String>>,
    pub(crate) track_effect_messages: &'a mut Vec<Vec<Option<(bool, String)>>>,
    pub(crate) send_effect_slots: &'a SendEffectSlots,
    pub(crate) send_effect_instances: &'a mut Vec<Vec<Option<PluginInstance<DawHost>>>>,
    pub(crate) send_effect_guis: &'a mut Vec<Vec<Option<PluginGuiHandle>>>,
    pub(crate) send_effect_paths: &'a mut Vec<Vec<String>>,
    pub(crate) send_effect_messages: &'a mut Vec<Vec<Option<(bool, String)>>>,
    pub(crate) submix_effect_slots: &'a SubmixEffectSlots,
    pub(crate) submix_effect_instances: &'a mut Vec<Vec<Option<PluginInstance<DawHost>>>>,
    pub(crate) submix_effect_guis: &'a mut Vec<Vec<Option<PluginGuiHandle>>>,
    pub(crate) submix_effect_paths: &'a mut Vec<Vec<String>>,
    pub(crate) submix_effect_messages: &'a mut Vec<Vec<Option<(bool, String)>>>,
    pub(crate) track_meters: &'a MeterHandles,
    pub(crate) submix_meters: &'a MeterHandles,
}

fn parse_track_kind(s: &str) -> Result<TrackKind, String> {
    match s {
        "step_grid" => Ok(TrackKind::StepGrid),
        "piano_roll" => Ok(TrackKind::PianoRoll),
        "audio" => Ok(TrackKind::Audio),
        other => Err(format!(
            "unknown track kind \"{other}\" (expected step_grid, piano_roll, or audio)"
        )),
    }
}

fn track_kind_str(kind: TrackKind) -> &'static str {
    match kind {
        TrackKind::StepGrid => "step_grid",
        TrackKind::PianoRoll => "piano_roll",
        TrackKind::Audio => "audio",
    }
}

fn parse_synth_engine(s: &str) -> Result<SynthEngine, String> {
    match s {
        "simple" => Ok(SynthEngine::Simple),
        "trine" => Ok(SynthEngine::Trine),
        "wave" => Ok(SynthEngine::Wave),
        other => Err(format!(
            "unknown synth engine \"{other}\" (expected simple, trine, or wave)"
        )),
    }
}

fn synth_engine_str(engine: SynthEngine) -> &'static str {
    match engine {
        SynthEngine::Simple => "simple",
        SynthEngine::Trine => "trine",
        SynthEngine::Wave => "wave",
    }
}

fn mcp_track_mut(song: &mut Song, index: usize) -> Result<&mut Track, String> {
    let track_count = song.tracks.len();
    song.tracks
        .get_mut(index)
        .ok_or_else(|| format!("no track at index {index} (song has {track_count} tracks)"))
}

#[derive(serde::Deserialize)]
struct McpAddTrackParams {
    name: String,
    kind: String,
    #[serde(default)]
    midi_channel: Option<u8>,
}

#[derive(serde::Deserialize)]
struct McpTrackIndexParams {
    track: usize,
}

#[derive(serde::Deserialize)]
struct McpSetTrackVolumeParams {
    track: usize,
    volume: f32,
}

#[derive(serde::Deserialize)]
struct McpSetTrackMuteParams {
    track: usize,
    muted: bool,
}

#[derive(serde::Deserialize)]
struct McpSetTrackSoloParams {
    track: usize,
    solo: bool,
}

#[derive(serde::Deserialize)]
struct McpAddRegionParams {
    track: usize,
    start_step: usize,
}

#[derive(serde::Deserialize)]
struct McpAddLaneParams {
    track: usize,
    name: String,
    pitch: u8,
}

#[derive(serde::Deserialize)]
struct McpSetStepParams {
    track: usize,
    region: usize,
    lane: usize,
    step: usize,
    /// 0 clears the step; 1-127 sets it with that velocity (mirrors `Lane::steps`' own encoding).
    velocity: u8,
}

#[derive(serde::Deserialize)]
struct McpAddNoteParams {
    track: usize,
    region: usize,
    pitch: u8,
    start_step: usize,
    length_steps: usize,
    velocity: u8,
}

#[derive(serde::Deserialize, Default)]
struct McpListPresetsParams {
    #[serde(default)]
    engine: Option<String>,
}

#[derive(serde::Deserialize)]
struct McpApplyPresetParams {
    track: usize,
    preset_name: String,
}

#[derive(serde::Deserialize)]
struct McpSetBpmParams {
    bpm: f32,
}

#[derive(serde::Deserialize)]
struct McpSaveSongParams {
    path: String,
}

#[derive(serde::Deserialize)]
struct McpLoadSongParams {
    path: String,
}

fn default_export_loops() -> u32 {
    1
}

#[derive(serde::Deserialize)]
struct McpExportWavParams {
    path: String,
    #[serde(default = "default_export_loops")]
    loops: u32,
}

/// Parses `params` into `T`, mapping a schema mismatch into the same `Result<_, String>` shape
/// every other MCP command handler returns — so a bad tool call from the LLM comes back as a
/// normal tool error instead of panicking the socket-handling thread.
fn parse_mcp_params<T: serde::de::DeserializeOwned>(params: serde_json::Value) -> Result<T, String> {
    serde_json::from_value(params).map_err(|err| format!("invalid parameters: {err}"))
}

/// Applies one queued MCP command (see `mcp_control::McpRequest`) to the live song, returning
/// either a JSON result payload or a human-readable error — the latter is sent back to
/// `simple-daw-mcp`, which surfaces it to the LLM as a tool error rather than a crash. Every
/// mutation here goes through the same model methods/helpers a UI button would use, so behavior
/// (and invariants like keeping `track_effect_*` aligned with `song.tracks`) stays identical to
/// driving the app by hand. Keep the tool names/params here in sync with the schemas declared in
/// `src/bin/simple-daw-mcp.rs`'s `tool_definitions()`.
pub(crate) fn apply_mcp_command(
    cmd: &str,
    params: serde_json::Value,
    song: &mut Song,
    ctx: &mut McpContext,
) -> Result<serde_json::Value, String> {
    use serde_json::json;

    match cmd {
        "list_tracks" => {
            let tracks: Vec<serde_json::Value> = song
                .tracks
                .iter()
                .enumerate()
                .map(|(index, track)| {
                    json!({
                        "index": index,
                        "name": track.name,
                        "kind": track_kind_str(track.kind),
                        "midi_channel": track.midi_channel,
                        "muted": track.muted,
                        "solo": track.solo,
                        "volume": track.volume,
                        "synth_engine": synth_engine_str(track.synth_engine),
                        "region_count": track.regions.len(),
                    })
                })
                .collect();
            Ok(json!({ "tracks": tracks }))
        }
        "add_track" => {
            let p: McpAddTrackParams = parse_mcp_params(params)?;
            let kind = parse_track_kind(&p.kind)?;
            let midi_channel = p
                .midi_channel
                .unwrap_or((song.tracks.len() as u8 % 16) + 1);
            let index = song.add_track(p.name, midi_channel, kind);
            resize_track_effects(
                ctx.track_effect_slots,
                ctx.track_effect_instances,
                ctx.track_effect_guis,
                ctx.track_effect_paths,
                ctx.track_effect_messages,
                song.tracks.len(),
            );
            resize_track_meters(ctx.track_meters, song.tracks.len());
            Ok(json!({ "index": index }))
        }
        "remove_track" => {
            let p: McpTrackIndexParams = parse_mcp_params(params)?;
            if p.track >= song.tracks.len() {
                return Err(format!(
                    "no track at index {} (song has {} tracks)",
                    p.track,
                    song.tracks.len()
                ));
            }
            song.remove_track(p.track);
            remove_track_effects(
                ctx.track_effect_slots,
                ctx.track_effect_instances,
                ctx.track_effect_guis,
                ctx.track_effect_paths,
                ctx.track_effect_messages,
                p.track,
            );
            remove_track_meter(ctx.track_meters, p.track);
            Ok(json!({}))
        }
        "set_track_volume" => {
            let p: McpSetTrackVolumeParams = parse_mcp_params(params)?;
            mcp_track_mut(song, p.track)?.volume = p.volume.max(0.0);
            Ok(json!({}))
        }
        "set_track_mute" => {
            let p: McpSetTrackMuteParams = parse_mcp_params(params)?;
            mcp_track_mut(song, p.track)?.muted = p.muted;
            Ok(json!({}))
        }
        "set_track_solo" => {
            let p: McpSetTrackSoloParams = parse_mcp_params(params)?;
            mcp_track_mut(song, p.track)?.solo = p.solo;
            Ok(json!({}))
        }
        "add_region" => {
            let p: McpAddRegionParams = parse_mcp_params(params)?;
            let steps_per_bar = song.steps_per_bar();
            let track = mcp_track_mut(song, p.track)?;
            if track.kind == TrackKind::Audio {
                return Err("audio tracks use audio_clips, not regions".to_string());
            }
            let region_index = track.add_region(p.start_step, steps_per_bar);
            Ok(json!({ "region_index": region_index }))
        }
        "add_lane" => {
            let p: McpAddLaneParams = parse_mcp_params(params)?;
            let track = mcp_track_mut(song, p.track)?;
            if track.kind != TrackKind::StepGrid {
                return Err("add_lane only applies to step_grid tracks".to_string());
            }
            track.add_lane(p.name, p.pitch);
            let lane_index = track
                .regions
                .first()
                .and_then(|region| match &region.content {
                    RegionContent::StepGrid(lanes) => Some(lanes.len().saturating_sub(1)),
                    _ => None,
                })
                .unwrap_or(0);
            Ok(json!({ "lane_index": lane_index }))
        }
        "set_step" => {
            let p: McpSetStepParams = parse_mcp_params(params)?;
            let track = mcp_track_mut(song, p.track)?;
            let region = track
                .regions
                .get_mut(p.region)
                .ok_or_else(|| format!("no region {} on track {}", p.region, p.track))?;
            let RegionContent::StepGrid(lanes) = &mut region.content else {
                return Err("region is not a step-grid region".to_string());
            };
            let lane = lanes
                .get_mut(p.lane)
                .ok_or_else(|| format!("no lane {} in region {}", p.lane, p.region))?;
            if p.step >= lane.steps.len() {
                return Err(format!(
                    "step {} out of range (region has {} steps)",
                    p.step,
                    lane.steps.len()
                ));
            }
            if p.velocity == 0 {
                lane.steps[p.step] = None;
            } else {
                lane.set_step(p.step, p.velocity.min(127));
            }
            Ok(json!({}))
        }
        "add_note" => {
            let p: McpAddNoteParams = parse_mcp_params(params)?;
            let steps_per_bar = song.steps_per_bar();
            let next_note_id = &mut song.next_note_id;
            let track_count = song.tracks.len();
            let track = song
                .tracks
                .get_mut(p.track)
                .ok_or_else(|| format!("no track at index {} (song has {track_count} tracks)", p.track))?;
            let region = track
                .regions
                .get_mut(p.region)
                .ok_or_else(|| format!("no region {} on track {}", p.region, p.track))?;
            let RegionContent::PianoRoll(notes) = &mut region.content else {
                return Err("region is not a piano-roll region".to_string());
            };
            let id = add_note(
                notes,
                next_note_id,
                p.pitch,
                p.start_step * TICKS_PER_STEP,
                p.length_steps * TICKS_PER_STEP,
                p.velocity.min(127),
            );
            model::grow_length_to_fit_notes(&mut region.content_length_steps, notes, steps_per_bar);
            Ok(json!({ "note_id": id }))
        }
        "list_presets" => {
            let p: McpListPresetsParams = parse_mcp_params(params)?;
            let engine_filter = p.engine.as_deref().map(parse_synth_engine).transpose()?;
            let presets: Vec<serde_json::Value> = factory_presets()
                .into_iter()
                .chain(song.synth_presets.clone())
                .filter(|preset| engine_filter.map_or(true, |e| preset.engine == e))
                .map(|preset| json!({"name": preset.name, "engine": synth_engine_str(preset.engine)}))
                .collect();
            Ok(json!({ "presets": presets }))
        }
        "apply_preset" => {
            let p: McpApplyPresetParams = parse_mcp_params(params)?;
            let preset = factory_presets()
                .into_iter()
                .chain(song.synth_presets.clone())
                .find(|preset| preset.name == p.preset_name)
                .ok_or_else(|| format!("no preset named \"{}\"", p.preset_name))?;
            let track = mcp_track_mut(song, p.track)?;
            track.synth_engine = preset.engine;
            match preset.engine {
                SynthEngine::Simple => track.synth = preset.params,
                SynthEngine::Trine => {
                    if let Some(trine) = preset.trine {
                        track.trine = trine;
                    }
                }
                SynthEngine::Wave => {
                    if let Some(wave) = preset.wave {
                        track.wave = wave;
                    }
                }
            }
            Ok(json!({}))
        }
        "set_bpm" => {
            let p: McpSetBpmParams = parse_mcp_params(params)?;
            if p.bpm <= 0.0 {
                return Err("bpm must be positive".to_string());
            }
            song.bpm = p.bpm;
            Ok(json!({}))
        }
        "play" => {
            ctx.transport.set_playing(true);
            Ok(json!({}))
        }
        "stop" => {
            ctx.transport.set_playing(false);
            Ok(json!({}))
        }
        "get_playback_state" => Ok(json!({
            "playing": ctx.transport.is_playing(),
            "current_tick": ctx.transport.current_tick(),
            "bpm": song.bpm,
            "song_name": song.name,
        })),
        "save_song" => {
            let p: McpSaveSongParams = parse_mcp_params(params)?;
            sync_song_effects(
                song,
                ctx.master_effect_paths,
                ctx.master_effect_slots,
                ctx.track_effect_paths,
                ctx.track_effect_slots,
                ctx.send_effect_paths,
                ctx.send_effect_slots,
                ctx.submix_effect_paths,
                ctx.submix_effect_slots,
            );
            let path = p.path.trim().to_string();
            let (ok, message) = perform_save(song, &path);
            if ok {
                *ctx.song_path = path;
                Ok(json!({ "message": message }))
            } else {
                Err(message)
            }
        }
        "load_song" => {
            let p: McpLoadSongParams = parse_mcp_params(params)?;
            let path = p.path.trim().to_string();
            let loaded = perform_load(&path, ctx.sample_rate)?;
            let track_count = loaded.tracks.len();
            let send_count = loaded.sends.len();
            let submix_count = loaded.submixes.len();
            let master_effect_specs = loaded.master_effects.clone();
            let track_effect_specs: Vec<Vec<TrackEffectConfig>> =
                loaded.tracks.iter().map(|t| t.effects.clone()).collect();
            let send_effect_specs: Vec<Vec<TrackEffectConfig>> =
                loaded.sends.iter().map(|s| s.effects.clone()).collect();
            let submix_effect_specs: Vec<Vec<TrackEffectConfig>> =
                loaded.submixes.iter().map(|s| s.effects.clone()).collect();
            *song = loaded;
            *ctx.song_path = path;
            resize_track_effects(
                ctx.track_effect_slots,
                ctx.track_effect_instances,
                ctx.track_effect_guis,
                ctx.track_effect_paths,
                ctx.track_effect_messages,
                track_count,
            );
            resize_track_meters(ctx.track_meters, track_count);
            resize_track_effects(
                ctx.send_effect_slots,
                ctx.send_effect_instances,
                ctx.send_effect_guis,
                ctx.send_effect_paths,
                ctx.send_effect_messages,
                send_count,
            );
            resize_track_effects(
                ctx.submix_effect_slots,
                ctx.submix_effect_instances,
                ctx.submix_effect_guis,
                ctx.submix_effect_paths,
                ctx.submix_effect_messages,
                submix_count,
            );
            resize_track_meters(ctx.submix_meters, submix_count);
            apply_loaded_effects(
                ctx.master_effect_paths,
                ctx.master_effect_instances,
                ctx.master_effect_guis,
                ctx.master_effect_slots,
                ctx.master_effect_messages,
                master_effect_specs,
                ctx.track_effect_paths,
                ctx.track_effect_instances,
                ctx.track_effect_guis,
                ctx.track_effect_messages,
                ctx.track_effect_slots,
                track_effect_specs,
                ctx.send_effect_paths,
                ctx.send_effect_instances,
                ctx.send_effect_guis,
                ctx.send_effect_messages,
                ctx.send_effect_slots,
                send_effect_specs,
                ctx.submix_effect_paths,
                ctx.submix_effect_instances,
                ctx.submix_effect_guis,
                ctx.submix_effect_messages,
                ctx.submix_effect_slots,
                submix_effect_specs,
                ctx.engine_config,
            );
            Ok(json!({ "track_count": track_count }))
        }
        "export_wav" => {
            let p: McpExportWavParams = parse_mcp_params(params)?;
            let sample_rate = ctx
                .sample_rate
                .ok_or_else(|| "no audio engine available (can't determine sample rate)".to_string())?;
            let path = std::path::Path::new(p.path.trim());
            audio::render_song_to_wav(song, sample_rate, p.loops.max(1), path)
                .map_err(|err| format!("{err:#}"))?;
            Ok(json!({ "path": p.path }))
        }
        other => Err(format!("unknown command \"{other}\"")),
    }
}
