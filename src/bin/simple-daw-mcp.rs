//! Standalone MCP (Model Context Protocol) server that lets an LLM (Claude Desktop, Claude Code,
//! etc.) drive a *running* `simple-daw` instance.
//!
//! This binary speaks MCP over stdio (newline-delimited JSON-RPC 2.0, per the MCP stdio
//! transport) to whatever launched it, and forwards each `tools/call` to `simple-daw`'s local
//! control socket (see `crate::mcp_control` in the main binary) over a Unix domain socket. It has
//! no dependency on `simple-daw`'s own types — every payload is opaque JSON, so this binary is
//! just a thin protocol translator, not a client of the `Song`/`Track` model.
//!
//! Point an MCP-capable client at this binary directly, e.g. in Claude Desktop's config:
//! ```json
//! {"mcpServers": {"simple-daw": {"command": "/path/to/target/release/simple-daw-mcp"}}}
//! ```
//!
//! Unix-only (matches `mcp_control`'s Unix-domain-socket transport) — not built on other
//! platforms.

use std::io::{self, BufRead, Write};
use std::os::unix::net::UnixStream;

use serde_json::{json, Value};

/// Must match `mcp_control::socket_path()` in the main `simple-daw` binary — duplicated rather
/// than shared via a library crate, since this binary otherwise has zero dependency on
/// `simple-daw`'s internals and everything it forwards is opaque JSON.
fn socket_path() -> std::path::PathBuf {
    std::env::temp_dir().join("simple-daw.sock")
}

/// One MCP tool's declared schema, used to answer `tools/list`. Keep this list in sync with the
/// `match cmd { ... }` arms in `apply_mcp_command` (src/main.rs) — this binary has no way to
/// verify that at compile time since the two never link against each other.
fn tool_definitions() -> Vec<Value> {
    vec![
        json!({
            "name": "list_tracks",
            "description": "List every track in the currently open song, in order.",
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": false},
        }),
        json!({
            "name": "add_track",
            "description": "Add a new track to the song. Returns the new track's index (0-based).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Track name."},
                    "kind": {"type": "string", "enum": ["step_grid", "piano_roll", "audio"], "description": "step_grid = drum-machine style grid, piano_roll = free-form melodic notes, audio = recorded/imported clips."},
                    "midi_channel": {"type": "integer", "minimum": 0, "maximum": 15, "description": "Optional; defaults to the next free channel."},
                },
                "required": ["name", "kind"],
                "additionalProperties": false,
            },
        }),
        json!({
            "name": "remove_track",
            "description": "Delete a track and everything it owns (its regions/audio clips).",
            "inputSchema": {
                "type": "object",
                "properties": {"track": {"type": "integer", "description": "Track index from list_tracks."}},
                "required": ["track"],
                "additionalProperties": false,
            },
        }),
        json!({
            "name": "set_track_volume",
            "description": "Set a track's mix volume (linear gain; 1.0 is unity).",
            "inputSchema": {
                "type": "object",
                "properties": {"track": {"type": "integer"}, "volume": {"type": "number", "minimum": 0}},
                "required": ["track", "volume"],
                "additionalProperties": false,
            },
        }),
        json!({
            "name": "set_track_mute",
            "description": "Mute or unmute a track.",
            "inputSchema": {
                "type": "object",
                "properties": {"track": {"type": "integer"}, "muted": {"type": "boolean"}},
                "required": ["track", "muted"],
                "additionalProperties": false,
            },
        }),
        json!({
            "name": "set_track_solo",
            "description": "Solo or unsolo a track. While any track is soloed, only soloed tracks play.",
            "inputSchema": {
                "type": "object",
                "properties": {"track": {"type": "integer"}, "solo": {"type": "boolean"}},
                "required": ["track", "solo"],
                "additionalProperties": false,
            },
        }),
        json!({
            "name": "add_region",
            "description": "Add a new one-bar region to a step_grid or piano_roll track at the given step offset. Returns the new region's index. A step_grid region's lanes are copied from the track's last existing region, if any.",
            "inputSchema": {
                "type": "object",
                "properties": {"track": {"type": "integer"}, "start_step": {"type": "integer", "minimum": 0, "description": "Position in 16th-note steps from the start of the song."}},
                "required": ["track", "start_step"],
                "additionalProperties": false,
            },
        }),
        json!({
            "name": "add_lane",
            "description": "Add a new step-grid lane (e.g. a drum voice) to every region on a step_grid track. Returns the new lane's index.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "track": {"type": "integer"},
                    "name": {"type": "string", "description": "e.g. \"Kick\", \"Snare\"."},
                    "pitch": {"type": "integer", "minimum": 0, "maximum": 127, "description": "MIDI note number this lane triggers."},
                },
                "required": ["track", "name", "pitch"],
                "additionalProperties": false,
            },
        }),
        json!({
            "name": "set_step",
            "description": "Set or clear one step in a step-grid lane — the core beat-programming tool.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "track": {"type": "integer"},
                    "region": {"type": "integer"},
                    "lane": {"type": "integer"},
                    "step": {"type": "integer", "minimum": 0, "description": "16th-note step index within the region."},
                    "velocity": {"type": "integer", "minimum": 0, "maximum": 127, "description": "0 clears the step; 1-127 sets it (127 = loudest)."},
                },
                "required": ["track", "region", "lane", "step", "velocity"],
                "additionalProperties": false,
            },
        }),
        json!({
            "name": "add_note",
            "description": "Add a note to a piano_roll region. Returns the new note's id. Overlapping notes at the same pitch are trimmed automatically.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "track": {"type": "integer"},
                    "region": {"type": "integer"},
                    "pitch": {"type": "integer", "minimum": 0, "maximum": 127, "description": "MIDI note number."},
                    "start_step": {"type": "integer", "minimum": 0, "description": "Position in 16th-note steps from the start of the region."},
                    "length_steps": {"type": "integer", "minimum": 1, "description": "Duration in 16th-note steps."},
                    "velocity": {"type": "integer", "minimum": 0, "maximum": 127},
                },
                "required": ["track", "region", "pitch", "start_step", "length_steps", "velocity"],
                "additionalProperties": false,
            },
        }),
        json!({
            "name": "list_presets",
            "description": "List available synth presets (factory + song-local), optionally filtered by engine.",
            "inputSchema": {
                "type": "object",
                "properties": {"engine": {"type": "string", "enum": ["simple", "trine", "wave"]}},
                "additionalProperties": false,
            },
        }),
        json!({
            "name": "apply_preset",
            "description": "Apply a synth preset (by name, from list_presets) to a track, switching its synth engine to match.",
            "inputSchema": {
                "type": "object",
                "properties": {"track": {"type": "integer"}, "preset_name": {"type": "string"}},
                "required": ["track", "preset_name"],
                "additionalProperties": false,
            },
        }),
        json!({
            "name": "set_bpm",
            "description": "Set the song's tempo in beats per minute.",
            "inputSchema": {
                "type": "object",
                "properties": {"bpm": {"type": "number", "exclusiveMinimum": 0}},
                "required": ["bpm"],
                "additionalProperties": false,
            },
        }),
        json!({
            "name": "play",
            "description": "Start playback from the current position.",
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": false},
        }),
        json!({
            "name": "stop",
            "description": "Stop playback.",
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": false},
        }),
        json!({
            "name": "get_playback_state",
            "description": "Get whether the song is currently playing, the playhead position, bpm, and song name.",
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": false},
        }),
        json!({
            "name": "save_song",
            "description": "Save the current song to a JSON file at the given path.",
            "inputSchema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": false,
            },
        }),
        json!({
            "name": "load_song",
            "description": "Load a song from a JSON file at the given path, replacing the currently open song.",
            "inputSchema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": false,
            },
        }),
        json!({
            "name": "export_wav",
            "description": "Bounce the current song to a WAV file (dry mix, no CLAP effects).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "loops": {"type": "integer", "minimum": 1, "description": "How many times to loop the arrangement; defaults to 1."},
                },
                "required": ["path"],
                "additionalProperties": false,
            },
        }),
    ]
}

/// Sends one `cmd`/`params` request to `simple-daw`'s control socket and returns its `result` (on
/// `ok: true`) or an error string (on `ok: false`, a timeout, or a connection failure — the last
/// meaning `simple-daw` isn't running).
fn call_daw(cmd: &str, params: Value) -> Result<Value, String> {
    let path = socket_path();
    let stream = UnixStream::connect(&path).map_err(|err| {
        format!(
            "couldn't reach simple-daw at {path:?} ({err}) — is the app running?"
        )
    })?;
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(15)))
        .ok();
    let mut writer = stream.try_clone().map_err(|err| err.to_string())?;
    let request = json!({"id": 1, "cmd": cmd, "params": params});
    writeln!(writer, "{request}").map_err(|err| err.to_string())?;

    let mut reader = io::BufReader::new(stream);
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .map_err(|err| format!("lost connection to simple-daw: {err}"))?;
    if line.trim().is_empty() {
        return Err("simple-daw closed the connection without a response".to_string());
    }
    let response: Value =
        serde_json::from_str(&line).map_err(|err| format!("malformed response: {err}"))?;
    if response.get("ok").and_then(Value::as_bool) == Some(true) {
        Ok(response.get("result").cloned().unwrap_or(Value::Null))
    } else {
        Err(response
            .get("error")
            .and_then(Value::as_str)
            .unwrap_or("unknown error")
            .to_string())
    }
}

/// Wraps a tool-call outcome in the MCP `tools/call` result shape: a single text content block,
/// with `isError` set when `outcome` was an error. Tool results are always JSON-serialized text
/// rather than a typed schema — MCP doesn't require structured tool output, and a JSON string is
/// simple for the model to read back.
fn tool_result(outcome: Result<Value, String>) -> Value {
    match outcome {
        Ok(value) => json!({
            "content": [{"type": "text", "text": value.to_string()}],
            "isError": false,
        }),
        Err(err) => json!({
            "content": [{"type": "text", "text": err}],
            "isError": true,
        }),
    }
}

fn handle_request(method: &str, params: &Value, protocol_version: &str) -> Option<Value> {
    match method {
        "initialize" => Some(json!({
            "protocolVersion": protocol_version,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "simple-daw-mcp", "version": env!("CARGO_PKG_VERSION")},
        })),
        "tools/list" => Some(json!({ "tools": tool_definitions() })),
        "tools/call" => {
            let name = params.get("name").and_then(Value::as_str).unwrap_or("");
            let arguments = params.get("arguments").cloned().unwrap_or_else(|| json!({}));
            Some(tool_result(call_daw(name, arguments)))
        }
        "ping" => Some(json!({})),
        _ => None,
    }
}

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let message: Value = match serde_json::from_str(line) {
            Ok(value) => value,
            Err(_) => continue, // not valid JSON-RPC — nothing sensible to reply with, skip it
        };
        let id = message.get("id").cloned();
        let Some(method) = message.get("method").and_then(Value::as_str) else {
            continue;
        };
        let empty_params = json!({});
        let params = message.get("params").unwrap_or(&empty_params);

        // A message with no "id" is a notification (e.g. "notifications/initialized") — MCP
        // requires no response, so just apply any side effect and move on.
        let Some(id) = id else {
            let _ = handle_request(method, params, "");
            continue;
        };

        let protocol_version = message
            .get("params")
            .and_then(|p| p.get("protocolVersion"))
            .and_then(Value::as_str)
            .unwrap_or("2025-06-18")
            .to_string();

        let response = match handle_request(method, params, &protocol_version) {
            Some(result) => json!({"jsonrpc": "2.0", "id": id, "result": result}),
            None => json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": {"code": -32601, "message": format!("method not found: {method}")},
            }),
        };
        if writeln!(stdout, "{response}").is_err() || stdout.flush().is_err() {
            break;
        }
    }
}
