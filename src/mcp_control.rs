//! Local control channel for the `simple-daw-mcp` companion binary (`src/bin/simple-daw-mcp.rs`).
//!
//! This module owns a background thread that listens on a Unix domain socket for
//! newline-delimited JSON requests (`{"id":.., "cmd":.., "params":..}`) and queues them for the
//! UI thread to apply — see `apply_mcp_command`/`McpContext` in `main.rs`. Commands are *not*
//! applied on this thread: several of them (adding/removing a track) have to stay in lockstep
//! with UI-only bookkeeping (`SimpleDawApp::track_effect_instances` and friends) that only the UI
//! thread touches, so every command is funneled through the same per-frame path a button click
//! would take. This also means a command's effects don't take place until the next `ui()` call —
//! `request_repaint()` below wakes the app up promptly instead of waiting for the next scheduled
//! frame.
//!
//! Unix-only for now (`UnixListener`/`UnixStream`); not compiled on other platforms. A Windows
//! build would need a named pipe or loopback TCP equivalent here plus in the companion binary.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::OnceLock;

use serde::Deserialize;
use serde_json::{json, Value};

/// One request forwarded from `simple-daw-mcp`, queued for `SimpleDawApp::ui` to apply on the
/// next frame. `reply` is a one-shot channel (a fresh `mpsc::channel()` per request) the UI
/// thread sends the result back through; the socket-handling thread blocks on it.
pub struct McpRequest {
    pub cmd: String,
    pub params: Value,
    pub reply: Sender<Result<Value, String>>,
}

#[derive(Deserialize)]
struct WireRequest {
    #[serde(default)]
    id: Value,
    cmd: String,
    #[serde(default)]
    params: Value,
}

/// Set once, from the first `ui()` call, so the socket thread (which has no other way to reach
/// the egui event loop) can nudge the app into rendering a frame right after queuing a command.
static REPAINT_CTX: OnceLock<egui::Context> = OnceLock::new();

/// Records the live `egui::Context` for `request_repaint` to use. Cheap to call every frame —
/// `OnceLock::set` after the first call is just an ignored no-op.
pub fn set_repaint_context(ctx: egui::Context) {
    let _ = REPAINT_CTX.set(ctx);
}

fn request_repaint() {
    if let Some(ctx) = REPAINT_CTX.get() {
        ctx.request_repaint();
    }
}

/// Fixed, well-known path so `simple-daw-mcp` doesn't need any configuration to find a running
/// `simple-daw` instance. Only one `simple-daw` is expected to run at a time per user.
pub fn socket_path() -> std::path::PathBuf {
    std::env::temp_dir().join("simple-daw.sock")
}

/// Spawns the listener thread and returns the receiving end of the command queue. Binding
/// failure (permissions, an already-running instance holding the socket) disables MCP control
/// for this run rather than aborting the app — logged to stderr, `rx` then simply never yields
/// anything.
pub fn spawn() -> Receiver<McpRequest> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let path = socket_path();
        // Remove a stale socket file left behind by a previous crashed run — `bind` fails with
        // AddrInUse otherwise even though nothing is actually listening.
        let _ = std::fs::remove_file(&path);
        match UnixListener::bind(&path) {
            Ok(listener) => {
                for stream in listener.incoming().flatten() {
                    let tx = tx.clone();
                    std::thread::spawn(move || handle_client(stream, &tx));
                }
            }
            Err(err) => {
                eprintln!("simple-daw: MCP control socket disabled ({err} at {path:?})");
            }
        }
    });
    rx
}

/// Reads newline-delimited JSON requests from one client connection until it disconnects,
/// dispatching each to the UI thread and writing back one JSON response line per request.
fn handle_client(stream: UnixStream, tx: &Sender<McpRequest>) {
    let Ok(reader_stream) = stream.try_clone() else {
        return;
    };
    let reader = BufReader::new(reader_stream);
    let mut writer = stream;
    for line in reader.lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let response = dispatch_line(&line, tx);
        if writeln!(writer, "{response}").is_err() {
            break;
        }
    }
}

fn dispatch_line(line: &str, tx: &Sender<McpRequest>) -> Value {
    let req: WireRequest = match serde_json::from_str(line) {
        Ok(req) => req,
        Err(err) => {
            return json!({"id": Value::Null, "ok": false, "error": format!("bad request: {err}")});
        }
    };
    let id = req.id.clone();
    let (reply_tx, reply_rx) = mpsc::channel();
    if tx
        .send(McpRequest {
            cmd: req.cmd,
            params: req.params,
            reply: reply_tx,
        })
        .is_err()
    {
        return json!({"id": id, "ok": false, "error": "simple-daw is shutting down"});
    }
    request_repaint();
    // The UI thread only drains the queue once per rendered frame; a generous timeout avoids a
    // hung client if the app is somehow stuck (e.g. a modal file dialog blocking the event loop).
    match reply_rx.recv_timeout(std::time::Duration::from_secs(10)) {
        Ok(Ok(result)) => json!({"id": id, "ok": true, "result": result}),
        Ok(Err(err)) => json!({"id": id, "ok": false, "error": err}),
        Err(_) => json!({"id": id, "ok": false, "error": "timed out waiting for simple-daw"}),
    }
}
