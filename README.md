# simple-daw

A native Rust MIDI sequencer / groovebox for composing video game music — not a general-purpose DAW.

Scope is deliberately narrow: step-sequencer + piano-roll MIDI editing, three built-in synth engines, sample playback, a Playlist/arrangement timeline, WAV bounce, MIDI import, JSON song persistence, CLAP effect hosting, composition tools (quantize/humanize/groove templates, tap tempo, audio tempo detection, and a tempo-map "Smart Tempo"), and non-destructive audio-clip editing (trim/fades, transient detection/Strip Silence, take-folder comping, and WSOLA-based time-stretch/Flex Time/Flex Pitch).

## Features

- **Step sequencer** — drum-machine style grid (`Lane`s with per-step velocity). Each lane can optionally be backed by a loaded WAV sample instead of the synth, or given its own synth engine/patch overriding the track's (via the 🎹 button on the lane row) — so one track can mix a kick patch on one lane with a hi-hat patch on another. A loaded sample still takes priority over any synth.
- **Piano roll** — free-form melodic editing on a custom-painted canvas: click-drag to draw notes of any length, drag to move/resize, multi-select (Ctrl/Cmd-click, Shift-drag box select) to move or delete several notes at once, with a per-note velocity lane. See [Using the piano roll](#using-the-piano-roll) below.
- **Three synth engines, selectable per track** — `Simple` (the original sine-family oscillator + exponential-decay envelope, percussive only), `Trine` (multi-oscillator with a filter/mod matrix), and `Wave` (two wavetable oscillators with position-morph and phase-warp, on the same filter/mod-matrix machinery as `Trine`). A library of built-in factory presets ships for all three.
- **Playlist / arrangement** — each track owns its own independently positioned `Region`s (step-grid or piano-roll content, Logic/Ableton-style — not a shared FL Studio–style pattern), dragged and resized along a shared song timeline.
- **Session View** — an Ableton-style clip-launching grid (tracks × rows) as an alternative to the Playlist: right-click an empty slot to assign an already-authored Region or audio clip into it, then click to launch/stop it, quantized to the next bar (or another note value, from the Quantize dropdown). Session View plays at most one clip per track at a time — launching a new one always stops whatever else was playing there. A "▶" button per row launches a whole scene (every track's clip at that row at once, leaving tracks with nothing there untouched). Right-click a filled slot for **Legato** (a launch continues the outgoing clip's playback phase instead of restarting silent-to-beat-one) and **Follow Action…** (auto-trigger another clip — Next/Previous/First/Last/Any/a specific row/itself again — after playing through a set number of times, with two independently-weighted candidate actions like Ableton's own). A toolbar toggle switches the transport between playing the Playlist arrangement and playing Session View — never both at once.
- **Sample playback** — WAV one-shots via a per-track 32-voice sample player pool, resampled to the output device's rate at load time.
- **Audio tracks** — record live input straight to a WAV file and drop it on the timeline as a clip, or import an existing WAV the same way. Recordings made from the same playhead position group into a "Take Folder" instead of piling up as overlapping clips (see **Take Folder comping** below).
- **Non-destructive clip trim and fades** — drag a clip's edges in the Playlist to trim its head/tail or set a fade in/out, without touching the source file.
- **Transient detection and Strip Silence** — a clip's waveform shows detected attacks as tick marks; right-click a clip and choose "Strip Silence" to split it into separate clips around the silent gaps, each still anchored to its original position in time.
- **Take Folder comping** — right-click a Take Folder to pick which whole take is heard, or double-click it to open the comp editor and drag across different takes' lanes to build a composite from pieces of each ("quick-swipe" comping).
- **Flex Time and Flex Pitch (time-stretch)** — right-click a clip → "Flex Time / Pitch…" for a hand-rolled, pitch-preserving time-stretch (WSOLA): drag warp points snapped to detected transients to stretch/compress the audio around them, or switch to the Pitch tab to drag a detected note to retarget its pitch. No external DSP library — verified for correct duration/pitch in tests, but not by ear, so listen before trusting it on anything you care about.
- **MIDI import** — load a standard `.mid` file into a track's piano roll.
- **Song persistence** — save/load the whole song (tracks, regions, sample paths, loaded effect paths and parameter values) to/from a JSON file via the File menu.
- **WAV export** — bounce the song to a file for a configurable number of loops; export uses the exact same sequencing/mixing code as live playback (dry only — effects, sends, submixes, and automation aren't included in the bounce, only each track's Volume/Pan and the raw synth/sample/step sequencing).
- **Effect chains, master bus, per track, per send, and per submix** — mix built-in DSP (delay, reverb, compressor, limiter, channel EQ, and more) with hosted CLAP plugins in any order, on the master bus, any track, any send bus, or any submix bus. New songs start with a default Limiter loaded on the master bus, the way Logic ships an Adaptive Limiter on master by default. CLAP hosting has host-side parameter editing (a generic "Params" window driven by the plugin's declared CLAP parameters) but no plugin GUI and no unload mid-session once a plugin is loaded.
- **Send buses** — aux buses with their own FX chain; each track gets a per-send level knob (post-fader) on its Mixer channel strip.
- **Submix buses (Track Stacks / alternate output routing)** — route a track's output into a shared submix bus instead of straight to master, picked from an "Output" dropdown on its Mixer channel strip. A submix has its own fader, Mute/Solo, and FX chain, so a group of tracks can share one compressor/reverb instance and be ridden together with one fader — Logic's Track Stack, minimal version.
- **Region fades and automation** — drag a Region's corner handles in the Playlist for a fade in/out; every Region also carries its own automation lanes (volume/pan/send-level/effect-param "rides," multi-point, edited under the Piano Roll/Beats grid) scoped to that Region's own track. Each point's segment into the next can be a straight line, an eased curve (Exponential/Logarithmic), or a Hold step — click an existing point to cycle its shape.
- **Metering** — peak/RMS bar meters and BS.1770-4 LUFS (momentary/short-term/integrated) on every Mixer channel strip, the master strip, and every submix strip.
- **Quantize, humanize, and groove templates** — snap piano-roll notes to a grid at an adjustable strength, randomly nudge timing/velocity, or apply a built-in swing/accent template (straight, swing 8th/16th, MPC push/lay-back) from the Piano Roll toolbar. Step-grid lanes get the same humanize/groove-template treatment from a per-lane "🎲" menu in the Beats window, including a small per-step timing offset (not just velocity) that the step-grid never had before.
- **Tap tempo and audio tempo detection** — tap the transport LCD's TAP button in time to set the song's BPM, or use File → Detect Tempo… to estimate a WAV file's BPM from its audio content (best on clearly rhythmic material like a drum loop or click track) and apply it to the song.
- **Tempo map ("Smart Tempo")** — the Playlist's Tempo Track panel lets you insert tempo-change points at the playhead, so a song's tempo can change partway through instead of staying fixed for the whole arrangement. Tempo changes are instant (a step function), not a smooth ramp.

## Building

### Prerequisites

- A Rust toolchain (install via [rustup](https://rustup.rs/))
- On Linux: ALSA development headers for cpal's audio backend, e.g. on Fedora:
  ```bash
  sudo dnf install alsa-lib-devel
  ```
- (Optional) A `.clap` plugin on disk if you want to test CLAP effect hosting — none ship with the repo. On Fedora, `clap-zam-plugins` is a convenient one to try:
  ```bash
  sudo dnf install clap-zam-plugins
  ```

## Using the piano roll

Each melodic track's pattern is edited on a free-form canvas: pitch on the vertical axis, time on the horizontal axis.

- **Click empty space** — drop a quick note at the default length (set with the "Length" buttons on the left).
- **Click-drag from empty space** — draw a new note out to any length.
- **Drag a note's body** — move it (both pitch and time).
- **Drag a note's right edge** — resize it (hover near the edge for the resize cursor).
- **Click a note** — select only that note, replacing whatever was selected before.
- **Ctrl/Cmd-click a note** — toggle it into or out of the current selection without touching the rest.
- **Shift-drag from empty space** — rubber-band a rectangle to select every note it touches.
- **Drag a note that's part of a multi-note selection** — move the whole selection together, preserving each note's position relative to the others.
- **Right-click a note** — delete it; if it's part of a multi-note selection, this deletes the whole selection instead.
- **Delete / Backspace** — delete the current selection.

The velocity of each note is set via the velocity lane below the grid: drag a note's bar up or down. The roll grows automatically as notes reach its right edge, and the "Zoom" / "Roll height" controls in the top toolbar resize every track's piano roll at once.

### Build & run

```bash
cargo build       # debug build
cargo run         # launch the GUI app
cargo run --bin simple-daw
cargo run --bin simple-daw-mcp
```

### Tests

```bash
cargo test              # unit tests: audio DSP math, model editing logic, WAV export
cargo test <test_name>  # run a single test
```

Unit tests cover synth/DSP math and the WAV exporter, but can't prove the live audio path actually produces sound — there's no automated harness for that. Audio changes should be verified manually by running the app.

## Architecture

- `src/main.rs` — egui/eframe UI. Owns the `Song` behind an `Arc<Mutex<Song>>`, shared with the audio thread. Also has the File menu (Load/Save/Save As/Export), the channel rack, piano roll and step-grid canvases, and the per-engine/per-effect parameter windows.
- `src/model.rs` — pure data model: `Song` → `Track` → `Region` → `RegionContent` (`Lane` steps or `Note`s). No audio, no UI. Also owns JSON save/load (`serde`/`serde_json`).
- `src/factory_presets.rs` — the built-in `SynthPreset` catalog shipped with the app, several patches per synth engine.
- `src/audio.rs` — the real-time engine: cpal stream setup, the step/tick clock and per-track synth voice pools (one per engine, plus a sample-playback pool), master-bus/per-track/per-send/per-submix effect chain processing (CLAP and built-in), submix output routing and mute/solo, region fade/automation evaluation, Session View clip triggering, and the offline WAV exporter.
- `src/session.rs` — pure Session View clip-slot state machine, launch-quantization, and follow-action resolution logic (no audio, no UI).
- `src/session_view_ui.rs` — the Session View clip-launching grid UI: launch/stop, scene buttons, and the Legato/Follow Action controls.
- `src/builtin_fx/` — DSP for the built-in (non-CLAP) effects, one file per effect: delay, bitcrusher, distortion, reverb, chorus, filter, tremolo, compressor, flanger, phaser, ring modulator, noise gate, phase invert, channel EQ, limiter.
- `src/plugin_host.rs` — CLAP plugin hosting (loading, activating, querying audio-port channel counts and plugin parameters, running audio through a loaded effect).
- `src/gui_embed/` — per-platform (Cocoa/Win32/X11) native embedding of a CLAP plugin's own GUI into a host window.
- `src/metering.rs` — peak/RMS/LUFS metering for the Mixer's channel strips.
- `src/sample.rs` — WAV decoding and resampling for one-shot sample playback.
- `src/wavetable.rs` — wavetable data and sampling for the `Wave` synth engine.
- `src/midi_import.rs` — standard MIDI file (`.mid`) import into piano-roll notes.
- `src/audio_input.rs` — live audio input capture for Audio-track recording.
- `src/groove.rs` — quantize/humanize/groove-template transforms for piano-roll notes and step-grid lanes.
- `src/tempo.rs` — tap-tempo BPM averaging.
- `src/tempo_detection.rs` — estimates a WAV file's BPM from its audio content.
- `src/transient_detection.rs` — attack/onset detection and silence-gate segmentation, behind a clip's transient markers and "Strip Silence".
- `src/stretch.rs` — WSOLA time-stretch, behind Flex Time.
- `src/pitch.rs` — pitch detection and pitch-shifting, behind Flex Pitch.
- `src/mcp_control.rs` / `src/bin/simple-daw-mcp.rs` — optional (Unix only) MCP server letting an LLM drive a running instance.

## License

GNU GENERAL PUBLIC LICENSE Version 3

