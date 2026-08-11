# simple-daw

A native Rust MIDI sequencer / groovebox for composing video game music — not a general-purpose DAW.

Scope is deliberately narrow: step-sequencer + piano-roll MIDI editing, three built-in synth engines, sample playback, a Playlist/arrangement timeline, WAV bounce, MIDI import, JSON song persistence, and CLAP effect hosting.

## Features

- **Step sequencer** — drum-machine style grid (`Lane`s with per-step velocity). Each lane can optionally be backed by a loaded WAV sample instead of the synth, or given its own synth engine/patch overriding the track's (via the 🎹 button on the lane row) — so one track can mix a kick patch on one lane with a hi-hat patch on another. A loaded sample still takes priority over any synth.
- **Piano roll** — free-form melodic editing on a custom-painted canvas: click-drag to draw notes of any length, drag to move/resize, multi-select (Ctrl/Cmd-click, Shift-drag box select) to move or delete several notes at once, with a per-note velocity lane. See [Using the piano roll](#using-the-piano-roll) below.
- **Three synth engines, selectable per track** — `Simple` (the original sine-family oscillator + exponential-decay envelope, percussive only), `Trine` (multi-oscillator with a filter/mod matrix), and `Wave` (two wavetable oscillators with position-morph and phase-warp, on the same filter/mod-matrix machinery as `Trine`). A library of built-in factory presets ships for all three.
- **Playlist / arrangement** — each track owns its own independently positioned `Region`s (step-grid or piano-roll content, Logic/Ableton-style — not a shared FL Studio–style pattern), dragged and resized along a shared song timeline.
- **Sample playback** — WAV one-shots via a per-track 32-voice sample player pool, resampled to the output device's rate at load time.
- **Audio tracks** — record live input straight to a WAV file and drop it on the timeline as a clip, or import an existing WAV the same way.
- **MIDI import** — load a standard `.mid` file into a track's piano roll.
- **Song persistence** — save/load the whole song (tracks, regions, sample paths, loaded effect paths and parameter values) to/from a JSON file via the File menu.
- **WAV export** — bounce the song to a file for a configurable number of loops; export uses the exact same sequencing/mixing code as live playback (dry only — CLAP effects aren't included in the bounce).
- **CLAP effect hosting** — load a CLAP plugin as a master-bus insert effect *and* as a per-track insert effect, with host-side parameter editing (a generic "Params" window driven by the plugin's declared CLAP parameters). No plugin GUI, no host-side automation, and no unload mid-session once a plugin is loaded.

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
- `src/audio.rs` — the real-time engine: cpal stream setup, the step/tick clock and per-track synth voice pools (one per engine, plus a sample-playback pool), CLAP master- and per-track-effect integration, and the offline WAV exporter.
- `src/builtin_fx.rs` — DSP for the built-in (non-CLAP) effects: delay, bitcrusher, distortion, reverb, chorus, filter, tremolo, compressor, flanger, phaser, ring modulator, noise gate.
- `src/plugin_host.rs` — CLAP plugin hosting (loading, activating, querying audio-port channel counts and plugin parameters, running audio through a loaded effect).
- `src/sample.rs` — WAV decoding and resampling for one-shot sample playback.
- `src/wavetable.rs` — wavetable data and sampling for the `Wave` synth engine.
- `src/midi_import.rs` — standard MIDI file (`.mid`) import into piano-roll notes.

## License

No license file yet — all rights reserved by default until one is added.
