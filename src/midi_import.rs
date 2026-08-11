//! Standard MIDI File (.mid) import: turns note events from an external MIDI file into new
//! piano-roll tracks the app can append to the current `Song`. Pure data conversion — no audio,
//! no UI, following the same "one job" split as `sample.rs`.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use midly::{MetaMessage, MidiMessage, Smf, Timing, TrackEventKind};

use crate::model::{Note, TICKS_PER_STEP, grow_length_to_fit_notes};

/// 4 steps/beat (16th-note steps) * `TICKS_PER_STEP`, matching the doc comment on
/// `TICKS_PER_STEP` ("24 ticks/step = 96 ticks/beat"). MIDI tick positions are rescaled into this
/// space using the file's own ticks-per-beat (PPQ), so imported notes land on the same musical
/// beats regardless of the source file's resolution.
const OUR_TICKS_PER_BEAT: f64 = (TICKS_PER_STEP * 4) as f64;

pub struct ImportedMidi {
    pub tracks: Vec<ImportedTrack>,
    /// BPM from the file's first "Set Tempo" meta event, if any. Only the first is used — `Song`
    /// has one global `bpm`, so mid-file tempo changes aren't representable and are ignored
    /// (a deliberate scope cut, not a bug).
    pub detected_bpm: Option<f32>,
}

/// One imported MIDI track's identity and note content — not a `model::Track` directly, since a
/// `Track` no longer owns its own pattern content (patterns are global, see `model::Pattern`).
/// The caller (`main.rs`'s Import MIDI flow) creates the `Track` via `Song::add_track` and drops
/// `notes`/`length_steps` into the song's first pattern for that track's new slot.
pub struct ImportedTrack {
    pub name: String,
    pub midi_channel: u8,
    pub notes: Vec<Note>,
    pub length_steps: usize,
}

struct MidiGroup {
    channel: u8,
    name: Option<String>,
    notes: Vec<Note>,
}

/// Parses a `.mid` file and returns one new piano-roll `Track` per (MIDI track, channel) pair
/// that contains at least one note — this covers both format-1 files (one instrument per track)
/// and format-0 files (multiple channels interleaved in a single track) without special-casing
/// either. Percussion channels aren't treated specially: every note becomes a piano-roll `Note`
/// at its MIDI key, since this app has no fixed drum-map concept for imported material.
///
/// `next_note_id` is threaded through and advanced exactly like `model::add_note` does, so
/// imported notes get ids that don't collide with the rest of the song. `steps_per_bar` (see
/// `Song::steps_per_bar`) sizes each imported track's default region length.
pub fn import_midi_file(
    path: &Path,
    next_note_id: &mut u64,
    steps_per_bar: usize,
) -> Result<ImportedMidi> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let smf = Smf::parse(&bytes).map_err(|err| anyhow!("not a valid MIDI file: {err}"))?;

    let midi_ppq = match smf.header.timing {
        Timing::Metrical(ticks_per_beat) => ticks_per_beat.as_int() as f64,
        Timing::Timecode(fps, subframe) => {
            return Err(anyhow!(
                "this file uses SMPTE timecode timing ({} fps, {} subframes/frame), which isn't \
                 supported — only ticks-per-beat MIDI files can be imported",
                fps.as_f32(),
                subframe
            ));
        }
    };
    let tick_scale = OUR_TICKS_PER_BEAT / midi_ppq;

    let mut detected_bpm = None;
    let mut groups: Vec<MidiGroup> = Vec::new();
    let mut group_index: HashMap<(usize, u8), usize> = HashMap::new();

    for (track_idx, track) in smf.tracks.iter().enumerate() {
        let mut abs_tick: u64 = 0;
        // (channel, key) -> (start_tick, velocity), tracking notes currently sounding so a
        // matching NoteOff (or zero-velocity NoteOn, its documented equivalent) can close them.
        let mut open_notes: HashMap<(u8, u8), (u64, u8)> = HashMap::new();
        let mut track_name: Option<String> = None;

        for event in track.iter() {
            abs_tick += event.delta.as_int() as u64;
            match event.kind {
                TrackEventKind::Meta(MetaMessage::Tempo(micros_per_beat)) => {
                    if detected_bpm.is_none() {
                        let micros = micros_per_beat.as_int() as f64;
                        if micros > 0.0 {
                            detected_bpm = Some((60_000_000.0 / micros) as f32);
                        }
                    }
                }
                TrackEventKind::Meta(MetaMessage::TrackName(name)) if track_name.is_none() => {
                    let name = String::from_utf8_lossy(name).trim().to_string();
                    if !name.is_empty() {
                        track_name = Some(name);
                    }
                }
                TrackEventKind::Midi { channel, message } => {
                    let channel = channel.as_int();
                    let closed = match message {
                        MidiMessage::NoteOn { key, vel } if vel.as_int() > 0 => {
                            open_notes.insert((channel, key.as_int()), (abs_tick, vel.as_int()));
                            None
                        }
                        MidiMessage::NoteOn { key, .. } | MidiMessage::NoteOff { key, .. } => {
                            open_notes
                                .remove(&(channel, key.as_int()))
                                .map(|(start, vel)| (key.as_int(), start, vel))
                        }
                        _ => None,
                    };
                    if let Some((pitch, start_tick, velocity)) = closed {
                        let idx = *group_index.entry((track_idx, channel)).or_insert_with(|| {
                            groups.push(MidiGroup {
                                channel,
                                name: None,
                                notes: Vec::new(),
                            });
                            groups.len() - 1
                        });
                        let start = (start_tick as f64 * tick_scale).round() as usize;
                        let end = (abs_tick as f64 * tick_scale).round() as usize;
                        let id = *next_note_id;
                        *next_note_id += 1;
                        groups[idx].notes.push(Note {
                            id,
                            pitch,
                            start_tick: start,
                            length_ticks: end.saturating_sub(start).max(1),
                            velocity: velocity.min(127),
                        });
                    }
                }
                _ => {}
            }
        }

        if let Some(name) = track_name {
            for (_, &idx) in group_index.iter().filter(|((t, _), _)| *t == track_idx) {
                groups[idx].name.get_or_insert_with(|| name.clone());
            }
        }
    }

    if groups.is_empty() {
        return Err(anyhow!("no note events found in this MIDI file"));
    }

    let tracks = groups
        .into_iter()
        .enumerate()
        .map(|(i, group)| {
            let mut notes = group.notes;
            notes.sort_by_key(|n| n.start_tick);
            let mut length_steps = steps_per_bar;
            grow_length_to_fit_notes(&mut length_steps, &notes, steps_per_bar);
            let name = group
                .name
                .unwrap_or_else(|| format!("MIDI Import {}", i + 1));
            let midi_channel = group.channel + 1;
            ImportedTrack {
                name,
                midi_channel,
                notes,
                length_steps,
            }
        })
        .collect();

    Ok(ImportedMidi {
        tracks,
        detected_bpm,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vlq(mut n: u32) -> Vec<u8> {
        let mut out = vec![(n & 0x7f) as u8];
        n >>= 7;
        while n > 0 {
            out.insert(0, ((n & 0x7f) as u8) | 0x80);
            n >>= 7;
        }
        out
    }

    fn track_chunk(events: &[u8]) -> Vec<u8> {
        let mut chunk = b"MTrk".to_vec();
        chunk.extend_from_slice(&(events.len() as u32).to_be_bytes());
        chunk.extend_from_slice(events);
        chunk
    }

    /// Builds a minimal format-1, 480-ticks/beat SMF: one tempo-only track (120 BPM) and one note
    /// track (channel 0, named "Lead") with two consecutive one-beat notes (C4 then E4).
    fn sample_midi_bytes() -> Vec<u8> {
        let mut header = b"MThd".to_vec();
        header.extend_from_slice(&6u32.to_be_bytes());
        header.extend_from_slice(&1u16.to_be_bytes()); // format 1
        header.extend_from_slice(&2u16.to_be_bytes()); // 2 tracks
        header.extend_from_slice(&480u16.to_be_bytes()); // ticks/beat

        let mut tempo_events = Vec::new();
        tempo_events.extend(vlq(0));
        tempo_events.extend([0xFF, 0x51, 0x03]);
        tempo_events.extend(&500_000u32.to_be_bytes()[1..]); // 500000 us/beat = 120 BPM
        tempo_events.extend(vlq(0));
        tempo_events.extend([0xFF, 0x2F, 0x00]);
        let tempo_track = track_chunk(&tempo_events);

        let mut note_events = Vec::new();
        note_events.extend(vlq(0));
        note_events.extend([0xFF, 0x03, 4]);
        note_events.extend(b"Lead");
        note_events.extend(vlq(0));
        note_events.extend([0x90, 60, 100]); // note on C4
        note_events.extend(vlq(480));
        note_events.extend([0x80, 60, 0]); // note off C4, one beat later
        note_events.extend(vlq(0));
        note_events.extend([0x90, 64, 90]); // note on E4, same tick
        note_events.extend(vlq(480));
        note_events.extend([0x80, 64, 0]); // note off E4, one beat later
        note_events.extend(vlq(0));
        note_events.extend([0xFF, 0x2F, 0x00]);
        let note_track = track_chunk(&note_events);

        [header, tempo_track, note_track].concat()
    }

    fn write_temp_midi(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(name);
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn import_converts_ticks_and_detects_tempo_and_track_name() {
        let path = write_temp_midi("simple_daw_import_test_basic.mid", &sample_midi_bytes());
        let mut next_id = 100;

        let imported = import_midi_file(&path, &mut next_id, 16).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(imported.detected_bpm, Some(120.0));
        assert_eq!(imported.tracks.len(), 1);

        let track = &imported.tracks[0];
        assert_eq!(track.name, "Lead");
        assert_eq!(track.midi_channel, 1);
        let notes = &track.notes;
        assert_eq!(notes.len(), 2);

        // 480 midi ticks/beat -> 96 of our ticks/beat, so one beat = 96 ticks and scale = 0.2.
        assert_eq!(notes[0].pitch, 60);
        assert_eq!(notes[0].start_tick, 0);
        assert_eq!(notes[0].length_ticks, 96);
        assert_eq!(notes[0].velocity, 100);
        assert_eq!(notes[0].id, 100);

        assert_eq!(notes[1].pitch, 64);
        assert_eq!(notes[1].start_tick, 96);
        assert_eq!(notes[1].length_ticks, 96);
        assert_eq!(notes[1].velocity, 90);
        assert_eq!(notes[1].id, 101);

        assert_eq!(next_id, 102);
    }

    #[test]
    fn import_rejects_a_file_with_no_notes() {
        let mut header = b"MThd".to_vec();
        header.extend_from_slice(&6u32.to_be_bytes());
        header.extend_from_slice(&1u16.to_be_bytes());
        header.extend_from_slice(&1u16.to_be_bytes());
        header.extend_from_slice(&480u16.to_be_bytes());
        let mut events = Vec::new();
        events.extend(vlq(0));
        events.extend([0xFF, 0x2F, 0x00]);
        let bytes = [header, track_chunk(&events)].concat();

        let path = write_temp_midi("simple_daw_import_test_empty.mid", &bytes);
        let mut next_id = 0;
        let result = import_midi_file(&path, &mut next_id, 16);
        std::fs::remove_file(&path).ok();

        assert!(result.is_err());
    }
}
