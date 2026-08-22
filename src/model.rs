use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::pitch::{self, PitchCorrection};
use crate::sample::SampleBuffer;
use crate::stretch::{self, WarpMarker};
use crate::wavetable::{WaveWarpMode, WavetableId};

/// One active step-grid trigger: a velocity plus a small timing nudge off the step's own grid
/// position, for groove templates/humanize (see `crate::groove`). `timing_offset_ticks` is
/// expected to stay within `-(TICKS_PER_STEP / 2 - 1)..=(TICKS_PER_STEP / 2 - 1)` (kept in range
/// by every setter — `Lane::set_step`, `Lane::set_step_timing_offset`, and `groove`'s step
/// functions — rather than enforced here) so a nudged step's trigger tick can never cross into a
/// neighboring step's own territory; `Sequencer::process`'s trigger scan (`audio::step_triggering_at`)
/// only looks at the two step boundaries nearest a given tick, relying on that invariant.
///
/// Deserializes from either this struct's own shape or a bare `u8` (`#[serde(from = "StepDataRepr")]`
/// below) so song files saved before timing offsets existed — where a step was just a velocity
/// byte — still load, with `timing_offset_ticks` defaulting to 0.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(from = "StepDataRepr")]
pub struct StepData {
    pub velocity: u8,
    pub timing_offset_ticks: i8,
}

/// Deserialization-only shape `StepData` converts from — see `StepData`'s doc comment.
#[derive(Deserialize)]
#[serde(untagged)]
enum StepDataRepr {
    /// A song file saved before timing offsets existed: a step was just a velocity byte.
    LegacyVelocity(u8),
    Full {
        velocity: u8,
        #[serde(default)]
        timing_offset_ticks: i8,
    },
}

impl From<StepDataRepr> for StepData {
    fn from(repr: StepDataRepr) -> Self {
        match repr {
            StepDataRepr::LegacyVelocity(velocity) => StepData { velocity, timing_offset_ticks: 0 },
            StepDataRepr::Full { velocity, timing_offset_ticks } => {
                StepData { velocity, timing_offset_ticks }
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Lane {
    pub name: String,
    pub pitch: u8,
    pub steps: Vec<Option<StepData>>,
    /// When set, triggering this lane plays the sample instead of the built-in synth.
    /// Not serialized — it's decoded audio, not song data; reloaded from `sample_path`
    /// after deserializing (see `Song::load_from_file`).
    #[serde(skip)]
    pub sample: Option<Arc<SampleBuffer>>,
    /// File path the user typed/loaded, kept around so the field survives even if loading failed.
    pub sample_path: String,
    #[serde(skip)]
    pub sample_error: Option<String>,
    /// When true, this lane renders with its own `synth_engine`/`synth`/`trine`/`wave` below
    /// instead of the track's — lets a step-grid track mix synth patches per lane (e.g. a kick
    /// patch on one lane, a hi-hat patch on another). `sample`, above, still takes priority over
    /// any synth (default or overridden) when triggering — unchanged from before this field
    /// existed. `#[serde(default)]` so song files saved before this field existed still load,
    /// defaulting every lane to the track's own synth (unchanged behavior).
    #[serde(default)]
    pub synth_override: bool,
    /// Mirrors `Track::synth_engine`, only used when `synth_override` is true.
    #[serde(default)]
    pub synth_engine: SynthEngine,
    /// Mirrors `Track::synth`, only used when `synth_override` is true and `synth_engine ==
    /// SynthEngine::Simple`.
    #[serde(default)]
    pub synth: SynthParams,
    /// Mirrors `Track::trine`, only used when `synth_override` is true and `synth_engine ==
    /// SynthEngine::Trine`.
    #[serde(default)]
    pub trine: TrineParams,
    /// Mirrors `Track::wave`, only used when `synth_override` is true and `synth_engine ==
    /// SynthEngine::Wave`.
    #[serde(default)]
    pub wave: WaveParams,
}

impl Lane {
    /// A new lane with `length_steps` empty steps and no sample/synth override.
    pub fn new(name: impl Into<String>, pitch: u8, length_steps: usize) -> Self {
        Self {
            name: name.into(),
            pitch,
            steps: vec![None; length_steps],
            sample: None,
            sample_path: String::new(),
            sample_error: None,
            synth_override: false,
            synth_engine: SynthEngine::default(),
            synth: SynthParams::default(),
            trine: TrineParams::default(),
            wave: WaveParams::default(),
        }
    }

    /// Sets step `index` to trigger at `velocity`, preserving that step's existing timing offset
    /// (if it was already active) rather than resetting it to straight.
    pub fn set_step(&mut self, index: usize, velocity: u8) {
        let timing_offset_ticks = self.steps[index].map_or(0, |step| step.timing_offset_ticks);
        self.steps[index] = Some(StepData { velocity, timing_offset_ticks });
    }


    /// Loads `sample_path`, resampled to `target_sample_rate`. On failure, falls back
    /// to the synth (clears `sample`) and records the error for the UI to display.
    pub fn load_sample(&mut self, target_sample_rate: u32) {
        let path = self.sample_path.trim();
        if path.is_empty() {
            self.sample = None;
            self.sample_error = None;
            return;
        }
        match SampleBuffer::load_wav_resampled(std::path::Path::new(path), target_sample_rate) {
            Ok(buffer) => {
                self.sample = Some(Arc::new(buffer));
                self.sample_error = None;
            }
            Err(err) => {
                self.sample = None;
                self.sample_error = Some(format!("{err:#}"));
            }
        }
    }

    /// Clears the loaded sample (if any) and any load error, reverting the lane to its synth.
    pub fn clear_sample(&mut self) {
        self.sample = None;
        self.sample_error = None;
    }
}

/// Piano-roll notes are positioned in ticks, finer than the step-grid's
/// 16th-note steps, so they can be dragged to (almost) any position/length
/// instead of snapping to one of 16 cells. 24 ticks/step = 96 ticks/beat,
/// a common MIDI PPQ-style resolution: fine enough that placement feels
/// free-form while staying exact integer arithmetic for the sequencer.
pub const TICKS_PER_STEP: usize = 24;

/// Largest magnitude a `StepData::timing_offset_ticks` may hold — half a step, minus one tick, so
/// a maximally-nudged step's trigger tick can never land exactly on (or past) a neighboring step's
/// own grid position. See `StepData`'s doc comment.
pub const MAX_STEP_TIMING_OFFSET_TICKS: i8 = (TICKS_PER_STEP / 2 - 1) as i8;

/// A melodic note in a piano-roll pattern: unlike a step-grid `Lane`, pitch
/// varies per note rather than being fixed per row, and position/length are
/// free (in ticks) rather than locked to a step. `id` is a stable per-song
/// identifier (not derived from Vec position) so the UI can track a note
/// across frames while it's being dragged, even as other notes are added or
/// removed around it.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Note {
    pub id: u64,
    pub pitch: u8,
    pub start_tick: usize,
    pub length_ticks: usize,
    pub velocity: u8,
}

/// Oscillator shape for a track's built-in synth voice (see `audio::Voice`).
/// Pure data — the actual waveform generation lives in `audio.rs`, which is
/// the only module allowed to know about DSP.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SynthWaveform {
    #[default]
    Sine,
    Saw,
    Square,
    Triangle,
    /// White noise — no meaningful pitch/phase, just a broadband texture. Currently only exposed
    /// in the Trine engine's oscillator/LFO pickers (see `TrineParams`); Simple Synth's UI doesn't
    /// offer it, though `audio::Voice` synthesizes it correctly if a saved file ever sets it.
    Noise,
}

/// Which tap of the per-voice resonant filter is sent to the output. The filter is a Zavalishin
/// TPT state-variable filter, which computes lowpass and bandpass simultaneously each sample —
/// highpass/notch are just different (cheap) combinations of those two, not a different filter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum FilterType {
    #[default]
    Lowpass,
    Highpass,
    Bandpass,
    Notch,
}

/// What the LFO modulates, if anything. Only one target is active at a time — matches the
/// existing single-target `filter_env_amount_hz` pattern already used by the filter envelope,
/// which keeps the UI/mental model simple instead of a combinatorial modulation matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum LfoTarget {
    #[default]
    None,
    Pitch,
    Amplitude,
    FilterCutoff,
}

/// Per-track synth voice settings: oscillator shape (+ detune/unison thickening), a real
/// attack/decay/sustain/release envelope, and a resonant filter (switchable type) with its own
/// decay envelope, plus a second oscillator, a sub-oscillator, an LFO, and glide/portamento.
/// Unlike a live instrument, this sequencer never gets an explicit "note off" event — every
/// trigger's total gate time (how long it stays "held" before Release begins) is known up front,
/// computed by the caller in `audio.rs::Sequencer::process` from the note's length (piano roll)
/// or `attack_seconds + decay_seconds` (step-grid hits, which have no natural length).
/// `#[serde(default)]` on the struct fills in any field missing from an older save file with this
/// type's `Default`, so songs saved by earlier versions of this feature still load.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SynthParams {
    pub waveform: SynthWaveform,
    /// Duty cycle of the Square waveform (0.5 = classic 50/50 square); ignored by other waveforms.
    pub pulse_width: f32,
    /// How many detuned copies of the oscillator to stack per note (1 = no detuning). Copies are
    /// spread symmetrically around the true pitch by `unison_detune_cents`.
    pub unison_voices: u8,
    pub unison_detune_cents: f32,
    /// How far unison voices spread across the stereo field, 0.0..1.0. 0.0 (default) keeps every
    /// unison voice centered — identical to this engine's pre-stereo behavior, so existing songs
    /// sound unchanged until this is raised. Only affects sound when `unison_voices > 1`; see
    /// `audio::Voice::next_sample`.
    pub unison_width: f32,

    pub attack_seconds: f32,
    pub decay_seconds: f32,
    /// Level (relative to the note's velocity-scaled peak) the envelope decays to and holds at
    /// until the gate closes. 0.0 reproduces the original always-decays-to-silence behavior.
    pub sustain_level: f32,
    pub release_seconds: f32,

    /// Base cutoff of the per-voice filter. Defaults high enough to be inaudible under the
    /// default `Lowpass` type, so a track with untouched filter settings sounds like the filter
    /// isn't there.
    pub filter_cutoff_hz: f32,
    pub filter_resonance: f32,
    /// Extra cutoff (Hz, can be negative) swept in at the note's start and decaying to 0 over the
    /// same `decay_seconds` window as the amplitude envelope's decay stage — independent of
    /// `sustain_level`, so the filter can keep closing even while the amplitude holds a sustain.
    pub filter_env_amount_hz: f32,
    pub filter_type: FilterType,

    /// A second oscillator, crossfaded against the first: 0.0 = osc1 only (default, unchanged
    /// sound), 1.0 = osc2 only. Detuned by `osc2_semitones` + `osc2_detune_cents` relative to the
    /// note's pitch. Does not get its own unison stacking (unison applies to osc1 only).
    pub osc2_waveform: SynthWaveform,
    pub osc2_semitones: i32,
    pub osc2_detune_cents: f32,
    pub osc2_mix: f32,
    /// Hard sync: resets osc2's phase to 0 every time osc1 (unison voice 0) completes a cycle.
    /// When osc2 is tuned away from osc1 this truncates its waveform mid-cycle, locking it to
    /// osc1's pitch and producing the bright, buzzy "sync" timbre classic analog synths are known
    /// for. False reproduces the original free-running behavior.
    pub osc2_sync: bool,
    /// A fixed sine oscillator one octave below the note, mixed in additively (not crossfaded)
    /// for extra low-end weight. 0.0 = off (default).
    pub sub_osc_mix: f32,

    /// Low-frequency oscillator; only affects sound once `lfo_target != LfoTarget::None`.
    pub lfo_waveform: SynthWaveform,
    pub lfo_rate_hz: f32,
    pub lfo_target: LfoTarget,
    /// 0.0..1.0. Meaning depends on `lfo_target`: max +/-1 semitone for Pitch, tremolo depth
    /// (1.0 = full on/off) for Amplitude, or a fraction of a fixed Hz sweep range for FilterCutoff.
    pub lfo_depth: f32,

    /// Seconds to glide from the previously played pitch to a new one, in log-frequency space
    /// (musically linear, not a linear-Hz sweep). 0.0 = instant, as before. Only applied to
    /// piano-roll notes (see `Sequencer::process`) — step-grid hits always trigger at pitch
    /// immediately, since portamento between unrelated drum hits doesn't make sense.
    pub glide_seconds: f32,
}

impl Default for SynthParams {
    fn default() -> Self {
        Self {
            waveform: SynthWaveform::Sine,
            pulse_width: 0.5,
            unison_voices: 1,
            unison_detune_cents: 12.0,
            unison_width: 0.0,
            attack_seconds: 0.0,
            decay_seconds: 0.25,
            sustain_level: 0.0,
            release_seconds: 0.05,
            filter_cutoff_hz: 20_000.0,
            filter_resonance: 0.707,
            filter_env_amount_hz: 0.0,
            filter_type: FilterType::Lowpass,
            osc2_waveform: SynthWaveform::Sine,
            osc2_semitones: 0,
            osc2_detune_cents: 0.0,
            osc2_mix: 0.0,
            osc2_sync: false,
            sub_osc_mix: 0.0,
            lfo_waveform: SynthWaveform::Sine,
            lfo_rate_hz: 5.0,
            lfo_target: LfoTarget::None,
            lfo_depth: 0.0,
            glide_seconds: 0.0,
        }
    }
}

/// Which synth engine a track's notes are rendered with — see `Track::synth_engine`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SynthEngine {
    /// The original built-in synth (`SynthParams`/`audio::Voice`).
    #[default]
    Simple,
    /// The second, more elaborate engine (`TrineParams`/`audio::TrineVoice`).
    Trine,
    /// The third engine (`WaveParams`/`audio::WaveVoice`): two wavetable oscillators with
    /// position-morphing and phase-warp, on top of the same dual-filter/mod-matrix machinery
    /// `Trine` uses.
    Wave,
}

/// How many poles a `TrineParams` filter has: one TPT SVF stage (12dB/octave, matching Simple
/// Synth's single-stage filter) or two cascaded stages (24dB/octave) — no new filter topology,
/// just running the existing stage once or twice.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum FilterSlope {
    #[default]
    Slope12,
    Slope24,
}

/// How `TrineParams`'s two filters combine. `Off` (default) means filter2 isn't in the signal path
/// at all, so an untouched track sounds identical to using filter1 alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum FilterRouting {
    #[default]
    Off,
    /// filter1's output feeds into filter2.
    Series,
    /// Both filters process the same input independently; their outputs are summed.
    Parallel,
}

/// A modulation source in `TrineParams`'s routing matrix — see `ModSlot`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ModSource {
    #[default]
    None,
    Lfo1,
    Lfo2,
    /// Free-running envelope, only audible once routed here — see `TrineParams::env1_attack_seconds`.
    Env1,
    /// Free-running envelope, only audible once routed here — see `TrineParams::env2_attack_seconds`.
    Env2,
    /// The note's velocity (0..1), fixed for the note's whole duration.
    Velocity,
}

/// A modulation target in `TrineParams`'s routing matrix — see `ModSlot`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ModTarget {
    #[default]
    None,
    Pitch,
    Osc1Level,
    Osc2Level,
    Osc3Level,
    PulseWidth,
    FilterCutoff,
    Filter2Cutoff,
    FilterResonance,
    FmAmount,
    RingModMix,
}

/// One routing in `TrineParams::mod_slots`: `source`'s current value (roughly -1.0..=1.0, or
/// 0.0..=1.0 for `ModSource::Velocity`), scaled by bipolar `amount`, is added to `target` each
/// sample. A slot with `target: ModTarget::None` (the default for a freshly-added row) is a no-op.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModSlot {
    pub source: ModSource,
    pub target: ModTarget,
    pub amount: f32,
}

impl Default for ModSlot {
    fn default() -> Self {
        Self {
            source: ModSource::None,
            target: ModTarget::None,
            amount: 0.0,
        }
    }
}

/// The second, independent synth engine ("Trine" in the UI) a track can opt into instead of the
/// default Simple Synth (`SynthParams`) — see `Track::synth_engine`. Three oscillators (with FM,
/// ring mod, and per-voice analog drift), a dual filter (series/parallel routable, switchable
/// slope), and a free modulation matrix (2 LFOs + 2 free envelopes + velocity, routable to any of
/// several targets) sit on top of an always-on third envelope driving amplitude — mirroring
/// Logic's ES2, where "ENV 3" is hardwired to volume while envelopes 1/2 are freely assignable via
/// its router. `#[serde(default)]` so song files saved before this engine existed still load, with
/// every track defaulting to `SynthEngine::Simple` (this struct's values are never read unless a
/// track opts in).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TrineParams {
    pub osc1_waveform: SynthWaveform,
    pub osc1_level: f32,
    /// Duty cycle used by any of the three oscillators currently set to `SynthWaveform::Square`
    /// (shared, like `SynthParams::pulse_width`).
    pub pulse_width: f32,

    pub osc2_waveform: SynthWaveform,
    pub osc2_semitones: i32,
    pub osc2_detune_cents: f32,
    pub osc2_level: f32,
    /// Hard sync: resets osc2's phase to 0 every time osc1 completes a cycle.
    pub osc2_sync: bool,

    pub osc3_waveform: SynthWaveform,
    pub osc3_semitones: i32,
    pub osc3_detune_cents: f32,
    pub osc3_level: f32,
    /// Hard sync: resets osc3's phase to 0 every time osc1 completes a cycle.
    pub osc3_sync: bool,

    /// Classic 2-op FM: osc2's raw output frequency-modulates osc1, independent of `osc2_level`
    /// so FM doesn't require osc2 to also be audible in the mix.
    pub fm_amount: f32,
    /// osc1 x osc2 sample product, mixed additively into the oscillator sum.
    pub ring_mod_mix: f32,
    /// Depth of a slow per-voice random walk applied to every oscillator's pitch, emulating
    /// analog oscillator drift. 0.0 = perfectly stable (default).
    pub analog_drift: f32,

    pub filter1_cutoff_hz: f32,
    pub filter1_resonance: f32,
    pub filter1_type: FilterType,
    pub filter1_slope: FilterSlope,
    pub filter2_cutoff_hz: f32,
    pub filter2_resonance: f32,
    pub filter2_type: FilterType,
    pub filter2_slope: FilterSlope,
    pub filter_routing: FilterRouting,
    /// Tanh soft-clip applied to the oscillator sum before it enters filter1. 0.0 = bypassed.
    pub filter_drive: f32,
    /// filter1 cutoff modulated directly (audio-rate) by osc2's instantaneous sample, independent
    /// of `mod_slots` — mirrors Logic's ES2 and its own fixed filter-FM knob.
    pub filter_fm_amount: f32,

    pub lfo1_waveform: SynthWaveform,
    pub lfo1_rate_hz: f32,
    pub lfo2_waveform: SynthWaveform,
    pub lfo2_rate_hz: f32,

    /// Free-running envelope, only audible once routed through `mod_slots` — unlike `env3_*`
    /// below, nothing hardwires this to a target.
    pub env1_attack_seconds: f32,
    pub env1_decay_seconds: f32,
    pub env1_sustain_level: f32,
    pub env1_release_seconds: f32,
    /// Free-running envelope, only audible once routed through `mod_slots`.
    pub env2_attack_seconds: f32,
    pub env2_decay_seconds: f32,
    pub env2_sustain_level: f32,
    pub env2_release_seconds: f32,

    /// The always-on amplitude envelope (mirroring Logic's ES2 "ENV 3 Vol") — every Trine voice is
    /// shaped by this regardless of `mod_slots`, so a freshly-selected Trine track is immediately audible.
    pub env3_attack_seconds: f32,
    pub env3_decay_seconds: f32,
    pub env3_sustain_level: f32,
    pub env3_release_seconds: f32,

    /// Source -> target routings, each with a bipolar amount. Empty by default, so a fresh Trine
    /// track behaves like a plain 3-oscillator/dual-filter/no-modulation synth until the user
    /// wires something up.
    pub mod_slots: Vec<ModSlot>,
}

impl Default for TrineParams {
    fn default() -> Self {
        Self {
            osc1_waveform: SynthWaveform::Saw,
            osc1_level: 1.0,
            pulse_width: 0.5,
            osc2_waveform: SynthWaveform::Saw,
            osc2_semitones: 0,
            osc2_detune_cents: 0.0,
            osc2_level: 0.0,
            osc2_sync: false,
            osc3_waveform: SynthWaveform::Saw,
            osc3_semitones: 0,
            osc3_detune_cents: 0.0,
            osc3_level: 0.0,
            osc3_sync: false,
            fm_amount: 0.0,
            ring_mod_mix: 0.0,
            analog_drift: 0.0,
            filter1_cutoff_hz: 20_000.0,
            filter1_resonance: 0.707,
            filter1_type: FilterType::Lowpass,
            filter1_slope: FilterSlope::Slope12,
            filter2_cutoff_hz: 20_000.0,
            filter2_resonance: 0.707,
            filter2_type: FilterType::Lowpass,
            filter2_slope: FilterSlope::Slope12,
            filter_routing: FilterRouting::Off,
            filter_drive: 0.0,
            filter_fm_amount: 0.0,
            lfo1_waveform: SynthWaveform::Sine,
            lfo1_rate_hz: 5.0,
            lfo2_waveform: SynthWaveform::Sine,
            lfo2_rate_hz: 5.0,
            env1_attack_seconds: 0.0,
            env1_decay_seconds: 0.25,
            env1_sustain_level: 0.0,
            env1_release_seconds: 0.05,
            env2_attack_seconds: 0.0,
            env2_decay_seconds: 0.25,
            env2_sustain_level: 0.0,
            env2_release_seconds: 0.05,
            env3_attack_seconds: 0.0,
            env3_decay_seconds: 0.25,
            env3_sustain_level: 0.0,
            env3_release_seconds: 0.05,
            mod_slots: Vec::new(),
        }
    }
}

/// A modulation source in `WaveParams`'s routing matrix — a separate enum from `ModSource` rather
/// than shared variants, so a `Wave` track's matrix only ever offers sources that actually exist
/// on it (and likewise a `Trine` track never sees `Wave`-only targets like `Osc1Position`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum WaveModSource {
    #[default]
    None,
    Lfo1,
    Lfo2,
    /// Free-running envelope, only audible once routed here — see `WaveParams::env1_attack_seconds`.
    Env1,
    /// Free-running envelope, only audible once routed here — see `WaveParams::env2_attack_seconds`.
    Env2,
    /// The note's velocity (0..1), fixed for the note's whole duration.
    Velocity,
}

/// A modulation target in `WaveParams`'s routing matrix — see `WaveModSource`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum WaveModTarget {
    #[default]
    None,
    Pitch,
    /// Scans Oscillator 1 through its wavetable's frames — see `WaveParams::osc1_position`.
    Osc1Position,
    /// Scans Oscillator 2 through its wavetable's frames — see `WaveParams::osc2_position`.
    Osc2Position,
    Osc1WarpAmount,
    Osc2WarpAmount,
    FilterCutoff,
    Filter2Cutoff,
    FilterResonance,
}

/// One routing in `WaveParams::mod_slots` — see `ModSlot`, the equivalent for `TrineParams`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct WaveModSlot {
    pub source: WaveModSource,
    pub target: WaveModTarget,
    pub amount: f32,
}

impl Default for WaveModSlot {
    fn default() -> Self {
        Self {
            source: WaveModSource::None,
            target: WaveModTarget::None,
            amount: 0.0,
        }
    }
}

/// The third synth engine ("Wave" in the UI) a track can opt into — see `Track::synth_engine`.
/// Loosely inspired by Xfer Serum 2's wavetable-synthesis approach (two wavetable oscillators
/// scanning through a table's frames, each with a phase-warp mode) rather than `Trine`/ES2's
/// fixed-waveform-plus-FM approach, but built entirely from procedurally generated tables (see
/// `wavetable` module) and simplified phase-domain warp math, not a reproduction of Serum's actual
/// engine. Reuses `Trine`'s dual-filter (series/parallel routable, switchable slope) and
/// free-modulation-matrix (2 LFOs + 2 free envelopes + velocity) machinery, plus a sub-oscillator
/// and noise oscillator mixed in additively, and an always-on amplitude envelope (`amp_*`)
/// separate from the two free envelopes. `#[serde(default)]` so song files saved before this
/// engine existed still load, with every track defaulting to `SynthEngine::Simple` (this struct's
/// values are never read unless a track opts in).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WaveParams {
    pub osc1_table: WavetableId,
    /// Scans Oscillator 1 through its wavetable's frames, 0.0..=1.0.
    pub osc1_position: f32,
    pub osc1_warp_mode: WaveWarpMode,
    pub osc1_warp_amount: f32,
    pub osc1_level: f32,

    pub osc2_table: WavetableId,
    pub osc2_position: f32,
    pub osc2_warp_mode: WaveWarpMode,
    pub osc2_warp_amount: f32,
    pub osc2_semitones: i32,
    pub osc2_detune_cents: f32,
    pub osc2_level: f32,

    /// How many detuned copies of Oscillator 1 to stack per note (1 = no detuning) — same
    /// mechanism as `SynthParams::unison_voices`, applied to Oscillator 1 only.
    pub unison_voices: u8,
    pub unison_detune_cents: f32,
    /// Same mechanism as `SynthParams::unison_width` — see that field's doc comment.
    pub unison_width: f32,

    /// A sine sub-oscillator mixed in additively (not crossfaded), for extra low-end weight.
    pub sub_osc_level: f32,
    /// Semitones below the note's pitch for the sub-oscillator (typically -12, one octave down).
    pub sub_osc_semitones: i32,
    /// Broadband noise mixed in additively.
    pub noise_level: f32,

    pub filter1_cutoff_hz: f32,
    pub filter1_resonance: f32,
    pub filter1_type: FilterType,
    pub filter1_slope: FilterSlope,
    pub filter2_cutoff_hz: f32,
    pub filter2_resonance: f32,
    pub filter2_type: FilterType,
    pub filter2_slope: FilterSlope,
    pub filter_routing: FilterRouting,
    /// Tanh soft-clip applied to the oscillator sum before it enters filter1. 0.0 = bypassed.
    pub filter_drive: f32,

    pub lfo1_waveform: SynthWaveform,
    pub lfo1_rate_hz: f32,
    pub lfo2_waveform: SynthWaveform,
    pub lfo2_rate_hz: f32,

    /// Free-running envelope, only audible once routed through `mod_slots`.
    pub env1_attack_seconds: f32,
    pub env1_decay_seconds: f32,
    pub env1_sustain_level: f32,
    pub env1_release_seconds: f32,
    /// Free-running envelope, only audible once routed through `mod_slots`.
    pub env2_attack_seconds: f32,
    pub env2_decay_seconds: f32,
    pub env2_sustain_level: f32,
    pub env2_release_seconds: f32,

    /// The always-on amplitude envelope — every `Wave` voice is shaped by this regardless of
    /// `mod_slots`, so a freshly-selected Wave track is immediately audible.
    pub amp_attack_seconds: f32,
    pub amp_decay_seconds: f32,
    pub amp_sustain_level: f32,
    pub amp_release_seconds: f32,

    /// Source -> target routings, each with a bipolar amount. Empty by default, so a fresh Wave
    /// track behaves like a plain 2-oscillator/dual-filter/no-modulation synth until the user
    /// wires something up.
    pub mod_slots: Vec<WaveModSlot>,
}

impl Default for WaveParams {
    fn default() -> Self {
        Self {
            osc1_table: WavetableId::ClassicMorph,
            osc1_position: 0.0,
            osc1_warp_mode: WaveWarpMode::Off,
            osc1_warp_amount: 0.0,
            osc1_level: 1.0,
            osc2_table: WavetableId::ClassicMorph,
            osc2_position: 0.0,
            osc2_warp_mode: WaveWarpMode::Off,
            osc2_warp_amount: 0.0,
            osc2_semitones: 0,
            osc2_detune_cents: 0.0,
            osc2_level: 0.0,
            unison_voices: 1,
            unison_detune_cents: 12.0,
            unison_width: 0.0,
            sub_osc_level: 0.0,
            sub_osc_semitones: -12,
            noise_level: 0.0,
            filter1_cutoff_hz: 20_000.0,
            filter1_resonance: 0.707,
            filter1_type: FilterType::Lowpass,
            filter1_slope: FilterSlope::Slope12,
            filter2_cutoff_hz: 20_000.0,
            filter2_resonance: 0.707,
            filter2_type: FilterType::Lowpass,
            filter2_slope: FilterSlope::Slope12,
            filter_routing: FilterRouting::Off,
            filter_drive: 0.0,
            lfo1_waveform: SynthWaveform::Sine,
            lfo1_rate_hz: 5.0,
            lfo2_waveform: SynthWaveform::Sine,
            lfo2_rate_hz: 5.0,
            env1_attack_seconds: 0.0,
            env1_decay_seconds: 0.25,
            env1_sustain_level: 0.0,
            env1_release_seconds: 0.05,
            env2_attack_seconds: 0.0,
            env2_decay_seconds: 0.25,
            env2_sustain_level: 0.0,
            env2_release_seconds: 0.05,
            amp_attack_seconds: 0.0,
            amp_decay_seconds: 0.25,
            amp_sustain_level: 0.0,
            amp_release_seconds: 0.05,
            mod_slots: Vec::new(),
        }
    }
}

/// A region's content is either a drum-machine style grid (fixed pitch per row, one lane per
/// instrument) or a piano roll (pitch varies per note), matching the owning track's `TrackKind`.
/// `Audio`-kind tracks don't use regions at all — see `Track::audio_clips`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum RegionContent {
    StepGrid(Vec<Lane>),
    PianoRoll(Vec<Note>),
}

/// A track's own musical content, positioned independently on that track's row in the Playlist
/// (Logic/Ableton-style): unlike the FL Studio–style shared "Pattern" this replaced, a `Region`'s
/// content belongs to exactly one placement on one track, so placing similar material twice means
/// two independent copies, never two references to the same data.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Region {
    pub name: String,
    pub start_tick: usize,
    /// The region's own natural length, auto-grown by `grow_length_to_fit_notes` for piano-roll
    /// content the same way a `Pattern`'s length used to be.
    pub content_length_steps: usize,
    /// The on-timeline span this region occupies. Shorter than `content_length_steps` truncates
    /// the content; longer loops it — dragging a region's right edge in the Playlist controls
    /// this, independent of the content's own length.
    pub loop_length_steps: usize,
    pub content: RegionContent,
    /// Ticks (from `start_tick`) over which this region's on-timeline output ramps up from
    /// silence — see `fade_gain_at`. `#[serde(default)]` so song files saved before fades existed
    /// still load with none.
    #[serde(default)]
    pub fade_in_ticks: usize,
    /// Ticks (into the end of the on-timeline span) over which this region's output ramps down to
    /// silence — see `fade_gain_at`. `#[serde(default)]` for the same reason as `fade_in_ticks`.
    #[serde(default)]
    pub fade_out_ticks: usize,
    /// This region's automation lanes (volume/pan/send-level/effect-param "rides" — see
    /// `AutomationLane`), evaluated against the same on-timeline offset `fade_gain_at` uses.
    /// `#[serde(default)]` so song files saved before automation existed still load with none.
    #[serde(default)]
    pub automation: Vec<AutomationLane>,
}

/// Which of a track's (or another track's, or a send/master bus's) continuously-varying
/// parameters an `AutomationLane` rides. A lane still always *lives on* one `Region`, owned by one
/// track — that doesn't change — but its target no longer has to be that same track: the
/// `OtherTrack*`/`SendEffectParam`/`MasterEffectParam` variants let a lane on one track's region
/// ride a different track's fader, another track's send level, a send bus's own effect chain, or
/// the master bus's effect chain. `audio::collect_automation` resolves every region's lanes into
/// per-owner buckets each buffer, regardless of which track's region a lane happened to live on.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum AutomationTarget {
    /// Rides this lane's own track's `Track::volume`.
    Volume,
    /// Rides this lane's own track's `Track::pan`.
    Pan,
    /// Rides this lane's own track's `Track::send_levels[send_index]` (index into `Song::sends`).
    SendLevel { send_index: usize },
    /// Rides one parameter of the effect loaded in this lane's own track's own
    /// `Track::effects[slot_index]` — which parameter is identified by `key`, since a CLAP
    /// plugin's parameters and a built-in effect's parameters are addressed differently (see
    /// `EffectParamKey`).
    EffectParam {
        slot_index: usize,
        key: EffectParamKey,
    },
    /// Rides `Song::tracks[track_index].volume` — a *different* track than the one this lane's
    /// region lives on.
    OtherTrackVolume { track_index: usize },
    /// Rides `Song::tracks[track_index].pan` — a *different* track than the one this lane's
    /// region lives on.
    OtherTrackPan { track_index: usize },
    /// Rides `Song::tracks[track_index].send_levels[send_index]` — a *different* track than the
    /// one this lane's region lives on.
    OtherTrackSendLevel { track_index: usize, send_index: usize },
    /// Rides one parameter of the effect loaded in a *different* track's
    /// `Track::effects[slot_index]` than the one this lane's region lives on.
    OtherTrackEffectParam {
        track_index: usize,
        slot_index: usize,
        key: EffectParamKey,
    },
    /// Rides one parameter of the effect loaded in `Song::sends[send_index]`'s own
    /// `SendBus::effects[slot_index]` — the send bus's own chain, distinct from
    /// `SendLevel`/`OtherTrackSendLevel`, which ride how much of a track feeds that bus.
    SendEffectParam {
        send_index: usize,
        slot_index: usize,
        key: EffectParamKey,
    },
    /// Rides one parameter of the effect loaded in `Song::master_effects[slot_index]` — the master
    /// bus's own chain.
    MasterEffectParam {
        slot_index: usize,
        key: EffectParamKey,
    },
}

/// Addresses one parameter within an effect chain slot, for `AutomationTarget::EffectParam`. CLAP
/// plugins expose a stable numeric id per parameter (`plugin_host::PluginParamInfo::id`, the same
/// id `TrackEffectConfig::Clap::params` is keyed by); built-in effects have no such id, only named
/// `pub` fields on their own struct, so they're addressed by name instead (see
/// `builtin_fx::BuiltInEffect::automatable_param_names`/`set_automatable_param`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum EffectParamKey {
    Clap { param_id: u32 },
    BuiltIn { param_name: String },
}

/// How `AutomationLane::value_at_fractional` interpolates the segment running *from* a point *to*
/// the next one — a property of the earlier point, not the lane as a whole, so different segments
/// of the same lane can shape differently (e.g. a fast attack into a held sustain). Fixed shapes
/// rather than a continuously adjustable curve amount (the way Logic/Cubase let you drag a curve's
/// bend) — the minimum needed to cover "fades in gently"/"fades in sharply"/"snaps" without a new
/// per-point drag gesture; `Exponential`/`Logarithmic` are eased power curves (`t*t`/
/// `1-(1-t)*(1-t)`), not literal `exp`/`log` of the automated value, since a literal log/exp domain
/// doesn't hold up across every automation target this app has (`Pan` spans negative values, for
/// one).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum CurveShape {
    #[default]
    Linear,
    /// Eases in: slow to leave this point's value, accelerating into the next.
    Exponential,
    /// Eases out: quick to leave this point's value, decelerating into the next.
    Logarithmic,
    /// Stays at this point's value for the whole segment, then jumps at the very end — a step,
    /// not a ramp.
    Hold,
}

impl CurveShape {
    /// Warps a linear `0.0..=1.0` segment-progress fraction into this shape's own progress at
    /// that same point — `value_at_fractional` then interpolates `before.value..after.value` by
    /// the warped fraction instead of the raw one. `t` is expected already clamped to `0.0..=1.0`
    /// by the caller; out-of-range input isn't clamped again here. `pub` so
    /// `automation_lane_graph_ui` (`main.rs`) can draw a faithful curve preview using the exact
    /// same math playback evaluates, not a decorative approximation of it.
    pub fn warp(self, t: f32) -> f32 {
        match self {
            Self::Linear => t,
            Self::Exponential => t * t,
            Self::Logarithmic => 1.0 - (1.0 - t) * (1.0 - t),
            Self::Hold => 0.0,
        }
    }

    /// Cycles to the next shape in menu order — `automation_lane_graph_ui`'s "click an existing
    /// point to cycle its curve shape" gesture (see `main.rs`).
    pub fn next(self) -> Self {
        match self {
            Self::Linear => Self::Exponential,
            Self::Exponential => Self::Logarithmic,
            Self::Logarithmic => Self::Hold,
            Self::Hold => Self::Linear,
        }
    }
}

/// One (tick, value) point on an `AutomationLane`'s curve — `tick` is relative to the owning
/// `Region`'s own `start_tick` (this region's local time, like `fade_in_ticks`/`fade_out_ticks`),
/// not an absolute song position. `value` is in whatever unit `AutomationLane::target` naturally
/// uses (linear gain for `Volume`/`SendLevel`, -1.0..1.0 for `Pan`, the parameter's own declared
/// range for `EffectParam`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AutomationPoint {
    pub tick: usize,
    pub value: f32,
    /// See `CurveShape`. Shapes the segment from this point to the next one; meaningless on the
    /// lane's last point (there's no next segment to shape). `#[serde(default)]` so lanes saved
    /// before curve shapes existed still load as straight lines — unchanged behavior.
    #[serde(default)]
    pub curve: CurveShape,
}

/// A single automated parameter's "ride" over a `Region`'s on-timeline span: a target plus an
/// ordered-by-nothing-in-particular list of (tick, value) points, linearly interpolated between —
/// see `value_at_fractional`. A `fade_in_ticks`/`fade_out_ticks` ramp is conceptually just a
/// 2-point `Volume` lane; this is the general form ("rides" — riding a fader up/down over multiple
/// points, not just a straight ramp).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AutomationLane {
    pub target: AutomationTarget,
    pub points: Vec<AutomationPoint>,
}

impl AutomationLane {
    /// This lane's value at `tick` (relative to the region's `start_tick`, same convention as
    /// `AutomationPoint::tick`), interpolated between the two points bracketing it by the earlier
    /// point's own `CurveShape` (straight-line by default) — holds the nearest point's value
    /// outside the lane's own range. `None` if the lane has no points at all, meaning "not
    /// automated yet" — callers fall back to the target's static value (`Track::volume`, etc.) in
    /// that case, the same way an empty `points` list reads as "no override" rather than "override
    /// to 0".
    ///
    /// Takes a fractional tick rather than a whole sequencer tick so a caller evaluating per audio
    /// sample (rather than per sequencer tick) gets a smoothly interpolated value instead of one
    /// quantized to tick boundaries — see `audio::TrackAutomationOverride`, which evaluates this
    /// once per output sample.
    ///
    /// Scans `points` rather than requiring them pre-sorted by tick, so a dragged point can be
    /// written in place (by index) during a UI drag without needing to keep the whole list sorted
    /// mid-gesture — an O(n) scan per lookup is the same not-fully-incremental trade-off
    /// `metering.rs`'s integrated-LUFS rescan already makes, and lanes are short (a handful of
    /// points, not thousands).
    pub fn value_at_fractional(&self, tick: f64) -> Option<f32> {
        if self.points.is_empty() {
            return None;
        }
        let before = self.points.iter().filter(|p| (p.tick as f64) <= tick).max_by_key(|p| p.tick);
        let after = self.points.iter().filter(|p| (p.tick as f64) > tick).min_by_key(|p| p.tick);
        Some(match (before, after) {
            (Some(before), Some(after)) => {
                let span = (after.tick - before.tick) as f64;
                let frac = if span > 0.0 {
                    (tick - before.tick as f64) / span
                } else {
                    0.0
                };
                let warped = before.curve.warp((frac as f32).clamp(0.0, 1.0));
                before.value + (after.value - before.value) * warped
            }
            (Some(before), None) => before.value,
            (None, Some(after)) => after.value,
            (None, None) => unreachable!("points is non-empty, so at least one side must match"),
        })
    }
}

impl Region {
    /// This region's fade gain (0.0..1.0) at `ticks_since_start` ticks past `start_tick`, against
    /// the on-timeline span (`0..loop_length_steps * TICKS_PER_STEP` — the same offset
    /// `Sequencer::process`'s active-region filter already computes, not the content's own,
    /// possibly-looped length). A region shorter than `fade_in_ticks + fade_out_ticks` has
    /// overlapping ramps; at any given point this takes whichever is more attenuated, rather than
    /// one silently overriding the other.
    pub fn fade_gain_at(&self, ticks_since_start: usize) -> f32 {
        let span_ticks = self.loop_length_steps * TICKS_PER_STEP;
        let mut gain = 1.0f32;
        if self.fade_in_ticks > 0 {
            gain = gain.min((ticks_since_start as f32 / self.fade_in_ticks as f32).clamp(0.0, 1.0));
        }
        if self.fade_out_ticks > 0 {
            let ticks_from_end = span_ticks.saturating_sub(ticks_since_start);
            gain = gain.min((ticks_from_end as f32 / self.fade_out_ticks as f32).clamp(0.0, 1.0));
        }
        gain
    }

    /// This region's content length (`content_length_steps`) in ticks.
    pub fn content_length_ticks(&self) -> usize {
        self.content_length_steps * TICKS_PER_STEP
    }
}

/// One clip slot's content in the Session View grid (see `Track::session_clips`) — an
/// independently launchable/stoppable loop, not pinned to any absolute song tick the way a
/// `Region`/`AudioClip` on the Playlist is. Reuses `RegionContent` for step-grid/piano-roll
/// material rather than duplicating `Lane`/`Note` storage a second time (the same content,
/// replayed a different way). The `Audio` variant wraps a whole `AudioClip` but its `start_tick`
/// is meaningless here (always `0`, see `SessionClip::from_audio_clip`) since a session clip's
/// timeline position is decided at launch time, not authored.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SessionClipContent {
    Region {
        content: RegionContent,
        /// Mirrors `Region::content_length_steps`.
        content_length_steps: usize,
        /// Mirrors `Region::loop_length_steps` — the slot's loop length in steps.
        loop_length_steps: usize,
    },
    Audio(AudioClip),
}

/// What a Session View slot does on its own, with no user input, once it's played through
/// `FollowActionConfig::times` times — Ableton's "follow action" concept. Every variant except
/// `Other` resolves relative to the clip's own row within its track (see
/// `session::follow_action_target`); `Other` names an absolute row index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FollowAction {
    /// Keep looping indefinitely — the default, identical to Session View's behavior before
    /// follow actions existed.
    None,
    Stop,
    /// Restart this same slot from the top (resets its own loop count).
    Again,
    Previous,
    Next,
    First,
    Last,
    /// A random row on the same track (uniformly, including possibly this same row again).
    Any,
    /// An absolute row index on the same track.
    Other(usize),
}

/// How a Session View slot responds to being launched — Ableton's four launch modes. `Toggle`
/// matches this app's original (pre-launch-mode) behavior exactly, so it's the default every
/// existing `SessionClip` keeps on load (`#[serde(default)]`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum LaunchMode {
    /// Click starts the clip; clicking it again while playing/queued stops it.
    #[default]
    Toggle,
    /// Click (re)starts the clip; clicking it again while playing retriggers it from the top
    /// rather than stopping it — stopping needs an explicit "Stop" (context menu, scene, or
    /// another exclusive launch on the same track).
    Trigger,
    /// Plays only while held down; releasing stops it immediately, bypassing launch
    /// quantization on the way out (matching a physical gate). Driven by
    /// `SessionLaunchRequest::held`, not `SessionLaunchRequest::intent`.
    Gate,
    /// Like `Gate`, but retriggers the clip from the top every time the launch-quantize
    /// boundary is crossed while still held — a stutter/roll effect.
    Repeat,
}

/// How many ticks a Session View launch/stop snaps forward to — see
/// `session::next_quantize_boundary`. `None` acts immediately, at the current tick. The grid-wide
/// setting lives as live UI state (`SimpleDawApp::session_quantize`), resolved to raw ticks once
/// per frame (`audio::Transport::session_quantize_ticks`) since that resolution already needs
/// `Song::steps_per_bar`; `SessionClip::quantize_override` reuses this same enum, not a raw tick
/// count, so a per-clip override stays correct across a time-signature change too instead of
/// freezing in whatever tick count happened to be true when it was set.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionQuantize {
    None,
    QuarterBar,
    HalfBar,
    #[default]
    OneBar,
    TwoBar,
    FourBar,
}

impl SessionQuantize {
    /// Ticks per quantize unit at `song`'s own time signature — see `Song::steps_per_bar`.
    pub fn ticks(self, song: &Song) -> usize {
        let bar_ticks = song.steps_per_bar() * TICKS_PER_STEP;
        match self {
            Self::None => 0,
            Self::QuarterBar => bar_ticks / 4,
            Self::HalfBar => bar_ticks / 2,
            Self::OneBar => bar_ticks,
            Self::TwoBar => bar_ticks * 2,
            Self::FourBar => bar_ticks * 4,
        }
    }
}

/// A Session View slot's follow action — see `FollowAction`. Ableton's model: two independent
/// candidate actions, picked by weighted random choice each time the follow action fires (not
/// picked once and fixed), after the clip has played through `times` times.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FollowActionConfig {
    pub times: u32,
    pub action_a: FollowAction,
    pub chance_a: f32,
    pub action_b: FollowAction,
    pub chance_b: f32,
}

impl Default for FollowActionConfig {
    /// "No follow action": plays once through `times` (`1`) and then just keeps looping, same as
    /// every `SessionClip` behaved before this field existed.
    fn default() -> Self {
        Self {
            times: 1,
            action_a: FollowAction::None,
            chance_a: 1.0,
            action_b: FollowAction::None,
            chance_b: 0.0,
        }
    }
}

/// One slot in the Session View grid (see `Track::session_clips`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SessionClip {
    pub name: String,
    pub content: SessionClipContent,
    /// See `FollowAction`. `#[serde(default)]` so session clips saved before follow actions
    /// existed still load with none (unchanged "loop forever" behavior).
    #[serde(default)]
    pub follow_action: FollowActionConfig,
    /// When true, launching this clip doesn't restart its own playhead at tick `0` — it continues
    /// from wherever the track's previously-playing slot's phase was (see
    /// `audio::Sequencer::trigger_session_clips`'s legato handling), for a seamless handoff between
    /// same-length variations. Still waits for the normal launch-quantize boundary like any other
    /// launch; only the starting phase changes. `#[serde(default)]` so session clips saved before
    /// legato existed still load with it off (unchanged restart-at-0 behavior).
    #[serde(default)]
    pub legato: bool,
    /// See `LaunchMode`. `#[serde(default)]` so session clips saved before launch modes existed
    /// still load as `Toggle` — this app's original click-to-start/click-to-stop behavior.
    #[serde(default)]
    pub launch_mode: LaunchMode,
    /// Overrides the Session View grid's own quantize setting for this clip alone, when set —
    /// see `SessionQuantize`. `#[serde(default)]` so session clips saved before this existed
    /// still load as `None` (always follow the grid-wide setting, unchanged prior behavior).
    #[serde(default)]
    pub quantize_override: Option<SessionQuantize>,
    /// This slot's own automation lanes ("rides" — see `AutomationLane`), evaluated against the
    /// slot's own `SlotState::Playing`/`QueuedStop` `local_tick` instead of a `Region`'s
    /// `region_local_tick` — see `audio::collect_automation`'s session-mode path. Distinct from
    /// arrangement-wide `Track::automation`, which still applies during Session View playback too.
    /// `#[serde(default)]` so session clips saved before this existed still load with none.
    #[serde(default)]
    pub automation: Vec<AutomationLane>,
}

impl SessionClip {
    /// Copies `region`'s content into a new session clip — one of two ways a slot gets `Region`
    /// content, alongside composing fresh content in place (see `new_piano_roll`/`to_region` for
    /// the reverse direction). See `session_view_ui`'s "Assign from Playlist" menu.
    pub fn from_region(region: &Region) -> Self {
        Self {
            name: region.name.clone(),
            content: SessionClipContent::Region {
                content: region.content.clone(),
                content_length_steps: region.content_length_steps,
                loop_length_steps: region.loop_length_steps,
            },
            follow_action: FollowActionConfig::default(),
            legato: false,
            launch_mode: LaunchMode::default(),
            quantize_override: None,
            automation: Vec::new(),
        }
    }

    /// Copies `clip`'s content into a new session clip, same idea as `from_region`. `start_tick`
    /// is reset to `0` — meaningless for session playback, see `SessionClipContent::Audio`'s doc
    /// comment.
    pub fn from_audio_clip(clip: &AudioClip) -> Self {
        let mut clip = clip.clone();
        clip.start_tick = 0;
        Self {
            name: clip.file_path.clone(),
            content: SessionClipContent::Audio(clip),
            follow_action: FollowActionConfig::default(),
            legato: false,
            launch_mode: LaunchMode::default(),
            quantize_override: None,
            automation: Vec::new(),
        }
    }

    /// A blank, freshly-composed piano-roll session clip — the Session View counterpart of
    /// `Track::add_region` for a `PianoRoll`-kind track, used by a slot's "Compose New Region…"
    /// instead of copying already-authored Playlist content (see `from_region`). `length_steps`
    /// is both the clip's initial content and loop length, matching `add_region`'s own default of
    /// one bar.
    pub fn new_piano_roll(name: impl Into<String>, length_steps: usize) -> Self {
        Self {
            name: name.into(),
            content: SessionClipContent::Region {
                content: RegionContent::PianoRoll(Vec::new()),
                content_length_steps: length_steps,
                loop_length_steps: length_steps,
            },
            follow_action: FollowActionConfig::default(),
            legato: false,
            launch_mode: LaunchMode::default(),
            quantize_override: None,
            automation: Vec::new(),
        }
    }

    /// A blank, freshly-composed step-grid session clip — the `StepGrid`-track counterpart of
    /// `new_piano_roll`, used by a slot's "Compose New Beats Lane…". Starts with no lanes at all
    /// (same as `Track::add_region`'s own `StepGrid` default when the track has no existing
    /// region to copy a lane layout from) — lanes are added from the Beats window's own "+ Lane"
    /// button once the slot is open for editing.
    pub fn new_step_grid(name: impl Into<String>, length_steps: usize) -> Self {
        Self {
            name: name.into(),
            content: SessionClipContent::Region {
                content: RegionContent::StepGrid(Vec::new()),
                content_length_steps: length_steps,
                loop_length_steps: length_steps,
            },
            follow_action: FollowActionConfig::default(),
            legato: false,
            launch_mode: LaunchMode::default(),
            quantize_override: None,
            automation: Vec::new(),
        }
    }

    /// Converts this clip's own `Region` content into a new Playlist `Region` at `start_tick` —
    /// the reverse of `from_region`, used by a slot's "Send to Playlist" action. A copy, not a
    /// move (the slot keeps its own content afterward), matching Ableton's own "drag a Session
    /// clip into the Arrangement" behavior — the two become independent from that point on. `None`
    /// for `Audio` content (out of scope for this action — Session View has no timeline position
    /// to place an audio clip's own fades/trim against the way a `PianoRoll`/`StepGrid` region's
    /// content ports over directly).
    pub fn to_region(&self, start_tick: usize) -> Option<Region> {
        let SessionClipContent::Region { content, content_length_steps, loop_length_steps } =
            &self.content
        else {
            return None;
        };
        Some(Region {
            name: self.name.clone(),
            start_tick,
            content_length_steps: *content_length_steps,
            loop_length_steps: *loop_length_steps,
            content: content.clone(),
            fade_in_ticks: 0,
            fade_out_ticks: 0,
            // Both are local-tick rides over a loop span (see this slot's own `automation` doc
            // comment), so copying carries over cleanly rather than losing the slot's rides.
            automation: self.automation.clone(),
        })
    }

    /// This clip's loop length in ticks — the audio engine's session playhead wraps modulo this.
    /// The `Audio` variant loops its whole `effective_length_ticks` (no separate loop-range trim
    /// in v1 — see `SessionClipContent::Audio`'s doc comment). `ticks_per_second` is the caller's
    /// job to supply, same reasoning as `AudioClip::full_length_ticks`, to avoid this module
    /// depending on `audio.rs`.
    pub fn loop_length_ticks(&self, ticks_per_second: f64) -> usize {
        match &self.content {
            SessionClipContent::Region { loop_length_steps, .. } => {
                loop_length_steps * TICKS_PER_STEP
            }
            SessionClipContent::Audio(clip) => clip.effective_length_ticks(ticks_per_second),
        }
    }
}

/// A pending user click on a Session View slot, communicated from the UI thread to the audio
/// thread through the same per-callback `Song` snapshot everything else here rides on (see
/// `audio::build_playback_stream`'s snapshot-clone). Not song data — never persisted (see
/// `Track::session_launch_requests`). `generation`/`intent` drive `LaunchMode::Toggle`/`Trigger`:
/// `generation` is bumped on every click, and the audio thread's own long-lived `Sequencer` state
/// (untouched by the snapshot clone) remembers the last generation it's seen per slot, only
/// acting when this one has moved on, since a cloned snapshot has no other way to signal "this
/// changed since last buffer." `held` instead drives `LaunchMode::Gate`/`Repeat`: it's a level
/// (not edge) signal, read fresh from whatever the latest snapshot carries every buffer — no
/// generation needed, since "is the button currently down" doesn't have a discrete "moment" to
/// miss the way a click does.
#[derive(Clone, Copy, Debug, Default)]
pub struct SessionLaunchRequest {
    pub generation: u64,
    pub intent: LaunchIntent,
    pub held: bool,
}

/// See `SessionLaunchRequest`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LaunchIntent {
    #[default]
    Play,
    Stop,
}

/// Adds a new note, clearing any existing note on the same pitch that it
/// would overlap (two notes overlapping at the same pitch is ambiguous for
/// playback — which one's on?). Returns the new note's id.
pub fn add_note(
    notes: &mut Vec<Note>,
    next_note_id: &mut u64,
    pitch: u8,
    start_tick: usize,
    length_ticks: usize,
    velocity: u8,
) -> u64 {
    let id = *next_note_id;
    *next_note_id += 1;
    let length_ticks = length_ticks.max(1);
    clear_overlaps(notes, id, pitch, start_tick, length_ticks);
    notes.push(Note {
        id,
        pitch,
        start_tick,
        length_ticks,
        velocity: velocity.min(127),
    });
    id
}

/// Removes the note with the given `id`, if present.
pub fn remove_note(notes: &mut Vec<Note>, id: u64) {
    notes.retain(|n| n.id != id);
}

/// Finds the note with the given `id`, if present.
pub fn find_note_mut(notes: &mut [Note], id: u64) -> Option<&mut Note> {
    notes.iter_mut().find(|n| n.id == id)
}

/// Removes any note other than `keep_id` on `pitch` whose span overlaps
/// `[start_tick, start_tick + length_ticks)`. Called after a move/resize/
/// create settles on a final position, not during an in-progress drag.
pub fn clear_overlaps(
    notes: &mut Vec<Note>,
    keep_id: u64,
    pitch: u8,
    start_tick: usize,
    length_ticks: usize,
) {
    let end = start_tick + length_ticks.max(1);
    notes.retain(|n| {
        n.id == keep_id
            || n.pitch != pitch
            || n.start_tick >= end
            || n.start_tick + n.length_ticks <= start_tick
    });
}

/// Which editing surface a track uses — fixed for the track's lifetime. A `StepGrid`/`PianoRoll`
/// track's actual musical content lives in its own `Track::regions`, each independently
/// positioned; the variant here just determines which of `RegionContent`'s cases those regions use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrackKind {
    StepGrid,
    PianoRoll,
    /// A track of recorded/imported audio clips (`Track::audio_clips`), positioned at absolute
    /// song ticks — the audio equivalent of a `StepGrid`/`PianoRoll` track's `regions`.
    Audio,
}

/// Grows `length_steps` (in whole bars of `steps_per_bar`, see `Song::steps_per_bar`) so it
/// always covers the furthest piano-roll note. Never shrinks — a piano roll can be edited past
/// its current end and the declared length follows rather than clipping playback/export, but
/// deleting notes doesn't retroactively shorten a pattern other tracks may be looping against.
pub fn grow_length_to_fit_notes(length_steps: &mut usize, notes: &[Note], steps_per_bar: usize) {
    let content_end_tick = notes
        .iter()
        .map(|n| n.start_tick + n.length_ticks)
        .max()
        .unwrap_or(0);
    let content_steps = content_end_tick.div_ceil(TICKS_PER_STEP);
    let bars = content_steps.div_ceil(steps_per_bar).max(1);
    *length_steps = (*length_steps).max(bars * steps_per_bar);
}

/// One effect in a track's insert chain: either a hosted CLAP plugin, or one of the app's
/// built-in DSP effects that need no external plugin file. Pure data — the live plugin instance
/// or DSP state lives outside the model (see `plugin_host::TrackEffectSlots`); the app layer
/// re-loads/re-creates each entry and re-applies its parameters after a `Song` is deserialized,
/// the same way it re-loads sample lanes. `#[serde(tag = "kind")]` so a saved song's JSON is
/// self-describing about which effect each chain slot is.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum TrackEffectConfig {
    /// A CLAP effect plugin: its file path, and its parameter values (by CLAP param id, not
    /// index, so a value still lands on the right parameter even if a plugin update reorders its
    /// declared parameter list) at last save.
    Clap {
        path: String,
        #[serde(default)]
        params: Vec<(u32, f64)>,
        /// Index into `Song.tracks` of a track whose own pre-effects signal feeds this plugin's
        /// sidechain key input — only takes effect if the loaded plugin actually declared a
        /// second input port (`plugin_host::LoadedEffect::has_sidechain_input`); otherwise
        /// silently ignored, same tolerance as a redirected automation target pointing at a track
        /// that doesn't exist. `None` (the default) means no sidechain routing.
        #[serde(default)]
        sidechain_source: Option<usize>,
    },
    /// A feedback delay line ("echo"). `feedback` is how much of the delayed signal feeds back
    /// into itself (0..1, higher means more repeats); `mix` is the dry/wet blend (0 = dry, 1 =
    /// fully wet).
    Delay {
        #[serde(default = "default_delay_time_ms")]
        time_ms: f32,
        #[serde(default = "default_delay_feedback")]
        feedback: f32,
        #[serde(default = "default_effect_mix")]
        mix: f32,
    },
    /// Sample-rate/bit-depth reduction for lo-fi/chiptune grit. `bit_depth` (1..16) sets the
    /// number of quantization steps; `rate_divisor` is how many samples each output value is
    /// held for before the next one is sampled.
    Bitcrusher {
        #[serde(default = "default_bitcrusher_bit_depth")]
        bit_depth: f32,
        #[serde(default = "default_bitcrusher_rate_divisor")]
        rate_divisor: u32,
        #[serde(default = "default_effect_mix_full")]
        mix: f32,
    },
    /// Tanh waveshaping drive.
    Distortion {
        #[serde(default = "default_distortion_drive")]
        drive: f32,
        #[serde(default = "default_effect_mix_full")]
        mix: f32,
    },
    /// A simple Schroeder-style (parallel comb + series allpass) mono reverb.
    Reverb {
        #[serde(default = "default_reverb_room_size")]
        room_size: f32,
        #[serde(default = "default_reverb_damping")]
        damping: f32,
        #[serde(default = "default_reverb_mix")]
        mix: f32,
    },
    /// An LFO-modulated short delay line ("chorus"/detune thickening).
    Chorus {
        #[serde(default = "default_chorus_rate_hz")]
        rate_hz: f32,
        #[serde(default = "default_chorus_depth_ms")]
        depth_ms: f32,
        #[serde(default = "default_effect_mix")]
        mix: f32,
    },
    /// A resonant state-variable filter, switchable between low-pass and high-pass.
    Filter {
        #[serde(default = "default_filter_cutoff_hz")]
        cutoff_hz: f32,
        #[serde(default = "default_filter_resonance")]
        resonance: f32,
        #[serde(default = "default_filter_mode")]
        mode: FilterMode,
        #[serde(default = "default_effect_mix_full")]
        mix: f32,
    },
    /// LFO-driven amplitude modulation.
    Tremolo {
        #[serde(default = "default_tremolo_rate_hz")]
        rate_hz: f32,
        #[serde(default = "default_tremolo_depth")]
        depth: f32,
    },
    /// Feedforward, dB-domain dynamics compressor (peak envelope follower with attack/release).
    Compressor {
        #[serde(default = "default_compressor_threshold_db")]
        threshold_db: f32,
        #[serde(default = "default_compressor_ratio")]
        ratio: f32,
        #[serde(default = "default_compressor_attack_ms")]
        attack_ms: f32,
        #[serde(default = "default_compressor_release_ms")]
        release_ms: f32,
        #[serde(default = "default_compressor_makeup_db")]
        makeup_db: f32,
        /// See `TrackEffectConfig::Clap`'s `sidechain_source` doc — same meaning, here driving
        /// the built-in compressor's envelope follower instead of a CLAP plugin's own port.
        #[serde(default)]
        sidechain_source: Option<usize>,
    },
    /// A short LFO-modulated delay with feedback ("flanger") — like `Chorus` but with a shorter
    /// delay range and a feedback path, giving it a more resonant, metallic sweep.
    Flanger {
        #[serde(default = "default_flanger_rate_hz")]
        rate_hz: f32,
        #[serde(default = "default_flanger_depth_ms")]
        depth_ms: f32,
        #[serde(default = "default_flanger_feedback")]
        feedback: f32,
        #[serde(default = "default_effect_mix")]
        mix: f32,
    },
    /// A cascade of LFO-swept first-order allpass stages ("phaser").
    Phaser {
        #[serde(default = "default_phaser_rate_hz")]
        rate_hz: f32,
        #[serde(default = "default_phaser_depth")]
        depth: f32,
        #[serde(default = "default_phaser_feedback")]
        feedback: f32,
        #[serde(default = "default_effect_mix")]
        mix: f32,
    },
    /// Multiplies the signal by a sine carrier oscillator for metallic/robotic tones.
    RingModulator {
        #[serde(default = "default_ring_mod_carrier_hz")]
        carrier_hz: f32,
        #[serde(default = "default_effect_mix")]
        mix: f32,
    },
    /// Attenuates the signal below a threshold, with independent attack/release smoothing and a
    /// configurable maximum attenuation (`range_db`) instead of hard-muting to silence.
    NoiseGate {
        #[serde(default = "default_noise_gate_threshold_db")]
        threshold_db: f32,
        #[serde(default = "default_noise_gate_attack_ms")]
        attack_ms: f32,
        #[serde(default = "default_noise_gate_release_ms")]
        release_ms: f32,
        #[serde(default = "default_noise_gate_range_db")]
        range_db: f32,
        /// See `TrackEffectConfig::Clap`'s `sidechain_source` doc — same meaning, here driving
        /// the built-in gate's envelope follower instead of a CLAP plugin's own port.
        #[serde(default)]
        sidechain_source: Option<usize>,
    },
    /// Flips signal polarity, independently per channel — for stereo phase troubleshooting (e.g.
    /// correcting an out-of-phase mic pair), not tone shaping, so it has no dry/wet mix.
    PhaseInvert {
        #[serde(default)]
        invert_left: bool,
        #[serde(default)]
        invert_right: bool,
    },
    /// A parametric multiband EQ modeled on Logic's Channel EQ: a fixed chain of 8 bands (low
    /// cut, low shelf, four peaking bands, high shelf, high cut), each independently switchable
    /// and tunable. See `EqBand`.
    ChannelEq {
        #[serde(default = "default_channel_eq_bands")]
        bands: Vec<EqBand>,
    },
    /// Look-ahead brickwall/peak limiter: `input_gain_db` drives the signal in, `ceiling_db` is
    /// the hard output ceiling, `release_ms` sets gain recovery speed after a loud passage. See
    /// `LimiterEffect` for why look-ahead makes this a hard ceiling rather than a soft target the
    /// way `Compressor` is.
    Limiter {
        #[serde(default = "default_limiter_input_gain_db")]
        input_gain_db: f32,
        #[serde(default = "default_limiter_ceiling_db")]
        ceiling_db: f32,
        #[serde(default = "default_limiter_release_ms")]
        release_ms: f32,
    },
}

/// Which frequencies a `TrackEffectConfig::Filter`/`FilterEffect` passes through.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum FilterMode {
    LowPass,
    HighPass,
}

/// The filter shape one `EqBand` applies. `HighPass`/`LowPass` ignore `EqBand::gain_db` (a cut
/// has no gain, only a corner frequency and a resonance); the other three shape gain around
/// `freq_hz`, wide (shelf) or narrow (peak).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum EqBandType {
    HighPass,
    LowShelf,
    #[default]
    Peak,
    HighShelf,
    LowPass,
}

/// One band of a `TrackEffectConfig::ChannelEq`/`ChannelEqEffect`: a single biquad stage with its
/// own shape, corner/center frequency, gain (peak/shelf only), resonance/bandwidth, and bypass.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct EqBand {
    pub band_type: EqBandType,
    pub freq_hz: f32,
    pub gain_db: f32,
    pub q: f32,
    pub enabled: bool,
}

impl Default for EqBand {
    fn default() -> Self {
        Self {
            band_type: EqBandType::Peak,
            freq_hz: 1000.0,
            gain_db: 0.0,
            q: 0.7,
            enabled: true,
        }
    }
}

/// The 8-band layout `TrackEffectConfig::default_channel_eq()` starts a new Channel EQ with,
/// mirroring Logic's Channel EQ: a low cut and high cut (off by default, since a cut is
/// immediately audible) bracketing a low shelf, four peaking bands spread across the audible
/// range, and a high shelf (all four on, at 0dB gain so they start transparent).
fn default_channel_eq_bands() -> Vec<EqBand> {
    vec![
        EqBand {
            band_type: EqBandType::HighPass,
            freq_hz: 30.0,
            gain_db: 0.0,
            q: 0.7,
            enabled: false,
        },
        EqBand {
            band_type: EqBandType::LowShelf,
            freq_hz: 100.0,
            gain_db: 0.0,
            q: 0.7,
            enabled: true,
        },
        EqBand {
            band_type: EqBandType::Peak,
            freq_hz: 250.0,
            gain_db: 0.0,
            q: 0.7,
            enabled: true,
        },
        EqBand {
            band_type: EqBandType::Peak,
            freq_hz: 1000.0,
            gain_db: 0.0,
            q: 0.7,
            enabled: true,
        },
        EqBand {
            band_type: EqBandType::Peak,
            freq_hz: 2500.0,
            gain_db: 0.0,
            q: 0.7,
            enabled: true,
        },
        EqBand {
            band_type: EqBandType::Peak,
            freq_hz: 6000.0,
            gain_db: 0.0,
            q: 0.7,
            enabled: true,
        },
        EqBand {
            band_type: EqBandType::HighShelf,
            freq_hz: 10000.0,
            gain_db: 0.0,
            q: 0.7,
            enabled: true,
        },
        EqBand {
            band_type: EqBandType::LowPass,
            freq_hz: 18000.0,
            gain_db: 0.0,
            q: 0.7,
            enabled: false,
        },
    ]
}

fn default_delay_time_ms() -> f32 {
    300.0
}
fn default_delay_feedback() -> f32 {
    0.35
}
fn default_effect_mix() -> f32 {
    0.35
}
fn default_effect_mix_full() -> f32 {
    1.0
}
fn default_bitcrusher_bit_depth() -> f32 {
    6.0
}
fn default_bitcrusher_rate_divisor() -> u32 {
    4
}
fn default_distortion_drive() -> f32 {
    4.0
}
fn default_reverb_room_size() -> f32 {
    0.5
}
fn default_reverb_damping() -> f32 {
    0.5
}
fn default_reverb_mix() -> f32 {
    0.3
}
fn default_chorus_rate_hz() -> f32 {
    1.2
}
fn default_chorus_depth_ms() -> f32 {
    8.0
}
fn default_filter_cutoff_hz() -> f32 {
    2000.0
}
fn default_filter_resonance() -> f32 {
    0.3
}
fn default_filter_mode() -> FilterMode {
    FilterMode::LowPass
}
fn default_tremolo_rate_hz() -> f32 {
    5.0
}
fn default_tremolo_depth() -> f32 {
    0.5
}
fn default_compressor_threshold_db() -> f32 {
    -18.0
}
fn default_compressor_ratio() -> f32 {
    4.0
}
fn default_compressor_attack_ms() -> f32 {
    10.0
}
fn default_compressor_release_ms() -> f32 {
    100.0
}
fn default_compressor_makeup_db() -> f32 {
    0.0
}
fn default_flanger_rate_hz() -> f32 {
    0.5
}
fn default_flanger_depth_ms() -> f32 {
    3.0
}
fn default_flanger_feedback() -> f32 {
    0.5
}
fn default_phaser_rate_hz() -> f32 {
    0.5
}
fn default_phaser_depth() -> f32 {
    0.7
}
fn default_phaser_feedback() -> f32 {
    0.3
}
fn default_ring_mod_carrier_hz() -> f32 {
    200.0
}
fn default_noise_gate_threshold_db() -> f32 {
    -40.0
}
fn default_noise_gate_attack_ms() -> f32 {
    2.0
}
fn default_noise_gate_release_ms() -> f32 {
    150.0
}
fn default_noise_gate_range_db() -> f32 {
    -60.0
}
fn default_limiter_input_gain_db() -> f32 {
    0.0
}
fn default_limiter_ceiling_db() -> f32 {
    -0.3
}
fn default_limiter_release_ms() -> f32 {
    50.0
}

impl TrackEffectConfig {
    /// Default parameter values used when a user picks "Delay / Echo" from the "+ Add Effect"
    /// menu (see `main.rs`'s `track_ui`).
    pub fn default_delay() -> Self {
        TrackEffectConfig::Delay {
            time_ms: default_delay_time_ms(),
            feedback: default_delay_feedback(),
            mix: default_effect_mix(),
        }
    }
    /// Default parameter values used when a user picks "Bitcrusher" from the "+ Add Effect" menu.
    pub fn default_bitcrusher() -> Self {
        TrackEffectConfig::Bitcrusher {
            bit_depth: default_bitcrusher_bit_depth(),
            rate_divisor: default_bitcrusher_rate_divisor(),
            mix: default_effect_mix_full(),
        }
    }
    /// Default parameter values used when a user picks "Distortion" from the "+ Add Effect" menu.
    pub fn default_distortion() -> Self {
        TrackEffectConfig::Distortion {
            drive: default_distortion_drive(),
            mix: default_effect_mix_full(),
        }
    }
    /// Default parameter values used when a user picks "Reverb" from the "+ Add Effect" menu.
    pub fn default_reverb() -> Self {
        TrackEffectConfig::Reverb {
            room_size: default_reverb_room_size(),
            damping: default_reverb_damping(),
            mix: default_reverb_mix(),
        }
    }
    /// Default parameter values used when a user picks "Chorus" from the "+ Add Effect" menu.
    pub fn default_chorus() -> Self {
        TrackEffectConfig::Chorus {
            rate_hz: default_chorus_rate_hz(),
            depth_ms: default_chorus_depth_ms(),
            mix: default_effect_mix(),
        }
    }
    /// Default parameter values used when a user picks "Filter" from the "+ Add Effect" menu.
    pub fn default_filter() -> Self {
        TrackEffectConfig::Filter {
            cutoff_hz: default_filter_cutoff_hz(),
            resonance: default_filter_resonance(),
            mode: default_filter_mode(),
            mix: default_effect_mix_full(),
        }
    }
    /// Default parameter values used when a user picks "Tremolo" from the "+ Add Effect" menu.
    pub fn default_tremolo() -> Self {
        TrackEffectConfig::Tremolo {
            rate_hz: default_tremolo_rate_hz(),
            depth: default_tremolo_depth(),
        }
    }
    /// Default parameter values used when a user picks "Compressor" from the "+ Add Effect" menu.
    pub fn default_compressor() -> Self {
        TrackEffectConfig::Compressor {
            threshold_db: default_compressor_threshold_db(),
            ratio: default_compressor_ratio(),
            attack_ms: default_compressor_attack_ms(),
            release_ms: default_compressor_release_ms(),
            makeup_db: default_compressor_makeup_db(),
            sidechain_source: None,
        }
    }
    /// Default parameter values used when a user picks "Flanger" from the "+ Add Effect" menu.
    pub fn default_flanger() -> Self {
        TrackEffectConfig::Flanger {
            rate_hz: default_flanger_rate_hz(),
            depth_ms: default_flanger_depth_ms(),
            feedback: default_flanger_feedback(),
            mix: default_effect_mix(),
        }
    }
    /// Default parameter values used when a user picks "Phaser" from the "+ Add Effect" menu.
    pub fn default_phaser() -> Self {
        TrackEffectConfig::Phaser {
            rate_hz: default_phaser_rate_hz(),
            depth: default_phaser_depth(),
            feedback: default_phaser_feedback(),
            mix: default_effect_mix(),
        }
    }
    /// Default parameter values used when a user picks "Ring Modulator" from the "+ Add Effect" menu.
    pub fn default_ring_modulator() -> Self {
        TrackEffectConfig::RingModulator {
            carrier_hz: default_ring_mod_carrier_hz(),
            mix: default_effect_mix(),
        }
    }
    /// Default parameter values used when a user picks "Noise Gate" from the "+ Add Effect" menu.
    pub fn default_noise_gate() -> Self {
        TrackEffectConfig::NoiseGate {
            threshold_db: default_noise_gate_threshold_db(),
            attack_ms: default_noise_gate_attack_ms(),
            release_ms: default_noise_gate_release_ms(),
            range_db: default_noise_gate_range_db(),
            sidechain_source: None,
        }
    }
    /// Default parameter values used when a user picks "Phase Invert" from the "+ Add Effect" menu.
    pub fn default_phase_invert() -> Self {
        TrackEffectConfig::PhaseInvert {
            invert_left: false,
            invert_right: false,
        }
    }
    /// Default parameter values used when a user picks "Channel EQ" from the "+ Add Effect" menu.
    pub fn default_channel_eq() -> Self {
        TrackEffectConfig::ChannelEq {
            bands: default_channel_eq_bands(),
        }
    }
    /// Default parameter values used when a user picks "Limiter" from the "+ Add Effect" menu.
    pub fn default_limiter() -> Self {
        TrackEffectConfig::Limiter {
            input_gain_db: default_limiter_input_gain_db(),
            ceiling_db: default_limiter_ceiling_db(),
            release_ms: default_limiter_release_ms(),
        }
    }
}

fn default_track_volume() -> f32 {
    1.0
}

fn default_input_gain() -> f32 {
    1.0
}

fn default_clip_gain() -> f32 {
    1.0
}

/// A recorded or imported audio region on an `Audio`-kind track, placed at an absolute song tick
/// (the audio equivalent of `Region`, which `StepGrid`/`PianoRoll` tracks use instead). Mirrors `Lane`'s
/// `sample_path`/`sample`/`sample_error` split: `file_path` is the persisted reference, `buffer`
/// is the decoded, resampled audio re-loaded from it after deserializing (see
/// `Song::load_from_file`). Trim/fade fields all default to `0`, meaning "no trim/fade yet" — a
/// freshly recorded/imported clip, and any clip from a song file saved before trimming existed,
/// plays its whole decoded buffer with no ramps, exactly as before this feature existed.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AudioClip {
    pub start_tick: usize,
    pub file_path: String,
    /// Linear gain applied to this clip's playback. 1.0 is unity.
    #[serde(default = "default_clip_gain")]
    pub gain: f32,
    /// Trim: the decoded (and, if `warp_markers`/`pitch_corrections` are set, already
    /// warped/pitch-corrected — see `load`) buffer's frame index playback starts from (head trim).
    /// Real sample frames, not ticks — trim itself stays non-tempo-elastic even though the
    /// underlying buffer might now be. `0` plays from the buffer's own start, same as before
    /// trimming existed.
    #[serde(default)]
    pub source_start_frame: usize,
    /// Trim: this clip's on-timeline duration in ticks. `0` is a sentinel meaning "play to the end
    /// of the decoded buffer from `source_start_frame`" — see `effective_length_ticks`.
    #[serde(default)]
    pub length_ticks: usize,
    /// Ticks (from `start_tick`) over which this clip's output ramps up from silence, the audio-clip
    /// counterpart of `Region::fade_in_ticks`. `0` means no fade.
    #[serde(default)]
    pub fade_in_ticks: usize,
    /// Ticks (into the end of `effective_length_ticks`) over which this clip's output ramps down to
    /// silence, the audio-clip counterpart of `Region::fade_out_ticks`. `0` means no fade.
    #[serde(default)]
    pub fade_out_ticks: usize,
    /// Flex Pitch: per-note pitch retargeting, applied by `load` before `warp_markers` (so a
    /// correction's frame range always indexes into the *unwarped* recording, regardless of
    /// whether the clip is also time-stretched) — see `pitch::apply_pitch_corrections`. Empty
    /// means untouched, same sentinel convention as every other field here.
    #[serde(default)]
    pub pitch_corrections: Vec<PitchCorrection>,
    /// Flex Time: a piecewise source-time-to-output-time map, applied by `load` last — see
    /// `stretch::warp_buffer`. Fewer than 2 markers (including the empty default) means
    /// untouched — one point alone doesn't define a span to stretch.
    #[serde(default)]
    pub warp_markers: Vec<WarpMarker>,
    #[serde(skip)]
    pub buffer: Option<Arc<SampleBuffer>>,
    #[serde(skip)]
    pub load_error: Option<String>,
}

impl AudioClip {
    /// A new clip referencing `file_path` at `start_tick`, unloaded until `load` is called. No
    /// trim/fade until explicitly set.
    pub fn new(start_tick: usize, file_path: impl Into<String>) -> Self {
        Self {
            start_tick,
            file_path: file_path.into(),
            gain: default_clip_gain(),
            source_start_frame: 0,
            length_ticks: 0,
            fade_in_ticks: 0,
            fade_out_ticks: 0,
            pitch_corrections: Vec::new(),
            warp_markers: Vec::new(),
            buffer: None,
            load_error: None,
        }
    }

    /// This clip's on-timeline duration in ticks if it were untrimmed at the tail — the cap
    /// `effective_length_ticks` falls back to when `length_ticks` is the "no trim" sentinel, and
    /// what a tail-trim (right-edge) drag can extend back out to. Converted from the buffer's
    /// remaining real-time duration (from `source_start_frame` onward) via `ticks_per_second` (see
    /// `audio::ticks_per_second`) — the caller's job, not this module's, to avoid `model.rs`
    /// depending on `audio.rs` (see `audio::arrangement_length_ticks`'s doc comment on the same
    /// tempo-at-start-tick approximation).
    pub fn full_length_ticks(&self, ticks_per_second: f64) -> usize {
        let Some(buffer) = &self.buffer else {
            return 0;
        };
        let remaining_frames = buffer.mono.len().saturating_sub(self.source_start_frame);
        let duration_seconds = remaining_frames as f64 / buffer.sample_rate.max(1) as f64;
        (duration_seconds * ticks_per_second).ceil() as usize
    }

    /// This clip's on-timeline duration in ticks. `length_ticks` of `0` (untrimmed clips, and song
    /// files saved before trim existed) falls back to `full_length_ticks`.
    pub fn effective_length_ticks(&self, ticks_per_second: f64) -> usize {
        if self.length_ticks > 0 {
            self.length_ticks
        } else {
            self.full_length_ticks(ticks_per_second)
        }
    }

    /// Loads `file_path`, resampled to `target_sample_rate` — see `Lane::load_sample`, the
    /// equivalent for step-grid one-shot samples. If `pitch_corrections`/`warp_markers` are set,
    /// the decoded buffer is pitch-corrected then time-warped before being stored, so every other
    /// consumer of `buffer` (playback, trim, fades, the offline bounce) sees the edited audio
    /// without needing to know Flex Time/Pitch exist — see `pitch::apply_pitch_corrections` and
    /// `stretch::warp_buffer`'s doc comments for why this "bake once at load time" approach was
    /// chosen over stretching/shifting live in the real-time engine.
    pub fn load(&mut self, target_sample_rate: u32) {
        let path = self.file_path.trim();
        if path.is_empty() {
            self.buffer = None;
            self.load_error = None;
            return;
        }
        match SampleBuffer::load_wav_resampled(std::path::Path::new(path), target_sample_rate) {
            Ok(mut buffer) => {
                if !self.pitch_corrections.is_empty() {
                    buffer = pitch::apply_pitch_corrections(&buffer, &self.pitch_corrections);
                }
                if self.warp_markers.len() >= 2 {
                    buffer = stretch::warp_buffer(&buffer, &self.warp_markers);
                }
                self.buffer = Some(Arc::new(buffer));
                self.load_error = None;
            }
            Err(err) => {
                self.buffer = None;
                self.load_error = Some(format!("{err:#}"));
            }
        }
    }
}

fn default_take_gain() -> f32 {
    1.0
}

/// One recorded pass inside a `TakeFolder` — the same "persisted reference plus decoded buffer"
/// split as `AudioClip`/`Lane`'s sample fields, but with no `start_tick`/gain/trim of its own:
/// every take in a folder starts at the folder's own `start_tick`, and which stretch of which
/// take is actually heard is entirely the `comp`'s job (see `TakeFolder`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Take {
    pub file_path: String,
    #[serde(skip)]
    pub buffer: Option<Arc<SampleBuffer>>,
    #[serde(skip)]
    pub load_error: Option<String>,
}

impl Take {
    pub fn new(file_path: impl Into<String>) -> Self {
        Self {
            file_path: file_path.into(),
            buffer: None,
            load_error: None,
        }
    }

    /// Loads `file_path`, resampled to `target_sample_rate` — see `AudioClip::load`, the
    /// equivalent for a plain (non-take) recorded/imported clip.
    pub fn load(&mut self, target_sample_rate: u32) {
        let path = self.file_path.trim();
        if path.is_empty() {
            self.buffer = None;
            self.load_error = None;
            return;
        }
        match SampleBuffer::load_wav_resampled(std::path::Path::new(path), target_sample_rate) {
            Ok(buffer) => {
                self.buffer = Some(Arc::new(buffer));
                self.load_error = None;
            }
            Err(err) => {
                self.buffer = None;
                self.load_error = Some(format!("{err:#}"));
            }
        }
    }
}

/// Which take is heard for one stretch of a `TakeFolder`'s span — `start_tick`/`end_tick` are
/// relative to the folder's own `start_tick`, not absolute song ticks (so moving the folder moves
/// every segment with it for free). A folder's `comp` is a sequence of these covering its whole
/// span with no gaps or overlaps — see `TakeFolder::set_active_take`, the only way `comp` is
/// mutated in this phase (segment-level comp-dragging is a later feature).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompSegment {
    pub start_tick: usize,
    pub end_tick: usize,
    pub take_index: usize,
}

/// A Logic-style "Take Folder": several recorded passes of the same performance
/// (`takes`), all starting at the same absolute `start_tick` and covering the same on-timeline
/// `length_ticks`, with `comp` choosing which take is heard for each stretch of that span. Lives in
/// its own `Track::take_folders` list, parallel to `Track::audio_clips` — comping only applies to
/// *recorded* material (see `main.rs`'s `finish_recording`); a plain imported/dragged-in file is
/// never grouped into one of these.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TakeFolder {
    pub start_tick: usize,
    /// Frozen at the first take's own recorded duration when the folder is created — later takes
    /// recorded into the same folder don't change it (see `main.rs`'s `finish_recording`), the same
    /// way a `Region`'s `loop_length_steps` doesn't grow just because its content did.
    pub length_ticks: usize,
    /// Linear gain applied to whichever take/segment is currently playing. 1.0 is unity.
    #[serde(default = "default_take_gain")]
    pub gain: f32,
    pub takes: Vec<Take>,
    pub comp: Vec<CompSegment>,
}

impl TakeFolder {
    /// A new folder with a single take occupying its whole span — the state right after the first
    /// recording into an empty spot on an armed track.
    pub fn new(start_tick: usize, length_ticks: usize, file_path: impl Into<String>) -> Self {
        Self {
            start_tick,
            length_ticks,
            gain: default_take_gain(),
            takes: vec![Take::new(file_path)],
            comp: vec![CompSegment {
                start_tick: 0,
                end_tick: length_ticks,
                take_index: 0,
            }],
        }
    }

    /// Adds a new take (a re-recording from the exact same `start_tick`, see `finish_recording`)
    /// and immediately comps the whole folder to it — "the take you just recorded is what you
    /// hear," until comped otherwise. Returns the new take's index.
    pub fn add_take_and_activate(&mut self, file_path: impl Into<String>) -> usize {
        self.takes.push(Take::new(file_path));
        let new_index = self.takes.len() - 1;
        self.set_active_take(new_index);
        new_index
    }

    /// Replaces `comp` with a single segment spanning the whole folder, assigned to `take_index` —
    /// the "pick one whole take" operation the right-click context menu exposes (see
    /// `assign_take_to_range` for the segment-level "quick-swipe" comp-editor's own mutation).
    /// Out-of-bounds `take_index` is a no-op.
    pub fn set_active_take(&mut self, take_index: usize) {
        if take_index >= self.takes.len() {
            return;
        }
        self.comp = vec![CompSegment {
            start_tick: 0,
            end_tick: self.length_ticks,
            take_index,
        }];
    }

    /// Reassigns `[start_tick, end_tick)` (clamped to the folder's own span) to `take_index`,
    /// splitting/trimming whichever existing segments it overlaps so `comp` stays sorted, gapless,
    /// and non-overlapping — the "quick-swipe" comp-editor's mutation (`main.rs`'s take-folder
    /// editor window), called live as the user drags across a take's lane. `start_tick >= end_tick`
    /// (a zero-width or backwards range) or an out-of-bounds `take_index` is a no-op.
    pub fn assign_take_to_range(&mut self, take_index: usize, start_tick: usize, end_tick: usize) {
        let start_tick = start_tick.min(self.length_ticks);
        let end_tick = end_tick.min(self.length_ticks);
        if start_tick >= end_tick || take_index >= self.takes.len() {
            return;
        }
        let mut rebuilt = Vec::with_capacity(self.comp.len() + 2);
        for segment in &self.comp {
            // The portion of this existing segment, if any, that survives before/after the
            // overwritten range — the range itself is pushed once, separately, below.
            if segment.start_tick < start_tick {
                rebuilt.push(CompSegment {
                    start_tick: segment.start_tick,
                    end_tick: segment.end_tick.min(start_tick),
                    take_index: segment.take_index,
                });
            }
            if segment.end_tick > end_tick {
                rebuilt.push(CompSegment {
                    start_tick: segment.start_tick.max(end_tick),
                    end_tick: segment.end_tick,
                    take_index: segment.take_index,
                });
            }
        }
        rebuilt.push(CompSegment {
            start_tick,
            end_tick,
            take_index,
        });
        rebuilt.retain(|s| s.start_tick < s.end_tick);
        rebuilt.sort_by_key(|s| s.start_tick);
        // Merge neighbors that ended up pointing at the same take (overwriting into the middle of
        // a same-take segment, or two neighbors landing on the same take after a drag) so `comp`
        // doesn't accumulate meaningless same-take splits.
        let mut merged: Vec<CompSegment> = Vec::with_capacity(rebuilt.len());
        for segment in rebuilt {
            match merged.last_mut() {
                Some(last) if last.take_index == segment.take_index && last.end_tick == segment.start_tick => {
                    last.end_tick = segment.end_tick;
                }
                _ => merged.push(segment),
            }
        }
        self.comp = merged;
    }

    /// Loads every take's buffer, resampled to `target_sample_rate` — see `Song::reload_samples`.
    pub fn load_all(&mut self, target_sample_rate: u32) {
        for take in &mut self.takes {
            if !take.file_path.trim().is_empty() {
                take.load(target_sample_rate);
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Track {
    pub name: String,
    pub midi_channel: u8,
    pub muted: bool,
    /// When any track in the song has `solo` set, only soloed tracks play (muted or not) and every
    /// other track goes silent — the usual DAW solo behavior. `#[serde(default)]` so song files
    /// saved before this field existed still load with nothing soloed (unchanged behavior).
    #[serde(default)]
    pub solo: bool,
    /// Fixed for the track's lifetime — see `TrackKind`. Determines which `RegionContent` variant
    /// this track's own `regions` use.
    pub kind: TrackKind,
    /// Length (in ticks) stamped onto a new note from a plain click (no
    /// drag) in the piano roll — dragging draws an explicit length instead.
    pub default_note_length_ticks: usize,
    /// This track's insert effect chain, processed in order (element 0 first, feeding into
    /// element 1, and so on). Empty means no effects loaded. `#[serde(default)]` so song files
    /// saved before this field existed (or before it was a chain rather than a single effect)
    /// still load, with no effects.
    #[serde(default)]
    pub effects: Vec<TrackEffectConfig>,
    /// Linear gain applied to this track's mix contribution (post effects chain, pre master bus).
    /// 1.0 is unity. `#[serde(default = "default_track_volume")]` so song files saved before this
    /// field existed still load at full volume rather than silent.
    #[serde(default = "default_track_volume")]
    pub volume: f32,
    /// Stereo position of this track's mix contribution, applied (as an equal-power gain split)
    /// after its effects chain, the same point `volume` is applied — -1.0 is hard left, 0.0 is
    /// center, 1.0 is hard right. `#[serde(default)]` so song files saved before this field
    /// existed still load centered.
    #[serde(default)]
    pub pan: f32,
    /// Linear gain applied to incoming audio while this track is armed and recording, before it's
    /// written to the take's WAV file — a preamp trim, distinct from `volume` (the mix fader
    /// applied on playback). Only meaningful for `TrackKind::Audio`. 1.0 is unity.
    /// `#[serde(default = "default_input_gain")]` so song files saved before this field existed
    /// still load at unity input gain.
    #[serde(default = "default_input_gain")]
    pub input_gain: f32,
    /// This track's built-in synth voice settings (waveform + attack/decay). `#[serde(default)]`
    /// so song files saved before this field existed still load, defaulting to the original
    /// sine-with-instant-attack sound.
    #[serde(default)]
    pub synth: SynthParams,
    /// Which synth engine renders this track's notes — `Simple` (`synth`, above) or `Trine` (`trine`,
    /// below). `#[serde(default)]` so song files saved before this engine existed still load,
    /// defaulting every track to `Simple` (unchanged behavior).
    #[serde(default)]
    pub synth_engine: SynthEngine,
    /// This track's Trine-engine settings, only used when `synth_engine == SynthEngine::Trine`.
    /// `#[serde(default)]` for the same reason as `synth_engine`.
    #[serde(default)]
    pub trine: TrineParams,
    /// This track's Wave-engine settings, only used when `synth_engine == SynthEngine::Wave`.
    /// `#[serde(default)]` for the same reason as `synth_engine`.
    #[serde(default)]
    pub wave: WaveParams,
    /// Recorded/imported audio clips, only used when `kind == TrackKind::Audio`. `#[serde(default)]`
    /// so song files saved before audio tracks existed still load, with no clips.
    #[serde(default)]
    pub audio_clips: Vec<AudioClip>,
    /// Take Folders (see `TakeFolder`) — recorded audio grouped for comping, kept separate from
    /// `audio_clips` since only recordings (not imports) ever land here. `#[serde(default)]` so
    /// song files saved before comping existed still load, with none.
    #[serde(default)]
    pub take_folders: Vec<TakeFolder>,
    /// This track's own independently-positioned regions (see `Region`), only used when
    /// `kind` is `StepGrid`/`PianoRoll` — the melodic/drum-grid equivalent of `audio_clips`.
    /// `#[serde(default)]` so song files saved before regions existed still load, with no regions
    /// (older formats route through `migrate_patterns_song` instead — see that function).
    #[serde(default)]
    pub regions: Vec<Region>,
    /// Linear send level (0.0 = no send) to each of `Song.sends`, index-aligned with that list and
    /// tapped post-fader/pan (the same point `volume`/`pan` apply) — see `SendBus`'s doc comment.
    /// Kept in sync with `Song.sends`'s length by `Song::add_send`/`remove_send`/`add_track`, not
    /// by this track alone. `#[serde(default)]` so song files saved before sends existed still
    /// load with none.
    #[serde(default)]
    pub send_levels: Vec<f32>,
    /// Where this track's post-fader/pan signal sums into — straight to the master bus (the
    /// original, still-default behavior) or into one of `Song.submixes` instead, exclusively (not
    /// in addition to master; the submix's own summed output is what reaches master). Distinct
    /// from `send_levels`: a send is a parallel tap, this replaces the track's direct contribution.
    /// `#[serde(default)]` so song files saved before submixes existed still load routed to master.
    #[serde(default)]
    pub output: TrackOutput,
    /// Track-wide automation "rides" — same `AutomationLane`/`AutomationTarget` shape as a
    /// `Region`'s own `automation`, but not tied to any one region's on-timeline span: a point's
    /// `tick` here is an *absolute* song tick (this track's placement in the overall arrangement),
    /// not region-local. Lets a lane ride a parameter across the whole song (or wherever this
    /// track has no active region at all) rather than only within one region's span. Where both a
    /// track-wide lane and an active region's own lane target the same parameter at the same tick,
    /// the region's lane wins — see `audio::collect_automation`. `#[serde(default)]` so song files
    /// saved before this field existed still load with none.
    #[serde(default)]
    pub automation: Vec<AutomationLane>,
    /// Whether this track is currently frozen: its own live synthesis (notes/steps/audio clips)
    /// and insert-effect chain (`effects`) are bypassed in the real-time engine in favor of
    /// playing back `frozen_clip`'s already-baked audio — volume/pan/mute/solo/sends/output stay
    /// live even while frozen, since those apply *after* a track's own chain either way. See
    /// `audio::render_track_to_buffer`. `#[serde(default)]` so song files saved before freeze
    /// existed still load unfrozen.
    #[serde(default)]
    pub frozen: bool,
    /// The baked audio a frozen track plays instead of live synthesis — an ordinary `AudioClip`
    /// (same file-backed reload-on-load machinery every other clip uses), spanning this track's
    /// whole arrangement from tick 0 as of the moment it was frozen. `None` whenever `frozen` is
    /// false. `#[serde(default)]` for the same reason as `frozen`.
    #[serde(default)]
    pub frozen_clip: Option<AudioClip>,
    /// Session View clip slots — see `SessionClip`. Slot index is aligned across every track
    /// (slot 0 in every track's own `Vec` is the grid's visual row 0); a track with fewer slots
    /// than another simply has empty rows past its own length. `#[serde(default)]` so song files
    /// saved before Session View existed still load with none.
    #[serde(default)]
    pub session_clips: Vec<Option<SessionClip>>,
    /// Pending Session View launch/stop clicks, index-aligned with `session_clips` — see
    /// `SessionLaunchRequest`. Transient UI state, not song data: `#[serde(skip)]` so it's never
    /// written to a song file and always starts empty on load.
    #[serde(skip)]
    pub session_launch_requests: Vec<SessionLaunchRequest>,
}

/// See `Track::output`. `Submix(index)` indexes into `Song.submixes`; kept valid by
/// `Song::remove_submix`, which resets any track pointing at the removed submix back to `Master`
/// and shifts indices of tracks pointing past it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum TrackOutput {
    #[default]
    Master,
    Submix(usize),
}

impl Track {
    /// A track ready for melodic (piano-roll) editing.
    pub fn new_piano_roll(name: impl Into<String>, midi_channel: u8) -> Self {
        Self {
            name: name.into(),
            midi_channel,
            muted: false,
            solo: false,
            kind: TrackKind::PianoRoll,
            default_note_length_ticks: 4 * TICKS_PER_STEP,
            effects: Vec::new(),
            volume: 1.0,
            pan: 0.0,
            input_gain: 1.0,
            synth: SynthParams::default(),
            synth_engine: SynthEngine::default(),
            trine: TrineParams::default(),
            wave: WaveParams::default(),
            audio_clips: Vec::new(),
            take_folders: Vec::new(),
            regions: Vec::new(),
            send_levels: Vec::new(),
            output: TrackOutput::Master,
            automation: Vec::new(),
            frozen: false,
            frozen_clip: None,
            session_clips: Vec::new(),
            session_launch_requests: Vec::new(),
        }
    }

    /// A track ready for step-grid (drum-machine) editing. See `new_piano_roll` for the rest.
    pub fn new_step_grid(name: impl Into<String>, midi_channel: u8) -> Self {
        Self {
            name: name.into(),
            midi_channel,
            muted: false,
            solo: false,
            kind: TrackKind::StepGrid,
            default_note_length_ticks: 4 * TICKS_PER_STEP,
            effects: Vec::new(),
            volume: 1.0,
            pan: 0.0,
            input_gain: 1.0,
            synth: SynthParams::default(),
            synth_engine: SynthEngine::default(),
            trine: TrineParams::default(),
            wave: WaveParams::default(),
            audio_clips: Vec::new(),
            take_folders: Vec::new(),
            regions: Vec::new(),
            send_levels: Vec::new(),
            output: TrackOutput::Master,
            automation: Vec::new(),
            frozen: false,
            frozen_clip: None,
            session_clips: Vec::new(),
            session_launch_requests: Vec::new(),
        }
    }

    /// A track ready for recorded/imported audio clips. See `new_piano_roll` for the rest.
    pub fn new_audio(name: impl Into<String>, midi_channel: u8) -> Self {
        Self {
            name: name.into(),
            midi_channel,
            muted: false,
            solo: false,
            kind: TrackKind::Audio,
            default_note_length_ticks: 4 * TICKS_PER_STEP,
            effects: Vec::new(),
            volume: 1.0,
            pan: 0.0,
            input_gain: 1.0,
            synth: SynthParams::default(),
            synth_engine: SynthEngine::default(),
            trine: TrineParams::default(),
            wave: WaveParams::default(),
            audio_clips: Vec::new(),
            take_folders: Vec::new(),
            regions: Vec::new(),
            send_levels: Vec::new(),
            output: TrackOutput::Master,
            automation: Vec::new(),
            frozen: false,
            frozen_clip: None,
            session_clips: Vec::new(),
            session_launch_requests: Vec::new(),
        }
    }

    /// Appends a new, empty region at `start_step`, sized to one bar (`steps_per_bar`, see
    /// `Song::steps_per_bar`) — the Playlist's "click empty space on a track's row" gesture. A new
    /// `StepGrid` region's lane layout (names/pitches, no step data) is copied from this track's
    /// most recently added region, if it has one, so a second region on the same track doesn't
    /// start bare — mirroring the old `Song::add_pattern`'s template-copy behavior, just scoped to
    /// one track. Returns the new region's index.
    pub fn add_region(&mut self, start_step: usize, steps_per_bar: usize) -> usize {
        let length_steps = steps_per_bar;
        let content = match self.kind {
            TrackKind::PianoRoll => RegionContent::PianoRoll(Vec::new()),
            TrackKind::StepGrid => {
                let lanes = match self.regions.last() {
                    Some(Region {
                        content: RegionContent::StepGrid(template_lanes),
                        ..
                    }) => template_lanes
                        .iter()
                        .map(|lane| {
                            let mut new_lane =
                                Lane::new(lane.name.clone(), lane.pitch, length_steps);
                            new_lane.sample_path = lane.sample_path.clone();
                            new_lane
                        })
                        .collect(),
                    _ => Vec::new(),
                };
                RegionContent::StepGrid(lanes)
            }
            TrackKind::Audio => unreachable!("Audio tracks use audio_clips, not regions"),
        };
        self.regions.push(Region {
            name: format!("Region {}", self.regions.len() + 1),
            start_tick: start_step * TICKS_PER_STEP,
            content_length_steps: length_steps,
            loop_length_steps: length_steps,
            content,
            fade_in_ticks: 0,
            fade_out_ticks: 0,
            automation: Vec::new(),
        });
        self.regions.len() - 1
    }

    /// The tick just past the end of this track's furthest-out region — `0` if it has none. Used
    /// by Session View's "Send to Playlist" action (`SessionClip::to_region`) to place a new
    /// region after everything already on this track's Playlist, rather than needing an
    /// interactive placement UI.
    pub fn end_of_regions_tick(&self) -> usize {
        self.regions
            .iter()
            .map(|region| region.start_tick + region.loop_length_steps * TICKS_PER_STEP)
            .max()
            .unwrap_or(0)
    }

    /// Appends a new lane to every `StepGrid` region on this track, keeping lanes index-aligned
    /// across this track's own regions the same way the old `Song::add_lane` kept them aligned
    /// across every pattern — a lane added here shows up (empty) in every existing region on this
    /// track, not just the one currently being edited.
    pub fn add_lane(&mut self, name: impl Into<String>, pitch: u8) {
        let name = name.into();
        for region in &mut self.regions {
            if let RegionContent::StepGrid(lanes) = &mut region.content {
                lanes.push(Lane::new(name.clone(), pitch, region.content_length_steps));
            }
        }
    }

    /// Removes `lane_index` from every `StepGrid` region on this track that has a lane at that
    /// index, mirroring `add_lane`'s all-regions scope.
    pub fn remove_lane(&mut self, lane_index: usize) {
        for region in &mut self.regions {
            if let RegionContent::StepGrid(lanes) = &mut region.content {
                if lane_index < lanes.len() {
                    lanes.remove(lane_index);
                }
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Song {
    pub name: String,
    /// The tempo from the start of the song up to `tempo_map`'s first point (if any) — the
    /// song's original single global tempo, still the only tempo field every one-shot
    /// read/write (the transport LCD's field, MCP `set_bpm`/`get_playback_state`, tap tempo,
    /// detect tempo, MIDI import's "apply tempo" checkbox) needs to know about. `tempo_map`
    /// only ever represents *changes* after this starting point — see `bpm_at`.
    pub bpm: f32,
    /// Tempo-change points after the start (see `bpm`), each holding constant until the next
    /// one — a step function, not a smooth ramp, edited via the Playlist's Tempo Track
    /// (`tempo_track_ui` in `main.rs`). Kept sorted by `tick` ascending by every editing method
    /// here (`set_tempo_at`/`remove_tempo_point`) rather than by the UI, the same convention
    /// `Song::sends`/`submixes` use for their own index-stability guarantees.
    /// `#[serde(default)]` so song files saved before tempo maps existed still load with a flat
    /// `bpm` and no changes.
    #[serde(default)]
    pub tempo_map: Vec<TempoPoint>,
    pub tracks: Vec<Track>,
    /// Monotonically increasing counter for `Note::id` allocation, shared
    /// across every track's piano roll so ids never collide.
    pub next_note_id: u64,
    /// The master bus's insert effect chain — same shape and processing order as
    /// `Track::effects`, just with no track attached. `#[serde(default)]` so song files saved
    /// before this was a chain (a single CLAP plugin path/params pair) still load; see
    /// `load_from_file`'s old-shape migration.
    #[serde(default)]
    pub master_effects: Vec<TrackEffectConfig>,
    /// Project-level library of imported CLAP plugins, each tagged with a mnemonic name so it can
    /// be picked by name — from the master Plugins window or a track's "+ Add Effect" menu —
    /// instead of by its (often long) filesystem path. Entries are just name/path pairs, not live
    /// plugin state: loading one still goes through the same `plugin_host::load_and_activate` call
    /// as typing the path directly. `#[serde(default)]` so song files saved before this field
    /// existed still load, with an empty library.
    #[serde(default)]
    pub plugins: Vec<ProjectPlugin>,
    /// Named, reusable synth patches, editable from any track's synth window and assignable to
    /// any track by picking one from the list — loading a preset copies its `params` into that
    /// track's `synth`, it isn't a live link back to this entry. `#[serde(default)]` so song files
    /// saved before this field existed still load, with an empty library.
    #[serde(default)]
    pub synth_presets: Vec<SynthPreset>,
    /// Time signature numerator (beats per bar). `#[serde(default = "...")]` so song files saved
    /// before this field existed still load as 4/4, the app's original fixed assumption.
    #[serde(default = "default_time_signature_numerator")]
    pub time_signature_numerator: u8,
    /// Time signature denominator (the beat's note value). Only 1/2/4/8/16 are reachable from the
    /// UI (see `transport_lcd_ui` in `main.rs`) so it always evenly divides `STEPS_PER_WHOLE_NOTE`
    /// — `steps_per_beat`/`steps_per_bar` assume that and don't re-validate it.
    #[serde(default = "default_time_signature_denominator")]
    pub time_signature_denominator: u8,
    /// Aux send buses (see `SendBus`) every track can feed via its own `Track::send_levels`.
    /// `#[serde(default)]` so song files saved before sends existed still load with none.
    #[serde(default)]
    pub sends: Vec<SendBus>,
    /// Submix buses (see `SubmixBus`) a track can route its output into instead of straight to
    /// master via `Track::output` — the "Track Stack"/alternate-output-routing mechanism.
    /// `#[serde(default)]` so song files saved before submixes existed still load with none (every
    /// track's `Track::output` also defaults to `Master`, so nothing changes for them).
    #[serde(default)]
    pub submixes: Vec<SubmixBus>,
    /// Session View scene rows (see `Scene`) — index-aligned to the grid's slot rows the way
    /// `Track::session_clips` already are, but grown lazily via `ensure_scene` rather than an
    /// explicit add/remove pair like `sends`/`submixes`, since a grid row isn't itself a managed
    /// collection (it just exists wherever a track has a clip). Shorter than the visible row count
    /// until a scene is actually named or given a tempo. `#[serde(default)]` so song files saved
    /// before scenes existed still load with none.
    #[serde(default)]
    pub scenes: Vec<Scene>,
}

fn default_time_signature_numerator() -> u8 {
    4
}

fn default_time_signature_denominator() -> u8 {
    4
}

/// One tempo-change point on `Song::tempo_map` — an absolute song tick (not region-local, unlike
/// `AutomationPoint::tick`, since a tempo map isn't scoped to one region/track) and the BPM to
/// hold from that tick until the next point.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TempoPoint {
    pub tick: usize,
    pub bpm: f32,
}

/// One row of `Song::scenes` — a Session View grid row's own name and, optionally, a tempo to
/// recall when that row's scene is launched (Ableton's "stateful scene" concept). Launching a
/// scene with `tempo: Some(bpm)` inserts a real `TempoPoint` at the launch-quantize boundary tick
/// (see `session_view_ui::launch_scene`), so the change is tick-accurate and survives a rewind
/// past that point the same way any other tempo-map point does. There's no time-signature
/// counterpart yet — `Song::time_signature_numerator`/`denominator` are single global fields with
/// no tick-indexed map to recall into (unlike `tempo_map`), so scene-based time-signature recall
/// is left for a future item once/if that map exists.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Scene {
    pub name: String,
    pub tempo: Option<f32>,
}

/// One imported CLAP plugin in the project library (`Song::plugins`) — a mnemonic name paired
/// with the plugin's filesystem path.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProjectPlugin {
    pub name: String,
    pub path: String,
}

/// An aux send bus (`Song::sends`): a named effect chain (same shape/processing order as
/// `Track::effects`) that every track can feed at an independent level via its own
/// `Track::send_levels`, tapped post-fader/pan like a classic send — see `audio.rs`'s mixdown for
/// where that tap happens. Index-aligned with `Song::sends`; `Song::add_send`/`remove_send` keep
/// every track's `send_levels` the same length as this list.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SendBus {
    pub name: String,
    #[serde(default)]
    pub effects: Vec<TrackEffectConfig>,
}

/// A summing bus a track can route its output into via `Track::output` (instead of straight to
/// master) — Logic's "Track Stack" made of one shared fader plus one shared insert chain, so N
/// tracks that used to each carry their own reverb/compressor can share one instance instead.
/// Unlike `SendBus`, this has its own `volume`/`muted`/`solo` since it stands in for its member
/// tracks' direct contribution to the mix rather than being a parallel tap; it has no `pan` — kept
/// minimal, matching this app's per-track (not per-group) stereo-field model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SubmixBus {
    pub name: String,
    #[serde(default)]
    pub effects: Vec<TrackEffectConfig>,
    #[serde(default = "default_track_volume")]
    pub volume: f32,
    /// Equal-power pan, same law and range as `Track::pan` — applied after this submix's own
    /// chain+volume, the same point a track's own pan pot sits (see `equal_power_pan_gains`).
    #[serde(default)]
    pub pan: f32,
    #[serde(default)]
    pub muted: bool,
    #[serde(default)]
    pub solo: bool,
}

/// A step is fixed at a sixteenth note regardless of time signature (see `TICKS_PER_STEP`) — this
/// is that resolution expressed as "steps per whole note", the fixed point `Song::steps_per_beat`
/// divides against to turn the denominator into a step count.
const STEPS_PER_WHOLE_NOTE: usize = 16;

/// A named, reusable snapshot of one track's synth settings for whichever `engine` it was saved
/// from, storable either inside a `Song` (see `Song::synth_presets`) or as its own standalone file
/// (`save_to_file`/`load_from_file`) for reuse across songs. Only the field matching `engine` is
/// populated; the other two stay at their `Default`, unused. `engine`/`trine`/`wave` are
/// `#[serde(default)]` so pre-existing preset files (saved before `Trine`/`Wave` presets existed,
/// always `Simple`) still load.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SynthPreset {
    pub name: String,
    #[serde(default)]
    pub engine: SynthEngine,
    pub params: SynthParams,
    #[serde(default)]
    pub trine: Option<TrineParams>,
    #[serde(default)]
    pub wave: Option<WaveParams>,
}

impl SynthPreset {
    /// Serializes this preset as pretty-printed JSON and writes it to `path`.
    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        let json =
            serde_json::to_string_pretty(self).context("failed to serialize synth preset")?;
        std::fs::write(path, json)
            .with_context(|| format!("failed to write {}", path.display()))?;
        Ok(())
    }

    /// Reads and deserializes a synth preset previously written by `save_to_file`.
    pub fn load_from_file(path: &Path) -> Result<Self> {
        let json = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        serde_json::from_str(&json).context("failed to parse synth preset file")
    }
}

impl Song {
    /// Steps (fixed sixteenth notes) per beat, where "beat" means the time signature's own
    /// denominator note value — e.g. an eighth-note beat (6/8) is 2 steps, a quarter-note beat
    /// (4/4) is 4 steps. Unrelated to `audio::ticks_per_second`'s own beat-based tick rate, which
    /// stays pinned to the quarter note: BPM conventionally names the quarter-note pulse
    /// regardless of the declared meter, so playback speed doesn't change with this.
    pub fn steps_per_beat(&self) -> usize {
        STEPS_PER_WHOLE_NOTE / self.time_signature_denominator.max(1) as usize
    }

    /// Steps (fixed sixteenth notes) per bar, i.e. `time_signature_numerator` beats of
    /// `steps_per_beat` each — the unit bar lines are drawn on and new regions/imports default to.
    pub fn steps_per_bar(&self) -> usize {
        self.time_signature_numerator as usize * self.steps_per_beat()
    }

    /// The tempo in effect at `tick` — `bpm` until `tempo_map`'s first point at or before `tick`,
    /// then that point's `bpm`, held constant until the next point (a step function; see
    /// `tempo_map`'s doc comment). `tempo_map` is assumed sorted by `tick` ascending, an invariant
    /// `set_tempo_at`/`remove_tempo_point` maintain.
    pub fn bpm_at(&self, tick: usize) -> f32 {
        self.tempo_map
            .iter()
            .rev()
            .find(|point| point.tick <= tick)
            .map_or(self.bpm, |point| point.bpm)
    }

    /// Inserts a new tempo-change point at `tick`, or updates its `bpm` if one already exists
    /// there, keeping `tempo_map` sorted by `tick`.
    pub fn set_tempo_at(&mut self, tick: usize, bpm: f32) {
        match self.tempo_map.iter_mut().find(|point| point.tick == tick) {
            Some(point) => point.bpm = bpm,
            None => {
                self.tempo_map.push(TempoPoint { tick, bpm });
                self.tempo_map.sort_by_key(|point| point.tick);
            }
        }
    }

    /// Removes the tempo-change point at index `index` into `tempo_map`, if any.
    pub fn remove_tempo_point(&mut self, index: usize) {
        if index < self.tempo_map.len() {
            self.tempo_map.remove(index);
        }
    }

    /// A starter song so the sequencer UI has something to show and edit.
    pub fn demo() -> Self {
        let mut drums = Track::new_step_grid("Drums", 10);
        let drums_region = drums.add_region(0, STEPS_PER_WHOLE_NOTE);
        drums.add_lane("Kick", 36);
        drums.add_lane("Snare", 38);
        drums.add_lane("Closed Hat", 42);
        drums.add_lane("Open Hat", 46);
        drums.add_lane("Clap", 39);
        drums.add_lane("Rim", 37);
        drums.add_lane("Low Tom", 45);
        drums.add_lane("Crash", 49);
        drums.add_lane("Mid Tom", 47);
        drums.add_lane("High Tom", 50);
        drums.add_lane("Ride", 51);
        drums.add_lane("Cowbell", 56);
        drums.add_lane("Shaker", 70);
        drums.add_lane("Tambourine", 54);
        drums.add_lane("Woodblock", 76);
        drums.add_lane("Triangle", 81);
        if let RegionContent::StepGrid(lanes) = &mut drums.regions[drums_region].content {
            for step in [0, 4, 8, 12] {
                lanes[0].set_step(step, 110);
            }
            for step in [4, 12] {
                lanes[1].set_step(step, 100);
            }
            for step in [0, 2, 4, 6, 8, 10, 12, 14] {
                lanes[2].set_step(step, 70);
            }
            for step in [4, 12] {
                lanes[4].set_step(step, 90);
            }
            for step in [2, 10] {
                lanes[5].set_step(step, 60);
            }
            for step in [14, 15] {
                lanes[6].set_step(step, 85);
            }
            lanes[7].set_step(0, 100);
            lanes[9].set_step(12, 85); // High Tom
            lanes[8].set_step(13, 80); // Mid Tom
            for step in [1, 9] {
                lanes[10].set_step(step, 55); // Ride
            }
            lanes[11].set_step(8, 75); // Cowbell
            for step in 0..16 {
                lanes[12].set_step(step, 45); // Shaker
            }
            for step in [2, 6, 10, 14] {
                lanes[13].set_step(step, 65); // Tambourine
            }
            for step in [3, 11] {
                lanes[14].set_step(step, 70); // Woodblock
            }
            lanes[15].set_step(15, 55); // Triangle
        }

        let mut bass = Track {
            synth: SynthParams {
                waveform: SynthWaveform::Saw,
                ..SynthParams::default()
            },
            ..Track::new_piano_roll("Bass", 1)
        };
        let mut next_note_id = 0u64;
        // The riff below spans exactly one bar (its furthest note ends at step 16), matching
        // `add_region`'s default one-bar `content_length_steps`/`loop_length_steps` — no need to
        // grow or adjust either after adding the notes.
        let bass_region = bass.add_region(0, STEPS_PER_WHOLE_NOTE);
        if let RegionContent::PianoRoll(notes) = &mut bass.regions[bass_region].content {
            let riff = [
                (36, 0, 4),  // C2
                (43, 4, 2),  // G2
                (41, 6, 2),  // F2
                (36, 8, 4),  // C2
                (38, 12, 2), // D2
                (40, 14, 2), // E2
            ];
            for (pitch, start_step, length_steps) in riff {
                add_note(
                    notes,
                    &mut next_note_id,
                    pitch,
                    start_step * TICKS_PER_STEP,
                    length_steps * TICKS_PER_STEP,
                    100,
                );
            }
        }

        Self {
            name: "New Song".to_string(),
            bpm: 120.0,
            tempo_map: Vec::new(),
            tracks: vec![drums, bass],
            next_note_id,
            // A default master-bus limiter, matching how e.g. Logic ships an Adaptive Limiter on
            // the master by default — a safety net against clipping rather than a creative choice,
            // so new songs get one without the user having to think about gain staging first.
            master_effects: vec![TrackEffectConfig::default_limiter()],
            plugins: Vec::new(),
            synth_presets: Vec::new(),
            time_signature_numerator: 4,
            time_signature_denominator: 4,
            sends: Vec::new(),
            submixes: Vec::new(),
            scenes: Vec::new(),
        }
    }

    /// Appends a new track, with a send level of 0.0 (no send) to every existing send bus so
    /// `Track::send_levels` starts index-aligned with `Song::sends`. Returns the new track's index.
    pub fn add_track(
        &mut self,
        name: impl Into<String>,
        midi_channel: u8,
        kind: TrackKind,
    ) -> usize {
        let mut track = match kind {
            TrackKind::PianoRoll => Track::new_piano_roll(name, midi_channel),
            TrackKind::StepGrid => Track::new_step_grid(name, midi_channel),
            TrackKind::Audio => Track::new_audio(name, midi_channel),
        };
        track.send_levels = vec![0.0; self.sends.len()];
        self.tracks.push(track);
        self.tracks.len() - 1
    }

    /// Removes a track and everything it owned (its own `regions`/`audio_clips` go with it —
    /// unlike the old shared-pattern model, nothing else references a track's content).
    pub fn remove_track(&mut self, index: usize) {
        self.tracks.remove(index);
    }

    /// Appends a new, empty-chain send bus and gives every existing track a 0.0 (no send) level
    /// for it, keeping every `Track::send_levels` index-aligned with `Song::sends`. Returns the
    /// new send's index.
    pub fn add_send(&mut self, name: impl Into<String>) -> usize {
        self.sends.push(SendBus { name: name.into(), effects: Vec::new() });
        for track in &mut self.tracks {
            track.send_levels.push(0.0);
        }
        self.sends.len() - 1
    }

    /// Removes a send bus and the corresponding entry from every track's `send_levels`, keeping
    /// them index-aligned with the now-shorter `Song::sends`.
    pub fn remove_send(&mut self, index: usize) {
        self.sends.remove(index);
        for track in &mut self.tracks {
            if index < track.send_levels.len() {
                track.send_levels.remove(index);
            }
        }
    }

    /// Appends a new, empty-chain, unity-volume submix bus. Returns the new submix's index.
    pub fn add_submix(&mut self, name: impl Into<String>) -> usize {
        self.submixes.push(SubmixBus {
            name: name.into(),
            effects: Vec::new(),
            volume: 1.0,
            pan: 0.0,
            muted: false,
            solo: false,
        });
        self.submixes.len() - 1
    }

    /// Removes a submix bus, routing any track that fed it back to `Master` (rather than leaving
    /// a dangling index) and shifting every other track's `TrackOutput::Submix` index down to
    /// stay valid against the now-shorter `Song.submixes`.
    pub fn remove_submix(&mut self, index: usize) {
        self.submixes.remove(index);
        for track in &mut self.tracks {
            track.output = match track.output {
                TrackOutput::Submix(i) if i == index => TrackOutput::Master,
                TrackOutput::Submix(i) if i > index => TrackOutput::Submix(i - 1),
                other => other,
            };
        }
    }

    /// Returns `scenes[index]`, growing the vec with default (unnamed, no tempo) `Scene`s up to
    /// `index` first if it's not long enough yet — see `Song::scenes`'s doc comment on why scene
    /// rows are grown lazily rather than through an explicit add/remove pair.
    pub fn ensure_scene(&mut self, index: usize) -> &mut Scene {
        if self.scenes.len() <= index {
            self.scenes.resize(index + 1, Scene::default());
        }
        &mut self.scenes[index]
    }

    /// Serializes the song to pretty-printed JSON. Loaded sample audio itself
    /// isn't part of the file (see `Lane::sample`'s `#[serde(skip)]`) — only
    /// `sample_path`, which `load_from_file` re-resolves on read.
    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self).context("failed to serialize song")?;
        std::fs::write(path, json)
            .with_context(|| format!("failed to write {}", path.display()))?;
        Ok(())
    }

    /// Deserializes a song from JSON, then re-loads every lane's sample from
    /// its stored `sample_path` (if any) so playback works immediately —
    /// `sample_rate` is `None` only if the audio engine failed to start, in
    /// which case lanes fall back to the synth until samples are loaded manually.
    ///
    /// Three save-format tiers, oldest first: (1) pre-global-patterns, one `Pattern` owned per
    /// track directly under a `"pattern"` (singular) key, no top-level `"patterns"` →
    /// `LegacySong`/`legacy_to_patterns_era`; (2) global shared `Pattern`s + `ArrangementClip`s,
    /// a top-level `"patterns"` key → `PatternsEraSong`; (3) current, independent per-track
    /// `Region`s, neither of the above → deserializes straight into `Song`. Tier 1 is upgraded to
    /// tier 2's shape first, then both tiers 1 and 2 finish through the same `migrate_patterns_song`
    /// call, so there's exactly one place that produces the final `Region`-based shape.
    ///
    /// Independently of tier, a save from before the master bus became a chain (`master_effects`)
    /// only has a top-level `master_effect_path`/`master_effect_params` pair (one CLAP plugin) —
    /// read directly off the raw JSON here, since none of the three tier shapes above carry it
    /// anymore, and wrapped into a one-entry `master_effects` chain if the new-shape field came
    /// back empty.
    pub fn load_from_file(path: &Path, sample_rate: Option<u32>) -> Result<Self> {
        let json = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let raw: serde_json::Value =
            serde_json::from_str(&json).context("failed to parse song file")?;
        let is_oldest_tier = raw
            .get("tracks")
            .and_then(|t| t.as_array())
            .and_then(|tracks| tracks.first())
            .is_some_and(|t| t.get("pattern").is_some());
        let old_master_effect_path = raw
            .get("master_effect_path")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let old_master_effect_params: Vec<(u32, f64)> = raw
            .get("master_effect_params")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();
        let mut song: Song = if raw.get("patterns").is_some() {
            let mid: PatternsEraSong =
                serde_json::from_value(raw).context("failed to parse song file")?;
            migrate_patterns_song(mid)
        } else if is_oldest_tier {
            let legacy: LegacySong =
                serde_json::from_value(raw).context("failed to parse legacy song file")?;
            migrate_patterns_song(legacy_to_patterns_era(legacy))
        } else {
            serde_json::from_value(raw).context("failed to parse song file")?
        };
        if song.master_effects.is_empty() && !old_master_effect_path.trim().is_empty() {
            song.master_effects = vec![TrackEffectConfig::Clap {
                path: old_master_effect_path,
                params: old_master_effect_params,
                sidechain_source: None,
            }];
        }
        if let Some(rate) = sample_rate {
            song.reload_samples(rate);
        }
        Ok(song)
    }

    /// Re-decodes and resamples every lane/clip with a sample loaded, at `sample_rate` — used
    /// both here (loading a song at the engine's current rate) and when the engine itself
    /// restarts at a different rate (see `main.rs`'s output device/rate picker), so already-loaded
    /// samples don't end up playing back at the wrong pitch/speed after the switch.
    pub fn reload_samples(&mut self, sample_rate: u32) {
        for track in &mut self.tracks {
            for region in &mut track.regions {
                if let RegionContent::StepGrid(lanes) = &mut region.content {
                    for lane in lanes {
                        if !lane.sample_path.trim().is_empty() {
                            lane.load_sample(sample_rate);
                        }
                    }
                }
            }
            for slot in track.session_clips.iter_mut().flatten() {
                if let SessionClipContent::Region { content: RegionContent::StepGrid(lanes), .. } =
                    &mut slot.content
                {
                    for lane in lanes {
                        if !lane.sample_path.trim().is_empty() {
                            lane.load_sample(sample_rate);
                        }
                    }
                }
            }
        }
        for track in &mut self.tracks {
            for clip in &mut track.audio_clips {
                if !clip.file_path.trim().is_empty() {
                    clip.load(sample_rate);
                }
            }
            for folder in &mut track.take_folders {
                folder.load_all(sample_rate);
            }
            for slot in track.session_clips.iter_mut().flatten() {
                if let SessionClipContent::Audio(clip) = &mut slot.content
                    && !clip.file_path.trim().is_empty()
                {
                    clip.load(sample_rate);
                }
            }
        }
    }
}

/// Deserialize-only mirror of the pre-global-patterns save format's per-pattern content: what's
/// now `RegionContent`, plus the `Audio` marker variant that era's format still carried (every
/// track kept exactly one slot back then, even audio tracks — which didn't actually exist yet, so
/// this is unreachable in practice, but kept exhaustive rather than a wildcard).
#[derive(Clone, Deserialize)]
enum LegacyRegionContent {
    StepGrid(Vec<Lane>),
    PianoRoll(Vec<Note>),
    Audio,
}

/// Mirrors the pre-global-patterns save format, where each track owned exactly one `Pattern`
/// (`content: LegacyRegionContent` directly, not one slot per track in a shared list). Deserialize-
/// only — never constructed for saving, since every save now goes through the current `Song`.
#[derive(Deserialize)]
struct LegacyPattern {
    #[allow(dead_code)]
    name: String,
    length_steps: usize,
    content: LegacyRegionContent,
}

#[derive(Deserialize)]
struct LegacyTrack {
    name: String,
    midi_channel: u8,
    #[serde(default)]
    muted: bool,
    pattern: LegacyPattern,
    default_note_length_ticks: usize,
    #[serde(default)]
    effects: Vec<TrackEffectConfig>,
    #[serde(default = "default_track_volume")]
    volume: f32,
    #[serde(default)]
    synth: SynthParams,
    #[serde(default)]
    synth_engine: SynthEngine,
    #[serde(default)]
    trine: TrineParams,
    #[serde(default)]
    wave: WaveParams,
}

#[derive(Deserialize)]
struct LegacySong {
    name: String,
    bpm: f32,
    tracks: Vec<LegacyTrack>,
    next_note_id: u64,
    #[serde(default)]
    synth_presets: Vec<SynthPreset>,
}

/// Deserialize-only mirror of the global-patterns-era `Pattern`/`ArrangementClip` shapes (this
/// app's save format between the original per-track-pattern era and the current per-track-region
/// era) — see `migrate_patterns_song`.
#[derive(Deserialize)]
struct PatternsEraPattern {
    name: String,
    length_steps: usize,
    track_contents: Vec<LegacyRegionContent>,
}

#[derive(Deserialize)]
struct PatternsEraClip {
    pattern_index: usize,
    start_step: usize,
    length_steps: usize,
    /// If set, this clip only ever played one track's slice of the pattern — a later addition to
    /// the patterns era, `#[serde(default)]` so clips saved before it still parse as "every
    /// track", matching their original behavior.
    #[serde(default)]
    track_index: Option<usize>,
}

#[derive(Deserialize)]
struct PatternsEraSong {
    name: String,
    bpm: f32,
    // The patterns era's `Track` shape is identical to today's live `Track` except for the
    // (new, `#[serde(default)]`) `regions` field, so it's safe to deserialize straight into it —
    // `regions` just comes back empty, and `migrate_patterns_song` fills it in.
    tracks: Vec<Track>,
    next_note_id: u64,
    #[serde(default)]
    synth_presets: Vec<SynthPreset>,
    #[serde(default)]
    patterns: Vec<PatternsEraPattern>,
    #[serde(default)]
    arrangement: Vec<PatternsEraClip>,
}

/// Upgrades a pre-global-patterns save into the patterns-era shape: every track's old, independent
/// `Pattern` becomes one slot in a single new global `Pattern` (named "Pattern 1", length = the
/// longest of the old per-track lengths — the same `max` the sequencer already used to pick its
/// loop point), and one `ArrangementClip` spanning that whole pattern reproduces the old
/// loop-forever behavior. Doesn't produce a `Song` directly — `load_from_file` feeds the result
/// through `migrate_patterns_song`, the same as a save from the patterns era itself would.
fn legacy_to_patterns_era(legacy: LegacySong) -> PatternsEraSong {
    let pattern_length_steps = legacy
        .tracks
        .iter()
        .map(|t| t.pattern.length_steps)
        .max()
        .unwrap_or(0);

    let mut tracks = Vec::with_capacity(legacy.tracks.len());
    let mut track_contents = Vec::with_capacity(legacy.tracks.len());
    for legacy_track in legacy.tracks {
        let kind = match &legacy_track.pattern.content {
            LegacyRegionContent::PianoRoll(_) => TrackKind::PianoRoll,
            LegacyRegionContent::StepGrid(_) => TrackKind::StepGrid,
            LegacyRegionContent::Audio => TrackKind::Audio,
        };
        track_contents.push(legacy_track.pattern.content);
        tracks.push(Track {
            name: legacy_track.name,
            midi_channel: legacy_track.midi_channel,
            muted: legacy_track.muted,
            solo: false,
            kind,
            default_note_length_ticks: legacy_track.default_note_length_ticks,
            effects: legacy_track.effects,
            volume: legacy_track.volume,
            pan: 0.0,
            input_gain: default_input_gain(),
            synth: legacy_track.synth,
            synth_engine: legacy_track.synth_engine,
            trine: legacy_track.trine,
            wave: legacy_track.wave,
            audio_clips: Vec::new(),
            take_folders: Vec::new(),
            regions: Vec::new(),
            send_levels: Vec::new(),
            output: TrackOutput::Master,
            automation: Vec::new(),
            frozen: false,
            frozen_clip: None,
            session_clips: Vec::new(),
            session_launch_requests: Vec::new(),
        });
    }

    let patterns = if tracks.is_empty() {
        Vec::new()
    } else {
        vec![PatternsEraPattern {
            name: "Pattern 1".to_string(),
            length_steps: pattern_length_steps,
            track_contents,
        }]
    };
    let arrangement = if patterns.is_empty() {
        Vec::new()
    } else {
        vec![PatternsEraClip {
            pattern_index: 0,
            start_step: 0,
            length_steps: pattern_length_steps,
            track_index: None,
        }]
    };

    PatternsEraSong {
        name: legacy.name,
        bpm: legacy.bpm,
        tracks,
        next_note_id: legacy.next_note_id,
        synth_presets: legacy.synth_presets,
        patterns,
        arrangement,
    }
}

/// Converts a patterns-era save (global, shared `Pattern`s placed by `ArrangementClip`s) into the
/// current shape: each clip becomes one independent `Region` per track it actually plays for
/// (respecting `track_index` scoping if the clip had it, otherwise one region per non-`Audio`
/// track slot in that pattern) — the whole point of the move to per-track regions is that this is
/// a one-way copy, not a reference, so two old clips sharing a pattern become two independent
/// regions that can then be edited separately.
fn migrate_patterns_song(mid: PatternsEraSong) -> Song {
    let mut tracks = mid.tracks;
    for clip in &mid.arrangement {
        let Some(pattern) = mid.patterns.get(clip.pattern_index) else {
            continue;
        };
        let targets: Vec<usize> = match clip.track_index {
            Some(t) => vec![t],
            None => (0..tracks.len()).collect(),
        };
        for track_index in targets {
            let Some(content) = pattern.track_contents.get(track_index) else {
                continue;
            };
            let region_content = match content {
                LegacyRegionContent::StepGrid(lanes) => RegionContent::StepGrid(lanes.clone()),
                LegacyRegionContent::PianoRoll(notes) => RegionContent::PianoRoll(notes.clone()),
                LegacyRegionContent::Audio => continue,
            };
            if let Some(track) = tracks.get_mut(track_index) {
                track.regions.push(Region {
                    name: pattern.name.clone(),
                    start_tick: clip.start_step * TICKS_PER_STEP,
                    content_length_steps: pattern.length_steps,
                    loop_length_steps: clip.length_steps,
                    content: region_content,
                    fade_in_ticks: 0,
                    fade_out_ticks: 0,
                    automation: Vec::new(),
                });
            }
        }
    }

    Song {
        name: mid.name,
        bpm: mid.bpm,
        tempo_map: Vec::new(),
        tracks,
        next_note_id: mid.next_note_id,
        // Populated uniformly by `load_from_file` from the raw JSON's old
        // `master_effect_path`/`master_effect_params` keys (present on saves from any era, this
        // migration function's own tier included), not threaded through here.
        master_effects: Vec::new(),
        plugins: Vec::new(),
        synth_presets: mid.synth_presets,
        time_signature_numerator: 4,
        time_signature_denominator: 4,
        sends: Vec::new(),
        submixes: Vec::new(),
        scenes: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn region_with_fades(
        loop_length_steps: usize,
        fade_in_ticks: usize,
        fade_out_ticks: usize,
    ) -> Region {
        Region {
            name: "Test".to_string(),
            start_tick: 0,
            content_length_steps: loop_length_steps,
            loop_length_steps,
            content: RegionContent::PianoRoll(Vec::new()),
            fade_in_ticks,
            fade_out_ticks,
            automation: Vec::new(),
        }
    }

    #[test]
    fn curve_shape_warp_matches_the_line_at_the_segment_endpoints() {
        for shape in [CurveShape::Linear, CurveShape::Exponential, CurveShape::Logarithmic] {
            assert_eq!(shape.warp(0.0), 0.0, "{shape:?} should start exactly at the first point");
            assert!((shape.warp(1.0) - 1.0).abs() < 1e-6, "{shape:?} should end exactly at the next point");
        }
        // Hold is the one shape that does *not* reach 1.0 at t=1.0 — see its own doc comment.
        assert_eq!(CurveShape::Hold.warp(1.0), 0.0);
    }

    #[test]
    fn curve_shape_next_cycles_through_every_shape_and_back() {
        assert_eq!(CurveShape::Linear.next(), CurveShape::Exponential);
        assert_eq!(CurveShape::Exponential.next(), CurveShape::Logarithmic);
        assert_eq!(CurveShape::Logarithmic.next(), CurveShape::Hold);
        assert_eq!(CurveShape::Hold.next(), CurveShape::Linear);
    }

    #[test]
    fn value_at_is_none_for_an_empty_lane() {
        let lane = AutomationLane { target: AutomationTarget::Volume, points: Vec::new() };
        assert_eq!(lane.value_at_fractional(0.0), None);
        assert_eq!(lane.value_at_fractional(100.0), None);
    }

    #[test]
    fn value_at_holds_the_single_point_everywhere() {
        let lane = AutomationLane {
            target: AutomationTarget::Volume,
            points: vec![AutomationPoint { tick: 50, value: 0.75, curve: CurveShape::default() }],
        };
        assert_eq!(lane.value_at_fractional(0.0), Some(0.75));
        assert_eq!(lane.value_at_fractional(50.0), Some(0.75));
        assert_eq!(lane.value_at_fractional(1000.0), Some(0.75));
    }

    #[test]
    fn value_at_interpolates_linearly_between_bracketing_points() {
        let lane = AutomationLane {
            target: AutomationTarget::Pan,
            points: vec![
                AutomationPoint { tick: 0, value: -1.0, curve: CurveShape::default() },
                AutomationPoint { tick: 100, value: 1.0, curve: CurveShape::default() },
            ],
        };
        assert_eq!(lane.value_at_fractional(0.0), Some(-1.0));
        assert!((lane.value_at_fractional(50.0).unwrap() - 0.0).abs() < 1e-6);
        assert_eq!(lane.value_at_fractional(100.0), Some(1.0));
    }

    #[test]
    fn value_at_holds_flat_then_jumps_for_a_hold_curve() {
        let lane = AutomationLane {
            target: AutomationTarget::Volume,
            points: vec![
                AutomationPoint { tick: 0, value: 0.0, curve: CurveShape::Hold },
                AutomationPoint { tick: 100, value: 1.0, curve: CurveShape::default() },
            ],
        };
        assert_eq!(lane.value_at_fractional(0.0), Some(0.0));
        assert_eq!(lane.value_at_fractional(50.0), Some(0.0));
        assert_eq!(lane.value_at_fractional(99.999), Some(0.0));
        // Exactly at the next point, that point's own value takes over.
        assert_eq!(lane.value_at_fractional(100.0), Some(1.0));
    }

    #[test]
    fn value_at_eases_in_for_an_exponential_curve() {
        let lane = AutomationLane {
            target: AutomationTarget::Volume,
            points: vec![
                AutomationPoint { tick: 0, value: 0.0, curve: CurveShape::Exponential },
                AutomationPoint { tick: 100, value: 1.0, curve: CurveShape::default() },
            ],
        };
        // Ease-in: stays below the straight-line value for the first half of the segment.
        let midpoint = lane.value_at_fractional(50.0).unwrap();
        assert!(midpoint < 0.5, "exponential should ease in below the linear midpoint, got {midpoint}");
        assert_eq!(lane.value_at_fractional(0.0), Some(0.0));
        assert_eq!(lane.value_at_fractional(100.0), Some(1.0));
    }

    #[test]
    fn value_at_eases_out_for_a_logarithmic_curve() {
        let lane = AutomationLane {
            target: AutomationTarget::Volume,
            points: vec![
                AutomationPoint { tick: 0, value: 0.0, curve: CurveShape::Logarithmic },
                AutomationPoint { tick: 100, value: 1.0, curve: CurveShape::default() },
            ],
        };
        // Ease-out: already past the straight-line value by the first half of the segment.
        let midpoint = lane.value_at_fractional(50.0).unwrap();
        assert!(midpoint > 0.5, "logarithmic should ease out above the linear midpoint, got {midpoint}");
        assert_eq!(lane.value_at_fractional(0.0), Some(0.0));
        assert_eq!(lane.value_at_fractional(100.0), Some(1.0));
    }

    #[test]
    fn a_points_own_curve_shape_round_trips_through_save_and_load() {
        let mut song = Song::demo();
        song.tracks[0].automation.push(AutomationLane {
            target: AutomationTarget::Volume,
            points: vec![
                AutomationPoint { tick: 0, value: 0.2, curve: CurveShape::Exponential },
                AutomationPoint { tick: 100, value: 0.8, curve: CurveShape::Hold },
            ],
        });

        let path = std::env::temp_dir().join(format!(
            "simple-daw-test-curve-shape-round-trip-{}.json",
            std::process::id()
        ));
        song.save_to_file(&path).unwrap();
        let loaded = Song::load_from_file(&path, None).unwrap();
        std::fs::remove_file(&path).ok();

        let points = &loaded.tracks[0].automation[0].points;
        assert_eq!(points[0].curve, CurveShape::Exponential);
        assert_eq!(points[1].curve, CurveShape::Hold);
    }

    #[test]
    fn automation_point_defaults_to_linear_curve_for_pre_existing_song_files() {
        // A song file saved before curve shapes existed: its automation point has no "curve" key.
        let json = r#"{
            "name": "Old Automation Song",
            "bpm": 120.0,
            "next_note_id": 0,
            "tracks": [
                {
                    "name": "Lead",
                    "midi_channel": 1,
                    "muted": false,
                    "kind": "PianoRoll",
                    "default_note_length_ticks": 96,
                    "automation": [
                        {
                            "target": { "kind": "Volume" },
                            "points": [{ "tick": 0, "value": 0.5 }]
                        }
                    ]
                }
            ]
        }"#;
        let path = std::env::temp_dir().join(format!(
            "simple-daw-test-legacy-curve-shape-{}.json",
            std::process::id()
        ));
        std::fs::write(&path, json).unwrap();

        let loaded = Song::load_from_file(&path, None).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(loaded.tracks[0].automation[0].points[0].curve, CurveShape::Linear);
    }

    #[test]
    fn value_at_fractional_interpolates_between_whole_ticks() {
        let lane = AutomationLane {
            target: AutomationTarget::Pan,
            points: vec![
                AutomationPoint { tick: 0, value: -1.0, curve: CurveShape::default() },
                AutomationPoint { tick: 100, value: 1.0, curve: CurveShape::default() },
            ],
        };
        assert!((lane.value_at_fractional(25.5).unwrap() - -0.49).abs() < 1e-6);
    }

    #[test]
    fn value_at_holds_the_nearest_point_outside_the_lanes_own_range() {
        let lane = AutomationLane {
            target: AutomationTarget::Volume,
            points: vec![
                AutomationPoint { tick: 20, value: 0.2, curve: CurveShape::default() },
                AutomationPoint { tick: 80, value: 0.8, curve: CurveShape::default() },
            ],
        };
        assert_eq!(lane.value_at_fractional(0.0), Some(0.2));
        assert_eq!(lane.value_at_fractional(1000.0), Some(0.8));
    }

    #[test]
    fn value_at_does_not_require_points_sorted_by_tick() {
        let lane = AutomationLane {
            target: AutomationTarget::Volume,
            points: vec![
                AutomationPoint { tick: 100, value: 1.0, curve: CurveShape::default() },
                AutomationPoint { tick: 0, value: 0.0, curve: CurveShape::default() },
            ],
        };
        assert!((lane.value_at_fractional(50.0).unwrap() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn fade_gain_at_is_full_with_no_fades_configured() {
        let region = region_with_fades(4, 0, 0);
        let span_ticks = 4 * TICKS_PER_STEP;
        assert_eq!(region.fade_gain_at(0), 1.0);
        assert_eq!(region.fade_gain_at(span_ticks / 2), 1.0);
        assert_eq!(region.fade_gain_at(span_ticks - 1), 1.0);
    }

    #[test]
    fn fade_gain_at_ramps_from_zero_to_one_over_fade_in_ticks() {
        let region = region_with_fades(4, 10, 0);
        assert_eq!(region.fade_gain_at(0), 0.0);
        assert!((region.fade_gain_at(5) - 0.5).abs() < 1e-6);
        assert_eq!(region.fade_gain_at(10), 1.0);
        assert_eq!(region.fade_gain_at(20), 1.0);
    }

    #[test]
    fn fade_gain_at_ramps_from_one_to_zero_over_the_last_fade_out_ticks() {
        let region = region_with_fades(4, 0, 10);
        let span_ticks = 4 * TICKS_PER_STEP;
        assert_eq!(region.fade_gain_at(0), 1.0);
        assert_eq!(region.fade_gain_at(span_ticks - 10), 1.0);
        assert!((region.fade_gain_at(span_ticks - 5) - 0.5).abs() < 1e-6);
        assert_eq!(region.fade_gain_at(span_ticks), 0.0);
    }

    #[test]
    fn fade_gain_at_takes_the_more_attenuated_ramp_when_fades_overlap() {
        // A region shorter than fade_in_ticks + fade_out_ticks: at the midpoint, both ramps are
        // partway through, and the more attenuated one should win rather than one overriding it.
        let region = region_with_fades(1, TICKS_PER_STEP, TICKS_PER_STEP);
        let span_ticks = TICKS_PER_STEP;
        let midpoint_gain = region.fade_gain_at(span_ticks / 2);
        assert!((midpoint_gain - 0.5).abs() < 1e-6);
        assert_eq!(region.fade_gain_at(0), 0.0);
        assert_eq!(region.fade_gain_at(span_ticks), 0.0);
    }

    #[test]
    fn save_then_load_round_trips_song_structure() {
        let song = Song::demo();
        let path =
            std::env::temp_dir().join(format!("simple-daw-test-{}.json", std::process::id()));

        song.save_to_file(&path).unwrap();
        let loaded = Song::load_from_file(&path, None).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(loaded.name, song.name);
        assert_eq!(loaded.bpm, song.bpm);
        assert_eq!(loaded.next_note_id, song.next_note_id);
        assert_eq!(loaded.tracks.len(), song.tracks.len());

        let (orig_drums, loaded_drums) = (&song.tracks[0], &loaded.tracks[0]);
        assert_eq!(loaded_drums.name, orig_drums.name);
        assert_eq!(loaded_drums.regions.len(), orig_drums.regions.len());

        match (
            &song.tracks[0].regions[0].content,
            &loaded.tracks[0].regions[0].content,
        ) {
            (RegionContent::StepGrid(orig_lanes), RegionContent::StepGrid(loaded_lanes)) => {
                assert_eq!(loaded_lanes.len(), orig_lanes.len());
                for (orig, loaded) in orig_lanes.iter().zip(loaded_lanes) {
                    assert_eq!(loaded.pitch, orig.pitch);
                    assert_eq!(loaded.steps, orig.steps);
                    // Sample audio isn't serialized; with no sample_rate passed
                    // to load_from_file it stays unloaded even though the demo
                    // song's lanes aren't pre-populated with a sample here anyway.
                    assert!(loaded.sample.is_none());
                }
            }
            _ => panic!("expected StepGrid content for the Drums track's region"),
        }

        match (
            &song.tracks[1].regions[0].content,
            &loaded.tracks[1].regions[0].content,
        ) {
            (RegionContent::PianoRoll(orig_notes), RegionContent::PianoRoll(loaded_notes)) => {
                assert_eq!(loaded_notes.len(), orig_notes.len());
                for (orig, loaded) in orig_notes.iter().zip(loaded_notes) {
                    assert_eq!(loaded.id, orig.id);
                    assert_eq!(loaded.pitch, orig.pitch);
                    assert_eq!(loaded.start_tick, orig.start_tick);
                    assert_eq!(loaded.length_ticks, orig.length_ticks);
                    assert_eq!(loaded.velocity, orig.velocity);
                }
            }
            _ => panic!("expected PianoRoll content for the Bass track's region"),
        }
    }

    #[test]
    fn save_then_load_round_trips_the_tempo_map() {
        let mut song = Song::demo();
        song.set_tempo_at(1000, 140.0);
        song.set_tempo_at(2000, 80.0);
        let path = std::env::temp_dir().join(format!(
            "simple-daw-test-tempo-map-{}.json",
            std::process::id()
        ));

        song.save_to_file(&path).unwrap();
        let loaded = Song::load_from_file(&path, None).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(loaded.tempo_map, song.tempo_map);
    }

    #[test]
    fn load_from_file_defaults_to_an_empty_tempo_map_for_pre_existing_song_files() {
        // A song file saved before tempo maps existed has no "tempo_map" key at all.
        let json = r#"{
            "name": "Old Song",
            "bpm": 120.0,
            "next_note_id": 0,
            "tracks": []
        }"#;
        let path = std::env::temp_dir().join(format!(
            "simple-daw-test-legacy-tempo-map-{}.json",
            std::process::id()
        ));
        std::fs::write(&path, json).unwrap();

        let loaded = Song::load_from_file(&path, None).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(loaded.bpm, 120.0);
        assert!(loaded.tempo_map.is_empty());
    }

    #[test]
    fn add_track_audio_has_no_regions() {
        let mut song = Song::demo();
        let new_index = song.add_track("Vocals", 5, TrackKind::Audio);

        assert_eq!(song.tracks[new_index].kind, TrackKind::Audio);
        assert!(song.tracks[new_index].regions.is_empty());
    }

    #[test]
    fn save_then_load_round_trips_audio_clips_but_not_buffer() {
        let mut song = Song::demo();
        let audio_index = song.add_track("Vocals", 5, TrackKind::Audio);
        let mut clip = AudioClip::new(48, "some/recording.wav");
        clip.gain = 0.8;
        clip.source_start_frame = 4_800;
        clip.length_ticks = 960;
        clip.fade_in_ticks = 96;
        clip.fade_out_ticks = 192;
        clip.pitch_corrections = vec![PitchCorrection {
            start_frame: 100,
            end_frame: 200,
            target_semitones: 2.5,
        }];
        clip.warp_markers = vec![
            WarpMarker { source_frame: 0, output_frame: 0 },
            WarpMarker { source_frame: 48_000, output_frame: 44_100 },
        ];
        song.tracks[audio_index].audio_clips.push(clip);

        let path = std::env::temp_dir().join(format!(
            "simple-daw-test-audio-clip-{}.json",
            std::process::id()
        ));
        song.save_to_file(&path).unwrap();
        // No sample_rate passed, so clips stay unloaded — this only checks the persisted fields.
        let loaded = Song::load_from_file(&path, None).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(loaded.tracks[audio_index].audio_clips.len(), 1);
        let loaded_clip = &loaded.tracks[audio_index].audio_clips[0];
        assert_eq!(loaded_clip.start_tick, 48);
        assert_eq!(loaded_clip.file_path, "some/recording.wav");
        assert_eq!(loaded_clip.gain, 0.8);
        assert_eq!(loaded_clip.source_start_frame, 4_800);
        assert_eq!(loaded_clip.length_ticks, 960);
        assert_eq!(loaded_clip.fade_in_ticks, 96);
        assert_eq!(loaded_clip.fade_out_ticks, 192);
        assert_eq!(loaded_clip.pitch_corrections.len(), 1);
        assert_eq!(loaded_clip.pitch_corrections[0].target_semitones, 2.5);
        assert_eq!(loaded_clip.warp_markers.len(), 2);
        assert_eq!(loaded_clip.warp_markers[1].output_frame, 44_100);
        assert!(
            loaded_clip.buffer.is_none(),
            "decoded audio isn't song data and must not be serialized"
        );
    }

    #[test]
    fn audio_clip_from_a_pre_trim_song_file_defaults_to_untrimmed_no_fade() {
        // A song file saved before trim/fade existed has none of the new fields in its JSON —
        // `#[serde(default)]` must fill them with the "no trim/fade yet" sentinel (0), not fail to
        // parse or silently drop the clip.
        let json = r#"{
            "start_tick": 48,
            "file_path": "some/recording.wav",
            "gain": 0.8
        }"#;
        let clip: AudioClip = serde_json::from_str(json).unwrap();
        assert_eq!(clip.source_start_frame, 0);
        assert_eq!(clip.length_ticks, 0);
        assert_eq!(clip.fade_in_ticks, 0);
        assert_eq!(clip.fade_out_ticks, 0);
        assert!(clip.pitch_corrections.is_empty());
        assert!(clip.warp_markers.is_empty());
    }

    /// Writes a short mono sine WAV to the system temp dir for `AudioClip::load` integration
    /// tests — the one place in this file that actually exercises the decode path, since
    /// `AudioClip`'s other tests only ever set `buffer` directly.
    fn write_test_wav(sample_rate: u32, duration_seconds: f32) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "simple-daw-test-flex-{}-{}.wav",
            std::process::id(),
            sample_rate
        ));
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&path, spec).unwrap();
        let n = (sample_rate as f32 * duration_seconds) as usize;
        for i in 0..n {
            let s = (2.0 * std::f32::consts::PI * 220.0 * i as f32 / sample_rate as f32).sin();
            writer.write_sample((s * i16::MAX as f32) as i16).unwrap();
        }
        writer.finalize().unwrap();
        path
    }

    #[test]
    fn load_applies_warp_markers_to_the_decoded_buffer() {
        let sample_rate = 48_000;
        let path = write_test_wav(sample_rate, 1.0);
        let mut clip = AudioClip::new(0, path.to_string_lossy().to_string());
        clip.warp_markers = vec![
            WarpMarker { source_frame: 0, output_frame: 0 },
            WarpMarker {
                source_frame: sample_rate as usize,
                output_frame: sample_rate as usize * 2,
            },
        ];
        clip.load(sample_rate);
        std::fs::remove_file(&path).ok();

        let buffer = clip.buffer.expect("clip should have loaded");
        assert_eq!(buffer.mono.len(), sample_rate as usize * 2);
    }

    #[test]
    fn load_with_no_warp_markers_or_pitch_corrections_leaves_the_buffer_untouched() {
        let sample_rate = 48_000;
        let path = write_test_wav(sample_rate, 0.5);
        let mut clip = AudioClip::new(0, path.to_string_lossy().to_string());
        clip.load(sample_rate);
        std::fs::remove_file(&path).ok();

        let buffer = clip.buffer.expect("clip should have loaded");
        assert_eq!(buffer.mono.len(), sample_rate as usize / 2);
    }

    #[test]
    fn effective_length_ticks_uses_length_ticks_when_trimmed() {
        let mut clip = AudioClip::new(0, "some/recording.wav");
        clip.length_ticks = 480;
        assert_eq!(clip.effective_length_ticks(1000.0), 480);
    }

    #[test]
    fn effective_length_ticks_falls_back_to_full_buffer_when_untrimmed() {
        let mut clip = AudioClip::new(0, "some/recording.wav");
        clip.buffer = Some(Arc::new(SampleBuffer {
            sample_rate: 48_000,
            mono: vec![0.0; 48_000],
        }));
        // 1 second of audio at 48kHz, converted at 1000 ticks/second, is 1000 ticks.
        assert_eq!(clip.effective_length_ticks(1000.0), 1000);
    }

    #[test]
    fn effective_length_ticks_accounts_for_source_start_frame() {
        let mut clip = AudioClip::new(0, "some/recording.wav");
        clip.source_start_frame = 24_000;
        clip.buffer = Some(Arc::new(SampleBuffer {
            sample_rate: 48_000,
            mono: vec![0.0; 48_000],
        }));
        // Only the remaining 0.5s (24_000 frames) past the trimmed head counts.
        assert_eq!(clip.effective_length_ticks(1000.0), 500);
    }

    #[test]
    fn take_folder_new_comps_the_whole_span_to_its_first_take() {
        let folder = TakeFolder::new(480, 960, "take1.wav");
        assert_eq!(folder.takes.len(), 1);
        assert_eq!(folder.comp.len(), 1);
        assert_eq!(folder.comp[0].take_index, 0);
        assert_eq!(folder.comp[0].start_tick, 0);
        assert_eq!(folder.comp[0].end_tick, 960);
    }

    #[test]
    fn add_take_and_activate_appends_and_comps_the_new_take() {
        let mut folder = TakeFolder::new(0, 960, "take1.wav");
        let new_index = folder.add_take_and_activate("take2.wav");
        assert_eq!(new_index, 1);
        assert_eq!(folder.takes.len(), 2);
        assert_eq!(folder.comp.len(), 1, "still one segment spanning the whole folder");
        assert_eq!(folder.comp[0].take_index, 1);
    }

    #[test]
    fn set_active_take_replaces_comp_with_a_single_whole_span_segment() {
        let mut folder = TakeFolder::new(0, 960, "take1.wav");
        folder.add_take_and_activate("take2.wav");
        folder.add_take_and_activate("take3.wav");
        folder.set_active_take(0);
        assert_eq!(folder.comp.len(), 1);
        assert_eq!(folder.comp[0].take_index, 0);
        assert_eq!(folder.comp[0].start_tick, 0);
        assert_eq!(folder.comp[0].end_tick, 960);
    }

    #[test]
    fn set_active_take_is_a_no_op_for_an_out_of_bounds_index() {
        let mut folder = TakeFolder::new(0, 960, "take1.wav");
        let before = folder.comp.clone();
        folder.set_active_take(5);
        assert_eq!(
            folder.comp.len(),
            before.len(),
            "comp segment count should be unchanged"
        );
        assert_eq!(folder.comp[0].take_index, before[0].take_index);
    }

    #[test]
    fn assign_take_to_range_splits_a_single_segment_into_three() {
        let mut folder = TakeFolder::new(0, 900, "take1.wav");
        folder.add_take_and_activate("take2.wav");
        // Whole span is take 1 right now; paint take 0 over the middle third.
        folder.assign_take_to_range(0, 300, 600);
        assert_eq!(folder.comp.len(), 3);
        assert_eq!(folder.comp[0], CompSegment { start_tick: 0, end_tick: 300, take_index: 1 });
        assert_eq!(folder.comp[1], CompSegment { start_tick: 300, end_tick: 600, take_index: 0 });
        assert_eq!(folder.comp[2], CompSegment { start_tick: 600, end_tick: 900, take_index: 1 });
    }

    #[test]
    fn assign_take_to_range_merges_adjacent_segments_on_the_same_take() {
        let mut folder = TakeFolder::new(0, 900, "take1.wav");
        folder.add_take_and_activate("take2.wav");
        folder.assign_take_to_range(0, 300, 600);
        // Now paint take 1 back over the same middle third — should merge back into one segment.
        folder.assign_take_to_range(1, 300, 600);
        assert_eq!(folder.comp, vec![CompSegment { start_tick: 0, end_tick: 900, take_index: 1 }]);
    }

    #[test]
    fn assign_take_to_range_clamps_to_the_folder_span() {
        let mut folder = TakeFolder::new(0, 900, "take1.wav");
        folder.add_take_and_activate("take2.wav");
        folder.assign_take_to_range(0, 600, 5_000);
        assert_eq!(
            folder.comp,
            vec![
                CompSegment { start_tick: 0, end_tick: 600, take_index: 1 },
                CompSegment { start_tick: 600, end_tick: 900, take_index: 0 },
            ]
        );
    }

    #[test]
    fn assign_take_to_range_is_a_no_op_for_a_backwards_or_empty_range() {
        let mut folder = TakeFolder::new(0, 900, "take1.wav");
        let before = folder.comp.clone();
        folder.assign_take_to_range(0, 600, 600);
        folder.assign_take_to_range(0, 600, 300);
        assert_eq!(folder.comp, before);
    }

    #[test]
    fn assign_take_to_range_is_a_no_op_for_an_out_of_bounds_take() {
        let mut folder = TakeFolder::new(0, 900, "take1.wav");
        let before = folder.comp.clone();
        folder.assign_take_to_range(5, 100, 200);
        assert_eq!(folder.comp, before);
    }

    #[test]
    fn save_then_load_round_trips_a_take_folder_and_its_comp() {
        let mut song = Song::demo();
        let audio_index = song.add_track("Vocals", 5, TrackKind::Audio);
        let mut folder = TakeFolder::new(48, 960, "take1.wav");
        folder.add_take_and_activate("take2.wav");
        folder.gain = 0.7;
        song.tracks[audio_index].take_folders.push(folder);

        let path = std::env::temp_dir().join(format!(
            "simple-daw-test-take-folder-{}.json",
            std::process::id()
        ));
        song.save_to_file(&path).unwrap();
        let loaded = Song::load_from_file(&path, None).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(loaded.tracks[audio_index].take_folders.len(), 1);
        let loaded_folder = &loaded.tracks[audio_index].take_folders[0];
        assert_eq!(loaded_folder.start_tick, 48);
        assert_eq!(loaded_folder.length_ticks, 960);
        assert_eq!(loaded_folder.gain, 0.7);
        assert_eq!(loaded_folder.takes.len(), 2);
        assert_eq!(loaded_folder.takes[0].file_path, "take1.wav");
        assert_eq!(loaded_folder.takes[1].file_path, "take2.wav");
        assert_eq!(loaded_folder.comp.len(), 1);
        assert_eq!(loaded_folder.comp[0].take_index, 1);
        assert!(
            loaded_folder.takes[0].buffer.is_none(),
            "decoded audio isn't song data and must not be serialized"
        );
    }

    #[test]
    fn take_folder_from_a_pre_comping_song_file_defaults_to_no_folders() {
        // A song file saved before comping existed has no `take_folders` key on its tracks at all
        // — `#[serde(default)]` must fill it with an empty list, not fail to parse. Derived from a
        // real `Track`'s own JSON (minus the one key under test) so every other field stays valid,
        // rather than hand-rolling a minimal JSON object that could drift from the real shape.
        let track = Track::new_audio("Vocals", 5);
        let mut value = serde_json::to_value(&track).unwrap();
        value.as_object_mut().unwrap().remove("take_folders");
        let loaded: Track = serde_json::from_value(value).unwrap();
        assert!(loaded.take_folders.is_empty());
    }

    #[test]
    fn save_then_load_round_trips_effect_state() {
        let mut song = Song::demo();
        song.master_effects = vec![TrackEffectConfig::Clap {
            path: "/usr/lib64/clap/ZamDelay.clap".to_string(),
            params: vec![(0, 0.5), (2, 1.0)],
            sidechain_source: None,
        }];
        song.tracks[1].effects = vec![TrackEffectConfig::Clap {
            path: "/usr/lib64/clap/ZamGate.clap".to_string(),
            params: vec![(1, 0.25)],
            sidechain_source: None,
        }];
        song.tracks[1].volume = 0.6;
        song.tracks[1].input_gain = 1.4;

        let path =
            std::env::temp_dir().join(format!("simple-daw-test-fx-{}.json", std::process::id()));
        song.save_to_file(&path).unwrap();
        let loaded = Song::load_from_file(&path, None).unwrap();
        std::fs::remove_file(&path).ok();

        match &loaded.master_effects[..] {
            [TrackEffectConfig::Clap { path, params, .. }] => {
                assert_eq!(path, "/usr/lib64/clap/ZamDelay.clap");
                assert_eq!(params, &vec![(0, 0.5), (2, 1.0)]);
            }
            other => panic!("expected a single master Clap effect, got {other:?}"),
        }
        assert_eq!(loaded.tracks[1].effects.len(), 1);
        match &loaded.tracks[1].effects[0] {
            TrackEffectConfig::Clap { path, params, .. } => {
                assert_eq!(path, "/usr/lib64/clap/ZamGate.clap");
                assert_eq!(params, &vec![(1, 0.25)]);
            }
            other => panic!("expected a Clap effect, got {other:?}"),
        }
        assert_eq!(loaded.tracks[1].volume, 0.6);
        assert_eq!(loaded.tracks[1].input_gain, 1.4);
        assert!(
            loaded.tracks[0].effects.is_empty(),
            "track with no effect loaded should round-trip as empty, not inherit another track's"
        );
    }

    #[test]
    fn save_then_load_round_trips_a_stacked_effect_chain_in_order() {
        // A mix of a CLAP plugin and two built-in effects, in a specific order — the chain must
        // round-trip both which kind each slot is and their parameter values.
        let mut song = Song::demo();
        song.tracks[0].effects = vec![
            TrackEffectConfig::Clap {
                path: "/usr/lib64/clap/ZamGate.clap".to_string(),
                params: vec![(1, 0.25)],
                sidechain_source: None,
            },
            TrackEffectConfig::Delay {
                time_ms: 450.0,
                feedback: 0.5,
                mix: 0.4,
            },
            TrackEffectConfig::Bitcrusher {
                bit_depth: 3.0,
                rate_divisor: 8,
                mix: 1.0,
            },
        ];

        let path = std::env::temp_dir().join(format!(
            "simple-daw-test-fx-chain-{}.json",
            std::process::id()
        ));
        song.save_to_file(&path).unwrap();
        let loaded = Song::load_from_file(&path, None).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(loaded.tracks[0].effects.len(), 3);
        match &loaded.tracks[0].effects[0] {
            TrackEffectConfig::Clap { path, .. } => {
                assert_eq!(path, "/usr/lib64/clap/ZamGate.clap")
            }
            other => panic!("expected a Clap effect, got {other:?}"),
        }
        match &loaded.tracks[0].effects[1] {
            TrackEffectConfig::Delay {
                time_ms,
                feedback,
                mix,
            } => {
                assert_eq!(*time_ms, 450.0);
                assert_eq!(*feedback, 0.5);
                assert_eq!(*mix, 0.4);
            }
            other => panic!("expected a Delay effect, got {other:?}"),
        }
        match &loaded.tracks[0].effects[2] {
            TrackEffectConfig::Bitcrusher {
                bit_depth,
                rate_divisor,
                mix,
            } => {
                assert_eq!(*bit_depth, 3.0);
                assert_eq!(*rate_divisor, 8);
                assert_eq!(*mix, 1.0);
            }
            other => panic!("expected a Bitcrusher effect, got {other:?}"),
        }
    }

    #[test]
    fn save_then_load_round_trips_chorus_filter_tremolo_and_compressor() {
        let mut song = Song::demo();
        song.tracks[0].effects = vec![
            TrackEffectConfig::Chorus {
                rate_hz: 2.0,
                depth_ms: 12.0,
                mix: 0.6,
            },
            TrackEffectConfig::Filter {
                cutoff_hz: 800.0,
                resonance: 0.7,
                mode: FilterMode::HighPass,
                mix: 1.0,
            },
            TrackEffectConfig::Tremolo {
                rate_hz: 6.0,
                depth: 0.8,
            },
            TrackEffectConfig::Compressor {
                threshold_db: -12.0,
                ratio: 3.0,
                attack_ms: 5.0,
                release_ms: 150.0,
                makeup_db: 2.0,
                sidechain_source: None,
            },
        ];

        let path = std::env::temp_dir().join(format!(
            "simple-daw-test-fx-new-{}.json",
            std::process::id()
        ));
        song.save_to_file(&path).unwrap();
        let loaded = Song::load_from_file(&path, None).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(loaded.tracks[0].effects.len(), 4);
        match &loaded.tracks[0].effects[0] {
            TrackEffectConfig::Chorus {
                rate_hz,
                depth_ms,
                mix,
            } => {
                assert_eq!(*rate_hz, 2.0);
                assert_eq!(*depth_ms, 12.0);
                assert_eq!(*mix, 0.6);
            }
            other => panic!("expected a Chorus effect, got {other:?}"),
        }
        match &loaded.tracks[0].effects[1] {
            TrackEffectConfig::Filter {
                cutoff_hz,
                resonance,
                mode,
                mix,
            } => {
                assert_eq!(*cutoff_hz, 800.0);
                assert_eq!(*resonance, 0.7);
                assert_eq!(*mode, FilterMode::HighPass);
                assert_eq!(*mix, 1.0);
            }
            other => panic!("expected a Filter effect, got {other:?}"),
        }
        match &loaded.tracks[0].effects[2] {
            TrackEffectConfig::Tremolo { rate_hz, depth } => {
                assert_eq!(*rate_hz, 6.0);
                assert_eq!(*depth, 0.8);
            }
            other => panic!("expected a Tremolo effect, got {other:?}"),
        }
        match &loaded.tracks[0].effects[3] {
            TrackEffectConfig::Compressor {
                threshold_db,
                ratio,
                attack_ms,
                release_ms,
                makeup_db,
                ..
            } => {
                assert_eq!(*threshold_db, -12.0);
                assert_eq!(*ratio, 3.0);
                assert_eq!(*attack_ms, 5.0);
                assert_eq!(*release_ms, 150.0);
                assert_eq!(*makeup_db, 2.0);
            }
            other => panic!("expected a Compressor effect, got {other:?}"),
        }
    }

    #[test]
    fn save_then_load_round_trips_flanger_phaser_ring_mod_and_noise_gate() {
        let mut song = Song::demo();
        song.tracks[0].effects = vec![
            TrackEffectConfig::Flanger {
                rate_hz: 0.8,
                depth_ms: 4.0,
                feedback: 0.6,
                mix: 0.5,
            },
            TrackEffectConfig::Phaser {
                rate_hz: 0.3,
                depth: 0.9,
                feedback: 0.4,
                mix: 0.5,
            },
            TrackEffectConfig::RingModulator {
                carrier_hz: 350.0,
                mix: 0.7,
            },
            TrackEffectConfig::NoiseGate {
                threshold_db: -30.0,
                attack_ms: 3.0,
                release_ms: 200.0,
                range_db: -70.0,
                sidechain_source: None,
            },
        ];

        let path = std::env::temp_dir().join(format!(
            "simple-daw-test-fx-newer-{}.json",
            std::process::id()
        ));
        song.save_to_file(&path).unwrap();
        let loaded = Song::load_from_file(&path, None).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(loaded.tracks[0].effects.len(), 4);
        match &loaded.tracks[0].effects[0] {
            TrackEffectConfig::Flanger {
                rate_hz,
                depth_ms,
                feedback,
                mix,
            } => {
                assert_eq!(*rate_hz, 0.8);
                assert_eq!(*depth_ms, 4.0);
                assert_eq!(*feedback, 0.6);
                assert_eq!(*mix, 0.5);
            }
            other => panic!("expected a Flanger effect, got {other:?}"),
        }
        match &loaded.tracks[0].effects[1] {
            TrackEffectConfig::Phaser {
                rate_hz,
                depth,
                feedback,
                mix,
            } => {
                assert_eq!(*rate_hz, 0.3);
                assert_eq!(*depth, 0.9);
                assert_eq!(*feedback, 0.4);
                assert_eq!(*mix, 0.5);
            }
            other => panic!("expected a Phaser effect, got {other:?}"),
        }
        match &loaded.tracks[0].effects[2] {
            TrackEffectConfig::RingModulator { carrier_hz, mix } => {
                assert_eq!(*carrier_hz, 350.0);
                assert_eq!(*mix, 0.7);
            }
            other => panic!("expected a RingModulator effect, got {other:?}"),
        }
        match &loaded.tracks[0].effects[3] {
            TrackEffectConfig::NoiseGate {
                threshold_db,
                attack_ms,
                release_ms,
                range_db,
                ..
            } => {
                assert_eq!(*threshold_db, -30.0);
                assert_eq!(*attack_ms, 3.0);
                assert_eq!(*release_ms, 200.0);
                assert_eq!(*range_db, -70.0);
            }
            other => panic!("expected a NoiseGate effect, got {other:?}"),
        }
    }

    #[test]
    fn save_then_load_round_trips_channel_eq_bands() {
        let mut song = Song::demo();
        song.tracks[0].effects = vec![TrackEffectConfig::ChannelEq {
            bands: vec![
                EqBand {
                    band_type: EqBandType::HighPass,
                    freq_hz: 45.0,
                    gain_db: 0.0,
                    q: 0.9,
                    enabled: true,
                },
                EqBand {
                    band_type: EqBandType::Peak,
                    freq_hz: 1200.0,
                    gain_db: -4.5,
                    q: 1.4,
                    enabled: false,
                },
            ],
        }];

        let path = std::env::temp_dir().join(format!(
            "simple-daw-test-fx-channel-eq-{}.json",
            std::process::id()
        ));
        song.save_to_file(&path).unwrap();
        let loaded = Song::load_from_file(&path, None).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(loaded.tracks[0].effects.len(), 1);
        match &loaded.tracks[0].effects[0] {
            TrackEffectConfig::ChannelEq { bands } => {
                assert_eq!(bands.len(), 2);
                assert_eq!(bands[0].band_type, EqBandType::HighPass);
                assert_eq!(bands[0].freq_hz, 45.0);
                assert_eq!(bands[0].q, 0.9);
                assert!(bands[0].enabled);
                assert_eq!(bands[1].band_type, EqBandType::Peak);
                assert_eq!(bands[1].gain_db, -4.5);
                assert!(!bands[1].enabled);
            }
            other => panic!("expected a ChannelEq effect, got {other:?}"),
        }
    }

    #[test]
    fn save_then_load_round_trips_track_wide_automation() {
        let mut song = Song::demo();
        song.tracks[0].automation = vec![AutomationLane {
            target: AutomationTarget::OtherTrackVolume { track_index: 1 },
            points: vec![
                AutomationPoint { tick: 0, value: 1.0, curve: CurveShape::default() },
                AutomationPoint { tick: 480, value: 0.25, curve: CurveShape::default() },
            ],
        }];

        let path = std::env::temp_dir().join(format!(
            "simple-daw-test-track-automation-{}.json",
            std::process::id()
        ));
        song.save_to_file(&path).unwrap();
        let loaded = Song::load_from_file(&path, None).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(loaded.tracks[0].automation.len(), 1);
        let lane = &loaded.tracks[0].automation[0];
        assert_eq!(lane.target, AutomationTarget::OtherTrackVolume { track_index: 1 });
        assert_eq!(lane.points.len(), 2);
        assert_eq!(lane.points[1].tick, 480);
        assert_eq!(lane.points[1].value, 0.25);
        // Untouched track: no automation set, none should appear after the round trip.
        assert!(loaded.tracks[1].automation.is_empty());
    }

    #[test]
    fn save_then_load_round_trips_limiter() {
        let mut song = Song::demo();
        song.tracks[0].effects = vec![TrackEffectConfig::Limiter {
            input_gain_db: 6.0,
            ceiling_db: -0.5,
            release_ms: 80.0,
        }];

        let path = std::env::temp_dir().join(format!(
            "simple-daw-test-fx-limiter-{}.json",
            std::process::id()
        ));
        song.save_to_file(&path).unwrap();
        let loaded = Song::load_from_file(&path, None).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(loaded.tracks[0].effects.len(), 1);
        match &loaded.tracks[0].effects[0] {
            TrackEffectConfig::Limiter {
                input_gain_db,
                ceiling_db,
                release_ms,
            } => {
                assert_eq!(*input_gain_db, 6.0);
                assert_eq!(*ceiling_db, -0.5);
                assert_eq!(*release_ms, 80.0);
            }
            other => panic!("expected a Limiter effect, got {other:?}"),
        }
    }

    #[test]
    fn load_from_file_defaults_effect_fields_for_pre_existing_song_files() {
        // A song file saved before effect persistence existed has no `effects` /
        // `master_effect_path` / `master_effect_params` keys at all.
        let json = r#"{
            "name": "Old Song",
            "bpm": 120.0,
            "next_note_id": 0,
            "tracks": [
                {
                    "name": "Drums",
                    "midi_channel": 10,
                    "muted": false,
                    "default_note_length_ticks": 96,
                    "pattern": { "name": "Drums 1", "length_steps": 16, "content": { "StepGrid": [] } }
                }
            ]
        }"#;
        let path = std::env::temp_dir().join(format!(
            "simple-daw-test-legacy-{}.json",
            std::process::id()
        ));
        std::fs::write(&path, json).unwrap();

        let loaded = Song::load_from_file(&path, None).unwrap();
        std::fs::remove_file(&path).ok();

        assert!(loaded.master_effects.is_empty());
        assert!(loaded.tracks[0].effects.is_empty());
        assert_eq!(loaded.tracks[0].volume, 1.0);
        assert!(loaded.tracks[0].automation.is_empty());
    }

    #[test]
    fn load_from_file_defaults_session_clips_for_pre_existing_song_files() {
        // A song file saved before Session View existed has no "session_clips" key at all.
        let json = r#"{
            "name": "Old Song",
            "bpm": 120.0,
            "next_note_id": 0,
            "tracks": [
                {
                    "name": "Drums",
                    "midi_channel": 10,
                    "muted": false,
                    "default_note_length_ticks": 96,
                    "pattern": { "name": "Drums 1", "length_steps": 16, "content": { "StepGrid": [] } }
                }
            ]
        }"#;
        let path = std::env::temp_dir().join(format!(
            "simple-daw-test-legacy-session-clips-{}.json",
            std::process::id()
        ));
        std::fs::write(&path, json).unwrap();

        let loaded = Song::load_from_file(&path, None).unwrap();
        std::fs::remove_file(&path).ok();

        assert!(loaded.tracks[0].session_clips.is_empty());
        assert!(loaded.tracks[0].session_launch_requests.is_empty());
    }

    #[test]
    fn session_clip_round_trips_through_save_and_load() {
        let mut song = Song::demo();
        let region = song.tracks[0].regions[0].clone();
        song.tracks[0].session_clips.push(Some(SessionClip::from_region(&region)));

        let path = std::env::temp_dir().join(format!(
            "simple-daw-test-session-clip-round-trip-{}.json",
            std::process::id()
        ));
        song.save_to_file(&path).unwrap();
        let loaded = Song::load_from_file(&path, None).unwrap();
        std::fs::remove_file(&path).ok();

        let slot = loaded.tracks[0].session_clips[0]
            .as_ref()
            .expect("session clip should round-trip");
        assert_eq!(slot.name, region.name);
        match &slot.content {
            SessionClipContent::Region { content_length_steps, loop_length_steps, .. } => {
                assert_eq!(*content_length_steps, region.content_length_steps);
                assert_eq!(*loop_length_steps, region.loop_length_steps);
            }
            SessionClipContent::Audio(_) => panic!("expected Region content"),
        }
        // Transient click state never round-trips.
        assert!(loaded.tracks[0].session_launch_requests.is_empty());
    }

    #[test]
    fn session_clip_follow_action_and_legato_round_trip() {
        let mut song = Song::demo();
        let region = song.tracks[0].regions[0].clone();
        let mut clip = SessionClip::from_region(&region);
        clip.legato = true;
        clip.follow_action = FollowActionConfig {
            times: 3,
            action_a: FollowAction::Next,
            chance_a: 0.75,
            action_b: FollowAction::Other(2),
            chance_b: 0.25,
        };
        song.tracks[0].session_clips.push(Some(clip));

        let path = std::env::temp_dir().join(format!(
            "simple-daw-test-follow-action-round-trip-{}.json",
            std::process::id()
        ));
        song.save_to_file(&path).unwrap();
        let loaded = Song::load_from_file(&path, None).unwrap();
        std::fs::remove_file(&path).ok();

        let slot = loaded.tracks[0].session_clips[0].as_ref().unwrap();
        assert!(slot.legato);
        assert_eq!(slot.follow_action.times, 3);
        assert_eq!(slot.follow_action.action_a, FollowAction::Next);
        assert_eq!(slot.follow_action.chance_a, 0.75);
        assert_eq!(slot.follow_action.action_b, FollowAction::Other(2));
        assert_eq!(slot.follow_action.chance_b, 0.25);
    }

    #[test]
    fn session_clip_defaults_follow_action_and_legato_for_pre_existing_song_files() {
        // A song file saved before follow actions/legato existed: its one session clip has no
        // "follow_action"/"legato" keys at all.
        let json = r#"{
            "name": "Old Session Song",
            "bpm": 120.0,
            "next_note_id": 0,
            "tracks": [
                {
                    "name": "Drums",
                    "midi_channel": 10,
                    "muted": false,
                    "kind": "StepGrid",
                    "default_note_length_ticks": 96,
                    "session_clips": [
                        {
                            "name": "Slot",
                            "content": { "Region": { "content": { "StepGrid": [] }, "content_length_steps": 4, "loop_length_steps": 4 } }
                        }
                    ]
                }
            ]
        }"#;
        let path = std::env::temp_dir().join(format!(
            "simple-daw-test-legacy-follow-action-{}.json",
            std::process::id()
        ));
        std::fs::write(&path, json).unwrap();

        let loaded = Song::load_from_file(&path, None).unwrap();
        std::fs::remove_file(&path).ok();

        let slot = loaded.tracks[0].session_clips[0].as_ref().unwrap();
        assert!(!slot.legato);
        assert_eq!(slot.follow_action, FollowActionConfig::default());
    }

    #[test]
    fn new_piano_roll_session_clip_starts_blank_with_matching_content_and_loop_length() {
        let clip = SessionClip::new_piano_roll("New Region", 16);
        assert_eq!(clip.name, "New Region");
        match &clip.content {
            SessionClipContent::Region { content, content_length_steps, loop_length_steps } => {
                assert!(matches!(content, RegionContent::PianoRoll(notes) if notes.is_empty()));
                assert_eq!(*content_length_steps, 16);
                assert_eq!(*loop_length_steps, 16);
            }
            SessionClipContent::Audio(_) => panic!("expected Region content"),
        }
    }

    #[test]
    fn new_step_grid_session_clip_starts_with_no_lanes_and_matching_content_and_loop_length() {
        let clip = SessionClip::new_step_grid("New Beats Lane", 16);
        assert_eq!(clip.name, "New Beats Lane");
        match &clip.content {
            SessionClipContent::Region { content, content_length_steps, loop_length_steps } => {
                assert!(matches!(content, RegionContent::StepGrid(lanes) if lanes.is_empty()));
                assert_eq!(*content_length_steps, 16);
                assert_eq!(*loop_length_steps, 16);
            }
            SessionClipContent::Audio(_) => panic!("expected Region content"),
        }
    }

    #[test]
    fn to_region_copies_a_step_grid_clips_content_at_the_given_start_tick() {
        let mut clip = SessionClip::new_step_grid("Drums", 4);
        if let SessionClipContent::Region { content: RegionContent::StepGrid(lanes), .. } =
            &mut clip.content
        {
            lanes.push(Lane::new("Kick", 36, 4));
        }

        let region = clip.to_region(240).expect("step-grid content should convert");
        assert_eq!(region.name, "Drums");
        assert_eq!(region.start_tick, 240);
        assert_eq!(region.content_length_steps, 4);
        assert_eq!(region.loop_length_steps, 4);
        assert!(matches!(&region.content, RegionContent::StepGrid(lanes) if lanes.len() == 1));
        assert!(region.automation.is_empty());
    }

    #[test]
    fn to_region_copies_a_piano_roll_clips_content_at_the_given_start_tick() {
        let mut clip = SessionClip::new_piano_roll("Melody", 8);
        if let SessionClipContent::Region { content: RegionContent::PianoRoll(notes), .. } =
            &mut clip.content
        {
            notes.push(crate::model::Note {
                id: 0,
                pitch: 60,
                start_tick: 0,
                length_ticks: TICKS_PER_STEP,
                velocity: 100,
            });
        }

        let region = clip.to_region(480).expect("piano-roll content should convert");
        assert_eq!(region.name, "Melody");
        assert_eq!(region.start_tick, 480);
        assert_eq!(region.content_length_steps, 8);
        assert_eq!(region.loop_length_steps, 8);
        assert!(matches!(&region.content, RegionContent::PianoRoll(notes) if notes.len() == 1));
        assert!(region.automation.is_empty());
    }

    #[test]
    fn to_region_returns_none_for_audio_content() {
        let clip = SessionClip::from_audio_clip(&AudioClip::new(0, "test.wav"));
        assert!(clip.to_region(0).is_none());
    }

    #[test]
    fn end_of_regions_tick_is_the_furthest_regions_own_end() {
        let mut track = Track::new_piano_roll("Lead", 1);
        assert_eq!(track.end_of_regions_tick(), 0, "an empty track has nothing to be past");

        track.add_region(0, 16);
        let second = track.add_region(32, 16);
        track.regions[second].loop_length_steps = 4;

        // Region 0: start_tick 0, loop_length_steps 16 -> ends at 16 steps.
        // Region 1 (shortened): start_tick 32 steps, loop_length_steps 4 -> ends at 36 steps.
        let expected = (32 + 4) * TICKS_PER_STEP;
        assert_eq!(track.end_of_regions_tick(), expected);
    }

    #[test]
    fn load_from_file_migrates_legacy_bare_velocity_steps_into_step_data() {
        // A song file saved before per-step timing offsets existed: each active step in a
        // lane's `steps` array is a bare velocity byte, not a `{velocity, timing_offset_ticks}`
        // object.
        let json = r#"{
            "name": "Old Song",
            "bpm": 120.0,
            "next_note_id": 0,
            "tracks": [
                {
                    "name": "Drums",
                    "midi_channel": 10,
                    "muted": false,
                    "default_note_length_ticks": 96,
                    "pattern": {
                        "name": "Drums 1",
                        "length_steps": 4,
                        "content": {
                            "StepGrid": [
                                { "name": "Kick", "pitch": 36, "steps": [100, null, null, 64], "sample_path": "" }
                            ]
                        }
                    }
                }
            ]
        }"#;
        let path = std::env::temp_dir().join(format!(
            "simple-daw-test-legacy-steps-{}.json",
            std::process::id()
        ));
        std::fs::write(&path, json).unwrap();

        let loaded = Song::load_from_file(&path, None).unwrap();
        std::fs::remove_file(&path).ok();

        let RegionContent::StepGrid(lanes) = &loaded.tracks[0].regions[0].content else {
            panic!("expected step-grid content");
        };
        assert_eq!(lanes[0].steps[0], Some(StepData { velocity: 100, timing_offset_ticks: 0 }));
        assert_eq!(lanes[0].steps[1], None);
        assert_eq!(lanes[0].steps[3], Some(StepData { velocity: 64, timing_offset_ticks: 0 }));
    }

    #[test]
    fn load_from_file_migrates_old_master_effect_path_into_master_effects_chain() {
        // A song saved before the master bus became a chain: a single top-level
        // `master_effect_path`/`master_effect_params` pair instead of `master_effects`.
        let json = r#"{
            "name": "Old Song",
            "bpm": 120.0,
            "next_note_id": 0,
            "master_effect_path": "/usr/lib64/clap/ZamDelay.clap",
            "master_effect_params": [[0, 0.5]],
            "tracks": []
        }"#;
        let path = std::env::temp_dir().join(format!(
            "simple-daw-test-migrate-master-effect-{}.json",
            std::process::id()
        ));
        std::fs::write(&path, json).unwrap();

        let loaded = Song::load_from_file(&path, None).unwrap();
        std::fs::remove_file(&path).ok();

        match &loaded.master_effects[..] {
            [TrackEffectConfig::Clap { path, params, sidechain_source }] => {
                assert_eq!(path, "/usr/lib64/clap/ZamDelay.clap");
                assert_eq!(params, &vec![(0, 0.5)]);
                assert_eq!(*sidechain_source, None);
            }
            other => panic!("expected a single master Clap effect, got {other:?}"),
        }
    }

    #[test]
    fn load_from_file_fills_in_new_synth_fields_for_songs_saved_before_they_existed() {
        // A song file saved right after the synth got its first 3 fields (waveform/attack/decay),
        // before sustain/release/pulse-width/unison/filter were added — the `synth` key is
        // present but only has the original 3 sub-fields.
        let json = r#"{
            "name": "Old Song",
            "bpm": 120.0,
            "next_note_id": 0,
            "tracks": [
                {
                    "name": "Bass",
                    "midi_channel": 1,
                    "muted": false,
                    "default_note_length_ticks": 96,
                    "pattern": { "name": "Bass 1", "length_steps": 16, "content": { "PianoRoll": [] } },
                    "synth": { "waveform": "Saw", "attack_seconds": 0.01, "decay_seconds": 0.5 }
                }
            ]
        }"#;
        let path = std::env::temp_dir().join(format!(
            "simple-daw-test-legacy-synth-{}.json",
            std::process::id()
        ));
        std::fs::write(&path, json).unwrap();

        let loaded = Song::load_from_file(&path, None).unwrap();
        std::fs::remove_file(&path).ok();

        let synth = &loaded.tracks[0].synth;
        // Fields present in the file keep their saved values...
        assert!(matches!(synth.waveform, SynthWaveform::Saw));
        assert_eq!(synth.attack_seconds, 0.01);
        assert_eq!(synth.decay_seconds, 0.5);
        // ...fields the old file predates fall back to `SynthParams::default()`.
        let default = SynthParams::default();
        assert_eq!(synth.sustain_level, default.sustain_level);
        assert_eq!(synth.release_seconds, default.release_seconds);
        assert_eq!(synth.pulse_width, default.pulse_width);
        assert_eq!(synth.unison_voices, default.unison_voices);
        assert_eq!(synth.filter_cutoff_hz, default.filter_cutoff_hz);
        assert!(matches!(synth.filter_type, FilterType::Lowpass));
        assert_eq!(synth.osc2_mix, default.osc2_mix);
        assert_eq!(synth.sub_osc_mix, default.sub_osc_mix);
        assert!(matches!(synth.lfo_target, LfoTarget::None));
        assert_eq!(synth.glide_seconds, default.glide_seconds);
    }

    #[test]
    fn save_then_load_round_trips_every_synth_param_field() {
        // Every field set to a distinctly non-default value, so a field silently falling back to
        // `SynthParams::default()` on load (e.g. from a missing serde attribute) would be caught,
        // not masked by it coincidentally matching the default.
        let synth = SynthParams {
            waveform: SynthWaveform::Square,
            pulse_width: 0.3,
            unison_voices: 3,
            unison_detune_cents: 25.0,
            unison_width: 0.6,
            attack_seconds: 0.2,
            decay_seconds: 0.6,
            sustain_level: 0.4,
            release_seconds: 0.3,
            filter_cutoff_hz: 800.0,
            filter_resonance: 3.5,
            filter_env_amount_hz: -2000.0,
            filter_type: FilterType::Bandpass,
            osc2_waveform: SynthWaveform::Triangle,
            osc2_semitones: -12,
            osc2_detune_cents: 7.0,
            osc2_mix: 0.65,
            osc2_sync: true,
            sub_osc_mix: 0.4,
            lfo_waveform: SynthWaveform::Saw,
            lfo_rate_hz: 3.5,
            lfo_target: LfoTarget::FilterCutoff,
            lfo_depth: 0.8,
            glide_seconds: 0.12,
        };

        let mut song = Song::demo();
        song.tracks[1].synth = synth;
        let path = std::env::temp_dir().join(format!(
            "simple-daw-test-synth-round-trip-{}.json",
            std::process::id()
        ));
        song.save_to_file(&path).unwrap();
        let loaded = Song::load_from_file(&path, None).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(loaded.tracks[1].synth, synth);
    }

    #[test]
    fn add_note_assigns_increasing_ids() {
        let mut notes = Vec::new();
        let mut next_id = 0u64;
        let a = add_note(&mut notes, &mut next_id, 40, 0, 24, 100);
        let b = add_note(&mut notes, &mut next_id, 43, 24, 24, 100);
        assert_ne!(a, b);
        assert_eq!(notes.len(), 2);
    }

    #[test]
    fn add_note_clears_overlap_on_same_pitch() {
        let mut notes = Vec::new();
        let mut next_id = 0u64;
        add_note(&mut notes, &mut next_id, 40, 0, 48, 100); // covers 0..48
        assert_eq!(notes.len(), 1);

        // Overlaps the first note's tail (24..72 overlaps 0..48 at 24..48).
        let second = add_note(&mut notes, &mut next_id, 40, 24, 48, 100);
        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0].id, second);
    }

    #[test]
    fn add_note_leaves_other_pitches_untouched() {
        let mut notes = Vec::new();
        let mut next_id = 0u64;
        add_note(&mut notes, &mut next_id, 40, 0, 24, 100);
        add_note(&mut notes, &mut next_id, 43, 0, 24, 100);
        assert_eq!(notes.len(), 2);
    }

    #[test]
    fn remove_note_deletes_only_the_matching_id() {
        let mut notes = Vec::new();
        let mut next_id = 0u64;
        let a = add_note(&mut notes, &mut next_id, 40, 0, 24, 100);
        add_note(&mut notes, &mut next_id, 43, 48, 24, 100);
        remove_note(&mut notes, a);
        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0].pitch, 43);
    }

    #[test]
    fn grow_length_to_fit_notes_rounds_up_to_the_next_bar() {
        let mut notes = Vec::new();
        let mut next_id = 0u64;
        // A note ending at tick 17*TICKS_PER_STEP needs 18 steps, which rounds up to 32 (2 bars).
        add_note(
            &mut notes,
            &mut next_id,
            40,
            17 * TICKS_PER_STEP,
            TICKS_PER_STEP,
            100,
        );

        let mut length_steps = 16;
        grow_length_to_fit_notes(&mut length_steps, &notes, 16);
        assert_eq!(length_steps, 32);
    }

    #[test]
    fn grow_length_to_fit_notes_never_shrinks() {
        let notes: Vec<Note> = Vec::new();
        let mut length_steps = 64;
        grow_length_to_fit_notes(&mut length_steps, &notes, 16);
        assert_eq!(length_steps, 64);
    }

    #[test]
    fn clear_overlaps_ignores_different_pitches_and_non_overlapping_spans() {
        let mut notes = Vec::new();
        let mut next_id = 0u64;
        let keep_id = add_note(&mut notes, &mut next_id, 40, 100, 24, 100);
        add_note(&mut notes, &mut next_id, 41, 100, 24, 100); // different pitch
        add_note(&mut notes, &mut next_id, 40, 0, 24, 100); // same pitch, no overlap with 100..124

        clear_overlaps(&mut notes, keep_id, 40, 100, 24);
        assert_eq!(
            notes.len(),
            3,
            "no overlap should mean nothing gets removed"
        );
    }

    #[test]
    fn save_then_load_round_trips_synth_presets() {
        let mut song = Song::demo();
        song.synth_presets.push(SynthPreset {
            name: "Lead Saw".to_string(),
            engine: SynthEngine::Simple,
            params: SynthParams {
                waveform: SynthWaveform::Saw,
                filter_cutoff_hz: 3000.0,
                ..SynthParams::default()
            },
            trine: None,
            wave: None,
        });
        let path = std::env::temp_dir().join(format!(
            "simple-daw-test-presets-{}.json",
            std::process::id()
        ));

        song.save_to_file(&path).unwrap();
        let loaded = Song::load_from_file(&path, None).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(loaded.synth_presets.len(), 1);
        assert_eq!(loaded.synth_presets[0].name, "Lead Saw");
        assert_eq!(loaded.synth_presets[0].params.waveform, SynthWaveform::Saw);
        assert_eq!(loaded.synth_presets[0].params.filter_cutoff_hz, 3000.0);
    }

    #[test]
    fn load_from_file_defaults_synth_presets_for_pre_existing_song_files() {
        let json = r#"{
            "name": "Old Song",
            "bpm": 120.0,
            "next_note_id": 0,
            "tracks": []
        }"#;
        let path = std::env::temp_dir().join(format!(
            "simple-daw-test-legacy-presets-{}.json",
            std::process::id()
        ));
        std::fs::write(&path, json).unwrap();

        let loaded = Song::load_from_file(&path, None).unwrap();
        std::fs::remove_file(&path).ok();

        assert!(loaded.synth_presets.is_empty());
    }

    #[test]
    fn synth_preset_defaults_engine_to_simple_for_pre_existing_preset_files() {
        let json = r#"{
            "name": "Old Preset",
            "params": { "waveform": "Saw" }
        }"#;
        let path = std::env::temp_dir().join(format!(
            "simple-daw-test-legacy-preset-file-{}.json",
            std::process::id()
        ));
        std::fs::write(&path, json).unwrap();

        let loaded = SynthPreset::load_from_file(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(loaded.engine, SynthEngine::Simple);
        assert_eq!(loaded.params.waveform, SynthWaveform::Saw);
        assert!(loaded.trine.is_none());
        assert!(loaded.wave.is_none());
    }

    #[test]
    fn synth_preset_save_then_load_round_trips_as_a_standalone_file() {
        let preset = SynthPreset {
            name: "Pluck Bass".to_string(),
            engine: SynthEngine::Simple,
            params: SynthParams {
                waveform: SynthWaveform::Square,
                pulse_width: 0.3,
                ..SynthParams::default()
            },
            trine: None,
            wave: None,
        };
        let path = std::env::temp_dir().join(format!(
            "simple-daw-test-preset-file-{}.json",
            std::process::id()
        ));

        preset.save_to_file(&path).unwrap();
        let loaded = SynthPreset::load_from_file(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(loaded.name, "Pluck Bass");
        assert_eq!(loaded.params.waveform, SynthWaveform::Square);
        assert_eq!(loaded.params.pulse_width, 0.3);
    }

    #[test]
    fn save_then_load_round_trips_trine_params() {
        let mut song = Song::demo();
        song.tracks[0].synth_engine = SynthEngine::Trine;
        song.tracks[0].trine = TrineParams {
            osc1_waveform: SynthWaveform::Noise,
            osc2_level: 0.4,
            filter_routing: FilterRouting::Series,
            filter2_slope: FilterSlope::Slope24,
            mod_slots: vec![ModSlot {
                source: ModSource::Env2,
                target: ModTarget::FilterCutoff,
                amount: 0.5,
            }],
            ..TrineParams::default()
        };
        let path =
            std::env::temp_dir().join(format!("simple-daw-test-trine-{}.json", std::process::id()));

        song.save_to_file(&path).unwrap();
        let loaded = Song::load_from_file(&path, None).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(loaded.tracks[0].synth_engine, SynthEngine::Trine);
        let trine = &loaded.tracks[0].trine;
        assert_eq!(trine.osc1_waveform, SynthWaveform::Noise);
        assert_eq!(trine.osc2_level, 0.4);
        assert_eq!(trine.filter_routing, FilterRouting::Series);
        assert_eq!(trine.filter2_slope, FilterSlope::Slope24);
        assert_eq!(
            trine.mod_slots,
            vec![ModSlot {
                source: ModSource::Env2,
                target: ModTarget::FilterCutoff,
                amount: 0.5
            }]
        );
    }

    #[test]
    fn save_then_load_round_trips_wave_params() {
        let mut song = Song::demo();
        song.tracks[0].synth_engine = SynthEngine::Wave;
        song.tracks[0].wave = WaveParams {
            osc1_table: WavetableId::Chip,
            osc1_position: 0.6,
            osc2_level: 0.4,
            osc2_warp_mode: WaveWarpMode::Bend,
            osc2_warp_amount: 0.8,
            filter_routing: FilterRouting::Series,
            filter2_slope: FilterSlope::Slope24,
            mod_slots: vec![WaveModSlot {
                source: WaveModSource::Env2,
                target: WaveModTarget::Osc1Position,
                amount: 0.5,
            }],
            ..WaveParams::default()
        };
        let path =
            std::env::temp_dir().join(format!("simple-daw-test-wave-{}.json", std::process::id()));

        song.save_to_file(&path).unwrap();
        let loaded = Song::load_from_file(&path, None).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(loaded.tracks[0].synth_engine, SynthEngine::Wave);
        let wave = &loaded.tracks[0].wave;
        assert_eq!(wave.osc1_table, WavetableId::Chip);
        assert_eq!(wave.osc1_position, 0.6);
        assert_eq!(wave.osc2_level, 0.4);
        assert_eq!(wave.osc2_warp_mode, WaveWarpMode::Bend);
        assert_eq!(wave.osc2_warp_amount, 0.8);
        assert_eq!(wave.filter_routing, FilterRouting::Series);
        assert_eq!(wave.filter2_slope, FilterSlope::Slope24);
        assert_eq!(
            wave.mod_slots,
            vec![WaveModSlot {
                source: WaveModSource::Env2,
                target: WaveModTarget::Osc1Position,
                amount: 0.5
            }]
        );
    }

    #[test]
    fn load_from_file_defaults_to_simple_engine_for_pre_existing_song_files() {
        let json = r#"{
            "name": "Old Song",
            "bpm": 120.0,
            "next_note_id": 0,
            "tracks": [
                {
                    "name": "Drums",
                    "midi_channel": 10,
                    "muted": false,
                    "default_note_length_ticks": 96,
                    "pattern": { "name": "Drums 1", "length_steps": 16, "content": { "StepGrid": [] } }
                }
            ]
        }"#;
        let path = std::env::temp_dir().join(format!(
            "simple-daw-test-legacy-engine-{}.json",
            std::process::id()
        ));
        std::fs::write(&path, json).unwrap();

        let loaded = Song::load_from_file(&path, None).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(loaded.tracks[0].synth_engine, SynthEngine::Simple);
        assert_eq!(loaded.tracks[0].trine.mod_slots.len(), 0);
    }

    #[test]
    fn load_from_file_migrates_a_legacy_two_track_song_into_one_looping_pattern() {
        // Pre-global-patterns save: each track owned its own independent `pattern`, and there was
        // no top-level "patterns"/"arrangement" key at all.
        let json = r#"{
            "name": "Old Song",
            "bpm": 120.0,
            "next_note_id": 3,
            "tracks": [
                {
                    "name": "Drums",
                    "midi_channel": 10,
                    "muted": false,
                    "default_note_length_ticks": 96,
                    "pattern": { "name": "Drums 1", "length_steps": 16, "content": { "StepGrid": [] } }
                },
                {
                    "name": "Bass",
                    "midi_channel": 1,
                    "muted": false,
                    "default_note_length_ticks": 96,
                    "pattern": { "name": "Bass 1", "length_steps": 32, "content": { "PianoRoll": [] } }
                }
            ]
        }"#;
        let path = std::env::temp_dir().join(format!(
            "simple-daw-test-migrate-{}.json",
            std::process::id()
        ));
        std::fs::write(&path, json).unwrap();

        let loaded = Song::load_from_file(&path, None).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(loaded.tracks[0].kind, TrackKind::StepGrid);
        assert_eq!(loaded.tracks[1].kind, TrackKind::PianoRoll);

        // Each track gets one region reproducing its old standalone pattern, sized to the longest
        // of the old per-track pattern lengths (32) — with the whole song looping end-to-end (see
        // `audio.rs`), a region that long on both tracks reproduces the old "loop forever" behavior.
        assert_eq!(loaded.tracks[0].regions.len(), 1);
        assert_eq!(loaded.tracks[0].regions[0].start_tick, 0);
        assert_eq!(loaded.tracks[0].regions[0].loop_length_steps, 32);
        assert!(matches!(
            loaded.tracks[0].regions[0].content,
            RegionContent::StepGrid(_)
        ));
        assert_eq!(loaded.tracks[1].regions.len(), 1);
        assert!(matches!(
            loaded.tracks[1].regions[0].content,
            RegionContent::PianoRoll(_)
        ));
    }

    #[test]
    fn load_from_file_migrates_a_patterns_era_song_into_independent_regions() {
        // Patterns-era save: a top-level "patterns" library plus "arrangement" clips placing them
        // — one clip scoped to just the Bass track (`track_index`), one unscoped (plays every
        // track). Two clips referencing the same pattern should become two independent regions.
        let json = r#"{
            "name": "Old Song",
            "bpm": 120.0,
            "next_note_id": 0,
            "tracks": [
                { "name": "Drums", "midi_channel": 10, "muted": false, "kind": "StepGrid", "default_note_length_ticks": 96 },
                { "name": "Bass", "midi_channel": 1, "muted": false, "kind": "PianoRoll", "default_note_length_ticks": 96 }
            ],
            "patterns": [
                { "name": "Pattern 1", "length_steps": 16, "track_contents": [{ "StepGrid": [] }, { "PianoRoll": [] }] }
            ],
            "arrangement": [
                { "pattern_index": 0, "start_step": 0, "length_steps": 16, "lane": 0 },
                { "pattern_index": 0, "start_step": 16, "length_steps": 16, "lane": 1, "track_index": 1 }
            ]
        }"#;
        let path = std::env::temp_dir().join(format!(
            "simple-daw-test-migrate-patterns-era-{}.json",
            std::process::id()
        ));
        std::fs::write(&path, json).unwrap();

        let loaded = Song::load_from_file(&path, None).unwrap();
        std::fs::remove_file(&path).ok();

        // The unscoped clip becomes one region per track (Drums + Bass); the Bass-scoped clip
        // becomes a second, independent region on Bass only.
        assert_eq!(loaded.tracks[0].regions.len(), 1);
        assert_eq!(loaded.tracks[0].regions[0].start_tick, 0);
        assert_eq!(loaded.tracks[1].regions.len(), 2);
        assert_eq!(loaded.tracks[1].regions[0].start_tick, 0);
        assert_eq!(loaded.tracks[1].regions[1].start_tick, 16 * TICKS_PER_STEP);
    }

    #[test]
    fn bpm_at_falls_back_to_base_bpm_with_no_tempo_map() {
        let song = Song::demo();
        assert_eq!(song.bpm_at(0), song.bpm);
        assert_eq!(song.bpm_at(100_000), song.bpm);
    }

    #[test]
    fn bpm_at_holds_each_points_bpm_until_the_next_one() {
        let mut song = Song::demo();
        song.bpm = 100.0;
        song.set_tempo_at(1000, 140.0);
        song.set_tempo_at(2000, 80.0);

        assert_eq!(song.bpm_at(0), 100.0);
        assert_eq!(song.bpm_at(999), 100.0);
        assert_eq!(song.bpm_at(1000), 140.0);
        assert_eq!(song.bpm_at(1500), 140.0);
        assert_eq!(song.bpm_at(2000), 80.0);
        assert_eq!(song.bpm_at(50_000), 80.0);
    }

    #[test]
    fn set_tempo_at_keeps_the_map_sorted_regardless_of_insertion_order() {
        let mut song = Song::demo();
        song.set_tempo_at(2000, 80.0);
        song.set_tempo_at(1000, 140.0);
        let ticks: Vec<usize> = song.tempo_map.iter().map(|p| p.tick).collect();
        assert_eq!(ticks, vec![1000, 2000]);
    }

    #[test]
    fn ensure_scene_grows_the_vec_and_returns_a_default_entry_beyond_its_current_length() {
        let mut song = Song::demo();
        assert!(song.scenes.is_empty());
        let scene = song.ensure_scene(2);
        assert_eq!(scene.name, "");
        assert_eq!(scene.tempo, None);
        assert_eq!(song.scenes.len(), 3);
    }

    #[test]
    fn ensure_scene_returns_the_existing_entry_without_resetting_it() {
        let mut song = Song::demo();
        song.ensure_scene(0).tempo = Some(140.0);
        assert_eq!(song.ensure_scene(0).tempo, Some(140.0));
        assert_eq!(song.scenes.len(), 1);
    }

    #[test]
    fn set_tempo_at_updates_an_existing_point_instead_of_duplicating_it() {
        let mut song = Song::demo();
        song.set_tempo_at(1000, 140.0);
        song.set_tempo_at(1000, 150.0);
        assert_eq!(song.tempo_map.len(), 1);
        assert_eq!(song.tempo_map[0].bpm, 150.0);
    }

    #[test]
    fn remove_tempo_point_drops_only_the_given_index() {
        let mut song = Song::demo();
        song.set_tempo_at(1000, 140.0);
        song.set_tempo_at(2000, 80.0);
        song.remove_tempo_point(0);
        assert_eq!(song.tempo_map.len(), 1);
        assert_eq!(song.tempo_map[0].tick, 2000);
    }

    #[test]
    fn add_track_returns_its_index_and_starts_with_no_regions() {
        let mut song = Song::demo();
        assert_eq!(song.tracks.len(), 2);

        let new_index = song.add_track("Lead", 2, TrackKind::PianoRoll);
        assert_eq!(new_index, 2);
        assert_eq!(song.tracks.len(), 3);
        assert!(song.tracks[new_index].regions.is_empty());
    }

    #[test]
    fn remove_track_drops_its_own_regions_and_only_its_own() {
        let mut song = Song::demo();
        assert_eq!(song.tracks[0].regions.len(), 1); // Drums
        assert_eq!(song.tracks[1].regions.len(), 1); // Bass

        song.remove_track(0);

        assert_eq!(song.tracks.len(), 1);
        // The surviving track is what used to be track index 1 (Bass) — its own region, and only
        // its own, comes with it; nothing referenced Drums' region, so there's nothing to fix up.
        assert_eq!(song.tracks[0].kind, TrackKind::PianoRoll);
        assert_eq!(song.tracks[0].regions.len(), 1);
        assert!(matches!(
            song.tracks[0].regions[0].content,
            RegionContent::PianoRoll(_)
        ));
    }

    #[test]
    fn remove_submix_routes_its_member_tracks_back_to_master() {
        let mut song = Song::demo();
        let submix_index = song.add_submix("Drum Bus");
        song.tracks[0].output = TrackOutput::Submix(submix_index);

        song.remove_submix(submix_index);

        assert!(song.submixes.is_empty());
        assert_eq!(song.tracks[0].output, TrackOutput::Master);
    }

    #[test]
    fn remove_submix_shifts_down_indices_of_tracks_routed_past_it() {
        let mut song = Song::demo();
        let first = song.add_submix("Drum Bus");
        let second = song.add_submix("Vocal Bus");
        song.tracks[0].output = TrackOutput::Submix(first);
        song.tracks[1].output = TrackOutput::Submix(second);

        song.remove_submix(first);

        assert_eq!(song.submixes.len(), 1);
        assert_eq!(song.submixes[0].name, "Vocal Bus");
        // Track 0 pointed at the removed submix, so it falls back to Master.
        assert_eq!(song.tracks[0].output, TrackOutput::Master);
        // Track 1 pointed past the removed submix, so its index shifts down to stay valid.
        assert_eq!(song.tracks[1].output, TrackOutput::Submix(0));
    }

    #[test]
    fn add_region_copies_step_grid_lane_layout_from_the_previous_region() {
        let mut song = Song::demo();
        let new_region = song.tracks[0].add_region(32, 16);

        let RegionContent::StepGrid(lanes) = &song.tracks[0].regions[new_region].content else {
            panic!("expected StepGrid content for a new region on the Drums track");
        };
        // Same lane names/pitches as the demo's first region on this track, but with fresh
        // (empty) steps.
        assert_eq!(lanes.len(), 16);
        assert_eq!(lanes[0].name, "Kick");
        assert_eq!(lanes[0].pitch, 36);
        assert!(lanes[0].steps.iter().all(|s| s.is_none()));
    }

    #[test]
    fn add_region_positions_it_at_the_given_step() {
        let mut song = Song::demo();
        let new_region = song.tracks[1].add_region(48, 16);

        assert_eq!(
            song.tracks[1].regions[new_region].start_tick,
            48 * TICKS_PER_STEP
        );
        assert_eq!(song.tracks[1].regions[new_region].loop_length_steps, 16);
    }
}
