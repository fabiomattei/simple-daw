//! Native DSP effects that need no external plugin file, as an alternative to CLAP hosting
//! (`plugin_host.rs`) in a track's insert chain. Each effect lives in its own submodule as a plain
//! Rust struct with public parameter fields (so the UI can bind sliders straight to them, the same
//! way `LoadedEffect`'s CLAP params are edited) plus whatever private DSP state it needs, and a
//! `process(&mut self, l: &mut [f32], r: &mut [f32])` method that runs it in place over one
//! stereo audio block.
//!
//! Every effect is **dual-mono**: it holds two independent copies of its internal state (delay
//! buffers, filters, envelope followers, LFO phase, ...) and runs the exact same per-channel
//! algorithm on `l` with one copy and on `r` with the other, sharing only its public parameters
//! and sample rate. LFO-driven effects (chorus, flanger, phaser, tremolo, ring mod) still end up
//! phase-locked across channels because both copies start from the same initial phase and advance
//! by the same increment every block — the width comes entirely from each channel's own audio
//! history, not from a deliberate L/R phase or tuning offset.

mod bitcrusher;
mod channel_eq;
mod chorus;
mod compressor;
mod delay;
mod distortion;
mod filter;
mod flanger;
mod limiter;
mod noise_gate;
mod phase_invert;
mod phaser;
mod reverb;
mod ring_modulator;
mod tremolo;

use crate::model::TrackEffectConfig;
// Re-exported (not just `use`d) so each leaf effect type is reachable as `builtin_fx::XEffect` —
// their own submodules stay private, but `main.rs` pattern-matches `BuiltInEffect`'s variants and
// reads/writes the bound struct's `pub` fields directly to build each effect's UI sliders.
pub(crate) use bitcrusher::BitcrusherEffect;
pub(crate) use channel_eq::ChannelEqEffect;
pub(crate) use chorus::ChorusEffect;
pub(crate) use compressor::CompressorEffect;
pub(crate) use delay::DelayEffect;
pub(crate) use distortion::DistortionEffect;
pub(crate) use filter::FilterEffect;
pub(crate) use flanger::FlangerEffect;
pub(crate) use limiter::LimiterEffect;
pub(crate) use noise_gate::NoiseGateEffect;
pub(crate) use phase_invert::PhaseInvertEffect;
pub(crate) use phaser::PhaserEffect;
pub(crate) use reverb::ReverbEffect;
pub(crate) use ring_modulator::RingModulatorEffect;
pub(crate) use tremolo::TremoloEffect;

/// A live, running built-in effect, owning whatever DSP state it needs (ring buffers, filter
/// state) across successive `process` calls. Mirrors `plugin_host::LoadedEffect`'s role for CLAP
/// effects, but construction is infallible (given a sample rate) since there's no external file
/// or plugin activation involved.
pub enum BuiltInEffect {
    Delay(DelayEffect),
    Bitcrusher(BitcrusherEffect),
    Distortion(DistortionEffect),
    Reverb(ReverbEffect),
    Chorus(ChorusEffect),
    Filter(FilterEffect),
    Tremolo(TremoloEffect),
    Compressor(CompressorEffect),
    Flanger(FlangerEffect),
    Phaser(PhaserEffect),
    RingModulator(RingModulatorEffect),
    NoiseGate(NoiseGateEffect),
    PhaseInvert(PhaseInvertEffect),
    ChannelEq(ChannelEqEffect),
    Limiter(LimiterEffect),
}

impl BuiltInEffect {
    /// Builds a live effect from its saved/edited parameters. `None` only if `config` is actually
    /// `TrackEffectConfig::Clap` — callers are expected to route CLAP configs through
    /// `plugin_host::load_and_activate` instead; this is a defensive fallback, not a normal path.
    pub fn from_config(config: &TrackEffectConfig, sample_rate: f32) -> Option<Self> {
        Some(match config {
            TrackEffectConfig::Delay {
                time_ms,
                feedback,
                mix,
            } => BuiltInEffect::Delay(DelayEffect::new(*time_ms, *feedback, *mix, sample_rate)),
            TrackEffectConfig::Bitcrusher {
                bit_depth,
                rate_divisor,
                mix,
            } => BuiltInEffect::Bitcrusher(BitcrusherEffect::new(*bit_depth, *rate_divisor, *mix)),
            TrackEffectConfig::Distortion { drive, mix } => {
                BuiltInEffect::Distortion(DistortionEffect::new(*drive, *mix))
            }
            TrackEffectConfig::Reverb {
                room_size,
                damping,
                mix,
            } => BuiltInEffect::Reverb(ReverbEffect::new(*room_size, *damping, *mix, sample_rate)),
            TrackEffectConfig::Chorus {
                rate_hz,
                depth_ms,
                mix,
            } => BuiltInEffect::Chorus(ChorusEffect::new(*rate_hz, *depth_ms, *mix, sample_rate)),
            TrackEffectConfig::Filter {
                cutoff_hz,
                resonance,
                mode,
                mix,
            } => BuiltInEffect::Filter(FilterEffect::new(
                *cutoff_hz,
                *resonance,
                *mode,
                *mix,
                sample_rate,
            )),
            TrackEffectConfig::Tremolo { rate_hz, depth } => {
                BuiltInEffect::Tremolo(TremoloEffect::new(*rate_hz, *depth, sample_rate))
            }
            TrackEffectConfig::Compressor {
                threshold_db,
                ratio,
                attack_ms,
                release_ms,
                makeup_db,
            } => BuiltInEffect::Compressor(CompressorEffect::new(
                *threshold_db,
                *ratio,
                *attack_ms,
                *release_ms,
                *makeup_db,
                sample_rate,
            )),
            TrackEffectConfig::Flanger {
                rate_hz,
                depth_ms,
                feedback,
                mix,
            } => BuiltInEffect::Flanger(FlangerEffect::new(
                *rate_hz,
                *depth_ms,
                *feedback,
                *mix,
                sample_rate,
            )),
            TrackEffectConfig::Phaser {
                rate_hz,
                depth,
                feedback,
                mix,
            } => BuiltInEffect::Phaser(PhaserEffect::new(
                *rate_hz,
                *depth,
                *feedback,
                *mix,
                sample_rate,
            )),
            TrackEffectConfig::RingModulator { carrier_hz, mix } => BuiltInEffect::RingModulator(
                RingModulatorEffect::new(*carrier_hz, *mix, sample_rate),
            ),
            TrackEffectConfig::NoiseGate {
                threshold_db,
                attack_ms,
                release_ms,
                range_db,
            } => BuiltInEffect::NoiseGate(NoiseGateEffect::new(
                *threshold_db,
                *attack_ms,
                *release_ms,
                *range_db,
                sample_rate,
            )),
            TrackEffectConfig::PhaseInvert {
                invert_left,
                invert_right,
            } => BuiltInEffect::PhaseInvert(PhaseInvertEffect::new(*invert_left, *invert_right)),
            TrackEffectConfig::ChannelEq { bands } => {
                BuiltInEffect::ChannelEq(ChannelEqEffect::new(bands.clone(), sample_rate))
            }
            TrackEffectConfig::Limiter {
                input_gain_db,
                ceiling_db,
                release_ms,
            } => BuiltInEffect::Limiter(LimiterEffect::new(
                *input_gain_db,
                *ceiling_db,
                *release_ms,
                sample_rate,
            )),
            TrackEffectConfig::Clap { .. } => return None,
        })
    }

    /// Snapshots this effect's current parameter values for persisting to a song file (see
    /// `main.rs`'s `sync_song_effects`) — the counterpart of `LoadedEffect::param_snapshot` for
    /// CLAP effects.
    pub fn to_config(&self) -> TrackEffectConfig {
        match self {
            BuiltInEffect::Delay(e) => TrackEffectConfig::Delay {
                time_ms: e.time_ms,
                feedback: e.feedback,
                mix: e.mix,
            },
            BuiltInEffect::Bitcrusher(e) => TrackEffectConfig::Bitcrusher {
                bit_depth: e.bit_depth,
                rate_divisor: e.rate_divisor,
                mix: e.mix,
            },
            BuiltInEffect::Distortion(e) => TrackEffectConfig::Distortion {
                drive: e.drive,
                mix: e.mix,
            },
            BuiltInEffect::Reverb(e) => TrackEffectConfig::Reverb {
                room_size: e.room_size,
                damping: e.damping,
                mix: e.mix,
            },
            BuiltInEffect::Chorus(e) => TrackEffectConfig::Chorus {
                rate_hz: e.rate_hz,
                depth_ms: e.depth_ms,
                mix: e.mix,
            },
            BuiltInEffect::Filter(e) => TrackEffectConfig::Filter {
                cutoff_hz: e.cutoff_hz,
                resonance: e.resonance,
                mode: e.mode,
                mix: e.mix,
            },
            BuiltInEffect::Tremolo(e) => TrackEffectConfig::Tremolo {
                rate_hz: e.rate_hz,
                depth: e.depth,
            },
            BuiltInEffect::Compressor(e) => TrackEffectConfig::Compressor {
                threshold_db: e.threshold_db,
                ratio: e.ratio,
                attack_ms: e.attack_ms,
                release_ms: e.release_ms,
                makeup_db: e.makeup_db,
            },
            BuiltInEffect::Flanger(e) => TrackEffectConfig::Flanger {
                rate_hz: e.rate_hz,
                depth_ms: e.depth_ms,
                feedback: e.feedback,
                mix: e.mix,
            },
            BuiltInEffect::Phaser(e) => TrackEffectConfig::Phaser {
                rate_hz: e.rate_hz,
                depth: e.depth,
                feedback: e.feedback,
                mix: e.mix,
            },
            BuiltInEffect::RingModulator(e) => TrackEffectConfig::RingModulator {
                carrier_hz: e.carrier_hz,
                mix: e.mix,
            },
            BuiltInEffect::NoiseGate(e) => TrackEffectConfig::NoiseGate {
                threshold_db: e.threshold_db,
                attack_ms: e.attack_ms,
                release_ms: e.release_ms,
                range_db: e.range_db,
            },
            BuiltInEffect::PhaseInvert(e) => TrackEffectConfig::PhaseInvert {
                invert_left: e.invert_left,
                invert_right: e.invert_right,
            },
            BuiltInEffect::ChannelEq(e) => TrackEffectConfig::ChannelEq {
                bands: e.bands.clone(),
            },
            BuiltInEffect::Limiter(e) => TrackEffectConfig::Limiter {
                input_gain_db: e.input_gain_db,
                ceiling_db: e.ceiling_db,
                release_ms: e.release_ms,
            },
        }
    }

    /// Short label for the FX chain row UI (see `main.rs`'s `track_ui`).
    pub fn label(&self) -> &'static str {
        match self {
            BuiltInEffect::Delay(_) => "Delay",
            BuiltInEffect::Bitcrusher(_) => "Bitcrusher",
            BuiltInEffect::Distortion(_) => "Distortion",
            BuiltInEffect::Reverb(_) => "Reverb",
            BuiltInEffect::Chorus(_) => "Chorus",
            BuiltInEffect::Filter(_) => "Filter",
            BuiltInEffect::Tremolo(_) => "Tremolo",
            BuiltInEffect::Compressor(_) => "Compressor",
            BuiltInEffect::Flanger(_) => "Flanger",
            BuiltInEffect::Phaser(_) => "Phaser",
            BuiltInEffect::RingModulator(_) => "Ring Mod",
            BuiltInEffect::NoiseGate(_) => "Noise Gate",
            BuiltInEffect::PhaseInvert(_) => "Phase Invert",
            BuiltInEffect::ChannelEq(_) => "Channel EQ",
            BuiltInEffect::Limiter(_) => "Limiter",
        }
    }

    /// Runs this effect over one stereo audio block, in place — each channel processed
    /// independently through its own internal state (see this module's doc for the dual-mono
    /// pattern every effect follows).
    pub fn process(&mut self, l: &mut [f32], r: &mut [f32]) {
        match self {
            BuiltInEffect::Delay(e) => e.process(l, r),
            BuiltInEffect::Bitcrusher(e) => e.process(l, r),
            BuiltInEffect::Distortion(e) => e.process(l, r),
            BuiltInEffect::Reverb(e) => e.process(l, r),
            BuiltInEffect::Chorus(e) => e.process(l, r),
            BuiltInEffect::Filter(e) => e.process(l, r),
            BuiltInEffect::Tremolo(e) => e.process(l, r),
            BuiltInEffect::Compressor(e) => e.process(l, r),
            BuiltInEffect::Flanger(e) => e.process(l, r),
            BuiltInEffect::Phaser(e) => e.process(l, r),
            BuiltInEffect::RingModulator(e) => e.process(l, r),
            BuiltInEffect::NoiseGate(e) => e.process(l, r),
            BuiltInEffect::PhaseInvert(e) => e.process(l, r),
            BuiltInEffect::ChannelEq(e) => e.process(l, r),
            BuiltInEffect::Limiter(e) => e.process(l, r),
        }
    }
}
