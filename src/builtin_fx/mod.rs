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
                sidechain_source,
            } => {
                let mut effect = CompressorEffect::new(
                    *threshold_db,
                    *ratio,
                    *attack_ms,
                    *release_ms,
                    *makeup_db,
                    sample_rate,
                );
                effect.sidechain_source = *sidechain_source;
                BuiltInEffect::Compressor(effect)
            }
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
                sidechain_source,
            } => {
                let mut effect =
                    NoiseGateEffect::new(*threshold_db, *attack_ms, *release_ms, *range_db, sample_rate);
                effect.sidechain_source = *sidechain_source;
                BuiltInEffect::NoiseGate(effect)
            }
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
                sidechain_source: e.sidechain_source,
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
                sidechain_source: e.sidechain_source,
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

    /// Live-sets one of `automatable_params_for_config`'s names on this effect (clamped to its
    /// declared range) — the real-time-safe write path automation uses every buffer (see
    /// `audio.rs`'s mixdown). Silently ignores an unknown name rather than panicking: an
    /// automation lane's saved param name isn't revalidated against whatever effect actually ended
    /// up loaded in that chain slot (e.g. after swapping which effect is loaded there).
    pub fn set_automatable_param(&mut self, name: &str, value: f32) {
        let clamp = |lo: f32, hi: f32| value.clamp(lo, hi);
        match self {
            BuiltInEffect::Delay(e) => match name {
                "Time" => e.time_ms = clamp(1.0, 2000.0),
                "Feedback" => e.feedback = clamp(0.0, 0.95),
                "Mix" => e.mix = clamp(0.0, 1.0),
                _ => {}
            },
            BuiltInEffect::Bitcrusher(e) => match name {
                "Bit depth" => e.bit_depth = clamp(1.0, 16.0),
                "Rate divisor" => e.rate_divisor = clamp(1.0, 50.0).round() as u32,
                "Mix" => e.mix = clamp(0.0, 1.0),
                _ => {}
            },
            BuiltInEffect::Distortion(e) => match name {
                "Drive" => e.drive = clamp(1.0, 20.0),
                "Mix" => e.mix = clamp(0.0, 1.0),
                _ => {}
            },
            BuiltInEffect::Reverb(e) => match name {
                "Room size" => e.room_size = clamp(0.0, 1.0),
                "Damping" => e.damping = clamp(0.0, 1.0),
                "Mix" => e.mix = clamp(0.0, 1.0),
                _ => {}
            },
            BuiltInEffect::Chorus(e) => match name {
                "Rate" => e.rate_hz = clamp(0.05, 10.0),
                "Depth" => e.depth_ms = clamp(0.0, 30.0),
                "Mix" => e.mix = clamp(0.0, 1.0),
                _ => {}
            },
            BuiltInEffect::Filter(e) => match name {
                "Cutoff" => e.cutoff_hz = clamp(20.0, 18000.0),
                "Resonance" => e.resonance = clamp(0.0, 0.99),
                "Mix" => e.mix = clamp(0.0, 1.0),
                _ => {}
            },
            BuiltInEffect::Tremolo(e) => match name {
                "Rate" => e.rate_hz = clamp(0.1, 20.0),
                "Depth" => e.depth = clamp(0.0, 1.0),
                _ => {}
            },
            BuiltInEffect::Compressor(e) => match name {
                "Threshold" => e.threshold_db = clamp(-60.0, 0.0),
                "Ratio" => e.ratio = clamp(1.0, 20.0),
                "Attack" => e.attack_ms = clamp(0.1, 200.0),
                "Release" => e.release_ms = clamp(5.0, 1000.0),
                "Makeup" => e.makeup_db = clamp(0.0, 24.0),
                _ => {}
            },
            BuiltInEffect::Flanger(e) => match name {
                "Rate" => e.rate_hz = clamp(0.05, 5.0),
                "Depth" => e.depth_ms = clamp(0.0, 10.0),
                "Feedback" => e.feedback = clamp(0.0, 0.95),
                "Mix" => e.mix = clamp(0.0, 1.0),
                _ => {}
            },
            BuiltInEffect::Phaser(e) => match name {
                "Rate" => e.rate_hz = clamp(0.05, 5.0),
                "Depth" => e.depth = clamp(0.0, 1.0),
                "Feedback" => e.feedback = clamp(0.0, 0.95),
                "Mix" => e.mix = clamp(0.0, 1.0),
                _ => {}
            },
            BuiltInEffect::RingModulator(e) => match name {
                "Carrier" => e.carrier_hz = clamp(20.0, 5000.0),
                "Mix" => e.mix = clamp(0.0, 1.0),
                _ => {}
            },
            BuiltInEffect::NoiseGate(e) => match name {
                "Threshold" => e.threshold_db = clamp(-80.0, 0.0),
                "Attack" => e.attack_ms = clamp(0.1, 200.0),
                "Release" => e.release_ms = clamp(5.0, 2000.0),
                "Range" => e.range_db = clamp(-80.0, 0.0),
                _ => {}
            },
            BuiltInEffect::PhaseInvert(_) | BuiltInEffect::ChannelEq(_) => {}
            BuiltInEffect::Limiter(e) => match name {
                "Input gain" => e.input_gain_db = clamp(-24.0, 24.0),
                "Ceiling" => e.ceiling_db = clamp(-12.0, 0.0),
                "Release" => e.release_ms = clamp(5.0, 1000.0),
                _ => {}
            },
        }
    }

    /// Runs this effect over one stereo audio block, in place — each channel processed
    /// independently through its own internal state (see this module's doc for the dual-mono
    /// pattern every effect follows). `sidechain`, a caller-supplied key signal (e.g. another
    /// track routed in for ducking), drives the envelope follower instead of `l`/`r`'s own —
    /// honored only by `Compressor` and `NoiseGate`, the two effect kinds whose gain is
    /// envelope-driven; every other effect kind ignores `sidechain`.
    pub fn process_with_sidechain(
        &mut self,
        l: &mut [f32],
        r: &mut [f32],
        sidechain: Option<(&[f32], &[f32])>,
    ) {
        match self {
            BuiltInEffect::Delay(e) => e.process(l, r),
            BuiltInEffect::Bitcrusher(e) => e.process(l, r),
            BuiltInEffect::Distortion(e) => e.process(l, r),
            BuiltInEffect::Reverb(e) => e.process(l, r),
            BuiltInEffect::Chorus(e) => e.process(l, r),
            BuiltInEffect::Filter(e) => e.process(l, r),
            BuiltInEffect::Tremolo(e) => e.process(l, r),
            BuiltInEffect::Compressor(e) => e.process_with_sidechain(l, r, sidechain),
            BuiltInEffect::Flanger(e) => e.process(l, r),
            BuiltInEffect::Phaser(e) => e.process(l, r),
            BuiltInEffect::RingModulator(e) => e.process(l, r),
            BuiltInEffect::NoiseGate(e) => e.process_with_sidechain(l, r, sidechain),
            BuiltInEffect::PhaseInvert(e) => e.process(l, r),
            BuiltInEffect::ChannelEq(e) => e.process(l, r),
            BuiltInEffect::Limiter(e) => e.process(l, r),
        }
    }

    /// This slot's configured sidechain key source (a `Song::tracks` index), for the two effect
    /// kinds whose gain is envelope-driven — `None` for every other kind, which has nowhere to
    /// use a key signal even if one were routed to it.
    pub fn sidechain_source(&self) -> Option<usize> {
        match self {
            BuiltInEffect::Compressor(e) => e.sidechain_source,
            BuiltInEffect::NoiseGate(e) => e.sidechain_source,
            _ => None,
        }
    }

    /// Mutable access to `sidechain_source`, for the FX chain UI's sidechain-source picker.
    /// `None` for every effect kind that doesn't carry a `sidechain_source` field at all.
    pub fn sidechain_source_mut(&mut self) -> Option<&mut Option<usize>> {
        match self {
            BuiltInEffect::Compressor(e) => Some(&mut e.sidechain_source),
            BuiltInEffect::NoiseGate(e) => Some(&mut e.sidechain_source),
            _ => None,
        }
    }
}

/// Every parameter name automation can target on the effect kind `config` names, in the same
/// order `main.rs`'s `built_in_effect_params_ui` shows their sliders, paired with that parameter's
/// declared range (used both for the automation lane point editor's value axis and to clamp
/// incoming automated values in `BuiltInEffect::set_automatable_param`). A "shape" parameter that
/// isn't a single ramping number (`Filter::mode`, `ChannelEq::bands`) has no entry here — it stays
/// manual-only, edited through `built_in_effect_params_ui`. Empty for `TrackEffectConfig::Clap`
/// (a CLAP plugin's parameters come from `plugin_host::PluginParamInfo` instead, once loaded).
///
/// Takes the saved `TrackEffectConfig` rather than a live `BuiltInEffect` instance so UI code (the
/// automation lane target picker) can query a chain slot's available parameters without spinning
/// up a throwaway effect instance (with its own delay lines/filter state) on every frame just to
/// ask what its parameters are called.
pub fn automatable_params_for_config(config: &TrackEffectConfig) -> &'static [(&'static str, f32, f32)] {
    match config {
        TrackEffectConfig::Clap { .. } => &[],
        TrackEffectConfig::Delay { .. } => {
            &[("Time", 1.0, 2000.0), ("Feedback", 0.0, 0.95), ("Mix", 0.0, 1.0)]
        }
        TrackEffectConfig::Bitcrusher { .. } => {
            &[("Bit depth", 1.0, 16.0), ("Rate divisor", 1.0, 50.0), ("Mix", 0.0, 1.0)]
        }
        TrackEffectConfig::Distortion { .. } => &[("Drive", 1.0, 20.0), ("Mix", 0.0, 1.0)],
        TrackEffectConfig::Reverb { .. } => {
            &[("Room size", 0.0, 1.0), ("Damping", 0.0, 1.0), ("Mix", 0.0, 1.0)]
        }
        TrackEffectConfig::Chorus { .. } => {
            &[("Rate", 0.05, 10.0), ("Depth", 0.0, 30.0), ("Mix", 0.0, 1.0)]
        }
        TrackEffectConfig::Filter { .. } => {
            &[("Cutoff", 20.0, 18000.0), ("Resonance", 0.0, 0.99), ("Mix", 0.0, 1.0)]
        }
        TrackEffectConfig::Tremolo { .. } => &[("Rate", 0.1, 20.0), ("Depth", 0.0, 1.0)],
        TrackEffectConfig::Compressor { .. } => &[
            ("Threshold", -60.0, 0.0),
            ("Ratio", 1.0, 20.0),
            ("Attack", 0.1, 200.0),
            ("Release", 5.0, 1000.0),
            ("Makeup", 0.0, 24.0),
        ],
        TrackEffectConfig::Flanger { .. } => &[
            ("Rate", 0.05, 5.0),
            ("Depth", 0.0, 10.0),
            ("Feedback", 0.0, 0.95),
            ("Mix", 0.0, 1.0),
        ],
        TrackEffectConfig::Phaser { .. } => &[
            ("Rate", 0.05, 5.0),
            ("Depth", 0.0, 1.0),
            ("Feedback", 0.0, 0.95),
            ("Mix", 0.0, 1.0),
        ],
        TrackEffectConfig::RingModulator { .. } => &[("Carrier", 20.0, 5000.0), ("Mix", 0.0, 1.0)],
        TrackEffectConfig::NoiseGate { .. } => &[
            ("Threshold", -80.0, 0.0),
            ("Attack", 0.1, 200.0),
            ("Release", 5.0, 2000.0),
            ("Range", -80.0, 0.0),
        ],
        TrackEffectConfig::PhaseInvert { .. } => &[],
        TrackEffectConfig::ChannelEq { .. } => &[],
        TrackEffectConfig::Limiter { .. } => {
            &[("Input gain", -24.0, 24.0), ("Ceiling", -12.0, 0.0), ("Release", 5.0, 1000.0)]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn every_default_config() -> Vec<TrackEffectConfig> {
        vec![
            TrackEffectConfig::default_delay(),
            TrackEffectConfig::default_bitcrusher(),
            TrackEffectConfig::default_distortion(),
            TrackEffectConfig::default_reverb(),
            TrackEffectConfig::default_chorus(),
            TrackEffectConfig::default_filter(),
            TrackEffectConfig::default_tremolo(),
            TrackEffectConfig::default_compressor(),
            TrackEffectConfig::default_flanger(),
            TrackEffectConfig::default_phaser(),
            TrackEffectConfig::default_ring_modulator(),
            TrackEffectConfig::default_noise_gate(),
            TrackEffectConfig::default_phase_invert(),
            TrackEffectConfig::default_channel_eq(),
            TrackEffectConfig::default_limiter(),
        ]
    }

    /// Every name `automatable_params_for_config` advertises for a given effect kind must actually
    /// be recognized by `set_automatable_param` on that same kind — a name only one of the two
    /// knows about (a typo in either list) would otherwise silently do nothing when automated,
    /// with no compiler error to catch it.
    #[test]
    fn every_automatable_param_name_is_recognized_by_set_automatable_param() {
        for config in every_default_config() {
            let mut effect = BuiltInEffect::from_config(&config, 48_000.0).unwrap();
            for &(name, min, max) in automatable_params_for_config(&config) {
                let before = format!("{:?}", effect.to_config());
                // Try both ends of the range: a param whose *default* happens to already sit at
                // one end (e.g. Bitcrusher's Mix defaults to fully wet, the same as its max) would
                // otherwise look like a no-op even though `set_automatable_param` handled it fine.
                effect.set_automatable_param(name, min);
                let after_min = format!("{:?}", effect.to_config());
                effect.set_automatable_param(name, max);
                let after_max = format!("{:?}", effect.to_config());
                assert!(
                    after_min != before || after_max != before,
                    "{}: setting {name:?} to its min ({min}) or max ({max}) had no effect on \
                     to_config() (before: {before})",
                    effect.label(),
                );
            }
        }
    }

    /// `set_automatable_param` clamps to the range `automatable_params_for_config` declares, the same way
    /// `plugin_host::LoadedEffect::set_param` clamps a CLAP parameter to its declared range.
    #[test]
    fn set_automatable_param_clamps_out_of_range_values() {
        let mut effect =
            BuiltInEffect::from_config(&TrackEffectConfig::default_delay(), 48_000.0).unwrap();
        effect.set_automatable_param("Feedback", 999.0);
        match effect.to_config() {
            TrackEffectConfig::Delay { feedback, .. } => assert_eq!(feedback, 0.95),
            other => panic!("expected Delay, got {other:?}"),
        }
        effect.set_automatable_param("Feedback", -999.0);
        match effect.to_config() {
            TrackEffectConfig::Delay { feedback, .. } => assert_eq!(feedback, 0.0),
            other => panic!("expected Delay, got {other:?}"),
        }
    }

    #[test]
    fn set_automatable_param_ignores_an_unknown_name() {
        let mut effect =
            BuiltInEffect::from_config(&TrackEffectConfig::default_delay(), 48_000.0).unwrap();
        let before = format!("{:?}", effect.to_config());
        effect.set_automatable_param("not_a_real_param", 1.0);
        let after = format!("{:?}", effect.to_config());
        assert_eq!(before, after);
    }
}
