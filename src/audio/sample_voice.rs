//! One-shot/looping WAV sample playback (`SampleVoice`) and `TrackVoices`, the per-track bundle
//! of all four always-allocated voice pools (`simple_voice::Voice`, `trine_voice::TrineVoice`,
//! `wave_voice::WaveVoice`, `SampleVoice`) plus each pool's own round-robin `next_*` index — an
//! idle pool just contributes silence, so a track only ever "pays" for the engine it's actually
//! using (see `model::SynthEngine`).

use std::sync::Arc;

use crate::sample::SampleBuffer;

use super::simple_voice::Voice;
use super::trine_voice::TrineVoice;
use super::wave_voice::WaveVoice;
use super::{SAMPLE_VOICE_COUNT, VOICE_COUNT};

/// Plays back a pre-resampled one-shot sample from `start_position` to `end_position` (exclusive),
/// with optional linear fade-in/fade-out ramps at the edges — the frame-domain counterpart of
/// `Region::fade_gain_at`, but evaluated per sample rather than per tick since a clip's playback
/// position isn't tick-quantized.
#[derive(Clone, Default)]
pub(crate) struct SampleVoice {
    pub(crate) buffer: Option<Arc<SampleBuffer>>,
    position: usize,
    start_position: usize,
    end_position: usize,
    gain: f32,
    fade_in_frames: usize,
    fade_out_frames: usize,
    /// When set, `next_sample` wraps `position` back to `start_position` instead of going silent
    /// past `end_position` — used only for Session View audio clips (`Sequencer::process`'s
    /// `trigger_session_clips`), which loop indefinitely until stopped rather than playing once.
    looping: bool,
}

impl SampleVoice {
    /// Plays `buffer` in full, from its own start to its own end, with no fades — used for
    /// velocity-triggered one-shot samples (drum-lane steps), not `AudioClip` playback (see
    /// `trigger_clip`).
    pub(crate) fn trigger(&mut self, buffer: Arc<SampleBuffer>, velocity: u8) {
        let gain = (velocity as f32 / 127.0).clamp(0.0, 1.0);
        self.trigger_clip(buffer, gain, 0, usize::MAX, 0, 0, false);
    }

    /// Same as `trigger`, but for `AudioClip` playback (see `model::AudioClip`): a continuous gain
    /// instead of a 0..127 velocity byte, plus a trim window (`start_frame..end_frame`, clamped to
    /// the buffer's own length) and fade-in/out ramp lengths in frames — both converted from the
    /// clip's tick-domain fields by the caller (`Sequencer::process`), since only that call site
    /// knows the tempo in effect at the clip's start tick. `looping` — see `SampleVoice::looping`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn trigger_clip(
        &mut self,
        buffer: Arc<SampleBuffer>,
        gain: f32,
        start_frame: usize,
        end_frame: usize,
        fade_in_frames: usize,
        fade_out_frames: usize,
        looping: bool,
    ) {
        let len = buffer.mono.len();
        self.start_position = start_frame.min(len);
        self.position = self.start_position;
        self.end_position = end_frame.min(len);
        self.buffer = Some(buffer);
        self.gain = gain;
        self.fade_in_frames = fade_in_frames;
        self.fade_out_frames = fade_out_frames;
        self.looping = looping;
    }

    pub(crate) fn next_sample(&mut self) -> f32 {
        if self.position >= self.end_position {
            if self.looping && self.end_position > self.start_position {
                self.position = self.start_position;
            } else {
                self.buffer = None;
                return 0.0;
            }
        }
        let Some(buffer) = &self.buffer else {
            return 0.0;
        };
        let Some(&s) = buffer.mono.get(self.position) else {
            self.buffer = None;
            return 0.0;
        };
        let elapsed = self.position - self.start_position;
        let remaining = self.end_position - self.position;
        let mut fade = 1.0f32;
        if self.fade_in_frames > 0 {
            fade = fade.min((elapsed as f32 / self.fade_in_frames as f32).clamp(0.0, 1.0));
        }
        if self.fade_out_frames > 0 {
            fade = fade.min((remaining as f32 / self.fade_out_frames as f32).clamp(0.0, 1.0));
        }
        self.position += 1;
        s * self.gain * fade
    }
}

/// One track's independent voice pools, so a busy drum track can never starve a melodic track's
/// polyphony (and so each track's dry signal can be kept separate for per-track CLAP effects).
pub(crate) struct TrackVoices {
    pub(crate) voices: [Voice; VOICE_COUNT],
    pub(crate) next_voice: usize,
    /// Independent voice pool for the Trine engine (see `SynthEngine::Trine`) — always allocated per
    /// track (a small, constant memory cost) but only ever triggered when that track's
    /// `synth_engine` is `Trine`; an idle pool just contributes silence to the mix.
    pub(crate) trine_voices: [TrineVoice; VOICE_COUNT],
    pub(crate) next_trine_voice: usize,
    /// Independent voice pool for the Wave engine (see `SynthEngine::Wave`) — same always-
    /// allocated-but-idle-when-unused arrangement as `trine_voices`.
    pub(crate) wave_voices: [WaveVoice; VOICE_COUNT],
    pub(crate) next_wave_voice: usize,
    pub(crate) sample_voices: [SampleVoice; SAMPLE_VOICE_COUNT],
    pub(crate) next_sample_voice: usize,
    /// Most recently triggered piano-roll pitch on this track, for `SynthParams::glide_seconds`
    /// to portamento from. Monophonic "last note" memory layered on top of the polyphonic voice
    /// pool — standard glide behavior even in an otherwise-polyphonic engine. Step-grid hits never
    /// read or write this (see `Sequencer::process`).
    pub(crate) last_freq: Option<f32>,
}

impl TrackVoices {
    pub(crate) fn new() -> Self {
        Self {
            voices: [Voice::default(); VOICE_COUNT],
            next_voice: 0,
            trine_voices: [TrineVoice::default(); VOICE_COUNT],
            next_trine_voice: 0,
            wave_voices: [WaveVoice::default(); VOICE_COUNT],
            next_wave_voice: 0,
            sample_voices: std::array::from_fn(|_| SampleVoice::default()),
            next_sample_voice: 0,
            last_freq: None,
        }
    }
}
