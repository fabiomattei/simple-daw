use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SampleFormat, SizedSample, Stream, StreamConfig};

/// Lists available audio input device names, in host-enumeration order, for a UI picker (see
/// `InputRecorder::start`'s `device_name`). Best-effort: a device whose name can't be queried is
/// skipped, since a name-only listing has no other way to represent it.
pub fn list_input_devices() -> Vec<String> {
    let Ok(devices) = cpal::default_host().input_devices() else {
        return Vec::new();
    };
    devices
        .filter_map(|d| Some(d.description().ok()?.name().to_string()))
        .collect()
}

/// A live microphone/line-in capture session. Symmetric with `audio::AudioEngine::start`, but for
/// input instead of output — kept in its own module rather than folded into `audio.rs` because
/// device capture is a distinct infra concern from the playback/mixing engine (the same reason
/// `plugin_host.rs` is split out from it), and captured samples are appended to a plain
/// `Arc<Mutex<Vec<f32>>>` from the input callback rather than the allocation-free scratch buffers
/// `audio.rs`'s real-time *playback* path is careful to use — recording is not that hot path, and
/// this mirrors the `Arc<Mutex<Song>>` sharing already used everywhere else in this app.
pub struct InputRecorder {
    _stream: Stream,
    buffer: Arc<Mutex<Vec<f32>>>,
    peak_bits: Arc<AtomicU32>,
    /// The input device's actual sample rate. Captured audio is handed back at this rate (see
    /// `stop`) and resampled to the playback engine's rate the same way any loaded WAV file is
    /// (`SampleBuffer::load_wav_resampled`), rather than resampling twice.
    pub sample_rate: u32,
}

impl InputRecorder {
    /// Starts capturing from the input device named `device_name`, falling back to the host's
    /// default input device if `None` or if no device with that name is found. `input_gain` is
    /// the armed track's preamp trim (see `Track::input_gain`) — applied once, in the input
    /// callback, before samples reach the capture buffer or the level meter, so the meter reflects
    /// what actually gets recorded. Fixed for the session's lifetime, same as `device_name`: this
    /// app has no live input monitoring, so there's no path by which changing the trim mid-take
    /// would need to take effect.
    pub fn start(device_name: Option<&str>, input_gain: f32) -> Result<Self> {
        let host = cpal::default_host();
        let named_device = device_name.and_then(|name| {
            host.input_devices().ok()?.find(|d| {
                d.description()
                    .map(|desc| desc.name() == name)
                    .unwrap_or(false)
            })
        });
        let device = named_device
            .or_else(|| host.default_input_device())
            .context("no audio input device available")?;

        let supported_config = device
            .default_input_config()
            .context("no default input config available")?;
        let sample_format = supported_config.sample_format();
        let config: StreamConfig = supported_config.into();
        let sample_rate = config.sample_rate;

        let buffer = Arc::new(Mutex::new(Vec::new()));
        let peak_bits = Arc::new(AtomicU32::new(0));

        let stream = match sample_format {
            SampleFormat::F32 => build_input_stream::<f32>(
                &device,
                &config,
                buffer.clone(),
                peak_bits.clone(),
                input_gain,
            )?,
            SampleFormat::I16 => build_input_stream::<i16>(
                &device,
                &config,
                buffer.clone(),
                peak_bits.clone(),
                input_gain,
            )?,
            SampleFormat::U16 => build_input_stream::<u16>(
                &device,
                &config,
                buffer.clone(),
                peak_bits.clone(),
                input_gain,
            )?,
            other => bail!("unsupported input sample format: {other:?}"),
        };
        stream.play().context("failed to start input stream")?;

        Ok(Self {
            _stream: stream,
            buffer,
            peak_bits,
            sample_rate,
        })
    }

    /// The most recently captured block's peak amplitude (0.0..~1.0), for a live input level
    /// meter. Reads a plain atomic, safe to poll from the UI thread every frame.
    pub fn level(&self) -> f32 {
        f32::from_bits(self.peak_bits.load(Ordering::Relaxed))
    }

    /// Stops capturing and returns every sample captured so far, mono, at `sample_rate`.
    pub fn stop(self) -> Vec<f32> {
        drop(self._stream);
        // The stream (and the buffer clone its callback held) is gone now, so this Arc has
        // exactly one owner left — `try_unwrap` succeeds and avoids a pointless clone of
        // (potentially minutes of) audio; the fallback only matters if that assumption is wrong.
        Arc::try_unwrap(self.buffer)
            .map(|mutex| mutex.into_inner().unwrap_or_default())
            .unwrap_or_else(|arc| arc.lock().map(|guard| guard.clone()).unwrap_or_default())
    }
}

fn build_input_stream<T>(
    device: &cpal::Device,
    config: &StreamConfig,
    buffer: Arc<Mutex<Vec<f32>>>,
    peak_bits: Arc<AtomicU32>,
    input_gain: f32,
) -> Result<Stream>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let channels = config.channels.max(1) as usize;
    let err_fn = |err| eprintln!("audio input stream error: {err}");
    let stream = device.build_input_stream(
        *config,
        move |data: &[T], _: &cpal::InputCallbackInfo| {
            let mut block_peak = 0.0f32;
            let mono: Vec<f32> = data
                .chunks(channels)
                .map(|frame| {
                    let sum: f32 = frame.iter().map(|s| f32::from_sample(*s)).sum();
                    let value = (sum / channels as f32) * input_gain;
                    block_peak = block_peak.max(value.abs());
                    value
                })
                .collect();
            if let Ok(mut buf) = buffer.lock() {
                buf.extend_from_slice(&mono);
            }
            peak_bits.store(block_peak.to_bits(), Ordering::Relaxed);
        },
        err_fn,
        None,
    )?;
    Ok(stream)
}
