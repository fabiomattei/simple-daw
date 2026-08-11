use std::fmt;
use std::path::Path;

use anyhow::{Context, Result};

/// A one-shot audio sample, downmixed to mono and pre-resampled to the
/// audio engine's output sample rate so playback is just a straight read.
pub struct SampleBuffer {
    pub sample_rate: u32,
    pub mono: Vec<f32>,
}

impl fmt::Debug for SampleBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SampleBuffer")
            .field("sample_rate", &self.sample_rate)
            .field("frames", &self.mono.len())
            .finish()
    }
}

impl SampleBuffer {
    /// Decode a WAV file at `path` (float or integer PCM, any channel count) into a
    /// mono buffer at the file's native sample rate. Returns an error if the file
    /// can't be opened or its samples can't be read.
    pub fn load_wav(path: &Path) -> Result<Self> {
        let mut reader = hound::WavReader::open(path)
            .with_context(|| format!("failed to open wav file: {}", path.display()))?;
        let spec = reader.spec();

        let interleaved: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => reader
                .samples::<f32>()
                .collect::<Result<_, _>>()
                .context("failed to read float wav samples")?,
            hound::SampleFormat::Int => {
                let max_amplitude = (1i64 << (spec.bits_per_sample - 1)) as f32;
                reader
                    .samples::<i32>()
                    .map(|s| s.map(|v| v as f32 / max_amplitude))
                    .collect::<Result<_, _>>()
                    .context("failed to read integer wav samples")?
            }
        };

        let channels = (spec.channels as usize).max(1);
        let mono: Vec<f32> = interleaved
            .chunks(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect();

        Ok(Self {
            sample_rate: spec.sample_rate,
            mono,
        })
    }

    /// Load a WAV file and resample it (linear interpolation) to `target_sample_rate`
    /// so the real-time playback path never needs to resample per-voice.
    pub fn load_wav_resampled(path: &Path, target_sample_rate: u32) -> Result<Self> {
        let loaded = Self::load_wav(path)?;
        if loaded.sample_rate == target_sample_rate || loaded.mono.is_empty() {
            return Ok(loaded);
        }
        Ok(Self {
            sample_rate: target_sample_rate,
            mono: resample_linear(&loaded.mono, loaded.sample_rate, target_sample_rate),
        })
    }
}

fn resample_linear(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    let ratio = from_rate as f64 / to_rate as f64;
    let out_len = ((input.len() as f64) / ratio).round().max(1.0) as usize;
    (0..out_len)
        .map(|i| {
            let src_pos = i as f64 * ratio;
            let idx = src_pos.floor() as usize;
            let frac = (src_pos - idx as f64) as f32;
            let a = input.get(idx).copied().unwrap_or(0.0);
            let b = input.get(idx + 1).copied().unwrap_or(a);
            a + (b - a) * frac
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_linear_preserves_endpoints_and_scales_length() {
        let input: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let out = resample_linear(&input, 44_100, 48_000);
        assert!((out.len() as f64 - 100.0 * 48_000.0 / 44_100.0).abs() < 2.0);
        assert!((out[0] - input[0]).abs() < 0.001);
    }

    #[test]
    fn resample_linear_is_noop_shape_when_rates_match() {
        let input = vec![0.1, 0.2, -0.3, 0.4];
        let out = resample_linear(&input, 44_100, 44_100);
        assert_eq!(out.len(), input.len());
        for (a, b) in input.iter().zip(out.iter()) {
            assert!((a - b).abs() < 0.0001);
        }
    }
}
