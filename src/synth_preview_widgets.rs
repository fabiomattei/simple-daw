//! Reusable synth-editor preview/picker widgets shared by all three built-in synth engines'
//! Device Panel UIs (`synth_simple_panel`, `synth_trine_panel`, `synth_wave_panel`) — generic
//! over raw parameter values (f32s, `FilterType`/`FilterSlope`/`FilterRouting`, `SynthWaveform`)
//! rather than any one engine's `SynthParams`/`TrineParams`/`WaveParams`, so a widget added here
//! for one engine is free for the others to reuse instead of re-implementing.

use crate::model::{FilterRouting, FilterSlope, FilterType, SynthWaveform};

/// Sample of a raw oscillator cycle in `[-1, 1]` for `phase` running `0..1`, for the small
/// waveform-preview canvases below. Mirrors `audio::waveform_sample` (kept private to the
/// real-time engine) since this is purely for drawing, not audio.
pub(crate) fn waveform_shape_sample(waveform: SynthWaveform, phase: f32, pulse_width: f32) -> f32 {
    match waveform {
        SynthWaveform::Sine => (phase * std::f32::consts::TAU).sin(),
        SynthWaveform::Saw => 2.0 * phase - 1.0,
        SynthWaveform::Square => {
            if phase < pulse_width {
                1.0
            } else {
                -1.0
            }
        }
        SynthWaveform::Triangle => 4.0 * (phase - 0.5).abs() - 1.0,
        // Mirrors `audio::hash_to_bipolar` — good enough for a jagged-looking preview.
        SynthWaveform::Noise => {
            let mut h = phase.to_bits();
            h ^= h >> 16;
            h = h.wrapping_mul(0x7feb_352d);
            h ^= h >> 15;
            h = h.wrapping_mul(0x846c_a68b);
            h ^= h >> 16;
            (h as f32 / u32::MAX as f32) * 2.0 - 1.0
        }
    }
}


/// Draws `cycles` repetitions of `waveform`'s shape across `rect`, scaled by `amplitude` (0..1)
/// and vertically centered. Used by the oscillator and LFO preview canvases.
pub(crate) fn paint_waveform_shape(
    painter: &egui::Painter,
    rect: egui::Rect,
    waveform: SynthWaveform,
    pulse_width: f32,
    amplitude: f32,
    cycles: f32,
    stroke: egui::Stroke,
) {
    let samples = 200;
    let mid_y = rect.center().y;
    let half_h = rect.height() * 0.5 * 0.9;
    let points: Vec<egui::Pos2> = (0..=samples)
        .map(|i| {
            let t = i as f32 / samples as f32;
            let phase = (t * cycles).fract();
            let sample = waveform_shape_sample(waveform, phase, pulse_width) * amplitude;
            egui::pos2(rect.left() + t * rect.width(), mid_y - sample * half_h)
        })
        .collect();
    painter.add(egui::Shape::line(points, stroke));
}


/// Samples one oscillator's shape across `cycles` of a reference oscillator's period, optionally
/// hard-synced to it — the same math `oscillator2_preview_ui` uses for Oscillator 2, generalized
/// so `TrineParams`'s three oscillators can share it (see `trine_oscillators_preview_ui`).
/// `ratio` is this oscillator's frequency relative to the reference (from semitone/cent tuning);
/// when `sync` is true the phase re-zeroes every reference cycle (`(fract(t) * ratio).fract()`),
/// mirroring `audio::Voice`'s hard-sync; when false it free-runs (`(t * ratio).fract()`).
pub(crate) fn synced_oscillator_points(
    rect: egui::Rect,
    waveform: SynthWaveform,
    pulse_width: f32,
    ratio: f32,
    sync: bool,
    amplitude: f32,
    cycles: f32,
    samples: usize,
) -> Vec<egui::Pos2> {
    let mid_y = rect.center().y;
    let half_h = rect.height() * 0.5 * 0.9;
    (0..=samples)
        .map(|i| {
            let t = i as f32 / samples as f32 * cycles;
            let phase = if sync {
                (t.fract() * ratio).fract()
            } else {
                (t * ratio).fract()
            };
            let sample = waveform_shape_sample(waveform, phase, pulse_width) * amplitude;
            egui::pos2(
                rect.left() + i as f32 / samples as f32 * rect.width(),
                mid_y - sample * half_h,
            )
        })
        .collect()
}


/// Draws the ADSR envelope shape. Since Sustain has no fixed duration (it holds until note-off),
/// its plateau is drawn as a fixed-width visual segment rather than to scale; Attack, Decay and
/// Release segments are sized proportionally to their actual values relative to one another.
/// Generic over the raw ADSR values so `TrineParams`/`WaveParams`'s multiple envelopes (which
/// aren't wrapped in a `SynthParams`) can reuse it — see `envelope_preview_ui` for the
/// `SynthParams` convenience wrapper.
pub(crate) fn adsr_preview_ui(
    ui: &mut egui::Ui,
    attack: f32,
    decay: f32,
    sustain: f32,
    release: f32,
    color: egui::Color32,
) {
    let (response, painter) =
        ui.allocate_painter(egui::vec2(ui.available_width(), 60.0), egui::Sense::hover());
    let rect = response.rect;
    painter.rect_filled(rect, 2.0, ui.visuals().extreme_bg_color);

    let attack = attack.max(0.01);
    let decay = decay.max(0.01);
    let release = release.max(0.01);
    let sustain_hold = 0.4; // fixed visual width standing in for the undefined hold duration
    let total = attack + decay + sustain_hold + release;

    let pad = 4.0;
    let x0 = rect.left() + pad;
    let usable_w = rect.width() - 2.0 * pad;
    let x_attack = x0 + usable_w * (attack / total);
    let x_decay = x_attack + usable_w * (decay / total);
    let x_hold = x_decay + usable_w * (sustain_hold / total);
    let x_release = x_hold + usable_w * (release / total);
    let y_bottom = rect.bottom() - pad;
    let y_top = rect.top() + pad;
    let y_sustain = y_bottom - (y_bottom - y_top) * sustain.clamp(0.0, 1.0);

    let points = vec![
        egui::pos2(x0, y_bottom),
        egui::pos2(x_attack, y_top),
        egui::pos2(x_decay, y_sustain),
        egui::pos2(x_hold, y_sustain),
        egui::pos2(x_release, y_bottom),
    ];
    painter.add(egui::Shape::line(points, egui::Stroke::new(2.0, color)));
}


/// Frequency-response magnitude curve for the per-voice TPT state-variable filter, using the
/// standard analog 2-pole lowpass/highpass/bandpass/notch prototype that filter approximates
/// (see `audio::Voice::process`'s SVF for the actual real-time DSP). Purely illustrative — close
/// enough to communicate cutoff/resonance shape without re-simulating the exact discrete filter.
pub(crate) fn filter_response_db(
    filter_type: FilterType,
    freq_hz: f32,
    cutoff_hz: f32,
    resonance: f32,
) -> f32 {
    let x = freq_hz / cutoff_hz.max(1.0);
    let q = resonance.max(0.05);
    let denom = ((1.0 - x * x).powi(2) + (x / q).powi(2)).sqrt().max(1e-6);
    let magnitude = match filter_type {
        FilterType::Lowpass => 1.0 / denom,
        FilterType::Highpass => (x * x) / denom,
        FilterType::Bandpass => (x / q) / denom,
        FilterType::Notch => (1.0 - x * x).abs() / denom,
    };
    20.0 * magnitude.max(1e-6).log10()
}


/// Draws the combined frequency response of `TrineParams`/`WaveParams`'s two filters, respecting
/// `FilterRouting`: `Off` shows filter1 alone, `Series` sums the two responses in dB (equivalent
/// to multiplying their linear magnitudes), `Parallel` sums their linear magnitudes before
/// converting back to dB. A second marker line (in filter2's color) appears next to filter1's
/// whenever filter2 is actually in the signal path.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dual_filter_preview_ui(
    ui: &mut egui::Ui,
    filter1_type: FilterType,
    cutoff1_hz: f32,
    resonance1: f32,
    filter2_type: FilterType,
    cutoff2_hz: f32,
    resonance2: f32,
    routing: FilterRouting,
) {
    let (response, painter) =
        ui.allocate_painter(egui::vec2(ui.available_width(), 60.0), egui::Sense::hover());
    let rect = response.rect;
    painter.rect_filled(rect, 2.0, ui.visuals().extreme_bg_color);

    let min_db = -36.0;
    let max_db = 18.0;
    let log_min = 20.0f32.log10();
    let log_max = 20_000.0f32.log10();
    let db_to_y = |db: f32| {
        let t = ((db - min_db) / (max_db - min_db)).clamp(0.0, 1.0);
        rect.bottom() - t * rect.height()
    };
    painter.line_segment(
        [
            egui::pos2(rect.left(), db_to_y(0.0)),
            egui::pos2(rect.right(), db_to_y(0.0)),
        ],
        egui::Stroke::new(0.5, ui.visuals().weak_text_color()),
    );

    let samples = 150;
    let points: Vec<egui::Pos2> = (0..=samples)
        .map(|i| {
            let t = i as f32 / samples as f32;
            let freq = 10f32.powf(log_min + t * (log_max - log_min));
            let db1 = filter_response_db(filter1_type, freq, cutoff1_hz, resonance1);
            let db = match routing {
                FilterRouting::Off => db1,
                FilterRouting::Series => {
                    db1 + filter_response_db(filter2_type, freq, cutoff2_hz, resonance2)
                }
                FilterRouting::Parallel => {
                    let db2 = filter_response_db(filter2_type, freq, cutoff2_hz, resonance2);
                    let linear_sum = 10f32.powf(db1 / 20.0) + 10f32.powf(db2 / 20.0);
                    20.0 * linear_sum.max(1e-6).log10()
                }
            };
            egui::pos2(
                rect.left() + t * rect.width(),
                db_to_y(db.clamp(min_db, max_db)),
            )
        })
        .collect();
    painter.add(egui::Shape::line(
        points,
        egui::Stroke::new(2.0, egui::Color32::from_rgb(230, 140, 140)),
    ));

    let cutoff_x = |cutoff_hz: f32| {
        let t = ((cutoff_hz.max(20.0).log10() - log_min) / (log_max - log_min)).clamp(0.0, 1.0);
        rect.left() + t * rect.width()
    };
    let x1 = cutoff_x(cutoff1_hz);
    painter.line_segment(
        [egui::pos2(x1, rect.top()), egui::pos2(x1, rect.bottom())],
        egui::Stroke::new(1.0, egui::Color32::YELLOW),
    );
    if routing != FilterRouting::Off {
        let x2 = cutoff_x(cutoff2_hz);
        painter.line_segment(
            [egui::pos2(x2, rect.top()), egui::pos2(x2, rect.bottom())],
            egui::Stroke::new(1.0, egui::Color32::from_rgb(120, 200, 255)),
        );
    }
}


/// Draws a few cycles of an LFO's waveform, scaled by `depth`; grayed out when `active` is false.
/// Generic over the raw values — see `lfo_preview_ui` for the `SynthParams` wrapper.
pub(crate) fn lfo_shape_preview_ui(ui: &mut egui::Ui, waveform: SynthWaveform, active: bool, depth: f32) {
    let (response, painter) =
        ui.allocate_painter(egui::vec2(ui.available_width(), 60.0), egui::Sense::hover());
    let rect = response.rect;
    painter.rect_filled(rect, 2.0, ui.visuals().extreme_bg_color);
    painter.line_segment(
        [
            egui::pos2(rect.left(), rect.center().y),
            egui::pos2(rect.right(), rect.center().y),
        ],
        egui::Stroke::new(0.5, ui.visuals().weak_text_color()),
    );

    let color = if active {
        egui::Color32::from_rgb(200, 140, 230)
    } else {
        ui.visuals().weak_text_color()
    };
    let amplitude = if active { depth.max(0.05) } else { 0.6 };
    paint_waveform_shape(
        &painter,
        rect,
        waveform,
        0.5,
        amplitude,
        4.0,
        egui::Stroke::new(2.0, color),
    );
}


/// A row of selectable labels for every `SynthWaveform` variant, mirroring the picker rows in
/// `synth_oscillators_ui` — shared here since Trine has five of these (three oscillators, two LFOs).
pub(crate) fn waveform_picker_ui(ui: &mut egui::Ui, current: &mut SynthWaveform) {
    ui.horizontal(|ui| {
        ui.label("Waveform:");
        for (label, waveform) in [
            ("Sine", SynthWaveform::Sine),
            ("Saw", SynthWaveform::Saw),
            ("Square", SynthWaveform::Square),
            ("Triangle", SynthWaveform::Triangle),
            ("Noise", SynthWaveform::Noise),
        ] {
            if ui.selectable_label(*current == waveform, label).clicked() {
                *current = waveform;
            }
        }
    });
}


pub(crate) fn filter_stage_ui(
    ui: &mut egui::Ui,
    cutoff_hz: &mut f32,
    resonance: &mut f32,
    filter_type: &mut FilterType,
    slope: &mut FilterSlope,
) {
    ui.horizontal(|ui| {
        ui.label("Type:");
        for (label, ft) in [
            ("Lowpass", FilterType::Lowpass),
            ("Highpass", FilterType::Highpass),
            ("Bandpass", FilterType::Bandpass),
            ("Notch", FilterType::Notch),
        ] {
            if ui.selectable_label(*filter_type == ft, label).clicked() {
                *filter_type = ft;
            }
        }
    });
    ui.horizontal(|ui| {
        ui.label("Slope:");
        for (label, s) in [
            ("12 dB/oct", FilterSlope::Slope12),
            ("24 dB/oct", FilterSlope::Slope24),
        ] {
            if ui.selectable_label(*slope == s, label).clicked() {
                *slope = s;
            }
        }
    });
    ui.add(
        egui::Slider::new(cutoff_hz, 20.0..=20_000.0)
            .logarithmic(true)
            .text("Cutoff")
            .suffix(" Hz"),
    );
    ui.add(egui::Slider::new(resonance, 0.3..=10.0).text("Resonance"));
}

