//! The `Simple` synth engine's Device Panel UI: waveform picker, oscillator/unison controls,
//! envelope, filter, and modulation sections for `Track::synth` — see `synth_params_ui`, called
//! from `device_panel_contents_ui` when `Track::synth_engine == SynthEngine::Simple`.

use crate::model::{FilterType, LfoTarget, SynthParams, SynthWaveform};
use crate::synth_preview_widgets::{
    adsr_preview_ui, filter_response_db, lfo_shape_preview_ui, paint_waveform_shape, waveform_shape_sample,
};

/// Renders the waveform picker and attack/decay sliders for a track's built-in synth voice,
/// shown inside the bottom Device Panel (see `device_panel_contents_ui`). Laid out as two
/// columns (oscillators | envelope/filter/LFO) to keep the panel from growing too tall.
pub(crate) fn synth_params_ui(ui: &mut egui::Ui, synth: &mut SynthParams) {
    ui.columns(2, |columns| {
        synth_oscillators_ui(&mut columns[0], synth);
        synth_modulation_ui(&mut columns[1], synth);
    });
}


/// Small canvas previewing the combined oscillator output: Oscillator 1 in the accent color,
/// Oscillator 2 faded in proportion to its mix, and the sub-oscillator as a thin low-amplitude
/// line when its level is above zero.
fn oscillator_preview_ui(ui: &mut egui::Ui, synth: &SynthParams) {
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

    if synth.sub_osc_mix > 0.0 {
        paint_waveform_shape(
            &painter,
            rect,
            SynthWaveform::Sine,
            0.5,
            synth.sub_osc_mix,
            1.0,
            egui::Stroke::new(1.0, ui.visuals().weak_text_color()),
        );
    }
    if synth.osc2_mix > 0.0 {
        let osc2_color =
            egui::Color32::from_rgb(230, 160, 90).gamma_multiply(synth.osc2_mix.max(0.25));
        paint_waveform_shape(
            &painter,
            rect,
            synth.osc2_waveform,
            0.5,
            1.0,
            2.0,
            egui::Stroke::new(1.5, osc2_color),
        );
    }
    let osc1_amplitude = if synth.osc2_mix > 0.0 {
        1.0 - synth.osc2_mix
    } else {
        1.0
    };
    paint_waveform_shape(
        &painter,
        rect,
        synth.waveform,
        synth.pulse_width,
        osc1_amplitude.max(0.15),
        2.0,
        egui::Stroke::new(2.0, egui::Color32::from_rgb(120, 170, 230)),
    );
}


/// Small canvas dedicated to Oscillator 2: a faint reference line for Oscillator 1 (one cycle
/// per `cycles`) and Oscillator 2's own shape overlaid in orange. Both phases are computed
/// analytically from the elapsed fraction of Oscillator 1's cycle, so this mirrors
/// `audio::Voice::next_sample`'s hard-sync math exactly without needing to simulate sample by
/// sample: free-running osc2 phase is `(t * ratio).fract()`, and — since sync resets osc2 to 0 at
/// every osc1 wrap — synced osc2 phase is `(fract(t) * ratio).fract()`, i.e. re-zeroed each cycle.
fn oscillator2_preview_ui(ui: &mut egui::Ui, synth: &SynthParams) {
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

    let ratio =
        2f32.powf(synth.osc2_semitones as f32 / 12.0) * 2f32.powf(synth.osc2_detune_cents / 1200.0);
    let cycles = 3.0;
    let samples = 300;
    let mid_y = rect.center().y;
    let half_h = rect.height() * 0.5 * 0.9;

    let osc1_points: Vec<egui::Pos2> = (0..=samples)
        .map(|i| {
            let t = i as f32 / samples as f32 * cycles;
            let sample = waveform_shape_sample(synth.waveform, t.fract(), synth.pulse_width) * 0.6;
            egui::pos2(
                rect.left() + i as f32 / samples as f32 * rect.width(),
                mid_y - sample * half_h,
            )
        })
        .collect();
    painter.add(egui::Shape::line(
        osc1_points,
        egui::Stroke::new(1.0, ui.visuals().weak_text_color()),
    ));

    let osc2_points: Vec<egui::Pos2> = (0..=samples)
        .map(|i| {
            let t = i as f32 / samples as f32 * cycles;
            let phase = if synth.osc2_sync {
                (t.fract() * ratio).fract()
            } else {
                (t * ratio).fract()
            };
            let sample = waveform_shape_sample(synth.osc2_waveform, phase, 0.5);
            egui::pos2(
                rect.left() + i as f32 / samples as f32 * rect.width(),
                mid_y - sample * half_h,
            )
        })
        .collect();
    painter.add(egui::Shape::line(
        osc2_points,
        egui::Stroke::new(2.0, egui::Color32::from_rgb(230, 160, 90)),
    ));
}


fn synth_oscillators_ui(ui: &mut egui::Ui, synth: &mut SynthParams) {
    ui.strong("Oscillator");
    ui.horizontal(|ui| {
        ui.label("Waveform:");
        for (label, waveform) in [
            ("Sine", SynthWaveform::Sine),
            ("Saw", SynthWaveform::Saw),
            ("Square", SynthWaveform::Square),
            ("Triangle", SynthWaveform::Triangle),
        ] {
            if ui
                .selectable_label(synth.waveform == waveform, label)
                .clicked()
            {
                synth.waveform = waveform;
            }
        }
    });
    ui.add_enabled(
        synth.waveform == SynthWaveform::Square,
        egui::Slider::new(&mut synth.pulse_width, 0.05..=0.95).text("Pulse width"),
    )
    .on_hover_text("Duty cycle of the Square wave; only applies to that waveform.");
    ui.horizontal(|ui| {
        ui.label("Unison:");
        for voices in 1..=3u8 {
            if ui
                .selectable_label(synth.unison_voices == voices, voices.to_string())
                .clicked()
            {
                synth.unison_voices = voices;
            }
        }
    });
    ui.add_enabled(
        synth.unison_voices > 1,
        egui::Slider::new(&mut synth.unison_detune_cents, 0.0..=50.0)
            .text("Detune")
            .suffix(" cents"),
    );
    ui.add_enabled(
        synth.unison_voices > 1,
        egui::Slider::new(&mut synth.unison_width, 0.0..=1.0).text("Width"),
    )
    .on_hover_text("Spreads unison voices across the stereo field. 0 keeps them centered.");
    oscillator_preview_ui(ui, synth);

    ui.separator();
    ui.strong("Oscillator 2");
    ui.horizontal(|ui| {
        ui.label("Waveform:");
        for (label, waveform) in [
            ("Sine", SynthWaveform::Sine),
            ("Saw", SynthWaveform::Saw),
            ("Square", SynthWaveform::Square),
            ("Triangle", SynthWaveform::Triangle),
        ] {
            if ui
                .selectable_label(synth.osc2_waveform == waveform, label)
                .clicked()
            {
                synth.osc2_waveform = waveform;
            }
        }
    });
    ui.add(
        egui::Slider::new(&mut synth.osc2_semitones, -24..=24)
            .text("Coarse tune")
            .suffix(" st"),
    );
    ui.add(
        egui::Slider::new(&mut synth.osc2_detune_cents, -50.0..=50.0)
            .text("Fine tune")
            .suffix(" cents"),
    );
    ui.add(egui::Slider::new(&mut synth.osc2_mix, 0.0..=1.0).text("Mix"));
    ui.weak("Mix crossfades between Oscillator 1 (0) and Oscillator 2 (1); 0 sounds exactly like before this existed.");
    ui.checkbox(&mut synth.osc2_sync, "Sync to Oscillator 1")
        .on_hover_text("Resets Oscillator 2's phase every time Oscillator 1 completes a cycle, locking it to Oscillator 1's pitch and truncating its waveform for a bright, buzzy timbre.");
    oscillator2_preview_ui(ui, synth);
    ui.add(egui::Slider::new(&mut synth.sub_osc_mix, 0.0..=1.0).text("Sub-osc level"));
    ui.weak("A fixed sine one octave below the note's pitch, mixed in on top (not crossfaded).");
}


fn envelope_preview_ui(ui: &mut egui::Ui, synth: &SynthParams) {
    adsr_preview_ui(
        ui,
        synth.attack_seconds,
        synth.decay_seconds,
        synth.sustain_level,
        synth.release_seconds,
        egui::Color32::from_rgb(120, 220, 140),
    );
}


/// Draws one filter's frequency response across 20Hz-20kHz (log-scaled x-axis) with a marker at
/// the current cutoff. Generic over the raw filter values — see `filter_preview_ui` for the
/// `SynthParams` wrapper and `dual_filter_preview_ui` for `TrineParams`/`WaveParams`'s two-filter
/// version.
fn filter_response_preview_ui(
    ui: &mut egui::Ui,
    filter_type: FilterType,
    cutoff_hz: f32,
    resonance: f32,
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
    // 0 dB reference line
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
            let db =
                filter_response_db(filter_type, freq, cutoff_hz, resonance).clamp(min_db, max_db);
            egui::pos2(rect.left() + t * rect.width(), db_to_y(db))
        })
        .collect();
    painter.add(egui::Shape::line(
        points,
        egui::Stroke::new(2.0, egui::Color32::from_rgb(230, 140, 140)),
    ));

    let cutoff_t = ((cutoff_hz.max(20.0).log10() - log_min) / (log_max - log_min)).clamp(0.0, 1.0);
    let cutoff_x = rect.left() + cutoff_t * rect.width();
    painter.line_segment(
        [
            egui::pos2(cutoff_x, rect.top()),
            egui::pos2(cutoff_x, rect.bottom()),
        ],
        egui::Stroke::new(1.0, egui::Color32::YELLOW),
    );
}


fn filter_preview_ui(ui: &mut egui::Ui, synth: &SynthParams) {
    filter_response_preview_ui(
        ui,
        synth.filter_type,
        synth.filter_cutoff_hz,
        synth.filter_resonance,
    );
}


fn synth_modulation_ui(ui: &mut egui::Ui, synth: &mut SynthParams) {
    ui.strong("Envelope");
    ui.add(
        egui::Slider::new(&mut synth.attack_seconds, 0.0..=0.5)
            .text("Attack")
            .suffix(" s"),
    );
    ui.add(
        egui::Slider::new(&mut synth.decay_seconds, 0.02..=2.0)
            .text("Decay")
            .suffix(" s"),
    );
    ui.add(egui::Slider::new(&mut synth.sustain_level, 0.0..=1.0).text("Sustain"));
    ui.add(
        egui::Slider::new(&mut synth.release_seconds, 0.02..=2.0)
            .text("Release")
            .suffix(" s"),
    );
    ui.weak(
        "Piano-roll notes hold Sustain for their drawn length, then Release. Step-grid hits have \
         no natural length, so they treat Attack + Decay as their held time, then Release.",
    );
    ui.add(
        egui::Slider::new(&mut synth.glide_seconds, 0.0..=1.0)
            .text("Glide")
            .suffix(" s"),
    );
    ui.weak("Portamento from the previously played pitch. Only applies to piano-roll notes, not step-grid hits.");
    envelope_preview_ui(ui, synth);

    ui.separator();
    ui.strong("Filter");
    ui.horizontal(|ui| {
        ui.label("Type:");
        for (label, filter_type) in [
            ("Lowpass", FilterType::Lowpass),
            ("Highpass", FilterType::Highpass),
            ("Bandpass", FilterType::Bandpass),
            ("Notch", FilterType::Notch),
        ] {
            if ui
                .selectable_label(synth.filter_type == filter_type, label)
                .clicked()
            {
                synth.filter_type = filter_type;
            }
        }
    });
    ui.add(
        egui::Slider::new(&mut synth.filter_cutoff_hz, 20.0..=20_000.0)
            .logarithmic(true)
            .text("Cutoff")
            .suffix(" Hz"),
    );
    ui.add(egui::Slider::new(&mut synth.filter_resonance, 0.3..=10.0).text("Resonance"));
    ui.add(
        egui::Slider::new(&mut synth.filter_env_amount_hz, -10_000.0..=10_000.0)
            .text("Env amount")
            .suffix(" Hz"),
    );
    ui.weak("Env amount sweeps the cutoff from note-on, decaying over the same time as the amplitude Decay above.");
    filter_preview_ui(ui, synth);

    ui.separator();
    ui.strong("LFO");
    ui.horizontal(|ui| {
        ui.label("Waveform:");
        for (label, waveform) in [
            ("Sine", SynthWaveform::Sine),
            ("Saw", SynthWaveform::Saw),
            ("Square", SynthWaveform::Square),
            ("Triangle", SynthWaveform::Triangle),
        ] {
            if ui
                .selectable_label(synth.lfo_waveform == waveform, label)
                .clicked()
            {
                synth.lfo_waveform = waveform;
            }
        }
    });
    ui.add(
        egui::Slider::new(&mut synth.lfo_rate_hz, 0.1..=20.0)
            .text("Rate")
            .suffix(" Hz"),
    );
    ui.horizontal(|ui| {
        ui.label("Target:");
        for (label, target) in [
            ("Off", LfoTarget::None),
            ("Pitch", LfoTarget::Pitch),
            ("Amplitude", LfoTarget::Amplitude),
            ("Filter cutoff", LfoTarget::FilterCutoff),
        ] {
            if ui
                .selectable_label(synth.lfo_target == target, label)
                .clicked()
            {
                synth.lfo_target = target;
            }
        }
    });
    ui.add_enabled(
        synth.lfo_target != LfoTarget::None,
        egui::Slider::new(&mut synth.lfo_depth, 0.0..=1.0).text("Depth"),
    );
    lfo_preview_ui(ui, synth);
}


fn lfo_preview_ui(ui: &mut egui::Ui, synth: &SynthParams) {
    lfo_shape_preview_ui(
        ui,
        synth.lfo_waveform,
        synth.lfo_target != LfoTarget::None,
        synth.lfo_depth,
    );
}

