//! The `Wave` synth engine's Device Panel UI: two wavetable oscillators with position-morph and
//! phase-warp, dual filter/mod matrix, LFOs, and envelopes for `Track::wave` — see
//! `wave_params_ui`, called from `device_panel_contents_ui` when
//! `Track::synth_engine == SynthEngine::Wave`.

use crate::model::{FilterRouting, WaveModSlot, WaveModSource, WaveModTarget, WaveParams};
use crate::synth_preview_widgets::{
    adsr_preview_ui, dual_filter_preview_ui, filter_stage_ui, lfo_shape_preview_ui, waveform_picker_ui,
};
use crate::wavetable::{self, WaveWarpMode, WavetableId};

/// A row of selectable labels for every `WavetableId` variant — see `waveform_picker_ui`, the
/// equivalent for classic waveforms.
fn wavetable_picker_ui(ui: &mut egui::Ui, current: &mut WavetableId) {
    ui.horizontal(|ui| {
        ui.label("Table:");
        for id in WavetableId::ALL {
            if ui.selectable_label(*current == id, id.label()).clicked() {
                *current = id;
            }
        }
    });
}


/// A row of selectable labels for every `WaveWarpMode` variant, plus an amount slider (disabled
/// while `Off`, matching `filter_stage_ui`'s "Off" no-op precedent elsewhere in this file).
fn warp_mode_picker_ui(ui: &mut egui::Ui, mode: &mut WaveWarpMode, amount: &mut f32) {
    ui.horizontal(|ui| {
        ui.label("Warp:");
        for (label, m) in [
            ("Off", WaveWarpMode::Off),
            ("Bend", WaveWarpMode::Bend),
            ("Sync", WaveWarpMode::Sync),
            ("Mirror", WaveWarpMode::Mirror),
            ("FM", WaveWarpMode::Fm),
        ] {
            if ui.selectable_label(*mode == m, label).clicked() {
                *mode = m;
            }
        }
    });
    ui.add_enabled(
        *mode != WaveWarpMode::Off,
        egui::Slider::new(amount, 0.0..=1.0).text("Warp amount"),
    );
}


/// Renders the Wave engine's settings, shown inside a track's synth window when
/// `Track::synth_engine == SynthEngine::Wave`. Laid out as three columns (oscillators | filter +
/// modulation matrix | LFOs + envelopes), the same structure as `trine_params_ui`.
pub(crate) fn wave_params_ui(ui: &mut egui::Ui, wave: &mut WaveParams) {
    ui.weak(
        "A third synth engine loosely inspired by wavetable synths like Serum: 2 wavetable \
         oscillators (each scanning its table's frames, with an optional phase-warp), a sub \
         oscillator, noise, a dual filter, and a free modulation matrix. The amplitude envelope \
         always drives volume; everything else only does something once routed in the \
         Modulation Matrix section below.",
    );
    ui.separator();
    ui.columns(3, |columns| {
        wave_oscillators_ui(&mut columns[0], wave);
        wave_filter_matrix_ui(&mut columns[1], wave);
        wave_lfos_envelopes_ui(&mut columns[2], wave);
    });
}


/// Samples one full cycle (`phase` 0..1) of a `WaveParams` oscillator directly from its actual
/// wavetable data — `wavetable::sample` at mip level 0 (the highest-fidelity mip; previews aren't
/// played back at pitch, so aliasing doesn't apply), through `wavetable::warp_phase` first so
/// Bend/Sync/Mirror/FM warp modes show up exactly as they'd sound, not an approximation.
fn wave_oscillator_points(
    rect: egui::Rect,
    table: WavetableId,
    position: f32,
    warp_mode: WaveWarpMode,
    warp_amount: f32,
    amplitude: f32,
    samples: usize,
) -> Vec<egui::Pos2> {
    let mid_y = rect.center().y;
    let half_h = rect.height() * 0.5 * 0.9;
    (0..=samples)
        .map(|i| {
            let phase = i as f32 / samples as f32;
            let warped = wavetable::warp_phase(phase, warp_mode, warp_amount);
            let sample = wavetable::sample(table, position, warped, 0) * amplitude;
            egui::pos2(rect.left() + phase * rect.width(), mid_y - sample * half_h)
        })
        .collect()
}


/// Small canvas overlaying Wave's two oscillators, each sampled straight from its actual
/// wavetable (see `wave_oscillator_points`): Oscillator 1 in blue, Oscillator 2 faded in
/// proportion to its Level in orange. Since a wavetable oscillator is periodic in phase 0..1
/// regardless of pitch, both are drawn as one cycle rather than pitch/sync-accurate like
/// `trine_oscillators_preview_ui` — this shows table/position/warp timbre, not tuning.
fn wave_oscillators_preview_ui(ui: &mut egui::Ui, wave: &WaveParams) {
    let (response, painter) =
        ui.allocate_painter(egui::vec2(ui.available_width(), 70.0), egui::Sense::hover());
    let rect = response.rect;
    painter.rect_filled(rect, 2.0, ui.visuals().extreme_bg_color);
    painter.line_segment(
        [
            egui::pos2(rect.left(), rect.center().y),
            egui::pos2(rect.right(), rect.center().y),
        ],
        egui::Stroke::new(0.5, ui.visuals().weak_text_color()),
    );

    let samples = 300;
    if wave.osc1_level > 0.0 {
        let points = wave_oscillator_points(
            rect,
            wave.osc1_table,
            wave.osc1_position,
            wave.osc1_warp_mode,
            wave.osc1_warp_amount,
            wave.osc1_level.max(0.15),
            samples,
        );
        painter.add(egui::Shape::line(
            points,
            egui::Stroke::new(2.0, egui::Color32::from_rgb(120, 170, 230)),
        ));
    }
    if wave.osc2_level > 0.0 {
        let points = wave_oscillator_points(
            rect,
            wave.osc2_table,
            wave.osc2_position,
            wave.osc2_warp_mode,
            wave.osc2_warp_amount,
            wave.osc2_level,
            samples,
        );
        let color = egui::Color32::from_rgb(230, 160, 90).gamma_multiply(wave.osc2_level.max(0.25));
        painter.add(egui::Shape::line(points, egui::Stroke::new(1.5, color)));
    }
}


/// Small canvas dedicated to Wave's Oscillator 2: a faint reference cycle for Oscillator 1 and
/// Oscillator 2's own shape overlaid in orange, both sampled directly from their wavetable data
/// via `wave_oscillator_points` — the single-oscillator counterpart to
/// `wave_oscillators_preview_ui`'s combined overlay.
fn wave_oscillator2_preview_ui(ui: &mut egui::Ui, wave: &WaveParams) {
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

    let samples = 300;
    let osc1_points = wave_oscillator_points(
        rect,
        wave.osc1_table,
        wave.osc1_position,
        wave.osc1_warp_mode,
        wave.osc1_warp_amount,
        0.6,
        samples,
    );
    painter.add(egui::Shape::line(
        osc1_points,
        egui::Stroke::new(1.0, ui.visuals().weak_text_color()),
    ));

    let osc2_points = wave_oscillator_points(
        rect,
        wave.osc2_table,
        wave.osc2_position,
        wave.osc2_warp_mode,
        wave.osc2_warp_amount,
        1.0,
        samples,
    );
    painter.add(egui::Shape::line(
        osc2_points,
        egui::Stroke::new(2.0, egui::Color32::from_rgb(230, 160, 90)),
    ));
}


fn wave_oscillators_ui(ui: &mut egui::Ui, wave: &mut WaveParams) {
    ui.strong("Oscillators");
    wave_oscillators_preview_ui(ui, wave);
    ui.separator();
    ui.strong("Oscillator 1");
    wavetable_picker_ui(ui, &mut wave.osc1_table);
    ui.add(egui::Slider::new(&mut wave.osc1_position, 0.0..=1.0).text("Position"));
    warp_mode_picker_ui(ui, &mut wave.osc1_warp_mode, &mut wave.osc1_warp_amount);
    ui.add(egui::Slider::new(&mut wave.osc1_level, 0.0..=1.0).text("Level"));
    ui.horizontal(|ui| {
        ui.label("Unison:");
        for voices in 1..=3u8 {
            if ui
                .selectable_label(wave.unison_voices == voices, voices.to_string())
                .clicked()
            {
                wave.unison_voices = voices;
            }
        }
    });
    ui.add_enabled(
        wave.unison_voices > 1,
        egui::Slider::new(&mut wave.unison_detune_cents, 0.0..=50.0)
            .text("Unison detune")
            .suffix(" cents"),
    );
    ui.add_enabled(
        wave.unison_voices > 1,
        egui::Slider::new(&mut wave.unison_width, 0.0..=1.0).text("Unison width"),
    )
    .on_hover_text("Spreads unison voices across the stereo field. 0 keeps them centered.");

    ui.separator();
    ui.strong("Oscillator 2");
    wavetable_picker_ui(ui, &mut wave.osc2_table);
    ui.add(egui::Slider::new(&mut wave.osc2_position, 0.0..=1.0).text("Position"));
    warp_mode_picker_ui(ui, &mut wave.osc2_warp_mode, &mut wave.osc2_warp_amount);
    ui.add(
        egui::Slider::new(&mut wave.osc2_semitones, -24..=24)
            .text("Coarse tune")
            .suffix(" st"),
    );
    ui.add(
        egui::Slider::new(&mut wave.osc2_detune_cents, -50.0..=50.0)
            .text("Fine tune")
            .suffix(" cents"),
    );
    ui.add(egui::Slider::new(&mut wave.osc2_level, 0.0..=1.0).text("Level"));
    wave_oscillator2_preview_ui(ui, wave);

    ui.separator();
    ui.strong("Sub / Noise");
    ui.add(
        egui::Slider::new(&mut wave.sub_osc_semitones, -24..=0)
            .text("Sub tune")
            .suffix(" st"),
    );
    ui.add(egui::Slider::new(&mut wave.sub_osc_level, 0.0..=1.0).text("Sub level"));
    ui.add(egui::Slider::new(&mut wave.noise_level, 0.0..=1.0).text("Noise level"));
}


/// Combines `wave_filter_ui` and `wave_matrix_ui` into Wave's middle column.
fn wave_filter_matrix_ui(ui: &mut egui::Ui, wave: &mut WaveParams) {
    ui.strong("Filter");
    wave_filter_ui(ui, wave);
    ui.separator();
    ui.strong("Modulation Matrix");
    wave_matrix_ui(ui, wave);
}


fn wave_filter_ui(ui: &mut egui::Ui, wave: &mut WaveParams) {
    ui.horizontal(|ui| {
        ui.label("Routing:");
        for (label, routing) in [
            ("Off", FilterRouting::Off),
            ("Series", FilterRouting::Series),
            ("Parallel", FilterRouting::Parallel),
        ] {
            if ui
                .selectable_label(wave.filter_routing == routing, label)
                .clicked()
            {
                wave.filter_routing = routing;
            }
        }
    });
    ui.weak("Off uses Filter 1 alone; Series feeds Filter 1 into Filter 2; Parallel sums both filters' output.");

    ui.strong("Filter 1");
    filter_stage_ui(
        ui,
        &mut wave.filter1_cutoff_hz,
        &mut wave.filter1_resonance,
        &mut wave.filter1_type,
        &mut wave.filter1_slope,
    );

    ui.add_enabled_ui(wave.filter_routing != FilterRouting::Off, |ui| {
        ui.separator();
        ui.strong("Filter 2");
        filter_stage_ui(
            ui,
            &mut wave.filter2_cutoff_hz,
            &mut wave.filter2_resonance,
            &mut wave.filter2_type,
            &mut wave.filter2_slope,
        );
    });

    ui.separator();
    ui.add(egui::Slider::new(&mut wave.filter_drive, 0.0..=1.0).text("Drive"))
        .on_hover_text("Soft-clip saturation applied before Filter 1.");

    dual_filter_preview_ui(
        ui,
        wave.filter1_type,
        wave.filter1_cutoff_hz,
        wave.filter1_resonance,
        wave.filter2_type,
        wave.filter2_cutoff_hz,
        wave.filter2_resonance,
        wave.filter_routing,
    );
}


fn wave_matrix_ui(ui: &mut egui::Ui, wave: &mut WaveParams) {
    ui.weak("Route a modulation source to a target with a bipolar amount. Empty by default.");
    let mut to_remove = None;
    for (index, slot) in wave.mod_slots.iter_mut().enumerate() {
        ui.horizontal(|ui| {
            egui::ComboBox::from_id_salt(("wave-mod-source", index))
                .selected_text(wave_mod_source_label(slot.source))
                .show_ui(ui, |ui| {
                    for source in [
                        WaveModSource::None,
                        WaveModSource::Lfo1,
                        WaveModSource::Lfo2,
                        WaveModSource::Env1,
                        WaveModSource::Env2,
                        WaveModSource::Velocity,
                    ] {
                        ui.selectable_value(
                            &mut slot.source,
                            source,
                            wave_mod_source_label(source),
                        );
                    }
                });
            ui.label("->");
            egui::ComboBox::from_id_salt(("wave-mod-target", index))
                .selected_text(wave_mod_target_label(slot.target))
                .show_ui(ui, |ui| {
                    for target in [
                        WaveModTarget::None,
                        WaveModTarget::Pitch,
                        WaveModTarget::Osc1Position,
                        WaveModTarget::Osc2Position,
                        WaveModTarget::Osc1WarpAmount,
                        WaveModTarget::Osc2WarpAmount,
                        WaveModTarget::FilterCutoff,
                        WaveModTarget::Filter2Cutoff,
                        WaveModTarget::FilterResonance,
                    ] {
                        ui.selectable_value(
                            &mut slot.target,
                            target,
                            wave_mod_target_label(target),
                        );
                    }
                });
            ui.add(egui::Slider::new(&mut slot.amount, -1.0..=1.0).text("Amount"));
            if ui.button("🗑").on_hover_text("Remove this mod slot").clicked() {
                to_remove = Some(index);
            }
        });
    }
    if let Some(index) = to_remove {
        wave.mod_slots.remove(index);
    }
    if ui.button("+ Add slot").clicked() {
        wave.mod_slots.push(WaveModSlot::default());
    }
}


fn wave_mod_source_label(source: WaveModSource) -> &'static str {
    match source {
        WaveModSource::None => "— none —",
        WaveModSource::Lfo1 => "LFO 1",
        WaveModSource::Lfo2 => "LFO 2",
        WaveModSource::Env1 => "Envelope 1",
        WaveModSource::Env2 => "Envelope 2",
        WaveModSource::Velocity => "Velocity",
    }
}


fn wave_mod_target_label(target: WaveModTarget) -> &'static str {
    match target {
        WaveModTarget::None => "— none —",
        WaveModTarget::Pitch => "Pitch",
        WaveModTarget::Osc1Position => "Osc 1 Position",
        WaveModTarget::Osc2Position => "Osc 2 Position",
        WaveModTarget::Osc1WarpAmount => "Osc 1 Warp Amount",
        WaveModTarget::Osc2WarpAmount => "Osc 2 Warp Amount",
        WaveModTarget::FilterCutoff => "Filter 1 Cutoff",
        WaveModTarget::Filter2Cutoff => "Filter 2 Cutoff",
        WaveModTarget::FilterResonance => "Filter 1 Resonance",
    }
}


/// Whether `source` is actually wired to something in `mod_slots` and, if so, the largest
/// magnitude it's routed at — see `trine_lfo_active_depth`, the `ModSlot` equivalent.
fn wave_lfo_active_depth(mod_slots: &[WaveModSlot], source: WaveModSource) -> (bool, f32) {
    let depth = mod_slots
        .iter()
        .filter(|slot| slot.source == source && slot.target != WaveModTarget::None)
        .map(|slot| slot.amount.abs())
        .fold(0.0f32, f32::max);
    (depth > 0.001, depth)
}


/// Combines Wave's LFOs and envelopes into the third column.
fn wave_lfos_envelopes_ui(ui: &mut egui::Ui, wave: &mut WaveParams) {
    ui.strong("LFOs");
    wave_lfos_ui(ui, wave);
    ui.separator();
    ui.strong("Envelopes");
    wave_envelopes_ui(ui, wave);
}


fn wave_lfos_ui(ui: &mut egui::Ui, wave: &mut WaveParams) {
    ui.strong("LFO 1");
    waveform_picker_ui(ui, &mut wave.lfo1_waveform);
    ui.add(
        egui::Slider::new(&mut wave.lfo1_rate_hz, 0.1..=20.0)
            .text("Rate")
            .suffix(" Hz"),
    );
    let (active1, depth1) = wave_lfo_active_depth(&wave.mod_slots, WaveModSource::Lfo1);
    lfo_shape_preview_ui(ui, wave.lfo1_waveform, active1, depth1);

    ui.separator();
    ui.strong("LFO 2");
    waveform_picker_ui(ui, &mut wave.lfo2_waveform);
    ui.add(
        egui::Slider::new(&mut wave.lfo2_rate_hz, 0.1..=20.0)
            .text("Rate")
            .suffix(" Hz"),
    );
    let (active2, depth2) = wave_lfo_active_depth(&wave.mod_slots, WaveModSource::Lfo2);
    lfo_shape_preview_ui(ui, wave.lfo2_waveform, active2, depth2);
}


fn wave_envelopes_ui(ui: &mut egui::Ui, wave: &mut WaveParams) {
    ui.strong("Envelope 1")
        .on_hover_text("Free — only audible once routed in the Modulation Matrix.");
    ui.add(
        egui::Slider::new(&mut wave.env1_attack_seconds, 0.0..=2.0)
            .text("Attack")
            .suffix(" s"),
    );
    ui.add(
        egui::Slider::new(&mut wave.env1_decay_seconds, 0.02..=2.0)
            .text("Decay")
            .suffix(" s"),
    );
    ui.add(egui::Slider::new(&mut wave.env1_sustain_level, 0.0..=1.0).text("Sustain"));
    ui.add(
        egui::Slider::new(&mut wave.env1_release_seconds, 0.02..=2.0)
            .text("Release")
            .suffix(" s"),
    );
    adsr_preview_ui(
        ui,
        wave.env1_attack_seconds,
        wave.env1_decay_seconds,
        wave.env1_sustain_level,
        wave.env1_release_seconds,
        egui::Color32::from_rgb(200, 140, 230),
    );

    ui.separator();
    ui.strong("Envelope 2")
        .on_hover_text("Free — only audible once routed in the Modulation Matrix.");
    ui.add(
        egui::Slider::new(&mut wave.env2_attack_seconds, 0.0..=2.0)
            .text("Attack")
            .suffix(" s"),
    );
    ui.add(
        egui::Slider::new(&mut wave.env2_decay_seconds, 0.02..=2.0)
            .text("Decay")
            .suffix(" s"),
    );
    ui.add(egui::Slider::new(&mut wave.env2_sustain_level, 0.0..=1.0).text("Sustain"));
    ui.add(
        egui::Slider::new(&mut wave.env2_release_seconds, 0.02..=2.0)
            .text("Release")
            .suffix(" s"),
    );
    adsr_preview_ui(
        ui,
        wave.env2_attack_seconds,
        wave.env2_decay_seconds,
        wave.env2_sustain_level,
        wave.env2_release_seconds,
        egui::Color32::from_rgb(200, 140, 230),
    );

    ui.separator();
    ui.strong("Amplitude Envelope")
        .on_hover_text("Always active — directly drives amplitude.");
    ui.add(
        egui::Slider::new(&mut wave.amp_attack_seconds, 0.0..=2.0)
            .text("Attack")
            .suffix(" s"),
    );
    ui.add(
        egui::Slider::new(&mut wave.amp_decay_seconds, 0.02..=2.0)
            .text("Decay")
            .suffix(" s"),
    );
    ui.add(egui::Slider::new(&mut wave.amp_sustain_level, 0.0..=1.0).text("Sustain"));
    ui.add(
        egui::Slider::new(&mut wave.amp_release_seconds, 0.02..=2.0)
            .text("Release")
            .suffix(" s"),
    );
    adsr_preview_ui(
        ui,
        wave.amp_attack_seconds,
        wave.amp_decay_seconds,
        wave.amp_sustain_level,
        wave.amp_release_seconds,
        egui::Color32::from_rgb(120, 220, 140),
    );
}

