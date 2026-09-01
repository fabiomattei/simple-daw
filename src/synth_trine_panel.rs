//! The `Trine` synth engine's Device Panel UI: three oscillators, dual filter/mod matrix, LFOs,
//! and envelopes for `Track::trine` — see `trine_params_ui`, called from
//! `device_panel_contents_ui` when `Track::synth_engine == SynthEngine::Trine`.

use crate::model::{FilterRouting, ModSlot, ModSource, ModTarget, SynthWaveform, TrineParams};
use crate::synth_preview_widgets::{
    adsr_preview_ui, dual_filter_preview_ui, filter_stage_ui, lfo_shape_preview_ui, synced_oscillator_points,
    waveform_picker_ui,
};

/// Renders the Trine engine's settings, shown inside the bottom Device Panel when
/// `Track::synth_engine == SynthEngine::Trine` (see `device_panel_contents_ui`). Laid out as
/// three columns (oscillators | filter + modulation matrix | LFOs + envelopes) so every section
/// is visible at once instead of stacked behind collapsing headers, mirroring `synth_params_ui`'s
/// two-column layout but wider since Trine has considerably more surface.
pub(crate) fn trine_params_ui(ui: &mut egui::Ui, trine: &mut TrineParams) {
    ui.weak(
        "A second, independent synth engine: 3 oscillators, a dual filter, and a free \
         modulation matrix. Envelope 3 always drives amplitude; everything else only does \
         something once routed in the Modulation Matrix section below.",
    );
    ui.separator();
    ui.columns(3, |columns| {
        trine_oscillators_ui(&mut columns[0], trine);
        trine_filter_matrix_ui(&mut columns[1], trine);
        trine_lfos_envelopes_ui(&mut columns[2], trine);
    });
}


/// Small canvas overlaying all three of Trine's oscillators: Oscillator 1 in blue (an undetuned,
/// 3-cycle reference), Oscillator 2 in orange and Oscillator 3 in green, both pitch- and
/// sync-accurate against Oscillator 1 using the same math as `oscillator2_preview_ui`, each
/// scaled by its own Level. FM and ring mod aren't reflected — they're audio-rate interactions a
/// static per-oscillator shape can't show — only mix level and pitch/sync relationships are.
fn trine_oscillators_preview_ui(ui: &mut egui::Ui, trine: &TrineParams) {
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

    let cycles = 3.0;
    let samples = 300;
    if trine.osc1_level > 0.0 {
        let points = synced_oscillator_points(
            rect,
            trine.osc1_waveform,
            trine.pulse_width,
            1.0,
            false,
            trine.osc1_level.max(0.15),
            cycles,
            samples,
        );
        painter.add(egui::Shape::line(
            points,
            egui::Stroke::new(2.0, egui::Color32::from_rgb(120, 170, 230)),
        ));
    }
    if trine.osc2_level > 0.0 {
        let ratio = 2f32.powf(trine.osc2_semitones as f32 / 12.0)
            * 2f32.powf(trine.osc2_detune_cents / 1200.0);
        let points = synced_oscillator_points(
            rect,
            trine.osc2_waveform,
            trine.pulse_width,
            ratio,
            trine.osc2_sync,
            trine.osc2_level,
            cycles,
            samples,
        );
        let color =
            egui::Color32::from_rgb(230, 160, 90).gamma_multiply(trine.osc2_level.max(0.25));
        painter.add(egui::Shape::line(points, egui::Stroke::new(1.5, color)));
    }
    if trine.osc3_level > 0.0 {
        let ratio = 2f32.powf(trine.osc3_semitones as f32 / 12.0)
            * 2f32.powf(trine.osc3_detune_cents / 1200.0);
        let points = synced_oscillator_points(
            rect,
            trine.osc3_waveform,
            trine.pulse_width,
            ratio,
            trine.osc3_sync,
            trine.osc3_level,
            cycles,
            samples,
        );
        let color =
            egui::Color32::from_rgb(140, 220, 170).gamma_multiply(trine.osc3_level.max(0.25));
        painter.add(egui::Shape::line(points, egui::Stroke::new(1.5, color)));
    }
}


/// Small canvas dedicated to one of Trine's Oscillator 2/3: a faint reference line for
/// Oscillator 1 and this oscillator's own shape overlaid in its accent color, pitch- and
/// sync-accurate against Oscillator 1 via `synced_oscillator_points` — the same math
/// `oscillator2_preview_ui` uses for the base synth, and the same per-oscillator math
/// `trine_oscillators_preview_ui` overlays for all three at once.
fn trine_oscillator_n_preview_ui(
    ui: &mut egui::Ui,
    trine: &TrineParams,
    waveform: SynthWaveform,
    semitones: i32,
    detune_cents: f32,
    sync: bool,
    color: egui::Color32,
) {
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

    let cycles = 3.0;
    let samples = 300;
    let osc1_points = synced_oscillator_points(
        rect,
        trine.osc1_waveform,
        trine.pulse_width,
        1.0,
        false,
        0.6,
        cycles,
        samples,
    );
    painter.add(egui::Shape::line(
        osc1_points,
        egui::Stroke::new(1.0, ui.visuals().weak_text_color()),
    ));

    let ratio = 2f32.powf(semitones as f32 / 12.0) * 2f32.powf(detune_cents / 1200.0);
    let osc_points = synced_oscillator_points(
        rect,
        waveform,
        trine.pulse_width,
        ratio,
        sync,
        1.0,
        cycles,
        samples,
    );
    painter.add(egui::Shape::line(osc_points, egui::Stroke::new(2.0, color)));
}


fn trine_oscillators_ui(ui: &mut egui::Ui, trine: &mut TrineParams) {
    ui.strong("Oscillators");
    trine_oscillators_preview_ui(ui, trine);
    ui.separator();
    ui.strong("Oscillator 1");
    waveform_picker_ui(ui, &mut trine.osc1_waveform);
    ui.add(egui::Slider::new(&mut trine.osc1_level, 0.0..=1.0).text("Level"));

    ui.separator();
    ui.strong("Oscillator 2");
    waveform_picker_ui(ui, &mut trine.osc2_waveform);
    ui.add(
        egui::Slider::new(&mut trine.osc2_semitones, -24..=24)
            .text("Coarse tune")
            .suffix(" st"),
    );
    ui.add(
        egui::Slider::new(&mut trine.osc2_detune_cents, -50.0..=50.0)
            .text("Fine tune")
            .suffix(" cents"),
    );
    ui.add(egui::Slider::new(&mut trine.osc2_level, 0.0..=1.0).text("Level"));
    ui.checkbox(&mut trine.osc2_sync, "Sync to Oscillator 1");
    trine_oscillator_n_preview_ui(
        ui,
        trine,
        trine.osc2_waveform,
        trine.osc2_semitones,
        trine.osc2_detune_cents,
        trine.osc2_sync,
        egui::Color32::from_rgb(230, 160, 90),
    );

    ui.separator();
    ui.strong("Oscillator 3");
    waveform_picker_ui(ui, &mut trine.osc3_waveform);
    ui.add(
        egui::Slider::new(&mut trine.osc3_semitones, -24..=24)
            .text("Coarse tune")
            .suffix(" st"),
    );
    ui.add(
        egui::Slider::new(&mut trine.osc3_detune_cents, -50.0..=50.0)
            .text("Fine tune")
            .suffix(" cents"),
    );
    ui.add(egui::Slider::new(&mut trine.osc3_level, 0.0..=1.0).text("Level"));
    ui.checkbox(&mut trine.osc3_sync, "Sync to Oscillator 1");
    trine_oscillator_n_preview_ui(
        ui,
        trine,
        trine.osc3_waveform,
        trine.osc3_semitones,
        trine.osc3_detune_cents,
        trine.osc3_sync,
        egui::Color32::from_rgb(140, 220, 170),
    );

    ui.separator();
    ui.add(
        egui::Slider::new(&mut trine.pulse_width, 0.05..=0.95)
            .text("Pulse width")
            .suffix(" (Square oscillators)"),
    );
    ui.add(egui::Slider::new(&mut trine.fm_amount, 0.0..=4.0).text("FM amount"))
        .on_hover_text("Oscillator 2 frequency-modulates Oscillator 1, independent of Oscillator 2's own level.");
    ui.add(egui::Slider::new(&mut trine.ring_mod_mix, 0.0..=1.0).text("Ring mod mix"))
        .on_hover_text(
            "Oscillator 1 x Oscillator 2, mixed in on top of the regular oscillator sum.",
        );
    ui.add(egui::Slider::new(&mut trine.analog_drift, 0.0..=1.0).text("Analog drift"))
        .on_hover_text(
            "Slow per-voice random pitch wander, emulating analog oscillator instability.",
        );
}


/// Combines `trine_filter_ui` and `trine_matrix_ui` into Trine's middle column.
fn trine_filter_matrix_ui(ui: &mut egui::Ui, trine: &mut TrineParams) {
    ui.strong("Filter");
    trine_filter_ui(ui, trine);
    ui.separator();
    ui.strong("Modulation Matrix");
    trine_matrix_ui(ui, trine);
}


fn trine_filter_ui(ui: &mut egui::Ui, trine: &mut TrineParams) {
    ui.horizontal(|ui| {
        ui.label("Routing:");
        for (label, routing) in [
            ("Off", FilterRouting::Off),
            ("Series", FilterRouting::Series),
            ("Parallel", FilterRouting::Parallel),
        ] {
            if ui
                .selectable_label(trine.filter_routing == routing, label)
                .clicked()
            {
                trine.filter_routing = routing;
            }
        }
    });
    ui.weak("Off uses Filter 1 alone; Series feeds Filter 1 into Filter 2; Parallel sums both filters' output.");

    ui.strong("Filter 1");
    filter_stage_ui(
        ui,
        &mut trine.filter1_cutoff_hz,
        &mut trine.filter1_resonance,
        &mut trine.filter1_type,
        &mut trine.filter1_slope,
    );

    ui.add_enabled_ui(trine.filter_routing != FilterRouting::Off, |ui| {
        ui.separator();
        ui.strong("Filter 2");
        filter_stage_ui(
            ui,
            &mut trine.filter2_cutoff_hz,
            &mut trine.filter2_resonance,
            &mut trine.filter2_type,
            &mut trine.filter2_slope,
        );
    });

    ui.separator();
    ui.add(egui::Slider::new(&mut trine.filter_drive, 0.0..=1.0).text("Drive"))
        .on_hover_text("Soft-clip saturation applied before Filter 1.");
    ui.add(egui::Slider::new(&mut trine.filter_fm_amount, 0.0..=1.0).text("Filter FM"))
        .on_hover_text("Filter 1 cutoff modulated directly by Oscillator 2's instantaneous output (audio-rate).");

    dual_filter_preview_ui(
        ui,
        trine.filter1_type,
        trine.filter1_cutoff_hz,
        trine.filter1_resonance,
        trine.filter2_type,
        trine.filter2_cutoff_hz,
        trine.filter2_resonance,
        trine.filter_routing,
    );
}


fn trine_matrix_ui(ui: &mut egui::Ui, trine: &mut TrineParams) {
    ui.weak("Route a modulation source to a target with a bipolar amount. Empty by default.");
    let mut to_remove = None;
    for (index, slot) in trine.mod_slots.iter_mut().enumerate() {
        ui.horizontal(|ui| {
            egui::ComboBox::from_id_salt(("trine-mod-source", index))
                .selected_text(mod_source_label(slot.source))
                .show_ui(ui, |ui| {
                    for source in [
                        ModSource::None,
                        ModSource::Lfo1,
                        ModSource::Lfo2,
                        ModSource::Env1,
                        ModSource::Env2,
                        ModSource::Velocity,
                    ] {
                        ui.selectable_value(&mut slot.source, source, mod_source_label(source));
                    }
                });
            ui.label("->");
            egui::ComboBox::from_id_salt(("trine-mod-target", index))
                .selected_text(mod_target_label(slot.target))
                .show_ui(ui, |ui| {
                    for target in [
                        ModTarget::None,
                        ModTarget::Pitch,
                        ModTarget::Osc1Level,
                        ModTarget::Osc2Level,
                        ModTarget::Osc3Level,
                        ModTarget::PulseWidth,
                        ModTarget::FilterCutoff,
                        ModTarget::Filter2Cutoff,
                        ModTarget::FilterResonance,
                        ModTarget::FmAmount,
                        ModTarget::RingModMix,
                    ] {
                        ui.selectable_value(&mut slot.target, target, mod_target_label(target));
                    }
                });
            ui.add(egui::Slider::new(&mut slot.amount, -1.0..=1.0).text("Amount"));
            if ui.button("🗑").on_hover_text("Remove this mod slot").clicked() {
                to_remove = Some(index);
            }
        });
    }
    if let Some(index) = to_remove {
        trine.mod_slots.remove(index);
    }
    if ui.button("+ Add slot").clicked() {
        trine.mod_slots.push(ModSlot::default());
    }
}


fn mod_source_label(source: ModSource) -> &'static str {
    match source {
        ModSource::None => "— none —",
        ModSource::Lfo1 => "LFO 1",
        ModSource::Lfo2 => "LFO 2",
        ModSource::Env1 => "Envelope 1",
        ModSource::Env2 => "Envelope 2",
        ModSource::Velocity => "Velocity",
    }
}


fn mod_target_label(target: ModTarget) -> &'static str {
    match target {
        ModTarget::None => "— none —",
        ModTarget::Pitch => "Pitch",
        ModTarget::Osc1Level => "Osc 1 Level",
        ModTarget::Osc2Level => "Osc 2 Level",
        ModTarget::Osc3Level => "Osc 3 Level",
        ModTarget::PulseWidth => "Pulse Width",
        ModTarget::FilterCutoff => "Filter 1 Cutoff",
        ModTarget::Filter2Cutoff => "Filter 2 Cutoff",
        ModTarget::FilterResonance => "Filter 1 Resonance",
        ModTarget::FmAmount => "FM Amount",
        ModTarget::RingModMix => "Ring Mod Mix",
    }
}


/// Whether `source` is actually wired to something in `mod_slots` (routed to a non-`None` target
/// with a non-zero amount) and, if so, the largest magnitude it's routed at — used to decide
/// whether an LFO's preview canvas should render as "live" or grayed-out, since Trine/Wave's LFOs
/// have no target/depth of their own the way `SynthParams`'s single LFO does.
fn trine_lfo_active_depth(mod_slots: &[ModSlot], source: ModSource) -> (bool, f32) {
    let depth = mod_slots
        .iter()
        .filter(|slot| slot.source == source && slot.target != ModTarget::None)
        .map(|slot| slot.amount.abs())
        .fold(0.0f32, f32::max);
    (depth > 0.001, depth)
}


/// Combines Trine's LFOs and envelopes into the third column.
fn trine_lfos_envelopes_ui(ui: &mut egui::Ui, trine: &mut TrineParams) {
    ui.strong("LFOs");
    trine_lfos_ui(ui, trine);
    ui.separator();
    ui.strong("Envelopes");
    trine_envelopes_ui(ui, trine);
}


fn trine_lfos_ui(ui: &mut egui::Ui, trine: &mut TrineParams) {
    ui.strong("LFO 1");
    waveform_picker_ui(ui, &mut trine.lfo1_waveform);
    ui.add(
        egui::Slider::new(&mut trine.lfo1_rate_hz, 0.1..=20.0)
            .text("Rate")
            .suffix(" Hz"),
    );
    let (active1, depth1) = trine_lfo_active_depth(&trine.mod_slots, ModSource::Lfo1);
    lfo_shape_preview_ui(ui, trine.lfo1_waveform, active1, depth1);

    ui.separator();
    ui.strong("LFO 2");
    waveform_picker_ui(ui, &mut trine.lfo2_waveform);
    ui.add(
        egui::Slider::new(&mut trine.lfo2_rate_hz, 0.1..=20.0)
            .text("Rate")
            .suffix(" Hz"),
    );
    let (active2, depth2) = trine_lfo_active_depth(&trine.mod_slots, ModSource::Lfo2);
    lfo_shape_preview_ui(ui, trine.lfo2_waveform, active2, depth2);
}


fn trine_envelopes_ui(ui: &mut egui::Ui, trine: &mut TrineParams) {
    ui.strong("Envelope 1")
        .on_hover_text("Free — only audible once routed in the Modulation Matrix.");
    ui.add(
        egui::Slider::new(&mut trine.env1_attack_seconds, 0.0..=2.0)
            .text("Attack")
            .suffix(" s"),
    );
    ui.add(
        egui::Slider::new(&mut trine.env1_decay_seconds, 0.02..=2.0)
            .text("Decay")
            .suffix(" s"),
    );
    ui.add(egui::Slider::new(&mut trine.env1_sustain_level, 0.0..=1.0).text("Sustain"));
    ui.add(
        egui::Slider::new(&mut trine.env1_release_seconds, 0.02..=2.0)
            .text("Release")
            .suffix(" s"),
    );
    adsr_preview_ui(
        ui,
        trine.env1_attack_seconds,
        trine.env1_decay_seconds,
        trine.env1_sustain_level,
        trine.env1_release_seconds,
        egui::Color32::from_rgb(200, 140, 230),
    );

    ui.separator();
    ui.strong("Envelope 2")
        .on_hover_text("Free — only audible once routed in the Modulation Matrix.");
    ui.add(
        egui::Slider::new(&mut trine.env2_attack_seconds, 0.0..=2.0)
            .text("Attack")
            .suffix(" s"),
    );
    ui.add(
        egui::Slider::new(&mut trine.env2_decay_seconds, 0.02..=2.0)
            .text("Decay")
            .suffix(" s"),
    );
    ui.add(egui::Slider::new(&mut trine.env2_sustain_level, 0.0..=1.0).text("Sustain"));
    ui.add(
        egui::Slider::new(&mut trine.env2_release_seconds, 0.02..=2.0)
            .text("Release")
            .suffix(" s"),
    );
    adsr_preview_ui(
        ui,
        trine.env2_attack_seconds,
        trine.env2_decay_seconds,
        trine.env2_sustain_level,
        trine.env2_release_seconds,
        egui::Color32::from_rgb(200, 140, 230),
    );

    ui.separator();
    ui.strong("Envelope 3 (Volume)")
        .on_hover_text("Always active — directly drives amplitude.");
    ui.add(
        egui::Slider::new(&mut trine.env3_attack_seconds, 0.0..=2.0)
            .text("Attack")
            .suffix(" s"),
    );
    ui.add(
        egui::Slider::new(&mut trine.env3_decay_seconds, 0.02..=2.0)
            .text("Decay")
            .suffix(" s"),
    );
    ui.add(egui::Slider::new(&mut trine.env3_sustain_level, 0.0..=1.0).text("Sustain"));
    ui.add(
        egui::Slider::new(&mut trine.env3_release_seconds, 0.02..=2.0)
            .text("Release")
            .suffix(" s"),
    );
    adsr_preview_ui(
        ui,
        trine.env3_attack_seconds,
        trine.env3_decay_seconds,
        trine.env3_sustain_level,
        trine.env3_release_seconds,
        egui::Color32::from_rgb(120, 220, 140),
    );
}

