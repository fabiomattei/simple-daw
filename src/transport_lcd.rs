//! The Logic Pro–style transport LCD (Bar/Beat/Div/Tick, Tempo, Tap Tempo, time signature) and
//! `toolbar_group`, the raised-panel visual grouping shared by every cluster of controls in the
//! main toolbar — both rendered from `SimpleDawApp::update`.

use crate::model::{Song, TICKS_PER_STEP};
use crate::tempo;

/// One Bar/Beat/Div/Tick (or Tempo/Sig) cell of `transport_lcd_ui`: a zero-padded number with
/// its leading padding digits dimmed (so "004" reads as a bright "4"), and a small caption
/// underneath. `width` is the total digit count; pass 1 for single-digit fields (no padding).
fn lcd_segment(ui: &mut egui::Ui, value: usize, width: usize, label: &str) {
    ui.vertical(|ui| {
        let text = format!("{value:0>width$}");
        let dim_count = text
            .chars()
            .take_while(|c| *c == '0')
            .count()
            .min(width - 1);
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 0.0;
            if dim_count > 0 {
                ui.label(
                    egui::RichText::new(&text[..dim_count])
                        .monospace()
                        .size(14.0)
                        .color(egui::Color32::from_gray(90)),
                );
            }
            ui.label(
                egui::RichText::new(&text[dim_count..])
                    .monospace()
                    .size(14.0)
                    .color(egui::Color32::WHITE),
            );
        });
        ui.label(
            egui::RichText::new(label)
                .size(8.0)
                .color(egui::Color32::from_gray(140)),
        );
    });
}

/// Thin vertical rule between `lcd_segment` cells.
fn lcd_divider(ui: &mut egui::Ui) {
    ui.add_space(6.0);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(1.0, 20.0), egui::Sense::hover());
    ui.painter()
        .rect_filled(rect, 0.0, egui::Color32::from_rgb(60, 62, 66));
    ui.add_space(6.0);
}

/// A visually distinct cluster within the toolbar — a subtly raised rounded panel that groups
/// related controls (transport, zoom, device picker, …) so the toolbar reads as separate
/// sections instead of one undifferentiated strip of widgets. Purely presentational: callers
/// place their existing widgets inside `add_contents` unchanged.
pub(crate) fn toolbar_group(ui: &mut egui::Ui, add_contents: impl FnOnce(&mut egui::Ui)) {
    egui::Frame::new()
        .fill(egui::Color32::from_rgb(38, 38, 38))
        .corner_radius(5.0)
        .inner_margin(egui::Margin::symmetric(10, 5))
        .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(20, 20, 20)))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 8.0;
                add_contents(ui);
            });
        });
}

/// Denominators reachable from the SIG picker — restricted to values that evenly divide a
/// sixteenth-note step (see `model::Song::steps_per_beat`), so `steps_per_bar`/`steps_per_beat`
/// never need to round or reject an input.
const TIME_SIGNATURE_DENOMINATORS: [u8; 5] = [1, 2, 4, 8, 16];

/// Logic Pro–style transport LCD: Bar/Beat/Div/Tick derived from the absolute tick counter and
/// the song's own time signature, plus editable Tempo/Signature fields, in one dark rounded
/// panel.
pub(crate) fn transport_lcd_ui(
    ui: &mut egui::Ui,
    tick: usize,
    song: &mut Song,
    tap_tempo: &mut tempo::TapTempo,
) {
    let steps_per_beat = song.steps_per_beat();
    let ticks_per_beat = steps_per_beat * TICKS_PER_STEP;
    let ticks_per_bar = song.steps_per_bar() * TICKS_PER_STEP;

    let bar = tick / ticks_per_bar + 1;
    let tick_in_bar = tick % ticks_per_bar;
    let beat = tick_in_bar / ticks_per_beat + 1;
    let tick_in_beat = tick_in_bar % ticks_per_beat;
    let division = tick_in_beat / TICKS_PER_STEP + 1;
    let sub_tick = tick_in_beat % TICKS_PER_STEP;

    egui::Frame::new()
        .fill(egui::Color32::from_rgb(35, 37, 40))
        .corner_radius(6.0)
        .inner_margin(egui::Margin::symmetric(10, 4))
        .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(15, 15, 15)))
        .show(ui, |ui| {
            ui.with_layout(egui::Layout::left_to_right(egui::Align::Min), |ui| {
                lcd_segment(ui, bar, 3, "BAR");
                lcd_divider(ui);
                lcd_segment(ui, beat, 1, "BEAT");
                lcd_divider(ui);
                lcd_segment(ui, division, 1, "DIV");
                lcd_divider(ui);
                lcd_segment(ui, sub_tick, 3, "TICK");

                ui.add_space(8.0);
                egui::Frame::new()
                    .fill(egui::Color32::from_rgb(46, 49, 53))
                    .corner_radius(4.0)
                    .inner_margin(egui::Margin {
                        left: 8,
                        right: 8,
                        top: 0,
                        bottom: 6,
                    })
                    .show(ui, |ui| {
                        ui.with_layout(egui::Layout::left_to_right(egui::Align::Min), |ui| {
                            ui.spacing_mut().button_padding.y = 0.0;
                            ui.vertical(|ui| {
                                ui.scope(|ui| {
                                    ui.style_mut().override_font_id =
                                        Some(egui::FontId::monospace(14.0));
                                    ui.add(
                                        egui::DragValue::new(&mut song.bpm)
                                            .range(20.0..=300.0)
                                            .fixed_decimals(0),
                                    );
                                });
                                ui.label(
                                    egui::RichText::new("TEMPO")
                                        .size(8.0)
                                        .color(egui::Color32::from_gray(140)),
                                );
                            });
                            lcd_divider(ui);
                            ui.vertical(|ui| {
                                if ui
                                    .add(egui::Button::new("TAP").small())
                                    .on_hover_text("Click on the beat a few times to set tempo")
                                    .clicked()
                                    && let Some(bpm) = tap_tempo.tap(std::time::Instant::now())
                                {
                                    song.bpm = bpm.clamp(20.0, 300.0);
                                }
                                ui.label(
                                    egui::RichText::new("TAP")
                                        .size(8.0)
                                        .color(egui::Color32::from_gray(140)),
                                );
                            });
                            lcd_divider(ui);
                            ui.vertical(|ui| {
                                ui.scope(|ui| {
                                    ui.spacing_mut().item_spacing.x = 2.0;
                                    ui.style_mut().override_font_id =
                                        Some(egui::FontId::monospace(13.0));
                                    ui.horizontal(|ui| {
                                        ui.add(
                                            egui::DragValue::new(
                                                &mut song.time_signature_numerator,
                                            )
                                            .range(1..=32),
                                        );
                                        ui.label("/");
                                        egui::ComboBox::from_id_salt("time_signature_denominator")
                                            .selected_text(format!(
                                                "{}",
                                                song.time_signature_denominator
                                            ))
                                            .width(36.0)
                                            .show_ui(ui, |ui| {
                                                for denominator in TIME_SIGNATURE_DENOMINATORS {
                                                    ui.selectable_value(
                                                        &mut song.time_signature_denominator,
                                                        denominator,
                                                        format!("{denominator}"),
                                                    );
                                                }
                                            });
                                    });
                                });
                                ui.label(
                                    egui::RichText::new("SIG")
                                        .size(8.0)
                                        .color(egui::Color32::from_gray(140)),
                                );
                            });
                        });
                    });
            });
        });
}
