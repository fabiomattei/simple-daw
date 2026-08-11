use std::ffi::{CStr, CString};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use clack_extensions::audio_ports::{AudioPortInfoBuffer, PluginAudioPorts};
use clack_extensions::gui::{
    GuiApiType, GuiConfiguration, GuiSize, HostGui, HostGuiImpl, PluginGui,
};
use clack_extensions::params::{ParamInfoBuffer, PluginParams};
use clack_host::events::event_types::ParamValueEvent;
use clack_host::prelude::*;
use clack_host::utils::Cookie;

use crate::builtin_fx::BuiltInEffect;

/// A CLAP parameter's static metadata, as declared by the plugin at load time (id, display name,
/// and its plain-value range). Doesn't change over the plugin's lifetime in this MVP — there's no
/// support for a plugin dynamically adding/removing parameters (`ParamRescanFlags::ALL`).
#[derive(Clone, Debug)]
pub struct PluginParamInfo {
    pub id: ClapId,
    pub name: String,
    pub min_value: f64,
    pub max_value: f64,
    pub default_value: f64,
}

/// A loaded, activated effect plugin, plus the channel counts it actually declared for its main
/// input/output ports (queried via the CLAP `audio-ports` extension — assuming stereo
/// unconditionally caused real plugins to misbehave, since not every effect is 2-in/2-out), and
/// its parameters. Used both for the single master-bus effect and per-track insert effects — the
/// only difference is how many of them exist and where their output gets summed.
pub struct LoadedEffect {
    pub processor: PluginAudioProcessor<DawHost>,
    pub input_channels: u32,
    pub output_channels: u32,
    pub params: Vec<PluginParamInfo>,
    /// Current value for each entry in `params` (same index). This app is the only writer of a
    /// plugin's parameters (no plugin GUI, no host-side automation), so tracking values here
    /// rather than re-querying the plugin every frame is safe and avoids a main-thread round trip
    /// from the audio callback.
    current_values: Vec<f64>,
    /// Set by `set_param`, cleared by `take_param_events` once the pending change(s) have been
    /// handed to the plugin.
    dirty: bool,
}

impl LoadedEffect {
    /// Sets parameter `index`'s value (clamped to its declared range) and marks it for delivery
    /// to the plugin on the next processed audio block.
    pub fn set_param(&mut self, index: usize, value: f64) {
        let Some(info) = self.params.get(index) else {
            return;
        };
        let clamped = value.clamp(info.min_value, info.max_value);
        if let Some(slot) = self.current_values.get_mut(index) {
            *slot = clamped;
            self.dirty = true;
        }
    }

    /// Current value of parameter `index`, or `None` if out of range.
    pub fn param_value(&self, index: usize) -> Option<f64> {
        self.current_values.get(index).copied()
    }

    /// Sets a parameter by its stable CLAP id rather than index — used to re-apply saved values
    /// after loading a plugin fresh (see `Song::save_to_file`/`load_from_file` and
    /// `Song::master_effect_params`/`TrackEffectConfig::params`). A no-op if the plugin has no
    /// parameter with that id.
    pub fn set_param_by_id(&mut self, id: u32, value: f64) {
        if let Some(index) = self.params.iter().position(|p| p.id.get() == id) {
            self.set_param(index, value);
        }
    }

    /// Snapshots every parameter's current value as (CLAP id, value) pairs, for persisting to a
    /// song file. Using ids rather than indices means values still land correctly even if a
    /// plugin update reorders its declared parameter list.
    pub fn param_snapshot(&self) -> Vec<(u32, f64)> {
        self.params
            .iter()
            .zip(&self.current_values)
            .map(|(info, value)| (info.id.get(), *value))
            .collect()
    }

    /// If any parameter changed since the last call, builds an event buffer with one
    /// `ParamValueEvent` per parameter's current value and clears the dirty flag; the caller
    /// feeds this to the plugin's next `process()` call. UI-thread edits are delivered this way
    /// rather than through the (main-thread-only) `flush` mechanism, since this plugin is always
    /// active once loaded — see `process_effect`.
    fn take_param_events(&mut self) -> Option<EventBuffer> {
        if !self.dirty {
            return None;
        }
        self.dirty = false;
        let mut buffer = EventBuffer::new();
        for (info, value) in self.params.iter().zip(&self.current_values) {
            let event =
                ParamValueEvent::new(0, info.id, Pckn::match_all(), *value, Cookie::empty());
            buffer.push(&event);
        }
        Some(buffer)
    }
}

/// Host-side bookkeeping for one plugin's floating GUI window. Returned by `load_and_activate`
/// alongside its `PluginInstance`/`LoadedEffect`; callers (`main.rs`) keep it next to the
/// `PluginInstance`, since opening/closing the GUI needs that instance's main-thread handle just
/// like `query_channel_count`/`query_params` do at load time.
pub struct PluginGuiHandle {
    /// `None` if the plugin doesn't implement the `gui` extension at all.
    gui: Option<PluginGui>,
    closed_by_plugin: Arc<AtomicBool>,
    is_open: bool,
}

impl PluginGuiHandle {
    /// Whether the loaded plugin implements the CLAP `gui` extension at all.
    pub fn is_supported(&self) -> bool {
        self.gui.is_some()
    }

    /// Whether this plugin's floating GUI window is currently open.
    pub fn is_open(&self) -> bool {
        self.is_open
    }
}

/// Shared between the UI thread (which loads/activates plugins) and the real-time audio callback
/// (which owns processing from then on). `None` means no master effect is loaded; the mix passes
/// through unchanged.
pub type MasterEffectSlot = Arc<Mutex<Option<LoadedEffect>>>;

/// One effect in a track's chain: either a hosted CLAP plugin, or one of the app's built-in DSP
/// effects (`builtin_fx::BuiltInEffect`) that need no external plugin file. The chain processes
/// both kinds interchangeably (see `process_effect_chain`), so a track can freely mix CLAP
/// plugins and built-ins in any order.
pub enum EffectInstance {
    Clap(LoadedEffect),
    BuiltIn(BuiltInEffect),
}

/// One effect chain per track, indexed the same as `Song::tracks`; each chain is indexed the same
/// as the UI's per-track list of effect slots, so a slot with no plugin loaded yet is `None`
/// rather than absent (built-in effects, needing no async load step, only ever occupy a slot as
/// `Some` — see `main.rs`'s "+ Add Effect" menu). Kept as a single `Mutex`-wrapped `Vec` (rather
/// than one `Arc<Mutex<_>>` per track) so the audio callback only takes one lock per buffer to
/// reach every track's chain, consistent with how `Song` itself is shared.
pub type TrackEffectSlots = Arc<Mutex<Vec<Vec<Option<EffectInstance>>>>>;

/// An empty `MasterEffectSlot` (no master-bus effect loaded).
pub fn new_master_effect_slot() -> MasterEffectSlot {
    Arc::new(Mutex::new(None))
}

/// A `TrackEffectSlots` with `track_count` empty per-track chains.
pub fn new_track_effect_slots(track_count: usize) -> TrackEffectSlots {
    // `LoadedEffect` isn't `Clone` (it owns a live plugin instance), so `vec![Vec::new(); n]`
    // isn't available here even though every element starts out empty.
    Arc::new(Mutex::new((0..track_count).map(|_| Vec::new()).collect()))
}

/// Minimal CLAP host: floating plugin GUI windows only (no embedding), no host-side automation,
/// no restart/reactivation support. Good enough to host effect plugins running with parameter
/// values we set directly via `set_param`. See the phase 7 plan for the deliberate scope cuts.
pub struct DawHost;

impl HostHandlers for DawHost {
    type Shared<'a> = DawHostShared;
    type MainThread<'a> = ();
    type AudioProcessor<'a> = ();

    fn declare_extensions(builder: &mut HostExtensions<Self>, _shared: &Self::Shared<'_>) {
        builder.register::<HostGui>();
    }
}

pub struct DawHostShared {
    /// Set by `closed` (which the plugin may call from a thread other than ours) when its
    /// floating GUI window is closed; polled and cleared on the UI thread by
    /// `plugin_gui_poll_closed`.
    gui_closed: Arc<AtomicBool>,
}

impl<'a> SharedHandler<'a> for DawHostShared {
    fn request_restart(&self) {
        // Not supported: we never reactivate a running plugin in this MVP.
    }

    fn request_process(&self) {
        // Not supported: our audio thread already drives processing continuously.
    }

    fn request_callback(&self) {
        // Not supported: plugins that need a main-thread callback to make
        // progress may not function correctly. Acceptable for this MVP's
        // effects-only scope (no host-side automation).
    }
}

/// Host side of the `gui` extension. Only floating windows are supported (see `open_plugin_gui`),
/// so the host never manages the window's size or visibility itself — those requests are
/// declined, matching a plugin-owned floating window's semantics.
impl HostGuiImpl for DawHostShared {
    fn resize_hints_changed(&self) {}

    fn request_resize(&self, _new_size: GuiSize) -> Result<(), HostError> {
        Err(HostError::Message(
            "resize requests aren't supported for floating plugin windows",
        ))
    }

    fn request_show(&self) -> Result<(), HostError> {
        Err(HostError::Message(
            "host-initiated show isn't supported for floating plugin windows",
        ))
    }

    fn request_hide(&self) -> Result<(), HostError> {
        Err(HostError::Message(
            "host-initiated hide isn't supported for floating plugin windows",
        ))
    }

    fn closed(&self, _was_destroyed: bool) {
        self.gui_closed.store(true, Ordering::Release);
    }
}

/// Queries the plugin's main audio port for its channel count, via the
/// `audio-ports` extension. Falls back to stereo (2) if the plugin doesn't
/// implement the extension, or has no ports in that direction.
fn query_channel_count(instance: &mut PluginInstance<DawHost>, is_input: bool) -> u32 {
    let mut handle = instance.plugin_handle();
    let Some(audio_ports) = handle.get_extension::<PluginAudioPorts>() else {
        return 2;
    };

    if audio_ports.count(&mut handle, is_input) == 0 {
        return if is_input { 0 } else { 2 };
    }

    let mut buffer = AudioPortInfoBuffer::new();
    audio_ports
        .get(&mut handle, 0, is_input, &mut buffer)
        .map(|info| info.channel_count)
        .unwrap_or(2)
}

/// Enumerates the plugin's parameters (if it implements the `params` extension), for display in
/// a generic parameter-editing panel. Returns an empty list for plugins with no parameters or no
/// `params` support.
fn query_params(instance: &mut PluginInstance<DawHost>) -> Vec<PluginParamInfo> {
    let mut handle = instance.plugin_handle();
    let Some(params_ext) = handle.get_extension::<PluginParams>() else {
        return Vec::new();
    };

    let count = params_ext.count(&mut handle);
    let mut buffer = ParamInfoBuffer::new();
    let mut infos = Vec::with_capacity(count as usize);
    for index in 0..count {
        if let Some(info) = params_ext.get_info(&mut handle, index, &mut buffer) {
            let name = String::from_utf8_lossy(info.name)
                .trim_end_matches('\0')
                .to_string();
            infos.push(PluginParamInfo {
                id: info.id,
                name,
                min_value: info.min_value,
                max_value: info.max_value,
                default_value: info.default_value,
            });
        }
    }
    infos
}

/// Queries whether the plugin implements the `gui` extension, for `PluginGuiHandle`.
fn query_gui_extension(instance: &mut PluginInstance<DawHost>) -> Option<PluginGui> {
    instance.plugin_handle().get_extension::<PluginGui>()
}

/// Loads a CLAP plugin bundle, instantiates its first plugin, and activates it
/// for audio processing. The returned `PluginInstance` must be kept alive for
/// as long as the returned `PluginAudioProcessor` is in use (it transitively
/// keeps the dynamic library loaded).
pub fn load_and_activate(
    path: &Path,
    sample_rate: f64,
    min_frames: u32,
    max_frames: u32,
) -> Result<(PluginInstance<DawHost>, LoadedEffect, PluginGuiHandle)> {
    // SAFETY: loading a `.clap` bundle runs arbitrary native code, same as
    // loading any other native plugin format (VST, LV2, ...). This is
    // inherent to hosting plugins at all, not specific to this call site.
    let entry = unsafe { PluginEntry::load(path) }
        .with_context(|| format!("failed to load CLAP bundle: {}", path.display()))?;

    let plugin_factory = entry
        .get_plugin_factory()
        .context("CLAP bundle has no plugin factory")?;

    let descriptor = plugin_factory
        .plugin_descriptors()
        .next()
        .context("CLAP bundle contains no plugins")?;
    let plugin_id: &CStr = descriptor.id().context("plugin descriptor has no id")?;

    let host_info = HostInfo::new(
        "simple-daw",
        "simple-daw",
        "https://example.invalid",
        "0.1.0",
    )
    .context("failed to build CLAP host info")?;

    let gui_closed = Arc::new(AtomicBool::new(false));
    let gui_closed_for_shared = gui_closed.clone();
    let mut instance = PluginInstance::<DawHost>::new(
        move |_| DawHostShared {
            gui_closed: gui_closed_for_shared,
        },
        |_| (),
        &entry,
        plugin_id,
        &host_info,
    )
    .context("failed to instantiate CLAP plugin")?;

    let input_channels = query_channel_count(&mut instance, true).clamp(1, 2);
    let output_channels = query_channel_count(&mut instance, false).clamp(1, 2);
    let params = query_params(&mut instance);
    let current_values = params.iter().map(|p| p.default_value).collect();
    let gui = query_gui_extension(&mut instance);

    let min_frames = min_frames.max(1);
    let configuration = PluginAudioConfiguration {
        sample_rate,
        min_frames_count: min_frames,
        max_frames_count: max_frames.max(min_frames),
    };

    let processor = instance
        .activate(|_, _| (), configuration)
        .context("failed to activate CLAP plugin")?;

    Ok((
        instance,
        LoadedEffect {
            processor: processor.into(),
            input_channels,
            output_channels,
            params,
            current_values,
            dirty: false,
        },
        PluginGuiHandle {
            gui,
            closed_by_plugin: gui_closed,
            is_open: false,
        },
    ))
}

/// Negotiates and opens `instance`'s floating GUI window (see the `gui` extension's negotiation
/// steps: `is_api_supported` → `create` → `suggest_title` → `show`). Errors if the plugin has no
/// `gui` extension, doesn't support a floating window on this platform, or the plugin itself
/// rejects one of these steps.
pub fn open_plugin_gui(
    instance: &mut PluginInstance<DawHost>,
    handle: &mut PluginGuiHandle,
    title: &str,
) -> Result<()> {
    let gui = handle.gui.context("plugin has no GUI")?;
    let mut plugin = instance.plugin_handle();

    let api_type = GuiApiType::default_for_current_platform()
        .context("no default floating GUI API for this platform")?;
    let configuration = GuiConfiguration {
        api_type,
        is_floating: true,
    };
    if !gui.is_api_supported(&mut plugin, configuration) {
        anyhow::bail!("plugin does not support a floating {api_type:?} window");
    }

    gui.create(&mut plugin, configuration)
        .context("failed to create plugin GUI")?;

    if let Ok(title) = CString::new(title) {
        gui.suggest_title(&mut plugin, &title);
    }

    if let Err(err) = gui.show(&mut plugin) {
        gui.destroy(&mut plugin);
        return Err(anyhow::Error::new(err).context("failed to show plugin GUI"));
    }

    handle.closed_by_plugin.store(false, Ordering::Release);
    handle.is_open = true;
    Ok(())
}

/// Hides and destroys `instance`'s floating GUI window. A no-op if none is open
/// (`PluginGuiHandle::is_open` is `false`).
pub fn close_plugin_gui(instance: &mut PluginInstance<DawHost>, handle: &mut PluginGuiHandle) {
    let (Some(gui), true) = (handle.gui, handle.is_open) else {
        return;
    };
    let mut plugin = instance.plugin_handle();
    let _ = gui.hide(&mut plugin);
    gui.destroy(&mut plugin);
    handle.is_open = false;
}

/// Polls whether the plugin closed its own floating window since the last call (e.g. the user hit
/// its close button), clearing the flag. If so, destroys the GUI's resources — required by CLAP
/// once the plugin reports the window closed — and updates `handle`. Returns whether the window
/// was just closed, so callers know to update any "GUI open" UI state.
pub fn plugin_gui_poll_closed(
    instance: &mut PluginInstance<DawHost>,
    handle: &mut PluginGuiHandle,
) -> bool {
    if !handle.closed_by_plugin.swap(false, Ordering::AcqRel) {
        return false;
    }
    if let Some(gui) = handle.gui {
        gui.destroy(&mut instance.plugin_handle());
    }
    handle.is_open = false;
    true
}

/// Reusable scratch buffers for one effect's audio-port plumbing, so `process_effect` doesn't
/// allocate on every call. Callers keep one `EffectScratch` per concurrently-processed effect
/// (the master bus, and one per track that has an effect loaded).
pub struct EffectScratch {
    in_l: Vec<f32>,
    in_r: Vec<f32>,
    out_l: Vec<f32>,
    out_r: Vec<f32>,
    input_ports: AudioPorts,
    output_ports: AudioPorts,
}

impl EffectScratch {
    /// Empty scratch buffers, ready for `reserve` and then `process_effect`.
    pub fn new() -> Self {
        Self {
            in_l: Vec::new(),
            in_r: Vec::new(),
            out_l: Vec::new(),
            out_r: Vec::new(),
            input_ports: AudioPorts::with_capacity(2, 1),
            output_ports: AudioPorts::with_capacity(2, 1),
        }
    }

    /// Pre-allocates the per-block buffers to `frames` capacity so the first real-time callback
    /// that actually processes audio through this effect doesn't have to grow them itself.
    pub fn reserve(&mut self, frames: usize) {
        self.in_l.reserve(frames);
        self.in_r.reserve(frames);
        self.out_l.reserve(frames);
        self.out_r.reserve(frames);
    }
}

/// Runs `effect` over a mono `input` buffer for one audio block, writing stereo results into
/// `out_l`/`out_r` (mono output is duplicated to both channels, matching how a mono-out plugin's
/// single channel is treated as center-panned). Also delivers any pending `set_param` changes as
/// part of this block's `process()` call. Returns `false` (leaving `out_l`/`out_r` untouched) if
/// the plugin isn't ready to process, so callers can fall back to the dry signal.
pub fn process_effect(
    effect: &mut LoadedEffect,
    input: &[f32],
    out_l: &mut [f32],
    out_r: &mut [f32],
    scratch: &mut EffectScratch,
) -> bool {
    let frames = input.len();
    scratch.in_l.resize(frames, 0.0);
    scratch.in_r.resize(frames, 0.0);
    scratch.in_l.copy_from_slice(input);
    if effect.input_channels > 1 {
        scratch.in_r.copy_from_slice(input);
    }
    scratch.out_l.resize(frames, 0.0);
    scratch.out_r.resize(frames, 0.0);

    let param_events = effect.take_param_events();
    let input_events = match &param_events {
        Some(buffer) => buffer.as_input(),
        None => InputEvents::empty(),
    };

    let processor = &mut effect.processor;
    let started = if processor.is_started() {
        processor.as_started_mut().ok()
    } else {
        processor.start_processing().ok()
    };
    let Some(started) = started else {
        return false;
    };

    let mut input_channels = vec![InputChannel::variable(&mut scratch.in_l)];
    if effect.input_channels > 1 {
        input_channels.push(InputChannel::variable(&mut scratch.in_r));
    }
    let input_audio = scratch.input_ports.with_input_buffers([AudioPortBuffer {
        latency: 0,
        channels: AudioPortBufferType::f32_input_only(input_channels.into_iter()),
    }]);

    let mut output_channels = vec![scratch.out_l.as_mut_slice()];
    if effect.output_channels > 1 {
        output_channels.push(scratch.out_r.as_mut_slice());
    }
    let mut output_audio = scratch.output_ports.with_output_buffers([AudioPortBuffer {
        latency: 0,
        channels: AudioPortBufferType::f32_output_only(output_channels.into_iter()),
    }]);

    let ok = started
        .process(
            &input_audio,
            &mut output_audio,
            &input_events,
            &mut OutputEvents::void(),
            None,
            None,
        )
        .is_ok();

    if ok {
        out_l.copy_from_slice(&scratch.out_l);
        if effect.output_channels > 1 {
            out_r.copy_from_slice(&scratch.out_r);
        } else {
            out_r.copy_from_slice(&scratch.out_l);
        }
    }
    ok
}

/// Runs `input` through each effect in `chain` in order (element 0 first, mixing CLAP plugins and
/// built-in DSP effects freely), downmixing one stage's stereo output to mono before feeding it
/// to the next stage — the same L/R-average downmix callers already use to sum a track's
/// post-effect signal into the master bus. A `None` slot (no CLAP plugin loaded there yet, or one
/// not ready to process — built-in effects never end up `None` once added, since creating one
/// can't fail or need an async load step) passes its input through unchanged, matching
/// `process_effect`'s own not-ready fallback.
///
/// `out_l`/`out_r` always end up holding the full chain's result (starting from `input`
/// duplicated to both channels if nothing in the chain processes). Returns whether at least one
/// stage actually processed, so callers can cheaply skip the mixed-down output and add the dry
/// signal straight in when a track's chain is empty or every CLAP plugin in it is unloaded.
pub fn process_effect_chain(
    chain: &mut [Option<EffectInstance>],
    input: &[f32],
    out_l: &mut [f32],
    out_r: &mut [f32],
    scratch: &mut [EffectScratch],
    mono_buf: &mut Vec<f32>,
) -> bool {
    let frames = input.len();
    mono_buf.resize(frames, 0.0);
    mono_buf.copy_from_slice(input);
    out_l.copy_from_slice(input);
    out_r.copy_from_slice(input);

    let mut any_used = false;
    for (slot, effect_scratch) in chain.iter_mut().zip(scratch.iter_mut()) {
        match slot {
            Some(EffectInstance::Clap(effect)) => {
                if process_effect(effect, mono_buf, out_l, out_r, effect_scratch) {
                    any_used = true;
                    for i in 0..frames {
                        mono_buf[i] = 0.5 * (out_l[i] + out_r[i]);
                    }
                }
            }
            Some(EffectInstance::BuiltIn(effect)) => {
                effect.process(mono_buf);
                out_l.copy_from_slice(mono_buf);
                out_r.copy_from_slice(mono_buf);
                any_used = true;
            }
            None => {}
        }
    }
    any_used
}
