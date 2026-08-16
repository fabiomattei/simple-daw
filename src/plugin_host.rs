use std::ffi::{CStr, CString};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use clack_extensions::audio_ports::{AudioPortInfoBuffer, PluginAudioPorts};
use clack_extensions::gui::{
    GuiApiType, GuiConfiguration, GuiSize, HostGui, HostGuiImpl, PluginGui,
};
use clack_extensions::latency::PluginLatency;
use clack_extensions::params::{ParamInfoBuffer, PluginParams};
use clack_host::events::event_types::ParamValueEvent;
use clack_host::prelude::*;
use clack_host::utils::Cookie;
use raw_window_handle::RawWindowHandle;

use crate::builtin_fx::BuiltInEffect;
use crate::gui_embed::EmbeddedContainer;
use crate::model::TrackEffectConfig;

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
    /// Whether the plugin declared a second input port (the CLAP convention for a sidechain key
    /// input — see the `audio-ports` extension's port-count query, distinct from `input_channels`
    /// which only describes the *main* port's channel count). When true, `process_effect` wires a
    /// caller-supplied key signal into that second port; when false, a supplied key signal is
    /// simply ignored, since the plugin never declared anywhere to put it.
    pub has_sidechain_input: bool,
    /// This plugin's self-reported processing latency in samples, queried once at load time via
    /// the CLAP `latency` extension (0 if the plugin doesn't implement it). Never re-queried after
    /// load — this host doesn't implement `HostLatencyImpl`'s "latency changed" callback, matching
    /// `DawHost`'s existing "no host-side automation, no restart/reactivation" MVP scope. Used for
    /// automatic plugin delay compensation — see `chain_latency_samples`.
    pub latency_samples: u32,
    /// Index into `Song::tracks` of this plugin's sidechain key source, mirroring
    /// `TrackEffectConfig::Clap::sidechain_source` — set from that field when the plugin is
    /// loaded, edited live from the FX chain UI same as any other parameter, and snapshotted back
    /// into `TrackEffectConfig` at save time. `None` means no sidechain routing.
    pub sidechain_source: Option<usize>,
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
    /// `TrackEffectConfig::Clap`'s `params`, the shape `Song::master_effects` and
    /// `Track::effects` both use). A no-op if the plugin has no parameter with that id.
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

/// Host-side bookkeeping for one plugin's GUI — either floating (a separate OS window, entirely
/// plugin-managed) or embedded (drawn into a host-owned container view docked inside our own
/// window, see `gui_embed`). Returned by `load_and_activate` alongside its
/// `PluginInstance`/`LoadedEffect`; callers (`main.rs`) keep it next to the `PluginInstance`,
/// since opening/closing the GUI needs that instance's main-thread handle just like
/// `query_channel_count`/`query_params` do at load time.
pub struct PluginGuiHandle {
    /// `None` if the plugin doesn't implement the `gui` extension at all.
    gui: Option<PluginGui>,
    closed_by_plugin: Arc<AtomicBool>,
    is_open: bool,
    /// `Some` while an embedded GUI is open; `None` for a floating GUI (or none at all). Dropping
    /// this tears the container down, so it's always cleared alongside `is_open` going false.
    embedded: Option<EmbeddedContainer>,
    /// The size negotiated with the plugin when the currently-open embedded GUI was opened (see
    /// `open_plugin_gui`'s returned `GuiSize`) — callers need this every frame, not just at open
    /// time, to keep reserving the same amount of space for it. `None` whenever `embedded` is.
    embedded_size: Option<GuiSize>,
}

impl PluginGuiHandle {
    /// Whether the loaded plugin implements the CLAP `gui` extension at all.
    pub fn is_supported(&self) -> bool {
        self.gui.is_some()
    }

    /// Whether this plugin's GUI (floating or embedded) is currently open.
    pub fn is_open(&self) -> bool {
        self.is_open
    }

    /// Whether the currently-open GUI is embedded (as opposed to floating). `false` if no GUI is
    /// open at all.
    pub fn is_embedded(&self) -> bool {
        self.embedded.is_some()
    }

    /// The size negotiated for the currently-open embedded GUI, if any — the amount of space a
    /// caller should reserve for it every frame (see `resize_embedded_plugin_gui`).
    pub fn embedded_size(&self) -> Option<GuiSize> {
        self.embedded_size
    }
}

/// One effect in a track's chain: either a hosted CLAP plugin, or one of the app's built-in DSP
/// effects (`builtin_fx::BuiltInEffect`) that need no external plugin file. The chain processes
/// both kinds interchangeably (see `process_effect_chain`), so a track can freely mix CLAP
/// plugins and built-ins in any order.
pub enum EffectInstance {
    Clap(LoadedEffect),
    BuiltIn(BuiltInEffect),
}

impl EffectInstance {
    /// This slot's configured sidechain key source (a `Song::tracks` index) — see
    /// `LoadedEffect::sidechain_source`/`BuiltInEffect::sidechain_source`. `None` for a built-in
    /// effect kind that has nowhere to use a key signal.
    pub fn sidechain_source(&self) -> Option<usize> {
        match self {
            EffectInstance::Clap(effect) => effect.sidechain_source,
            EffectInstance::BuiltIn(effect) => effect.sidechain_source(),
        }
    }

    /// Mutable access to `sidechain_source`, for the FX chain UI's sidechain-source picker.
    /// `None` for a built-in effect kind that doesn't carry a `sidechain_source` field at all.
    pub fn sidechain_source_mut(&mut self) -> Option<&mut Option<usize>> {
        match self {
            EffectInstance::Clap(effect) => Some(&mut effect.sidechain_source),
            EffectInstance::BuiltIn(effect) => effect.sidechain_source_mut(),
        }
    }
}

/// Total latency (in samples) a chain adds to whatever signal passes through it — the sum of
/// every loaded CLAP stage's own `LoadedEffect::latency_samples`. A `None` slot (nothing loaded
/// yet) and every built-in effect (all real-time, zero-lookahead algorithms in this codebase)
/// contribute 0. Used for automatic plugin delay compensation, computed fresh each buffer/chunk
/// from whichever chains are currently loaded — see `audio::pdc_delay_samples_for_track`.
pub fn chain_latency_samples(chain: &[Option<EffectInstance>]) -> u32 {
    chain
        .iter()
        .map(|slot| match slot {
            Some(EffectInstance::Clap(effect)) => effect.latency_samples,
            _ => 0,
        })
        .sum()
}

/// One effect chain per track, indexed the same as `Song::tracks`; each chain is indexed the same
/// as the UI's per-track list of effect slots, so a slot with no plugin loaded yet is `None`
/// rather than absent (built-in effects, needing no async load step, only ever occupy a slot as
/// `Some` — see `main.rs`'s "+ Add Effect" menu). Kept as a single `Mutex`-wrapped `Vec` (rather
/// than one `Arc<Mutex<_>>` per track) so the audio callback only takes one lock per buffer to
/// reach every track's chain, consistent with how `Song` itself is shared.
pub type TrackEffectSlots = Arc<Mutex<Vec<Vec<Option<EffectInstance>>>>>;

/// The master bus's own effect chain — structurally identical to `TrackEffectSlots`, just always
/// exactly one "track" long (index 0 is the whole chain; there's no second row). Reusing the same
/// type means the master chain processes through the exact same `process_effect_chain` call, and
/// its "+ Add Effect" UI reuses the exact same per-track chain UI, as any real track's chain does.
pub type MasterEffectSlots = TrackEffectSlots;

/// An empty `MasterEffectSlots` (no master-bus effects loaded).
pub fn new_master_effect_slots() -> MasterEffectSlots {
    new_track_effect_slots(1)
}

/// One effect chain per send bus, indexed the same as `Song::sends` — structurally identical to
/// `TrackEffectSlots` (one row per send, resized in lockstep with `Song::sends` the same way
/// `TrackEffectSlots` resizes with `Song::tracks`), and built with the same `new_track_effect_slots`
/// constructor.
pub type SendEffectSlots = TrackEffectSlots;

/// One effect chain per submix bus, indexed the same as `Song::submixes` — structurally identical
/// to `TrackEffectSlots`/`SendEffectSlots`, same reasoning.
pub type SubmixEffectSlots = TrackEffectSlots;

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

/// Whether the plugin declared a second input port, the CLAP convention for an optional sidechain
/// key input. Plugins without the `audio-ports` extension, or with only one input port, report
/// `false` — the common case for the overwhelming majority of effects.
fn has_sidechain_input_port(instance: &mut PluginInstance<DawHost>) -> bool {
    let mut handle = instance.plugin_handle();
    let Some(audio_ports) = handle.get_extension::<PluginAudioPorts>() else {
        return false;
    };
    audio_ports.count(&mut handle, true) >= 2
}

/// Queries the plugin's self-reported processing latency (CLAP `latency` extension), 0 if it
/// doesn't implement the extension.
fn query_latency(instance: &mut PluginInstance<DawHost>) -> u32 {
    let mut handle = instance.plugin_handle();
    let Some(latency) = handle.get_extension::<PluginLatency>() else {
        return 0;
    };
    latency.get(&mut handle)
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
    let has_sidechain_input = has_sidechain_input_port(&mut instance);
    let latency_samples = query_latency(&mut instance);
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
            has_sidechain_input,
            latency_samples,
            sidechain_source: None,
            params,
            current_values,
            dirty: false,
        },
        PluginGuiHandle {
            gui,
            closed_by_plugin: gui_closed,
            is_open: false,
            embedded: None,
            embedded_size: None,
        },
    ))
}

/// A temporary effect chain loaded fresh for one-shot offline rendering (`audio::render_song_to_wav`)
/// — distinct from `TrackEffectSlots` and friends, the *live* chains loaded once via the UI's
/// "Load" button and reused every real-time callback for as long as a plugin stays loaded. Every
/// loaded CLAP plugin's `PluginInstance` must be kept alive exactly as long as its `LoadedEffect`
/// (see `LoadedEffect`'s doc comment) — `_instances` exists purely to hold that alive via RAII,
/// dropped together with `chain` when this whole value drops at the end of the bounce. Never read
/// directly; a bounce has no GUI to open and no need to touch a `PluginInstance` beyond keeping it
/// alive.
pub struct OfflineEffectChain {
    pub chain: Vec<Option<EffectInstance>>,
    _instances: Vec<Option<PluginInstance<DawHost>>>,
}

/// Loads `specs` into a fresh `OfflineEffectChain` at `sample_rate`/`block_size` — which may differ
/// from any live session's, since a bounce picks its own sample rate. Activates each CLAP plugin
/// with `min_frames_count = 1` regardless of `block_size` (unlike a live chain, which activates at
/// the audio device's own reported minimum) — the offline mixdown's automation sub-chunking
/// (`process_chain_with_automation`) can hand a plugin fewer frames than `block_size` at an
/// automation breakpoint, and this keeps every call within the plugin's declared range. A CLAP
/// plugin that fails to load (missing file, incompatible bundle, ...) leaves that slot `None`,
/// silently skipped, the same tolerance a live chain already has for a not-yet-loaded slot — one
/// broken effect shouldn't fail the whole bounce.
pub fn load_offline_chain(specs: &[TrackEffectConfig], sample_rate: f64, block_size: u32) -> OfflineEffectChain {
    let mut chain = Vec::with_capacity(specs.len());
    let mut instances = Vec::with_capacity(specs.len());
    for spec in specs {
        match spec {
            TrackEffectConfig::Clap { path, params, sidechain_source } => {
                if path.trim().is_empty() {
                    chain.push(None);
                    instances.push(None);
                    continue;
                }
                match load_and_activate(Path::new(path.trim()), sample_rate, 1, block_size) {
                    Ok((instance, mut effect, _gui)) => {
                        for (id, value) in params {
                            effect.set_param_by_id(*id, *value);
                        }
                        effect.sidechain_source = *sidechain_source;
                        chain.push(Some(EffectInstance::Clap(effect)));
                        instances.push(Some(instance));
                    }
                    Err(_) => {
                        chain.push(None);
                        instances.push(None);
                    }
                }
            }
            builtin_spec => {
                let effect = BuiltInEffect::from_config(builtin_spec, sample_rate as f32);
                chain.push(effect.map(EffectInstance::BuiltIn));
                instances.push(None);
            }
        }
    }
    OfflineEffectChain { chain, _instances: instances }
}

/// Negotiates and opens `instance`'s GUI, preferring an embedded view docked in the host's own
/// window over a floating OS window when possible.
///
/// If `embed_target` is `Some((parent_window, scale_factor))` and the plugin's `is_api_supported`
/// accepts an embedded configuration for this platform's default `GuiApiType`, follows CLAP's
/// embedded negotiation (`create` → `set_scale` → `get_size` → build a `gui_embed::EmbeddedContainer`
/// sized to the plugin's own preferred size → `set_parent` → `show`) and returns `Ok(Some(size))`
/// so the caller knows how much space to reserve for it. Otherwise falls back to the floating-window
/// negotiation (`is_api_supported` → `create` → `suggest_title` → `show`) and returns `Ok(None)`.
///
/// Errors if the plugin has no `gui` extension at all, or if whichever path was attempted fails —
/// this does not silently fall back from a failed embedded attempt to floating, since a plugin
/// that claimed to support embedding but then failed partway through is a real problem worth
/// surfacing, not papering over.
pub fn open_plugin_gui(
    instance: &mut PluginInstance<DawHost>,
    handle: &mut PluginGuiHandle,
    title: &str,
    embed_target: Option<(RawWindowHandle, f64)>,
) -> Result<Option<GuiSize>> {
    let gui = handle.gui.context("plugin has no GUI")?;
    let mut plugin = instance.plugin_handle();

    if let Some((parent, scale_factor)) = embed_target
        && let Some(api_type) = GuiApiType::default_for_current_platform()
    {
        let embed_configuration = GuiConfiguration {
            api_type,
            is_floating: false,
        };
        if gui.is_api_supported(&mut plugin, embed_configuration) {
            let (container, size) =
                open_embedded_gui(&gui, &mut plugin, api_type, parent, scale_factor)?;
            handle.embedded = Some(container);
            handle.embedded_size = Some(size);
            handle.closed_by_plugin.store(false, Ordering::Release);
            handle.is_open = true;
            return Ok(Some(size));
        }
    }

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
    Ok(None)
}

/// The embedded half of `open_plugin_gui`'s negotiation, split out to keep that function's two
/// paths (embedded vs. floating) readable. On any failure after `create` succeeds, calls
/// `destroy` before returning the error so the plugin's GUI resources aren't left half-allocated.
fn open_embedded_gui(
    gui: &PluginGui,
    plugin: &mut PluginMainThreadHandle,
    api_type: GuiApiType,
    parent: RawWindowHandle,
    scale_factor: f64,
) -> Result<(EmbeddedContainer, GuiSize)> {
    let configuration = GuiConfiguration {
        api_type,
        is_floating: false,
    };
    gui.create(plugin, configuration)
        .context("failed to create plugin GUI")?;

    // Cocoa uses logical points and handles its own scaling; `set_scale` should not be called
    // for it (see `GuiApiType::COCOA`'s doc comment). Not fatal if a Win32/X11 plugin declines
    // it — the container itself still applies `scale_factor` to its own sizing regardless.
    if api_type != GuiApiType::COCOA {
        let _ = gui.set_scale(plugin, scale_factor);
    }

    let size = gui.get_size(plugin).unwrap_or(GuiSize {
        width: 480,
        height: 320,
    });

    let mut container = match EmbeddedContainer::create(parent, scale_factor) {
        Ok(container) => container,
        Err(err) => {
            gui.destroy(plugin);
            return Err(err);
        }
    };
    container.set_frame(0.0, 0.0, size.width as f64, size.height as f64);

    // SAFETY: `container` is kept alive in `PluginGuiHandle::embedded` for as long as this GUI
    // stays open, and is only dropped (tearing down and removing the native view/window) after
    // `destroy` has been called for it, in `close_plugin_gui`/`plugin_gui_poll_closed` — matching
    // `set_parent`'s requirement that the parent stay valid until then.
    if let Err(err) = unsafe { gui.set_parent(plugin, container.clap_window()) } {
        gui.destroy(plugin);
        return Err(anyhow::Error::new(err).context("failed to set plugin GUI parent"));
    }

    if let Err(err) = gui.show(plugin) {
        gui.destroy(plugin);
        return Err(anyhow::Error::new(err).context("failed to show plugin GUI"));
    }

    Ok((container, size))
}

/// Repositions/resizes the currently-open embedded GUI's container to `x`/`y`/`width`/`height`
/// (top-left-origin egui logical points, relative to the parent window's content area) — called
/// every frame while the reserved panel for it is on screen. A no-op if no GUI is open, or the
/// open one is floating rather than embedded.
pub fn resize_embedded_plugin_gui(handle: &mut PluginGuiHandle, x: f64, y: f64, width: f64, height: f64) {
    if let Some(container) = handle.embedded.as_mut() {
        container.set_frame(x, y, width, height);
    }
}

/// Hides and destroys `instance`'s GUI (floating or embedded). A no-op if none is open
/// (`PluginGuiHandle::is_open` is `false`).
pub fn close_plugin_gui(instance: &mut PluginInstance<DawHost>, handle: &mut PluginGuiHandle) {
    let (Some(gui), true) = (handle.gui, handle.is_open) else {
        return;
    };
    let mut plugin = instance.plugin_handle();
    let _ = gui.hide(&mut plugin);
    gui.destroy(&mut plugin);
    handle.embedded = None;
    handle.embedded_size = None;
    handle.is_open = false;
}

/// Polls whether the plugin closed its own floating window since the last call (e.g. the user hit
/// its close button), clearing the flag. If so, destroys the GUI's resources — required by CLAP
/// once the plugin reports the window closed — and updates `handle`. Returns whether the window
/// was just closed, so callers know to update any "GUI open" UI state. Embedded GUIs don't self-
/// close this way in practice (there's no close button on a bare container view), but the same
/// bookkeeping applies if a plugin ever reports it regardless.
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
    handle.embedded = None;
    handle.embedded_size = None;
    handle.is_open = false;
    true
}

/// Reusable scratch buffers for one effect's audio-port plumbing, so `process_effect` doesn't
/// allocate on every call. Callers keep one `EffectScratch` per concurrently-processed effect
/// (the master bus, and one per track that has an effect loaded).
pub struct EffectScratch {
    in_l: Vec<f32>,
    in_r: Vec<f32>,
    /// Sidechain key signal, copied in only when the effect declared a second input port and a
    /// caller supplied a key signal — see `process_effect`.
    sc_l: Vec<f32>,
    sc_r: Vec<f32>,
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
            sc_l: Vec::new(),
            sc_r: Vec::new(),
            out_l: Vec::new(),
            out_r: Vec::new(),
            // 2 ports of capacity: the main input port, plus an optional sidechain port.
            input_ports: AudioPorts::with_capacity(4, 2),
            output_ports: AudioPorts::with_capacity(2, 1),
        }
    }

    /// Pre-allocates the per-block buffers to `frames` capacity so the first real-time callback
    /// that actually processes audio through this effect doesn't have to grow them itself.
    pub fn reserve(&mut self, frames: usize) {
        self.in_l.reserve(frames);
        self.in_r.reserve(frames);
        self.sc_l.reserve(frames);
        self.sc_r.reserve(frames);
        self.out_l.reserve(frames);
        self.out_r.reserve(frames);
    }
}

/// Runs `effect` over one stereo audio block (`input_l`/`input_r`) for one audio block, writing
/// stereo results into `out_l`/`out_r`. A plugin that only declares one input channel gets a mono
/// downmix (`(l+r)/2`) as its single input — that's the plugin's own declared I/O shape, not a
/// choice this function makes about the chain; a mono-output plugin's single channel is likewise
/// duplicated to both output channels, treated as center-panned. Also delivers any pending
/// `set_param` changes as part of this block's `process()` call. `sidechain`, when supplied and
/// the plugin declared a second input port (`LoadedEffect::has_sidechain_input`), feeds that key
/// signal into the plugin's sidechain port; ignored otherwise (a supplied key signal for a plugin
/// with nowhere to put it is silently dropped, same tolerance already applied to a redirected
/// automation target on a track/slot that doesn't exist). Returns `false` (leaving `out_l`/`out_r`
/// untouched) if the plugin isn't ready to process, so callers can fall back to the dry signal.
pub fn process_effect(
    effect: &mut LoadedEffect,
    input_l: &[f32],
    input_r: &[f32],
    out_l: &mut [f32],
    out_r: &mut [f32],
    scratch: &mut EffectScratch,
    sidechain: Option<(&[f32], &[f32])>,
) -> bool {
    let frames = input_l.len();
    scratch.in_l.resize(frames, 0.0);
    scratch.in_r.resize(frames, 0.0);
    if effect.input_channels > 1 {
        scratch.in_l.copy_from_slice(input_l);
        scratch.in_r.copy_from_slice(input_r);
    } else {
        for i in 0..frames {
            scratch.in_l[i] = 0.5 * (input_l[i] + input_r[i]);
        }
    }
    let use_sidechain = effect.has_sidechain_input && sidechain.is_some();
    if let Some((key_l, key_r)) = sidechain.filter(|_| use_sidechain) {
        scratch.sc_l.resize(frames, 0.0);
        scratch.sc_r.resize(frames, 0.0);
        scratch.sc_l.copy_from_slice(key_l);
        scratch.sc_r.copy_from_slice(key_r);
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
    let mut input_ports = vec![AudioPortBuffer {
        latency: 0,
        channels: AudioPortBufferType::f32_input_only(input_channels.into_iter()),
    }];
    if use_sidechain {
        let sidechain_channels =
            vec![InputChannel::variable(&mut scratch.sc_l), InputChannel::variable(&mut scratch.sc_r)];
        input_ports.push(AudioPortBuffer {
            latency: 0,
            channels: AudioPortBufferType::f32_input_only(sidechain_channels.into_iter()),
        });
    }
    let input_audio = scratch.input_ports.with_input_buffers(input_ports);

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

/// Runs `input_l`/`input_r` through each effect in `chain` in order (element 0 first, mixing CLAP
/// plugins and built-in DSP effects freely), carrying the real stereo signal from one stage to the
/// next — a stage only loses width if it itself declares mono I/O (see `process_effect`'s own
/// downmix-on-mono-input behavior), never because the chain machinery flattens it between stages.
/// A `None` slot (no CLAP plugin loaded there yet, or one not ready to process — built-in effects
/// never end up `None` once added, since creating one can't fail or need an async load step)
/// passes its input through unchanged, matching `process_effect`'s own not-ready fallback.
///
/// `out_l`/`out_r` always end up holding the full chain's result (starting from `input_l`/
/// `input_r` unchanged if nothing in the chain processes). `run_l`/`run_r` are reusable scratch
/// for the signal in flight between stages — needed because a CLAP stage's `process_effect` call
/// can't read and write the same buffer. Returns whether at least one stage actually processed, so
/// callers can cheaply skip the processed output and add the dry signal straight in when a
/// track's chain is empty or every CLAP plugin in it is unloaded.
///
/// `all_track_dry` is every track's own pre-effects, pre-volume/pan synthesized signal for this
/// same block, indexed the same as `Song::tracks` (so `all_track_dry[i]` is track `i`'s own dry
/// signal, regardless of which chain — a track's, a send's, a submix's, or the master's — is
/// being processed here). A slot whose own `sidechain_source` (`EffectInstance::sidechain_source`)
/// names a valid track index gets that track's dry signal as its key input; `None`, or an index
/// past the end of `all_track_dry`, means no sidechain for that slot. Only `EffectInstance::Clap`
/// (when the plugin itself declared a sidechain port) and the built-in Compressor/NoiseGate honor
/// a resolved key signal; every other slot kind ignores its own `sidechain_source`.
#[allow(clippy::too_many_arguments)]
pub fn process_effect_chain(
    chain: &mut [Option<EffectInstance>],
    input_l: &[f32],
    input_r: &[f32],
    out_l: &mut [f32],
    out_r: &mut [f32],
    scratch: &mut [EffectScratch],
    run_l: &mut Vec<f32>,
    run_r: &mut Vec<f32>,
    all_track_dry: &[(&[f32], &[f32])],
) -> bool {
    let frames = input_l.len();
    run_l.resize(frames, 0.0);
    run_r.resize(frames, 0.0);
    run_l.copy_from_slice(input_l);
    run_r.copy_from_slice(input_r);
    out_l.copy_from_slice(input_l);
    out_r.copy_from_slice(input_r);

    let mut any_used = false;
    for (slot, effect_scratch) in chain.iter_mut().zip(scratch.iter_mut()) {
        let sidechain = slot
            .as_ref()
            .and_then(EffectInstance::sidechain_source)
            .and_then(|index| all_track_dry.get(index).copied());
        match slot {
            Some(EffectInstance::Clap(effect)) => {
                if process_effect(effect, run_l, run_r, out_l, out_r, effect_scratch, sidechain) {
                    any_used = true;
                    run_l.copy_from_slice(out_l);
                    run_r.copy_from_slice(out_r);
                }
            }
            Some(EffectInstance::BuiltIn(effect)) => {
                effect.process_with_sidechain(run_l, run_r, sidechain);
                out_l.copy_from_slice(run_l);
                out_r.copy_from_slice(run_r);
                any_used = true;
            }
            None => {}
        }
    }
    any_used
}
