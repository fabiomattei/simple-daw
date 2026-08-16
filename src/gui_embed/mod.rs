//! Native container-view creation for embedded CLAP plugin GUIs (see `plugin_host::open_plugin_gui`).
//!
//! CLAP's `gui` extension has no "position within parent" call — a host that wants a plugin's
//! embedded GUI to sit in a specific spot inside its own layout (rather than covering the whole
//! parent window) has to own a small positionable native container itself, hand *that* to the
//! plugin as its parent, and reposition/resize it every frame to track wherever the host's UI
//! decides it should go. One backend module per platform implements that container; this module
//! just picks the right one and exposes a single platform-agnostic type.
//!
//! Only macOS (`macos`) has been run and visually verified. `win32`/`x11` are written from the
//! Win32/Xlib/CLAP specs and their crates' own source, but this project only has a macOS target
//! installed, so they've never been compiled, let alone tested against a real embedding plugin —
//! treat them as unverified until someone checks them on those platforms.

use anyhow::Result;
use clack_extensions::gui::Window as ClapWindow;
use raw_window_handle::RawWindowHandle;

#[cfg(target_os = "macos")]
mod macos;
#[cfg(all(unix, not(target_os = "macos")))]
mod x11;
#[cfg(target_os = "windows")]
mod win32;

#[cfg(target_os = "macos")]
use macos::Container as PlatformContainer;
#[cfg(all(unix, not(target_os = "macos")))]
use x11::Container as PlatformContainer;
#[cfg(target_os = "windows")]
use win32::Container as PlatformContainer;

/// A host-owned native view/window, parented into the app's own window, that a CLAP plugin's
/// embedded GUI is in turn parented into. Dropping this tears the container down and removes it
/// from the app's window.
pub struct EmbeddedContainer(PlatformContainer);

impl EmbeddedContainer {
    /// Creates a new container parented into `parent`, the app's own window. `scale_factor` is
    /// the app window's current DPI scale factor — needed on the physical-pixel platforms
    /// (Win32/X11, see `GuiApiType::uses_logical_size`); ignored on macOS, which uses logical
    /// points throughout, same as egui.
    pub fn create(parent: RawWindowHandle, scale_factor: f64) -> Result<Self> {
        PlatformContainer::create(parent, scale_factor).map(Self)
    }

    /// Repositions/resizes the container. `x`/`y`/`width`/`height` are top-left-origin egui
    /// logical points, relative to the parent window's own content area; each backend converts to
    /// its own native coordinate convention and pixel unit internally.
    pub fn set_frame(&mut self, x: f64, y: f64, width: f64, height: f64) {
        self.0.set_frame(x, y, width, height);
    }

    /// The container's own handle, ready to hand to `PluginGui::set_parent` as the plugin's
    /// embedding target.
    pub fn clap_window(&self) -> ClapWindow<'static> {
        self.0.clap_window()
    }
}
