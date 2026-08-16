//! X11 backend for `gui_embed` — **unverified**: this project only has an
//! `aarch64-apple-darwin` Rust target installed, so this module has never been compiled, let
//! alone run against a real embedding plugin. Written from the CLAP spec and `x11rb`'s own
//! source; treat it as a first draft until someone builds and tests it on Linux/X11. See the
//! module doc comment on `gui_embed` for the wider context. No Wayland support — CLAP's `gui`
//! extension itself has none (`GuiApiType::supports_embedding` is `false` only for Wayland); the
//! existing floating-window path already covers that case.

use anyhow::{Context, Result};
use clack_extensions::gui::Window as ClapWindow;
use raw_window_handle::RawWindowHandle;
use x11rb::connection::Connection;
use x11rb::protocol::xproto::{
    configure_window, create_window, destroy_window, map_window, ConfigureWindowAux,
    CreateWindowAux, WindowClass,
};
use x11rb::rust_connection::RustConnection;

/// A plain child window under the embedding target's own top-level X window, positioned/sized in
/// the parent's own coordinate space. Opens its own XCB connection — winit's own connection isn't
/// exposed to embedder code, and a second connection to the same X server for this purpose is a
/// well-established pattern among embedding hosts.
pub struct Container {
    conn: RustConnection,
    window: u32,
    /// X11 windows are sized/positioned in physical pixels, unlike egui's logical points, so
    /// `set_frame`'s input needs converting every call. Fixed at creation time — sufficient here
    /// since nothing in this app currently reacts to a live DPI change while a plugin GUI is open.
    scale_factor: f64,
}

impl Container {
    pub fn create(parent: RawWindowHandle, scale_factor: f64) -> Result<Self> {
        let RawWindowHandle::Xlib(handle) = parent else {
            anyhow::bail!("expected an Xlib window handle on X11");
        };
        let parent_window = handle.window as u32;

        let (conn, _screen_num) =
            x11rb::connect(None).context("failed to open a second X11 connection")?;
        let window = conn
            .generate_id()
            .context("failed to allocate an X11 window id")?;

        // SAFETY: none — X11 requests here are all safe Rust calls. `COPY_DEPTH_FROM_PARENT`/
        // `COPY_FROM_PARENT` for depth and visual means the child always matches whatever
        // depth/visual `parent_window` itself uses, avoiding a `BadMatch` from a mismatched
        // visual on `INPUT_OUTPUT` windows.
        create_window(
            &conn,
            x11rb::COPY_DEPTH_FROM_PARENT,
            window,
            parent_window,
            0,
            0,
            1,
            1,
            0,
            WindowClass::COPY_FROM_PARENT,
            x11rb::COPY_FROM_PARENT,
            &CreateWindowAux::new(),
        )
        .context("failed to create X11 container window")?;
        map_window(&conn, window).context("failed to map X11 container window")?;
        conn.flush().context("failed to flush X11 connection")?;

        Ok(Self {
            conn,
            window,
            scale_factor,
        })
    }

    pub fn set_frame(&mut self, x: f64, y: f64, width: f64, height: f64) {
        let to_px = |v: f64| (v * self.scale_factor).round() as i32;
        let aux = ConfigureWindowAux::new()
            .x(to_px(x))
            .y(to_px(y))
            .width(to_px(width).max(1) as u32)
            .height(to_px(height).max(1) as u32);
        if configure_window(&self.conn, self.window, &aux).is_ok() {
            let _ = self.conn.flush();
        }
    }

    pub fn clap_window(&self) -> ClapWindow<'static> {
        ClapWindow::from_x11_handle(self.window as std::ffi::c_ulong)
    }
}

impl Drop for Container {
    fn drop(&mut self) {
        if destroy_window(&self.conn, self.window).is_ok() {
            let _ = self.conn.flush();
        }
    }
}
