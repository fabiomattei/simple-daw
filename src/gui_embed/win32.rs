//! Win32 backend for `gui_embed` — **unverified**: this project only has an
//! `aarch64-apple-darwin` Rust target installed, so this module has never been compiled, let
//! alone run against a real embedding plugin. Written from the Win32 API and `windows` crate
//! docs/source; treat it as a first draft until someone builds and tests it on Windows. See the
//! module doc comment on `gui_embed` for the wider context.

use anyhow::{Context, Result};
use clack_extensions::gui::Window as ClapWindow;
use raw_window_handle::RawWindowHandle;
use windows::core::w;
use windows::Win32::Foundation::HWND;
use windows::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DestroyWindow, MoveWindow, WINDOW_EX_STYLE, WS_CHILD, WS_VISIBLE,
};

/// A child `HWND` of the built-in `"STATIC"` window class (no custom window procedure needed —
/// we never process messages for it ourselves), positioned/sized in the parent's own client area.
pub struct Container {
    hwnd: HWND,
    /// Win32 windows are sized/positioned in physical pixels, unlike egui's logical points, so
    /// `set_frame`'s input needs converting every call. Fixed at creation time — sufficient here
    /// since nothing in this app currently reacts to a live DPI change while a plugin GUI is open.
    scale_factor: f64,
}

impl Container {
    pub fn create(parent: RawWindowHandle, scale_factor: f64) -> Result<Self> {
        let RawWindowHandle::Win32(handle) = parent else {
            anyhow::bail!("expected a Win32 window handle on Windows");
        };
        let parent_hwnd = HWND(handle.hwnd.get() as *mut std::ffi::c_void);

        // SAFETY: `parent_hwnd` is guaranteed valid for the lifetime of the `RawWindowHandle`
        // borrow by `HasWindowHandle`'s contract (see `eframe::Frame::window_handle`, our only
        // caller). `"STATIC"` is a built-in system window class, always registered — no
        // `RegisterClassExW` call is needed. The child outlives this call via its own HWND, torn
        // down explicitly in `Drop`.
        let hwnd = unsafe {
            CreateWindowExW(
                WINDOW_EX_STYLE(0),
                w!("STATIC"),
                w!(""),
                WS_CHILD | WS_VISIBLE,
                0,
                0,
                0,
                0,
                Some(parent_hwnd),
                None,
                None,
                None,
            )
        }
        .context("failed to create Win32 container window")?;

        Ok(Self {
            hwnd,
            scale_factor,
        })
    }

    pub fn set_frame(&mut self, x: f64, y: f64, width: f64, height: f64) {
        let to_px = |v: f64| (v * self.scale_factor).round() as i32;
        // SAFETY: `self.hwnd` is a valid, live window.
        let _ = unsafe {
            MoveWindow(
                self.hwnd,
                to_px(x),
                to_px(y),
                to_px(width),
                to_px(height),
                true,
            )
        };
    }

    pub fn clap_window(&self) -> ClapWindow<'static> {
        ClapWindow::from_win32_hwnd(self.hwnd.0)
    }
}

impl Drop for Container {
    fn drop(&mut self) {
        // SAFETY: `self.hwnd` is a valid, live window; destroying it is always safe to call.
        let _ = unsafe { DestroyWindow(self.hwnd) };
    }
}
