//! macOS/Cocoa backend for `gui_embed` — the only platform actually run/verified (see the module
//! doc comment on `gui_embed` itself).

use anyhow::{Context, Result};
use clack_extensions::gui::Window as ClapWindow;
use objc2::rc::Retained;
use objc2_app_kit::NSView;
use objc2_foundation::{MainThreadMarker, NSPoint, NSRect, NSSize};
use raw_window_handle::RawWindowHandle;

/// A plain `NSView` added as a subview of the app's own content view, plus a strong reference to
/// that parent (queried on every `set_frame` call for its current size and flip state, both of
/// which can change if the app window resizes or moves between screens).
pub struct Container {
    view: Retained<NSView>,
    parent: Retained<NSView>,
}

impl Container {
    pub fn create(parent: RawWindowHandle, _scale_factor: f64) -> Result<Self> {
        let RawWindowHandle::AppKit(handle) = parent else {
            anyhow::bail!("expected an AppKit window handle on macOS");
        };
        let mtm = MainThreadMarker::new()
            .context("embedded CLAP GUIs must be opened from the main thread")?;

        // SAFETY: `handle.ns_view` is guaranteed valid for the lifetime of the `RawWindowHandle`
        // borrow by `HasWindowHandle`'s contract (see `eframe::Frame::window_handle`, our only
        // caller). `retain` takes our own strong reference so we can keep querying it after this
        // call returns; that's sound as long as the app's own window outlives this `Container`,
        // which holds — a `Container` only ever lives as long as the "FX Params" editor window
        // that opened it, itself long gone before the app's main window closes.
        let parent_view: Retained<NSView> =
            unsafe { Retained::retain(handle.ns_view.as_ptr().cast()) }
                .context("parent NSView pointer was null")?;

        // SAFETY: `mtm` proves we're on the main thread, `NSView::new`'s only precondition.
        let view = unsafe { NSView::new(mtm) };
        // SAFETY: `view` and `parent_view` are both valid, live Objective-C objects.
        unsafe { parent_view.addSubview(&view) };

        Ok(Self {
            view,
            parent: parent_view,
        })
    }

    pub fn set_frame(&mut self, x: f64, y: f64, width: f64, height: f64) {
        // AppKit's default view coordinate system has its origin at the bottom-left with y
        // increasing upward, the opposite of egui's top-left/y-down; a flipped view matches egui
        // directly. Checking `isFlipped` at each call (rather than assuming one or the other)
        // means this is correct regardless of how winit's own content view is configured. Cocoa
        // uses logical points throughout, the same unit egui already gives us, so — unlike
        // Win32/X11 — no DPI scaling math is needed here.
        let parent_height = self.parent.bounds().size.height;
        let origin_y = if self.parent.isFlipped() {
            y
        } else {
            parent_height - y - height
        };
        let frame = NSRect {
            origin: NSPoint { x, y: origin_y },
            size: NSSize { width, height },
        };
        // SAFETY: `self.view` is a valid, live Objective-C object.
        unsafe { self.view.setFrame(frame) };
    }

    pub fn clap_window(&self) -> ClapWindow<'static> {
        let ptr = Retained::as_ptr(&self.view) as *mut std::ffi::c_void;
        ClapWindow::from_cocoa_nsview(ptr)
    }
}

impl Drop for Container {
    fn drop(&mut self) {
        // SAFETY: `self.view` is a valid, live Objective-C object; removing a view from its
        // superview is always safe to call, even if it's already been removed.
        unsafe { self.view.removeFromSuperview() };
    }
}
