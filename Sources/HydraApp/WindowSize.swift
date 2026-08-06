import AppKit
import SwiftUI

/// Gives the window a sensible starting size, once.
///
/// `Scene.defaultSize` is supposed to handle this, but has no effect here: the window opens
/// at its minimum width and the full height of the screen. Rather than imposing size
/// constraints on the view — which put the window's constraints in conflict with the
/// columns' and made AppKit give up — we set the size on the window itself, constraining
/// nothing.
///
/// Applies only when opening a window that has no saved geometry: as soon as macOS has
/// remembered one, that is the user's choice and it wins.
struct InitialWindowSize: NSViewRepresentable {
    let width: CGFloat
    let height: CGFloat

    func makeNSView(context: Context) -> NSView {
        let view = NSView(frame: .zero)
        DispatchQueue.main.async { apply(to: view.window) }
        return view
    }

    func updateNSView(_ nsView: NSView, context: Context) {}

    private func apply(to window: NSWindow?) {
        guard let window, let screen = window.screen ?? NSScreen.main else { return }
        guard !context(window) else { return }

        let visible = screen.visibleFrame
        let size = CGSize(
            width: min(width, visible.width), height: min(height, visible.height))
        let origin = CGPoint(
            x: visible.midX - size.width / 2, y: visible.midY - size.height / 2)
        window.setFrame(CGRect(origin: origin, size: size), display: true)
    }

    /// True if macOS has already restored a geometry for this window: its height then
    /// differs from what the minimum constraints alone would impose.
    private func context(_ window: NSWindow) -> Bool {
        window.frameAutosaveName.isEmpty == false
            && UserDefaults.standard.string(forKey: "NSWindow Frame " + window.frameAutosaveName)
                != nil
    }
}

extension View {
    func initialWindowSize(width: CGFloat, height: CGFloat) -> some View {
        background(InitialWindowSize(width: width, height: height))
    }
}
