import AppKit
import SwiftUI

/// Donne à la fenêtre une taille de départ raisonnable, une seule fois.
///
/// `Scene.defaultSize` est censé s'en charger, mais reste sans effet ici : la fenêtre
/// s'ouvre à sa largeur minimale et sur toute la hauteur de l'écran. Plutôt que d'imposer
/// des contraintes de taille à la vue — ce qui avait mis les contraintes de la fenêtre en
/// conflit avec celles des colonnes et faisait abandonner AppKit —, on pose la taille sur
/// la fenêtre elle-même, sans rien contraindre.
///
/// Ne s'applique qu'à l'ouverture d'une fenêtre encore vierge : dès que macOS a mémorisé
/// une géométrie, c'est le choix de l'utilisateur et il prime.
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

    /// Vrai si macOS a déjà restauré une géométrie pour cette fenêtre : la hauteur diffère
    /// alors de celle qu'imposent les contraintes minimales seules.
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
