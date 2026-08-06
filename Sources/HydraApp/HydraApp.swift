import AppKit
import HydraMetal
import SwiftUI

@main
struct HydraApp: App {
    @State private var model = AppModel()

    init() {
        // `--self-test` vérifie, sans interface, que le paquet est complet : les sources
        // Metal sont des ressources SwiftPM, et si elles n'accompagnent pas l'exécutable
        // l'application se lance mais échoue au premier chargement de modèle. Mieux vaut
        // le savoir à la construction qu'après un téléchargement de soixante gigaoctets.
        if CommandLine.arguments.contains("--self-test") {
            HydraApp.runSelfTest()
        }
        // `--ui-smoke` ouvre la fenêtre, la met au premier plan et rend la main. Une
        // interface peut se compiler, s'ouvrir, et mourir d'une boucle de contraintes
        // AppKit dès qu'elle passe au premier plan — c'est arrivé, et rien dans les tests
        // ne l'aurait vu. Le plantage est une exception non rattrapée : il suffit de
        // laisser vivre la fenêtre quelques secondes pour qu'il se manifeste.
        if CommandLine.arguments.contains("--ui-smoke") {
            DispatchQueue.main.asyncAfter(deadline: .now() + 1) {
                NSApp.activate(ignoringOtherApps: true)
                NSApp.windows.first(where: \.isVisible)?.makeKeyAndOrderFront(nil)
            }
            DispatchQueue.main.asyncAfter(deadline: .now() + 4) {
                func report(_ line: String) {
                    FileHandle.standardError.write(Data((line + "\n").utf8))
                }
                report("  \(NSApp.windows.count) fenêtre(s)")
                for window in NSApp.windows where window.contentView != nil {
                    report(String(
                        format: "  fenêtre %.0f×%.0f · min %.0f×%.0f · écran %.0f",
                        window.frame.width, window.frame.height,
                        window.contentMinSize.width, window.contentMinSize.height,
                        NSScreen.main?.visibleFrame.height ?? 0))
                    func walk(_ view: NSView, _ depth: Int) {
                        if let split = view as? NSSplitView {
                            for pane in split.arrangedSubviews {
                                report(String(format: "    colonne %.0f×%.0f",
                                              pane.frame.width, pane.frame.height))
                            }
                        }
                        if depth < 8 { view.subviews.forEach { walk($0, depth + 1) } }
                    }
                    walk(window.contentView!, 0)
                }
                exit(0)
            }
        }
        // `--smoke-test` exerce le chemin réel de l'application — moteur, tokeniseur,
        // format Harmony, génération — sans interface. Un autotest qui ne fait que
        // compiler les noyaux ne prouverait pas que l'application sait répondre.
        if let index = CommandLine.arguments.firstIndex(of: "--smoke-test") {
            let prompt = CommandLine.arguments.count > index + 1
                ? CommandLine.arguments[index + 1] : "Dis bonjour en trois mots."
            HydraApp.runSmokeTest(prompt: prompt)
        }
    }

    static func runSelfTest() -> Never {
        var failures: [String] = []
        do {
            let context = try MetalContext()
            print("✔ Metal : \(context.device.name), famille \(context.gpuFamily)")
            for kernel in [
                "mxfp4_gemv_vectorized", "bf16_gemv", "bf16_gemm_tiled", "mxfp4_gemm_tiled",
                "attention_decode", "attention_prefill", "rms_norm", "swiglu", "router_topk",
            ] {
                _ = try context.pipeline(kernel)
            }
            print("✔ noyaux Metal compilés depuis le paquet")
        } catch {
            failures.append("Metal : \(error)")
        }

        do {
            let directory = try ModelLocations.directory()
            print("✔ répertoire des modèles : \(directory.path)")
            for entry in CatalogEntry.all {
                print("  \(entry.displayName) : \(ModelLocations.state(of: entry))")
            }
        } catch {
            failures.append("modèles : \(error)")
        }

        if failures.isEmpty {
            print("\nautotest réussi")
            exit(0)
        }
        for failure in failures { print("✘ \(failure)") }
        exit(1)
    }

    static func runSmokeTest(prompt: String) -> Never {
        guard let entry = CatalogEntry.all.first(where: {
            ModelLocations.state(of: $0).isInstalled
        }) else {
            print("✘ aucun modèle installé")
            exit(1)
        }
        print("modèle : \(entry.displayName)")

        let engine = InferenceEngine()
        let done = DispatchSemaphore(value: 0)
        nonisolated(unsafe) var failure: String?

        engine.load(entry: entry, contextLength: 4096, slotsPerLayer: 8) { step in
            print("  \(step)")
        } completion: { result in
            if case .failure(let error) = result { failure = error.description }
            done.signal()
        }
        done.wait()
        if let failure {
            print("✘ chargement : \(failure)")
            exit(1)
        }

        var settings = GenerationSettings()
        settings.maximumTokens = 64
        settings.reasoningEffort = "low"

        nonisolated(unsafe) var text = ""
        nonisolated(unsafe) var rate = 0.0
        nonisolated(unsafe) var ttft = 0.0
        engine.generate(turns: [.user(prompt)], settings: settings) { event in
            switch event {
            case .text(let fragment): text += fragment
            case .firstToken(let seconds): ttft = seconds
            case .started, .reasoning: break
            case .finished(let tokens, let seconds, _):
                rate = seconds > 0 ? Double(tokens) / seconds : 0
                done.signal()
            case .failed(let reason):
                failure = reason
                done.signal()
            }
        }
        done.wait()

        if let failure {
            print("✘ génération : \(failure)")
            exit(1)
        }
        print("\n\(text.trimmingCharacters(in: .whitespacesAndNewlines))\n")
        print(String(format: "✔ %.2f jetons/s · %.1f s avant réponse · empreinte %.0f Mio",
                     rate, ttft, Double(MemoryFootprint.current()) / 1_048_576))

        // Second tour : même conversation, une question de plus. C'est là que la
        // réutilisation du cache doit se voir.
        nonisolated(unsafe) var followUpTTFT = 0.0
        engine.generate(
            turns: [.user(prompt), .assistant(text), .user("Et pourquoi rouge au coucher ?")],
            settings: settings
        ) { event in
            switch event {
            case .firstToken(let seconds): followUpTTFT = seconds
            case .finished: done.signal()
            case .failed(let reason): failure = reason; done.signal()
            default: break
            }
        }
        done.wait()
        print(String(format: "✔ tour de suite : %.1f s avant réponse", followUpTTFT))
        exit(text.isEmpty ? 1 : 0)
    }

    var body: some Scene {
        WindowGroup {
            // Pas de `.frame(minWidth:)` ici : imposer une taille minimale à la racine
            // d'un NavigationSplitView met ses contraintes en conflit avec celles des
            // colonnes. La fenêtre s'ouvrait alors à 980 points de large alors que les
            // colonnes en réclamaient 1308, et AppKit relançait indéfiniment sa passe de
            // mise à jour des contraintes jusqu'à lever une exception. Chaque colonne
            // déclare son propre minimum, et la fenêtre en hérite.
            ContentView(model: model)
                .initialWindowSize(width: 1240, height: 820)
                .onDisappear { model.flush() }
        }
        .defaultSize(width: 1240, height: 800)
        .windowToolbarStyle(.unified)
        .commands {
            CommandGroup(after: .newItem) {
                Button("Nouvelle conversation") { model.newConversation() }
                    .keyboardShortcut("n", modifiers: .command)
            }
        }
    }
}
