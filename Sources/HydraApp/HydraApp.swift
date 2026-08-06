import AppKit
import HydraMetal
import SwiftUI

@main
struct HydraApp: App {
    @State private var model = AppModel()

    init() {
        // `--self-test` checks, without a UI, that the bundle is complete: the Metal sources
        // are SwiftPM resources, and if they do not accompany the executable the app launches
        // but fails on the first model load. Better to know that at build time than after a
        // sixty-gigabyte download.
        if CommandLine.arguments.contains("--self-test") {
            HydraApp.runSelfTest()
        }
        // `--ui-smoke` opens the window, brings it to the front and returns. A UI can
        // compile, open, and die of an AppKit constraint loop the moment it comes to the
        // front — that happened, and nothing in the tests would have caught it. The crash is
        // an uncaught exception: letting the window live a few seconds is enough for it to
        // show.
        if CommandLine.arguments.contains("--ui-smoke") {
            DispatchQueue.main.asyncAfter(deadline: .now() + 1) {
                NSApp.activate(ignoringOtherApps: true)
                NSApp.windows.first(where: \.isVisible)?.makeKeyAndOrderFront(nil)
            }
            DispatchQueue.main.asyncAfter(deadline: .now() + 4) {
                func report(_ line: String) {
                    FileHandle.standardError.write(Data((line + "\n").utf8))
                }
                report("  \(NSApp.windows.count) window(s)")
                for window in NSApp.windows where window.contentView != nil {
                    report(String(
                        format: "  window %.0f×%.0f · min %.0f×%.0f · screen %.0f",
                        window.frame.width, window.frame.height,
                        window.contentMinSize.width, window.contentMinSize.height,
                        NSScreen.main?.visibleFrame.height ?? 0))
                    func walk(_ view: NSView, _ depth: Int) {
                        if let split = view as? NSSplitView {
                            for pane in split.arrangedSubviews {
                                report(String(format: "    column %.0f×%.0f",
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
        // `--smoke-test` exercises the application's real path — engine, tokenizer, Harmony
        // format, generation — without a UI. A self-test that only compiles the kernels would
        // not prove the application can answer.
        if let index = CommandLine.arguments.firstIndex(of: "--smoke-test") {
            let prompt = CommandLine.arguments.count > index + 1
                ? CommandLine.arguments[index + 1] : "Say hello in three words."
            HydraApp.runSmokeTest(prompt: prompt)
        }
    }

    static func runSelfTest() -> Never {
        var failures: [String] = []
        do {
            let context = try MetalContext()
            print("✔ Metal: \(context.device.name), family \(context.gpuFamily)")
            for kernel in [
                "mxfp4_gemv_vectorized", "bf16_gemv", "bf16_gemm_tiled", "mxfp4_gemm_tiled",
                "attention_decode", "attention_prefill", "rms_norm", "swiglu", "router_topk",
            ] {
                _ = try context.pipeline(kernel)
            }
            print("✔ Metal kernels compiled from the bundle")
        } catch {
            failures.append("Metal: \(error)")
        }

        do {
            let directory = try ModelLocations.directory()
            print("✔ models directory: \(directory.path)")
            for entry in CatalogEntry.all {
                print("  \(entry.displayName) : \(ModelLocations.state(of: entry))")
            }
        } catch {
            failures.append("models: \(error)")
        }

        if failures.isEmpty {
            print("\nself-test passed")
            exit(0)
        }
        for failure in failures { print("✘ \(failure)") }
        exit(1)
    }

    static func runSmokeTest(prompt: String) -> Never {
        guard let entry = CatalogEntry.all.first(where: {
            ModelLocations.state(of: $0).isInstalled
        }) else {
            print("✘ no model installed")
            exit(1)
        }
        print("model: \(entry.displayName)")

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
            print("✘ load: \(failure)")
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
            print("✘ generation: \(failure)")
            exit(1)
        }
        print("\n\(text.trimmingCharacters(in: .whitespacesAndNewlines))\n")
        print(String(format: "✔ %.2f tok/s · %.1f s to first token · footprint %.0f MiB",
                     rate, ttft, Double(MemoryFootprint.current()) / 1_048_576))

        // Second turn: same conversation, one more question. This is where cache reuse
        // should show.
        nonisolated(unsafe) var followUpTTFT = 0.0
        engine.generate(
            turns: [.user(prompt), .assistant(text), .user("And why red at sunset?")],
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
        print(String(format: "✔ follow-up turn: %.1f s to first token", followUpTTFT))
        exit(text.isEmpty ? 1 : 0)
    }

    var body: some Scene {
        WindowGroup {
            // No `.frame(minWidth:)` here: imposing a minimum size on the root of a
            // NavigationSplitView puts its constraints in conflict with the columns'. The
            // window then opened 980 points wide while the columns demanded 1308, and AppKit
            // restarted its constraint-update pass indefinitely until it threw. Each column
            // declares its own minimum, and the window inherits it.
            ContentView(model: model)
                .initialWindowSize(width: 1240, height: 820)
                .onDisappear { model.flush() }
        }
        .defaultSize(width: 1240, height: 800)
        .windowToolbarStyle(.unified)
        .commands {
            CommandGroup(after: .newItem) {
                Button("New conversation") { model.newConversation() }
                    .keyboardShortcut("n", modifiers: .command)
            }
        }
    }
}
