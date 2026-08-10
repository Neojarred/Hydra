import Foundation
import HydraCore
import HydraFormat
import HydraInstall
import HydraMetal
import HydraTokenize

/// A conversation with an installed model, in whichever format that model speaks.
enum Chat {

    struct Options {
        var tokenCount = 512
        var contextLength = 4096
        var slotsPerLayer: Int?
        var temperature: Float = 1.0
        var topP: Float = 1.0
        var reasoning: ReasoningLevel = .medium
        var showAnalysis = false
        var instructions: String?
        /// Print every generated token, id and piece, before the parser sees it.
        var dumpTokens = false
    }

    static func run(
        model: any ModelDescriptor, root: URL, prompt: String, options: Options
    ) throws {
        guard TokenizerInstaller.isInstalled(at: root) else {
            print("tokenizer missing — run first: hydra tokenizer")
            throw ExitError.planInvalid
        }

        var start = Date()
        let tokenizer = try TokenizerInstaller.load(
            from: root, architecture: model.architecture)
        let tokenizerTime = Date().timeIntervalSince(start)

        let context = try MetalContext()
        let profile = context.hardwareProfile(memoryBandwidth: 94e9, diskBandwidth: 5.5e9)
        let policy: ExpertCachePolicy = options.slotsPerLayer.map { .slotsPerLayer($0) } ?? .minimal
        let budget = MemoryBudget(
            config: model, hardware: profile,
            contextLength: options.contextLength, policy: policy)

        start = Date()
        let mapping = try ModelMapping(root: root, model: model, device: context.device)
        let expertCache = ExpertSlotCache(
            root: root, model: model,
            slotsPerLayer: budget.expertSlotsPerLayer, device: context.device)
        let runner = try ModelRuntime.makeRunner(
            model: model, context: context, mapping: mapping,
            expertCache: expertCache, contextLength: options.contextLength)
        // Bring the pages in with a sequential read rather than through scattered page
        // faults during the first pass.
        let warmStart = Date()
        mapping.prefault()
        let warmTime = Date().timeIntervalSince(warmStart)
        let loadTime = Date().timeIntervalSince(start)

        FileHandle.standardError.write(Data(
            ("\(model.name) — \(budget.expertSlotsPerLayer)/\(model.expertCount) experts cached, "
             + "expected footprint \(gib(budget.totalFootprintBytes))\n"
             + String(format: "tokenizer %.1f s, model %.1f s (of which %.1f s prefaulting)\n\n",
                       tokenizerTime, loadTime, warmTime)).utf8))

        // --- Prompt, in the loaded model's own format ---
        let format = ConversationFormats.format(for: runner.architecture)
        let promptSettings = PromptSettings(
            reasoning: options.reasoning, instructions: options.instructions)
        let rendered = format.render(turns: [.user(prompt)], settings: promptSettings)
        let promptTokens = tokenizer.encode(rendered, allowSpecial: true)

        FileHandle.standardError.write(Data(
            "prompt: \(promptTokens.count) tokens — prefill…\n".utf8))

        // --- Batched prefill ---
        //
        // The prompt's tokens are known in advance: nothing forces us to process them one
        // at a time. In batches the dense weights are read once for the whole batch instead
        // of once per token — the same computation, a different order.
        start = Date()
        var distribution = try runner.prefill(tokens: promptTokens)
        let prefillTime = Date().timeIntervalSince(start)
        let t = runner.lastTimings
        FileHandle.standardError.write(Data(
            String(format: "  %d tokens in %.1f s (%.0f tokens/s)\n"
                   + "  cb1 %.2f s · expert I/O %.2f s · experts %.2f s · head %.2f s\n"
                   + "  read from SSD: %.2f GiB\n\n",
                   promptTokens.count, prefillTime,
                   Double(promptTokens.count) / prefillTime,
                   t.attentionAndRouter, t.expertIO, t.mixture, t.head,
                   Double(expertCache.statisticsSnapshot().bytesRead) / 1_073_741_824).utf8))

        // --- Generation ---
        let parser = format.makeParser(tokenizer: tokenizer, settings: promptSettings)
        let sampling = ModelRunner.Sampling(
            temperature: options.temperature, topP: options.topP)

        var generated = 0
        start = Date()
        var analysisShown = false
        var reasoningCharacters = 0

        while generated < options.tokenCount && !parser.isFinished {
            let token = runner.sample(from: distribution, using: sampling)
            generated += 1

            if options.dumpTokens {
                let piece = tokenizer.name(of: token)
                    ?? String(decoding: tokenizer.bytes(for: token), as: UTF8.self)
                FileHandle.standardError.write(Data("[\(token) «\(piece)»] ".utf8))
            }

            for event in parser.consume(token) {
                switch event {
                case .answer(let text):
                    // The reasoning block closes when the answer starts, whichever format
                    // signalled the transition.
                    if analysisShown {
                        FileHandle.standardError.write(Data("\u{1B}[0m\n\n".utf8))
                        analysisShown = false
                    }
                    print(text, terminator: "")
                    fflush(stdout)
                case .reasoning(let text):
                    reasoningCharacters += text.count
                    guard options.showAnalysis else { break }
                    if !analysisShown {
                        FileHandle.standardError.write(Data("\u{1B}[2m[reasoning] ".utf8))
                        analysisShown = true
                    }
                    FileHandle.standardError.write(Data(text.utf8))
                case .stopped:
                    break
                }
            }
            if parser.isFinished { break }
            distribution = try runner.forward(token: token, needsLogits: true)
        }
        if analysisShown { FileHandle.standardError.write(Data("\u{1B}[0m\n\n".utf8)) }
        let generationTime = Date().timeIntervalSince(start)

        let stats = expertCache.statisticsSnapshot()
        FileHandle.standardError.write(Data(
            String(format:
                "\n\n— %d tokens in %.1f s (%.2f tok/s) — cache hits %.0f %% — footprint %@\n",
                generated, generationTime, Double(generated) / generationTime,
                stats.hitRate * 100, mib(MemoryFootprint.current())).utf8))

        if !options.showAnalysis && reasoningCharacters > 0 {
            let hidden = "  (\(reasoningCharacters) characters of reasoning "
                + "hidden, --analysis to see them)\n"
            FileHandle.standardError.write(Data(hidden.utf8))
        }
    }
}
