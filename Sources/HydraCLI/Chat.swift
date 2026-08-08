import Foundation
import HydraCore
import HydraFormat
import HydraInstall
import HydraMetal
import HydraTokenize

/// A conversation with an installed model, in the Harmony format.
enum Chat {

    struct Options {
        var tokenCount = 512
        var contextLength = 4096
        var slotsPerLayer: Int?
        var temperature: Float = 1.0
        var topP: Float = 1.0
        var reasoning: Harmony.ReasoningEffort = .medium
        var showAnalysis = false
        var instructions: String?
    }

    static func run(config: GptOssConfig, root: URL, prompt: String, options: Options) throws {
        guard TokenizerInstaller.isInstalled(at: root) else {
            print("tokenizer missing — run first: hydra tokenizer")
            throw ExitError.planInvalid
        }

        var start = Date()
        let tokenizer = try TokenizerInstaller.load(from: root)
        let tokenizerTime = Date().timeIntervalSince(start)

        let context = try MetalContext()
        let profile = context.hardwareProfile(memoryBandwidth: 94e9, diskBandwidth: 5.5e9)
        let policy: ExpertCachePolicy = options.slotsPerLayer.map { .slotsPerLayer($0) } ?? .minimal
        let budget = MemoryBudget(
            config: config, hardware: profile,
            contextLength: options.contextLength, policy: policy)

        start = Date()
        let mapping = try ModelMapping(root: root, model: config, device: context.device)
        let expertCache = ExpertSlotCache(
root: root, model: config,
            slotsPerLayer: budget.expertSlotsPerLayer, device: context.device)
        let runner = try ModelRunner(
            config: config, context: context, mapping: mapping,
            expertCache: expertCache, contextLength: options.contextLength)
        // Bring the pages in with a sequential read rather than through scattered page
        // faults during the first pass.
        let warmStart = Date()
        mapping.prefault()
        let warmTime = Date().timeIntervalSince(warmStart)
        let loadTime = Date().timeIntervalSince(start)

        FileHandle.standardError.write(Data(
            ("\(config.name) — \(budget.expertSlotsPerLayer)/\(config.expertCount) experts cached, "
             + "expected footprint \(gib(budget.totalFootprintBytes))\n"
             + String(format: "tokenizer %.1f s, model %.1f s (of which %.1f s prefaulting)\n\n",
                       tokenizerTime, loadTime, warmTime)).utf8))

        // --- Harmony prompt ---
        let renderer = Harmony.Renderer(
            reasoningEffort: options.reasoning, instructions: options.instructions)
        let rendered = renderer.render(turns: [.user(prompt)])
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
        let parser = Harmony.Parser(tokenizer: tokenizer)
        var session = Harmony.Parser.Session()
        let sampling = ModelRunner.Sampling(
            temperature: options.temperature, topP: options.topP)

        var generated = 0
        start = Date()
        var analysisShown = false

        while generated < options.tokenCount && !session.isFinished {
            let token = runner.sample(from: distribution, using: sampling)
            generated += 1

            for event in parser.consume(token, session: &session) {
                switch event {
                case .text(let channel, let text):
                    switch channel {
                    case .final:
                        print(text, terminator: "")
                        fflush(stdout)
                    case .analysis where options.showAnalysis:
                        if !analysisShown {
                            FileHandle.standardError.write(Data("\u{1B}[2m[reasoning] ".utf8))
                            analysisShown = true
                        }
                        FileHandle.standardError.write(Data(text.utf8))
                    default:
                        break
                    }
                case .channelEnded(let channel):
                    if channel == .analysis, analysisShown {
                        FileHandle.standardError.write(Data("\u{1B}[0m\n\n".utf8))
                    }
                case .stopped:
                    break
                }
            }
            if session.isFinished { break }
            distribution = try runner.forward(token: token)
        }
        let generationTime = Date().timeIntervalSince(start)

        let stats = expertCache.statisticsSnapshot()
        FileHandle.standardError.write(Data(
            String(format:
                "\n\n— %d tokens in %.1f s (%.2f tok/s) — cache hits %.0f %% — footprint %@\n",
                generated, generationTime, Double(generated) / generationTime,
                stats.hitRate * 100, mib(MemoryFootprint.current())).utf8))

        if !options.showAnalysis && !session.analysisText.isEmpty {
            let hidden = "  (\(session.analysisText.count) characters of reasoning "
                + "hidden, --analysis to see them)\n"
            FileHandle.standardError.write(Data(hidden.utf8))
        }
    }
}
