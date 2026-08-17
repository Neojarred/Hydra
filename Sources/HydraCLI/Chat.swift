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
        var prefillChunk: Int?
        /// `nil` means "whatever the model publishes". A flag on the command line overrides it.
        var temperature: Float?
        var topKOverride: Int?
        var presencePenaltyOverride: Float?
        var seed: UInt64 = 0x5EED_1234
        var topP: Float?
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
            print("tokenizer missing, run first: hydra tokenizer")
            throw ExitError.planInvalid
        }

        var start = Date()
        let tokenizer = try TokenizerInstaller.load(
            from: root, architecture: model.architecture)
        let tokenizerTime = Date().timeIntervalSince(start)

        let context = try MetalContext()
        let profile = context.hardwareProfile(memoryBandwidth: 94e9, diskBandwidth: 5.5e9)
        let policy: ExpertCachePolicy = options.slotsPerLayer.map { .slotsPerLayer($0) } ?? .balanced
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
            expertCache: expertCache, contextLength: options.contextLength,
            prefillChunk: options.prefillChunk)
        // Bring the pages in with a sequential read rather than through scattered page
        // faults during the first pass.
        let warmStart = Date()
        mapping.prefault()
        let warmTime = Date().timeIntervalSince(warmStart)
        let loadTime = Date().timeIntervalSince(start)

        FileHandle.standardError.write(Data(
            ("\(model.name), \(budget.expertSlotsPerLayer)/\(model.expertCount) experts cached, "
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
            "prompt: \(promptTokens.count) tokens, prefill…\n".utf8))

        // --- Batched prefill ---
        //
        // The prompt's tokens are known in advance: nothing forces us to process them one
        // at a time. In batches the dense weights are read once for the whole batch instead
        // of once per token, the same computation, a different order.
        start = Date()
        ForwardEncoder.dispatchCounter.enabled = true
        ForwardEncoder.dispatchCounter.reset()
        var distribution = try runner.prefill(tokens: promptTokens)
        let prefillTime = Date().timeIntervalSince(start)
        let prefillDispatches = ForwardEncoder.dispatchCounter.snapshot()
        ForwardEncoder.dispatchCounter.enabled = false
        let prefillGPU = (runner as? Qwen35MoeRunner)?.lastGPUSeconds
            ?? (runner as? Gemma4ModelRunner)?.lastGPUSeconds
        let t = runner.lastTimings
        FileHandle.standardError.write(Data(
            String(format: "  %d tokens in %.1f s (%.0f tokens/s)\n"
                   + "  cb1 %.2f s · expert I/O %.2f s · experts %.2f s · head %.2f s\n"
                   + "  read from SSD: %.2f GiB\n\n",
                   promptTokens.count, prefillTime,
                   Double(promptTokens.count) / prefillTime,
                   t.attentionAndRouter, t.expertIO, t.mixture, t.head,
                   Double(expertCache.statisticsSnapshot().bytesRead) / 1_073_741_824).utf8))

        if let gpu = prefillGPU {
            let total = prefillDispatches.values.reduce(0, +)
            FileHandle.standardError.write(Data(
                String(format:
                    "  GPU busy %.1f s of %.1f s (%.0f %%) · %d dispatches "
                    + "(%.0f a token)\n",
                    gpu, prefillTime, gpu / prefillTime * 100, total,
                    Double(total) / Double(max(promptTokens.count, 1))).utf8))
            for (name, count) in prefillDispatches.sorted(by: { $0.value > $1.value }).prefix(5) {
                FileHandle.standardError.write(Data(
                    String(format: "    %-28s %7d\n",
                           (name as NSString).utf8String!, count).utf8))
            }
            FileHandle.standardError.write(Data("\n".utf8))
        }

        // The dispatch count for one decoded token, which is what ranks "fewer launches"
        // against every other candidate optimization.
        ForwardEncoder.dispatchCounter.enabled = true
        ForwardEncoder.dispatchCounter.reset()

        // --- Generation ---
        let parser = format.makeParser(tokenizer: tokenizer, settings: promptSettings)
        let sampling = ModelRunner.Sampling(
            temperature: options.temperature ?? model.samplingDefaults.temperature,
            topP: options.topP ?? model.samplingDefaults.topP,
            topK: options.topKOverride ?? model.samplingDefaults.topK,
            presencePenalty: options.presencePenaltyOverride
                ?? model.samplingDefaults.presencePenalty,
            seed: options.seed)

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
        let dispatches = ForwardEncoder.dispatchCounter.snapshot()
        ForwardEncoder.dispatchCounter.enabled = false
        if generated > 0 {
            let total = dispatches.values.reduce(0, +)
            FileHandle.standardError.write(Data(
                String(format: "\n  %d dispatches over %d tokens = %.0f a token\n",
                       total, generated, Double(total) / Double(generated)).utf8))
            let widths = ForwardEncoder.dispatchCounter.widths()
            FileHandle.standardError.write(Data(
                "    kernel                        count  a token  threadgroups\n".utf8))
            for (name, count) in dispatches.sorted(by: { $0.value > $1.value }).prefix(8) {
                FileHandle.standardError.write(Data(
                    String(format: "    %-28s %6d  %7.0f  %12.0f\n",
                           (name as NSString).utf8String!, count,
                           Double(count) / Double(generated), widths[name] ?? 0).utf8))
            }
        }

        // The last decode step's own breakdown. Prefill's is printed above and says nothing
        // about decoding, which is where a chat actually spends its time.
        //
        // "merged buffer" is the wait on the one buffer a layer, which holds the previous
        // layer's experts *and* this layer's mixer and router: both models merge them, so the
        // execution of the experts is in that figure and not in "encode", which is only the
        // CPU-side encoding of the next one.
        let d = runner.lastTimings
        // The one measurement that separates "the kernels are slow" from "the GPU is idle".
        // `expertIO` is nested inside `mixture` for Qwen, so the wall figure adds only the
        // buckets that do not overlap.
        let gpuBusy: Double? = (runner as? Gemma4ModelRunner)?.lastGPUSeconds
            ?? (runner as? Qwen35MoeRunner)?.lastGPUSeconds
        if let busy = gpuBusy {
            let wall = runner is Qwen35MoeRunner
                ? d.attentionAndRouter + d.mixture + d.head
                : d.attentionAndRouter + d.expertIO + d.mixture + d.head
            FileHandle.standardError.write(Data(
                String(format:
                    "\n  GPU busy %.1f ms of %.1f ms wall (%.0f %%), the rest is CPU\n",
                    busy * 1000, wall * 1000,
                    wall > 0 ? busy / wall * 100 : 0).utf8))
        }
        FileHandle.standardError.write(Data(
            String(format:
                "\n  decode step: merged buffer %.1f ms · expert I/O %.1f ms · encode %.1f ms · head %.1f ms"
                + "  (sum %.1f ms)\n",
                d.attentionAndRouter * 1000, d.expertIO * 1000, d.mixture * 1000,
                d.head * 1000,
                (d.attentionAndRouter + d.expertIO + d.mixture + d.head) * 1000).utf8))

        let stats = expertCache.statisticsSnapshot()
        FileHandle.standardError.write(Data(
            String(format:
                "\n\n%d tokens in %.1f s (%.2f tok/s), cache hits %.0f %%, footprint %@\n",
                generated, generationTime, Double(generated) / generationTime,
                stats.hitRate * 100, mib(MemoryFootprint.current())).utf8))

        if !options.showAnalysis && reasoningCharacters > 0 {
            let hidden = "  (\(reasoningCharacters) characters of reasoning "
                + "hidden, --analysis to see them)\n"
            FileHandle.standardError.write(Data(hidden.utf8))
        }
    }
}
