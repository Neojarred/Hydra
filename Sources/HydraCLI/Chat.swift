import Foundation
import HydraCore
import HydraFormat
import HydraInstall
import HydraMetal
import HydraSearch
import HydraVision
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
        /// How far back the presence penalty looks. Zero is the whole generation, which is the
        /// shipped default and what the model cards mean by `presence_penalty`.
        var repeatWindowOverride: Int?
        var frequencyPenaltyOverride: Float?
        var seed: UInt64 = 0x5EED_1234
        var images: [String] = []
        var topP: Float?
        var reasoning: ReasoningLevel = .medium
        var showAnalysis = false
        var instructions: String?
        /// Print every generated token, id and piece, before the parser sees it.
        var dumpTokens = false
        /// Declare the search tool and run the calls that come back.
        ///
        /// The same loop the application runs, on the command line, because the failure this
        /// feature had was behavioural — a model that deliberated instead of searching and then
        /// never answered — and that is not a thing a unit test sees. It needs the real
        /// checkpoint, the real prompt and a real query.
        var searches = false
        /// Replay a recorded search response instead of calling the endpoint.
        ///
        /// The query pass is skipped with it, deliberately: comparing two answer-phase prompts
        /// needs everything before the answer held fixed, and a model-written query is not
        /// fixed once the prompt it was written from has changed.
        var searchFrom: String?
    }

    static func run(
        model: any ModelDescriptor, root: URL, prompt: String, options: Options
    ) async throws {
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
        var promptSettings = PromptSettings(
            reasoning: options.reasoning, instructions: options.instructions)
        if options.searches || options.searchFrom != nil {
            guard format.supportsTools else {
                print("this model has no search dialect yet; use qwen-q4 or qwen-q8")
                throw ExitError.planInvalid
            }
            promptSettings.searching = true
        }
        var turn = ChatTurn.user(prompt)
        turn.images = options.images.count
        // Open when searching: the turn is continued twice from the same point.
        let rendered = (options.searches || options.searchFrom != nil)
            ? format.renderOpen(turns: [turn], settings: promptSettings)
            : format.render(turns: [turn], settings: promptSettings)
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
        var distribution: UnsafeBufferPointer<Float>
        if options.images.isEmpty {
            distribution = try runner.prefill(tokens: promptTokens)
        } else {
            // The end-to-end picture path, the same one the app takes: split the prompt at each
            // pad, run the tower, hand the runner text runs and embeddings in order.
            // Gemma has its own tower, its own placeholder and its own splice.
            if let gemma = runner as? Gemma4ModelRunner {
                var visionConfig = Gemma4VisionConfig.a4b
                // The checkpoint accepts (70, 140, 280, 560, 1120) and nothing else.
                if let requested = ProcessInfo.processInfo.environment["HYDRA_GEMMA_SOFT_TOKENS"],
                    let value = Int(requested) { visionConfig.softTokens = value }
                guard let pieces = Gemma4Prompt.split(
                    tokens: promptTokens, atPlaceholder: visionConfig.imageTokenID,
                    images: options.images.count)
                else {
                    print("the prompt's image placeholders do not match the images given")
                    throw ExitError.planInvalid
                }
                let mapping = try Gemma4VisionMapping(root: root, device: context.device)
                let tower = Gemma4VisionTower(
                    config: visionConfig, context: context, weights: mapping)
                let patcher = Gemma4ImagePatcher(config: visionConfig)

                var elements: [Gemma4ModelRunner.PromptElement] = []
                for piece in pieces {
                    switch piece {
                    case .text(let tokens):
                        elements.append(.text(tokens))
                    case .image(let index):
                        let started = Date()
                        let patched = try patcher.patch(
                            contentsOf: URL(fileURLWithPath: options.images[index]))
                        let embedded = try tower.forward(
                            patches: patched.values,
                            gridHeight: patched.gridHeight, gridWidth: patched.gridWidth)
                        FileHandle.standardError.write(Data(String(
                            format: "  image %d: %dx%d, %d tokens, tower %.1f s\n",
                            index + 1, patched.pixelWidth, patched.pixelHeight, patched.tokens,
                            Date().timeIntervalSince(started)).utf8))
                        FileHandle.standardError.write(Data(String(
                            format: "    a scaled text embedding: mean |x| %.4f\n",
                            gemma.scaledEmbeddingMagnitude(token: promptTokens.first ?? 1)).utf8))
                        let mean = embedded.reduce(0) { $0 + abs($1) } / Float(embedded.count)
                        let peak = embedded.map { abs($0) }.max() ?? 0
                        FileHandle.standardError.write(Data(String(
                            format: "    embeddings: %d rows, mean |x| %.4f, peak %.3f, finite %@\n",
                            patched.tokens, mean, peak,
                            embedded.allSatisfy { $0.isFinite } ? "yes" : "NO").utf8))
                        elements.append(.image(
                            embeddings: embedded, tokens: patched.tokens))
                    }
                }
                distribution = try gemma.prefill(elements: elements)
            } else {
            guard let qwen = runner as? Qwen35MoeRunner,
                let pieces = QwenFormat.split(
                    tokens: promptTokens, atImagePad: Qwen35VisionConfig.a3b.imageTokenID,
                    images: options.images.count)
            else {
                print("this model cannot read images")
                throw ExitError.planInvalid
            }
            let visionConfig = Qwen35VisionConfig.a3b
            let mapping = try VisionMapping(root: root, device: context.device)
            let tower = VisionTower(
                config: visionConfig, context: context, weights: mapping)
            let patcher = ImagePatcher(config: visionConfig)

            var elements: [Qwen35MoeRunner.PromptElement] = []
            for piece in pieces {
                switch piece {
                case .text(let tokens):
                    elements.append(.text(tokens))
                case .image(let index):
                    let started = Date()
                    let patched = try patcher.patch(
                        contentsOf: URL(fileURLWithPath: options.images[index]))
                    let embedded = try tower.forward(
                        patches: patched.values, grid: patched.grid)
                    let tokens = visionConfig.tokenCount(for: patched.grid)
                    FileHandle.standardError.write(Data(String(
                        format: "  image %d: %dx%d, %d tokens, tower %.1f s\n",
                        index + 1, patched.pixelWidth, patched.pixelHeight, tokens,
                        Date().timeIntervalSince(started)).utf8))
                    elements.append(.image(
                        embeddings: embedded, frames: patched.grid.temporal,
                        height: patched.grid.height / visionConfig.spatialMergeSize,
                        width: patched.grid.width / visionConfig.spatialMergeSize))
                }
            }
            distribution = try qwen.prefill(elements: elements)
            }
        }
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

        // --- The search pass, before the answer ---
        //
        // The same two-continuations-from-one-point shape the application runs: ask for a
        // query without thinking, rewind to the checkpoint `prefill` just took, feed the
        // results, then answer. Nothing of the conversation is processed twice.
        if let path = options.searchFrom {
            // Replay: no query pass, no network, no credit. The turn is a pure function of the
            // seed and the prompt, which is what an A/B of two prompts requires.
            let data = try Data(contentsOf: URL(fileURLWithPath: path))
            let response = try TavilyClient.decode(data, query: prompt)
            let rendered = SearchBlock.render(response, budget: 1000) {
                tokenizer.encode($0, allowSpecial: false).count
            }
            FileHandle.standardError.write(Data(String(
                format: "\u{1B}[2m[replay] %d of %d results, %d tokens\u{1B}[0m\n",
                rendered.included, rendered.included + rendered.dropped,
                rendered.tokens).utf8))
            distribution = try runner.prefill(tokens: tokenizer.encode(
                format.renderSearchResults(rendered.text, settings: promptSettings),
                allowSpecial: true))
        } else if options.searches {
            let anchor = promptTokens.count
            distribution = try runner.prefill(
                tokens: tokenizer.encode(format.renderQueryRequest(), allowSpecial: true))

            var querySettings = promptSettings
            querySettings.reasoning = .off
            let queryParser = format.makeParser(
                tokenizer: tokenizer, settings: querySettings)
            let querySampling = ModelRunner.Sampling(
                temperature: 0.7, topP: 0.8, topK: 20, seed: options.seed)
            var raw = ""
            var spent = 0
            while spent < 64 && !queryParser.isFinished {
                let token = runner.sample(from: distribution, using: querySampling)
                spent += 1
                for event in queryParser.consume(token) {
                    if case .answer(let piece) = event { raw += piece }
                }
                if queryParser.isFinished { break }
                distribution = try runner.forward(token: token, needsLogits: true)
            }
            let query = WebSearchTool.cleanQuery(raw)
            let shown = query.isEmpty ? "(none)" : query
            FileHandle.standardError.write(Data(
                ("\u{1B}[2m[query] " + shown + " (\(spent) tokens)\u{1B}[0m\n").utf8))

            let resumable = runner.reusablePrefix(atMost: anchor)
            guard resumable == anchor else {
                print("the model could not resume after the query pass")
                throw ExitError.planInvalid
            }
            runner.rewind(to: anchor)

            var block = WebSearchTool.failureNote("no query could be formed")
            if !query.isEmpty {
                do {
                    let client = try TavilyClient.fromEnvironment()
                    let response = try await client.search(SearchQuery(text: query))
                    let rendered = SearchBlock.render(response, budget: 1000) {
                        tokenizer.encode($0, allowSpecial: false).count
                    }
                    FileHandle.standardError.write(Data(String(
                        format: "\u{1B}[2m[search] %d of %d results, %d tokens\u{1B}[0m\n",
                        rendered.included, rendered.included + rendered.dropped,
                        rendered.tokens).utf8))
                    if !rendered.isEmpty { block = rendered.text }
                } catch {
                    let reason = (error as? SearchError)?.description ?? "\(error)"
                    FileHandle.standardError.write(
                        Data("\u{1B}[2m[search failed] \(reason)\u{1B}[0m\n".utf8))
                    block = WebSearchTool.failureNote(reason)
                }
            }
            distribution = try runner.prefill(tokens: tokenizer.encode(
                format.renderSearchResults(block, settings: promptSettings),
                allowSpecial: true))
        }

        // --- Generation ---
        runner.beginGeneration()
        var parser = format.makeParser(tokenizer: tokenizer, settings: promptSettings)
        let sampling = ModelRunner.Sampling(
            temperature: options.temperature ?? model.samplingDefaults.temperature,
            topP: options.topP ?? model.samplingDefaults.topP,
            topK: options.topKOverride ?? model.samplingDefaults.topK,
            presencePenalty: options.presencePenaltyOverride
                ?? model.samplingDefaults.presencePenalty,
            repeatWindow: options.repeatWindowOverride ?? model.samplingDefaults.repeatWindow,
            frequencyPenalty: options.frequencyPenaltyOverride
                ?? model.samplingDefaults.frequencyPenalty,
            seed: options.seed)

        var generated = 0
        start = Date()
        var analysisShown = false
        var reasoningCharacters = 0
        var producedAnswer = false

        var rounds = 0
        var pendingCall: ToolCall?
        toolLoop: while true {
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
                    producedAnswer = true
                case .reasoning(let text):
                    reasoningCharacters += text.count
                    guard options.showAnalysis else { break }
                    if !analysisShown {
                        FileHandle.standardError.write(Data("\u{1B}[2m[reasoning] ".utf8))
                        analysisShown = true
                    }
                    FileHandle.standardError.write(Data(text.utf8))
                case .toolCall(let call):
                    let arguments = call.arguments.keys.sorted()
                        .map { "\($0)=\(call.arguments[$0] ?? "")" }
                        .joined(separator: ", ")
                    FileHandle.standardError.write(
                        Data("\n\u{1B}[2m[tool call] \(call.name)(\(arguments))\u{1B}[0m\n".utf8))
                    pendingCall = call
                case .stopped:
                    break
                }
            }
            if parser.isFinished { break }
            distribution = try runner.forward(token: token, needsLogits: true)
        }

        // --- The tool round, exactly as the application runs it ---
        guard options.searches, let call = pendingCall, rounds < 2 else { break toolLoop }
        pendingCall = nil
        rounds += 1

        let result: String
        if let query = WebSearchTool.query(from: call) {
            do {
                let client = try TavilyClient.fromEnvironment()
                let searchStart = Date()
                let response = try await client.search(query)
                let block = SearchBlock.render(response, budget: 1000) {
                    tokenizer.encode($0, allowSpecial: false).count
                }
                FileHandle.standardError.write(Data(String(
                    format: "\u{1B}[2m[search] %.2f s, %d of %d results, %d tokens\u{1B}[0m\n",
                    Date().timeIntervalSince(searchStart), block.included,
                    block.included + block.dropped, block.tokens).utf8))
                result = block.isEmpty
                    ? WebSearchTool.failureNote("the search returned nothing usable") : block.text
            } catch {
                let reason = (error as? SearchError)?.description ?? "\(error)"
                FileHandle.standardError.write(
                    Data("\u{1B}[2m[search failed] \(reason)\u{1B}[0m\n".utf8))
                result = WebSearchTool.failureNote(reason)
            }
        } else {
            result = "The search needs a non-empty `query` parameter."
        }

        let appended = tokenizer.encode(
            format.renderToolResult(result, settings: promptSettings), allowSpecial: true)
        let appendStart = Date()
        distribution = try runner.prefill(tokens: appended)
        FileHandle.standardError.write(Data(String(
            format: "\u{1B}[2m[fed] %d tokens in %.1f s\u{1B}[0m\n\n",
            appended.count, Date().timeIntervalSince(appendStart)).utf8))
        parser = format.makeParser(tokenizer: tokenizer, settings: promptSettings)
        }
        if analysisShown { FileHandle.standardError.write(Data("\u{1B}[0m\n\n".utf8)) }
        // The same thing the application now says, because the turn that reasons and then
        // stops without answering is the failure this feature actually had, and a diagnostic
        // tool that shows it as a blank line is no use for the next round of work.
        if !producedAnswer {
            FileHandle.standardError.write(Data(
                ("\u{1B}[2m[no answer] the model ended its turn after \(generated) tokens "
                 + "without writing one"
                 + (rounds > 0 ? ", having searched \(rounds) time(s)" : "")
                 + "\u{1B}[0m\n").utf8))
        }
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
