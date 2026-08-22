import Foundation
import HydraCore
import HydraInstall
import HydraMetal
import HydraSearch
import HydraTokenize
import HydraVision

/// Runs inference off the main thread and reports as it goes.
///
/// The runtime is blocking by construction: every layer commits a command buffer and
/// waits. Running it on the main thread would freeze the interface for the whole
/// generation. So execution is isolated on a dedicated queue, and events come back up on the
/// main thread.
public final class InferenceEngine: @unchecked Sendable {

    public struct Loaded: Sendable {
        public let entry: CatalogEntry
        public let contextLength: Int
        public let slotsPerLayer: Int
        /// Mapped weights, file-backed.
        public let mappedBytes: Int
        /// Memory reserved by the runtime: expert slots, KV, scratch, logits.
        public let reservedBytes: Int
    }

    /// A load failure, carrying a readable message.
    public struct LoadFailure: Error, CustomStringConvertible, Sendable {
        public let description: String
    }

    public enum Event: Sendable {
        /// Emitted as soon as the prompt is encoded: gives context usage before generation.
        /// - Parameter newTokens: the part of the prompt not already in the cache, which is
        ///   what the wait before the first token is actually proportional to.
        case started(promptTokens: Int, newTokens: Int, contextLength: Int)
        /// Latency to the first visible token, prefill included.
        case firstToken(seconds: Double)
        case reasoning(String)
        case text(String)
        /// A tool call was read back and is being run. The query is shown, because the whole
        /// premise of the feature is that the user knows what left their machine.
        case searching(query: String, provider: String)
        /// What came back, priced before it is fed. The same contract as an image's chip: the
        /// cost is visible before it is paid, not after.
        case searched(SearchOutcome)
        /// The search failed. Not a failed generation: the model is told and answers anyway.
        case searchFailed(String)
        case finished(tokens: Int, seconds: Double, contextUsed: Int)
        case failed(String)
    }

    /// One completed search, as the interface needs it: what was asked, what is being read,
    /// and what it costs.
    public struct SearchOutcome: Sendable {
        public let query: String
        /// The sources actually fed to the model, in rank order. Not everything the provider
        /// returned: listing a source under an answer that never saw it is a citation to
        /// something the model did not read.
        public let sources: [SearchResult]
        public let dropped: Int
        public let tokens: Int
    }

    private let queue = DispatchQueue(label: "hydra.inference", qos: .userInitiated)
    private var context: MetalContext?
    /// Held through the seam: past `ModelRuntime.makeRunner` nothing here knows or asks
    /// which architecture is loaded.
    private var runner: (any TextModelRunner)?
    private var tokenizer: BPETokenizer?
    private var mapping: ModelMapping?
    private(set) public var loaded: Loaded?

    /// The sampler's seed, drawn once per model load.
    ///
    /// `TokenSampler` reads `Sampling.seed` only when its stream has not started, so this is the
    /// draw that decides the **first** generation after a load and nothing after it. Left at the
    /// type's fixed default, that made every user's first answer on a fresh model the same one,
    /// and M-077 found the constant `0x5EED1234` to be a bad draw: on the reported prompt it
    /// degenerates where six other seeds do not. A fixed seed is what makes a measurement
    /// reproducible and what makes one unlucky stream everybody's first impression.
    ///
    /// The CLI keeps the constant, deliberately: `hydra chat` is an instrument and its whole
    /// value is that two runs of it agree.
    private var samplerSeed: UInt64 = .random(in: 1...UInt64.max)

    private let cancelled = Flag()

    public init() {}

    // MARK: - Loading

    public func load(
        entry: CatalogEntry, contextLength: Int, slotsPerLayer: Int?,
        progress: @escaping @Sendable (String) -> Void,
        completion: @escaping @Sendable (Result<Loaded, LoadFailure>) -> Void
    ) {
        queue.async { [self] in
            do {
                let root = try ModelLocations.root(for: entry)
                progress("Reading the tokenizer…")
                let tokenizer = try TokenizerInstaller.load(
                    from: root, architecture: entry.model.architecture)

                progress("Initializing the GPU…")
                let context = try self.context ?? MetalContext()
                self.context = context

                let profile = context.hardwareProfile(
                    memoryBandwidth: 94e9, diskBandwidth: 5.5e9)
                let policy: ExpertCachePolicy =
                    slotsPerLayer.map { .slotsPerLayer($0) } ?? .balanced
                let budget = MemoryBudget(
                    config: entry.model, hardware: profile,
                    contextLength: contextLength, policy: policy)

                progress("Opening the weights…")
                let mapping = try ModelMapping(
                    root: root, model: entry.model, device: context.device)
                let cache = ExpertSlotCache(
                    root: root, model: entry.model,
                    slotsPerLayer: budget.expertSlotsPerLayer, device: context.device)
                // The architecture is decided here and nowhere after: everything below holds
                // `any TextModelRunner`.
                let runner = try ModelRuntime.makeRunner(
                    model: entry.model, context: context, mapping: mapping,
                    expertCache: cache, contextLength: contextLength)

                // Bring the pages in with a sequential read rather than by scattered faults
                // during the first generation: measured, that halved the time of the first
                // prompt.
                progress("Prefaulting the weights…")
                mapping.prefault()

                self.tokenizer = tokenizer
                self.runner = runner
                // A fresh stream for a freshly loaded model, so an unlucky one is not permanent.
                self.samplerSeed = .random(in: 1...UInt64.max)
                self.mapping = mapping
                let loaded = Loaded(
                    entry: entry, contextLength: contextLength,
                    slotsPerLayer: budget.expertSlotsPerLayer,
                    mappedBytes: mapping.mappedByteCount,
                    reservedBytes: runner.reservedBytes)
                self.loaded = loaded
                completion(.success(loaded))
            } catch {
                completion(.failure(LoadFailure(description: "\(error)")))
            }
        }
    }

    public func unload() {
        // Cancel **before** queueing. `unload` runs on the inference queue, which is serial,
        // so it otherwise waits behind a generation that may have minutes left to run: the
        // interface greys out the load button, keeps saying "thinking", and nothing it does
        // can reach the work already in flight.
        cancelled.set(true)
        queue.async { [self] in
            runner = nil
            mapping = nil
            tokenizer = nil
            tower = nil          // 851 MiB, and it belongs to the model that just went away
            gemmaTower = nil     // and Gemma's is a gigabyte
            loaded = nil
        }
    }

    // MARK: - Generation

    public func cancel() { cancelled.set(true) }

    private var tower: VisionTower?
    private var gemmaTower: Gemma4VisionTower?

    /// Gemma's tower, built once and dropped with the model. A gigabyte should not be mapped
    /// for a conversation that never sends a picture.
    private func gemmaVisionTower() throws -> Gemma4VisionTower {
        if let gemmaTower { return gemmaTower }
        guard let context, let root = mapping?.root else {
            throw MetalContext.ContextError.noDevice
        }
        let built = try Gemma4VisionMapping(root: root, device: context.device)
        let made = Gemma4VisionTower(
            config: Gemma4VisionConfig.a4b, context: context, weights: built)
        gemmaTower = made
        return made
    }

    /// The vision tower, built once and kept for the life of the loaded model.
    ///
    /// Lazily, because 851 MiB should not be mapped for a conversation that never sends a
    /// picture, and every conversation so far has been one of those.
    private func visionTower() throws -> VisionTower {
        if let tower { return tower }
        guard let context, let root = mapping?.root else {
            throw MetalContext.ContextError.noDevice
        }
        let built = try VisionMapping(root: root, device: context.device)
        let made = VisionTower(
            config: Qwen35VisionConfig.a3b, context: context, weights: built)
        tower = made
        return made
    }

    public func generate(
        turns: [ChatTurn], settings: GenerationSettings, images: [String] = [],
        onEvent: @escaping @Sendable (Event) -> Void
    ) {
        cancelled.set(false)
        queue.async { [self] in
            guard let runner, let tokenizer, let model = loaded?.entry.model else {
                onEvent(.failed("no model loaded"))
                return
            }
            do {
                // The prompt format is chosen from the loaded model, once, here.
                let format = ConversationFormats.format(for: runner.architecture)

                // Declared only if we can both render the dialect and run what comes back.
                //
                // Neither half is optional. A declaration a format cannot parse gets a call
                // printed into the answer as text; a declaration with no client behind it gets
                // a model that promises to look something up and then cannot. Both cost the
                // conversation its cached prefix as well, since the block sits at the head of
                // the prompt — so when search is unavailable this renders byte for byte what it
                // rendered before search existed.
                var promptSettings = settings.prompt
                // Tools are no longer declared to the model at all.
                //
                // Letting it decide whether to search meant letting it deliberate, and
                // deliberation is where this model fails (M-077): asked whether something it
                // had never heard of existed, it spent its whole budget arguing and then
                // answered nothing. The turn is split instead — the query is asked for
                // directly, without thinking, and the answer is written afterwards with the
                // facts already in hand. That also returns the ~317 tokens the declaration
                // cost at the head of every prompt.
                promptSettings.tools = []
                let searches = settings.usesWebSearch && format.supportsTools
                    && webSearch != nil
                // The prompt says what will actually happen, not what was asked for: a system
                // turn promising results that never arrive is worse than no promise at all.
                promptSettings.searching = searches

                // A searching turn answers **without thinking**, whatever the switch says.
                //
                // Not a preference. Deliberation is where this model fails, and a turn that has
                // already been handed the facts has nothing left to deliberate about: measured
                // over the same prompt and seeds, thinking on degenerates on half of them and
                // thinking off on none of eight (M-077). The one place reasoning still earns
                // its keep is writing the query, and that pass forces it off too — a query is
                // a dozen words.
                //
                // The interface says so rather than hiding it: a switch that silently does
                // nothing is worse than one that is honestly overridden.
                if searches { promptSettings.reasoning = .off }

                // The conversation without its generation prompt, so the turn can be opened
                // from this point twice: once to ask for a query, once to answer.
                let open = format.renderOpen(turns: turns, settings: promptSettings)
                let prompt = tokenizer.encode(
                    open + (searches ? "" : format.generationPrompt(promptSettings)),
                    allowSpecial: true)
                let started = Date()

                // The previous turn's work is reused.
                //
                // Every turn re-renders the whole conversation: by the tenth exchange the
                // prompt is thousands of tokens of which three are new, and re-prefilling
                // them all redoes work already done, sixteen seconds measured on a
                // thousand tokens. We keep only the prefix common to both prompts, and
                // compute only what follows.
                //
                // `cachedTokens` is the exact sequence passed to the model, answer included:
                // it is what describes the cache's contents, not the prompt alone. An edit
                // or a regeneration shortens the common prefix, and resumption happens by
                // itself at the right place.
                // `canRewind` is the wrong question, and asking it cost every Gemma turn a
                // full re-prefill: it is false for any model with a sliding window, five
                // layers in six, here, so this branch never ran and a thousand-token
                // conversation paid 38 s to recompute what it already had. Resuming needs to
                // go back a handful of tokens, which a bounded ring holds.
                // A conversation carrying a picture does not reuse anything.
                //
                // The reuse below reasons entirely in flat token arrays, and an image is not
                // one: its tokens carry embeddings from the tower and three rotary positions
                // apiece, so a prefix that matches by token id can still mean something
                // different. Rather than make that machinery multimodal, such a conversation
                // reprefills from nothing each turn. It costs time and it cannot be subtly
                // wrong, which for a first version is the right way round. What should be
                // cached here eventually is the tower's output, not the tokens.
                var candidate = images.isEmpty
                    ? min(commonPrefixLength(cachedTokens, prompt), runner.position) : 0
                // At least one token must be processed to obtain a distribution.
                if candidate >= prompt.count { candidate = max(0, prompt.count - 1) }
                // The longest prefix the runner can *actually* resume from, which for a
                // recurrent model is the nearest turn boundary at or below the candidate and
                // not the candidate itself.
                let reusable = runner.reusablePrefix(atMost: candidate)

                // Announced here rather than before the reuse is known, because the number
                // that explains the wait is the new part, not the prompt's length. Without it
                // the time to first token looks arbitrary: a 2400-token turn resuming a
                // conversation is quicker than a fresh 800-token paste, and nothing on screen
                // said why.
                onEvent(.started(
                    promptTokens: prompt.count, newTokens: prompt.count - reusable,
                    contextLength: runner.kvCache.contextLength))

                if reusable > 0 {
                    runner.rewind(to: reusable)
                } else {
                    runner.reset()
                }

                var fed = Array(prompt[0..<reusable])
                let remaining = Array(prompt[reusable...])

                // Prefilled in chunks so that stopping is possible while it runs.
                //
                // One call would be simpler and is what this used to do. But prefill is the
                // longest uninterruptible stretch of a turn, twenty seconds on Gemma, more on
                // a long conversation, and during it the stop button did nothing at all. The
                // chunk is the granularity of "stop now"; it is large enough not to disturb
                // the batched path, which chunks internally anyway.
                // Never below the runner's own chunk, which differs per model: 256 for Gemma
                // and 512 for Qwen (M-057). Slicing finer hands the batched path half-chunks
                // and pays the expert reads twice, which is the failure the constant's comment
                // warned about and could not prevent while it was a constant.
                let cancellationChunk = max(
                    Self.prefillCancellationChunk, runner.prefillChunkTokens)
                var distribution = UnsafeBufferPointer<Float>(start: nil, count: 0)
                var offset = 0
                var stoppedDuringPrefill = false
                while offset < remaining.count {
                    if cancelled.value { stoppedDuringPrefill = true; break }
                    let end = min(offset + cancellationChunk, remaining.count)
                    let slice = Array(remaining[offset..<end])
                    distribution = try runner.prefill(tokens: slice)
                    fed += slice
                    offset = end
                }

                // --- The picture path: one prefill, with the tower's output spliced in ---
                if !images.isEmpty, let gemma = runner as? Gemma4ModelRunner {
                    let visionConfig = Gemma4VisionConfig.a4b
                    guard let pieces = Gemma4Prompt.split(
                        tokens: prompt, atPlaceholder: visionConfig.imageTokenID,
                        images: images.count)
                    else {
                        onEvent(.failed("the prompt's image placeholders do not match"))
                        return
                    }
                    let tower = try gemmaVisionTower()
                    let patcher = Gemma4ImagePatcher(config: visionConfig)
                    var elements: [Gemma4ModelRunner.PromptElement] = []
                    for piece in pieces {
                        switch piece {
                        case .text(let tokens):
                            elements.append(.text(tokens))
                        case .image(let index):
                            if cancelled.value { break }
                            let patched = try patcher.patch(
                                contentsOf: URL(fileURLWithPath: images[index]))
                            let embedded = try tower.forward(
                                patches: patched.values,
                                gridHeight: patched.gridHeight, gridWidth: patched.gridWidth)
                            elements.append(.image(
                                embeddings: embedded, tokens: patched.tokens))
                        }
                    }
                    distribution = try gemma.prefill(elements: elements)
                    fed = prompt
                } else if !images.isEmpty {
                    guard let qwen = runner as? Qwen35MoeRunner,
                        let pieces = QwenFormat.split(
                            tokens: prompt, atImagePad: Qwen35VisionConfig.a3b.imageTokenID,
                            images: images.count)
                    else {
                        onEvent(.failed("this model cannot read images"))
                        return
                    }
                    let tower = try visionTower()
                    let patcher = ImagePatcher(config: Qwen35VisionConfig.a3b)

                    var elements: [Qwen35MoeRunner.PromptElement] = []
                    for piece in pieces {
                        switch piece {
                        case .text(let tokens):
                            elements.append(.text(tokens))
                        case .image(let index):
                            if cancelled.value { break }
                            let patched = try patcher.patch(
                                contentsOf: URL(fileURLWithPath: images[index]))
                            let embedded = try tower.forward(
                                patches: patched.values, grid: patched.grid)
                            elements.append(.image(
                                embeddings: embedded, frames: patched.grid.temporal,
                                height: patched.grid.height / 2,
                                width: patched.grid.width / 2))
                        }
                    }
                    distribution = try qwen.prefill(elements: elements)
                    fed = prompt
                }
                guard !stoppedDuringPrefill else {
                    // The cache holds a prefix of the prompt, which is exactly what the next
                    // turn wants to reuse, so `fed` is kept rather than discarded.
                    cachedTokens = fed
                    onEvent(.finished(
                        tokens: 0, seconds: 0, contextUsed: fed.count))
                    return
                }
                // --- The search pass, before a word of the answer is written ---
                //
                // Two continuations from one point. The conversation has just been prefilled
                // and `prefill` ends by checkpointing the recurrence, so the position the query
                // pass starts from is exactly the position it can be rewound to. Nothing of the
                // conversation is processed twice.
                var searchOutcome: SearchOutcome?
                if searches {
                    let anchor = fed.count
                    let request = tokenizer.encode(
                        format.renderQueryRequest(), allowSpecial: true)
                    distribution = try runner.prefill(tokens: request)

                    // Short, and unthinking. The model is not being asked whether to search.
                    let query = try readQuery(
                        runner: runner, tokenizer: tokenizer, format: format,
                        settings: promptSettings, from: &distribution)

                    // Back to the anchor. The query pass never happened as far as the answer is
                    // concerned: it leaves no tokens in `fed` and none in the answer's context.
                    let resumable = runner.reusablePrefix(atMost: anchor)
                    guard resumable == anchor else {
                        // Only reachable if the checkpoint the prefill takes were removed. Said
                        // out loud rather than silently reprocessing the conversation.
                        onEvent(.searchFailed(
                            "the model could not resume after the query pass"))
                        throw RunnerFailure.cannotResume
                    }
                    runner.rewind(to: anchor)

                    let block = runSearch(
                        query, tokenizer: tokenizer, budget: settings.searchTokenBudget,
                        onEvent: onEvent)
                    searchOutcome = block.outcome

                    let results = tokenizer.encode(
                        format.renderSearchResults(block.text, settings: promptSettings),
                        allowSpecial: true)
                    var offset = 0
                    while offset < results.count {
                        if cancelled.value { break }
                        let end = min(offset + cancellationChunk, results.count)
                        distribution = try runner.prefill(
                            tokens: Array(results[offset..<end]))
                        offset = end
                    }
                    fed += results
                } else if searches == false, settings.usesWebSearch {
                    onEvent(.searchFailed(
                        webSearch == nil
                            ? "no search key is configured"
                            : "this model cannot search yet"))
                }

                let prefilled = Date()

                // The presence penalty describes this answer, not the conversation.
                //
                // Without this the history accumulates for the life of the loaded model, and by
                // the fifth turn every ordinary word carries a permanent penalty. It cannot be
                // folded into `resetSampling()`, which also restarts the random stream and would
                // make every regeneration return the same text.
                runner.beginGeneration()

                var parser = format.makeParser(
                    tokenizer: tokenizer, settings: promptSettings)
                // What the model itself publishes, unless the user has moved a slider.
                //
                // `topK` and the presence penalty are taken from the model either way: they are
                // not on any slider, and the presence penalty is the only thing in the sampler
                // that can break a repetition loop once one has started (M-069). Leaving them at
                // zero because the user nudged the temperature would reintroduce exactly the
                // failure this resolves.
                let published = model.samplingDefaults
                let sampling = ModelRunner.Sampling(
                    temperature: settings.followsModel
                        ? published.temperature : Float(settings.temperature),
                    topP: settings.followsModel ? published.topP : Float(settings.topP),
                    topK: published.topK,
                    presencePenalty: published.presencePenalty,
                    repeatWindow: published.repeatWindow,
                    frequencyPenalty: published.frequencyPenalty,
                    seed: samplerSeed)

                // Fragments are batched before being published.
                //
                // One token produces one fragment, and every fragment used to trigger a full
                // re-render of the conversation: a Markdown re-parse over the whole message,
                // relayout, animated scrolling. That work runs on the main thread, but it
                // consumes memory bandwidth, the very thing that limits decoding.
                // Throughput fell from 9.2 to 5.4 tok/s.
                //
                // At twenty refreshes per second the text scrolls smoothly to the eye, for a
                // fraction of the renders.
                var pendingText = ""
                var pendingReasoning = ""
                var lastFlush = Date()

                func flush(force: Bool = false) {
                    guard force || Date().timeIntervalSince(lastFlush) >= 0.05 else { return }
                    if !pendingText.isEmpty {
                        onEvent(.text(pendingText))
                        pendingText = ""
                    }
                    if !pendingReasoning.isEmpty {
                        onEvent(.reasoning(pendingReasoning))
                        pendingReasoning = ""
                    }
                    lastFlush = Date()
                }

                // The budget is bounded by the remaining context: exceeding it would overflow
                // the KV cache and fail the generation mid-sentence.
                // We keep a margin for the format's closing markers.
                // Recomputed rather than fixed, because a tool result lands in the same
                // context and the room left after it is not the room there was before.
                func remainingRoom() -> Int {
                    runner.kvCache.contextLength - fed.count - format.reservedStopTokens
                }
                let room = runner.kvCache.contextLength - prompt.count
                    - format.reservedStopTokens
                var budget = max(1, min(settings.maximumTokens, room))

                // Speculative decoding: the drafts come from the prompt itself.
                //
                // An ordinary pass re-reads every weight for a single token. Verifying four
                // candidates in one pass re-reads them once, the dense weights as well as
                // the shared experts. The draft costs nothing: it is a pattern search in
                // what has already been written.
                //
                // The output is identical token for token, draft right or wrong: a rejected
                // candidate only wastes computation. Four tests verify this on both
                // sampling modes.
                // `HYDRA_NOSPEC` disables speculation, for paired measurement.
                let speculates = ProcessInfo.processInfo.environment["HYDRA_NOSPEC"] == nil
                let drafter = NGramDrafter()
                var history = fed

                var produced = 0
                var pendingCall: ToolCall?
                var announcedFirstToken = false
                func announceFirstToken() {
                    guard !announcedFirstToken else { return }
                    announcedFirstToken = true
                    onEvent(.firstToken(seconds: Date().timeIntervalSince(started)))
                }
                var producedFinalText = false

                // The turn is a loop, not a single pass.
                //
                // Decode until the model stops or asks for a function; run it; append the
                // result and keep going. The append is at **token** level, onto the cache the
                // decode just filled: the call the model wrote stays where it is and only the
                // result's tokens are prefilled. Re-rendering the conversation instead would
                // mean re-encoding that call, and byte-level BPE makes no promise that the
                // same characters come back as the same ids across the join — it would work
                // most of the time, which is the worst way for it to be wrong.
                // The tool-call loop, retained and currently unreachable.
                //
                // Nothing declares tools any more (see above), so `pendingCall` is never set
                // and this runs exactly once before breaking — it is the plain decode loop with
                // a lid on it. It is kept rather than deleted because the design it belongs to
                // may be the right one for a model that can be trusted to deliberate: Qwen 3.8
                // ships `reasoning_effort` levels 3.6 does not have, and if bounded reasoning
                // turns out to be reliable, letting the model choose when to search is better
                // than searching on every turn. Deleting the protocol would mean rebuilding the
                // parser, the renderers and their tests to find that out.
                //
                // **It must stay unreachable until then.** Re-enabling it is one line —
                // restoring the tool declaration — and that line is the whole experiment.
                var rounds = 0
                toolLoop: while true {
                    outer: while produced < budget && !parser.isFinished {
                        if cancelled.value { break toolLoop }

                        let draft = speculates ? drafter.propose(history: history) : []
                        let outcome = try runner.step(
                            from: distribution, draft: draft, sampling: sampling)
                        distribution = outcome.next

                        for token in outcome.tokens {
                            produced += 1
                            history.append(token)
                            fed.append(token)

                            for event in parser.consume(token) {
                                switch event {
                                case .answer(let fragment):
                                    announceFirstToken()
                                    pendingText += fragment
                                    producedFinalText = true
                                    flush()
                                case .reasoning(let fragment):
                                    announceFirstToken()
                                    pendingReasoning += fragment
                                    flush()
                                case .toolCall(let call):
                                    pendingCall = call
                                case .stopped:
                                    break
                                }
                            }
                            // A batch may contain tokens beyond the end marker. They stay in
                            // the KV cache but not in `fed`, which describes what was kept:
                            // the next turn will rewind to the common prefix, hence below
                            // them, and they will disappear on their own.
                            if parser.isFinished || produced >= budget { break outer }
                        }
                    }

                    guard let call = pendingCall, rounds < Self.maximumToolRounds else { break }
                    pendingCall = nil
                    rounds += 1
                    flush(force: true)

                    let result = runTool(
                        call, tokenizer: tokenizer, budget: settings.searchTokenBudget,
                        onEvent: onEvent)
                    if cancelled.value { break }

                    // Only the new text is encoded and prefilled. `fed` grows by exactly these
                    // tokens, so the next turn's prefix match still describes the cache.
                    let appended = tokenizer.encode(
                        format.renderToolResult(result, settings: promptSettings),
                        allowSpecial: true)
                    var offset = 0
                    while offset < appended.count {
                        if cancelled.value { break toolLoop }
                        let end = min(offset + cancellationChunk, appended.count)
                        distribution = try runner.prefill(tokens: Array(appended[offset..<end]))
                        offset = end
                    }
                    fed += appended
                    history += appended

                    // A fresh parser, because the old one is finished and because the block
                    // just appended reopened the turn: it must start in the same reasoning
                    // state the prompt left it in.
                    parser = format.makeParser(tokenizer: tokenizer, settings: promptSettings)
                    budget = min(settings.maximumTokens, produced + remainingRoom())
                }
                cachedTokens = fed
                flush(force: true)

                // A turn that ends with no answer must say why.
                //
                // It used to say so only when the budget ran out, and there is a second way to
                // get here that is just as blank: the model closes its turn of its own accord
                // having written nothing but reasoning. Measured on a searching turn, that is
                // exactly what happened — 2,894 tokens produced, two characters of answer, and
                // an interface that showed a fold of reasoning and then silence. Below the
                // budget, so the old condition never fired and nothing was said at all.
                //
                // Not said when the user pressed stop: they know why it ended.
                if !producedFinalText && !cancelled.value {
                    onEvent(.text(Self.silentEndNote(
                        produced: produced, budget: budget, room: room,
                        rounds: searchOutcome == nil ? 0 : 1,
                        maximumTokens: settings.maximumTokens)))
                }
                // Throughput is measured over decoding alone.
                //
                // It used to count from the start of the prefill. On an established
                // conversation the prompt is several thousand tokens and processing it
                // weighs more than the entire answer: the same engine showed 6 tok/s on a
                // fresh conversation and 4 on a loaded one, while decoding at exactly the
                // same speed. The cost of the prefill is not hidden for all that, it is
                // what the time to first token measures, shown right beside it.
                //
                // `fed` is the exact sequence the cache holds, which after a search includes
                // the thousand tokens of results. `prompt.count + produced` was right until
                // this turn could grow in the middle: it reported 3,246 of 8k on a conversation
                // whose cache actually held 4,263, and the gauge is the only thing telling a
                // user how close to the wall they are.
                onEvent(.finished(
                    tokens: produced, seconds: Date().timeIntervalSince(prefilled),
                    contextUsed: fed.count))
            } catch {
                // The cache's state is no longer known with certainty: we forget it rather
                // than risk reusing a prefix that corresponds to nothing.
                cachedTokens = []
                onEvent(.failed("\(error)"))
            }
        }
    }

    // MARK: - Tools

    /// The search client, or `nil` when the user has not supplied a key.
    ///
    /// Set from outside rather than built here: whether search is available is a question about
    /// the user's settings, and the engine's job is to run what it is given.
    public var webSearch: (any WebSearch)?

    /// Runs one call and returns the text to feed back.
    ///
    /// Always returns something. Every failure below — an unknown function, a missing query, a
    /// used-up allowance, a dead network — is answered with a note rather than an error, because
    /// the model is mid-turn with a question in front of it and its own weights to answer from.
    /// Ending the generation would turn a flaky network into a blank reply.
    private func runTool(
        _ call: ToolCall, tokenizer: BPETokenizer, budget: Int,
        onEvent: @escaping @Sendable (Event) -> Void
    ) -> String {
        guard let client = webSearch else {
            return WebSearchTool.failureNote("web search is not configured")
        }
        guard let query = WebSearchTool.query(from: call) else {
            // The model named a function we do not have, or called ours with nothing to search
            // for. Both are its mistakes to make, and telling it so costs a sentence.
            return call.name == WebSearchTool.name
                ? "The search needs a non-empty `query` parameter."
                : "There is no function called `\(call.name)`. "
                    + "The only one available is `\(WebSearchTool.name)`."
        }

        onEvent(.searching(query: query.text, provider: client.name))

        let outcome = Self.awaitingSearch(client, query)
        switch outcome {
        case .failure(let error):
            let reason = (error as? SearchError)?.description ?? "\(error)"
            onEvent(.searchFailed(reason))
            return WebSearchTool.failureNote(reason)
        case .success(let response):
            let rendered = SearchBlock.render(response, budget: budget) {
                tokenizer.encode($0, allowSpecial: false).count
            }
            onEvent(.searched(SearchOutcome(
                query: query.text,
                sources: Array(response.results.prefix(rendered.included)),
                dropped: rendered.dropped, tokens: rendered.tokens)))
            guard !rendered.isEmpty else {
                return WebSearchTool.failureNote("the search returned nothing usable")
            }
            return rendered.text
        }
    }

    /// Blocks the inference queue on one asynchronous search.
    ///
    /// The runtime below is synchronous by construction — every layer commits a command buffer
    /// and waits — so the turn cannot become `async` without rewriting all of it for a call that
    /// takes about a second. This is the seam where the two models meet, and it is safe
    /// precisely because the queue it blocks is the dedicated inference queue and never the main
    /// thread. The timeout is the part that matters: without it an endpoint that never answers
    /// holds the model, the stop button and `unload` behind it forever.
    private static func awaitingSearch(
        _ client: any WebSearch, _ query: SearchQuery
    ) -> Result<SearchResponse, Error> {
        let semaphore = DispatchSemaphore(value: 0)
        let box = Box<Result<SearchResponse, Error>>(
            .failure(SearchError.badStatus(-1, provider: client.name, message: "timed out")))
        Task {
            do { box.value = .success(try await client.search(query)) }
            catch { box.value = .failure(error) }
            semaphore.signal()
        }
        _ = semaphore.wait(timeout: .now() + searchTimeout)
        return box.value
    }

    /// A value handed between the task and the queue waiting on it.
    private final class Box<T>: @unchecked Sendable {
        private var storage: T
        private let lock = NSLock()
        init(_ value: T) { storage = value }
        var value: T {
            get { lock.lock(); defer { lock.unlock() }; return storage }
            set { lock.lock(); storage = newValue; lock.unlock() }
        }
    }

    /// Why a turn ended without an answer, in the terms the user can act on.
    static func silentEndNote(
        produced: Int, budget: Int, room: Int, rounds: Int, maximumTokens: Int
    ) -> String {
        if produced >= budget {
            return room <= maximumTokens
                ? "_(stopped: only \(room) tokens of context were left. "
                    + "Start a new conversation or raise the context length.)_"
                : "_(stopped: reasoning used up all \(budget) allowed tokens. "
                    + "Lower the reasoning effort.)_"
        }
        let searched = rounds > 0
            ? " It searched the web \(rounds == 1 ? "once" : "\(rounds) times") first."
            : ""
        return "_(the model ended its turn after \(produced) tokens without writing an "
            + "answer.\(searched) It reasoned and then stopped; ask again, or lower the "
            + "reasoning effort.)_"
    }

    enum RunnerFailure: Error, CustomStringConvertible {
        case cannotResume
        var description: String {
            "the recurrence could not resume after the search pass"
        }
    }

    /// Decodes the query the model was asked for, and nothing more.
    ///
    /// Bounded hard. A query is a dozen words, and the bound is what keeps a model that has
    /// started to ramble from turning the cheap half of the turn into the expensive one. The
    /// first line is taken: the instruction asks for the query alone, and when the model adds a
    /// sentence anyway the first line is still the query.
    private func readQuery(
        runner: any TextModelRunner, tokenizer: BPETokenizer, format: any ConversationFormat,
        settings: PromptSettings, from distribution: inout UnsafeBufferPointer<Float>
    ) throws -> String {
        // Thinking is closed by `renderQueryRequest`, so the parser is told the same.
        var querySettings = settings
        querySettings.reasoning = .off
        let parser = format.makeParser(tokenizer: tokenizer, settings: querySettings)
        let sampling = ModelRunner.Sampling(temperature: 0.7, topP: 0.8, topK: 20, seed: 0)

        var text = ""
        var produced = 0
        while produced < Self.maximumQueryTokens && !parser.isFinished {
            if cancelled.value { break }
            let token = runner.sample(from: distribution, using: sampling)
            produced += 1
            for event in parser.consume(token) {
                if case .answer(let fragment) = event { text += fragment }
            }
            if parser.isFinished { break }
            distribution = try runner.forward(token: token, needsLogits: true)
        }
        return WebSearchTool.cleanQuery(text)
    }

    /// Runs the search for a query the model wrote, and renders the block.
    private func runSearch(
        _ query: String, tokenizer: BPETokenizer, budget: Int,
        onEvent: @escaping @Sendable (Event) -> Void
    ) -> (text: String, outcome: SearchOutcome?) {
        guard let client = webSearch else {
            return (WebSearchTool.failureNote("web search is not configured"), nil)
        }
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            onEvent(.searchFailed("the model wrote no query"))
            return (WebSearchTool.failureNote("no query could be formed"), nil)
        }

        onEvent(.searching(query: trimmed, provider: client.name))
        switch Self.awaitingSearch(client, SearchQuery(text: trimmed)) {
        case .failure(let error):
            let reason = (error as? SearchError)?.description ?? "\(error)"
            onEvent(.searchFailed(reason))
            return (WebSearchTool.failureNote(reason), nil)
        case .success(let response):
            let rendered = SearchBlock.render(response, budget: budget) {
                tokenizer.encode($0, allowSpecial: false).count
            }
            guard !rendered.isEmpty else {
                onEvent(.searchFailed("the search returned nothing usable"))
                return (WebSearchTool.failureNote("the search returned nothing usable"), nil)
            }
            let outcome = SearchOutcome(
                query: trimmed, sources: Array(response.results.prefix(rendered.included)),
                dropped: rendered.dropped, tokens: rendered.tokens)
            onEvent(.searched(outcome))
            return (rendered.text, outcome)
        }
    }

    /// How many tokens the query pass may spend. A query is a dozen words.
    static let maximumQueryTokens = 64

    /// How many times one turn may call a function before it must answer.
    ///
    /// Two. One search answers a question; a second refines it after seeing what came back.
    /// Beyond that the model is not converging, and each round is 15 seconds of prefill and a
    /// credit off a metered allowance, so the bound is low and deliberate rather than generous.
    static let maximumToolRounds = 2

    /// How long a search may take before the turn gives up on it and answers without one.
    ///
    /// The inference queue is serial and blocked while this runs, so an endpoint that never
    /// answers would otherwise hold the model, the stop button and `unload` behind it.
    static let searchTimeout: TimeInterval = 20

    /// How many prompt tokens are processed between two checks of the stop flag.
    ///
    /// A compromise, and the two sides pull in opposite directions: smaller reacts faster,
    /// larger keeps the batched prefill efficient.
    ///
    /// It must not be *below* the prefill runner's own chunk, or the batching never sees a
    /// full one and the tuning inside it is wasted: at 128 against Gemma's 256 the engine was
    /// handing the runner half-chunks and paying the expert reads twice.
    private static let prefillCancellationChunk = 256

    /// The exact sequence of tokens currently represented in the KV cache.
    ///
    /// Only touched from the inference queue, which is serial.
    private var cachedTokens: [Int] = []

    /// A flag shared between the main thread and the inference queue.
    private final class Flag: @unchecked Sendable {
        private var storage = false
        private let lock = NSLock()
        func set(_ value: Bool) {
            lock.lock()
            storage = value
            lock.unlock()
        }
        var value: Bool {
            lock.lock()
            defer { lock.unlock() }
            return storage
        }
    }
}
