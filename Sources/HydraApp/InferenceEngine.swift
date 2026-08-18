import Foundation
import HydraCore
import HydraInstall
import HydraMetal
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
        case finished(tokens: Int, seconds: Double, contextUsed: Int)
        case failed(String)
    }

    private let queue = DispatchQueue(label: "hydra.inference", qos: .userInitiated)
    private var context: MetalContext?
    /// Held through the seam: past `ModelRuntime.makeRunner` nothing here knows or asks
    /// which architecture is loaded.
    private var runner: (any TextModelRunner)?
    private var tokenizer: BPETokenizer?
    private var mapping: ModelMapping?
    private(set) public var loaded: Loaded?

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
            loaded = nil
        }
    }

    // MARK: - Generation

    public func cancel() { cancelled.set(true) }

    private var tower: VisionTower?

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
                let prompt = tokenizer.encode(
                    format.render(turns: turns, settings: settings.prompt), allowSpecial: true)
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
                if !images.isEmpty {
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
                let prefilled = Date()

                let parser = format.makeParser(
                    tokenizer: tokenizer, settings: settings.prompt)
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
                    presencePenalty: published.presencePenalty)

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
                let room = runner.kvCache.contextLength - prompt.count
                    - format.reservedStopTokens
                let budget = max(1, min(settings.maximumTokens, room))

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
                var announcedFirstToken = false
                func announceFirstToken() {
                    guard !announcedFirstToken else { return }
                    announcedFirstToken = true
                    onEvent(.firstToken(seconds: Date().timeIntervalSince(started)))
                }
                var producedFinalText = false

                outer: while produced < budget && !parser.isFinished {
                    if cancelled.value { break }

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
                            case .stopped:
                                break
                            }
                        }
                        // A batch may contain tokens beyond the end marker. They stay in the
                        // KV cache but not in `fed`, which describes what was kept: the next
                        // turn will rewind to the common prefix, hence below them, and they
                        // will disappear on their own.
                        if parser.isFinished || produced >= budget { break outer }
                    }
                }
                cachedTokens = fed
                flush(force: true)

                // Stopping on the budget without having written anything leaves the user with
                // reasoning followed by nothing. Saying so beats letting them guess.
                if produced >= budget && !producedFinalText {
                    onEvent(.text(
                        room <= settings.maximumTokens
                            ? "_(stopped: only \(room) tokens of context were left. "
                                + "Start a new conversation or raise the context length.)_"
                            : "_(stopped: reasoning used up all \(budget) allowed "
                                + "tokens. Lower the reasoning effort.)_"))
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
                onEvent(.finished(
                    tokens: produced, seconds: Date().timeIntervalSince(prefilled),
                    contextUsed: prompt.count + produced))
            } catch {
                // The cache's state is no longer known with certainty: we forget it rather
                // than risk reusing a prefix that corresponds to nothing.
                cachedTokens = []
                onEvent(.failed("\(error)"))
            }
        }
    }

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
