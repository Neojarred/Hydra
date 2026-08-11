import Foundation
import HydraCore
import HydraFormat
import Metal

/// Runs Gemma 4, one token at a time.
///
/// A sibling of `ModelRunner` rather than a branch inside it: the two share every mechanism
/// that does not depend on the architecture — the slot cache, the KV cache, the command-buffer
/// discipline — and differ in the one thing that does, the forward pass (D-023).
///
/// Still simpler than `ModelRunner`: no speculative decoding and no read/compute overlap. Both
/// were added there on the back of a measurement, and neither has one here yet.
///
/// Prefill is no longer among them. It goes through `Gemma4PrefillRunner`, which reorders the
/// work layer-major so a layer's experts are read once per chunk instead of once per token —
/// the measurement that justified it being that prefill ran at about one token a second and
/// moved 1.7 GiB from SSD for each of them.
public final class Gemma4ModelRunner: @unchecked Sendable {

    public let config: Gemma4Config
    public let context: MetalContext
    public let mapping: ModelMapping
    public let expertCache: ExpertSlotCache
    public let kvCache: KVCache

    private let encoder: ForwardEncoder
    private let layerRunner: Gemma4LayerRunner
    private let scratch: Gemma4DecodeScratch
    /// One table per layer type, built once: the frequencies depend on the pattern, not on
    /// the position.
    private let ropeTables: [Gemma4RoPETables]
    private let logits: MTLBuffer
    /// Built once and reused. Prefill is where a conversation spends its visible time.
    private let prefillRunner: Gemma4PrefillRunner
    /// Chosen once, here, and never asked about again (D-023).
    private let weights: any Gemma4Weights

    public private(set) var position = 0
    public private(set) var lastTimings = ModelRunner.Timings()

    /// Seconds the GPU actually spent executing the last token's command buffers.
    ///
    /// Wall time minus this is the CPU's share — encoding, `pread`, and waiting. Five
    /// structural changes in a row moved nothing (M-031 to M-034), which is what a wrong model
    /// of the bottleneck looks like; this is the measurement that distinguishes "the kernels
    /// are slow" from "the GPU is idle" and should have been the first one taken.
    public private(set) var lastGPUSeconds = 0.0

    /// Called at named points inside each layer, when set.
    ///
    /// A permanent seam rather than a temporary `print`. The per-operator and whole-model tests
    /// run a 6-layer, 64-wide configuration; the real model is 30 layers and 2816 wide, and the
    /// first failure it produced was every logit `NaN` — a result those tests cannot reach and
    /// that says nothing about *where* it went wrong. Narrowing that to one layer took one run
    /// with this set; narrowing it to one stage inside the layer took a second.
    ///
    /// Costs one nil check per stage when unused.
    public var stageObserver: ((_ layer: Int, _ stage: String, _ values: [Float]) -> Void)?

    private func observe(_ layer: Int, _ stage: String, _ buffer: MTLBuffer, count: Int) {
        guard let stageObserver else { return }
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: count)
        stageObserver(layer, stage, Array(UnsafeBufferPointer(start: pointer, count: count)))
    }

    public enum RunnerError: Error, CustomStringConvertible {
        case allocationFailed(String)
        case contextExhausted(position: Int, capacity: Int)

        public var description: String {
            switch self {
            case .allocationFailed(let what): return "cannot allocate \(what)"
            case let .contextExhausted(position, capacity):
                return "context exhausted: position \(position) of \(capacity)"
            }
        }
    }

    /// - Parameter weights: where the weights live and how they decode. Defaults to the BF16
    ///   build; the MLX 4-bit build passes its own, and `config` is then the *geometry* of that
    ///   checkpoint — identical, which is the reason `Gemma4MLXConfig` wraps this type rather
    ///   than restating it.
    public init(
        config: Gemma4Config, context: MetalContext, mapping: ModelMapping,
        expertCache: ExpertSlotCache, contextLength: Int,
        weights: (any Gemma4Weights)? = nil
    ) throws {
        self.config = config
        self.context = context
        self.mapping = mapping
        self.expertCache = expertCache
        self.encoder = ForwardEncoder(context: context)
        self.scratch = try Gemma4DecodeScratch(config: config, device: context.device)
        self.kvCache = try KVCache(
            model: config, contextLength: contextLength, device: context.device)
        let source = weights ?? Gemma4BF16Weights(config: config, mapping: mapping)
        self.weights = source
        self.layerRunner = Gemma4LayerRunner(
            config: config, encoder: encoder, mapping: mapping, weights: source)
        self.ropeTables = (0..<config.layerCount).map {
            Gemma4RoPETables(config: config, layer: $0)
        }
        guard let logits = context.device.makeBuffer(
            length: config.vocabSize * MemoryLayout<Float>.size, options: .storageModeShared)
        else { throw RunnerError.allocationFailed("logits") }
        self.logits = logits
        self.prefillRunner = try Gemma4PrefillRunner(
            config: config, encoder: encoder, layerRunner: layerRunner,
            mapping: mapping, device: context.device)
    }

    public var reservedBytes: Int {
        expertCache.reservedBytes + kvCache.byteCount + scratch.byteCount + logits.length
            + prefillRunner.byteCount
    }

    // MARK: - State

    public func reset() {
        position = 0
        // The cache's length is its own state and must go back with the position, or the
        // next `rewind` measures against a history that no longer exists.
        kvCache.reset()
        expertCache.resetStatistics()
    }

    public var canRewind: Bool { kvCache.canRewind }

    public func rewind(to tokens: Int) {
        guard canRewind, tokens >= 0, tokens <= position else { return }
        kvCache.rewind(to: tokens)
        position = tokens
    }

    // MARK: - Forward

    private func commandBuffer() throws -> MTLCommandBuffer {
        guard let buffer = context.commandQueue.makeCommandBuffer() else {
            throw RunnerError.allocationFailed("command buffer")
        }
        return buffer
    }

    /// Writes the rotary tables for one layer at the current position.
    ///
    /// Per layer rather than once per token, because the frequencies differ between sliding
    /// and full layers — two thetas, and a zero-padded tail on the full ones.
    private func writeRopeTables(layer: Int, position: Int) {
        let tables = ropeTables[layer].tables(at: position)
        let pairs = tables.cos.count
        let cos = scratch.cosTable.contents().bindMemory(to: Float.self, capacity: pairs)
        let sin = scratch.sinTable.contents().bindMemory(to: Float.self, capacity: pairs)
        for i in 0..<pairs {
            cos[i] = Float(tables.cos[i])
            sin[i] = Float(tables.sin[i])
        }
    }

    /// The experts the router chose, read back on the CPU.
    private func selectedExperts() -> [Int] {
        let pointer = scratch.routerIndices.contents().bindMemory(
            to: UInt32.self, capacity: config.expertsPerToken)
        return (0..<config.expertsPerToken).map { Int(pointer[$0]) }
    }

    @discardableResult
    public func forward(
        token: Int, needsLogits: Bool = true
    ) throws -> UnsafeBufferPointer<Float> {
        guard position < kvCache.contextLength else {
            throw RunnerError.contextExhausted(
                position: position, capacity: kvCache.contextLength)
        }
        var timings = ModelRunner.Timings()
        var gpuSeconds = 0.0

        // --- Embedding, scaled by sqrt(hiddenSize) ---
        //
        // The scale is a constant in the model code and appears nowhere in the checkpoint.
        // Omitting it mis-scales the entire forward pass (D-022).
        let hidden = UnsafeMutableBufferPointer(
            start: scratch.hidden.contents().bindMemory(
                to: Float.self, capacity: config.hiddenSize),
            count: config.hiddenSize)
        weights.readEmbedding(token: token, into: hidden)
        let scale = config.embeddingScale
        for i in 0..<config.hiddenSize { hidden[i] *= scale }

        // One command buffer a layer instead of two.
        //
        // The loop used to commit twice a layer and wait on both — 60 round trips a token, and
        // on each one the GPU drains while the CPU encodes the next batch and the CPU idles
        // while the GPU runs it. Only one of the two waits earns its place: the router picks
        // the experts, and the CPU cannot `pread` them until it has read that choice back.
        //
        // The second wait was never synchronizing anything the GPU needed. Layer N's experts
        // and layer N+1's attention are sequentially dependent, but nothing between them
        // requires the CPU, so they belong in one buffer — where the shared serial encoder
        // already orders them and makes the mixture's writes visible to the attention that
        // reads them.
        //
        // Not merged when a stage observer is attached: it reads the scratch buffers after
        // each wait, and merging would let the next layer's attention overwrite them before
        // they were read. Diagnostics keep the slower path and the exact values.
        var pending: MTLCommandBuffer? = nil
        var pinnedLayer: Int? = nil

        for layer in 0..<config.layerCount {
            var start = Date()
            let buffer: MTLCommandBuffer
            if let merged = pending {
                buffer = merged
                pending = nil
            } else {
                writeRopeTables(layer: layer, position: position)
                buffer = try commandBuffer()
                try encodeAttentionAndRouter(
                    layer: layer, scratch: scratch, kvCache: kvCache, in: buffer)
            }
            context.commit(buffer)
            buffer.waitUntilCompleted()
            gpuSeconds += buffer.gpuEndTime - buffer.gpuStartTime
            timings.attentionAndRouter += Date().timeIntervalSince(start)

            // The previous layer's experts finished inside the buffer just waited on.
            if let pinned = pinnedLayer {
                expertCache.release(layer: pinned)
                pinnedLayer = nil
            }

            observe(layer, "attention", scratch.attention, count: config.hiddenSize)
            observe(layer, "projected", scratch.projected, count: config.hiddenSize)
            observe(layer, "routerLogits", scratch.routerLogits, count: config.expertCount)
            if stageObserver != nil {
                let indices = scratch.routerIndices.contents().bindMemory(
                    to: UInt32.self, capacity: config.expertsPerToken)
                stageObserver?(
                    layer, "expertIndices",
                    (0..<config.expertsPerToken).map { Float(indices[$0]) })
                observe(layer, "routerWeights", scratch.routerWeights,
                    count: config.expertsPerToken)
            }

            // The only unavoidable synchronization point: the CPU must know which experts
            // were chosen before it can read them.
            let selected = selectedExperts()

            start = Date()
            try expertCache.load(layer: layer, experts: selected)
            timings.expertIO += Date().timeIntervalSince(start)

            start = Date()
            let next = try commandBuffer()
            try layerRunner.encodeMixtureStart(scratch: scratch, in: next)
            for (slot, expert) in selected.enumerated() {
                let (blob, _) = try expertCache.expert(layer: layer, expert: expert, pin: true)
                try layerRunner.encodeSingleExpert(
                    buffer: blob, weightIndex: slot, scratch: scratch, in: next)
            }
            try layerRunner.encodeCombineBranches(
                layer: layer, count: selected.count, scratch: scratch, in: next)
            pinnedLayer = layer

            if stageObserver == nil && layer + 1 < config.layerCount {
                writeRopeTables(layer: layer + 1, position: position)
                try encodeAttentionAndRouter(
                    layer: layer + 1, scratch: scratch, kvCache: kvCache, in: next)
                pending = next
            } else {
                context.commit(next)
                next.waitUntilCompleted()
                gpuSeconds += next.gpuEndTime - next.gpuStartTime
                expertCache.release(layer: layer)
                pinnedLayer = nil
            }
            timings.mixture += Date().timeIntervalSince(start)

            // The mixture's own intermediates, read after the buffer completed. `denseOutput`
            // and `expertInput` are computed once; the per-expert buffers hold the last
            // expert's values, which is enough to tell a NaN apart from a finite result.
            observe(layer, "residual", scratch.residual, count: config.hiddenSize)
            observe(layer, "expertInput", scratch.expertInput, count: config.hiddenSize)
            observe(layer, "denseGate", scratch.denseGate, count: config.intermediateSize)
            observe(layer, "denseUp", scratch.denseUp, count: config.intermediateSize)
            observe(
                layer, "denseActivated", scratch.denseActivated,
                count: config.intermediateSize)
            observe(layer, "denseOutput", scratch.denseOutput, count: config.hiddenSize)
            observe(layer, "expertGate", scratch.expertGate, count: config.moeIntermediateSize)
            observe(
                layer, "expertSlices", scratch.expertSlices,
                count: config.expertsPerToken * config.hiddenSize)
            observe(layer, "hidden", scratch.hidden, count: config.hiddenSize)
        }

        // The cache's own bookkeeping, which is not the same as `position`.
        //
        // Omitting this was invisible for a whole conversation turn: the rows are written at
        // `position`, so attention was correct and the model answered. But `length` stayed at
        // zero, so the second turn's cache reuse called `rewind` and tripped its precondition,
        // crashing the app — and the capacity check inside `advance` had never run at all,
        // which is what stops a long conversation from writing past the buffer.
        try kvCache.advance()
        position += 1

        guard needsLogits else {
            lastTimings = timings
            return UnsafeBufferPointer(start: nil, count: 0)
        }

        // --- Final norm, the tied head, and softcapping ---
        let start = Date()
        let head = try commandBuffer()
        try encodeHead(in: head)
        context.commit(head)
        head.waitUntilCompleted()
        gpuSeconds += head.gpuEndTime - head.gpuStartTime
        timings.head = Date().timeIntervalSince(start)

        lastTimings = timings
        lastGPUSeconds = gpuSeconds
        return UnsafeBufferPointer(
            start: logits.contents().bindMemory(to: Float.self, capacity: config.vocabSize),
            count: config.vocabSize)
    }

    /// Processes a prompt, returning the distribution that follows it.
    ///
    /// Layer-major and chunked, through `Gemma4PrefillRunner`. Token-major prefill re-read a
    /// layer's experts for every token — 1.7 GiB from SSD per token on the real model, about
    /// one token a second, so a twenty-five token prompt cost twenty seconds before anything
    /// appeared. Grouping the experts within a layer reads each one once per chunk instead.
    ///
    /// A single token still goes through `forward`: there is nothing to share, and the batch
    /// buffers would be set up for nothing.
    @discardableResult
    public func prefill(tokens: [Int]) throws -> UnsafeBufferPointer<Float> {
        guard tokens.count > 1 else {
            return tokens.isEmpty
                ? UnsafeBufferPointer<Float>(start: nil, count: 0)
                : try forward(token: tokens[0])
        }
        guard position + tokens.count <= kvCache.contextLength else {
            throw RunnerError.contextExhausted(
                position: position + tokens.count, capacity: kvCache.contextLength)
        }

        var timings = ModelRunner.Timings()
        var offset = 0
        while offset < tokens.count {
            let end = min(offset + Gemma4PrefillRunner.chunk, tokens.count)
            let slice = Array(tokens[offset..<end])
            let scale = config.embeddingScale

            _ = try prefillRunner.run(
                tokenCount: slice.count, firstPosition: position,
                embeddings: { [self] index, row in
                    weights.readEmbedding(token: slice[index], into: row)
                    for i in 0..<config.hiddenSize { row[i] *= scale }
                },
                scratch: scratch, kvCache: kvCache, expertCache: expertCache,
                ropeTables: ropeTables, commandBuffer: commandBuffer, timings: &timings)

            for _ in slice { try kvCache.advance() }
            position += slice.count
            offset = end
        }

        // The head, over the last token's state, which the chunk left in the scratch.
        let start = Date()
        let head = try commandBuffer()
        try encodeHead(in: head)
        context.commit(head)
        head.waitUntilCompleted()
        timings.head = Date().timeIntervalSince(start)

        lastTimings = timings
        return UnsafeBufferPointer(
            start: logits.contents().bindMemory(to: Float.self, capacity: config.vocabSize),
            count: config.vocabSize)
    }

    // MARK: - Sampling

    /// Its own pseudo-random stream, held by value. Nothing about drawing a token depends on
    /// the architecture, so this is the shared implementation rather than a second copy.
    private var sampler = TokenSampler()

    public func sample(
        from distribution: UnsafeBufferPointer<Float>, using sampling: ModelRunner.Sampling
    ) -> Int {
        sampler.sample(from: distribution, using: sampling)
    }

    public func resetSampling() {
        sampler.reset()
    }

    // MARK: - Verification

    /// Grown on demand and reused, so that every row handed back stays valid together.
    private var batchLogits: MTLBuffer?

    /// Processes `tokens` and returns the logits of every position, row `i` predicting what
    /// follows `tokens[i]`.
    ///
    /// **Sequential, where `ModelRunner`'s is batched.** The result is identical; the cost is
    /// not. `ModelRunner` re-reads the dense weights once to verify `n` tokens, which is the
    /// entire reason speculative decoding pays there. Here each position is a full pass, so
    /// verifying `n` costs exactly what decoding `n` costs.
    ///
    /// That is why this is correct but unused by `step` below. It exists because the exactness
    /// harness needs per-position logits, and because a batched implementation later is a
    /// change to this method's body and to nothing else.
    ///
    /// The returned rows point into one shared buffer and remain valid until the next call.
    public func verify(tokens: [Int]) throws -> [UnsafeBufferPointer<Float>] {
        precondition(!tokens.isEmpty, "nothing to verify")

        let bytes = tokens.count * config.vocabSize * MemoryLayout<Float>.size
        if batchLogits == nil || batchLogits!.length < bytes {
            guard let buffer = context.device.makeBuffer(
                length: bytes, options: .storageModeShared)
            else { throw RunnerError.allocationFailed("batchLogits") }
            batchLogits = buffer
        }
        guard let output = batchLogits else {
            throw RunnerError.allocationFailed("batchLogits")
        }
        let base = output.contents().bindMemory(
            to: Float.self, capacity: tokens.count * config.vocabSize)

        // Copied out on each step: `forward` hands back the one logits buffer, which the next
        // position overwrites. Rows that alias would be a bug the caller could not see.
        for (index, token) in tokens.enumerated() {
            let row = try forward(token: token)
            base.advanced(by: index * config.vocabSize)
                .update(from: row.baseAddress!, count: config.vocabSize)
        }

        return (0..<tokens.count).map {
            UnsafeBufferPointer(
                start: base.advanced(by: $0 * config.vocabSize), count: config.vocabSize)
        }
    }

    /// One decoding step. The draft is deliberately ignored.
    ///
    /// Speculative decoding is a trade: verify `n` candidates for the price of re-reading the
    /// weights once, and keep the prefix that was right. Without a batched pass there is no
    /// trade — verifying the draft costs one forward per token, exactly what decoding them
    /// costs — so accepting a draft here would buy nothing and could only lose, since a
    /// rejected token is work thrown away.
    ///
    /// Returning a single token is within the contract: the caller accepts *the longest
    /// correct prefix*, and one token is always a correct prefix. Output is identical to
    /// `ModelRunner`'s at equal seed, because a step that consumes exactly one draw is what
    /// ordinary decoding already does.
    public func step(
        from distribution: UnsafeBufferPointer<Float>, draft: [Int],
        sampling: ModelRunner.Sampling
    ) throws -> (tokens: [Int], next: UnsafeBufferPointer<Float>) {
        let token = sample(from: distribution, using: sampling)
        return ([token], try forward(token: token))
    }

    /// The final norm, the tied head and the softcap, over whatever is in `scratch.hidden`.
    ///
    /// Shared by decoding and prefill rather than written in both: `tie_word_embeddings` means
    /// the embedding **is** the output projection, and a second transcription of that is a
    /// second chance to look for an `lm_head` tensor that does not exist.
    /// A layer's attention, its dense branch, and the router's top-k, into one buffer.
    ///
    /// Called from two places now — once to prime the first layer, and once per layer to
    /// append the next one's attention behind the current one's experts — so it lives here
    /// rather than being written twice.
    private func encodeAttentionAndRouter(
        layer: Int, scratch: Gemma4DecodeScratch, kvCache: KVCache,
        in commandBuffer: MTLCommandBuffer
    ) throws {
        try layerRunner.encodeAttentionAndRouter(
            layer: layer, position: position, scratch: scratch, kvCache: kvCache,
            in: commandBuffer)
        let routerScale = try weights.plain("router.per_expert_scale", layer: layer)
        try encoder.gemmaRouterTopK(
            logits: scratch.routerLogits,
            perExpertScale: routerScale.buffer, perExpertScaleOffset: routerScale.offset,
            indices: scratch.routerIndices, weights: scratch.routerWeights,
            expertCount: config.expertCount, topK: config.expertsPerToken, in: commandBuffer)
    }

    private func encodeHead(in commandBuffer: MTLCommandBuffer) throws {
        let finalNorm = try weights.finalNorm()
        try encoder.rmsNorm(
            input: scratch.hidden, scale: finalNorm.buffer, scaleOffset: finalNorm.offset,
            output: scratch.normed, size: config.hiddenSize, eps: config.rmsNormEps,
            in: commandBuffer)
        try encoder.encodeProjection(
            try weights.head(rows: config.vocabSize, cols: config.hiddenSize),
            input: scratch.normed, inputOffset: 0, output: logits, outputOffset: 0,
            rows: config.vocabSize, cols: config.hiddenSize, in: commandBuffer)
        try encoder.softcapLogits(
            logits, size: config.vocabSize, cap: config.finalLogitSoftcapping, in: commandBuffer)
    }

    public func greedyToken(from distribution: UnsafeBufferPointer<Float>) -> Int {
        TokenSampler.greedyToken(from: distribution)
    }
}
