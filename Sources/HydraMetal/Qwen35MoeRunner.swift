import Foundation
import HydraCore
import HydraFormat
import Metal

/// Runs Qwen 3.5/3.6 MoE, one token at a time.
///
/// The third sibling of `ModelRunner` and `Gemma4ModelRunner`, and the one that differs most:
/// three layers in four have no key/value cache at all, and carry a fixed recurrent state
/// instead. `RecurrentStateCache` holds that state, `KVCache` holds histories for the other ten
/// layers and nothing for the thirty, and both are indexed by the layer's own number so no
/// caller translates between two numbering schemes.
///
/// Prefill goes through `QwenPrefillRunner`, layer-major and chunked, so a layer's experts are
/// read once a chunk instead of once a token. Token by token a prompt moves 540 MiB from SSD
/// for each of its tokens; grouped, a chunk reads at most the whole 16.9 GiB pool once.
public final class Qwen35MoeRunner: @unchecked Sendable {

    public let config: Qwen35MoeConfig
    public let context: MetalContext
    public let mapping: ModelMapping
    public let expertCache: ExpertSlotCache
    public let kvCache: KVCache
    /// The linear layers' whole memory. Fixed size, and it does not grow with the context.
    public let state: RecurrentStateCache

    private let encoder: ForwardEncoder
    private let layerRunner: QwenLayerRunner
    private let weights: Qwen35MoeWeights
    /// Resolved once, when the runner is built. A decoding step never looks up a tensor name.
    private let layerWeights: [QwenLayerRunner.LayerWeights]
    private let scratch: QwenLayerRunner.Scratch
    /// Built once, because prompt processing is where a conversation spends its visible time.
    private let prefillRunner: QwenPrefillRunner

    private let hidden: MTLBuffer
    private let normed: MTLBuffer
    private let logits: MTLBuffer
    private let cosTable: MTLBuffer
    private let sinTable: MTLBuffer
    /// An unreachable sink a head, this project's way of saying "no learned sink".
    private let sinks: MTLBuffer
    /// `1 / theta^(2i/headDim)` for the rotating pairs, zero for the rest.
    private let inverseFrequencies: [Double]

    public private(set) var position = 0
    public private(set) var lastTimings = ModelRunner.Timings()

    /// Seconds the GPU actually spent executing the last token's command buffers.
    ///
    /// Wall time minus this is the CPU's share: encoding, `pread`, and waiting. The one
    /// measurement that says whether a kernel is slow or the GPU is simply idle.
    public private(set) var lastGPUSeconds = 0.0

    public enum RunnerError: Error, CustomStringConvertible {
        case contextExhausted(position: Int, capacity: Int)

        public var description: String {
            switch self {
            case let .contextExhausted(position, capacity):
                return "context exhausted: position \(position) of \(capacity)"
            }
        }
    }

    public init(
        config: Qwen35MoeConfig, context: MetalContext, mapping: ModelMapping,
        expertCache: ExpertSlotCache, contextLength: Int,
        prefillChunk: Int = QwenPrefillRunner.chunk
    ) throws {
        self.config = config
        self.context = context
        self.mapping = mapping
        self.expertCache = expertCache
        self.encoder = ForwardEncoder(context: context)
        self.kvCache = try KVCache(
            model: config, contextLength: contextLength, device: context.device)
        self.state = try RecurrentStateCache(
            model: config,
            geometry: RecurrentGeometry(
                valueHeads: config.linearValueHeads, keyHeads: config.linearKeyHeads,
                keyDim: config.linearKeyHeadDim, valueDim: config.linearValueHeadDim,
                convDim: config.linearConvDim, convKernel: config.linearConvKernel),
            device: context.device)
        self.layerRunner = QwenLayerRunner(
            config: config, encoder: ForwardEncoder(context: context), context: context)
        self.scratch = try QwenLayerRunner.Scratch(config: config, device: context.device)

        let source = Qwen35MoeWeights(config: config, mapping: mapping)
        self.weights = source
        self.layerWeights = try (0..<config.layerCount).map {
            try source.layerWeights(layer: $0)
        }

        func make(_ floats: Int, _ name: String) throws -> MTLBuffer {
            guard let buffer = context.device.makeBuffer(
                length: max(floats, 1) * 4, options: .storageModeShared)
            else { throw ModelRunner.RunnerError.allocationFailed(name) }
            return buffer
        }
        self.prefillRunner = try QwenPrefillRunner(
            config: config, encoder: encoder, weights: source, layerWeights: layerWeights,
            device: context.device)
        self.hidden = try make(config.hiddenSize, "qwen.hidden")
        self.normed = try make(config.hiddenSize, "qwen.normed")
        self.logits = try make(config.vocabSize, "qwen.logits")
        self.cosTable = try make(config.headDim / 2, "qwen.cos")
        self.sinTable = try make(config.headDim / 2, "qwen.sin")

        // BF16, because the attention kernel reads its sinks as BF16. Unreachably low, so the
        // running maximum a sink seeds never wins and no probability mass lands on it.
        let sinkBits = [UInt16](
            repeating: BF16.fromFloat(-1e30), count: max(config.attentionHeadCount, 1))
        guard let sinks = sinkBits.withUnsafeBytes({
            context.device.makeBuffer(
                bytes: $0.baseAddress!, length: max($0.count, 4), options: .storageModeShared)
        }) else { throw ModelRunner.RunnerError.allocationFailed("qwen.sinks") }
        self.sinks = sinks

        // Partial rotary: only the first quarter of a head turns, and the rest is left alone.
        // Expressed as zero inverse frequencies rather than a shorter table, so the kernel has
        // no special case and cannot rotate a dimension it should not (D-027).
        let rotatingPairs = Int(Double(config.headDim) * config.partialRotaryFactor) / 2
        self.inverseFrequencies = (0..<(config.headDim / 2)).map { index in
            index < rotatingPairs
                ? 1.0 / pow(config.ropeTheta, Double(2 * index) / Double(config.headDim))
                : 0
        }
    }

    public var reservedBytes: Int {
        expertCache.reservedBytes + kvCache.byteCount + state.byteCount
            + logits.length + hidden.length + normed.length + prefillRunner.byteCount
    }

    // MARK: - State

    public func reset() {
        position = 0
        kvCache.reset()
        state.reset()
        expertCache.resetStatistics()
    }

    public var canRewind: Bool { kvCache.canRewind && state.canRewind(to: position) }

    /// Both memories have to agree, and the recurrent one is the stricter.
    ///
    /// A KV cache can be truncated to any position it still holds. A decayed running sum cannot
    /// be un-summed, so the recurrent state can only return to a position it checkpointed. The
    /// conjunction is not a conservative guess: rewinding one and not the other would answer
    /// from an attention history and a recurrent state describing different prefixes.
    public func canRewind(to tokens: Int) -> Bool {
        tokens <= position && kvCache.canRewind(to: tokens) && state.canRewind(to: tokens)
    }

    public func rewind(to tokens: Int) {
        guard canRewind(to: tokens) else { return }
        kvCache.rewind(to: tokens)
        try? state.rewind(to: tokens)
        position = tokens
    }

    // MARK: - Forward

    private func commandBuffer() throws -> MTLCommandBuffer {
        guard let buffer = context.commandQueue.makeCommandBuffer() else {
            throw ModelRunner.RunnerError.allocationFailed("command buffer")
        }
        return buffer
    }

    /// The rotary tables at one position. One table for the whole model: unlike Gemma, every
    /// attending layer here shares a single theta and a single rotating fraction.
    private func writeRopeTables(position: Int) {
        let pairs = config.headDim / 2
        let cos = cosTable.contents().bindMemory(to: Float.self, capacity: pairs)
        let sin = sinTable.contents().bindMemory(to: Float.self, capacity: pairs)
        for i in 0..<pairs {
            let angle = Double(position) * inverseFrequencies[i]
            cos[i] = Float(Foundation.cos(angle))
            sin[i] = Float(Foundation.sin(angle))
        }
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

        // The embedding, read straight in. **No scale**: Gemma multiplies by `sqrt(hidden)`
        // and this model does not, and applying it here would inflate every activation by 45.
        let row = UnsafeMutableBufferPointer(
            start: hidden.contents().bindMemory(to: Float.self, capacity: config.hiddenSize),
            count: config.hiddenSize)
        weights.readEmbedding(token: token, into: row)

        writeRopeTables(position: position)

        // One command buffer a layer instead of two.
        //
        // Only one of the two waits earns its place: the router picks the experts and the CPU
        // cannot `pread` them until it has read that choice back. The second was never
        // synchronizing anything the GPU needed. Layer N's experts and layer N+1's mixer are
        // sequentially dependent, but nothing between them requires the CPU, so they belong in
        // one buffer, where the shared serial encoder already orders them and makes the
        // mixture's writes visible to the mixer that reads them.
        //
        // Measured at 44 % GPU-busy before this, which is what says the round trips are the
        // cost rather than the kernels (M-053). Gemma took the same step for a measured +8 %
        // (M-042).
        var gpuSeconds = 0.0
        var pending: MTLCommandBuffer? = nil
        var pinnedLayer: Int? = nil

        for layer in 0..<config.layerCount {
            var start = Date()
            let buffer: MTLCommandBuffer
            if let merged = pending {
                buffer = merged
                pending = nil
            } else {
                buffer = try commandBuffer()
                try layerRunner.encodeMixerAndRouter(
                    layer, hidden: hidden, weights: layerWeights[layer], scratch: scratch,
                    kvCache: kvCache, state: state, position: position,
                    cos: cosTable, sin: sinTable, sinks: sinks, in: buffer)
            }
            context.commit(buffer)
            try context.wait(buffer)
            gpuSeconds += buffer.gpuEndTime - buffer.gpuStartTime
            timings.attentionAndRouter += Date().timeIntervalSince(start)

            // The previous layer's experts finished inside the buffer just waited on.
            if let pinned = pinnedLayer {
                expertCache.release(layer: pinned)
                pinnedLayer = nil
            }

            // The only unavoidable synchronization: the CPU must know which experts were
            // chosen before it can read them.
            let selected = layerRunner.selectedExperts(scratch)

            start = Date()
            try expertCache.load(layer: layer, experts: selected)
            timings.expertIO += Date().timeIntervalSince(start)

            start = Date()
            let next = try commandBuffer()
            let experts = try selected.map { expert -> QwenMixtureBlock.Expert in
                let (blob, offset) = try expertCache.expert(
                    layer: layer, expert: expert, pin: true)
                return weights.expert(blob: blob, offset: offset)
            }
            try layerRunner.encodeExperts(
                experts, hidden: hidden, scratch: scratch, in: next)
            pinnedLayer = layer

            if layer + 1 < config.layerCount {
                try layerRunner.encodeMixerAndRouter(
                    layer + 1, hidden: hidden, weights: layerWeights[layer + 1],
                    scratch: scratch, kvCache: kvCache, state: state, position: position,
                    cos: cosTable, sin: sinTable, sinks: sinks, in: next)
                pending = next
            } else {
                context.commit(next)
                try context.wait(next)
                gpuSeconds += next.gpuEndTime - next.gpuStartTime
                expertCache.release(layer: layer)
                pinnedLayer = nil
            }
            timings.mixture += Date().timeIntervalSince(start)
        }

        // Both memories advance. `KVCache.advance` skips the recurrent layers itself, and the
        // recurrent cache carries only a position: its state was already advanced in place by
        // each layer's delta-rule step.
        try kvCache.advance()
        state.advance(by: 1)
        position += 1

        guard needsLogits else {
            lastTimings = timings
            return UnsafeBufferPointer(start: nil, count: 0)
        }

        let start = Date()
        let head = try commandBuffer()
        try encodeHead(in: head)
        context.commit(head)
        try context.wait(head)
        gpuSeconds += head.gpuEndTime - head.gpuStartTime
        timings.head = Date().timeIntervalSince(start)

        lastTimings = timings
        lastGPUSeconds = gpuSeconds
        return UnsafeBufferPointer(
            start: logits.contents().bindMemory(to: Float.self, capacity: config.vocabSize),
            count: config.vocabSize)
    }

    /// The final norm and the untied head. **No softcap**: Gemma caps its logits and this model
    /// does not, and a cap applied here would flatten the distribution at every position.
    private func encodeHead(in commandBuffer: MTLCommandBuffer) throws {
        let finalNorm = try weights.finalNorm()
        try encoder.rmsNorm(
            input: hidden, scale: finalNorm.buffer, scaleOffset: finalNorm.offset,
            output: normed, size: config.hiddenSize, eps: config.rmsNormEps,
            in: commandBuffer)
        try encoder.encodeProjection(
            try weights.head(),
            input: normed, inputOffset: 0, output: logits, outputOffset: 0,
            rows: config.vocabSize, cols: config.hiddenSize, in: commandBuffer)
    }

    /// Processes a prompt, chunked and layer-major, and checkpoints the state at the end.
    ///
    /// The checkpoint is what makes the next turn reusable: without one the recurrent state can
    /// never go back, and a follow-up question reprocesses the whole conversation.
    ///
    /// A single token still goes through `forward`. There is nothing to group, and the chunk
    /// buffers would be staged for one row.
    @discardableResult
    public func prefill(tokens: [Int]) throws -> UnsafeBufferPointer<Float> {
        guard tokens.count > 1 else {
            if tokens.isEmpty { return UnsafeBufferPointer(start: nil, count: 0) }
            let result = try forward(token: tokens[0])
            state.checkpoint()
            return result
        }
        guard position + tokens.count <= kvCache.contextLength else {
            throw RunnerError.contextExhausted(
                position: position + tokens.count, capacity: kvCache.contextLength)
        }

        var timings = ModelRunner.Timings()
        var offset = 0
        var lastChunk = 0
        while offset < tokens.count {
            let end = min(offset + prefillRunner.chunkTokens, tokens.count)
            let slice = Array(tokens[offset..<end])

            try prefillRunner.run(
                tokenCount: slice.count, firstPosition: position,
                embeddings: { [self] index, row in
                    weights.readEmbedding(token: slice[index], into: row)
                },
                kvCache: kvCache, state: state, expertCache: expertCache,
                inverseFrequencies: inverseFrequencies,
                commandBuffer: commandBuffer, timings: &timings)

            for _ in slice { try kvCache.advance() }
            state.advance(by: slice.count)
            position += slice.count
            lastChunk = slice.count
            offset = end
        }

        // The head reads the last token's state, which the chunk left in the batch buffer.
        let start = Date()
        let head = try commandBuffer()
        try prefillRunner.copyLastRow(tokenCount: lastChunk, into: hidden, in: head)
        try encodeHead(in: head)
        context.commit(head)
        try context.wait(head)
        timings.head = Date().timeIntervalSince(start)

        state.checkpoint()
        lastTimings = timings
        return UnsafeBufferPointer(
            start: logits.contents().bindMemory(to: Float.self, capacity: config.vocabSize),
            count: config.vocabSize)
    }

    // MARK: - Sampling

    private var sampler = TokenSampler()

    public func sample(
        from distribution: UnsafeBufferPointer<Float>, using sampling: ModelRunner.Sampling
    ) -> Int {
        sampler.sample(from: distribution, using: sampling)
    }

    public func resetSampling() { sampler.reset() }

    public func greedyToken(from distribution: UnsafeBufferPointer<Float>) -> Int {
        TokenSampler.greedyToken(from: distribution)
    }

    // MARK: - Verification

    private var batchLogits: MTLBuffer?

    /// Per-position logits, computed sequentially, as `Gemma4ModelRunner` does.
    ///
    /// A batched form would need a batched delta rule, which is the same missing piece prefill
    /// wants. Until it exists, verifying `n` tokens costs exactly what decoding `n` costs.
    public func verify(tokens: [Int]) throws -> [UnsafeBufferPointer<Float>] {
        precondition(!tokens.isEmpty, "nothing to verify")
        let bytes = tokens.count * config.vocabSize * MemoryLayout<Float>.size
        if batchLogits == nil || batchLogits!.length < bytes {
            guard let buffer = context.device.makeBuffer(
                length: bytes, options: .storageModeShared)
            else { throw ModelRunner.RunnerError.allocationFailed("batchLogits") }
            batchLogits = buffer
        }
        guard let output = batchLogits else {
            throw ModelRunner.RunnerError.allocationFailed("batchLogits")
        }
        let base = output.contents().bindMemory(
            to: Float.self, capacity: tokens.count * config.vocabSize)
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

    /// One decoding step; the draft is ignored, for the reason `Gemma4ModelRunner` gives.
    /// Returning a single token is within the contract, since one token is always a correct
    /// prefix of any draft.
    public func step(
        from distribution: UnsafeBufferPointer<Float>, draft: [Int],
        sampling: ModelRunner.Sampling
    ) throws -> (tokens: [Int], next: UnsafeBufferPointer<Float>) {
        let token = sample(from: distribution, using: sampling)
        return ([token], try forward(token: token))
    }
}

extension Qwen35MoeRunner: TextModelRunner {
    public var architecture: ModelArchitecture { config.architecture }
}
