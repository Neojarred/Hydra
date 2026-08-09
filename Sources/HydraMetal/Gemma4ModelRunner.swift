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
/// Deliberately simpler than `ModelRunner` for now. It has no speculative decoding, no
/// batched prefill and no read/compute overlap: those were each added to `ModelRunner` on the
/// back of a measurement, and adding them here before Gemma has produced a single correct
/// token would be optimizing something unproven. `prefill` therefore loops over `forward`.
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

    public private(set) var position = 0
    public private(set) var lastTimings = ModelRunner.Timings()

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

    public init(
        config: Gemma4Config, context: MetalContext, mapping: ModelMapping,
        expertCache: ExpertSlotCache, contextLength: Int
    ) throws {
        self.config = config
        self.context = context
        self.mapping = mapping
        self.expertCache = expertCache
        self.encoder = ForwardEncoder(context: context)
        self.scratch = try Gemma4DecodeScratch(config: config, device: context.device)
        self.kvCache = try KVCache(
            model: config, contextLength: contextLength, device: context.device)
        self.layerRunner = Gemma4LayerRunner(
            config: config, encoder: encoder, mapping: mapping)
        self.ropeTables = (0..<config.layerCount).map {
            Gemma4RoPETables(config: config, layer: $0)
        }
        guard let logits = context.device.makeBuffer(
            length: config.vocabSize * MemoryLayout<Float>.size, options: .storageModeShared)
        else { throw RunnerError.allocationFailed("logits") }
        self.logits = logits
    }

    public var reservedBytes: Int {
        expertCache.reservedBytes + kvCache.byteCount + scratch.byteCount + logits.length
    }

    // MARK: - State

    public func reset() {
        position = 0
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

        // --- Embedding, scaled by sqrt(hiddenSize) ---
        //
        // The scale is a constant in the model code and appears nowhere in the checkpoint.
        // Omitting it mis-scales the entire forward pass (D-022).
        let hidden = UnsafeMutableBufferPointer(
            start: scratch.hidden.contents().bindMemory(
                to: Float.self, capacity: config.hiddenSize),
            count: config.hiddenSize)
        mapping.readEmbedding(token: token, into: hidden)
        let scale = config.embeddingScale
        for i in 0..<config.hiddenSize { hidden[i] *= scale }

        for layer in 0..<config.layerCount {
            writeRopeTables(layer: layer, position: position)

            var start = Date()
            let first = try commandBuffer()
            try layerRunner.encodeAttentionAndRouter(
                layer: layer, position: position, scratch: scratch, kvCache: kvCache, in: first)
            let routerScale = try mapping.residentTensor(
                "model.language_model.layers.\(layer).router.per_expert_scale")
            try encoder.gemmaRouterTopK(
                logits: scratch.routerLogits,
                perExpertScale: routerScale.buffer, perExpertScaleOffset: routerScale.offset,
                indices: scratch.routerIndices, weights: scratch.routerWeights,
                expertCount: config.expertCount, topK: config.expertsPerToken, in: first)
            first.commit()
            first.waitUntilCompleted()
            timings.attentionAndRouter += Date().timeIntervalSince(start)

            // The only unavoidable synchronization point: the CPU must know which experts
            // were chosen before it can read them.
            let selected = selectedExperts()

            start = Date()
            try expertCache.load(layer: layer, experts: selected)
            timings.expertIO += Date().timeIntervalSince(start)

            start = Date()
            let second = try commandBuffer()
            try layerRunner.encodeMixtureStart(scratch: scratch, in: second)
            for (slot, expert) in selected.enumerated() {
                let (buffer, _) = try expertCache.expert(layer: layer, expert: expert, pin: true)
                try layerRunner.encodeSingleExpert(
                    buffer: buffer, weightIndex: slot, scratch: scratch, in: second)
            }
            try layerRunner.encodeCombineBranches(
                layer: layer, count: selected.count, scratch: scratch, in: second)
            second.commit()
            second.waitUntilCompleted()
            expertCache.release(layer: layer)
            timings.mixture += Date().timeIntervalSince(start)
        }

        position += 1

        guard needsLogits else {
            lastTimings = timings
            return UnsafeBufferPointer(start: nil, count: 0)
        }

        // --- Final norm, the tied head, and softcapping ---
        let start = Date()
        let head = try commandBuffer()
        let finalNorm = try mapping.residentTensor("model.language_model.norm.weight")
        try encoder.rmsNorm(
            input: scratch.hidden, scale: finalNorm.buffer, scaleOffset: finalNorm.offset,
            output: scratch.normed, size: config.hiddenSize, eps: config.rmsNormEps, in: head)

        // `tie_word_embeddings`: the embedding **is** the output projection. There is no
        // `lm_head` tensor to look for.
        let embedding = try mapping.residentTensor(
            "model.language_model.embed_tokens.weight")
        try encoder.denseProjection(
            weights: embedding.buffer, weightsOffset: embedding.offset,
            bias: nil, biasOffset: 0,
            input: scratch.normed, inputOffset: 0, output: logits, outputOffset: 0,
            rows: config.vocabSize, cols: config.hiddenSize, in: head)
        try encoder.softcapLogits(
            logits, size: config.vocabSize, cap: config.finalLogitSoftcapping, in: head)
        head.commit()
        head.waitUntilCompleted()
        timings.head = Date().timeIntervalSince(start)

        lastTimings = timings
        return UnsafeBufferPointer(
            start: logits.contents().bindMemory(to: Float.self, capacity: config.vocabSize),
            count: config.vocabSize)
    }

    /// Processes a prompt, returning the distribution that follows it.
    ///
    /// One token at a time. `ModelRunner`'s batched prefill exists because measurement showed
    /// it worth 39 s on a thousand tokens; reproducing it here before Gemma has produced a
    /// correct token would be optimizing an unproven path.
    @discardableResult
    public func prefill(tokens: [Int]) throws -> UnsafeBufferPointer<Float> {
        var distribution = UnsafeBufferPointer<Float>(start: nil, count: 0)
        for (index, token) in tokens.enumerated() {
            distribution = try forward(token: token, needsLogits: index == tokens.count - 1)
        }
        return distribution
    }

    public func greedyToken(from distribution: UnsafeBufferPointer<Float>) -> Int {
        var best = 0
        var bestValue = -Float.greatestFiniteMagnitude
        for (index, value) in distribution.enumerated() where value > bestValue {
            bestValue = value
            best = index
        }
        return best
    }
}
