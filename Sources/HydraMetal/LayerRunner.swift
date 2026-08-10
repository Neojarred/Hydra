import Foundation
import HydraCore
import HydraFormat
import Metal

/// Working buffers reused from one token to the next.
///
/// Everything is allocated **once** at load time. Nothing is allocated during decoding: one
/// Metal allocation per token would cost more than the compute, and the footprint would stop
/// being predictable — which would ruin the one property the project has to demonstrate.
public final class DecodeScratch: @unchecked Sendable {

    public let hidden: MTLBuffer       // the residual state, [hiddenSize]
    public let normed: MTLBuffer       // [hiddenSize]
    public let query: MTLBuffer        // [qHeads * headDim]
    public let key: MTLBuffer          // [kvHeads * headDim]
    public let value: MTLBuffer        // [kvHeads * headDim]
    public let attention: MTLBuffer    // [qHeads * headDim]
    public let projected: MTLBuffer    // [hiddenSize]
    public let routerLogits: MTLBuffer // [expertCount]
    public let routerIndices: MTLBuffer  // [topK] UInt32
    public let routerWeights: MTLBuffer  // [topK] Float
    public let mixture: MTLBuffer      // [hiddenSize]
    public let gateUp: MTLBuffer       // [2 * intermediateSize]
    public let activated: MTLBuffer    // [intermediateSize]
    public let expertOutput: MTLBuffer // [hiddenSize]
    /// One slot per selected expert: [expertsPerToken × hiddenSize].
    public let expertSlices: MTLBuffer
    public let cosTable: MTLBuffer     // [headDim / 2]
    public let sinTable: MTLBuffer     // [headDim / 2]

    public let byteCount: Int

    public enum ScratchError: Error, CustomStringConvertible {
        case allocationFailed(String, bytes: Int)
        public var description: String {
            switch self {
            case let .allocationFailed(name, bytes):
                return "scratch: cannot allocate \(bytes) B for \"\(name)\""
            }
        }
    }

    public init(config: GptOssConfig, device: MTLDevice) throws {
        var total = 0
        func make(_ name: String, floats: Int) throws -> MTLBuffer {
            let bytes = max(floats * MemoryLayout<Float>.size, 16)
            guard let buffer = device.makeBuffer(length: bytes, options: .storageModeShared)
            else { throw ScratchError.allocationFailed(name, bytes: bytes) }
            total += bytes
            return buffer
        }

        let qDim = config.attentionHeadCount * config.headDim
        let kvDim = config.keyValueHeadCount * config.headDim

        hidden = try make("hidden", floats: config.hiddenSize)
        normed = try make("normed", floats: config.hiddenSize)
        query = try make("query", floats: qDim)
        key = try make("key", floats: kvDim)
        value = try make("value", floats: kvDim)
        attention = try make("attention", floats: qDim)
        projected = try make("projected", floats: config.hiddenSize)
        routerLogits = try make("routerLogits", floats: config.expertCount)
        routerIndices = try make("routerIndices", floats: config.expertsPerToken)
        routerWeights = try make("routerWeights", floats: config.expertsPerToken)
        mixture = try make("mixture", floats: config.hiddenSize)
        gateUp = try make("gateUp", floats: 2 * config.intermediateSize)
        activated = try make("activated", floats: config.intermediateSize)
        expertOutput = try make("expertOutput", floats: config.hiddenSize)
        expertSlices = try make(
            "expertSlices", floats: config.expertsPerToken * config.hiddenSize)
        cosTable = try make("cosTable", floats: config.headDim / 2)
        sinTable = try make("sinTable", floats: config.headDim / 2)

        byteCount = total
    }

    public func hiddenState() -> UnsafeMutableBufferPointer<Float> {
        UnsafeMutableBufferPointer(
            start: hidden.contents().bindMemory(to: Float.self, capacity: 1), count: 1)
    }
}

/// Runs one transformer layer on the GPU.
///
/// The graph is cut into two command buffers, and that cut is **imposed by the
/// architecture**, not chosen: the router produces expert identifiers on the GPU, and the
/// CPU must read them to know which blobs to load from SSD. No reordering escapes that
/// dependency.
///
/// ```
/// cb1 : norm → QKV → RoPE → KV write → attention → O projection → residual
///       → post-attention norm → router logits → top-k
/// I/O : read the identifiers, load the missing experts in parallel
/// cb2 : for each expert — gate_up → SwiGLU → down → weighted accumulation
///       → residual
/// ```
///
/// Since GPT-OSS **has no shared expert**, there is no dense branch to compute during the
/// reads. The overlap that hides latency in TurboFieldfare is therefore structurally absent
/// here — a documented limit, not an oversight.
public struct LayerRunner: Sendable {

    public let config: GptOssConfig
    public let encoder: ForwardEncoder
    public let mapping: ModelMapping
    public let cache: ExpertSlotCache

    public init(
        config: GptOssConfig, encoder: ForwardEncoder,
        mapping: ModelMapping, cache: ExpertSlotCache
    ) {
        self.config = config
        self.encoder = encoder
        self.mapping = mapping
        self.cache = cache
    }

    private var qDim: Int { config.attentionHeadCount * config.headDim }
    private var kvDim: Int { config.keyValueHeadCount * config.headDim }
    private var smScale: Float { 1.0 / Float(config.headDim).squareRoot() }

    private func tensor(_ suffix: String, layer: Int) throws -> (MTLBuffer, Int) {
        let (buffer, offset, _) = try mapping.residentTensor("model.layers.\(layer).\(suffix)")
        return (buffer, offset)
    }

    /// First half: everything that precedes knowing which experts are needed.
    public func encodeAttentionAndRouter(
        layer: Int, position: Int, scratch: DecodeScratch, kvCache: KVCache,
        in commandBuffer: MTLCommandBuffer
    ) throws {
        let ring = kvCache.layers[layer].ringSize

        // --- Input normalization ---
        let inputNorm = try tensor("input_layernorm.weight", layer: layer)
        try encoder.rmsNorm(
            input: scratch.hidden, scale: inputNorm.0, scaleOffset: inputNorm.1,
            output: scratch.normed, size: config.hiddenSize, eps: config.rmsNormEps,
            in: commandBuffer)

        // --- Projections Q, K, V ---
        for (suffix, output, rows) in [
            ("q_proj", scratch.query, qDim),
            ("k_proj", scratch.key, kvDim),
            ("v_proj", scratch.value, kvDim),
        ] {
            let weight = try tensor("self_attn.\(suffix).weight", layer: layer)
            let bias = try tensor("self_attn.\(suffix).bias", layer: layer)
            try encoder.denseProjection(
                weights: weight.0, weightsOffset: weight.1,
                bias: bias.0, biasOffset: bias.1,
                input: scratch.normed, inputOffset: 0,
                output: output, outputOffset: 0,
                rows: rows, cols: config.hiddenSize, in: commandBuffer)
        }

        // --- RoPE on Q and K, tables already carrying the YaRN concentration ---
        try encoder.applyRoPE(
            vector: scratch.query, vectorOffset: 0,
            cos: scratch.cosTable, sin: scratch.sinTable, tableOffset: 0,
            heads: config.attentionHeadCount, headDim: config.headDim, in: commandBuffer)
        try encoder.applyRoPE(
            vector: scratch.key, vectorOffset: 0,
            cos: scratch.cosTable, sin: scratch.sinTable, tableOffset: 0,
            heads: config.keyValueHeadCount, headDim: config.headDim, in: commandBuffer)

        // --- Write into the cache, then attention ---
        try encoder.writeKeyValue(
            key: scratch.key, keyOffset: 0, value: scratch.value, valueOffset: 0,
            keyCache: kvCache.layers[layer].keys, valueCache: kvCache.layers[layer].values,
            kvHeads: config.keyValueHeadCount, headDim: config.headDim,
            position: position, ringSize: ring, in: commandBuffer)

        let visible = kvCache.visibleRange(layer: layer, position: position)
        let sinks = try tensor("self_attn.sinks", layer: layer)
        try encoder.attention(
            query: scratch.query, queryOffset: 0,
            keyCache: kvCache.layers[layer].keys, valueCache: kvCache.layers[layer].values,
            sinks: sinks.0, sinksOffset: sinks.1,
            output: scratch.attention, outputOffset: 0,
            qHeads: config.attentionHeadCount, kvHeads: config.keyValueHeadCount,
            headDim: config.headDim, keyCount: visible.count,
            ringSize: ring, startPosition: visible.start, smScale: smScale,
            in: commandBuffer)

        // --- Output projection and residual ---
        let outWeight = try tensor("self_attn.o_proj.weight", layer: layer)
        let outBias = try tensor("self_attn.o_proj.bias", layer: layer)
        try encoder.denseProjection(
            weights: outWeight.0, weightsOffset: outWeight.1,
            bias: outBias.0, biasOffset: outBias.1,
            input: scratch.attention, inputOffset: 0,
            output: scratch.projected, outputOffset: 0,
            rows: config.hiddenSize, cols: qDim, in: commandBuffer)
        try encoder.addInPlace(
            target: scratch.hidden, targetOffset: 0,
            addend: scratch.projected, addendOffset: 0,
            size: config.hiddenSize, in: commandBuffer)

        // --- Normalisation post-attention et routeur ---
        let postNorm = try tensor("post_attention_layernorm.weight", layer: layer)
        try encoder.rmsNorm(
            input: scratch.hidden, scale: postNorm.0, scaleOffset: postNorm.1,
            output: scratch.normed, size: config.hiddenSize, eps: config.rmsNormEps,
            in: commandBuffer)

        let routerWeight = try tensor("mlp.router.weight", layer: layer)
        let routerBias = try tensor("mlp.router.bias", layer: layer)
        try encoder.denseProjection(
            weights: routerWeight.0, weightsOffset: routerWeight.1,
            bias: routerBias.0, biasOffset: routerBias.1,
            input: scratch.normed, inputOffset: 0,
            output: scratch.routerLogits, outputOffset: 0,
            rows: config.expertCount, cols: config.hiddenSize, in: commandBuffer)

        try encoder.routerTopK(
            logits: scratch.routerLogits, logitsOffset: 0,
            indices: scratch.routerIndices, weights: scratch.routerWeights,
            expertCount: config.expertCount, topK: config.expertsPerToken, in: commandBuffer)
    }

    /// The chosen expert identifiers, read after `cb1` has been committed.
    public func selectedExperts(_ scratch: DecodeScratch) -> [Int] {
        let raw = UnsafeBufferPointer(
            start: scratch.routerIndices.contents().bindMemory(
                to: UInt32.self, capacity: config.expertsPerToken),
            count: config.expertsPerToken)
        return raw.map(Int.init)
    }

    /// Zeroes the mixture accumulator.
    public func encodeMixtureStart(
        scratch: DecodeScratch, in commandBuffer: MTLCommandBuffer
    ) throws {
        try encoder.fillZero(
            scratch.mixture, offset: 0, size: config.hiddenSize, in: commandBuffer)
    }

    /// Adds the mixture residual to the hidden state.
    public func encodeMixtureEnd(
        scratch: DecodeScratch, in commandBuffer: MTLCommandBuffer
    ) throws {
        try encoder.addInPlace(
            target: scratch.hidden, targetOffset: 0,
            addend: scratch.mixture, addendOffset: 0,
            size: config.hiddenSize, in: commandBuffer)
    }

    /// A single expert, so its compute can be submitted as soon as it is loaded.
    public func encodeSingleExpert(
        layer: Int, expert: Int, weightIndex: Int,
        scratch: DecodeScratch, in commandBuffer: MTLCommandBuffer
    ) throws {
        let blob = config.expertBlobLayout
        // Blocks until **this** expert is ready, not until all of them are.
        let (buffer, blobOffset) = try cache.expert(layer: layer, expert: expert, pin: true)

        try encoder.expertProjection(
            blocks: buffer, blocksOffset: blobOffset + blob.gateUpBlocks.offset,
            scales: buffer, scalesOffset: blobOffset + blob.gateUpScales.offset,
            bias: buffer, biasOffset: blobOffset + blob.gateUpBias.offset,
            input: scratch.normed, inputOffset: 0,
            output: scratch.gateUp, outputOffset: 0,
            rows: 2 * config.intermediateSize, cols: config.hiddenSize, in: commandBuffer)
        try encoder.swiglu(
            input: scratch.gateUp, inputOffset: 0,
            output: scratch.activated, outputOffset: 0,
            size: config.intermediateSize,
            alpha: 1.702, limit: config.swigluLimit, in: commandBuffer)
        try encoder.expertProjection(
            blocks: buffer, blocksOffset: blobOffset + blob.downBlocks.offset,
            scales: buffer, scalesOffset: blobOffset + blob.downScales.offset,
            bias: buffer, biasOffset: blobOffset + blob.downBias.offset,
            input: scratch.activated, inputOffset: 0,
            output: scratch.expertOutput, outputOffset: 0,
            rows: config.hiddenSize, cols: config.intermediateSize, in: commandBuffer)
        try encoder.writeExpertScaled(
            into: scratch.expertSlices,
            outputOffset: weightIndex * config.hiddenSize * MemoryLayout<Float>.size,
            contribution: scratch.expertOutput, contributionOffset: 0,
            weights: scratch.routerWeights, weightIndex: weightIndex,
            size: config.hiddenSize, in: commandBuffer)
    }

    /// Sums the slots in slot order, then adds the residual.
    ///
    /// Here, and only here, is the addition order fixed — so it does not depend on the order
    /// in which the experts were computed, nor on the state of the cache.
    public func encodeCombineSlices(
        count: Int, scratch: DecodeScratch, in commandBuffer: MTLCommandBuffer
    ) throws {
        try encoder.sumExpertSlices(
            into: scratch.mixture, slices: scratch.expertSlices,
            size: config.hiddenSize, count: count, in: commandBuffer)
        try encoder.addInPlace(
            target: scratch.hidden, targetOffset: 0,
            addend: scratch.mixture, addendOffset: 0,
            size: config.hiddenSize, in: commandBuffer)
    }

    /// Second half: the selected experts are cached, compute can proceed.
    public func encodeMixtureOfExperts(
        layer: Int, experts: [Int], scratch: DecodeScratch,
        in commandBuffer: MTLCommandBuffer
    ) throws {
        let blob = config.expertBlobLayout
        try encoder.fillZero(
            scratch.mixture, offset: 0, size: config.hiddenSize, in: commandBuffer)

        for (slotIndex, expert) in experts.enumerated() {
            // Pinned: the pass encoded below will reference this buffer until the command
            // buffer executes, long after this call returns.
            let (buffer, blobOffset) = try cache.expert(layer: layer, expert: expert, pin: true)

            // gate_up : [2 × intermediate, hidden]
            try encoder.expertProjection(
                blocks: buffer, blocksOffset: blobOffset + blob.gateUpBlocks.offset,
                scales: buffer, scalesOffset: blobOffset + blob.gateUpScales.offset,
                bias: buffer, biasOffset: blobOffset + blob.gateUpBias.offset,
                input: scratch.normed, inputOffset: 0,
                output: scratch.gateUp, outputOffset: 0,
                rows: 2 * config.intermediateSize, cols: config.hiddenSize, in: commandBuffer)

            // SwiGLU: even/odd index split, asymmetric clamping, +1.
            try encoder.swiglu(
                input: scratch.gateUp, inputOffset: 0,
                output: scratch.activated, outputOffset: 0,
                size: config.intermediateSize,
                alpha: 1.702, limit: config.swigluLimit, in: commandBuffer)

            // down : [hidden, intermediate]
            try encoder.expertProjection(
                blocks: buffer, blocksOffset: blobOffset + blob.downBlocks.offset,
                scales: buffer, scalesOffset: blobOffset + blob.downScales.offset,
                bias: buffer, biasOffset: blobOffset + blob.downBias.offset,
                input: scratch.activated, inputOffset: 0,
                output: scratch.expertOutput, outputOffset: 0,
                rows: config.hiddenSize, cols: config.intermediateSize, in: commandBuffer)

            try encoder.accumulateExpert(
                into: scratch.mixture, outputOffset: 0,
                contribution: scratch.expertOutput, contributionOffset: 0,
                weights: scratch.routerWeights, weightIndex: slotIndex,
                size: config.hiddenSize, in: commandBuffer)
        }

        try encoder.addInPlace(
            target: scratch.hidden, targetOffset: 0,
            addend: scratch.mixture, addendOffset: 0,
            size: config.hiddenSize, in: commandBuffer)
    }
}
