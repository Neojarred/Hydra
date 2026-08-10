import Foundation
import HydraCore
import HydraFormat
import Metal

/// Buffers for one prefill chunk, allocated once.
///
/// For a 128-token chunk of the 20B, the whole set weighs about ten megabytes — against the
/// 1.18 GiB of the expert cache. Chunked prefill therefore does not move the memory budget;
/// it only changes the order of the reads.
public final class PrefillScratch: @unchecked Sendable {

    public let maximumTokens: Int

    public let hidden: MTLBuffer        // [N][hidden]
    public let normed: MTLBuffer        // [N][hidden]
    public let query: MTLBuffer         // [N][qDim]
    public let key: MTLBuffer           // [N][kvDim]
    public let value: MTLBuffer         // [N][kvDim]
    public let attention: MTLBuffer     // [N][qDim]
    public let projected: MTLBuffer     // [N][hidden]
    public let routerLogits: MTLBuffer  // [N][experts]
    public let routerIndices: MTLBuffer // [N][topK] UInt32
    public let routerWeights: MTLBuffer // [N][topK]
    public let mixture: MTLBuffer       // [N][hidden]
    public let gateUp: MTLBuffer        // [N][2 * inter]
    public let activated: MTLBuffer     // [N][inter]
    public let expertOutput: MTLBuffer  // [N][hidden]
    public let cosTable: MTLBuffer      // [N][headDim/2]
    public let sinTable: MTLBuffer
    /// The tokens assigned to each expert of a tile, and their routing weights.
    ///
    /// **One region per expert of the tile**, not a single shared one. Passes are encoded
    /// before they run: if the CPU rewrote the same region for the next expert, the already
    /// encoded pass of the previous one would read the wrong indices. It is the same class
    /// of error as evicting a slot still referenced, and it shows up the same way — wrong
    /// results with no error at all.
    public let gatherIndices: MTLBuffer  // [experts][N] UInt32
    public let gatherWeights: MTLBuffer  // [experts][N]
    /// Entries reserved for each expert in the buffers above.
    public let gatherStride: Int
    /// Identity indices, for passes working on data that is already compacted.
    public let identityIndices: MTLBuffer

    public let byteCount: Int

    public init(config: GptOssConfig, maximumTokens: Int, device: MTLDevice) throws {
        self.maximumTokens = maximumTokens
        var total = 0
        func make(_ name: String, floats: Int) throws -> MTLBuffer {
            let bytes = max(floats * MemoryLayout<Float>.size, 16)
            guard let buffer = device.makeBuffer(length: bytes, options: .storageModeShared)
            else { throw DecodeScratch.ScratchError.allocationFailed(name, bytes: bytes) }
            total += bytes
            return buffer
        }

        let n = maximumTokens
        let qDim = config.attentionHeadCount * config.headDim
        let kvDim = config.keyValueHeadCount * config.headDim

        hidden = try make("hidden", floats: n * config.hiddenSize)
        normed = try make("normed", floats: n * config.hiddenSize)
        query = try make("query", floats: n * qDim)
        key = try make("key", floats: n * kvDim)
        value = try make("value", floats: n * kvDim)
        attention = try make("attention", floats: n * qDim)
        projected = try make("projected", floats: n * config.hiddenSize)
        routerLogits = try make("routerLogits", floats: n * config.expertCount)
        routerIndices = try make("routerIndices", floats: n * config.expertsPerToken)
        routerWeights = try make("routerWeights", floats: n * config.expertsPerToken)
        mixture = try make("mixture", floats: n * config.hiddenSize)
        gateUp = try make("gateUp", floats: n * 2 * config.intermediateSize)
        activated = try make("activated", floats: n * config.intermediateSize)
        expertOutput = try make("expertOutput", floats: n * config.hiddenSize)
        cosTable = try make("cosTable", floats: n * config.headDim / 2)
        sinTable = try make("sinTable", floats: n * config.headDim / 2)
        // One region per expert: a few tens of KiB in total, negligible.
        gatherStride = n
        gatherIndices = try make("gatherIndices", floats: config.expertCount * n)
        gatherWeights = try make("gatherWeights", floats: config.expertCount * n)
        identityIndices = try make("identityIndices", floats: n)

        // The identity indices never change: we fill them once.
        let identity = identityIndices.contents().bindMemory(to: UInt32.self, capacity: n)
        for i in 0..<n { identity[i] = UInt32(i) }

        byteCount = total
    }
}

/// Runs a chunk of tokens through one layer.
///
/// The difference from `LayerRunner` lies entirely in the weights being read **once for the
/// whole chunk**. On a 78-token prompt of the 20B, that takes dense-weight re-reads from
/// 92.9 GiB down to 1.2 GiB, for a strictly identical computation.
///
/// The expert mixture proceeds **expert by expert**: for each one we gather the tokens the
/// router assigned to it, compute, and scatter. Only one expert slot is occupied at a time,
/// exactly as in decoding — which is what makes chunked prefill cost no memory.
public struct PrefillRunner: Sendable {

    public let config: GptOssConfig
    public let encoder: BatchEncoder
    public let mapping: ModelMapping
    public let cache: ExpertSlotCache

    public init(
        config: GptOssConfig, encoder: BatchEncoder,
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

    /// First half: attention and router, for the whole chunk.
    public func encodeAttentionAndRouter(
        layer: Int, tokens: Int, firstPosition: Int,
        scratch: PrefillScratch, kvCache: KVCache,
        in commandBuffer: MTLCommandBuffer
    ) throws {
        let ring = kvCache.layers[layer].ringSize
        let sliding = config.attentionPattern(atLayer: layer) == .sliding

        let inputNorm = try tensor("input_layernorm.weight", layer: layer)
        try encoder.rmsNorm(
            input: scratch.hidden, scale: inputNorm.0, scaleOffset: inputNorm.1,
            output: scratch.normed, size: config.hiddenSize, tokens: tokens,
            eps: config.rmsNormEps, in: commandBuffer)

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
                input: scratch.normed, output: output,
                rows: rows, cols: config.hiddenSize, tokens: tokens, in: commandBuffer)
        }

        try encoder.applyRoPE(
            vector: scratch.query, cos: scratch.cosTable, sin: scratch.sinTable,
            heads: config.attentionHeadCount, headDim: config.headDim, tokens: tokens,
            in: commandBuffer)
        try encoder.applyRoPE(
            vector: scratch.key, cos: scratch.cosTable, sin: scratch.sinTable,
            heads: config.keyValueHeadCount, headDim: config.headDim, tokens: tokens,
            in: commandBuffer)

        try encoder.writeKeyValue(
            key: scratch.key, value: scratch.value,
            keyCache: kvCache.layers[layer].keys, valueCache: kvCache.layers[layer].values,
            kvHeads: config.keyValueHeadCount, headDim: config.headDim,
            tokens: tokens, firstPosition: firstPosition, ringSize: ring, in: commandBuffer)

        let sinks = try tensor("self_attn.sinks", layer: layer)
        try encoder.attention(
            query: scratch.query,
            keyCache: kvCache.layers[layer].keys, valueCache: kvCache.layers[layer].values,
            sinks: sinks.0, sinksOffset: sinks.1, output: scratch.attention,
            qHeads: config.attentionHeadCount, kvHeads: config.keyValueHeadCount,
            headDim: config.headDim, tokens: tokens,
            ringSize: ring, firstPosition: firstPosition,
            slidingWindow: sliding ? config.slidingWindow : 0,
            smScale: smScale, in: commandBuffer)

        let outWeight = try tensor("self_attn.o_proj.weight", layer: layer)
        let outBias = try tensor("self_attn.o_proj.bias", layer: layer)
        try encoder.denseProjection(
            weights: outWeight.0, weightsOffset: outWeight.1,
            bias: outBias.0, biasOffset: outBias.1,
            input: scratch.attention, output: scratch.projected,
            rows: config.hiddenSize, cols: qDim, tokens: tokens, in: commandBuffer)
        try encoder.addInPlace(
            target: scratch.hidden, addend: scratch.projected,
            size: tokens * config.hiddenSize, in: commandBuffer)

        let postNorm = try tensor("post_attention_layernorm.weight", layer: layer)
        try encoder.rmsNorm(
            input: scratch.hidden, scale: postNorm.0, scaleOffset: postNorm.1,
            output: scratch.normed, size: config.hiddenSize, tokens: tokens,
            eps: config.rmsNormEps, in: commandBuffer)

        let routerWeight = try tensor("mlp.router.weight", layer: layer)
        let routerBias = try tensor("mlp.router.bias", layer: layer)
        try encoder.denseProjection(
            weights: routerWeight.0, weightsOffset: routerWeight.1,
            bias: routerBias.0, biasOffset: routerBias.1,
            input: scratch.normed, output: scratch.routerLogits,
            rows: config.expertCount, cols: config.hiddenSize, tokens: tokens, in: commandBuffer)

        try encoder.routerTopK(
            logits: scratch.routerLogits, indices: scratch.routerIndices,
            weights: scratch.routerWeights, expertCount: config.expertCount,
            topK: config.expertsPerToken, tokens: tokens, in: commandBuffer)
    }

    /// The tokens assigned to each expert, read after the first half has been committed.
    public struct Assignment: Sendable {
        public let expert: Int
        public let rows: [UInt32]
        public let weights: [Float]
    }

    public func assignments(_ scratch: PrefillScratch, tokens: Int) -> [Assignment] {
        let topK = config.expertsPerToken
        let indices = scratch.routerIndices.contents().bindMemory(
            to: UInt32.self, capacity: tokens * topK)
        let weights = scratch.routerWeights.contents().bindMemory(
            to: Float.self, capacity: tokens * topK)

        var rowsByExpert: [Int: [(UInt32, Float)]] = [:]
        for token in 0..<tokens {
            for k in 0..<topK {
                let expert = Int(indices[token * topK + k])
                rowsByExpert[expert, default: []].append(
                    (UInt32(token), weights[token * topK + k]))
            }
        }
        // Deterministic order: two identical runs must produce the same outputs, and the
        // accumulation order affects the last bits.
        return rowsByExpert.keys.sorted().map { expert in
            let entries = rowsByExpert[expert]!
            return Assignment(
                expert: expert, rows: entries.map(\.0), weights: entries.map(\.1))
        }
    }

    /// Second half: one expert at a time, over its own tokens only.
    /// - Parameter slot: the expert's rank within the current tile. Determines which region
    ///   of the index buffers is reserved for it — two experts encoded into the same command
    ///   buffer must not share one.
    public func encodeExpert(
        layer: Int, assignment: Assignment, slot: Int, scratch: PrefillScratch,
        in commandBuffer: MTLCommandBuffer
    ) throws {
        let blob = config.expertBlobLayout
        let count = assignment.rows.count
        guard count > 0 else { return }

        let base = slot * scratch.gatherStride
        let capacity = config.expertCount * scratch.gatherStride
        let indices = scratch.gatherIndices.contents().bindMemory(
            to: UInt32.self, capacity: capacity)
        let weights = scratch.gatherWeights.contents().bindMemory(
            to: Float.self, capacity: capacity)
        for (index, row) in assignment.rows.enumerated() { indices[base + index] = row }
        for (index, weight) in assignment.weights.enumerated() { weights[base + index] = weight }
        let byteOffset = base * MemoryLayout<UInt32>.size

        let (buffer, blobOffset) = try cache.expert(
            layer: layer, expert: assignment.expert, pin: true)

        try encoder.expertProjection(
            blocks: buffer, blocksOffset: blobOffset + blob.gateUpBlocks.offset,
            scales: buffer, scalesOffset: blobOffset + blob.gateUpScales.offset,
            bias: buffer, biasOffset: blobOffset + blob.gateUpBias.offset,
            input: scratch.normed, output: scratch.gateUp,
            rowIndices: scratch.gatherIndices, rowIndicesOffset: byteOffset,
            rows: 2 * config.intermediateSize, cols: config.hiddenSize,
            count: count, in: commandBuffer)

        try encoder.swiglu(
            input: scratch.gateUp, output: scratch.activated,
            size: config.intermediateSize, tokens: count,
            alpha: 1.702, limit: config.swigluLimit, in: commandBuffer)

        // The activations are already compacted: the second projection reads the rows in
        // order, hence the identity indices.
        try encoder.expertProjection(
            blocks: buffer, blocksOffset: blobOffset + blob.downBlocks.offset,
            scales: buffer, scalesOffset: blobOffset + blob.downScales.offset,
            bias: buffer, biasOffset: blobOffset + blob.downBias.offset,
            input: scratch.activated, output: scratch.expertOutput,
            rowIndices: scratch.identityIndices, rowIndicesOffset: 0,
            rows: config.hiddenSize, cols: config.intermediateSize,
            count: count, in: commandBuffer)

        try encoder.scatterExpert(
            into: scratch.mixture, outputs: scratch.expertOutput,
            rowIndices: scratch.gatherIndices, rowIndicesOffset: byteOffset,
            weights: scratch.gatherWeights, weightsOffset: byteOffset,
            size: config.hiddenSize, count: count, in: commandBuffer)
    }

    public func encodeMixtureStart(
        tokens: Int, scratch: PrefillScratch, in commandBuffer: MTLCommandBuffer
    ) throws {
        try encoder.fillZero(
            scratch.mixture, size: tokens * config.hiddenSize, in: commandBuffer)
    }

    public func encodeMixtureEnd(
        tokens: Int, scratch: PrefillScratch, in commandBuffer: MTLCommandBuffer
    ) throws {
        try encoder.addInPlace(
            target: scratch.hidden, addend: scratch.mixture,
            size: tokens * config.hiddenSize, in: commandBuffer)
    }
}
