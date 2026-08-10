import Foundation
import HydraCore
import HydraFormat
import Metal

/// Working buffers for a Gemma 4 decoding step, allocated once at load time.
///
/// Sized on the **largest** layer geometry rather than a representative one: sliding layers
/// carry 16 heads of 256 and full layers 16 of 512, so a buffer sized on the first would be
/// overrun by the second. Allocating per layer would put a Metal allocation on the per-token
/// path, which D-012 rules out.
public final class Gemma4DecodeScratch: @unchecked Sendable {

    public let hidden: MTLBuffer         // the residual state, [hiddenSize]
    public let residual: MTLBuffer       // a copy held across the two branches
    public let normed: MTLBuffer         // [hiddenSize]
    public let query: MTLBuffer          // [heads * maxHeadDim]
    public let key: MTLBuffer            // [maxKeyValueDim]
    public let value: MTLBuffer          // [maxKeyValueDim]
    public let attention: MTLBuffer      // [heads * maxHeadDim]
    public let projected: MTLBuffer      // [hiddenSize]

    /// The dense branch, which runs beside the experts rather than before them.
    public let denseGate: MTLBuffer      // [intermediateSize]
    public let denseUp: MTLBuffer        // [intermediateSize]
    public let denseActivated: MTLBuffer // [intermediateSize]
    public let denseOutput: MTLBuffer    // [hiddenSize]

    public let routerLogits: MTLBuffer   // [expertCount]
    public let routerIndices: MTLBuffer  // [expertsPerToken] UInt32
    public let routerWeights: MTLBuffer  // [expertsPerToken] Float
    public let expertInput: MTLBuffer    // [hiddenSize]
    public let expertGate: MTLBuffer     // [moeIntermediateSize]
    public let expertUp: MTLBuffer       // [moeIntermediateSize]
    public let expertActivated: MTLBuffer  // [moeIntermediateSize]
    /// One slot per selected expert, so the sum's order does not depend on which arrived first.
    public let expertSlices: MTLBuffer   // [expertsPerToken * hiddenSize]
    public let mixture: MTLBuffer        // [hiddenSize]

    public let cosTable: MTLBuffer       // [maxHeadDim / 2]
    public let sinTable: MTLBuffer       // [maxHeadDim / 2]

    /// A per-head sink no score can reach.
    ///
    /// Gemma has no attention sinks, and the shared kernel seeds its online softmax on one.
    /// Filled once with −1e30 in BF16: the sink's term becomes `exp(−1e30 − max) = 0` while
    /// the denominator stays at one, which is arithmetically identical to having none. The
    /// equivalence is measured in `GemmaKernelTests`, not assumed.
    public let unreachableSinks: MTLBuffer

    public let byteCount: Int

    public init(config: Gemma4Config, device: MTLDevice) throws {
        var total = 0
        func make(_ name: String, _ floats: Int) throws -> MTLBuffer {
            let bytes = max(floats * 4, 256)
            guard let buffer = device.makeBuffer(length: bytes, options: .storageModeShared) else {
                throw DecodeScratch.ScratchError.allocationFailed(name, bytes: bytes)
            }
            total += bytes
            return buffer
        }

        // The widest geometry across layers, not the first one's.
        let geometries = (0..<config.layerCount).map(config.attentionGeometry(atLayer:))
        let maxQueryDim = geometries.map(\.queryDim).max() ?? 0
        let maxKeyValueDim = geometries.map(\.keyValueDim).max() ?? 0
        let maxHeadDim = geometries.map(\.headDim).max() ?? 0

        let h = config.hiddenSize
        hidden = try make("hidden", h)
        residual = try make("residual", h)
        normed = try make("normed", h)
        query = try make("query", maxQueryDim)
        key = try make("key", maxKeyValueDim)
        value = try make("value", maxKeyValueDim)
        attention = try make("attention", maxQueryDim)
        projected = try make("projected", h)

        denseGate = try make("denseGate", config.intermediateSize)
        denseUp = try make("denseUp", config.intermediateSize)
        denseActivated = try make("denseActivated", config.intermediateSize)
        denseOutput = try make("denseOutput", h)

        routerLogits = try make("routerLogits", config.expertCount)
        routerIndices = try make("routerIndices", config.expertsPerToken)
        routerWeights = try make("routerWeights", config.expertsPerToken)
        expertInput = try make("expertInput", h)
        expertGate = try make("expertGate", config.moeIntermediateSize)
        expertUp = try make("expertUp", config.moeIntermediateSize)
        expertActivated = try make("expertActivated", config.moeIntermediateSize)
        expertSlices = try make("expertSlices", config.expertsPerToken * h)
        mixture = try make("mixture", h)

        cosTable = try make("cosTable", maxHeadDim / 2)
        sinTable = try make("sinTable", maxHeadDim / 2)

        let heads = config.attentionHeadCount
        guard let sinks = device.makeBuffer(
            length: max(heads * 2, 256), options: .storageModeShared)
        else {
            throw DecodeScratch.ScratchError.allocationFailed("sinks", bytes: heads * 2)
        }
        let pointer = sinks.contents().bindMemory(to: UInt16.self, capacity: heads)
        for i in 0..<heads { pointer[i] = BF16.fromFloat(-1e30) }
        unreachableSinks = sinks
        total += sinks.length

        byteCount = total
    }
}

/// Encodes one Gemma 4 layer, in the topology `Gemma4ReferenceLayer` pins down.
///
/// The structure is not GPT-OSS's with different constants. Two departures decide the shape of
/// this file, and both are in D-022:
///
/// - **`post_attention_layernorm` is applied before the residual add** — post-norm, where
///   GPT-OSS is pre-norm;
/// - **the expert branch reads the residual**, the state before the dense MLP, so the two are
///   parallel branches over the same input and are summed at the end.
///
/// The split into two command buffers is the same as GPT-OSS's and for the same reason: the CPU
/// must read the router's chosen experts before it knows which blobs to fetch from disk.
public struct Gemma4LayerRunner: Sendable {

    public let config: Gemma4Config
    public let encoder: ForwardEncoder
    public let mapping: ModelMapping
    /// Where the weights come from and how they decode.
    ///
    /// The topology below is one implementation serving two checkpoints. Every projection is
    /// `y = W · x`; only the encoding of `W` differs, and resolving that through this seam is
    /// what keeps the BF16 and MLX builds from becoming two copies of a forward pass that must
    /// agree forever (D-023).
    public let weights: any Gemma4Weights

    public init(
        config: Gemma4Config, encoder: ForwardEncoder, mapping: ModelMapping,
        weights: (any Gemma4Weights)? = nil
    ) {
        self.config = config
        self.encoder = encoder
        self.mapping = mapping
        self.weights = weights ?? Gemma4BF16Weights(config: config, mapping: mapping)
    }

    /// An unquantized tensor: the norms, the router's scales, `layer_scalar`.
    private func tensor(_ suffix: String, layer: Int) throws -> (MTLBuffer, Int) {
        let resolved = try weights.plain(suffix, layer: layer)
        return (resolved.buffer, resolved.offset)
    }

    /// A projection, by the stem the checkpoint names it with — `.weight` and, where the
    /// build is quantized, `.scales` and `.biases` are resolved behind this.
    private func projection(
        _ stem: String, layer: Int, rows: Int, cols: Int
    ) throws -> ForwardEncoder.ProjectionSource {
        try weights.projection(stem, layer: layer, rows: rows, cols: cols)
    }

    /// Everything up to and including the router, after which the caller knows which experts
    /// to read.
    /// - Parameters:
    ///   - ropeCos, ropeSin, ropeOffset: where to read this token's rotary tables. Defaults to
    ///     the scratch's single pair, which decoding rewrites before every token. Prefill passes
    ///     a table per token instead, because it encodes a whole chunk before committing and the
    ///     CPU cannot rewrite one buffer between dispatches that have not run yet.
    public func encodeAttentionAndRouter(
        layer: Int, position: Int, scratch: Gemma4DecodeScratch, kvCache: KVCache,
        ropeCos: MTLBuffer? = nil, ropeSin: MTLBuffer? = nil, ropeOffset: Int = 0,
        in commandBuffer: MTLCommandBuffer
    ) throws {
        let geometry = config.attentionGeometry(atLayer: layer)
        let ring = kvCache.layers[layer].ringSize

        // The residual is kept for the feed-forward branches, which both read it.
        try encoder.copy(
            into: scratch.residual, destinationOffset: 0,
            from: scratch.hidden, sourceOffset: 0, size: config.hiddenSize,
            in: commandBuffer)

        let inputNorm = try tensor("input_layernorm.weight", layer: layer)
        try encoder.rmsNorm(
            input: scratch.hidden, scale: inputNorm.0, scaleOffset: inputNorm.1,
            output: scratch.normed, size: config.hiddenSize, eps: config.rmsNormEps,
            in: commandBuffer)

        // --- Q, K and, on sliding layers only, V ---
        try encoder.encodeProjection(
            try projection(
                "self_attn.q_proj", layer: layer,
                rows: geometry.queryDim, cols: config.hiddenSize),
            input: scratch.normed, inputOffset: 0, output: scratch.query, outputOffset: 0,
            rows: geometry.queryDim, cols: config.hiddenSize, in: commandBuffer)

        try encoder.encodeProjection(
            try projection(
                "self_attn.k_proj", layer: layer,
                rows: geometry.keyValueDim, cols: config.hiddenSize),
            input: scratch.normed, inputOffset: 0, output: scratch.key, outputOffset: 0,
            rows: geometry.keyValueDim, cols: config.hiddenSize, in: commandBuffer)

        // `attention_k_eq_v`: V reuses the key projection's **output**, taken before k_norm and
        // before RoPE, then normalized on its own. Copying it here is what makes that ordering
        // explicit rather than accidental.
        if config.hasValueProjection(atLayer: layer) {
            try encoder.encodeProjection(
                try projection(
                    "self_attn.v_proj", layer: layer,
                    rows: geometry.keyValueDim, cols: config.hiddenSize),
                input: scratch.normed, inputOffset: 0, output: scratch.value, outputOffset: 0,
                rows: geometry.keyValueDim, cols: config.hiddenSize, in: commandBuffer)
        } else {
            try encoder.copy(
                into: scratch.value, destinationOffset: 0,
                from: scratch.key, sourceOffset: 0, size: geometry.keyValueDim,
                in: commandBuffer)
        }

        // --- The norms, before RoPE for Q and K; V takes a weightless one and no rotation ---
        let qNorm = try tensor("self_attn.q_norm.weight", layer: layer)
        for head in 0..<geometry.attentionHeadCount {
            try encoder.rmsNorm(
                input: scratch.query, inputOffset: head * geometry.headDim * 4,
                scale: qNorm.0, scaleOffset: qNorm.1,
                output: scratch.query, outputOffset: head * geometry.headDim * 4,
                size: geometry.headDim, eps: config.rmsNormEps, in: commandBuffer)
        }
        let kNorm = try tensor("self_attn.k_norm.weight", layer: layer)
        for head in 0..<geometry.keyValueHeadCount {
            try encoder.rmsNorm(
                input: scratch.key, inputOffset: head * geometry.headDim * 4,
                scale: kNorm.0, scaleOffset: kNorm.1,
                output: scratch.key, outputOffset: head * geometry.headDim * 4,
                size: geometry.headDim, eps: config.rmsNormEps, in: commandBuffer)
        }
        // Per key/value head, not over the concatenated vector.
        //
        // RMS normalization is **not separable**: normalizing `keyValueDim` values together is
        // a different operation from normalizing each head's slice. The two coincide when
        // there is one key/value head, which is why a full-attention layer agreed with the
        // oracle while a sliding one — two heads — diverged by 42 %.
        for head in 0..<geometry.keyValueHeadCount {
            try encoder.rmsNormUnscaled(
                input: scratch.value, inputOffset: head * geometry.headDim * 4,
                output: scratch.value, outputOffset: head * geometry.headDim * 4,
                size: geometry.headDim, eps: config.rmsNormEps, in: commandBuffer)
        }

        try encoder.applyRoPE(
            vector: scratch.query, vectorOffset: 0,
            cos: ropeCos ?? scratch.cosTable, sin: ropeSin ?? scratch.sinTable,
            tableOffset: ropeOffset,
            heads: geometry.attentionHeadCount, headDim: geometry.headDim, in: commandBuffer)
        try encoder.applyRoPE(
            vector: scratch.key, vectorOffset: 0,
            cos: ropeCos ?? scratch.cosTable, sin: ropeSin ?? scratch.sinTable,
            tableOffset: ropeOffset,
            heads: geometry.keyValueHeadCount, headDim: geometry.headDim, in: commandBuffer)

        try encoder.writeKeyValue(
            key: scratch.key, keyOffset: 0, value: scratch.value, valueOffset: 0,
            keyCache: kvCache.layers[layer].keys, valueCache: kvCache.layers[layer].values,
            kvHeads: geometry.keyValueHeadCount, headDim: geometry.headDim,
            position: position, ringSize: ring, in: commandBuffer)

        let visible = kvCache.visibleRange(layer: layer, position: position)
        try encoder.attention(
            query: scratch.query, queryOffset: 0,
            keyCache: kvCache.layers[layer].keys, valueCache: kvCache.layers[layer].values,
            sinks: scratch.unreachableSinks, sinksOffset: 0,
            output: scratch.attention, outputOffset: 0,
            qHeads: geometry.attentionHeadCount, kvHeads: geometry.keyValueHeadCount,
            headDim: geometry.headDim, keyCount: visible.count,
            ringSize: ring, startPosition: visible.start,
            // 1.0, not 1/sqrt(headDim): the query norm absorbs the scale (D-022).
            smScale: 1.0, in: commandBuffer)

        try encoder.encodeProjection(
            try projection(
                "self_attn.o_proj", layer: layer,
                rows: config.hiddenSize, cols: geometry.queryDim),
            input: scratch.attention, inputOffset: 0,
            output: scratch.projected, outputOffset: 0,
            rows: config.hiddenSize, cols: geometry.queryDim, in: commandBuffer)

        // **Post-norm**: the attention output is normalized before it rejoins the residual.
        let postAttention = try tensor("post_attention_layernorm.weight", layer: layer)
        try encoder.rmsNorm(
            input: scratch.projected, scale: postAttention.0, scaleOffset: postAttention.1,
            output: scratch.projected, size: config.hiddenSize, eps: config.rmsNormEps,
            in: commandBuffer)
        try encoder.addInPlace(
            target: scratch.hidden, targetOffset: 0,
            addend: scratch.projected, addendOffset: 0,
            size: config.hiddenSize, in: commandBuffer)

        // The residual the two feed-forward branches share.
        try encoder.copy(
            into: scratch.residual, destinationOffset: 0,
            from: scratch.hidden, sourceOffset: 0, size: config.hiddenSize,
            in: commandBuffer)

        // --- The dense branch, which does not wait for the experts ---
        let preFF = try tensor("pre_feedforward_layernorm.weight", layer: layer)
        try encoder.rmsNorm(
            input: scratch.residual, scale: preFF.0, scaleOffset: preFF.1,
            output: scratch.normed, size: config.hiddenSize, eps: config.rmsNormEps,
            in: commandBuffer)
        try encoder.encodeProjection(
            try projection(
                "mlp.gate_proj", layer: layer,
                rows: config.intermediateSize, cols: config.hiddenSize),
            input: scratch.normed, inputOffset: 0, output: scratch.denseGate, outputOffset: 0,
            rows: config.intermediateSize, cols: config.hiddenSize, in: commandBuffer)
        try encoder.encodeProjection(
            try projection(
                "mlp.up_proj", layer: layer,
                rows: config.intermediateSize, cols: config.hiddenSize),
            input: scratch.normed, inputOffset: 0, output: scratch.denseUp, outputOffset: 0,
            rows: config.intermediateSize, cols: config.hiddenSize, in: commandBuffer)
        try encoder.geluMultiply(
            gate: scratch.denseGate, up: scratch.denseUp, output: scratch.denseActivated,
            size: config.intermediateSize, in: commandBuffer)
        try encoder.encodeProjection(
            try projection(
                "mlp.down_proj", layer: layer,
                rows: config.hiddenSize, cols: config.intermediateSize),
            input: scratch.denseActivated, inputOffset: 0,
            output: scratch.denseOutput, outputOffset: 0,
            rows: config.hiddenSize, cols: config.intermediateSize, in: commandBuffer)
        let postFF1 = try tensor("post_feedforward_layernorm_1.weight", layer: layer)
        try encoder.rmsNorm(
            input: scratch.denseOutput, scale: postFF1.0, scaleOffset: postFF1.1,
            output: scratch.denseOutput, size: config.hiddenSize, eps: config.rmsNormEps,
            in: commandBuffer)

        // --- The router, reading the residual rather than the dense branch's output ---
        try encoder.rmsNormUnscaled(
            input: scratch.residual, output: scratch.expertInput,
            size: config.hiddenSize, eps: config.rmsNormEps, in: commandBuffer)
        let routerScale = try tensor("router.scale", layer: layer)
        try encoder.scaleByBF16(
            target: scratch.expertInput, scale: routerScale.0, scaleOffset: routerScale.1,
            factor: Foundation.pow(Float(config.hiddenSize), -0.5),
            size: config.hiddenSize, in: commandBuffer)
        try encoder.encodeProjection(
            try projection(
                "router.proj", layer: layer,
                rows: config.expertCount, cols: config.hiddenSize),
            input: scratch.expertInput, inputOffset: 0,
            output: scratch.routerLogits, outputOffset: 0,
            rows: config.expertCount, cols: config.hiddenSize, in: commandBuffer)

        // The expert branch's own normalization, of the residual and not of the router's input.
        let preFF2 = try tensor("pre_feedforward_layernorm_2.weight", layer: layer)
        try encoder.rmsNorm(
            input: scratch.residual, scale: preFF2.0, scaleOffset: preFF2.1,
            output: scratch.expertInput, size: config.hiddenSize, eps: config.rmsNormEps,
            in: commandBuffer)
    }

    // MARK: - The expert branch, and the sum of the two

    /// A single expert, encoded as soon as its blob is resident.
    ///
    /// The weights are plain BF16 matrices, so this is `denseProjection` twice around
    /// `geluMultiply` — no dequantization step, which is the only reason GPT-OSS needs its own
    /// expert kernel.
    ///
    /// Each contribution is written into **its own slot** rather than accumulated in place, so
    /// the sum's order is fixed by the slot index and not by which expert happened to arrive
    /// first. Without that, the same prompt would give different logits depending on the state
    /// of the cache.
    /// - Parameters:
    ///   - destination: where the scaled contribution lands, and at which slot. Defaults to the
    ///     decode scratch's own slices. Prefill passes a batch-wide buffer instead, because it
    ///     runs **expert-major**: one expert serves every token that chose it, so token *t*'s
    ///     slots cannot live in a scratch that the next token overwrites.
    ///   - routerWeights: the router weights to scale by. Same reason — prefill holds one row
    ///     per token and indexes into it, where decode has only the current token's.
    ///
    /// Parameterized rather than duplicated. A second copy of these four projections is exactly
    /// the kind of drift D-023 warns about: it would run, return finite numbers, and disagree
    /// with the decode path by a little.
    public func encodeSingleExpert(
        buffer: MTLBuffer, blobOffset: Int = 0, weightIndex: Int,
        scratch: Gemma4DecodeScratch,
        destination: MTLBuffer? = nil, destinationSlot: Int? = nil,
        routerWeights: MTLBuffer? = nil, weightSlot: Int? = nil,
        in commandBuffer: MTLCommandBuffer
    ) throws {
        let inner = config.moeIntermediateSize

        // `gate` and `up` are two halves of one fused tensor in the BF16 build and two separate
        // matrices in the MLX one. Asking the weight source by name is what lets the same three
        // projections serve both.
        try encoder.encodeProjection(
            weights.expert(.gate, blob: buffer, blobOffset: blobOffset),
            input: scratch.expertInput, inputOffset: 0,
            output: scratch.expertGate, outputOffset: 0,
            rows: inner, cols: config.hiddenSize, in: commandBuffer)
        try encoder.encodeProjection(
            weights.expert(.up, blob: buffer, blobOffset: blobOffset),
            input: scratch.expertInput, inputOffset: 0,
            output: scratch.expertUp, outputOffset: 0,
            rows: inner, cols: config.hiddenSize, in: commandBuffer)
        try encoder.geluMultiply(
            gate: scratch.expertGate, up: scratch.expertUp,
            output: scratch.expertActivated, size: inner, in: commandBuffer)
        try encoder.encodeProjection(
            weights.expert(.down, blob: buffer, blobOffset: blobOffset),
            input: scratch.expertActivated, inputOffset: 0,
            output: scratch.mixture, outputOffset: 0,
            rows: config.hiddenSize, cols: inner, in: commandBuffer)
        try encoder.writeExpertScaled(
            into: destination ?? scratch.expertSlices,
            outputOffset: (destinationSlot ?? weightIndex)
                * config.hiddenSize * MemoryLayout<Float>.size,
            contribution: scratch.mixture, contributionOffset: 0,
            weights: routerWeights ?? scratch.routerWeights,
            weightIndex: weightSlot ?? weightIndex,
            size: config.hiddenSize, in: commandBuffer)
    }

    /// Sums the expert slots, normalizes each branch, adds them, and closes the layer.
    ///
    /// This is where the parallel structure becomes visible: `denseOutput` was produced before
    /// any expert ran, and neither branch has seen the other until this point.
    public func encodeCombineBranches(
        layer: Int, count: Int, scratch: Gemma4DecodeScratch,
        in commandBuffer: MTLCommandBuffer
    ) throws {
        try encoder.sumExpertSlices(
            into: scratch.mixture, slices: scratch.expertSlices,
            size: config.hiddenSize, count: count, in: commandBuffer)

        let postFF2 = try tensor("post_feedforward_layernorm_2.weight", layer: layer)
        try encoder.rmsNorm(
            input: scratch.mixture, scale: postFF2.0, scaleOffset: postFF2.1,
            output: scratch.mixture, size: config.hiddenSize, eps: config.rmsNormEps,
            in: commandBuffer)

        // branch1 + branch2, then the layer's own norm, then the residual.
        try encoder.addInPlace(
            target: scratch.mixture, targetOffset: 0,
            addend: scratch.denseOutput, addendOffset: 0,
            size: config.hiddenSize, in: commandBuffer)
        let postFF = try tensor("post_feedforward_layernorm.weight", layer: layer)
        try encoder.rmsNorm(
            input: scratch.mixture, scale: postFF.0, scaleOffset: postFF.1,
            output: scratch.mixture, size: config.hiddenSize, eps: config.rmsNormEps,
            in: commandBuffer)

        try encoder.copy(
            into: scratch.hidden, destinationOffset: 0,
            from: scratch.residual, sourceOffset: 0, size: config.hiddenSize,
            in: commandBuffer)
        try encoder.addInPlace(
            target: scratch.hidden, targetOffset: 0,
            addend: scratch.mixture, addendOffset: 0,
            size: config.hiddenSize, in: commandBuffer)

        // `layer_scalar`, applied last: one stored value multiplying the whole hidden state,
        // so it needs the broadcasting kernel rather than the per-element one.
        let scalar = try tensor("layer_scalar", layer: layer)
        try encoder.scaleByScalar(
            target: scratch.hidden, scale: scalar.0, scaleOffset: scalar.1,
            size: config.hiddenSize, in: commandBuffer)
    }

    /// Zeroes the slots before an expert writes into any of them.
    public func encodeMixtureStart(
        scratch: Gemma4DecodeScratch, in commandBuffer: MTLCommandBuffer
    ) throws {
        try encoder.fillZero(
            scratch.expertSlices, offset: 0,
            size: config.expertsPerToken * config.hiddenSize, in: commandBuffer)
    }
}
