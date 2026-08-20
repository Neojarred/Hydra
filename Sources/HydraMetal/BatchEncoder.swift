import Foundation
import HydraCore
import Metal

/// Encodes the passes of a prefill chunk.
///
/// Same role as `ForwardEncoder`, over kernels that process several tokens at once. The two
/// coexist: decoding stays one token at a time by nature, each token depends on the
/// previous, whereas prefill knows the whole prompt in advance.
public struct BatchEncoder: Sendable {

    public let context: MetalContext

    public init(context: MetalContext) {
        self.context = context
    }

    /// Tokens per threadgroup in the GEMM kernels. Must match `TOKEN_TILE` on the Metal
    /// side: that is the only coupling between the two, and breaking it silently would
    /// produce partial results.
    public static let tokenTile = 16

    private func encodeGrid(
        _ function: String, in commandBuffer: MTLCommandBuffer,
        threadgroups: MTLSize, width: Int,
        _ configure: (MTLComputeCommandEncoder) -> Void
    ) throws {
        let pipeline = try context.pipeline(function)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalContext.ContextError.functionMissing(function)
        }
        encoder.setComputePipelineState(pipeline)
        configure(encoder)
        encoder.dispatchThreadgroups(
            threadgroups,
            threadsPerThreadgroup: MTLSize(
                width: min(width, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func encodeLinear(
        _ function: String, in commandBuffer: MTLCommandBuffer, elements: Int,
        _ configure: (MTLComputeCommandEncoder) -> Void
    ) throws {
        guard elements > 0 else { return }
        let pipeline = try context.pipeline(function)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalContext.ContextError.functionMissing(function)
        }
        encoder.setComputePipelineState(pipeline)
        configure(encoder)
        encoder.dispatchThreads(
            MTLSize(width: elements, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: min(elements, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// GEMM tiling dimensions, to be kept identical to those in `tiled.metal`.
    /// Memory traffic is `cols × rows × tokens × (2/tileTokens + 4/tileRows)`: these two
    /// numbers determine performance, not how finely the loops are unrolled.
    public static let tileRows = 128
    public static let tileTokens = 64
    private static let tileThreads = 256

    private func gemmGrid(rows: Int, tokens: Int) -> MTLSize {
        MTLSize(
            width: (rows + Self.tileRows - 1) / Self.tileRows,
            height: (tokens + Self.tileTokens - 1) / Self.tileTokens,
            depth: 1)
    }

    private var gemmWidth: Int { Self.tileThreads }

    // MARK: - Passes

    public func rmsNorm(
        input: MTLBuffer, scale: MTLBuffer, scaleOffset: Int, output: MTLBuffer,
        size: Int, tokens: Int, eps: Float, in commandBuffer: MTLCommandBuffer
    ) throws {
        var dims = SIMD2<UInt32>(UInt32(size), UInt32(tokens))
        var epsilon = eps
        try encodeGrid(
            "rms_norm_batch", in: commandBuffer,
            threadgroups: MTLSize(width: tokens, height: 1, depth: 1), width: 256
        ) {
            $0.setBuffer(input, offset: 0, index: 0)
            $0.setBuffer(scale, offset: scaleOffset, index: 1)
            $0.setBuffer(output, offset: 0, index: 2)
            $0.setBytes(&dims, length: MemoryLayout<SIMD2<UInt32>>.size, index: 3)
            $0.setBytes(&epsilon, length: 4, index: 4)
        }
    }

    public func denseProjection(
        weights: MTLBuffer, weightsOffset: Int, bias: MTLBuffer?, biasOffset: Int,
        input: MTLBuffer, output: MTLBuffer,
        rows: Int, cols: Int, tokens: Int, in commandBuffer: MTLCommandBuffer
    ) throws {
        // `bf16_gemm` reads the weight eight BF16 at a time as a `uint4`, and the activation
        // as two `float4`, so `cols / 8` is the loop bound and a remainder is not read at all.
        // Every shape this project ships is a multiple of eight, which is exactly why nothing
        // caught it: a tiny fixture with 12 columns silently used 8 of them and produced a
        // finite, wrong answer. The same shape of fault as M-051.
        precondition(
            cols % 8 == 0,
            "bf16_gemm reads columns in groups of eight; \(cols) would drop \(cols % 8)")
        var dims = SIMD4<UInt32>(
            UInt32(rows), UInt32(cols), UInt32(tokens), UInt32(bias == nil ? 0 : 1))
        let tiled = tokens >= Self.tiledThreshold
        try encodeGrid(
            tiled ? "bf16_gemm_tiled" : "bf16_gemm", in: commandBuffer,
            threadgroups: tiled
                ? gemmGrid(rows: rows, tokens: tokens)
                : MTLSize(width: (rows + 3) / 4,
                          height: (tokens + Self.tokenTile - 1) / Self.tokenTile, depth: 1),
            width: tiled ? gemmWidth : 128
        ) {
            $0.setBuffer(weights, offset: weightsOffset, index: 0)
            $0.setBuffer(bias ?? weights, offset: bias == nil ? 0 : biasOffset, index: 1)
            $0.setBuffer(input, offset: 0, index: 2)
            $0.setBuffer(output, offset: 0, index: 3)
            $0.setBytes(&dims, length: MemoryLayout<SIMD4<UInt32>>.size, index: 4)
        }
    }

    /// Switch-over threshold between the two expert kernels.
    ///
    /// The tiled kernel reduces memory traffic, but with `TILE_ROWS = 128` it launches only
    /// `rows/128` threadgroups, 45 for `gate_up`, far short of what it takes to occupy the
    /// GPU. And in prefill an expert serves only a fraction of the chunk: about eight tokens
    /// out of sixty-eight. Below this threshold the one-row-per-SIMD-group variant launches
    /// thirty times more threadgroups and wins comfortably.
    ///
    /// The right kernel therefore depends on the token count, not on the kind of operation.
    public static let tiledThreshold = 32

    public func expertProjection(
        blocks: MTLBuffer, blocksOffset: Int, scales: MTLBuffer, scalesOffset: Int,
        bias: MTLBuffer?, biasOffset: Int,
        input: MTLBuffer, output: MTLBuffer,
        rowIndices: MTLBuffer, rowIndicesOffset: Int,
        rows: Int, cols: Int, count: Int, in commandBuffer: MTLCommandBuffer
    ) throws {
        var dims = SIMD4<UInt32>(
            UInt32(rows), UInt32(cols), UInt32(count), UInt32(bias == nil ? 0 : 1))
        let tiled = count >= Self.tiledThreshold
        try encodeGrid(
            tiled ? "mxfp4_gemm_tiled" : "mxfp4_gemm_gathered", in: commandBuffer,
            threadgroups: tiled
                ? gemmGrid(rows: rows, tokens: count)
                : MTLSize(width: (rows + 3) / 4,
                          height: (count + Self.tokenTile - 1) / Self.tokenTile, depth: 1),
            width: tiled ? gemmWidth : 128
        ) {
            $0.setBuffer(blocks, offset: blocksOffset, index: 0)
            $0.setBuffer(scales, offset: scalesOffset, index: 1)
            $0.setBuffer(bias ?? blocks, offset: bias == nil ? 0 : biasOffset, index: 2)
            $0.setBuffer(input, offset: 0, index: 3)
            $0.setBuffer(output, offset: 0, index: 4)
            $0.setBuffer(rowIndices, offset: rowIndicesOffset, index: 5)
            $0.setBytes(&dims, length: MemoryLayout<SIMD4<UInt32>>.size, index: 6)
        }
    }

    public func applyRoPE(
        vector: MTLBuffer, cos: MTLBuffer, sin: MTLBuffer,
        heads: Int, headDim: Int, tokens: Int, in commandBuffer: MTLCommandBuffer
    ) throws {
        var dims = SIMD4<UInt32>(UInt32(heads), UInt32(headDim), UInt32(tokens), 0)
        try encodeLinear(
            "rope_apply_batch", in: commandBuffer, elements: tokens * heads * headDim / 2
        ) {
            $0.setBuffer(vector, offset: 0, index: 0)
            $0.setBuffer(cos, offset: 0, index: 1)
            $0.setBuffer(sin, offset: 0, index: 2)
            $0.setBytes(&dims, length: MemoryLayout<SIMD4<UInt32>>.size, index: 3)
        }
    }

    public func writeKeyValue(
        key: MTLBuffer, value: MTLBuffer, keyCache: MTLBuffer, valueCache: MTLBuffer,
        kvHeads: Int, headDim: Int, tokens: Int, firstPosition: Int, ringSize: Int,
        in commandBuffer: MTLCommandBuffer
    ) throws {
        var dims = SIMD4<UInt32>(
            UInt32(kvHeads), UInt32(headDim), UInt32(tokens), UInt32(ringSize))
        var first = UInt32(firstPosition)
        try encodeLinear(
            "kv_cache_write_batch", in: commandBuffer, elements: tokens * kvHeads * headDim
        ) {
            $0.setBuffer(key, offset: 0, index: 0)
            $0.setBuffer(value, offset: 0, index: 1)
            $0.setBuffer(keyCache, offset: 0, index: 2)
            $0.setBuffer(valueCache, offset: 0, index: 3)
            $0.setBytes(&dims, length: MemoryLayout<SIMD4<UInt32>>.size, index: 4)
            $0.setBytes(&first, length: 4, index: 5)
        }
    }

    public func attention(
        query: MTLBuffer, keyCache: MTLBuffer, valueCache: MTLBuffer,
        sinks: MTLBuffer, sinksOffset: Int, output: MTLBuffer,
        qHeads: Int, kvHeads: Int, headDim: Int, tokens: Int,
        ringSize: Int, firstPosition: Int, slidingWindow: Int, smScale: Float,
        /// One entry a token, the end of its bidirectional block, or `nil` for none.
        blockEnds: MTLBuffer? = nil,
        in commandBuffer: MTLCommandBuffer
    ) throws {
        var dims = SIMD4<UInt32>(
            UInt32(qHeads), UInt32(kvHeads), UInt32(headDim), UInt32(tokens))
        // A zeroed scratch when there are no blocks: the kernel reads one entry a token either
        // way, and zero means "ordinary token, causal only".
        let blocks = blockEnds
            ?? context.scratch("attention.blocks", bytes: max(tokens, 1) * 4)
        var window = SIMD4<UInt32>(
            UInt32(ringSize), UInt32(firstPosition), UInt32(slidingWindow), 0)
        var scale = smScale
        try encodeGrid(
            // Must match what the kernel expects and what decoding dispatches: the key range
            // is split across the threadgroup's simdgroups, so 32 threads would be one
            // simdgroup and a different reassociation from `attention_decode`.
            "attention_prefill", in: commandBuffer,
            threadgroups: MTLSize(width: qHeads, height: tokens, depth: 1),
            width: ForwardEncoder.attentionThreads
        ) {
            $0.setBuffer(query, offset: 0, index: 0)
            $0.setBuffer(keyCache, offset: 0, index: 1)
            $0.setBuffer(valueCache, offset: 0, index: 2)
            $0.setBuffer(sinks, offset: sinksOffset, index: 3)
            $0.setBuffer(output, offset: 0, index: 4)
            $0.setBytes(&dims, length: MemoryLayout<SIMD4<UInt32>>.size, index: 5)
            $0.setBytes(&window, length: MemoryLayout<SIMD4<UInt32>>.size, index: 6)
            $0.setBytes(&scale, length: 4, index: 7)
            $0.setBuffer(blocks, offset: 0, index: 8)
        }
    }

    public func swiglu(
        input: MTLBuffer, output: MTLBuffer, size: Int, tokens: Int,
        alpha: Float, limit: Float, in commandBuffer: MTLCommandBuffer
    ) throws {
        var dims = SIMD2<UInt32>(UInt32(size), UInt32(tokens))
        var params = SIMD2<Float>(alpha, limit)
        try encodeLinear("swiglu_batch", in: commandBuffer, elements: size * tokens) {
            $0.setBuffer(input, offset: 0, index: 0)
            $0.setBuffer(output, offset: 0, index: 1)
            $0.setBytes(&dims, length: MemoryLayout<SIMD2<UInt32>>.size, index: 2)
            $0.setBytes(&params, length: MemoryLayout<SIMD2<Float>>.size, index: 3)
        }
    }

    public func routerTopK(
        logits: MTLBuffer, indices: MTLBuffer, weights: MTLBuffer,
        expertCount: Int, topK: Int, tokens: Int, in commandBuffer: MTLCommandBuffer
    ) throws {
        var dims = SIMD4<UInt32>(UInt32(expertCount), UInt32(topK), UInt32(tokens), 0)
        try encodeGrid(
            "router_topk_batch", in: commandBuffer,
            threadgroups: MTLSize(width: tokens, height: 1, depth: 1), width: 32
        ) {
            $0.setBuffer(logits, offset: 0, index: 0)
            $0.setBuffer(indices, offset: 0, index: 1)
            $0.setBuffer(weights, offset: 0, index: 2)
            $0.setBytes(&dims, length: MemoryLayout<SIMD4<UInt32>>.size, index: 3)
        }
    }

    public func scatterExpert(
        into mixture: MTLBuffer, outputs: MTLBuffer,
        rowIndices: MTLBuffer, rowIndicesOffset: Int,
        weights: MTLBuffer, weightsOffset: Int,
        size: Int, count: Int, in commandBuffer: MTLCommandBuffer
    ) throws {
        var dims = SIMD2<UInt32>(UInt32(size), UInt32(count))
        try encodeLinear("scatter_expert", in: commandBuffer, elements: size * count) {
            $0.setBuffer(mixture, offset: 0, index: 0)
            $0.setBuffer(outputs, offset: 0, index: 1)
            $0.setBuffer(rowIndices, offset: rowIndicesOffset, index: 2)
            $0.setBuffer(weights, offset: weightsOffset, index: 3)
            $0.setBytes(&dims, length: MemoryLayout<SIMD2<UInt32>>.size, index: 4)
        }
    }

    public func addInPlace(
        target: MTLBuffer, addend: MTLBuffer, size: Int, in commandBuffer: MTLCommandBuffer
    ) throws {
        var count = UInt32(size)
        try encodeLinear("add_inplace_batch", in: commandBuffer, elements: size) {
            $0.setBuffer(target, offset: 0, index: 0)
            $0.setBuffer(addend, offset: 0, index: 1)
            $0.setBytes(&count, length: 4, index: 2)
        }
    }

    public func fillZero(
        _ buffer: MTLBuffer, size: Int, in commandBuffer: MTLCommandBuffer
    ) throws {
        var count = UInt32(size)
        try encodeLinear("fill_zero", in: commandBuffer, elements: size) {
            $0.setBuffer(buffer, offset: 0, index: 0)
            $0.setBytes(&count, length: 4, index: 1)
        }
    }
    // MARK: - The vision tower

    /// LayerNorm over a batch of patches. Unlike `rmsNorm` above, the mean is removed and a
    /// bias is added, which is what the tower's tensors call for.
    public func visionLayerNorm(
        input: MTLBuffer, weight: MTLBuffer, weightOffset: Int,
        bias: MTLBuffer, biasOffset: Int, output: MTLBuffer,
        width: Int, tokens: Int, eps: Float = 1e-6, in commandBuffer: MTLCommandBuffer
    ) throws {
        var dims = SIMD2<UInt32>(UInt32(width), UInt32(tokens))
        var epsilon = eps
        try encodeGrid(
            "vision_layer_norm", in: commandBuffer,
            threadgroups: MTLSize(width: tokens, height: 1, depth: 1), width: 256
        ) {
            $0.setBuffer(input, offset: 0, index: 0)
            $0.setBuffer(weight, offset: weightOffset, index: 1)
            $0.setBuffer(bias, offset: biasOffset, index: 2)
            $0.setBuffer(output, offset: 0, index: 3)
            $0.setBytes(&dims, length: MemoryLayout<SIMD2<UInt32>>.size, index: 4)
            $0.setBytes(&epsilon, length: 4, index: 5)
        }
    }

    public func visionGELU(
        _ buffer: MTLBuffer, count: Int, in commandBuffer: MTLCommandBuffer
    ) throws {
        var size = UInt32(count)
        try encodeLinear("vision_gelu", in: commandBuffer, elements: count) {
            $0.setBuffer(buffer, offset: 0, index: 0)
            $0.setBytes(&size, length: 4, index: 1)
        }
    }

    /// Turns the query and key halves of a packed `qkv` in place, leaving the values alone.
    public func visionRotary(
        qkv: MTLBuffer, angles: MTLBuffer,
        patches: Int, heads: Int, headDim: Int, in commandBuffer: MTLCommandBuffer
    ) throws {
        var dims = SIMD4<UInt32>(UInt32(patches), UInt32(heads), UInt32(headDim), 0)
        let pipeline = try context.pipeline("vision_rotary")
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalContext.ContextError.functionMissing("vision_rotary")
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(qkv, offset: 0, index: 0)
        encoder.setBuffer(angles, offset: 0, index: 1)
        encoder.setBytes(&dims, length: MemoryLayout<SIMD4<UInt32>>.size, index: 2)
        encoder.dispatchThreads(
            MTLSize(width: heads * headDim / 2, height: patches, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))
        encoder.endEncoding()
    }

    /// Bidirectional attention over one image's patches.
    public func visionAttention(
        qkv: MTLBuffer, output: MTLBuffer,
        patches: Int, heads: Int, headDim: Int, scale requested: Float? = nil,
        in commandBuffer: MTLCommandBuffer
    ) throws {
        var dims = SIMD4<UInt32>(UInt32(patches), UInt32(heads), UInt32(headDim), 0)
        // Qwen's `1/sqrt(headDim)` unless the caller says otherwise. Gemma's tower asks for 1,
        // because its queries and keys are RMS-normalized before the product.
        var scale = requested ?? 1 / Float(headDim).squareRoot()
        // The specialized kernel when the width is one we compiled, the generic one otherwise.
        // A width with no specialization still runs, just without the unrolling.
        let specialized = [64, 72, 128].contains(headDim)
        try encodeGrid(
            specialized ? "vision_attention_\(headDim)" : "vision_attention", in: commandBuffer,
            // Eight queries a threadgroup, one to a simdgroup, so a key tile read once serves
            // eight of them (M-070).
            threadgroups: MTLSize(width: heads, height: (patches + 7) / 8, depth: 1), width: 256
        ) {
            $0.setBuffer(qkv, offset: 0, index: 0)
            $0.setBuffer(output, offset: 0, index: 1)
            $0.setBytes(&dims, length: MemoryLayout<SIMD4<UInt32>>.size, index: 2)
            $0.setBytes(&scale, length: 4, index: 3)
        }
    }
    // MARK: - Gemma's vision tower

    /// RMSNorm with no learned weight, over a batch of tokens.
    public func rmsNormUnscaledBatch(
        input: MTLBuffer, output: MTLBuffer, size: Int, tokens: Int, eps: Float,
        in commandBuffer: MTLCommandBuffer
    ) throws {
        var dims = SIMD2<UInt32>(UInt32(size), UInt32(tokens))
        var epsilon = eps
        try encodeGrid(
            "rms_norm_unscaled_batch", in: commandBuffer,
            threadgroups: MTLSize(width: tokens, height: 1, depth: 1), width: 256
        ) {
            $0.setBuffer(input, offset: 0, index: 0)
            $0.setBuffer(output, offset: 0, index: 1)
            $0.setBytes(&dims, length: MemoryLayout<SIMD2<UInt32>>.size, index: 2)
            $0.setBytes(&epsilon, length: 4, index: 3)
        }
    }

    /// Per-head RMSNorm over a batch, for q, k and v. `scale` is `nil` for v, which has none.
    public func rmsNormHeadsBatch(
        buffer: MTLBuffer, scale: MTLBuffer?, scaleOffset: Int,
        headDim: Int, heads: Int, tokens: Int, eps: Float, in commandBuffer: MTLCommandBuffer
    ) throws {
        // `rms_norm_heads` indexes by threadgroup alone, so a batch is just more threadgroups:
        // token `t` head `h` sits at `(t * heads + h) * headDim`, which is what it computes.
        var dims = SIMD2<UInt32>(UInt32(headDim), UInt32(scale == nil ? 0 : 1))
        var epsilon = eps
        try encodeGrid(
            "rms_norm_heads", in: commandBuffer,
            threadgroups: MTLSize(width: tokens * heads, height: 1, depth: 1), width: 128
        ) {
            $0.setBuffer(buffer, offset: 0, index: 0)
            $0.setBuffer(scale ?? buffer, offset: scale == nil ? 0 : scaleOffset, index: 1)
            $0.setBytes(&dims, length: MemoryLayout<SIMD2<UInt32>>.size, index: 2)
            $0.setBytes(&epsilon, length: 4, index: 3)
        }
    }

    /// Gemma's two-dimensional rotary, in place.
    public func gemmaVisionRotary(
        buffer: MTLBuffer, angles: MTLBuffer,
        tokens: Int, heads: Int, headDim: Int, perAxis: Int,
        in commandBuffer: MTLCommandBuffer
    ) throws {
        var dims = SIMD4<UInt32>(
            UInt32(tokens), UInt32(heads), UInt32(headDim), UInt32(perAxis))
        let pipeline = try context.pipeline("gemma_vision_rotary")
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalContext.ContextError.functionMissing("gemma_vision_rotary")
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(buffer, offset: 0, index: 0)
        encoder.setBuffer(angles, offset: 0, index: 1)
        encoder.setBytes(&dims, length: MemoryLayout<SIMD4<UInt32>>.size, index: 2)
        encoder.dispatchThreads(
            MTLSize(width: heads * perAxis / 2 * 2, height: tokens, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))
        encoder.endEncoding()
    }

    /// The 3x3 average pool, the `sqrt(hidden)` scale and the standardization, fused.
    public func gemmaVisionPool(
        patches: MTLBuffer, bias: MTLBuffer, biasOffset: Int,
        scale: MTLBuffer, scaleOffset: Int, output: MTLBuffer,
        hidden: Int, gridWidth: Int, kernelSize: Int, pooledWidth: Int, pooledTokens: Int,
        in commandBuffer: MTLCommandBuffer
    ) throws {
        var dims = SIMD4<UInt32>(
            UInt32(hidden), UInt32(gridWidth), UInt32(kernelSize), UInt32(pooledWidth))
        try encodeGrid(
            "gemma_vision_pool", in: commandBuffer,
            threadgroups: MTLSize(width: pooledTokens, height: 1, depth: 1), width: 256
        ) {
            $0.setBuffer(patches, offset: 0, index: 0)
            $0.setBuffer(bias, offset: biasOffset, index: 1)
            $0.setBuffer(scale, offset: scaleOffset, index: 2)
            $0.setBuffer(output, offset: 0, index: 3)
            $0.setBytes(&dims, length: MemoryLayout<SIMD4<UInt32>>.size, index: 4)
        }
    }
    /// Gathers q, k and v into the packed layout the attention kernel reads.
    public func gemmaVisionPackQKV(
        query: MTLBuffer, key: MTLBuffer, value: MTLBuffer, qkv: MTLBuffer,
        hidden: Int, tokens: Int, in commandBuffer: MTLCommandBuffer
    ) throws {
        var dims = SIMD2<UInt32>(UInt32(hidden), UInt32(tokens))
        let pipeline = try context.pipeline("gemma_vision_pack_qkv")
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalContext.ContextError.functionMissing("gemma_vision_pack_qkv")
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(query, offset: 0, index: 0)
        encoder.setBuffer(key, offset: 0, index: 1)
        encoder.setBuffer(value, offset: 0, index: 2)
        encoder.setBuffer(qkv, offset: 0, index: 3)
        encoder.setBytes(&dims, length: MemoryLayout<SIMD2<UInt32>>.size, index: 4)
        encoder.dispatchThreads(
            MTLSize(width: hidden, height: tokens, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 4, depth: 1))
        encoder.endEncoding()
    }
}
