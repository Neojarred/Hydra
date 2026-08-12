import Foundation
import HydraCore
import Metal

/// Encodes the passes of a decoding step into a **shared** command buffer.
///
/// This is the essential difference from the test wrappers: those create a buffer, commit
/// it and wait, which costs **45 µs measured** per call. One 20B token needs close to two
/// hundred passes; issuing them separately would cost more time in synchronization than in
/// compute. Here everything accumulates into one buffer the caller commits once.
///
/// Splitting into two buffers per layer is not an aesthetic choice: the CPU must read the
/// expert identifiers the router produces before it knows which blobs to read from SSD.
/// That dependency imposes the boundary, and it is what gives the pipeline its
/// `cb1` → I/O → `cb2` shape.
public struct ForwardEncoder: Sendable {

    public let context: MetalContext

    public init(context: MetalContext) {
        self.context = context
    }

    // MARK: - Utilitaires d'encodage

    /// Commits a command buffer, closing the shared compute encoder first.
    ///
    /// Dispatches now share one encoder per command buffer, so it is still open when the work
    /// is done being encoded. Committing without closing it is a Metal assertion failure.
    public func commit(_ commandBuffer: MTLCommandBuffer) {
        context.commit(commandBuffer)
    }

    private func encode(
        _ function: String, in commandBuffer: MTLCommandBuffer,
        threadgroups: Int, threadsPerThreadgroup: Int,
        _ configure: (MTLComputeCommandEncoder) -> Void
    ) throws {
        let pipeline = try context.pipeline(function)
        let encoder = try context.sharedEncoder(for: commandBuffer)
        encoder.setComputePipelineState(pipeline)
        configure(encoder)
        encoder.dispatchThreadgroups(
            MTLSize(width: threadgroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: min(threadsPerThreadgroup, pipeline.maxTotalThreadsPerThreadgroup),
                height: 1, depth: 1))
    }

    private func encodeLinear(
        _ function: String, in commandBuffer: MTLCommandBuffer, elements: Int,
        _ configure: (MTLComputeCommandEncoder) -> Void
    ) throws {
        let pipeline = try context.pipeline(function)
        let encoder = try context.sharedEncoder(for: commandBuffer)
        encoder.setComputePipelineState(pipeline)
        configure(encoder)
        encoder.dispatchThreads(
            MTLSize(width: elements, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: min(elements, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
    }

    /// Threadgroup width for a GEMV: enough lanes to cover a row's work groups, rounded up
    /// to a SIMD group.
    /// Rows a threadgroup covers when one simdgroup owns a row. Eight gives 256 threads,
    /// which is the widest a threadgroup runs without spilling occupancy on this family.
    static let rowsPerThreadgroup = 8

    /// Threads a decode-attention threadgroup runs, so eight simdgroups split the key range.
    /// Public because the test harness must dispatch the same shape production does.
    public static let attentionThreads = 256

    /// The widest attention head `attention_decode` can serve, matching
    /// `kMaxAttentionHeadDim` in the shader. Gemma's full-attention layers are 512.
    public static let maxAttentionHeadDim = 512

    /// Experts the router kernels can select for one token, matching `kMaxRouterTopK` in the
    /// shader.
    ///
    /// Qwen3.6-35B-A3B routes to exactly eight, so this is met with no margin. The kernel
    /// clamps to it so it cannot read past its own arrays, which means exceeding it would
    /// route on the first eight and produce plausible wrong output. The dispatch sites refuse
    /// instead. The guard used to exist only in the test harness, which is how a bound met
    /// exactly by production goes unnoticed.
    public static let maxRouterTopK = 8

    private func gemvWidth(units: Int) -> Int {
        min(256, max(32, (units + 31) / 32 * 32))
    }

    // MARK: - Passes

    public func rmsNorm(
        input: MTLBuffer, inputOffset: Int = 0,
        scale: MTLBuffer, scaleOffset: Int,
        output: MTLBuffer, outputOffset: Int = 0,
        size: Int, eps: Float,
        in commandBuffer: MTLCommandBuffer
    ) throws {
        var count = UInt32(size)
        var epsilon = eps
        try encode("rms_norm", in: commandBuffer, threadgroups: 1, threadsPerThreadgroup: 256) {
            $0.setBuffer(input, offset: inputOffset, index: 0)
            $0.setBuffer(scale, offset: scaleOffset, index: 1)
            $0.setBuffer(output, offset: outputOffset, index: 2)
            $0.setBytes(&count, length: 4, index: 3)
            $0.setBytes(&epsilon, length: 4, index: 4)
        }
    }

    /// Dense BF16 projection: `y = W·x + bias`.
    public func denseProjection(
        weights: MTLBuffer, weightsOffset: Int,
        bias: MTLBuffer?, biasOffset: Int,
        input: MTLBuffer, inputOffset: Int,
        output: MTLBuffer, outputOffset: Int,
        rows: Int, cols: Int,
        in commandBuffer: MTLCommandBuffer
    ) throws {
        var dims = SIMD2<UInt32>(UInt32(rows), UInt32(cols))
        var hasBias = UInt32(bias == nil ? 0 : 1)
        try encode(
            "bf16_gemv", in: commandBuffer,
            threadgroups: rows, threadsPerThreadgroup: gemvWidth(units: cols / 8)
        ) {
            $0.setBuffer(weights, offset: weightsOffset, index: 0)
            $0.setBuffer(bias ?? weights, offset: bias == nil ? 0 : biasOffset, index: 1)
            $0.setBuffer(input, offset: inputOffset, index: 2)
            $0.setBuffer(output, offset: outputOffset, index: 3)
            $0.setBytes(&dims, length: MemoryLayout<SIMD2<UInt32>>.size, index: 4)
            $0.setBytes(&hasBias, length: 4, index: 5)
        }
    }

    // MARK: - Gemma 4

    /// RMSNorm with no learned scale, `v_norm` and the router's, which have no tensor.
    public func rmsNormUnscaled(
        input: MTLBuffer, inputOffset: Int = 0,
        output: MTLBuffer, outputOffset: Int = 0,
        size: Int, eps: Float, in commandBuffer: MTLCommandBuffer
    ) throws {
        var count = UInt32(size)
        var epsilon = eps
        try encode(
            "rms_norm_unscaled", in: commandBuffer, threadgroups: 1, threadsPerThreadgroup: 256
        ) {
            $0.setBuffer(input, offset: inputOffset, index: 0)
            $0.setBuffer(output, offset: outputOffset, index: 1)
            $0.setBytes(&count, length: 4, index: 2)
            $0.setBytes(&epsilon, length: 4, index: 3)
        }
    }

    /// `gelu_pytorch_tanh(gate) · up`, over two separate vectors.
    public func geluMultiply(
        gate: MTLBuffer, gateOffset: Int = 0,
        up: MTLBuffer, upOffset: Int = 0,
        output: MTLBuffer, outputOffset: Int = 0,
        size: Int, in commandBuffer: MTLCommandBuffer
    ) throws {
        var count = UInt32(size)
        try encodeLinear("gelu_mul", in: commandBuffer, elements: size) {
            $0.setBuffer(gate, offset: gateOffset, index: 0)
            $0.setBuffer(up, offset: upOffset, index: 1)
            $0.setBuffer(output, offset: outputOffset, index: 2)
            $0.setBytes(&count, length: 4, index: 3)
        }
    }

    /// `x · w · factor`, with `w` in BF16. The router's scale.
    public func scaleByBF16(
        target: MTLBuffer, targetOffset: Int = 0,
        scale: MTLBuffer, scaleOffset: Int, factor: Float, size: Int,
        in commandBuffer: MTLCommandBuffer
    ) throws {
        var count = UInt32(size)
        var multiplier = factor
        try encodeLinear("scale_by_bf16", in: commandBuffer, elements: size) {
            $0.setBuffer(target, offset: targetOffset, index: 0)
            $0.setBuffer(scale, offset: scaleOffset, index: 1)
            $0.setBytes(&count, length: 4, index: 2)
            $0.setBytes(&multiplier, length: 4, index: 3)
        }
    }

    /// `x · s`, with a single BF16 value broadcast over the vector.
    public func scaleByScalar(
        target: MTLBuffer, targetOffset: Int = 0,
        scale: MTLBuffer, scaleOffset: Int, size: Int,
        in commandBuffer: MTLCommandBuffer
    ) throws {
        var count = UInt32(size)
        try encodeLinear("scale_by_bf16_scalar", in: commandBuffer, elements: size) {
            $0.setBuffer(target, offset: targetOffset, index: 0)
            $0.setBuffer(scale, offset: scaleOffset, index: 1)
            $0.setBytes(&count, length: 4, index: 2)
        }
    }

    /// Gemma's router selection: softmax over all experts, top-k, renormalize, per-expert
    /// scale. Not interchangeable with `routerTopK`.
    public func gemmaRouterTopK(
        logits: MTLBuffer, logitsOffset: Int = 0,
        perExpertScale: MTLBuffer, perExpertScaleOffset: Int,
        indices: MTLBuffer, indicesOffset: Int = 0,
        weights: MTLBuffer, weightsOffset: Int = 0,
        expertCount: Int, topK: Int, in commandBuffer: MTLCommandBuffer
    ) throws {
        precondition(
            topK <= Self.maxRouterTopK,
            "the router kernel selects at most \(Self.maxRouterTopK) experts, asked \(topK)")
        var dims = SIMD2<UInt32>(UInt32(expertCount), UInt32(topK))
        try encode(
            "gemma_router_topk", in: commandBuffer, threadgroups: 1, threadsPerThreadgroup: 1
        ) {
            $0.setBuffer(logits, offset: logitsOffset, index: 0)
            $0.setBuffer(perExpertScale, offset: perExpertScaleOffset, index: 1)
            $0.setBuffer(indices, offset: indicesOffset, index: 2)
            $0.setBuffer(weights, offset: weightsOffset, index: 3)
            $0.setBytes(&dims, length: MemoryLayout<SIMD2<UInt32>>.size, index: 4)
        }
    }

    /// Where a projection's weights come from, and how they are decoded.
    ///
    /// The seam that lets one Gemma topology serve two checkpoints. Every projection in the
    /// layer runner, attention, the dense MLP, the router, the experts, the head, is
    /// `y = W · x`; what differs between the BF16 build and the MLX 4-bit one is only how `W`
    /// is written down. Resolving that to a value here keeps the forward pass a single
    /// implementation, which is the one thing D-023 will not trade away: a second copy would
    /// run, return finite numbers, and disagree by a little.
    public enum ProjectionSource {
        case bf16(buffer: MTLBuffer, offset: Int)
        case mlxAffine(
            words: MTLBuffer, wordsOffset: Int,
            scales: MTLBuffer, scalesOffset: Int,
            biases: MTLBuffer, biasesOffset: Int,
            bits: Int, groupSize: Int)
    }

    /// `y = W · x`, whichever way `W` is stored.
    public func encodeProjection(
        _ source: ProjectionSource,
        input: MTLBuffer, inputOffset: Int,
        output: MTLBuffer, outputOffset: Int,
        rows: Int, cols: Int,
        in commandBuffer: MTLCommandBuffer
    ) throws {
        switch source {
        case let .bf16(buffer, offset):
            try denseProjection(
                weights: buffer, weightsOffset: offset, bias: nil, biasOffset: 0,
                input: input, inputOffset: inputOffset,
                output: output, outputOffset: outputOffset,
                rows: rows, cols: cols, in: commandBuffer)
        case let .mlxAffine(words, wordsOffset, scales, scalesOffset, biases, biasesOffset,
                            bits, groupSize):
            try mlxAffineProjection(
                words: words, wordsOffset: wordsOffset,
                scales: scales, scalesOffset: scalesOffset,
                biases: biases, biasesOffset: biasesOffset,
                input: input, inputOffset: inputOffset,
                output: output, outputOffset: outputOffset,
                rows: rows, cols: cols, bits: bits, groupSize: groupSize,
                in: commandBuffer)
        }
    }

    /// `y = W · x` for an MLX affine-quantized matrix: packed values, per-group scale and bias.
    public func mlxAffineProjection(
        words: MTLBuffer, wordsOffset: Int,
        scales: MTLBuffer, scalesOffset: Int,
        biases: MTLBuffer, biasesOffset: Int,
        input: MTLBuffer, inputOffset: Int,
        output: MTLBuffer, outputOffset: Int,
        rows: Int, cols: Int, bits: Int, groupSize: Int,
        in commandBuffer: MTLCommandBuffer
    ) throws {
        var dims = SIMD4<UInt32>(
            UInt32(rows), UInt32(cols), UInt32(bits), UInt32(groupSize))
        // One simdgroup a row, eight rows a threadgroup: the reduction becomes a single
        // `simd_sum` with no barrier, and the threadgroup count drops eightfold.
        // Specialized per width: see the kernel's own note.
        guard bits == 4 || bits == 8 else {
            throw MetalContext.ContextError.functionMissing("mlx_affine_gemv_\(bits)")
        }
        try encode(
            "mlx_affine_gemv_\(bits)", in: commandBuffer,
            threadgroups: (rows + Self.rowsPerThreadgroup - 1) / Self.rowsPerThreadgroup,
            threadsPerThreadgroup: Self.rowsPerThreadgroup * 32
        ) {
            $0.setBuffer(words, offset: wordsOffset, index: 0)
            $0.setBuffer(scales, offset: scalesOffset, index: 1)
            $0.setBuffer(biases, offset: biasesOffset, index: 2)
            $0.setBuffer(input, offset: inputOffset, index: 3)
            $0.setBuffer(output, offset: outputOffset, index: 4)
            $0.setBytes(&dims, length: MemoryLayout<SIMD4<UInt32>>.size, index: 5)
        }
    }

    /// `Y = W · X` for an MLX affine matrix against `tokens` vectors at once.
    ///
    /// Bit-identical to calling `mlxAffineProjection` once per token, the kernel keeps each
    /// token's accumulation order, so it is a drop-in for prefill's per-token loop and the
    /// test can assert equality rather than a tolerance.
    ///
    /// - Parameters:
    ///   - input: `[tokens][cols]`, row-major.
    ///   - output: `[tokens][rows]`, row-major.
    /// Rows a simdgroup carries together. Matches `RB` in the kernel.
    public static let rowBlock = 4

    /// Tokens carried together by the batched projection. Matches `TB` in the kernel.
    public static let batchTile = 8

    /// `tokens` rounded up to the tile, which is how a transposed activation buffer is sized.
    public static func paddedTokens(_ tokens: Int) -> Int {
        (tokens + batchTile - 1) / batchTile * batchTile
    }

    /// Rearranges a chunk's activations from `[tokens][cols]` to `[cols][paddedTokens]`.
    ///
    /// The batched projection wants a column's tile of tokens contiguous. Padding rows are
    /// zeroed so the kernel needs no bounds test.
    public func transposeActivations(
        input: MTLBuffer, inputOffset: Int, output: MTLBuffer, outputOffset: Int,
        tokens: Int, cols: Int, in commandBuffer: MTLCommandBuffer
    ) throws {
        let padded = Self.paddedTokens(tokens)
        var dims = SIMD3<UInt32>(UInt32(tokens), UInt32(cols), UInt32(padded))
        let pipeline = try context.pipeline("transpose_activations")
        let encoder = try context.sharedEncoder(for: commandBuffer)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: inputOffset, index: 0)
        encoder.setBuffer(output, offset: outputOffset, index: 1)
        encoder.setBytes(&dims, length: MemoryLayout<SIMD3<UInt32>>.size, index: 2)
        encoder.dispatchThreads(
            MTLSize(width: cols, height: padded, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 8, depth: 1))
    }

    /// Chunks a column range divides into at a given bit width, which is how `chunkSums` is
    /// sized. A chunk is four 32-bit words, so 32 values at 4 bits and 16 at 8.
    public static func chunkCount(cols: Int, bits: Int) -> Int {
        cols / (32 / bits) / 4
    }

    /// Per-chunk sums of the activations, for the batched projection's bias term.
    ///
    /// `Σx` is row-independent, so computing it once here removes one instruction per value
    /// per token from the projection's inner loop, half of it. Depends on the bit width,
    /// because the chunk's width in columns does.
    public func chunkSums(
        input: MTLBuffer, inputOffset: Int, output: MTLBuffer, outputOffset: Int,
        tokens: Int, cols: Int, bits: Int, in commandBuffer: MTLCommandBuffer
    ) throws {
        let padded = Self.paddedTokens(tokens)
        let chunks = Self.chunkCount(cols: cols, bits: bits)
        guard chunks > 0 else { return }
        var dims = SIMD3<UInt32>(
            UInt32(chunks), UInt32(padded), UInt32(4 * (32 / bits)))
        let pipeline = try context.pipeline("chunk_sums")
        let encoder = try context.sharedEncoder(for: commandBuffer)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: inputOffset, index: 0)
        encoder.setBuffer(output, offset: outputOffset, index: 1)
        encoder.setBytes(&dims, length: MemoryLayout<SIMD3<UInt32>>.size, index: 2)
        encoder.dispatchThreads(
            MTLSize(width: chunks, height: padded, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 8, depth: 1))
    }

    /// - Parameters:
    ///   - input: `[cols][paddedTokens]`, from `transposeActivations`.
    ///   - sums: `[chunks][paddedTokens]`, from `chunkSums` at the same bit width.
    ///   - output: `[tokens][rows]`, row-major.
    public func mlxAffineBatchedProjection(
        words: MTLBuffer, wordsOffset: Int,
        scales: MTLBuffer, scalesOffset: Int,
        biases: MTLBuffer, biasesOffset: Int,
        input: MTLBuffer, inputOffset: Int,
        sums: MTLBuffer, sumsOffset: Int,
        output: MTLBuffer, outputOffset: Int,
        rows: Int, cols: Int, tokens: Int, bits: Int, groupSize: Int,
        in commandBuffer: MTLCommandBuffer
    ) throws {
        guard bits == 4 || bits == 8 else {
            throw MetalContext.ContextError.functionMissing("mlx_affine_gemm_\(bits)")
        }
        var dims = SIMD4<UInt32>(
            UInt32(rows), UInt32(cols), UInt32(tokens), UInt32(groupSize))
        var padded = UInt32(Self.paddedTokens(tokens))
        try encode(
            "mlx_affine_gemm_\(bits)", in: commandBuffer,
            threadgroups: (rows + Self.rowsPerThreadgroup * Self.rowBlock - 1)
                / (Self.rowsPerThreadgroup * Self.rowBlock),
            threadsPerThreadgroup: Self.rowsPerThreadgroup * 32
        ) {
            $0.setBuffer(words, offset: wordsOffset, index: 0)
            $0.setBuffer(scales, offset: scalesOffset, index: 1)
            $0.setBuffer(biases, offset: biasesOffset, index: 2)
            $0.setBuffer(input, offset: inputOffset, index: 3)
            $0.setBuffer(output, offset: outputOffset, index: 4)
            $0.setBytes(&dims, length: MemoryLayout<SIMD4<UInt32>>.size, index: 5)
            $0.setBytes(&padded, length: 4, index: 6)
            $0.setBuffer(sums, offset: sumsOffset, index: 7)
        }
    }

    /// `hidden += rmsNorm(x, scale)` and `residual = hidden`, in one dispatch.
    public func fusedNormAddCopy(
        input: MTLBuffer, scale: MTLBuffer, scaleOffset: Int,
        hidden: MTLBuffer, residual: MTLBuffer,
        size: Int, eps: Float, in commandBuffer: MTLCommandBuffer
    ) throws {
        var count = UInt32(size)
        var epsilon = eps
        try encode(
            "fused_norm_add_copy", in: commandBuffer,
            threadgroups: 1, threadsPerThreadgroup: 256
        ) {
            $0.setBuffer(input, offset: 0, index: 0)
            $0.setBuffer(scale, offset: scaleOffset, index: 1)
            $0.setBuffer(hidden, offset: 0, index: 2)
            $0.setBuffer(residual, offset: 0, index: 3)
            $0.setBytes(&count, length: 4, index: 4)
            $0.setBytes(&epsilon, length: 4, index: 5)
        }
    }

    /// `out = rmsNorm(x) · scale · factor`, the router's input, in one dispatch.
    public func fusedUnscaledNormScale(
        input: MTLBuffer, scale: MTLBuffer, scaleOffset: Int, output: MTLBuffer,
        size: Int, eps: Float, factor: Float, in commandBuffer: MTLCommandBuffer
    ) throws {
        var count = UInt32(size)
        var epsilon = eps
        var multiplier = factor
        try encode(
            "fused_unscaled_norm_scale", in: commandBuffer,
            threadgroups: 1, threadsPerThreadgroup: 256
        ) {
            $0.setBuffer(input, offset: 0, index: 0)
            $0.setBuffer(scale, offset: scaleOffset, index: 1)
            $0.setBuffer(output, offset: 0, index: 2)
            $0.setBytes(&count, length: 4, index: 3)
            $0.setBytes(&epsilon, length: 4, index: 4)
            $0.setBytes(&multiplier, length: 4, index: 5)
        }
    }

    /// RMS-normalizes every head of a vector in one dispatch, one threadgroup to a head.
    ///
    /// - Parameter scale: the per-head weight, shared across heads, or `nil` for `v_norm`,
    ///   which is built without one and has no tensor in the checkpoint.
    public func rmsNormHeads(
        vector: MTLBuffer, scale: MTLBuffer?, scaleOffset: Int,
        heads: Int, headDim: Int, eps: Float, in commandBuffer: MTLCommandBuffer
    ) throws {
        var dims = SIMD2<UInt32>(UInt32(headDim), scale == nil ? 0 : 1)
        var epsilon = eps
        try encode(
            "rms_norm_heads", in: commandBuffer,
            threadgroups: heads, threadsPerThreadgroup: min(256, max(32, headDim))
        ) {
            $0.setBuffer(vector, offset: 0, index: 0)
            // A kernel argument must be bound even when the flag says it is unused.
            $0.setBuffer(scale ?? vector, offset: scale == nil ? 0 : scaleOffset, index: 1)
            $0.setBytes(&dims, length: MemoryLayout<SIMD2<UInt32>>.size, index: 2)
            $0.setBytes(&epsilon, length: 4, index: 3)
        }
    }

    /// True when a projection source has a batched form. Only the MLX affine build does;
    /// BF16 has no GEMM and its caller falls back to the per-token path.
    public static func supportsBatching(_ source: ProjectionSource) -> Bool {
        if case .mlxAffine = source { return true }
        return false
    }

    /// The bit width a projection decodes at, or nil if it has no batched form.
    public static func bits(of source: ProjectionSource) -> Int? {
        if case .mlxAffine(_, _, _, _, _, _, let bits, _) = source { return bits }
        return nil
    }

    /// `Y = W · X` over a chunk, from a resolved projection source.
    public func encodeBatchedProjection(
        _ source: ProjectionSource,
        input: MTLBuffer, sums: MTLBuffer, output: MTLBuffer,
        rows: Int, cols: Int, tokens: Int, in commandBuffer: MTLCommandBuffer
    ) throws {
        guard case .mlxAffine(
            let words, let wordsOffset, let scales, let scalesOffset,
            let biases, let biasesOffset, let bits, let groupSize) = source
        else { throw MetalContext.ContextError.functionMissing("mlx_affine_gemm") }
        try mlxAffineBatchedProjection(
            words: words, wordsOffset: wordsOffset, scales: scales, scalesOffset: scalesOffset,
            biases: biases, biasesOffset: biasesOffset, input: input, inputOffset: 0,
            sums: sums, sumsOffset: 0, output: output, outputOffset: 0,
            rows: rows, cols: cols, tokens: tokens, bits: bits, groupSize: groupSize,
            in: commandBuffer)
    }

    // MARK: - Batched forms, for prefill

    /// Attention for a whole prefill chunk, one threadgroup a (token, head).
    ///
    /// - Parameter window: the sliding bound, or 0 for a full-attention layer. It has to
    ///   reproduce `KVCache.visibleRange`, which is where it comes from.
    public func attentionPrefill(
        query: MTLBuffer, keyCache: MTLBuffer, valueCache: MTLBuffer,
        sinks: MTLBuffer, sinksOffset: Int, output: MTLBuffer,
        qHeads: Int, kvHeads: Int, headDim: Int, tokens: Int,
        firstPosition: Int, window: Int, ringSize: Int, smScale: Float,
        in commandBuffer: MTLCommandBuffer
    ) throws {
        precondition(
            headDim <= Self.maxAttentionHeadDim,
            "headDim \(headDim) exceeds attention_prefill's accumulator")
        var dims = SIMD4<UInt32>(
            UInt32(qHeads), UInt32(kvHeads), UInt32(headDim), UInt32(tokens))
        var span = SIMD4<UInt32>(
            UInt32(ringSize), UInt32(firstPosition), UInt32(window), 0)
        var scale = smScale
        let pipeline = try context.pipeline("attention_prefill")
        let encoder = try context.sharedEncoder(for: commandBuffer)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(query, offset: 0, index: 0)
        encoder.setBuffer(keyCache, offset: 0, index: 1)
        encoder.setBuffer(valueCache, offset: 0, index: 2)
        encoder.setBuffer(sinks, offset: sinksOffset, index: 3)
        encoder.setBuffer(output, offset: 0, index: 4)
        encoder.setBytes(&dims, length: MemoryLayout<SIMD4<UInt32>>.size, index: 5)
        encoder.setBytes(&span, length: MemoryLayout<SIMD4<UInt32>>.size, index: 6)
        encoder.setBytes(&scale, length: 4, index: 7)
        encoder.dispatchThreadgroups(
            MTLSize(width: qHeads, height: tokens, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: min(Self.attentionThreads, pipeline.maxTotalThreadsPerThreadgroup),
                height: 1, depth: 1))
    }

    /// `rmsNorm` over a chunk, one threadgroup a token.
    ///
    /// Same thread count and reduction order as the single-token kernel, so a chunk's result
    /// is bit-identical to normalizing its tokens one at a time, which prefill's contract
    /// requires.
    public func rmsNormBatched(
        input: MTLBuffer, inputOffset: Int, scale: MTLBuffer, scaleOffset: Int,
        output: MTLBuffer, outputOffset: Int,
        tokens: Int, size: Int, eps: Float, in commandBuffer: MTLCommandBuffer
    ) throws {
        var count = UInt32(size)
        var epsilon = eps
        try encode(
            "rms_norm_batched", in: commandBuffer,
            threadgroups: tokens, threadsPerThreadgroup: 256
        ) {
            $0.setBuffer(input, offset: inputOffset, index: 0)
            $0.setBuffer(scale, offset: scaleOffset, index: 1)
            $0.setBuffer(output, offset: outputOffset, index: 2)
            $0.setBytes(&count, length: 4, index: 3)
            $0.setBytes(&epsilon, length: 4, index: 4)
        }
    }

    /// `fusedNormAddCopy` over a chunk, one threadgroup a token.
    public func fusedNormAddCopyBatched(
        input: MTLBuffer, inputOffset: Int, scale: MTLBuffer, scaleOffset: Int,
        hidden: MTLBuffer, hiddenOffset: Int, residual: MTLBuffer, residualOffset: Int,
        tokens: Int, size: Int, eps: Float, in commandBuffer: MTLCommandBuffer
    ) throws {
        var count = UInt32(size)
        var epsilon = eps
        try encode(
            "fused_norm_add_copy_batched", in: commandBuffer,
            threadgroups: tokens, threadsPerThreadgroup: 256
        ) {
            $0.setBuffer(input, offset: inputOffset, index: 0)
            $0.setBuffer(scale, offset: scaleOffset, index: 1)
            $0.setBuffer(hidden, offset: hiddenOffset, index: 2)
            $0.setBuffer(residual, offset: residualOffset, index: 3)
            $0.setBytes(&count, length: 4, index: 4)
            $0.setBytes(&epsilon, length: 4, index: 5)
        }
    }

    /// `fusedUnscaledNormScale` over a chunk, one threadgroup a token.
    public func fusedUnscaledNormScaleBatched(
        input: MTLBuffer, inputOffset: Int, scale: MTLBuffer, scaleOffset: Int,
        output: MTLBuffer, outputOffset: Int,
        tokens: Int, size: Int, eps: Float, factor: Float,
        in commandBuffer: MTLCommandBuffer
    ) throws {
        var count = UInt32(size)
        var epsilon = eps
        var multiplier = factor
        try encode(
            "fused_unscaled_norm_scale_batched", in: commandBuffer,
            threadgroups: tokens, threadsPerThreadgroup: 256
        ) {
            $0.setBuffer(input, offset: inputOffset, index: 0)
            $0.setBuffer(scale, offset: scaleOffset, index: 1)
            $0.setBuffer(output, offset: outputOffset, index: 2)
            $0.setBytes(&count, length: 4, index: 3)
            $0.setBytes(&epsilon, length: 4, index: 4)
            $0.setBytes(&multiplier, length: 4, index: 5)
        }
    }

    /// `applyRoPE` over a chunk, each token against its own rotary table.
    public func applyRoPEBatched(
        vector: MTLBuffer, vectorOffset: Int,
        cos: MTLBuffer, sin: MTLBuffer, tableStride: Int,
        tokens: Int, heads: Int, headDim: Int, in commandBuffer: MTLCommandBuffer
    ) throws {
        var dims = SIMD4<UInt32>(
            UInt32(heads), UInt32(headDim), UInt32(headDim / 2), UInt32(tableStride))
        let pipeline = try context.pipeline("rope_apply_batched")
        let encoder = try context.sharedEncoder(for: commandBuffer)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(vector, offset: vectorOffset, index: 0)
        encoder.setBuffer(cos, offset: 0, index: 1)
        encoder.setBuffer(sin, offset: 0, index: 2)
        encoder.setBytes(&dims, length: MemoryLayout<SIMD4<UInt32>>.size, index: 3)
        encoder.dispatchThreads(
            MTLSize(width: heads * headDim / 2, height: tokens, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 4, depth: 1))
    }

    /// `cap · tanh(logits / cap)`, in place.
    public func softcapLogits(
        _ logits: MTLBuffer, offset: Int = 0, size: Int, cap: Float,
        in commandBuffer: MTLCommandBuffer
    ) throws {
        var count = UInt32(size)
        var limit = cap
        try encodeLinear("logit_softcap", in: commandBuffer, elements: size) {
            $0.setBuffer(logits, offset: offset, index: 0)
            $0.setBytes(&count, length: 4, index: 1)
            $0.setBytes(&limit, length: 4, index: 2)
        }
    }

    /// MXFP4 expert projection: `y = W·x + bias`.
    public func expertProjection(
        blocks: MTLBuffer, blocksOffset: Int,
        scales: MTLBuffer, scalesOffset: Int,
        bias: MTLBuffer?, biasOffset: Int,
        input: MTLBuffer, inputOffset: Int,
        output: MTLBuffer, outputOffset: Int,
        rows: Int, cols: Int,
        in commandBuffer: MTLCommandBuffer
    ) throws {
        var dims = SIMD2<UInt32>(UInt32(rows), UInt32(cols))
        var hasBias = UInt32(bias == nil ? 0 : 1)
        // Variant chosen after a paired measurement: ×1.04 over the reference, with no
        // change in outputs (docs/02-MEASUREMENTS.md, M-005).
        try encode(
            "mxfp4_gemv_vectorized", in: commandBuffer,
            threadgroups: rows, threadsPerThreadgroup: gemvWidth(units: cols / 32)
        ) {
            $0.setBuffer(blocks, offset: blocksOffset, index: 0)
            $0.setBuffer(scales, offset: scalesOffset, index: 1)
            $0.setBuffer(bias ?? blocks, offset: bias == nil ? 0 : biasOffset, index: 2)
            $0.setBuffer(input, offset: inputOffset, index: 3)
            $0.setBuffer(output, offset: outputOffset, index: 4)
            $0.setBytes(&dims, length: MemoryLayout<SIMD2<UInt32>>.size, index: 5)
            $0.setBytes(&hasBias, length: 4, index: 6)
        }
    }

    public func applyRoPE(
        vector: MTLBuffer, vectorOffset: Int,
        cos: MTLBuffer, sin: MTLBuffer, tableOffset: Int,
        heads: Int, headDim: Int,
        in commandBuffer: MTLCommandBuffer
    ) throws {
        var dims = SIMD2<UInt32>(UInt32(heads), UInt32(headDim))
        try encodeLinear("rope_apply", in: commandBuffer, elements: heads * headDim / 2) {
            $0.setBuffer(vector, offset: vectorOffset, index: 0)
            $0.setBuffer(cos, offset: tableOffset, index: 1)
            $0.setBuffer(sin, offset: tableOffset, index: 2)
            $0.setBytes(&dims, length: MemoryLayout<SIMD2<UInt32>>.size, index: 3)
        }
    }

    public func writeKeyValue(
        key: MTLBuffer, keyOffset: Int, value: MTLBuffer, valueOffset: Int,
        keyCache: MTLBuffer, valueCache: MTLBuffer,
        kvHeads: Int, headDim: Int, position: Int, ringSize: Int,
        in commandBuffer: MTLCommandBuffer
    ) throws {
        var dims = SIMD2<UInt32>(UInt32(kvHeads), UInt32(headDim))
        var slot = SIMD2<UInt32>(UInt32(position), UInt32(ringSize))
        try encodeLinear("kv_cache_write", in: commandBuffer, elements: kvHeads * headDim) {
            $0.setBuffer(key, offset: keyOffset, index: 0)
            $0.setBuffer(value, offset: valueOffset, index: 1)
            $0.setBuffer(keyCache, offset: 0, index: 2)
            $0.setBuffer(valueCache, offset: 0, index: 3)
            $0.setBytes(&dims, length: MemoryLayout<SIMD2<UInt32>>.size, index: 4)
            $0.setBytes(&slot, length: MemoryLayout<SIMD2<UInt32>>.size, index: 5)
        }
    }

    public func attention(
        query: MTLBuffer, queryOffset: Int,
        keyCache: MTLBuffer, valueCache: MTLBuffer,
        sinks: MTLBuffer, sinksOffset: Int,
        output: MTLBuffer, outputOffset: Int,
        qHeads: Int, kvHeads: Int, headDim: Int, keyCount: Int,
        ringSize: Int, startPosition: Int, smScale: Float,
        in commandBuffer: MTLCommandBuffer
    ) throws {
        precondition(
            headDim <= Self.maxAttentionHeadDim,
            "headDim \(headDim) exceeds attention_decode's accumulator")
        var dims = SIMD4<UInt32>(
            UInt32(qHeads), UInt32(kvHeads), UInt32(headDim), UInt32(keyCount))
        var ring = SIMD2<UInt32>(UInt32(ringSize), UInt32(startPosition))
        var scale = smScale
        try encode(
            // Eight simdgroups split the key range. Clamped by `encode` if the pipeline
            // allows fewer; the kernel reads its own simdgroup count and adapts.
            "attention_decode", in: commandBuffer,
            threadgroups: qHeads, threadsPerThreadgroup: Self.attentionThreads
        ) {
            $0.setBuffer(query, offset: queryOffset, index: 0)
            $0.setBuffer(keyCache, offset: 0, index: 1)
            $0.setBuffer(valueCache, offset: 0, index: 2)
            $0.setBuffer(sinks, offset: sinksOffset, index: 3)
            $0.setBuffer(output, offset: outputOffset, index: 4)
            $0.setBytes(&dims, length: MemoryLayout<SIMD4<UInt32>>.size, index: 5)
            $0.setBytes(&ring, length: MemoryLayout<SIMD2<UInt32>>.size, index: 6)
            $0.setBytes(&scale, length: 4, index: 7)
        }
    }

    public func swiglu(
        input: MTLBuffer, inputOffset: Int,
        output: MTLBuffer, outputOffset: Int,
        size: Int, alpha: Float, limit: Float,
        in commandBuffer: MTLCommandBuffer
    ) throws {
        var count = UInt32(size)
        var params = SIMD2<Float>(alpha, limit)
        try encodeLinear("swiglu", in: commandBuffer, elements: size) {
            $0.setBuffer(input, offset: inputOffset, index: 0)
            $0.setBuffer(output, offset: outputOffset, index: 1)
            $0.setBytes(&count, length: 4, index: 2)
            $0.setBytes(&params, length: MemoryLayout<SIMD2<Float>>.size, index: 3)
        }
    }

    public func routerTopK(
        logits: MTLBuffer, logitsOffset: Int,
        indices: MTLBuffer, weights: MTLBuffer,
        expertCount: Int, topK: Int,
        in commandBuffer: MTLCommandBuffer
    ) throws {
        precondition(
            topK <= Self.maxRouterTopK,
            "the router kernel selects at most \(Self.maxRouterTopK) experts, asked \(topK)")
        var dims = SIMD2<UInt32>(UInt32(expertCount), UInt32(topK))
        try encode("router_topk", in: commandBuffer, threadgroups: 1, threadsPerThreadgroup: 32) {
            $0.setBuffer(logits, offset: logitsOffset, index: 0)
            $0.setBuffer(indices, offset: 0, index: 1)
            $0.setBuffer(weights, offset: 0, index: 2)
            $0.setBytes(&dims, length: MemoryLayout<SIMD2<UInt32>>.size, index: 3)
        }
    }

    public func addInPlace(
        target: MTLBuffer, targetOffset: Int,
        addend: MTLBuffer, addendOffset: Int, size: Int,
        in commandBuffer: MTLCommandBuffer
    ) throws {
        var count = UInt32(size)
        try encodeLinear("add_inplace", in: commandBuffer, elements: size) {
            $0.setBuffer(target, offset: targetOffset, index: 0)
            $0.setBuffer(addend, offset: addendOffset, index: 1)
            $0.setBytes(&count, length: 4, index: 2)
        }
    }

    public func copy(
        into destination: MTLBuffer, destinationOffset: Int,
        from source: MTLBuffer, sourceOffset: Int, size: Int,
        in commandBuffer: MTLCommandBuffer
    ) throws {
        var count = UInt32(size)
        try encodeLinear("copy_buffer", in: commandBuffer, elements: size) {
            $0.setBuffer(destination, offset: destinationOffset, index: 0)
            $0.setBuffer(source, offset: sourceOffset, index: 1)
            $0.setBytes(&count, length: 4, index: 2)
        }
    }

    public func fillZero(
        _ buffer: MTLBuffer, offset: Int, size: Int, in commandBuffer: MTLCommandBuffer
    ) throws {
        var count = UInt32(size)
        try encodeLinear("fill_zero", in: commandBuffer, elements: size) {
            $0.setBuffer(buffer, offset: offset, index: 0)
            $0.setBytes(&count, length: 4, index: 1)
        }
    }

    /// `out += weight[index] · contribution`. The weight is read from a GPU buffer: the
    /// expert identifiers come from the router, and bringing their weights back to the CPU
    /// would cost a synchronization round trip.
    public func writeExpertScaled(
        into output: MTLBuffer, outputOffset: Int,
        contribution: MTLBuffer, contributionOffset: Int,
        weights: MTLBuffer, weightIndex: Int, size: Int,
        in commandBuffer: MTLCommandBuffer
    ) throws {
        var dims = SIMD2<UInt32>(UInt32(size), UInt32(weightIndex))
        try encodeLinear("write_expert_scaled", in: commandBuffer, elements: size) {
            $0.setBuffer(output, offset: outputOffset, index: 0)
            $0.setBuffer(contribution, offset: contributionOffset, index: 1)
            $0.setBuffer(weights, offset: 0, index: 2)
            $0.setBytes(&dims, length: MemoryLayout<SIMD2<UInt32>>.size, index: 3)
        }
    }

    public func sumExpertSlices(
        into output: MTLBuffer, slices: MTLBuffer, size: Int, count: Int,
        in commandBuffer: MTLCommandBuffer
    ) throws {
        var dims = SIMD2<UInt32>(UInt32(size), UInt32(count))
        try encodeLinear("sum_expert_slices", in: commandBuffer, elements: size) {
            $0.setBuffer(output, offset: 0, index: 0)
            $0.setBuffer(slices, offset: 0, index: 1)
            $0.setBytes(&dims, length: MemoryLayout<SIMD2<UInt32>>.size, index: 2)
        }
    }

    public func accumulateExpert(
        into output: MTLBuffer, outputOffset: Int,
        contribution: MTLBuffer, contributionOffset: Int,
        weights: MTLBuffer, weightIndex: Int, size: Int,
        in commandBuffer: MTLCommandBuffer
    ) throws {
        var dims = SIMD2<UInt32>(UInt32(size), UInt32(weightIndex))
        try encodeLinear("accumulate_expert", in: commandBuffer, elements: size) {
            $0.setBuffer(output, offset: outputOffset, index: 0)
            $0.setBuffer(contribution, offset: contributionOffset, index: 1)
            $0.setBuffer(weights, offset: 0, index: 2)
            $0.setBytes(&dims, length: MemoryLayout<SIMD2<UInt32>>.size, index: 3)
        }
    }
}
