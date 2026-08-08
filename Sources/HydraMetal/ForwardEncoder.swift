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

    private func encode(
        _ function: String, in commandBuffer: MTLCommandBuffer,
        threadgroups: Int, threadsPerThreadgroup: Int,
        _ configure: (MTLComputeCommandEncoder) -> Void
    ) throws {
        let pipeline = try context.pipeline(function)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalContext.ContextError.functionMissing(function)
        }
        encoder.setComputePipelineState(pipeline)
        configure(encoder)
        encoder.dispatchThreadgroups(
            MTLSize(width: threadgroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: min(threadsPerThreadgroup, pipeline.maxTotalThreadsPerThreadgroup),
                height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func encodeLinear(
        _ function: String, in commandBuffer: MTLCommandBuffer, elements: Int,
        _ configure: (MTLComputeCommandEncoder) -> Void
    ) throws {
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

    /// Threadgroup width for a GEMV: enough lanes to cover a row's work groups, rounded up
    /// to a SIMD group.
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

    /// RMSNorm with no learned scale — `v_norm` and the router's, which have no tensor.
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
        var dims = SIMD4<UInt32>(
            UInt32(qHeads), UInt32(kvHeads), UInt32(headDim), UInt32(keyCount))
        var ring = SIMD2<UInt32>(UInt32(ringSize), UInt32(startPosition))
        var scale = smScale
        try encode(
            "attention_decode", in: commandBuffer,
            threadgroups: qHeads, threadsPerThreadgroup: 32
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
