import Foundation
import HydraCore
import HydraFormat
import Metal

/// Swift wrappers around the non-MoE operators.
///
/// As with MXFP4, these wrappers serve **validation** first: they make every kernel
/// comparable with `HydraReference`'s CPU implementation. The inference graph encodes its
/// passes into a shared command buffer, a CPU-GPU round trip costs 45 µs, more than a
/// kernel itself, so issuing them one by one would be ruinous.
public struct AttentionKernels: Sendable {

    public let context: MetalContext

    public init(context: MetalContext) {
        self.context = context
    }

    public enum KernelError: Error, CustomStringConvertible {
        case allocationFailed(bytes: Int)
        case encodingFailed
        case dimensionMismatch(String)

        public var description: String {
            switch self {
            case .allocationFailed(let bytes): return "cannot allocate \(bytes) bytes on the GPU"
            case .encodingFailed: return "cannot encode the Metal pass"
            case .dimensionMismatch(let detail): return "inconsistent dimensions: \(detail)"
            }
        }
    }

    private func buffer(_ values: [Float]) throws -> MTLBuffer {
        try buffer(bytes: values.withUnsafeBytes { Data($0) })
    }

    private func buffer(bytes: Data) throws -> MTLBuffer {
        guard let out = bytes.withUnsafeBytes({ raw in
            context.device.makeBuffer(
                bytes: raw.baseAddress!, length: max(raw.count, 4), options: .storageModeShared)
        }) else {
            throw KernelError.allocationFailed(bytes: bytes.count)
        }
        return out
    }

    private func output(count: Int) throws -> MTLBuffer {
        guard let out = context.device.makeBuffer(
            length: max(count * MemoryLayout<Float>.size, 4), options: .storageModeShared)
        else {
            throw KernelError.allocationFailed(bytes: count * 4)
        }
        return out
    }

    private func run(_ encode: (MTLComputeCommandEncoder) -> Void) throws {
        guard let commandBuffer = context.commandQueue.makeCommandBuffer(),
            let encoder = commandBuffer.makeComputeCommandEncoder()
        else { throw KernelError.encodingFailed }
        encode(encoder)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }

    private func read(_ buffer: MTLBuffer, count: Int) -> [Float] {
        Array(
            UnsafeBufferPointer(
                start: buffer.contents().bindMemory(to: Float.self, capacity: count), count: count))
    }

    // MARK: - RMSNorm

    public func rmsNorm(_ x: [Float], scale: [Float], eps: Float = 1e-5) throws -> [Float] {
        guard x.count == scale.count else {
            throw KernelError.dimensionMismatch("x and scale have different sizes")
        }
        let pipeline = try context.pipeline("rms_norm")
        let xBuffer = try buffer(x)
        let scaleBuffer = try buffer(bytes: BF16.encode(scale))
        let out = try output(count: x.count)

        var size = UInt32(x.count)
        var epsilon = eps
        try run { encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(xBuffer, offset: 0, index: 0)
            encoder.setBuffer(scaleBuffer, offset: 0, index: 1)
            encoder.setBuffer(out, offset: 0, index: 2)
            encoder.setBytes(&size, length: 4, index: 3)
            encoder.setBytes(&epsilon, length: 4, index: 4)
            encoder.dispatchThreadgroups(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        return read(out, count: x.count)
    }

    // MARK: - RoPE

    public func applyRoPE(
        _ x: [Float], heads: Int, headDim: Int, cos: [Float], sin: [Float]
    ) throws -> [Float] {
        guard x.count == heads * headDim, cos.count == headDim / 2, sin.count == headDim / 2 else {
            throw KernelError.dimensionMismatch("RoPE: inconsistent sizes")
        }
        let pipeline = try context.pipeline("rope_apply")
        let xBuffer = try buffer(x)
        let cosBuffer = try buffer(cos)
        let sinBuffer = try buffer(sin)

        var dims = SIMD2<UInt32>(UInt32(heads), UInt32(headDim))
        let threads = heads * headDim / 2
        try run { encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(xBuffer, offset: 0, index: 0)
            encoder.setBuffer(cosBuffer, offset: 0, index: 1)
            encoder.setBuffer(sinBuffer, offset: 0, index: 2)
            encoder.setBytes(&dims, length: MemoryLayout<SIMD2<UInt32>>.size, index: 3)
            encoder.dispatchThreads(
                MTLSize(width: threads, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(
                    width: min(threads, pipeline.maxTotalThreadsPerThreadgroup),
                    height: 1, depth: 1))
        }
        return read(xBuffer, count: x.count)
    }

    // MARK: - SwiGLU

    public func swiglu(_ x: [Float], alpha: Float = 1.702, limit: Float = 7.0) throws -> [Float] {
        guard x.count % 2 == 0 else {
            throw KernelError.dimensionMismatch("SwiGLU expects an even number of inputs")
        }
        let size = x.count / 2
        let pipeline = try context.pipeline("swiglu")
        let xBuffer = try buffer(x)
        let out = try output(count: size)

        var count = UInt32(size)
        var params = SIMD2<Float>(alpha, limit)
        try run { encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(xBuffer, offset: 0, index: 0)
            encoder.setBuffer(out, offset: 0, index: 1)
            encoder.setBytes(&count, length: 4, index: 2)
            encoder.setBytes(&params, length: MemoryLayout<SIMD2<Float>>.size, index: 3)
            encoder.dispatchThreads(
                MTLSize(width: size, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(
                    width: min(size, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        }
        return read(out, count: size)
    }

    // MARK: - Attention

    /// Decoding attention over an FP16 KV cache.
    ///
    /// - Parameters:
    ///   - kCache, vCache: `[capacity][kvHeads][headDim]` en FP16.
    ///   - ringSize: 0 for linear storage, otherwise the ring capacity of the
    ///     sliding-window layers.
    ///   - startPosition: absolute position of the first visible key.
    public func attentionDecode(
        query: [Float], kCache: [Float16], vCache: [Float16], sinks: [Float],
        qHeads: Int, kvHeads: Int, headDim: Int, keyCount: Int,
        ringSize: Int = 0, startPosition: Int = 0, smScale: Float
    ) throws -> [Float] {
        guard query.count == qHeads * headDim, sinks.count == qHeads else {
            throw KernelError.dimensionMismatch("attention: query or sinks badly sized")
        }
        // The bound the kernel actually has. It read 256 while Gemma's full-attention layers
        // are 512 wide, so this rejected the one geometry that needed testing, and the
        // production path, which has no such guard, passed 512 through to a kernel sized for
        // 256. A limit asserted only where it is not exceeded protects nothing.
        guard headDim <= ForwardEncoder.maxAttentionHeadDim else {
            throw KernelError.dimensionMismatch(
                "headDim \(headDim) exceeds the kernel accumulator")
        }
        let pipeline = try context.pipeline("attention_decode")
        let qBuffer = try buffer(query)
        let kBuffer = try buffer(bytes: kCache.withUnsafeBytes { Data($0) })
        let vBuffer = try buffer(bytes: vCache.withUnsafeBytes { Data($0) })
        let sinkBuffer = try buffer(bytes: BF16.encode(sinks))
        let out = try output(count: qHeads * headDim)

        var dims = SIMD4<UInt32>(
            UInt32(qHeads), UInt32(kvHeads), UInt32(headDim), UInt32(keyCount))
        var ring = SIMD2<UInt32>(UInt32(ringSize), UInt32(startPosition))
        var scale = smScale
        try run { encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(qBuffer, offset: 0, index: 0)
            encoder.setBuffer(kBuffer, offset: 0, index: 1)
            encoder.setBuffer(vBuffer, offset: 0, index: 2)
            encoder.setBuffer(sinkBuffer, offset: 0, index: 3)
            encoder.setBuffer(out, offset: 0, index: 4)
            encoder.setBytes(&dims, length: MemoryLayout<SIMD4<UInt32>>.size, index: 5)
            encoder.setBytes(&ring, length: MemoryLayout<SIMD2<UInt32>>.size, index: 6)
            encoder.setBytes(&scale, length: 4, index: 7)
            // Must match what `ForwardEncoder.attention` dispatches. It hardcoded 32, one
            // simdgroup, while production ran 256, so every test through this harness
            // validated a configuration the model never uses. That is how a NaN in the
            // split-K merge reached the model with the whole attention suite green.
            encoder.dispatchThreadgroups(
                MTLSize(width: qHeads, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(
                    width: min(ForwardEncoder.attentionThreads,
                        pipeline.maxTotalThreadsPerThreadgroup),
                    height: 1, depth: 1))
        }
        return read(out, count: qHeads * headDim)
    }

    // MARK: - Routeur

    public func routerTopK(_ logits: [Float], topK: Int) throws -> (indices: [Int], weights: [Float]) {
        guard topK <= ForwardEncoder.maxRouterTopK else {
            throw KernelError.dimensionMismatch(
                "the router kernel selects at most \(ForwardEncoder.maxRouterTopK) experts")
        }
        let pipeline = try context.pipeline("router_topk")
        let logitBuffer = try buffer(logits)
        guard let indexBuffer = context.device.makeBuffer(
            length: topK * MemoryLayout<UInt32>.size, options: .storageModeShared)
        else { throw KernelError.allocationFailed(bytes: topK * 4) }
        let weightBuffer = try output(count: topK)

        var dims = SIMD2<UInt32>(UInt32(logits.count), UInt32(topK))
        try run { encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(logitBuffer, offset: 0, index: 0)
            encoder.setBuffer(indexBuffer, offset: 0, index: 1)
            encoder.setBuffer(weightBuffer, offset: 0, index: 2)
            encoder.setBytes(&dims, length: MemoryLayout<SIMD2<UInt32>>.size, index: 3)
            encoder.dispatchThreadgroups(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        }

        let raw = UnsafeBufferPointer(
            start: indexBuffer.contents().bindMemory(to: UInt32.self, capacity: topK), count: topK)
        return (raw.map(Int.init), read(weightBuffer, count: topK))
    }
}
