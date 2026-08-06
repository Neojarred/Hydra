import Foundation
import HydraCore
import HydraFormat
import Metal

/// Enveloppes Swift des opérateurs hors MoE.
///
/// Comme pour MXFP4, ces enveloppes servent d'abord à la **validation** : elles rendent
/// chaque noyau comparable à l'implémentation CPU de `HydraReference`. Le graphe
/// d'inférence encodera ses passes dans un tampon de commandes partagé — un aller-retour
/// CPU-GPU coûte 45 µs, soit plus qu'un noyau, donc les émettre un par un serait ruineux.
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
            case .allocationFailed(let bytes): return "allocation Metal impossible : \(bytes) o"
            case .encodingFailed: return "encodage de la passe Metal impossible"
            case .dimensionMismatch(let detail): return "dimensions incohérentes : \(detail)"
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
            throw KernelError.dimensionMismatch("x et échelle de tailles différentes")
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
            throw KernelError.dimensionMismatch("RoPE : tailles incohérentes")
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
            throw KernelError.dimensionMismatch("SwiGLU attend un nombre pair d'entrées")
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

    /// Attention de décodage sur un cache KV en FP16.
    ///
    /// - Parameters:
    ///   - kCache, vCache: `[capacity][kvHeads][headDim]` en FP16.
    ///   - ringSize: 0 pour un stockage linéaire, sinon la capacité de l'anneau des
    ///     couches à fenêtre glissante.
    ///   - startPosition: position absolue de la première clé visible.
    public func attentionDecode(
        query: [Float], kCache: [Float16], vCache: [Float16], sinks: [Float],
        qHeads: Int, kvHeads: Int, headDim: Int, keyCount: Int,
        ringSize: Int = 0, startPosition: Int = 0, smScale: Float
    ) throws -> [Float] {
        guard query.count == qHeads * headDim, sinks.count == qHeads else {
            throw KernelError.dimensionMismatch("attention : requête ou puits mal dimensionnés")
        }
        guard headDim <= 256 else {
            throw KernelError.dimensionMismatch("headDim > 256 dépasse l'accumulateur du noyau")
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
            encoder.dispatchThreadgroups(
                MTLSize(width: qHeads, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        }
        return read(out, count: qHeads * headDim)
    }

    // MARK: - Routeur

    public func routerTopK(_ logits: [Float], topK: Int) throws -> (indices: [Int], weights: [Float]) {
        guard topK <= 8 else {
            throw KernelError.dimensionMismatch("le noyau du routeur est borné à top-8")
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
