import Foundation
import HydraCore
import Metal

/// Encode les passes d'un pas de décodage dans un tampon de commandes **partagé**.
///
/// C'est la différence essentielle avec les enveloppes de test : celles-ci créent un
/// tampon, le valident et attendent, ce qui coûte **45 µs mesurés** par appel. Un token du
/// 20B demande près de deux cents passes ; les émettre séparément coûterait plus de temps
/// en synchronisation qu'en calcul. Ici tout s'accumule dans un tampon que l'appelant
/// valide une fois.
///
/// Le découpage en deux tampons par couche n'est pas un choix esthétique : le CPU doit
/// lire les identifiants d'experts produits par le routeur avant de savoir quels blobs
/// lire sur le SSD. Cette dépendance impose la frontière, et c'est elle qui donne au
/// pipeline sa forme `cb1` → I/O → `cb2`.
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

    /// Largeur de threadgroup pour un GEMV : assez de voies pour couvrir les groupes de
    /// travail d'une ligne, arrondie au groupe SIMD.
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

    /// Projection dense BF16 : `y = W·x + biais`.
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

    /// Projection d'expert MXFP4 : `y = W·x + biais`.
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
        // Variante retenue après mesure appariée : ×1,04 sur la référence, sans changer
        // les sorties (docs/02-MESURES.md, M-005).
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

    /// `out += poids[index] · contribution`. Le poids est lu dans un tampon GPU : les
    /// identifiants d'experts viennent du routeur, et redescendre leurs poids côté CPU
    /// coûterait un aller-retour de synchronisation.
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
