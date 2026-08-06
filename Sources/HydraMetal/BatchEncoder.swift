import Foundation
import HydraCore
import Metal

/// Encode les passes d'un bloc de prefill.
///
/// Même rôle que `ForwardEncoder`, sur des noyaux qui traitent plusieurs jetons à la
/// fois. Les deux coexistent : le décodage reste unitaire par nature — un jeton dépend
/// du précédent — alors que le prefill connaît toute l'invite d'avance.
public struct BatchEncoder: Sendable {

    public let context: MetalContext

    public init(context: MetalContext) {
        self.context = context
    }

    /// Jetons par threadgroup dans les noyaux GEMM. Doit correspondre à `TOKEN_TILE`
    /// côté Metal : c'est le seul couplage entre les deux, et le franchir silencieusement
    /// produirait des résultats partiels.
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

    /// Dimensions de tuilage du GEMM, à garder identiques à celles de `tiled.metal`.
    /// Le trafic mémoire vaut `cols × rows × tokens × (2/tileTokens + 4/tileRows)` :
    /// ce sont ces deux nombres qui déterminent la performance, pas la finesse du
    /// déroulage.
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

    /// Seuil de bascule entre les deux noyaux d'expert.
    ///
    /// Le noyau tuilé réduit le trafic mémoire, mais avec `TILE_ROWS = 128` il ne lance
    /// que `rows/128` threadgroups — 45 pour `gate_up`, très en deçà de ce qu'il faut pour
    /// occuper le GPU. Or en prefill, un expert ne sert qu'une fraction du bloc : environ
    /// huit jetons sur soixante-huit. Sous ce seuil, la variante à une ligne par groupe
    /// SIMD lance trente fois plus de threadgroups et l'emporte largement.
    ///
    /// Le bon noyau dépend donc du nombre de jetons, pas du type d'opération.
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
        in commandBuffer: MTLCommandBuffer
    ) throws {
        var dims = SIMD4<UInt32>(
            UInt32(qHeads), UInt32(kvHeads), UInt32(headDim), UInt32(tokens))
        var window = SIMD4<UInt32>(
            UInt32(ringSize), UInt32(firstPosition), UInt32(slidingWindow), 0)
        var scale = smScale
        try encodeGrid(
            "attention_prefill", in: commandBuffer,
            threadgroups: MTLSize(width: qHeads, height: tokens, depth: 1), width: 32
        ) {
            $0.setBuffer(query, offset: 0, index: 0)
            $0.setBuffer(keyCache, offset: 0, index: 1)
            $0.setBuffer(valueCache, offset: 0, index: 2)
            $0.setBuffer(sinks, offset: sinksOffset, index: 3)
            $0.setBuffer(output, offset: 0, index: 4)
            $0.setBytes(&dims, length: MemoryLayout<SIMD4<UInt32>>.size, index: 5)
            $0.setBytes(&window, length: MemoryLayout<SIMD4<UInt32>>.size, index: 6)
            $0.setBytes(&scale, length: 4, index: 7)
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
}
