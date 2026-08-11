import Foundation
import HydraCore
import HydraFormat
import Metal

/// The per-chunk buffers prefill's staged path works in, all `[tokens][…]`.
///
/// The decode scratch holds one token's vectors. Prefill used it once per token, which is why
/// phase A issued about 334,000 dispatches for a 557-token prompt — twenty of them a layer a
/// token, each moving a few kilobytes. These buffers hold the whole chunk, so a stage is one
/// dispatch instead of one per token.
///
/// Sized on the widest geometry the model has, not on the common one: Gemma's full-attention
/// layers are 512 wide with 2 KV heads against the sliding layers' 256 with 8, and the query
/// vector is the larger of the two.
public final class Gemma4ChunkScratch {

    public let tokens: Int
    public let paddedTokens: Int

    public let normed: MTLBuffer
    public let query: MTLBuffer
    public let key: MTLBuffer
    public let value: MTLBuffer
    public let attention: MTLBuffer
    public let projected: MTLBuffer
    public let denseGate: MTLBuffer
    public let denseUp: MTLBuffer
    public let denseActivated: MTLBuffer
    public let routerLogits: MTLBuffer

    /// The expert branch, for the tokens of one expert's group: their inputs gathered
    /// contiguously, the two inner projections, and the output before it is scattered back.
    public let expertInput: MTLBuffer
    public let expertGate: MTLBuffer
    public let expertUp: MTLBuffer
    public let expertActivated: MTLBuffer
    public let expertOutput: MTLBuffer

    /// Activations rearranged to `[cols][paddedTokens]` for the batched projection, and the
    /// per-chunk sums of its bias term. One pair, reused by every projection in the layer:
    /// they are consumed by the dispatch that follows them.
    public let transposed: MTLBuffer
    public let sums: MTLBuffer

    /// Sinks the attention kernel can never reach, shared by every token.
    public let unreachableSinks: MTLBuffer

    public init(config: Gemma4Config, tokens: Int, device: MTLDevice) throws {
        self.tokens = tokens
        self.paddedTokens = ForwardEncoder.paddedTokens(tokens)

        let geometries = (0..<config.layerCount).map { config.attentionGeometry(atLayer: $0) }
        let maxQuery = geometries.map(\.queryDim).max() ?? 0
        let maxKV = geometries.map(\.keyValueDim).max() ?? 0
        let widest = max(
            config.hiddenSize, maxQuery, config.intermediateSize, config.moeIntermediateSize)

        func make(_ floats: Int, _ name: String) throws -> MTLBuffer {
            guard let buffer = device.makeBuffer(
                length: max(floats, 1) * 4, options: .storageModeShared)
            else { throw ModelRunner.RunnerError.allocationFailed(name) }
            return buffer
        }

        normed = try make(tokens * config.hiddenSize, "chunk.normed")
        query = try make(tokens * maxQuery, "chunk.query")
        key = try make(tokens * maxKV, "chunk.key")
        value = try make(tokens * maxKV, "chunk.value")
        attention = try make(tokens * maxQuery, "chunk.attention")
        projected = try make(tokens * config.hiddenSize, "chunk.projected")
        denseGate = try make(tokens * config.intermediateSize, "chunk.denseGate")
        denseUp = try make(tokens * config.intermediateSize, "chunk.denseUp")
        denseActivated = try make(tokens * config.intermediateSize, "chunk.denseActivated")
        routerLogits = try make(tokens * config.expertCount, "chunk.routerLogits")

        expertInput = try make(tokens * config.hiddenSize, "chunk.expertInput")
        expertGate = try make(tokens * config.moeIntermediateSize, "chunk.expertGate")
        expertUp = try make(tokens * config.moeIntermediateSize, "chunk.expertUp")
        expertActivated = try make(
            tokens * config.moeIntermediateSize, "chunk.expertActivated")
        expertOutput = try make(tokens * config.hiddenSize, "chunk.expertOutput")

        transposed = try make(widest * paddedTokens, "chunk.transposed")
        // Four bits is the fewest, so the most chunks a column range divides into.
        sums = try make(
            ForwardEncoder.chunkCount(cols: widest, bits: 4) * paddedTokens + paddedTokens,
            "chunk.sums")

        // Shared by every token: the sink is a constant, not per-position.
        unreachableSinks = try make(max(config.attentionHeadCount, 64), "chunk.sinks")
        let sinks = unreachableSinks.contents().bindMemory(
            to: UInt16.self, capacity: config.attentionHeadCount)
        for i in 0..<config.attentionHeadCount { sinks[i] = BF16.fromFloat(-1e30) }
    }
}
