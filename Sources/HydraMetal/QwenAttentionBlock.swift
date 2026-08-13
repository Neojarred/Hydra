import Foundation
import HydraCore
import Metal

/// One Qwen full-attention block, encoded onto the GPU.
///
/// The other kind of layer, one in four. Every kernel it uses already existed for Gemma; what
/// differs is the gated query projection and, quietly, the scale.
///
/// **The attention scale is `1/sqrt(headDim)`.** Gemma passes 1.0 because its query norm absorbs
/// it (D-022), and borrowing that constant here divides every logit by sixteen at a 256-wide
/// head, which softmax turns into a flatter distribution rather than an obvious fault (D-027).
public struct QwenAttentionBlock {

    public final class Scratch {
        public let normed: MTLBuffer
        /// `q_proj`'s output, query and gate interleaved per head.
        public let combined: MTLBuffer
        public let query: MTLBuffer
        public let gate: MTLBuffer
        public let key: MTLBuffer
        public let value: MTLBuffer
        public let attended: MTLBuffer
        public let projected: MTLBuffer

        public init(config: Qwen35MoeConfig, device: MTLDevice) throws {
            func make(_ floats: Int, _ name: String) throws -> MTLBuffer {
                guard let buffer = device.makeBuffer(
                    length: max(floats, 1) * 4, options: .storageModeShared)
                else { throw ModelRunner.RunnerError.allocationFailed(name) }
                return buffer
            }
            normed = try make(config.hiddenSize, "qwen.attn.normed")
            combined = try make(config.queryProjectionRows, "qwen.attn.combined")
            query = try make(config.queryDim, "qwen.attn.query")
            gate = try make(config.queryDim, "qwen.attn.gate")
            key = try make(config.keyValueDim, "qwen.attn.key")
            value = try make(config.keyValueDim, "qwen.attn.value")
            attended = try make(config.queryDim, "qwen.attn.attended")
            projected = try make(config.hiddenSize, "qwen.attn.projected")
        }
    }

    public struct Weights {
        public let inputNorm: (buffer: MTLBuffer, offset: Int)
        public let qProj: ForwardEncoder.ProjectionSource
        public let kProj: ForwardEncoder.ProjectionSource
        public let vProj: ForwardEncoder.ProjectionSource
        public let oProj: ForwardEncoder.ProjectionSource
        public let qNorm: (buffer: MTLBuffer, offset: Int)
        public let kNorm: (buffer: MTLBuffer, offset: Int)

        public init(
            inputNorm: (buffer: MTLBuffer, offset: Int),
            qProj: ForwardEncoder.ProjectionSource, kProj: ForwardEncoder.ProjectionSource,
            vProj: ForwardEncoder.ProjectionSource, oProj: ForwardEncoder.ProjectionSource,
            qNorm: (buffer: MTLBuffer, offset: Int), kNorm: (buffer: MTLBuffer, offset: Int)
        ) {
            self.inputNorm = inputNorm
            self.qProj = qProj
            self.kProj = kProj
            self.vProj = vProj
            self.oProj = oProj
            self.qNorm = qNorm
            self.kNorm = kNorm
        }
    }

    private let config: Qwen35MoeConfig
    private let encoder: ForwardEncoder

    public init(config: Qwen35MoeConfig, encoder: ForwardEncoder) {
        self.config = config
        self.encoder = encoder
    }

    /// Encodes one token. `hidden` is read and written; the residual add happens in place.
    ///
    /// - Parameters:
    ///   - cos, sin: the rotary tables for this position, `[headDim / 2]` each, with zero
    ///     frequencies past the rotating quarter so the rest of the head does not turn.
    ///   - sinks: an unreachable sink a head, which is how this project's attention kernel
    ///     expresses "no learned sink" (Gemma does the same).
    public func encode(
        hidden: MTLBuffer, weights: Weights, scratch: Scratch,
        keyCache: MTLBuffer, valueCache: MTLBuffer, sinks: MTLBuffer,
        position: Int, visibleStart: Int, visibleCount: Int, ringSize: Int,
        cos: MTLBuffer, sin: MTLBuffer,
        in commandBuffer: MTLCommandBuffer
    ) throws {
        let hiddenSize = config.hiddenSize
        let heads = config.attentionHeadCount
        let kvHeads = config.keyValueHeadCount
        let headDim = config.headDim

        try encoder.rmsNorm(
            input: hidden, inputOffset: 0,
            scale: weights.inputNorm.buffer, scaleOffset: weights.inputNorm.offset,
            output: scratch.normed, outputOffset: 0,
            size: hiddenSize, eps: config.rmsNormEps, in: commandBuffer)

        // The query projection carries the gate with it, interleaved per head.
        try encoder.encodeProjection(
            weights.qProj, input: scratch.normed, inputOffset: 0,
            output: scratch.combined, outputOffset: 0,
            rows: config.queryProjectionRows, cols: hiddenSize, in: commandBuffer)
        try encoder.qwenSplitQueryGate(
            combined: scratch.combined, query: scratch.query, gate: scratch.gate,
            heads: heads, headDim: headDim, in: commandBuffer)

        try encoder.encodeProjection(
            weights.kProj, input: scratch.normed, inputOffset: 0,
            output: scratch.key, outputOffset: 0,
            rows: config.keyValueDim, cols: hiddenSize, in: commandBuffer)
        try encoder.encodeProjection(
            weights.vProj, input: scratch.normed, inputOffset: 0,
            output: scratch.value, outputOffset: 0,
            rows: config.keyValueDim, cols: hiddenSize, in: commandBuffer)

        // The norms are per head and come **before** the rotary.
        try encoder.rmsNormHeads(
            vector: scratch.query, scale: weights.qNorm.buffer,
            scaleOffset: weights.qNorm.offset, heads: heads, headDim: headDim,
            eps: config.rmsNormEps, in: commandBuffer)
        try encoder.rmsNormHeads(
            vector: scratch.key, scale: weights.kNorm.buffer,
            scaleOffset: weights.kNorm.offset, heads: kvHeads, headDim: headDim,
            eps: config.rmsNormEps, in: commandBuffer)

        try encoder.applyRoPE(
            vector: scratch.query, vectorOffset: 0, cos: cos, sin: sin, tableOffset: 0,
            heads: heads, headDim: headDim, in: commandBuffer)
        try encoder.applyRoPE(
            vector: scratch.key, vectorOffset: 0, cos: cos, sin: sin, tableOffset: 0,
            heads: kvHeads, headDim: headDim, in: commandBuffer)

        try encoder.writeKeyValue(
            key: scratch.key, keyOffset: 0, value: scratch.value, valueOffset: 0,
            keyCache: keyCache, valueCache: valueCache,
            kvHeads: kvHeads, headDim: headDim, position: position, ringSize: ringSize,
            in: commandBuffer)

        try encoder.attention(
            query: scratch.query, queryOffset: 0,
            keyCache: keyCache, valueCache: valueCache, sinks: sinks, sinksOffset: 0,
            output: scratch.attended, outputOffset: 0,
            qHeads: heads, kvHeads: kvHeads, headDim: headDim, keyCount: visibleCount,
            ringSize: ringSize, startPosition: visibleStart,
            // Not Gemma's 1.0: see the note on this type.
            smScale: 1.0 / Float(headDim).squareRoot(), in: commandBuffer)

        // The gate multiplies what attention returned, not what it attended with.
        try encoder.qwenApplyOutputGate(
            output: scratch.attended, gate: scratch.gate, count: config.queryDim,
            in: commandBuffer)

        try encoder.encodeProjection(
            weights.oProj, input: scratch.attended, inputOffset: 0,
            output: scratch.projected, outputOffset: 0,
            rows: hiddenSize, cols: config.queryDim, in: commandBuffer)

        try encoder.addInPlace(
            target: hidden, targetOffset: 0, addend: scratch.projected, addendOffset: 0,
            size: hiddenSize, in: commandBuffer)
    }
}
