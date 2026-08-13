import Foundation
import HydraCore
import Metal

/// One Qwen linear attention block, encoded onto the GPU.
///
/// The kernels underneath are each checked against `QwenReferenceOps`. This is the composition,
/// and it takes its projections as resolved `ProjectionSource` values rather than resolving them
/// itself, for two reasons: the dispatch on how a checkpoint is encoded belongs at the seam
/// (D-023), and a test can then drive the whole block with BF16 weights it builds by hand
/// instead of needing a quantized checkpoint to exist first.
///
/// The order is `QwenReferenceLayer`'s, which is `Qwen3NextDecoderLayer`'s (D-027):
///
/// ```
/// normed = input_layernorm(x)
/// qkvRaw = in_proj_qkv(normed)   z = in_proj_z(normed)   a, b = in_proj_a/b(normed)
/// qkv    = silu(causal_conv(qkvRaw))       window advances here
/// out    = delta_rule(split(qkv), g(a), beta(b))    state advances here
/// out    = gated_rms_norm(out, norm.weight, gate: z)
/// x      = x + out_proj(out)
/// ```
public struct QwenLinearBlock {

    /// Working buffers for one token, allocated once.
    public final class Scratch {
        public let normed: MTLBuffer
        public let qkvRaw: MTLBuffer
        public let qkv: MTLBuffer
        public let z: MTLBuffer
        public let a: MTLBuffer
        public let b: MTLBuffer
        public let mixed: MTLBuffer
        public let gated: MTLBuffer
        public let projected: MTLBuffer

        public init(config: Qwen35MoeConfig, device: MTLDevice) throws {
            func make(_ floats: Int, _ name: String) throws -> MTLBuffer {
                guard let buffer = device.makeBuffer(
                    length: max(floats, 1) * 4, options: .storageModeShared)
                else { throw ModelRunner.RunnerError.allocationFailed(name) }
                return buffer
            }
            let zDim = config.linearValueHeads * config.linearValueHeadDim
            normed = try make(config.hiddenSize, "qwen.normed")
            qkvRaw = try make(config.linearConvDim, "qwen.qkvRaw")
            qkv = try make(config.linearConvDim, "qwen.qkv")
            z = try make(zDim, "qwen.z")
            a = try make(config.linearValueHeads, "qwen.a")
            b = try make(config.linearValueHeads, "qwen.b")
            mixed = try make(zDim, "qwen.mixed")
            gated = try make(zDim, "qwen.gated")
            projected = try make(config.hiddenSize, "qwen.projected")
        }
    }

    /// The weights one linear block reads, already resolved.
    public struct Weights {
        public let inputNorm: (buffer: MTLBuffer, offset: Int)
        public let qkv: ForwardEncoder.ProjectionSource
        public let z: ForwardEncoder.ProjectionSource
        public let a: ForwardEncoder.ProjectionSource
        public let b: ForwardEncoder.ProjectionSource
        public let outProj: ForwardEncoder.ProjectionSource
        public let convWeight: MTLBuffer
        public let convBias: MTLBuffer?
        public let logA: MTLBuffer
        public let dtBias: MTLBuffer
        public let normWeight: (buffer: MTLBuffer, offset: Int)

        public init(
            inputNorm: (buffer: MTLBuffer, offset: Int),
            qkv: ForwardEncoder.ProjectionSource, z: ForwardEncoder.ProjectionSource,
            a: ForwardEncoder.ProjectionSource, b: ForwardEncoder.ProjectionSource,
            outProj: ForwardEncoder.ProjectionSource,
            convWeight: MTLBuffer, convBias: MTLBuffer?,
            logA: MTLBuffer, dtBias: MTLBuffer,
            normWeight: (buffer: MTLBuffer, offset: Int)
        ) {
            self.inputNorm = inputNorm
            self.qkv = qkv
            self.z = z
            self.a = a
            self.b = b
            self.outProj = outProj
            self.convWeight = convWeight
            self.convBias = convBias
            self.logA = logA
            self.dtBias = dtBias
            self.normWeight = normWeight
        }
    }

    private let config: Qwen35MoeConfig
    private let encoder: ForwardEncoder

    public init(config: Qwen35MoeConfig, encoder: ForwardEncoder) {
        self.config = config
        self.encoder = encoder
    }

    /// Encodes one token. `hidden` is read and written: the residual add happens in place.
    ///
    /// `state` and `window` are advanced by the kernels that read them, so the caller does not
    /// advance anything and cannot forget to.
    public func encode(
        hidden: MTLBuffer, weights: Weights, scratch: Scratch,
        state: MTLBuffer, stateOffset: Int, window: MTLBuffer, windowOffset: Int,
        in commandBuffer: MTLCommandBuffer
    ) throws {
        let hiddenSize = config.hiddenSize
        let zDim = config.linearValueHeads * config.linearValueHeadDim
        let keySpan = config.linearKeyHeads * config.linearKeyHeadDim
        let float = MemoryLayout<Float>.size

        try encoder.rmsNorm(
            input: hidden, inputOffset: 0,
            scale: weights.inputNorm.buffer, scaleOffset: weights.inputNorm.offset,
            output: scratch.normed, outputOffset: 0,
            size: hiddenSize, eps: config.rmsNormEps, in: commandBuffer)

        // The four projections all read the normalized input.
        try encoder.encodeProjection(
            weights.qkv, input: scratch.normed, inputOffset: 0,
            output: scratch.qkvRaw, outputOffset: 0,
            rows: config.linearConvDim, cols: hiddenSize, in: commandBuffer)
        try encoder.encodeProjection(
            weights.z, input: scratch.normed, inputOffset: 0,
            output: scratch.z, outputOffset: 0,
            rows: zDim, cols: hiddenSize, in: commandBuffer)
        try encoder.encodeProjection(
            weights.a, input: scratch.normed, inputOffset: 0,
            output: scratch.a, outputOffset: 0,
            rows: config.linearValueHeads, cols: hiddenSize, in: commandBuffer)
        try encoder.encodeProjection(
            weights.b, input: scratch.normed, inputOffset: 0,
            output: scratch.b, outputOffset: 0,
            rows: config.linearValueHeads, cols: hiddenSize, in: commandBuffer)

        // The convolution sees q, k and v together, before the split, and advances its window.
        try encoder.qwenCausalConvStep(
            window: window, windowOffset: windowOffset,
            input: scratch.qkvRaw, weight: weights.convWeight, bias: weights.convBias,
            output: scratch.qkv, convDim: config.linearConvDim,
            kernel: config.linearConvKernel, in: commandBuffer)

        // q, k and v are contiguous spans of the convolved vector, so the split is an offset
        // and not a copy.
        try encoder.qwenDeltaRuleStep(
            state: state, stateOffset: stateOffset,
            query: scratch.qkv, key: scratch.qkv, value: scratch.qkv,
            queryOffset: 0, keyOffset: keySpan * float, valueOffset: 2 * keySpan * float,
            a: scratch.a, b: scratch.b, logA: weights.logA, dtBias: weights.dtBias,
            output: scratch.mixed,
            valueHeads: config.linearValueHeads, keyHeads: config.linearKeyHeads,
            keyDim: config.linearKeyHeadDim, valueDim: config.linearValueHeadDim,
            eps: config.rmsNormEps, in: commandBuffer)

        try encoder.qwenGatedRMSNormHeads(
            input: scratch.mixed,
            weight: weights.normWeight.buffer, weightOffset: weights.normWeight.offset,
            gate: scratch.z, output: scratch.gated,
            heads: config.linearValueHeads, dim: config.linearValueHeadDim,
            eps: config.rmsNormEps, in: commandBuffer)

        try encoder.encodeProjection(
            weights.outProj, input: scratch.gated, inputOffset: 0,
            output: scratch.projected, outputOffset: 0,
            rows: hiddenSize, cols: zDim, in: commandBuffer)

        // The residual is the block's input, not its normalized form.
        try encoder.addInPlace(
            target: hidden, targetOffset: 0, addend: scratch.projected, addendOffset: 0,
            size: hiddenSize, in: commandBuffer)
    }
}
