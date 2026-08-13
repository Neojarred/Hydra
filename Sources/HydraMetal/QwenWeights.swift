import Foundation
import HydraCore
import HydraFormat
import Metal

/// Where a Qwen layer's weights live, and how they decode.
///
/// The same arrangement as `Gemma4Weights` and for the same reason (D-023): the dispatch on
/// which checkpoint is being run happens once, when the runner is built, never inside a
/// decoding step.
///
/// Only one encoding exists so far. The published MLX builds are affine group 64 with
/// per-tensor overrides, which `MLXAffineLayout` already decodes, so unlike Gemma there is no
/// second conformer for a BF16 release. There is no BF16 runner for this architecture and this
/// type does not pretend otherwise.
public struct QwenMLXWeights: Sendable {

    private let config: Qwen35MoeConfig
    private let mapping: ModelMapping

    public init(config: Qwen35MoeConfig, mapping: ModelMapping) {
        self.config = config
        self.mapping = mapping
    }

    /// The bit width a tensor decodes at, read from the checkpoint's own per-tensor map.
    ///
    /// Not a constant: the router and the shared expert's gate are 8-bit in both published
    /// builds while everything else follows the base width, so a single width would decode
    /// eighty tensors at half their intended shape from bytes that are all present (D-024).
    public func bits(for suffix: String) -> Int {
        suffix == "mlp.gate" || suffix == "mlp.shared_expert_gate"
            ? config.gateBits : config.quantBits
    }

    private func triple(_ stem: String, bits: Int) throws -> ForwardEncoder.ProjectionSource {
        let words = try mapping.residentTensor("\(stem).weight")
        let scales = try mapping.residentTensor("\(stem).scales")
        let biases = try mapping.residentTensor("\(stem).biases")
        return .mlxAffine(
            words: words.buffer, wordsOffset: words.offset,
            scales: scales.buffer, scalesOffset: scales.offset,
            biases: biases.buffer, biasesOffset: biases.offset,
            bits: bits, groupSize: config.groupSize)
    }

    /// A quantized projection, by the suffix the checkpoint spells it with.
    public func projection(_ suffix: String, layer: Int) throws -> ForwardEncoder.ProjectionSource {
        try triple(config.layerTensor(suffix, layer: layer), bits: bits(for: suffix))
    }

    /// An unquantized tensor: the norms, `A_log`, `dt_bias`, the convolution's kernel.
    ///
    /// BF16 in the checkpoint. `A_log` and `dt_bias` are one value a value head and the
    /// convolution's weight is `[convDim][kernel]`, none of which is worth quantizing and none
    /// of which the published builds do.
    public func plain(_ suffix: String, layer: Int) throws -> (buffer: MTLBuffer, offset: Int) {
        let tensor = try mapping.residentTensor(config.layerTensor(suffix, layer: layer))
        return (tensor.buffer, tensor.offset)
    }

    public func finalNorm() throws -> (buffer: MTLBuffer, offset: Int) {
        let tensor = try mapping.residentTensor(config.finalNormTensor)
        return (tensor.buffer, tensor.offset)
    }

    /// The output head, **its own tensor**.
    ///
    /// `tie_word_embeddings` is false here where Gemma ties them, so this is not the embedding
    /// table read a second time: both are resident and they are different weights.
    public func head() throws -> ForwardEncoder.ProjectionSource {
        try triple("language_model.lm_head", bits: config.quantBits)
    }

    /// One of an expert's three matrices, inside a blob the slot cache has made resident.
    public func expert(
        _ part: Gemma4ExpertPart, blob: MTLBuffer
    ) -> ForwardEncoder.ProjectionSource {
        let layout = MLXExpertBlobLayout(
            hiddenSize: config.hiddenSize, moeIntermediateSize: config.moeIntermediateSize,
            bits: config.quantBits, groupSize: config.groupSize)
        let slots: (ExpertBlobLayout.Slot, ExpertBlobLayout.Slot, ExpertBlobLayout.Slot)
        switch part {
        case .gate: slots = (layout.gateWeights, layout.gateScales, layout.gateBiases)
        case .up: slots = (layout.upWeights, layout.upScales, layout.upBiases)
        case .down: slots = (layout.downWeights, layout.downScales, layout.downBiases)
        }
        return .mlxAffine(
            words: blob, wordsOffset: slots.0.offset,
            scales: blob, scalesOffset: slots.1.offset,
            biases: blob, biasesOffset: slots.2.offset,
            bits: config.quantBits, groupSize: config.groupSize)
    }

    // MARK: - The names, in one place

    /// Every tensor suffix a linear layer reads, quantized ones first.
    ///
    /// Listed rather than spelled at each use so the runner and the descriptor can be checked
    /// against each other: a name this asks for that the descriptor never makes resident fails
    /// at load, and a name the descriptor carries that nothing asks for is dead weight in a
    /// 20 GB download.
    public static let linearProjections = [
        "linear_attn.in_proj_qkv", "linear_attn.in_proj_z",
        "linear_attn.in_proj_a", "linear_attn.in_proj_b", "linear_attn.out_proj",
    ]
    public static let linearPlain = [
        "linear_attn.conv1d.weight", "linear_attn.A_log", "linear_attn.dt_bias",
        "linear_attn.norm.weight",
    ]
    public static let attentionProjections = [
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    ]
    public static let attentionPlain = ["self_attn.q_norm.weight", "self_attn.k_norm.weight"]
    public static let sharedProjections = [
        "mlp.gate", "mlp.shared_expert_gate",
        "mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj",
        "mlp.shared_expert.down_proj",
    ]
    public static let sharedPlain = ["input_layernorm.weight", "post_attention_layernorm.weight"]
}
