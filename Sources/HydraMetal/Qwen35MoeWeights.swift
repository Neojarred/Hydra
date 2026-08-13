import Foundation
import HydraCore
import HydraFormat
import Metal

/// Where a Qwen layer's weights live, and how they decode.
///
/// One conformance rather than Gemma's two, because only quantized builds of this model exist
/// and the BF16 original is 70 GB. So this is a struct and not a protocol: a second checkpoint
/// encoding would be the reason to introduce one, and there is no second encoding.
///
/// Resolution happens **when the runner is built**, not inside a decoding step (D-023). The
/// layer runner is handed finished `Weights` values and never learns a tensor name.
///
/// The two layer kinds ask for entirely different inventories, which is the clearest statement
/// of what this architecture is: a recurrent layer has no `q_proj`, no `k_proj` and no cache,
/// and carries a convolution, two per-head parameter vectors and a gated norm that a full
/// attention layer has never heard of.
public struct Qwen35MoeWeights: Sendable {

    private let config: Qwen35MoeConfig
    private let mapping: ModelMapping

    public init(config: Qwen35MoeConfig, mapping: ModelMapping) {
        self.config = config
        self.mapping = mapping
    }

    /// The bit width of a projection, from the checkpoint's own per-tensor overrides.
    ///
    /// The router and the shared expert's gate are 8-bit in both published builds and
    /// everything else follows the base width (D-026). A single width would decode those two
    /// tensors a layer at half their intended shape, from bytes that are all present.
    private func bits(for suffix: String) -> Int {
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

    private func projection(
        _ suffix: String, layer: Int
    ) throws -> ForwardEncoder.ProjectionSource {
        try triple(config.layerTensor(suffix, layer: layer), bits: bits(for: suffix))
    }

    /// An unquantized tensor: the norms, `A_log`, `dt_bias`, the convolution's kernel. BF16,
    /// because quantizing a per-channel scale saves nothing and costs accuracy (D-015).
    private func plain(_ suffix: String, layer: Int) throws -> (buffer: MTLBuffer, offset: Int) {
        let tensor = try mapping.residentTensor(config.layerTensor(suffix, layer: layer))
        return (tensor.buffer, tensor.offset)
    }

    // MARK: - Per-layer inventories

    public func linearWeights(layer: Int) throws -> QwenLinearBlock.Weights {
        QwenLinearBlock.Weights(
            inputNorm: try plain("input_layernorm.weight", layer: layer),
            qkv: try projection("linear_attn.in_proj_qkv", layer: layer),
            z: try projection("linear_attn.in_proj_z", layer: layer),
            a: try projection("linear_attn.in_proj_a", layer: layer),
            b: try projection("linear_attn.in_proj_b", layer: layer),
            outProj: try projection("linear_attn.out_proj", layer: layer),
            convWeight: try plain("linear_attn.conv1d.weight", layer: layer),
            // The published checkpoint has no `conv1d.bias`. Absent rather than zero, so a
            // build that grows one is a missing tensor and not a silently ignored one.
            convBias: nil,
            logA: try plain("linear_attn.A_log", layer: layer),
            dtBias: try plain("linear_attn.dt_bias", layer: layer),
            normWeight: try plain("linear_attn.norm.weight", layer: layer))
    }

    public func attentionWeights(layer: Int) throws -> QwenAttentionBlock.Weights {
        QwenAttentionBlock.Weights(
            inputNorm: try plain("input_layernorm.weight", layer: layer),
            qProj: try projection("self_attn.q_proj", layer: layer),
            kProj: try projection("self_attn.k_proj", layer: layer),
            vProj: try projection("self_attn.v_proj", layer: layer),
            oProj: try projection("self_attn.o_proj", layer: layer),
            qNorm: try plain("self_attn.q_norm.weight", layer: layer),
            kNorm: try plain("self_attn.k_norm.weight", layer: layer))
    }

    public func mixtureWeights(layer: Int) throws -> QwenMixtureBlock.Weights {
        QwenMixtureBlock.Weights(
            postAttentionNorm: try plain("post_attention_layernorm.weight", layer: layer),
            router: try projection("mlp.gate", layer: layer),
            sharedGate: try projection("mlp.shared_expert_gate", layer: layer),
            shared: QwenMixtureBlock.Expert(
                gate: try projection("mlp.shared_expert.gate_proj", layer: layer),
                up: try projection("mlp.shared_expert.up_proj", layer: layer),
                down: try projection("mlp.shared_expert.down_proj", layer: layer)))
    }

    public func layerWeights(layer: Int) throws -> QwenLayerRunner.LayerWeights {
        let mixer: QwenLayerRunner.TokenMixer =
            config.attentionPattern(atLayer: layer) == .full
            ? .attention(try attentionWeights(layer: layer))
            : .linear(try linearWeights(layer: layer))
        return QwenLayerRunner.LayerWeights(
            mixer: mixer, mixture: try mixtureWeights(layer: layer))
    }

    // MARK: - One routed expert, inside a slot the cache has made resident

    public func expert(blob: MTLBuffer, offset: Int = 0) -> QwenMixtureBlock.Expert {
        let layout = config.expertBlobLayout
        func part(
            _ words: ExpertBlobLayout.Slot, _ scales: ExpertBlobLayout.Slot,
            _ biases: ExpertBlobLayout.Slot
        ) -> ForwardEncoder.ProjectionSource {
            .mlxAffine(
                words: blob, wordsOffset: offset + words.offset,
                scales: blob, scalesOffset: offset + scales.offset,
                biases: blob, biasesOffset: offset + biases.offset,
                bits: config.quantBits, groupSize: config.groupSize)
        }
        return QwenMixtureBlock.Expert(
            gate: part(layout.gateWeights, layout.gateScales, layout.gateBiases),
            up: part(layout.upWeights, layout.upScales, layout.upBiases),
            down: part(layout.downWeights, layout.downScales, layout.downBiases))
    }

    // MARK: - The ends of the model

    public func finalNorm() throws -> (buffer: MTLBuffer, offset: Int) {
        let tensor = try mapping.residentTensor(config.finalNormTensor)
        return (tensor.buffer, tensor.offset)
    }

    /// The output projection, which is **not** the embedding table.
    ///
    /// `tie_word_embeddings` is false here where it is true for Gemma, so this reads `lm_head`
    /// and the two tables are genuinely different weights. Reading the embedding here instead
    /// would produce a model that runs and answers, from a head it was never trained with.
    public func head() throws -> ForwardEncoder.ProjectionSource {
        try triple(config.headStem, bits: config.quantBits)
    }

    /// One embedding row, unpacked and dequantized on the CPU.
    public func readEmbedding(
        token: Int, into destination: UnsafeMutableBufferPointer<Float>
    ) {
        let layout = MLXAffineLayout(
            bits: config.quantBits, groupSize: config.groupSize,
            rows: config.vocabSize, cols: config.hiddenSize)
        let stem = config.embeddingStem

        // Not a silent return. Leaving `destination` holding the previous token's embedding
        // lets the forward pass run to completion and produce a finite, plausible, wrong
        // answer; a missing embedding table means the installation is broken and that is worth
        // saying where it is discovered.
        guard let words = try? mapping.residentTensor("\(stem).weight"),
            let scales = try? mapping.residentTensor("\(stem).scales"),
            let biases = try? mapping.residentTensor("\(stem).biases")
        else {
            preconditionFailure(
                "the embedding table is missing from the installation: "
                    + "\(stem).weight/.scales/.biases")
        }

        mapping.resident.withBytes { raw in
            MLXAffineRow.read(
                row: token, into: destination, bytes: raw, layout: layout,
                words: words.offset, scales: scales.offset, biases: biases.offset,
                bits: config.quantBits, groupSize: config.groupSize)
        }
    }
}
