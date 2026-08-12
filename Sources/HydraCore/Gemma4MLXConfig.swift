import Foundation

/// Gemma 4 as the MLX 4-bit QAT conversion stores it.
///
/// **A wrapper around `Gemma4Config`, not a copy of it.** The architecture is identical, same
/// layers, same two attention geometries, same 5:1 pattern, same operator semantics from D-022.
/// What differs is how the weights are written down. Restating the geometry here would give two
/// sets of constants that must agree forever, which is the drift D-023 exists to prevent; the
/// only things overridden are the ones that genuinely changed.
///
/// Three of them, and each is a trap recorded in D-024:
///
/// - **Names are inverted.** `language_model.model.layers.N…` where the BF16 build writes
///   `model.language_model.layers.N…`.
/// - **Mixed precision.** 4 bits for attention, the embedding and the experts; **8 bits** for
///   every layer's dense MLP and router, 120 tensors, not a special case for layer 0.
/// - **Experts are unfused.** `experts.switch_glu.{gate,up,down}_proj`, three matrices each
///   carrying its own scales and biases, where the BF16 build fuses gate and up into one.
public struct Gemma4MLXConfig: Sendable, Equatable {

    /// The architecture, unchanged. Everything geometric is read through here.
    public let base: Gemma4Config

    /// Bits for attention, the embedding and the experts.
    public let quantBits: Int
    /// Bits for the dense MLP and the router, the 120 tensors the config overrides.
    public let denseBits: Int
    public let groupSize: Int

    public init(
        base: Gemma4Config = .a4b,
        quantBits: Int = 4, denseBits: Int = 8, groupSize: Int = 64
    ) {
        self.base = base
        self.quantBits = quantBits
        self.denseBits = denseBits
        self.groupSize = groupSize
    }

    public static let a4b = Gemma4MLXConfig()

    /// A tiny configuration for tests, mirroring `Gemma4Config.tiny`.
    ///
    /// The group size is 16 rather than the published 64 because the tiny model is 64 wide with
    /// a 16-wide expert intermediate, and a group has to divide a row. The published value is
    /// asserted separately on `a4b`; what this exercises is the wiring, which does not depend
    /// on the number.
    public static let tiny = Gemma4MLXConfig(base: .tiny, groupSize: 16)

    // MARK: - Names

    /// The prefix, which is the inverted one.
    public static let prefix = "language_model.model"

    public func layerTensor(_ suffix: String, layer: Int) -> String {
        "\(Self.prefix).layers.\(layer).\(suffix)"
    }

    public var finalNormTensor: String { "\(Self.prefix).norm.weight" }
    public var embeddingTensor: String { "\(Self.prefix).embed_tokens.weight" }

    // MARK: - Quantized tensor sizes

    /// The three parts of one affine-quantized matrix, named and sized.
    public struct QuantizedTensor: Sendable, Equatable {
        public let weight: String
        public let scales: String
        public let biases: String
        public let layout: MLXAffineLayout

        public var parts: [(name: String, byteCount: Int)] {
            [
                (weight, layout.weightBytes),
                (scales, layout.scaleBytes),
                (biases, layout.biasBytes),
            ]
        }
        public var totalBytes: Int { layout.totalBytes }
    }

    public func quantized(
        _ stem: String, rows: Int, cols: Int, bits: Int
    ) -> QuantizedTensor {
        QuantizedTensor(
            weight: "\(stem).weight", scales: "\(stem).scales", biases: "\(stem).biases",
            layout: MLXAffineLayout(
                bits: bits, groupSize: groupSize, rows: rows, cols: cols))
    }

    // MARK: - The resident tensors

    /// Everything that must stay in memory, in placement order.
    ///
    /// The per-layer list follows the checkpoint's own inventory: 45 tensors on a sliding
    /// layer, 42 on a full one, the difference being `v_proj`, which full layers do not have
    /// because `attention_k_eq_v` makes V reuse the key projection (D-022). Nine of the 45 are
    /// the experts, which live in their own files rather than here.
    public var residentTensors: [(name: String, byteCount: Int)] {
        var out: [(name: String, byteCount: Int)] = []
        let hidden = base.hiddenSize

        func plain(_ name: String, _ elements: Int) {
            out.append((name, elements * 2))
        }
        func quant(_ stem: String, rows: Int, cols: Int, bits: Int) {
            for part in quantized(stem, rows: rows, cols: cols, bits: bits).parts {
                out.append(part)
            }
        }

        for layer in 0..<base.layerCount {
            let geometry = base.attentionGeometry(atLayer: layer)
            func name(_ suffix: String) -> String { layerTensor(suffix, layer: layer) }

            plain(name("input_layernorm.weight"), hidden)
            quant(name("self_attn.q_proj"), rows: geometry.queryDim, cols: hidden, bits: quantBits)
            plain(name("self_attn.q_norm.weight"), geometry.headDim)
            quant(
                name("self_attn.k_proj"), rows: geometry.keyValueDim, cols: hidden,
                bits: quantBits)
            plain(name("self_attn.k_norm.weight"), geometry.headDim)
            // Only sliding layers project their own values.
            if base.attentionPattern(atLayer: layer) == .sliding {
                quant(
                    name("self_attn.v_proj"), rows: geometry.keyValueDim, cols: hidden,
                    bits: quantBits)
            }
            quant(
                name("self_attn.o_proj"), rows: hidden, cols: geometry.queryDim,
                bits: quantBits)
            plain(name("post_attention_layernorm.weight"), hidden)

            plain(name("pre_feedforward_layernorm.weight"), hidden)
            quant(
                name("mlp.gate_proj"), rows: base.intermediateSize, cols: hidden,
                bits: denseBits)
            quant(
                name("mlp.up_proj"), rows: base.intermediateSize, cols: hidden, bits: denseBits)
            quant(
                name("mlp.down_proj"), rows: hidden, cols: base.intermediateSize,
                bits: denseBits)
            plain(name("post_feedforward_layernorm_1.weight"), hidden)

            plain(name("pre_feedforward_layernorm_2.weight"), hidden)
            quant(
                name("router.proj"), rows: base.expertCount, cols: hidden, bits: denseBits)
            plain(name("router.scale"), hidden)
            plain(name("router.per_expert_scale"), base.expertCount)
            plain(name("post_feedforward_layernorm_2.weight"), hidden)

            plain(name("post_feedforward_layernorm.weight"), hidden)
            plain(name("layer_scalar"), 1)
        }

        plain(finalNormTensor, hidden)
        // Tied to the output head and read in full every token, so it is resident, and it is
        // quantized here, where the BF16 build stores it plain.
        quant("\(Self.prefix).embed_tokens", rows: base.vocabSize, cols: hidden, bits: quantBits)
        return out
    }

    /// The nine sub-tensors of one expert.
    public var expertBlobLayout: MLXExpertBlobLayout {
        MLXExpertBlobLayout(
            hiddenSize: base.hiddenSize, moeIntermediateSize: base.moeIntermediateSize,
            bits: quantBits, groupSize: groupSize)
    }

    /// The checkpoint's names for one layer's experts, in blob order.
    public func expertTensors(layer: Int) -> [(stem: String, rows: Int, cols: Int)] {
        let stem = layerTensor("experts.switch_glu", layer: layer)
        return [
            ("\(stem).gate_proj", base.moeIntermediateSize, base.hiddenSize),
            ("\(stem).up_proj", base.moeIntermediateSize, base.hiddenSize),
            ("\(stem).down_proj", base.hiddenSize, base.moeIntermediateSize),
        ]
    }
}
