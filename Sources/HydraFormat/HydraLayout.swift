import Foundation
import HydraCore

/// Disposition physique d'une installation `.hydra`.
///
/// ```
/// gpt-oss-20b.hydra/
///   manifest.json
///   verified-install.json
///   resident.bin            attention, routeurs, normes, sinks, lm_head
///   embed.bin               embed_tokens, mappé mais hors working set Metal
///   experts/
///     layout.json
///     layer_00.bin ... layer_NN.bin
///   tokenizer/
/// ```
///
/// La mise en page d'un blob d'expert n'est **pas** recalculée ici : elle vient de
/// `config.expertBlobLayout`, dans `HydraCore`, pour que le format sur disque et le
/// dimensionnement des slots mémoire ne puissent pas diverger.
public struct HydraLayout: Sendable {

    public static let pageAlignment = ExpertBlobLayout.pageAlignment
    public static let tensorAlignment = ExpertBlobLayout.tensorAlignment

    public let config: GptOssConfig
    public let expertBlob: ExpertBlobLayout
    public let resident: [TensorPlacement]
    public let residentBytes: Int

    /// Emplacement d'un tenseur source dans un fichier destination.
    public struct TensorPlacement: Sendable, Equatable {
        /// Clé safetensors d'origine, conservée pour la traçabilité et la vérification.
        public let sourceName: String
        public let offset: Int
        public let byteCount: Int
        public var end: Int { offset + byteCount }
    }

    public init(config: GptOssConfig) {
        self.config = config
        self.expertBlob = config.expertBlobLayout
        let (placements, total) = Self.makeResidentPlacements(config: config)
        self.resident = placements
        self.residentBytes = total
    }

    // MARK: - Experts

    /// Décalage du blob de l'expert `index` dans son fichier de couche.
    public func expertOffset(_ index: Int) -> Int {
        index * expertBlob.strideBytes
    }

    /// Taille d'un fichier `experts/layer_XX.bin`.
    public var expertLayerFileBytes: Int {
        config.expertCount * expertBlob.strideBytes
    }

    // MARK: - Résidents

    /// Nom safetensors des tenseurs résidents d'une couche, dans l'ordre de placement.
    public static func residentTensorNames(
        layer: Int
    ) -> [(name: String, bytes: (GptOssConfig) -> Int)] {
        let l = layer
        return [
            ("model.layers.\(l).input_layernorm.weight", { 2 * $0.hiddenSize }),
            ("model.layers.\(l).self_attn.q_proj.weight", { 2 * $0.attentionHeadCount * $0.headDim * $0.hiddenSize }),
            ("model.layers.\(l).self_attn.q_proj.bias", { 2 * $0.attentionHeadCount * $0.headDim }),
            ("model.layers.\(l).self_attn.k_proj.weight", { 2 * $0.keyValueHeadCount * $0.headDim * $0.hiddenSize }),
            ("model.layers.\(l).self_attn.k_proj.bias", { 2 * $0.keyValueHeadCount * $0.headDim }),
            ("model.layers.\(l).self_attn.v_proj.weight", { 2 * $0.keyValueHeadCount * $0.headDim * $0.hiddenSize }),
            ("model.layers.\(l).self_attn.v_proj.bias", { 2 * $0.keyValueHeadCount * $0.headDim }),
            ("model.layers.\(l).self_attn.o_proj.weight", { 2 * $0.hiddenSize * $0.attentionHeadCount * $0.headDim }),
            ("model.layers.\(l).self_attn.o_proj.bias", { 2 * $0.hiddenSize }),
            ("model.layers.\(l).self_attn.sinks", { 2 * $0.attentionHeadCount }),
            ("model.layers.\(l).post_attention_layernorm.weight", { 2 * $0.hiddenSize }),
            ("model.layers.\(l).mlp.router.weight", { 2 * $0.expertCount * $0.hiddenSize }),
            ("model.layers.\(l).mlp.router.bias", { 2 * $0.expertCount }),
        ]
    }

    private static func makeResidentPlacements(
        config: GptOssConfig
    ) -> ([TensorPlacement], Int) {
        var out: [TensorPlacement] = []
        var cursor = 0
        func place(_ name: String, _ size: Int) {
            cursor = alignUp(cursor, to: tensorAlignment)
            out.append(TensorPlacement(sourceName: name, offset: cursor, byteCount: size))
            cursor += size
        }

        // Les couches d'abord, dans l'ordre d'exécution.
        for layer in 0..<config.layerCount {
            for entry in residentTensorNames(layer: layer) {
                place(entry.name, entry.bytes(config))
            }
        }
        place("model.norm.weight", 2 * config.hiddenSize)
        // La tête LM en dernier : c'est le plus gros bloc et il n'est lu qu'une fois par token.
        place("lm_head.weight", config.lmHeadBytes)

        return (out, alignUp(cursor, to: pageAlignment))
    }

    /// Le seul tenseur de `embed.bin`.
    public var embeddingByteCount: Int { config.embeddingBytes }

    public func placement(of name: String) -> TensorPlacement? {
        resident.first { $0.sourceName == name }
    }
}
