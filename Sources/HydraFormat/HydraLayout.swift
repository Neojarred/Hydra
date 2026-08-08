import Foundation
import HydraCore

/// The physical layout of a `.hydra` installation.
///
/// ```
/// gpt-oss-20b.hydra/
///   manifest.json
///   verified-install.json
///   resident.bin            attention, routers, norms, sinks, lm_head
///   embed.bin               embed_tokens, mapped but outside the Metal working set
///   experts/
///     layout.json
///     layer_00.bin ... layer_NN.bin
///   tokenizer/
/// ```
///
/// An expert blob's layout is **not** recomputed here: it comes from
/// `config.expertBlobLayout`, in `HydraCore`, so that the on-disk format and the sizing of
/// memory slots cannot diverge.
public struct HydraLayout: Sendable {

    public static let pageAlignment = ExpertBlobLayout.pageAlignment
    public static let tensorAlignment = ExpertBlobLayout.tensorAlignment

    public let expertBlob: any ExpertBlob
    public let expertCount: Int
    public let resident: [TensorPlacement]
    public let residentBytes: Int

    /// The size of `embed.bin`, or **zero when the model has no separate embedding file**.
    ///
    /// GPT-OSS keeps its embedding out of `resident.bin` on purpose: only one row is read per
    /// token, so there is no reason to hold 1.08 GiB in the working set. Gemma 4 ties its
    /// embedding to the output head, which means the whole matrix is read on every token — it
    /// belongs in `resident.bin`, and no separate file exists.
    public let embeddingBytes: Int

    /// Where a source tensor sits in a destination file.
    public struct TensorPlacement: Sendable, Equatable {
        /// The original safetensors key, kept for traceability and verification.
        public let sourceName: String
        public let offset: Int
        public let byteCount: Int
        public var end: Int { offset + byteCount }
    }

    /// The architecture-neutral initializer: a layout is an ordered list of tensors, an
    /// expert blob and a count. Nothing here knows which model produced them, which is what
    /// lets a second architecture reuse the placement rules rather than re-derive them.
    public init(
        residentTensors: [(name: String, byteCount: Int)],
        expertBlob: any ExpertBlob, expertCount: Int, embeddingBytes: Int
    ) {
        self.expertBlob = expertBlob
        self.expertCount = expertCount
        self.embeddingBytes = embeddingBytes

        var placements: [TensorPlacement] = []
        var cursor = 0
        for tensor in residentTensors {
            cursor = alignUp(cursor, to: Self.tensorAlignment)
            placements.append(
                TensorPlacement(
                    sourceName: tensor.name, offset: cursor, byteCount: tensor.byteCount))
            cursor += tensor.byteCount
        }
        self.resident = placements
        self.residentBytes = alignUp(cursor, to: Self.pageAlignment)
    }

    public init(config: GptOssConfig) {
        var tensors: [(name: String, byteCount: Int)] = []
        for layer in 0..<config.layerCount {
            for entry in Self.residentTensorNames(layer: layer) {
                tensors.append((entry.name, entry.bytes(config)))
            }
        }
        tensors.append(("model.norm.weight", 2 * config.hiddenSize))
        // The LM head last: the largest block, read once per token.
        tensors.append(("lm_head.weight", config.lmHeadBytes))

        self.init(
            residentTensors: tensors, expertBlob: config.expertBlobLayout,
            expertCount: config.expertCount, embeddingBytes: config.embeddingBytes)
    }

    public init(config: Gemma4Config) {
        self.init(
            residentTensors: config.residentTensors, expertBlob: config.expertBlobLayout,
            expertCount: config.expertCount,
            // Tied to the output head, so the embedding lives in resident.bin and there is
            // no separate file.
            embeddingBytes: 0)
    }

    // MARK: - Experts

    /// The offset of expert `index`'s blob within its layer file.
    public func expertOffset(_ index: Int) -> Int {
        index * expertBlob.strideBytes
    }

    /// The size of an `experts/layer_XX.bin` file.
    public var expertLayerFileBytes: Int {
        expertCount * expertBlob.strideBytes
    }

    // MARK: - Residents

    /// The safetensors names of a layer's resident tensors, in placement order.
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


    public func placement(of name: String) -> TensorPlacement? {
        resident.first { $0.sourceName == name }
    }
}
