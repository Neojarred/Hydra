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
    /// embedding to the output head, which means the whole matrix is read on every token, it
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

    /// Places any model that can describe itself. There is no per-architecture branch here:
    /// a layout is a tensor list, a blob and a count (D-023).
    public init(model: any ModelDescriptor) {
        self.init(
            residentTensors: model.residentTensors, expertBlob: model.expertBlob,
            expertCount: model.expertCount, embeddingBytes: model.embeddingFileBytes)
    }

    public init(config: GptOssConfig) { self.init(model: config) }
    public init(config: Gemma4Config) { self.init(model: config) }

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

    /// Kept as a forwarding alias: the list now belongs to the configuration, because each
    /// architecture owns the tensors it declares (D-023).
    public static func residentTensorNames(
        layer: Int
    ) -> [(name: String, bytes: (GptOssConfig) -> Int)] {
        GptOssConfig.residentTensorNames(layer: layer)
    }


    public func placement(of name: String) -> TensorPlacement? {
        resident.first { $0.sourceName == name }
    }
}
