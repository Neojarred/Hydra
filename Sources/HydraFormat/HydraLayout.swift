import Foundation
import HydraCore

/// Disposition physique d'une installation `.hydra`.
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

    public let config: GptOssConfig
    public let expertBlob: ExpertBlobLayout
    public let resident: [TensorPlacement]
    public let residentBytes: Int
    /// What each role is stored at in this installation.
    public let precision: PrecisionPolicy

    /// What a resident tensor is **for**, which is what decides the precision it may be
    /// stored at.
    ///
    /// The role lives here rather than in the quantization code on purpose: when a second
    /// model arrives it declares the role of each of its tensors — something it has to do
    /// anyway — and inherits the precision policy without a line of its own. No knowledge of
    /// Gemma in the quantizer, no knowledge of the quantizer in Gemma.
    ///
    /// It is also what makes D-020's exclusions structural rather than a matter of review:
    /// a router cannot be quantized by accident, because its role forbids it at the source.
    public enum TensorRole: String, Sendable, Equatable, Codable {
        /// q/k/v/o weights. 35 % of the bytes read per token on the 20B.
        case attentionProjection
        /// The output projection. 31 % on its own.
        case lmHead
        /// Never quantized (D-020). 184 KB per layer on the 20B — nothing to win — and a
        /// routing error is not a small perturbation but a **discrete** one: the token goes
        /// to the wrong expert entirely and nothing downstream recovers it.
        case router
        /// Never quantized: negligible in size, and scales multiply everything after them.
        case norm
        /// Never quantized: one value per row.
        case bias
        /// Never quantized: one value per head, and it sets the softmax denominator.
        case sink

        /// Whether D-020 allows this role to leave BF16.
        public var isQuantizable: Bool {
            switch self {
            case .attentionProjection, .lmHead: return true
            case .router, .norm, .bias, .sink: return false
            }
        }
    }

    /// Where a source tensor sits in a destination file.
    public struct TensorPlacement: Sendable, Equatable {
        /// The original safetensors key, kept for traceability and verification.
        public let sourceName: String
        public let offset: Int
        /// Bytes occupied **as stored**, which depends on `precision`.
        public let byteCount: Int
        /// What the tensor is for, hence what precision it may be stored at.
        public let role: TensorRole
        /// The format it is actually stored in.
        public let precision: WeightPrecision
        public var end: Int { offset + byteCount }

        /// Values the tensor holds, whatever the storage format.
        public var valueCount: Int {
            switch precision {
            case .bf16: return byteCount / 2
            case .q8: return byteCount / Q8.bytesPerBlock * Q8.blockSize
            case .mxfp4:
                return byteCount / (MXFP4.packedBytesPerBlock + MXFP4.scaleBytesPerBlock)
                    * MXFP4.blockSize
            }
        }

        /// Where the Q8 scales start, relative to the tensor. Levels occupy everything
        /// before it. Meaningless for other formats.
        public var scaleOffset: Int { offset + Q8.levelBytes(values: valueCount) }
    }

    public init(config: GptOssConfig, precision: PrecisionPolicy = .published) {
        self.config = config
        self.precision = precision
        self.expertBlob = config.expertBlobLayout
        let (placements, total) = Self.makeResidentPlacements(
            config: config, precision: precision)
        self.resident = placements
        self.residentBytes = total
    }

    // MARK: - Experts

    /// The offset of expert `index`'s blob within its layer file.
    public func expertOffset(_ index: Int) -> Int {
        index * expertBlob.strideBytes
    }

    /// The size of an `experts/layer_XX.bin` file.
    public var expertLayerFileBytes: Int {
        config.expertCount * expertBlob.strideBytes
    }

    // MARK: - Residents

    /// The safetensors names of a layer's resident tensors, in placement order.
    public static func residentTensorNames(
        layer: Int
    ) -> [(name: String, bytes: (GptOssConfig) -> Int, role: TensorRole)] {
        let l = layer
        return [
            ("model.layers.\(l).input_layernorm.weight", { 2 * $0.hiddenSize }, .norm),
            ("model.layers.\(l).self_attn.q_proj.weight", { 2 * $0.attentionHeadCount * $0.headDim * $0.hiddenSize }, .attentionProjection),
            ("model.layers.\(l).self_attn.q_proj.bias", { 2 * $0.attentionHeadCount * $0.headDim }, .bias),
            ("model.layers.\(l).self_attn.k_proj.weight", { 2 * $0.keyValueHeadCount * $0.headDim * $0.hiddenSize }, .attentionProjection),
            ("model.layers.\(l).self_attn.k_proj.bias", { 2 * $0.keyValueHeadCount * $0.headDim }, .bias),
            ("model.layers.\(l).self_attn.v_proj.weight", { 2 * $0.keyValueHeadCount * $0.headDim * $0.hiddenSize }, .attentionProjection),
            ("model.layers.\(l).self_attn.v_proj.bias", { 2 * $0.keyValueHeadCount * $0.headDim }, .bias),
            ("model.layers.\(l).self_attn.o_proj.weight", { 2 * $0.hiddenSize * $0.attentionHeadCount * $0.headDim }, .attentionProjection),
            ("model.layers.\(l).self_attn.o_proj.bias", { 2 * $0.hiddenSize }, .bias),
            ("model.layers.\(l).self_attn.sinks", { 2 * $0.attentionHeadCount }, .sink),
            ("model.layers.\(l).post_attention_layernorm.weight", { 2 * $0.hiddenSize }, .norm),
            ("model.layers.\(l).mlp.router.weight", { 2 * $0.expertCount * $0.hiddenSize }, .router),
            ("model.layers.\(l).mlp.router.bias", { 2 * $0.expertCount }, .bias),
        ]
    }

    private static func makeResidentPlacements(
        config: GptOssConfig, precision: PrecisionPolicy
    ) -> ([TensorPlacement], Int) {
        var out: [TensorPlacement] = []
        var cursor = 0

        /// `publishedBytes` is what the tensor occupies as BF16 upstream; the stored size is
        /// derived from it, so a role that changes format cannot silently keep the old size.
        func place(_ name: String, _ publishedBytes: Int, _ role: TensorRole) {
            let format = precision.precision(for: role)
            let values = publishedBytes / 2
            // A tensor that is not a whole number of blocks stays as published rather than
            // being quantized on a truncated range. None of GPT-OSS's dimensions produce
            // one; a future model might, and this must not become a silent corruption.
            let usable = format != .bf16 && values % Q8.blockSize == 0 ? format : .bf16
            let size = usable.byteCount(values: values)

            cursor = alignUp(cursor, to: tensorAlignment)
            out.append(
                TensorPlacement(
                    sourceName: name, offset: cursor, byteCount: size,
                    role: role, precision: usable))
            cursor += size
        }

        // The layers first, in execution order.
        for layer in 0..<config.layerCount {
            for entry in residentTensorNames(layer: layer) {
                place(entry.name, entry.bytes(config), entry.role)
            }
        }
        place("model.norm.weight", 2 * config.hiddenSize, .norm)
        // The LM head last: it is the largest block and is read only once per token.
        place("lm_head.weight", config.lmHeadBytes, .lmHead)

        return (out, alignUp(cursor, to: pageAlignment))
    }

    /// The only tensor in `embed.bin`.
    public var embeddingByteCount: Int { config.embeddingBytes }

    public func placement(of name: String) -> TensorPlacement? {
        resident.first { $0.sourceName == name }
    }
}
