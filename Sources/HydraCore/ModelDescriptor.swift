import Foundation

/// Which family a model belongs to. The one place dispatch happens (D-023).
public enum ModelArchitecture: String, Sendable, Codable, CaseIterable {
    case gptOss = "gpt-oss"
    case gemma4 = "gemma-4"

    public var label: String {
        switch self {
        case .gptOss: return "GPT-OSS"
        case .gemma4: return "Gemma 4"
        }
    }
}

/// The attention geometry of one layer.
///
/// Uniform across layers in GPT-OSS and **not** in Gemma 4, whose full-attention layers carry a
/// different head dimension and key/value head count from its sliding ones — so `q_proj` has a
/// different shape depending on the layer. Anything derived from this must be asked for per
/// layer, never taken once and reused.
public struct AttentionGeometry: Sendable, Equatable {
    public let headDim: Int
    public let attentionHeadCount: Int
    public let keyValueHeadCount: Int

    public init(headDim: Int, attentionHeadCount: Int, keyValueHeadCount: Int) {
        self.headDim = headDim
        self.attentionHeadCount = attentionHeadCount
        self.keyValueHeadCount = keyValueHeadCount
    }

    /// The GQA group: how many query heads share one key/value head.
    public var groupedQueryFactor: Int { attentionHeadCount / keyValueHeadCount }
    public var queryDim: Int { attentionHeadCount * headDim }
    public var keyValueDim: Int { keyValueHeadCount * headDim }
}

/// What every consumer that **sizes** things needs to know about a model, whatever it is.
///
/// The dividing line, from D-023: this contract holds what is shared in *kind* — a layer count
/// is a layer count — and never what is shared only in *shape*. It deliberately says nothing
/// about what is inside an expert blob, how a layer computes, or how a prompt is written. Those
/// differ in ways that would make one model declare another's fields, which is how a BF16
/// checkpoint ends up with an imaginary scales tensor.
///
/// `HydraLayout`, `MemoryBudget` and the slot cache consume this. The kernels do not.
public protocol ModelDescriptor: Sendable {

    var architecture: ModelArchitecture { get }
    var name: String { get }

    var layerCount: Int { get }
    var hiddenSize: Int { get }
    var vocabSize: Int { get }
    var rmsNormEps: Float { get }
    var maxPositionEmbeddings: Int { get }

    /// Routed experts per layer, and how many are active per token.
    var expertCount: Int { get }
    var expertsPerToken: Int { get }

    /// The reach of a sliding-attention layer.
    var slidingWindow: Int { get }
    /// Each layer's attention pattern, in order. Read, never derived from the index: GPT-OSS
    /// alternates and Gemma runs five sliding to one full.
    var layerTypes: [AttentionPattern] { get }
    func attentionGeometry(atLayer index: Int) -> AttentionGeometry

    /// One expert's on-disk shape. Three numbers, never its contents.
    var expertBlob: any ExpertBlob { get }

    /// How the experts are stored, as the **source checkpoint** publishes them.
    ///
    /// Recorded, not chosen. GPT-OSS ships MXFP4 because OpenAI trained it that way; Gemma 4
    /// ships BF16, and its QAT sibling will ship `q4_0`. A manifest that hardcoded one value
    /// would make every new format a new branch rather than a new value.
    var expertFormat: String { get }

    /// Every tensor that must be resident, in placement order, with its size.
    var residentTensors: [(name: String, byteCount: Int)] { get }

    /// The size of a separate embedding file, or **zero when the embedding is resident**
    /// because it is tied to the output head.
    var embeddingFileBytes: Int { get }

    /// The resident tensor holding the embedding, when there is no separate file.
    ///
    /// `nil` for a model that keeps its embedding out of the working set, which is what
    /// GPT-OSS does deliberately: one row is read per token, so wiring down 1.08 GiB would be
    /// waste. Gemma ties it to the output head, so the whole matrix is read every token and
    /// belongs with the resident weights — and the runtime has to be told where.
    var residentEmbeddingTensor: String? { get }
}

extension ModelDescriptor {
    public func attentionPattern(atLayer index: Int) -> AttentionPattern { layerTypes[index] }

    public var fullAttentionLayerCount: Int { layerTypes.count { $0 == .full } }
    public var slidingAttentionLayerCount: Int { layerTypes.count { $0 == .sliding } }

    public var expertSlotBytes: Int { expertBlob.strideBytes }
    public var expertPoolBytes: Int { layerCount * expertCount * expertBlob.sourceBytes }
    public var residentBytes: Int { residentTensors.reduce(0) { $0 + $1.byteCount } }

    /// What lands on disk: the experts, the resident weights, and a separate embedding file
    /// where one exists.
    public var installedBytes: Int {
        expertPoolBytes + residentBytes + embeddingFileBytes
    }

    // MARK: - KV cache

    /// The physical rows of a sliding layer's ring: the window, plus one prefill chunk of
    /// margin so a chunk can be written before the oldest rows are needed.
    public var slidingRingRows: Int { slidingWindow + 128 }

    /// KV bytes per token for one layer, keys and values, in FP16.
    ///
    /// **Per layer, and that is the whole point.** GPT-OSS has one attention geometry, so a
    /// single number sufficed and the old formula multiplied it by a layer count. Gemma's
    /// sliding layers carry 8 key/value heads of 256 and its full layers 2 of 512 — 8 KiB
    /// against 4 KiB per token — so a model-wide constant would misbudget every layer.
    public func kvBytesPerToken(atLayer index: Int) -> Int {
        2 * attentionGeometry(atLayer: index).keyValueDim * 2
    }

    /// The FP16 KV cache for a given context.
    ///
    /// Full layers hold the whole context; sliding layers hold a fixed ring however long the
    /// conversation runs, which is what keeps a long context affordable.
    public func kvCacheBytes(contextLength: Int) -> Int {
        (0..<layerCount).reduce(0) { total, layer in
            let rows = layerTypes[layer] == .full ? contextLength : slidingRingRows
            return total + kvBytesPerToken(atLayer: layer) * rows
        }
    }

    // MARK: - Per-token volumes

    /// The bytes the GPU must move to decode one token, a perfect expert cache included: the
    /// selected experts' weights are read whatever happens.
    public var gpuBytesPerDecodedToken: Int {
        residentBytes + layerCount * expertsPerToken * expertBlob.sourceBytes
    }

    /// The bytes to read from SSD for one token, as a function of the cache hit rate.
    public func diskBytesPerDecodedToken(cacheHitRate: Double) -> Int {
        let all = layerCount * expertsPerToken * expertBlob.sourceBytes
        return Int(Double(all) * min(max(1.0 - cacheHitRate, 0), 1))
    }
}

// MARK: - Conformances

extension GptOssConfig: ModelDescriptor {
    public var architecture: ModelArchitecture { .gptOss }
    public var expertBlob: any ExpertBlob { expertBlobLayout }
    public var expertFormat: String { "mxfp4" }
    public var embeddingFileBytes: Int { embeddingBytes }
    public var residentEmbeddingTensor: String? { nil }

    /// GPT-OSS's resident tensors: the layers in execution order, the final norm, then the LM
    /// head — the largest block, and read once per token.
    public var residentTensors: [(name: String, byteCount: Int)] {
        var out: [(name: String, byteCount: Int)] = []
        for layer in 0..<layerCount {
            for entry in Self.residentTensorNames(layer: layer) {
                out.append((entry.name, entry.bytes(self)))
            }
        }
        out.append(("model.norm.weight", 2 * hiddenSize))
        out.append(("lm_head.weight", lmHeadBytes))
        return out
    }
}

extension Gemma4MLXConfig: ModelDescriptor {
    /// The same architecture as the BF16 build — it is the *encoding* that differs, and
    /// `expertFormat` is what carries that (D-023: recorded, not chosen).
    public var architecture: ModelArchitecture { .gemma4 }
    public var expertFormat: String { "mlx-affine-\(quantBits)bit-g\(groupSize)" }

    public var name: String { base.name + " (MLX 4-bit)" }
    public var layerCount: Int { base.layerCount }
    public var hiddenSize: Int { base.hiddenSize }
    public var vocabSize: Int { base.vocabSize }
    public var rmsNormEps: Float { base.rmsNormEps }
    public var maxPositionEmbeddings: Int { base.maxPositionEmbeddings }
    public var expertCount: Int { base.expertCount }
    public var expertsPerToken: Int { base.expertsPerToken }
    public var slidingWindow: Int { base.slidingWindow }
    public var layerTypes: [AttentionPattern] { base.layerTypes }
    public func attentionGeometry(atLayer index: Int) -> AttentionGeometry {
        base.attentionGeometry(atLayer: index)
    }

    public var expertBlob: any ExpertBlob { expertBlobLayout }
    /// Tied to the output head, so no separate file — as in the BF16 build.
    public var embeddingFileBytes: Int { 0 }
    public var residentEmbeddingTensor: String? { embeddingTensor }
}

extension Gemma4Config: ModelDescriptor {
    public var architecture: ModelArchitecture { .gemma4 }
    public var expertBlob: any ExpertBlob { expertBlobLayout }
    /// As published. The QAT sibling will report `q4_0` here and change nothing else.
    public var expertFormat: String { "bf16" }
    /// Tied to the output head, so the embedding is a resident tensor and no separate file
    /// exists.
    public var embeddingFileBytes: Int { 0 }
    public var residentEmbeddingTensor: String? { "model.language_model.embed_tokens.weight" }
}
