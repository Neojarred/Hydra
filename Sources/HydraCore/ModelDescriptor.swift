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
}

// MARK: - Conformances

extension GptOssConfig: ModelDescriptor {
    public var architecture: ModelArchitecture { .gptOss }
    public var expertBlob: any ExpertBlob { expertBlobLayout }
    public var expertFormat: String { "mxfp4" }
    public var embeddingFileBytes: Int { embeddingBytes }

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

extension Gemma4Config: ModelDescriptor {
    public var architecture: ModelArchitecture { .gemma4 }
    public var expertBlob: any ExpertBlob { expertBlobLayout }
    /// As published. The QAT sibling will report `q4_0` here and change nothing else.
    public var expertFormat: String { "bf16" }
    /// Tied to the output head, so the embedding is a resident tensor and no separate file
    /// exists.
    public var embeddingFileBytes: Int { 0 }
}
