import Foundation

/// A layer's attention pattern.
public enum AttentionPattern: String, Sendable, Codable {
    /// Causal attention bounded to a sliding window.
    case sliding
    /// Attention causale sur tout le contexte.
    case full
}

/// GPT-OSS's configuration, transcribed from the real `config.json` of the repositories
/// `openai/gpt-oss-20b` et `openai/gpt-oss-120b`.
///
/// Per the brief, this is a **concrete structure, not a generic model contract**. The
/// abstraction will only be extracted in phase 3, from two engines
/// qui fonctionnent (docs/00-FEASIBILITY.md, §6).
public struct GptOssConfig: Sendable, Equatable {

    public let name: String
    public let layerCount: Int
    public let expertCount: Int

    // The defaults are the ones shared by the 20B and the 120B. They are parameterizable
    // only to allow miniature configurations in tests: checking the repack on a real model
    // would mean downloading 12.8 GiB, whereas the logic to validate is purely
    // structural.
    public let expertsPerToken: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let headDim: Int
    public let attentionHeadCount: Int
    public let keyValueHeadCount: Int
    public let vocabSize: Int
    public let slidingWindow: Int
    public let ropeTheta: Float
    public let rmsNormEps: Float
    public let swigluLimit: Float
    public let maxPositionEmbeddings: Int

    // YaRN: base 4096 extended to 131072.
    public let yarnFactor: Float
    public let yarnBetaFast: Float
    public let yarnBetaSlow: Float
    public let yarnOriginalContext: Int

    public static let b20 = GptOssConfig(name: "GPT-OSS 20B", layerCount: 24, expertCount: 32)
    public static let b120 = GptOssConfig(name: "GPT-OSS 120B", layerCount: 36, expertCount: 128)

    /// A miniature configuration, reserved for tests. Keeps every structural invariant of the
    /// real model — multiples of the MXFP4 block size, GQA, alternating attention patterns —
    /// for a few tens of KiB instead of 12.8 GiB.
    public static let tiny = GptOssConfig(
        name: "GPT-OSS tiny (test)", layerCount: 4, expertCount: 6,
        expertsPerToken: 2, hiddenSize: 64, intermediateSize: 64,
        headDim: 16, attentionHeadCount: 4, keyValueHeadCount: 2,
        vocabSize: 128, slidingWindow: 8)

    public init(
        name: String,
        layerCount: Int,
        expertCount: Int,
        expertsPerToken: Int = 4,
        hiddenSize: Int = 2880,
        intermediateSize: Int = 2880,
        headDim: Int = 64,
        attentionHeadCount: Int = 64,
        keyValueHeadCount: Int = 8,
        vocabSize: Int = 201_088,
        slidingWindow: Int = 128,
        ropeTheta: Float = 150_000,
        rmsNormEps: Float = 1e-5,
        swigluLimit: Float = 7.0,
        maxPositionEmbeddings: Int = 131_072,
        yarnFactor: Float = 32,
        yarnBetaFast: Float = 32,
        yarnBetaSlow: Float = 1,
        yarnOriginalContext: Int = 4096
    ) {
        precondition(hiddenSize % MXFP4Layout.blockSize == 0)
        precondition(intermediateSize % MXFP4Layout.blockSize == 0)
        precondition(attentionHeadCount % keyValueHeadCount == 0)
        self.name = name
        self.layerCount = layerCount
        self.expertCount = expertCount
        self.expertsPerToken = expertsPerToken
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.headDim = headDim
        self.attentionHeadCount = attentionHeadCount
        self.keyValueHeadCount = keyValueHeadCount
        self.vocabSize = vocabSize
        self.slidingWindow = slidingWindow
        self.ropeTheta = ropeTheta
        self.rmsNormEps = rmsNormEps
        self.swigluLimit = swigluLimit
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.yarnFactor = yarnFactor
        self.yarnBetaFast = yarnBetaFast
        self.yarnBetaSlow = yarnBetaSlow
        self.yarnOriginalContext = yarnOriginalContext
    }

    /// `layer_types` alternates starting from `sliding_attention` at layer 0.
    public func attentionPattern(atLayer index: Int) -> AttentionPattern {
        index.isMultiple(of: 2) ? .sliding : .full
    }

    public var fullAttentionLayerCount: Int { layerCount / 2 }
    public var slidingAttentionLayerCount: Int { layerCount - fullAttentionLayerCount }

    /// The GQA group: how many query heads share one key/value head.
    public var groupedQueryFactor: Int { attentionHeadCount / keyValueHeadCount }

    // MARK: - Exact sizes, derived from the safetensors headers

    private func bf16(_ dims: Int...) -> Int { dims.reduce(2, *) }

    /// An expert blob's layout. The single source of truth, shared by the on-disk format and
    /// the sizing of memory slots.
    public var expertBlobLayout: ExpertBlobLayout { ExpertBlobLayout(config: self) }

    /// The size of an MXFP4 expert blob **in the source checkpoint**, BF16 biases included.
    /// It is 13,236,480 bytes for both models.
    public var expertBlobBytes: Int { expertBlobLayout.sourceBytes }

    /// The size of a memory slot: the laid-out blob, page-aligned.
    /// It is this value, not `expertBlobBytes`, that sizes the cache.
    public var expertSlotBytes: Int { expertBlobLayout.strideBytes }

    /// The full pool of routed experts, as it lives on disk.
    public var expertPoolBytes: Int { layerCount * expertCount * expertBlobBytes }

    /// A layer's attention, router and norm weights. All in BF16.
    public var residentPerLayerBytes: Int {
        let qDim = attentionHeadCount * headDim
        let kvDim = keyValueHeadCount * headDim
        let q = bf16(qDim, hiddenSize) + bf16(qDim)
        let k = bf16(kvDim, hiddenSize) + bf16(kvDim)
        let v = k
        let o = bf16(hiddenSize, qDim) + bf16(hiddenSize)
        let sinks = bf16(attentionHeadCount)
        let router = bf16(expertCount, hiddenSize) + bf16(expertCount)
        let norms = 2 * bf16(hiddenSize)
        return q + k + v + o + sinks + router + norms
    }

    /// The embedding table. Deliberately **excluded** from the resident weights: we read only
    /// one row per token, so it stays mapped and paged on demand rather than wired into the
    /// Metal working set (§2.2c of the feasibility study).
    public var embeddingBytes: Int { bf16(vocabSize, hiddenSize) }

    /// The LM head. Read in full on every token: this one must stay resident.
    public var lmHeadBytes: Int { bf16(vocabSize, hiddenSize) }

    /// The weights that must occupy the Metal working set permanently.
    public var residentBytes: Int {
        lmHeadBytes + layerCount * residentPerLayerBytes + bf16(hiddenSize)
    }

    /// The size of a complete installation on disk.
    public var installedBytes: Int {
        expertPoolBytes + residentBytes + embeddingBytes
    }

    // MARK: - KV cache

    /// KV bytes per token per full-attention layer, in FP16.
    public var kvBytesPerTokenPerFullLayer: Int { 2 * keyValueHeadCount * headDim * 2 }

    /// The physical rows of a sliding-window layer's ring.
    /// The window is 128 tokens; we add a margin of one prefill chunk.
    public var slidingRingRows: Int { slidingWindow + 128 }

    /// The total size of the FP16 KV cache for a given context.
    public func kvCacheBytes(contextLength: Int) -> Int {
        let full = fullAttentionLayerCount * kvBytesPerTokenPerFullLayer * contextLength
        let sliding = slidingAttentionLayerCount * kvBytesPerTokenPerFullLayer * slidingRingRows
        return full + sliding
    }

    // MARK: - Volumes par token

    /// The bytes the GPU must move to decode one token, a perfect expert cache included: the
    /// selected experts' weights are read whatever happens.
    public var gpuBytesPerDecodedToken: Int {
        layerCount * residentPerLayerBytes
            + lmHeadBytes
            + layerCount * expertsPerToken * expertBlobBytes
    }

    /// The bytes to read from SSD for one token, as a function of the cache hit rate.
    public func diskBytesPerDecodedToken(cacheHitRate: Double) -> Int {
        let all = layerCount * expertsPerToken * expertBlobBytes
        return Int(Double(all) * (1.0 - cacheHitRate).clamped(to: 0...1))
    }
}

/// The MXFP4 layout constants, duplicated here to avoid HydraCore depending on HydraFormat.
/// The two definitions are locked together by a consistency test.
public enum MXFP4Layout {
    public static let blockSize = 32
    public static let packedBytesPerBlock = 16
    public static let scaleBytesPerBlock = 1
}

extension Comparable {
    fileprivate func clamped(to range: ClosedRange<Self>) -> Self {
        min(max(self, range.lowerBound), range.upperBound)
    }
}
