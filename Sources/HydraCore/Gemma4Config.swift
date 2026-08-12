import Foundation

/// Gemma 4 26B-A4B's configuration, transcribed from the real `config.json` of
/// `google/gemma-4-26B-A4B`.
///
/// Concrete, like `GptOssConfig`, and for the same reason: the shared contract is being
/// extracted from two working engines rather than designed in advance
/// (docs/00-FEASIBILITY.md, §6). What the two already share lives in `ExpertBlob` and
/// `AttentionPattern`; the rest is deliberately duplicated until a third model shows which
/// parts are genuinely common.
///
/// The operator semantics that go with these numbers, and that cannot be guessed from
/// them, are recorded in **D-022**. Read that before implementing anything here.
public struct Gemma4Config: Sendable, Equatable {

    public let name: String
    public let layerCount: Int
    public let expertCount: Int
    public let expertsPerToken: Int
    public let hiddenSize: Int
    /// The dense MLP beside the experts, always active. Not the experts' width.
    public let intermediateSize: Int
    /// One expert's width. A quarter of the dense MLP's.
    public let moeIntermediateSize: Int
    public let attentionHeadCount: Int
    public let vocabSize: Int
    public let slidingWindow: Int
    public let rmsNormEps: Float
    public let maxPositionEmbeddings: Int

    /// Sliding-attention layers: 16 heads of 256, 8 key/value heads.
    public let slidingHeadDim: Int
    public let slidingKeyValueHeadCount: Int
    /// Full-attention layers: 16 heads of **512**, and only **2** key/value heads. The
    /// geometry is not the sliding one, so `q_proj` has a different shape depending on the
    /// layer, 8192 rows here against 4096 there.
    public let globalHeadDim: Int
    public let globalKeyValueHeadCount: Int

    /// θ for each pattern. Full layers also rotate only a quarter of the head dimension; see
    /// `partialRotaryFactor`.
    public let slidingRopeTheta: Float
    public let globalRopeTheta: Float
    /// `rope_type: "proportional"` on full layers. The first quarter of the frequency pairs
    /// rotate; the rest get an inverse frequency of **zero**, which is `cos = 1, sin = 0`,
    /// the identity. That is why partial rotation needs no kernel change (D-022).
    public let partialRotaryFactor: Float

    /// `logits = c · tanh(logits / c)` before sampling.
    public let finalLogitSoftcapping: Float
    /// Embeddings are multiplied by `sqrt(hiddenSize)` on lookup. Omitting it mis-scales the
    /// entire forward pass, and nothing in the checkpoint reveals it.
    public var embeddingScale: Float { Float(hiddenSize).squareRoot() }

    /// K and V share one projection on **full-attention layers only**, so those layers have
    /// no `v_proj` tensor at all.
    public let keyEqualsValue: Bool

    public static let a4b = Gemma4Config(name: "Gemma 4 26B-A4B")

    /// A miniature configuration for tests, keeping every structural invariant: the 5:1
    /// pattern, two head geometries, a dense MLP beside the experts.
    public static let tiny = Gemma4Config(
        name: "Gemma 4 tiny (test)", layerCount: 6, expertCount: 8, expertsPerToken: 2,
        hiddenSize: 64, intermediateSize: 48, moeIntermediateSize: 16,
        attentionHeadCount: 4, vocabSize: 256, slidingWindow: 8,
        slidingHeadDim: 16, slidingKeyValueHeadCount: 2,
        globalHeadDim: 32, globalKeyValueHeadCount: 1)

    public init(
        name: String,
        layerCount: Int = 30,
        expertCount: Int = 128,
        expertsPerToken: Int = 8,
        hiddenSize: Int = 2816,
        intermediateSize: Int = 2112,
        moeIntermediateSize: Int = 704,
        attentionHeadCount: Int = 16,
        vocabSize: Int = 262_144,
        slidingWindow: Int = 1024,
        rmsNormEps: Float = 1e-6,
        maxPositionEmbeddings: Int = 262_144,
        slidingHeadDim: Int = 256,
        slidingKeyValueHeadCount: Int = 8,
        globalHeadDim: Int = 512,
        globalKeyValueHeadCount: Int = 2,
        slidingRopeTheta: Float = 10_000,
        globalRopeTheta: Float = 1_000_000,
        partialRotaryFactor: Float = 0.25,
        finalLogitSoftcapping: Float = 30,
        keyEqualsValue: Bool = true
    ) {
        self.name = name
        self.layerCount = layerCount
        self.expertCount = expertCount
        self.expertsPerToken = expertsPerToken
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.moeIntermediateSize = moeIntermediateSize
        self.attentionHeadCount = attentionHeadCount
        self.vocabSize = vocabSize
        self.slidingWindow = slidingWindow
        self.rmsNormEps = rmsNormEps
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.slidingHeadDim = slidingHeadDim
        self.slidingKeyValueHeadCount = slidingKeyValueHeadCount
        self.globalHeadDim = globalHeadDim
        self.globalKeyValueHeadCount = globalKeyValueHeadCount
        self.slidingRopeTheta = slidingRopeTheta
        self.globalRopeTheta = globalRopeTheta
        self.partialRotaryFactor = partialRotaryFactor
        self.finalLogitSoftcapping = finalLogitSoftcapping
        self.keyEqualsValue = keyEqualsValue
    }

    // MARK: - Layers

    /// `layer_types`: **five sliding to one full**, repeating. Not GPT-OSS's alternation, and
    /// not derivable from parity.
    public var layerTypes: [AttentionPattern] {
        (0..<layerCount).map { ($0 + 1) % 6 == 0 ? .full : .sliding }
    }

    public func attentionPattern(atLayer index: Int) -> AttentionPattern { layerTypes[index] }

    public func attentionGeometry(atLayer index: Int) -> AttentionGeometry {
        let full = attentionPattern(atLayer: index) == .full
        return AttentionGeometry(
            headDim: full ? globalHeadDim : slidingHeadDim,
            attentionHeadCount: attentionHeadCount,
            keyValueHeadCount: full ? globalKeyValueHeadCount : slidingKeyValueHeadCount)
    }

    /// Whether a layer projects its own values. False on full layers, where `attention_k_eq_v`
    /// makes V reuse `k_proj`'s output, with a different normalization, and no RoPE (D-022).
    public func hasValueProjection(atLayer index: Int) -> Bool {
        !(keyEqualsValue && attentionPattern(atLayer: index) == .full)
    }

    public func ropeTheta(atLayer index: Int) -> Float {
        attentionPattern(atLayer: index) == .full ? globalRopeTheta : slidingRopeTheta
    }

    /// How many of a head's frequency pairs actually rotate. The remainder carry a zero
    /// inverse frequency and pass through unchanged.
    public func rotatingPairs(atLayer index: Int) -> Int {
        let geometry = attentionGeometry(atLayer: index)
        guard attentionPattern(atLayer: index) == .full else { return geometry.headDim / 2 }
        return Int(partialRotaryFactor * Float(geometry.headDim)) / 2
    }

    // MARK: - Sizes

    private func bf16(_ dims: Int...) -> Int { dims.reduce(2, *) }

    public var expertBlobLayout: GemmaExpertBlobLayout {
        GemmaExpertBlobLayout(hiddenSize: hiddenSize, moeIntermediateSize: moeIntermediateSize)
    }

    public var expertBlobBytes: Int { expertBlobLayout.sourceBytes }
    public var expertSlotBytes: Int { expertBlobLayout.strideBytes }
    public var expertPoolBytes: Int { layerCount * expertCount * expertBlobBytes }

    /// The embedding, which is **also the output head**, `tie_word_embeddings` is true and no
    /// `lm_head` tensor exists. It is therefore read on every token and must stay resident,
    /// where GPT-OSS deliberately maps its embedding outside the working set.
    public var embeddingBytes: Int { bf16(vocabSize, hiddenSize) }

    /// Every resident tensor of one layer, in execution order, with its size.
    ///
    /// Names carry the `model.language_model.` prefix the checkpoint uses; the vision and
    /// audio towers live under different prefixes and are excluded from the plan (D-021).
    ///
    /// Note what is **absent**: no `v_norm`, it is built `with_scale=False` and has no
    /// tensor, and no `v_proj` on full-attention layers.
    public func residentTensors(atLayer index: Int) -> [(name: String, byteCount: Int)] {
        let l = "model.language_model.layers.\(index)"
        let g = attentionGeometry(atLayer: index)
        var out: [(String, Int)] = [
            ("\(l).input_layernorm.weight", bf16(hiddenSize)),
            ("\(l).self_attn.q_proj.weight", bf16(g.queryDim, hiddenSize)),
            ("\(l).self_attn.q_norm.weight", bf16(g.headDim)),
            ("\(l).self_attn.k_proj.weight", bf16(g.keyValueDim, hiddenSize)),
            ("\(l).self_attn.k_norm.weight", bf16(g.headDim)),
        ]
        if hasValueProjection(atLayer: index) {
            out.append(("\(l).self_attn.v_proj.weight", bf16(g.keyValueDim, hiddenSize)))
        }
        out += [
            ("\(l).self_attn.o_proj.weight", bf16(hiddenSize, g.queryDim)),
            ("\(l).post_attention_layernorm.weight", bf16(hiddenSize)),
            // The dense MLP, always active, the I/O-overlap partner GPT-OSS lacks.
            ("\(l).pre_feedforward_layernorm.weight", bf16(hiddenSize)),
            ("\(l).mlp.gate_proj.weight", bf16(intermediateSize, hiddenSize)),
            ("\(l).mlp.up_proj.weight", bf16(intermediateSize, hiddenSize)),
            ("\(l).mlp.down_proj.weight", bf16(hiddenSize, intermediateSize)),
            ("\(l).post_feedforward_layernorm_1.weight", bf16(hiddenSize)),
            // The expert branch, which reads the residual rather than the MLP's output.
            ("\(l).pre_feedforward_layernorm_2.weight", bf16(hiddenSize)),
            ("\(l).router.proj.weight", bf16(expertCount, hiddenSize)),
            ("\(l).router.scale", bf16(hiddenSize)),
            ("\(l).router.per_expert_scale", bf16(expertCount)),
            ("\(l).post_feedforward_layernorm_2.weight", bf16(hiddenSize)),
            ("\(l).post_feedforward_layernorm.weight", bf16(hiddenSize)),
            ("\(l).layer_scalar", bf16(1)),
        ]
        return out.map { (name: $0.0, byteCount: $0.1) }
    }

    /// Every resident tensor, layers then the tensors outside them.
    public var residentTensors: [(name: String, byteCount: Int)] {
        var out = (0..<layerCount).flatMap { residentTensors(atLayer: $0) }
        out.append(("model.language_model.norm.weight", bf16(hiddenSize)))
        // Last, as GPT-OSS places its LM head: the largest block, read once per token.
        out.append(("model.language_model.embed_tokens.weight", embeddingBytes))
        return out
    }

    public var residentBytes: Int { residentTensors.reduce(0) { $0 + $1.byteCount } }

    /// What lands on disk for the text model. The vision and audio towers are excluded.
    public var installedBytes: Int { expertPoolBytes + residentBytes }
}
