import Foundation

/// Qwen3.5/3.6 MoE, transcribed from the real `config.json` of `Qwen/Qwen3.6-35B-A3B` and the
/// tensor inventory of `lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit`.
///
/// The release is named 3.6 and its `model_type` is `qwen3_5_moe`: it reuses the 3.5
/// architecture class, so the reference implementation to read is `modeling_qwen3_5.py`, which
/// is where D-027 comes from.
///
/// Both published MLX builds are affine, group 64, with per-tensor overrides carried in the
/// config, which is the format `MLXAffineLayout` already decodes. The quantization work is
/// therefore nil; the cost of this model is its attention (D-026).
public struct Qwen35MoeConfig: Sendable {

    public let name: String

    // MARK: - Text model

    public let layerCount: Int
    public let hiddenSize: Int
    public let vocabSize: Int
    public let rmsNormEps: Float
    public let maxPositionEmbeddings: Int

    /// Every fourth layer is full attention and the rest are recurrent, `full_attention_interval`
    /// being 4. Ten of forty, which is why long context is cheap here: the other thirty keep a
    /// fixed state instead of a growing cache (D-027).
    public let fullAttentionInterval: Int

    // MARK: - Full-attention layers

    public let attentionHeadCount: Int
    public let keyValueHeadCount: Int
    public let headDim: Int
    public let partialRotaryFactor: Double
    public let ropeTheta: Double

    /// `attn_output_gate`, which changes the shape of `q_proj`.
    ///
    /// When set, the query projection emits a gate alongside the query, so its row count is
    /// **twice** `attentionHeadCount · headDim`. This is the one shape here inferred from the
    /// flag rather than read from a tensor, and the installer checks it against the
    /// checkpoint's own header rather than trusting it (see `queryProjectionRows`).
    public let attentionOutputGate: Bool

    // MARK: - Linear attention layers

    public let linearKeyHeads: Int
    public let linearValueHeads: Int
    public let linearKeyHeadDim: Int
    public let linearValueHeadDim: Int
    public let linearConvKernel: Int

    // MARK: - Mixture of experts

    public let expertCount: Int
    public let expertsPerToken: Int
    public let moeIntermediateSize: Int
    /// A shared expert, always active, alongside the routed ones. The same role Gemma's dense
    /// MLP plays: a branch that does not wait on the SSD.
    public let sharedExpertIntermediateSize: Int

    // MARK: - Quantization

    public let quantBits: Int
    /// The width of the router and the shared expert's gate, which both MLX builds keep at 8
    /// whatever the base width. Read per tensor rather than assumed, as for Gemma (D-024).
    public let gateBits: Int
    public let groupSize: Int

    public init(
        name: String = "Qwen3.6 35B-A3B",
        layerCount: Int = 40,
        hiddenSize: Int = 2048,
        vocabSize: Int = 248_320,
        rmsNormEps: Float = 1e-6,
        maxPositionEmbeddings: Int = 262_144,
        fullAttentionInterval: Int = 4,
        attentionHeadCount: Int = 16,
        keyValueHeadCount: Int = 2,
        headDim: Int = 256,
        partialRotaryFactor: Double = 0.25,
        ropeTheta: Double = 10_000_000,
        attentionOutputGate: Bool = true,
        linearKeyHeads: Int = 16,
        linearValueHeads: Int = 32,
        linearKeyHeadDim: Int = 128,
        linearValueHeadDim: Int = 128,
        linearConvKernel: Int = 4,
        expertCount: Int = 256,
        expertsPerToken: Int = 8,
        moeIntermediateSize: Int = 512,
        sharedExpertIntermediateSize: Int = 512,
        quantBits: Int = 4,
        gateBits: Int = 8,
        groupSize: Int = 64
    ) {
        self.name = name
        self.layerCount = layerCount
        self.hiddenSize = hiddenSize
        self.vocabSize = vocabSize
        self.rmsNormEps = rmsNormEps
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.fullAttentionInterval = fullAttentionInterval
        self.attentionHeadCount = attentionHeadCount
        self.keyValueHeadCount = keyValueHeadCount
        self.headDim = headDim
        self.partialRotaryFactor = partialRotaryFactor
        self.ropeTheta = ropeTheta
        self.attentionOutputGate = attentionOutputGate
        self.linearKeyHeads = linearKeyHeads
        self.linearValueHeads = linearValueHeads
        self.linearKeyHeadDim = linearKeyHeadDim
        self.linearValueHeadDim = linearValueHeadDim
        self.linearConvKernel = linearConvKernel
        self.expertCount = expertCount
        self.expertsPerToken = expertsPerToken
        self.moeIntermediateSize = moeIntermediateSize
        self.sharedExpertIntermediateSize = sharedExpertIntermediateSize
        self.quantBits = quantBits
        self.gateBits = gateBits
        self.groupSize = groupSize
    }

    /// The published 4-bit MLX build.
    public static let a3bQ4 = Qwen35MoeConfig()
    /// The published 8-bit MLX build. Larger and slower, not smaller: it buys accuracy (D-026).
    public static let a3bQ8 = Qwen35MoeConfig(name: "Qwen3.6 35B-A3B (8-bit)", quantBits: 8)

    /// A miniature for tests, the same shape at a size that fits in a fixture.
    public static let tiny = Qwen35MoeConfig(
        name: "Qwen tiny", layerCount: 8, hiddenSize: 64, vocabSize: 256,
        maxPositionEmbeddings: 512, attentionHeadCount: 4, keyValueHeadCount: 2, headDim: 16,
        linearKeyHeads: 2, linearValueHeads: 4, linearKeyHeadDim: 16, linearValueHeadDim: 16,
        expertCount: 8, expertsPerToken: 2, moeIntermediateSize: 16,
        sharedExpertIntermediateSize: 16, groupSize: 16)

    // MARK: - Derived shapes

    public var queryDim: Int { attentionHeadCount * headDim }
    public var keyValueDim: Int { keyValueHeadCount * headDim }

    /// The rows `q_proj` emits, doubled when the layer gates its own output.
    public var queryProjectionRows: Int { attentionOutputGate ? queryDim * 2 : queryDim }

    /// The channels the depthwise convolution runs over: q, k and v concatenated, before the
    /// split. 8192 at the published shape.
    public var linearConvDim: Int {
        2 * linearKeyHeads * linearKeyHeadDim + linearValueHeads * linearValueHeadDim
    }

    public var linearLayerCount: Int { layerCount - fullAttentionLayerCount }
    public var fullAttentionLayerCount: Int { layerCount / fullAttentionInterval }

    public static let prefix = "language_model.model"

    public func layerTensor(_ suffix: String, layer: Int) -> String {
        "\(Self.prefix).layers.\(layer).\(suffix)"
    }

    public var finalNormTensor: String { "\(Self.prefix).norm.weight" }
    public var embeddingTensor: String { "\(Self.prefix).embed_tokens.weight" }
    /// **Not tied.** `tie_word_embeddings` is false, so the head is its own tensor and both it
    /// and the embedding stay resident. Gemma ties them and reads the same table twice.
    public var headTensor: String { "language_model.lm_head.weight" }

    /// The stems, without a part suffix.
    ///
    /// Quantized, so each of these is three tensors, `.weight`, `.scales` and `.biases`, and
    /// the name that resolves them is the stem. Both forms are needed and neither is derivable
    /// from the other by string surgery a reader should have to verify.
    public var embeddingStem: String { "\(Self.prefix).embed_tokens" }
    public var headStem: String { "language_model.lm_head" }
}

// MARK: - ModelDescriptor

extension Qwen35MoeConfig: ModelDescriptor {

    public var architecture: ModelArchitecture { .qwen35Moe }

    /// Every fourth layer attends; the rest recur.
    ///
    /// Taken from `layer_types` in the config, which lists three `linear_attention` to one
    /// `full_attention` and repeats. Written as arithmetic on the interval rather than as a
    /// literal list so a differently sized sibling of this model stays correct.
    public var layerTypes: [AttentionPattern] {
        (0..<layerCount).map {
            ($0 + 1) % fullAttentionInterval == 0 ? .full : .linear
        }
    }

    /// The geometry of the **attention** layers.
    ///
    /// Every full layer here has the same shape, so the index changes nothing. It is not
    /// meaningful for a `linear` layer, whose shape is `recurrentGeometry`, and a caller that
    /// reaches this without checking the pattern first is asking the wrong question: that is
    /// what `AttentionPattern.keepsKeyValueHistory` is for.
    public func attentionGeometry(atLayer index: Int) -> AttentionGeometry {
        AttentionGeometry(
            headDim: headDim, attentionHeadCount: attentionHeadCount,
            keyValueHeadCount: keyValueHeadCount)
    }

    /// The shape of one recurrent layer.
    public var recurrentGeometry: RecurrentGeometryDescription {
        RecurrentGeometryDescription(
            valueHeads: linearValueHeads, keyHeads: linearKeyHeads,
            keyDim: linearKeyHeadDim, valueDim: linearValueHeadDim,
            convDim: linearConvDim, convKernel: linearConvKernel)
    }

    /// Fixed, and independent of the context: 62.8 MiB at the published shape.
    ///
    /// The state is float32 whatever the checkpoint stores (D-027), so this does not shrink
    /// with the quantization. The convolution's window is counted with it because it is state
    /// in the same sense and forgotten just as silently.
    public var recurrentStateBytes: Int {
        let state = linearValueHeads * linearKeyHeadDim * linearValueHeadDim * 4
        let window = (linearConvKernel - 1) * linearConvDim * 4
        return linearLayerCount * (state + window)
    }

    /// **Zero: this model has no sliding-window layer.**
    ///
    /// Its attention layers see the whole context and its other layers see everything through a
    /// recurrence. Reporting a window would size a ring for layers that never use one.
    public var slidingWindow: Int { 0 }

    /// The nine sub-tensors of one expert, as a concrete layout.
    ///
    /// Named as well as returned through `expertBlob` because the install plan needs the slots
    /// and their byte counts, and an existential hides both.
    public var expertBlobLayout: MLXExpertBlobLayout {
        MLXExpertBlobLayout(
            hiddenSize: hiddenSize, moeIntermediateSize: moeIntermediateSize,
            bits: quantBits, groupSize: groupSize)
    }

    public var expertBlob: any ExpertBlob {
        MLXExpertBlobLayout(
            hiddenSize: hiddenSize, moeIntermediateSize: moeIntermediateSize,
            bits: quantBits, groupSize: groupSize)
    }

    public var expertFormat: String { "mlx-affine" }

    /// **Zero: there is no `embed.bin`.**
    ///
    /// This is the size of a *separate embedding file*, not of the embedding. The table is in
    /// `residentTensors` and lives in `resident.bin`, so a non-zero value here made
    /// `ModelMapping` open a file the installer never writes, and counted the table twice in
    /// `installedBytes`, which is what the catalogue shows before a download.
    public var embeddingFileBytes: Int { 0 }

    public var residentEmbeddingTensor: String? { embeddingTensor }

    /// Everything that stays in memory, in placement order.
    ///
    /// The two layer kinds have entirely different inventories, which is the clearest statement
    /// of what this architecture is: a recurrent layer has no `q_proj`, no `k_proj` and no
    /// cache, and carries a convolution, two per-head parameter vectors and a gated norm that
    /// an attention layer does not.
    public var residentTensors: [(name: String, byteCount: Int)] {
        var out: [(name: String, byteCount: Int)] = []

        func plain(_ name: String, _ elements: Int) { out.append((name, elements * 2)) }
        func quant(_ stem: String, rows: Int, cols: Int, bits: Int) {
            let layout = MLXAffineLayout(
                bits: bits, groupSize: groupSize, rows: rows, cols: cols)
            out.append(("\(stem).weight", layout.weightBytes))
            out.append(("\(stem).scales", layout.scaleBytes))
            out.append(("\(stem).biases", layout.biasBytes))
        }

        for layer in 0..<layerCount {
            func name(_ suffix: String) -> String { layerTensor(suffix, layer: layer) }
            plain(name("input_layernorm.weight"), hiddenSize)

            if attentionPattern(atLayer: layer) == .full {
                quant(name("self_attn.q_proj"), rows: queryProjectionRows, cols: hiddenSize,
                      bits: quantBits)
                quant(name("self_attn.k_proj"), rows: keyValueDim, cols: hiddenSize,
                      bits: quantBits)
                quant(name("self_attn.v_proj"), rows: keyValueDim, cols: hiddenSize,
                      bits: quantBits)
                quant(name("self_attn.o_proj"), rows: hiddenSize, cols: queryDim,
                      bits: quantBits)
                plain(name("self_attn.q_norm.weight"), headDim)
                plain(name("self_attn.k_norm.weight"), headDim)
            } else {
                quant(name("linear_attn.in_proj_qkv"), rows: linearConvDim, cols: hiddenSize,
                      bits: quantBits)
                quant(name("linear_attn.in_proj_z"),
                      rows: linearValueHeads * linearValueHeadDim, cols: hiddenSize,
                      bits: quantBits)
                quant(name("linear_attn.in_proj_b"), rows: linearValueHeads, cols: hiddenSize,
                      bits: quantBits)
                quant(name("linear_attn.in_proj_a"), rows: linearValueHeads, cols: hiddenSize,
                      bits: quantBits)
                quant(name("linear_attn.out_proj"), rows: hiddenSize,
                      cols: linearValueHeads * linearValueHeadDim, bits: quantBits)
                // Depthwise: one kernel a channel, and no channel mixing.
                plain(name("linear_attn.conv1d.weight"), linearConvDim * linearConvKernel)
                plain(name("linear_attn.A_log"), linearValueHeads)
                plain(name("linear_attn.dt_bias"), linearValueHeads)
                plain(name("linear_attn.norm.weight"), linearValueHeadDim)
            }

            plain(name("post_attention_layernorm.weight"), hiddenSize)

            // The router and the shared expert's gate, both 8-bit in the published builds.
            quant(name("mlp.gate"), rows: expertCount, cols: hiddenSize, bits: gateBits)
            quant(name("mlp.shared_expert_gate"), rows: 1, cols: hiddenSize, bits: gateBits)
            for part in ["gate_proj", "up_proj"] {
                quant(name("mlp.shared_expert.\(part)"), rows: sharedExpertIntermediateSize,
                      cols: hiddenSize, bits: quantBits)
            }
            quant(name("mlp.shared_expert.down_proj"), rows: hiddenSize,
                  cols: sharedExpertIntermediateSize, bits: quantBits)
        }

        plain(finalNormTensor, hiddenSize)
        // Both, because the embeddings are not tied.
        quant("\(Self.prefix).embed_tokens", rows: vocabSize, cols: hiddenSize, bits: quantBits)
        quant("language_model.lm_head", rows: vocabSize, cols: hiddenSize, bits: quantBits)
        return out
    }

    /// The checkpoint's names for one layer's experts, in blob order.
    ///
    /// `switch_mlp` rather than Gemma's `switch_glu`, and the same three matrices.
    public func expertTensors(layer: Int) -> [(stem: String, rows: Int, cols: Int)] {
        let stem = layerTensor("mlp.switch_mlp", layer: layer)
        return [
            ("\(stem).gate_proj", moeIntermediateSize, hiddenSize),
            ("\(stem).up_proj", moeIntermediateSize, hiddenSize),
            ("\(stem).down_proj", hiddenSize, moeIntermediateSize),
        ]
    }
}

/// The shape of a recurrent layer, as the descriptor reports it.
///
/// Deliberately a plain description in `HydraCore` rather than the `RecurrentGeometry` the
/// Metal side builds its buffers from: the core module knows shapes and the runtime knows
/// buffers, and this is the seam between them (D-023).
public struct RecurrentGeometryDescription: Sendable, Equatable {
    public let valueHeads: Int
    public let keyHeads: Int
    public let keyDim: Int
    public let valueDim: Int
    public let convDim: Int
    public let convKernel: Int

    public init(
        valueHeads: Int, keyHeads: Int, keyDim: Int, valueDim: Int,
        convDim: Int, convKernel: Int
    ) {
        self.valueHeads = valueHeads
        self.keyHeads = keyHeads
        self.keyDim = keyDim
        self.valueDim = valueDim
        self.convDim = convDim
        self.convKernel = convKernel
    }
}
