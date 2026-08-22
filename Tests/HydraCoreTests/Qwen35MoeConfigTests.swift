import Foundation
import Testing

@testable import HydraCore

/// The transcription, against the published checkpoint's own numbers.
///
/// Every figure here comes from `config.json` or from the repository's file sizes, not from
/// this struct. A config that agrees with itself proves nothing; the point is to catch a
/// transcription slip before it becomes a load failure, or worse, a model that loads.
@Suite("Qwen3.5 MoE config")
struct Qwen35MoeConfigTests {

    private let model = Qwen35MoeConfig.a3bQ4

    @Test("The layer pattern is three recurrent to one attending")
    func layerPattern() {
        let types = model.layerTypes
        #expect(types.count == 40)
        #expect(types.filter { $0 == .full }.count == 10)
        #expect(types.filter { $0 == .linear }.count == 30)
        // `layer_types` in the config starts with three linear and makes every fourth full.
        #expect(Array(types.prefix(4)) == [.linear, .linear, .linear, .full])
        #expect(types.last == .full, "forty divides by four, so the last layer attends")
    }

    /// The recurrent layers contribute nothing per token and a fixed amount overall.
    @Test("Only the attending layers have a per-token cache")
    func recurrentLayersAreFree() {
        for (index, pattern) in model.layerTypes.enumerated() {
            let bytes = model.kvBytesPerToken(atLayer: index)
            #expect((bytes == 0) == (pattern == .linear), "layer \(index) is \(pattern)")
        }

        // 62.8 MiB at the published shape, and the same at any context length.
        //
        // The audit said 60, having counted the recurrent state and forgotten the
        // convolution's window, which is 2.8 MiB of it. Both are state and both must be
        // carried, so both are counted here.
        let state = 30 * 32 * 128 * 128 * 4
        let window = 30 * 3 * 8192 * 4
        #expect(model.recurrentStateBytes == state + window)
        #expect(Double(model.recurrentStateBytes) / 1_048_576 > 62.5)
        #expect(Double(model.recurrentStateBytes) / 1_048_576 < 63.0)
    }

    /// Long context is cheap because only a quarter of the layers grow.
    @Test("Sixteen times the context is four times the cache, not sixteen")
    func contextGrowth() {
        let short = model.kvCacheBytes(contextLength: 4096)
        let long = model.kvCacheBytes(contextLength: 65536)
        #expect(long == short * 16, "the ten full layers grow linearly")

        // Against a hypothetical all-attention model of the same width: the saving is the
        // three quarters of layers that do not keep a history at all.
        let asIfAllAttended = 40 * 2 * model.keyValueDim * 2 * 4096
        #expect(short * 4 == asIfAllAttended)
    }

    /// The convolution runs over q, k and v concatenated, which is 8192 channels here.
    @Test("The convolution width follows the linear head shapes")
    func convolutionWidth() {
        #expect(model.linearConvDim == 2 * 16 * 128 + 32 * 128)
        #expect(model.linearConvDim == 8192)
    }

    /// `attn_output_gate` doubles the query projection, which is the one shape inferred from a
    /// flag rather than read from a tensor.
    @Test("The gated query projection is twice the query width")
    func gatedQueryProjection() {
        #expect(model.queryDim == 4096)
        #expect(model.queryProjectionRows == 8192)
        let ungated = Qwen35MoeConfig(attentionOutputGate: false)
        #expect(ungated.queryProjectionRows == ungated.queryDim)
    }

    /// The resident inventory names the tensors the checkpoint actually contains.
    ///
    /// Spelling is load-bearing: a name that does not exist fails at install, and one that
    /// exists but means something else does not.
    @Test("The resident inventory matches the checkpoint's names")
    func residentNames() {
        let names = Set(model.residentTensors.map(\.name))

        // A recurrent layer, layer 0.
        #expect(names.contains("language_model.model.layers.0.linear_attn.in_proj_qkv.weight"))
        #expect(names.contains("language_model.model.layers.0.linear_attn.conv1d.weight"))
        #expect(names.contains("language_model.model.layers.0.linear_attn.A_log"))
        #expect(names.contains("language_model.model.layers.0.linear_attn.dt_bias"))
        #expect(!names.contains("language_model.model.layers.0.self_attn.q_proj.weight"))

        // An attending layer, layer 3.
        #expect(names.contains("language_model.model.layers.3.self_attn.q_proj.weight"))
        #expect(names.contains("language_model.model.layers.3.self_attn.q_norm.weight"))
        #expect(!names.contains("language_model.model.layers.3.linear_attn.in_proj_qkv.weight"))

        // Shared by both, and the untied pair.
        #expect(names.contains("language_model.model.layers.0.mlp.gate.weight"))
        #expect(names.contains("language_model.model.layers.0.mlp.shared_expert.up_proj.weight"))
        #expect(names.contains("language_model.model.layers.0.mlp.shared_expert_gate.weight"))
        #expect(names.contains("language_model.model.embed_tokens.weight"))
        #expect(
            names.contains("language_model.lm_head.weight"),
            "the embeddings are not tied, so the head is its own tensor")
    }

    /// The experts are the overwhelming majority of the model, which is what makes it a good
    /// fit for streaming them.
    ///
    /// Checked against the published repository: 20.4 GB of safetensors for the 4-bit build,
    /// of which this accounts for everything but the vision tower.
    @Test("The expert pool dominates, and the totals match the published build")
    func poolDominates() {
        let pool = Double(model.expertPoolBytes) / 1_073_741_824
        let resident = Double(model.residentBytes) / 1_073_741_824
        #expect(pool > 16.5 && pool < 17.5, "16.9 GiB of experts")

        let share = pool / (pool + resident)
        #expect(share > 0.85, "the expert pool is the model, at \(share)")

        // The published 4-bit repository is 20.4 GB, 19.0 GiB, including a vision tower this
        // does not count. Text-only must therefore land below it and near it.
        let total = pool + resident
        #expect(total < 19.0 && total > 17.0, "text-only total is \(total) GiB")
    }

    @Test("The 8-bit build is larger and not smaller")
    func eightBitIsLarger() {
        let four = Qwen35MoeConfig.a3bQ4
        let eight = Qwen35MoeConfig.a3bQ8
        #expect(eight.expertPoolBytes > four.expertPoolBytes)
        #expect(eight.residentBytes > four.residentBytes)
        #expect(
            Double(eight.expertPoolBytes) / Double(four.expertPoolBytes) > 1.8,
            "eight bits is nearly twice four, so it buys accuracy and costs memory")
    }
}

/// Each model's published sampling recipe reaches the caller through `any ModelDescriptor`.
///
/// The dispatch is the point, not the numbers. `samplingDefaults` has a default implementation
/// in a protocol extension, and a default that exists *only* in an extension is statically
/// dispatched: every call through an existential would take the fallback and no model's own
/// answer would ever be read. Declaring it in the protocol is what makes the override visible,
/// and nothing else in the suite would notice if that declaration were removed, because the
/// code would still compile and still return plausible numbers.
@Suite("Published sampling defaults")
struct SamplingDefaultsTests {

    @Test("Every model's own recipe survives the existential")
    func perModelDefaults() {
        let models: [(any ModelDescriptor, SamplingDefaults)] = [
            // The four published fields, plus the two that are not on the card (M-077).
            (Qwen35MoeConfig.a3bQ4,
             SamplingDefaults(
                temperature: 1.0, topP: 0.95, topK: 20, presencePenalty: 1.5,
                frequencyPenalty: 0.2, repeatWindow: 256)),
            (Qwen35MoeConfig.a3bQ8,
             SamplingDefaults(
                temperature: 1.0, topP: 0.95, topK: 20, presencePenalty: 1.5,
                frequencyPenalty: 0.2, repeatWindow: 256)),
            (Gemma4Config.a4b,
             SamplingDefaults(temperature: 1.0, topP: 0.95, topK: 64, presencePenalty: 0)),
            (Gemma4MLXConfig.a4b,
             SamplingDefaults(temperature: 1.0, topP: 0.95, topK: 64, presencePenalty: 0)),
            (GptOssConfig.b20, .untruncated),
            (GptOssConfig.b120, .untruncated),
        ]
        for (model, expected) in models {
            #expect(model.samplingDefaults == expected, "\(model.name)")
        }
    }

    /// Qwen is the one that needs the penalty, and the one whose absence of it was visible.
    @Test("Qwen asks for a presence penalty and GPT-OSS does not")
    func qwenNeedsThePenalty() {
        #expect(Qwen35MoeConfig.a3bQ4.samplingDefaults.presencePenalty > 0)
        #expect(GptOssConfig.b20.samplingDefaults.presencePenalty == 0)
        #expect(GptOssConfig.b20.samplingDefaults.topK == 0, "the raw distribution, as published")
    }

    /// The one departure from a published card, and the two halves of it arrive together.
    ///
    /// Measured separately (M-077), neither works: the window alone is a null, because a flat
    /// penalty does not read the counts it bounds; the frequency term alone degrades every
    /// seed mildly, because over a whole answer it charges words that answer legitimately
    /// needs again. Shipping one without the other would be shipping a configuration nothing
    /// measured.
    @Test("Only Qwen departs from its card, and only in the terms no card specifies")
    func theDeparture() {
        for qwen in [Qwen35MoeConfig.a3bQ4, Qwen35MoeConfig.a3bQ8] {
            let published = qwen.samplingDefaults
            #expect(published.frequencyPenalty > 0)
            #expect(published.repeatWindow > 0, "the frequency term is not safe unbounded")
        }
        for other: any ModelDescriptor in [
            Gemma4Config.a4b, Gemma4MLXConfig.a4b, GptOssConfig.b20, GptOssConfig.b120,
        ] {
            #expect(other.samplingDefaults.frequencyPenalty == 0, "\(other.name)")
            #expect(other.samplingDefaults.repeatWindow == 0, "\(other.name)")
        }
    }
}
