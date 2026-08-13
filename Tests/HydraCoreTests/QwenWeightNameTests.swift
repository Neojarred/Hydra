import Foundation
import Testing

@testable import HydraCore

/// The names the runner will ask for, against the names the installer will make resident.
///
/// These two lists are written in different files for different reasons and nothing makes them
/// agree. A name asked for but never installed fails at load, after a 20 GB download. A name
/// installed but never asked for is dead weight carried forever, and nothing at all reports it.
///
/// Duplicated deliberately in `QwenMLXWeights`, as a list rather than spelled at each use, so
/// this comparison is possible at all.
@Suite("Qwen tensor names")
struct QwenWeightNameTests {

    private let model = Qwen35MoeConfig.a3bQ4

    /// The suffixes the runner reads, mirrored from `QwenMLXWeights` so `HydraCore` does not
    /// have to depend on `HydraMetal` to be checked against it.
    private let linearProjections = [
        "linear_attn.in_proj_qkv", "linear_attn.in_proj_z",
        "linear_attn.in_proj_a", "linear_attn.in_proj_b", "linear_attn.out_proj",
    ]
    private let linearPlain = [
        "linear_attn.conv1d.weight", "linear_attn.A_log", "linear_attn.dt_bias",
        "linear_attn.norm.weight",
    ]
    private let attentionProjections = [
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    ]
    private let attentionPlain = ["self_attn.q_norm.weight", "self_attn.k_norm.weight"]
    private let sharedProjections = [
        "mlp.gate", "mlp.shared_expert_gate",
        "mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj",
        "mlp.shared_expert.down_proj",
    ]
    private let sharedPlain = ["input_layernorm.weight", "post_attention_layernorm.weight"]

    @Test("Everything the runner reads is something the installer makes resident")
    func everyReadIsInstalled() {
        let resident = Set(model.residentTensors.map(\.name))

        for layer in 0..<model.layerCount {
            let isLinear = model.attentionPattern(atLayer: layer) == .linear
            let projections = (isLinear ? linearProjections : attentionProjections)
                + sharedProjections
            let plain = (isLinear ? linearPlain : attentionPlain) + sharedPlain

            for suffix in projections {
                for part in ["weight", "scales", "biases"] {
                    let name = model.layerTensor("\(suffix).\(part)", layer: layer)
                    #expect(resident.contains(name), "layer \(layer) reads \(name), never installed")
                }
            }
            for suffix in plain {
                let name = model.layerTensor(suffix, layer: layer)
                #expect(resident.contains(name), "layer \(layer) reads \(name), never installed")
            }
        }

        for name in [model.finalNormTensor, model.embeddingTensor, model.headTensor] {
            #expect(resident.contains(name), "\(name) is read and never installed")
        }
    }

    /// And nothing is installed that nothing reads.
    ///
    /// The other direction, which fails silently rather than loudly: a tensor carried in the
    /// download, mapped into memory, and never touched.
    @Test("Everything the installer carries is something the runner reads")
    func everyInstalledIsRead() {
        var expected = Set<String>()
        for layer in 0..<model.layerCount {
            let isLinear = model.attentionPattern(atLayer: layer) == .linear
            for suffix in (isLinear ? linearProjections : attentionProjections) + sharedProjections {
                for part in ["weight", "scales", "biases"] {
                    expected.insert(model.layerTensor("\(suffix).\(part)", layer: layer))
                }
            }
            for suffix in (isLinear ? linearPlain : attentionPlain) + sharedPlain {
                expected.insert(model.layerTensor(suffix, layer: layer))
            }
        }
        for stem in ["\(Qwen35MoeConfig.prefix).embed_tokens", "language_model.lm_head"] {
            for part in ["weight", "scales", "biases"] { expected.insert("\(stem).\(part)") }
        }
        expected.insert(model.finalNormTensor)

        let resident = Set(model.residentTensors.map(\.name))
        let unread = resident.subtracting(expected)
        #expect(unread.isEmpty, "installed and never read: \(unread.sorted().prefix(5))")
    }

    /// The router and the shared expert's gate are 8-bit; everything else follows the build.
    @Test("Only the two gates keep their own width")
    func gateWidths() {
        for suffix in ["mlp.gate", "mlp.shared_expert_gate"] {
            let layout = MLXAffineLayout(
                bits: model.gateBits, groupSize: model.groupSize,
                rows: model.expertCount, cols: model.hiddenSize)
            #expect(layout.bits == 8, "\(suffix) is 8-bit in both published builds")
        }
        #expect(model.quantBits == 4, "the 4-bit build")
        #expect(Qwen35MoeConfig.a3bQ8.gateBits == 8, "and the 8-bit build's gates are 8 too")
    }
}
