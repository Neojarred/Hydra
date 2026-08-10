import Foundation
import Testing

@testable import HydraCore

/// The MLX descriptor against the published checkpoint's own inventory.
///
/// Counts and names were read from `model.safetensors.index.json`, not derived: 1,697 tensors
/// in total, 1,339 of them the language model, 358 the vision tower, and the language layers
/// split **25 at 45 tensors and 5 at 42**. That split is `attention_k_eq_v` — layers 5, 11, 17,
/// 23 and 29 have no `v_proj` because V reuses the key projection (D-022) — and a descriptor
/// that declared one for them would ask the repacker for a tensor the checkpoint does not have.
@Suite("Gemma 4 MLX descriptor")
struct Gemma4MLXConfigTests {

    private let config = Gemma4MLXConfig.a4b

    /// The architecture is the BF16 build's. Only the encoding changed.
    @Test("Geometry is inherited, not restated")
    func geometryMatchesTheBF16Build() {
        let bf16 = Gemma4Config.a4b
        #expect(config.layerCount == bf16.layerCount)
        #expect(config.hiddenSize == bf16.hiddenSize)
        #expect(config.vocabSize == bf16.vocabSize)
        #expect(config.expertCount == bf16.expertCount)
        #expect(config.expertsPerToken == bf16.expertsPerToken)
        #expect(config.layerTypes == bf16.layerTypes)
        #expect(config.architecture == bf16.architecture)

        for layer in 0..<config.layerCount {
            #expect(
                config.attentionGeometry(atLayer: layer)
                    == bf16.attentionGeometry(atLayer: layer))
        }
    }

    /// The names are the inverted ones, and getting them the BF16 way round asks the checkpoint
    /// for tensors that are not in it.
    @Test("Names carry the inverted prefix")
    func namesAreInverted() {
        let name = config.layerTensor("self_attn.q_proj", layer: 3)
        #expect(name == "language_model.model.layers.3.self_attn.q_proj")
        #expect(!name.hasPrefix("model.language_model"))
        #expect(config.finalNormTensor == "language_model.model.norm.weight")
        #expect(config.embeddingTensor == "language_model.model.embed_tokens.weight")
        #expect(config.residentEmbeddingTensor == config.embeddingTensor)
    }

    /// 45 tensors on a sliding layer and 42 on a full one, the difference being `v_proj`'s
    /// triple — the checkpoint's own split.
    @Test("Per-layer tensor counts match the checkpoint")
    func perLayerCountsMatchCheckpoint() {
        var perLayer: [Int: Int] = [:]
        for (name, _) in config.residentTensors {
            guard name.contains(".layers.") else { continue }
            let index = Int(name.split(separator: ".")[3])!
            perLayer[index, default: 0] += 1
        }

        // Nine of the checkpoint's 45 are the experts, which live in their own files.
        let expertTensors = 9
        for layer in 0..<config.layerCount {
            let expected = config.layerTypes[layer] == .sliding ? 45 : 42
            #expect(
                perLayer[layer]! + expertTensors == expected,
                "layer \(layer) has \(perLayer[layer]! + expertTensors) tensors, expected \(expected)")
        }

        // Stated the other way, so the reason is visible and not just the arithmetic.
        let full = config.layerTypes.enumerated().filter { $0.element == .full }.map(\.offset)
        #expect(full == [5, 11, 17, 23, 29])
        for layer in full {
            #expect(
                !config.residentTensors.contains {
                    $0.name == config.layerTensor("self_attn.v_proj.weight", layer: layer)
                },
                "a full layer must not declare v_proj")
        }
    }

    /// Every quantized tensor is a triple, and the dense MLP and router are the 8-bit ones.
    @Test("The dense MLP and router are 8-bit, everything else 4")
    func mixedPrecisionIsPerTensor() {
        let hidden = config.hiddenSize
        let mlp = config.quantized(
            config.layerTensor("mlp.gate_proj", layer: 7),
            rows: config.base.intermediateSize, cols: hidden, bits: config.denseBits)
        #expect(mlp.layout.bits == 8)
        #expect(mlp.layout.wordsPerRow == hidden / 4)

        let attention = config.quantized(
            config.layerTensor("self_attn.q_proj", layer: 7),
            rows: 4096, cols: hidden, bits: config.quantBits)
        #expect(attention.layout.bits == 4)
        #expect(attention.layout.wordsPerRow == hidden / 8)

        // 120 eight-bit tensors: four per layer, thirty layers.
        let eightBitStems = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "router.proj"]
        #expect(eightBitStems.count * config.layerCount == 120)

        // Each contributes a weight, a scales and a biases.
        for stem in eightBitStems {
            let full = config.layerTensor(stem, layer: 0)
            let names = config.residentTensors.map(\.name)
            #expect(names.contains("\(full).weight"))
            #expect(names.contains("\(full).scales"))
            #expect(names.contains("\(full).biases"))
        }
    }

    /// The number the whole exercise is for.
    @Test("The install is about a third of the BF16 build")
    func installShrinks() {
        let bf16 = Gemma4Config.a4b
        #expect(config.expertBlob.sourceBytes == 3_345_408)
        #expect(bf16.expertBlob.sourceBytes == 11_894_784)

        let pool = Double(config.expertPoolBytes) / 1e9
        #expect(pool > 12.5 && pool < 13.0, "expert pool \(pool) GB")

        let installed = Double(config.installedBytes) / 1e9
        #expect(installed > 13 && installed < 17, "install \(installed) GB")
        #expect(config.installedBytes < bf16.installedBytes / 3)

        // What actually streams per token, which is the bottleneck M-028 and M-029 name.
        let perToken = Double(config.diskBytesPerDecodedToken(cacheHitRate: 0)) / 1e9
        let bf16PerToken = Double(bf16.diskBytesPerDecodedToken(cacheHitRate: 0)) / 1e9
        #expect(perToken < bf16PerToken / 3.5, "\(perToken) GB against \(bf16PerToken)")
    }

    /// The three expert matrices, unfused, in blob order.
    @Test("Experts are three separate matrices")
    func expertsAreUnfused() {
        let tensors = config.expertTensors(layer: 2)
        #expect(tensors.count == 3)
        #expect(tensors[0].stem.hasSuffix("experts.switch_glu.gate_proj"))
        #expect(tensors[1].stem.hasSuffix("experts.switch_glu.up_proj"))
        #expect(tensors[2].stem.hasSuffix("experts.switch_glu.down_proj"))

        // gate and up share a shape; down transposes it.
        #expect(tensors[0].rows == tensors[1].rows && tensors[0].cols == tensors[1].cols)
        #expect(tensors[2].rows == tensors[0].cols && tensors[2].cols == tensors[0].rows)

        // Nine slots to a blob, against the BF16 build's two.
        #expect(config.expertBlobLayout.slots.count == 9)
    }
}
