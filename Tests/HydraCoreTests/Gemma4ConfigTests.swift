import Foundation
import Testing

@testable import HydraCore

/// Gemma 4 26B-A4B's structure, asserted against the published `config.json` and weight index.
///
/// Every number here decides either what is downloaded or what stays in memory, and a wrong one
/// fails silently — the model would load and produce degraded text. The figures D-021 was
/// argued on are asserted rather than trusted, so that if any of them moves, the decision fails
/// a test instead of quietly becoming wrong.
@Suite("Gemma 4 config")
struct Gemma4ConfigTests {

    private let config = Gemma4Config.a4b

    // MARK: - Layer pattern

    /// Five sliding to one full, repeating. Parity — GPT-OSS's rule — would give a completely
    /// different set of layers, and attention would silently reach the wrong distance.
    @Test("The 5:1 pattern matches layer_types")
    func layerPattern() {
        let types = config.layerTypes
        #expect(types.count == 30)
        // From config.json: indices 5, 11, 17, 23, 29 are full_attention.
        let full = types.indices.filter { types[$0] == .full }
        #expect(full == [5, 11, 17, 23, 29])
        #expect(types.count { $0 == .sliding } == 25)
    }

    @Test("Full layers carry a different geometry from sliding ones")
    func twoGeometries() {
        let sliding = config.attentionGeometry(atLayer: 0)
        let full = config.attentionGeometry(atLayer: 5)

        #expect(sliding.headDim == 256 && sliding.keyValueHeadCount == 8)
        #expect(full.headDim == 512 && full.keyValueHeadCount == 2)
        // The consequence that breaks a uniform layout: q_proj is not the same shape.
        #expect(sliding.queryDim == 4096)
        #expect(full.queryDim == 8192)
        // And GQA groups differ: 2 on sliding layers, 8 on full ones.
        #expect(sliding.groupedQueryFactor == 2)
        #expect(full.groupedQueryFactor == 8)
    }

    /// `attention_k_eq_v` applies to full layers only, so those have no `v_proj` tensor.
    /// Expecting one would make the repack plan ask for a tensor the checkpoint does not have.
    @Test("Only sliding layers project their own values")
    func valueProjectionPresence() {
        #expect(config.hasValueProjection(atLayer: 0))
        #expect(!config.hasValueProjection(atLayer: 5))

        let slidingNames = config.residentTensors(atLayer: 0).map(\.name)
        let fullNames = config.residentTensors(atLayer: 5).map(\.name)
        #expect(slidingNames.contains { $0.hasSuffix("v_proj.weight") })
        #expect(!fullNames.contains { $0.hasSuffix("v_proj.weight") })

        // v_norm has with_scale=False, so it has no tensor on any layer (D-022).
        #expect(!slidingNames.contains { $0.contains("v_norm") })
        #expect(!fullNames.contains { $0.contains("v_norm") })
    }

    // MARK: - RoPE

    /// Partial rotation expressed as a pair count: a quarter of 512 is 128 values, i.e. 64
    /// pairs rotate and the remaining 192 pass through. That is what makes it free in the
    /// kernel.
    @Test("Full layers rotate a quarter of the head dimension")
    func partialRotation() {
        #expect(config.rotatingPairs(atLayer: 0) == 128)  // sliding: all 256/2 pairs
        #expect(config.rotatingPairs(atLayer: 5) == 64)  // full: 0.25 × 512 / 2
        #expect(config.ropeTheta(atLayer: 0) == 10_000)
        #expect(config.ropeTheta(atLayer: 5) == 1_000_000)
    }

    // MARK: - Sizes

    @Test("The embedding is the output head and must stay resident")
    func tiedEmbedding() {
        #expect(config.embeddingBytes == 262_144 * 2816 * 2)
        let names = config.residentTensors.map(\.name)
        #expect(names.contains("model.language_model.embed_tokens.weight"))
        // tie_word_embeddings is true: there is no separate head to look for.
        #expect(!names.contains { $0.contains("lm_head") })
    }

    /// D-021 accepted a resident floor of roughly 4.5 GB. This is where that number comes from.
    @Test("The resident floor is what D-021 accepted")
    func residentFloor() {
        let gib = Double(config.residentBytes) / 1_073_741_824
        #expect(gib > 4.2 && gib < 4.8, "resident floor is \(gib) GiB")
        // Nearly twice GPT-OSS 20B's, driven by the tied embedding and the dense MLP.
        #expect(config.residentBytes > GptOssConfig.b20.residentBytes)
    }

    @Test("The install size is what D-021 planned for")
    func installSize() {
        let gb = Double(config.installedBytes) / 1_000_000_000
        #expect(gb > 48 && gb < 52, "install is \(gb) GB")
    }

    /// The router carries two learned scales beyond its projection, and the layer a scalar.
    /// Missing any of them changes the expert weighting with no error raised.
    @Test("The router's scales and the layer scalar are accounted for")
    func routerExtras() {
        let names = config.residentTensors(atLayer: 0).map(\.name)
        #expect(names.contains { $0.hasSuffix("router.proj.weight") })
        #expect(names.contains { $0.hasSuffix("router.scale") })
        #expect(names.contains { $0.hasSuffix("router.per_expert_scale") })
        #expect(names.contains { $0.hasSuffix("layer_scalar") })
    }

    /// Seven normalizations per layer, not the two GPT-OSS has. The MoE branch reads the
    /// residual, so it carries its own pre- and post-norms alongside the dense branch's.
    @Test("Seven norms per layer")
    func normCount() {
        let norms = config.residentTensors(atLayer: 0).filter {
            $0.name.contains("layernorm")
        }
        #expect(norms.count == 7, "found \(norms.map(\.name))")
    }

    @Test("Names carry the language-model prefix, never a tower's")
    func namesAreTextOnly() {
        for tensor in config.residentTensors {
            #expect(tensor.name.hasPrefix("model.language_model."))
            #expect(!tensor.name.contains("vision_tower"))
            #expect(!tensor.name.contains("audio"))
        }
    }

    @Test("The tiny config keeps the structural invariants")
    func tinyIsRepresentative() {
        let tiny = Gemma4Config.tiny
        #expect(tiny.layerTypes.contains(.full) && tiny.layerTypes.contains(.sliding))
        #expect(tiny.attentionGeometry(atLayer: 0).headDim
            != tiny.attentionGeometry(atLayer: 5).headDim)
        #expect(!tiny.hasValueProjection(atLayer: 5))
        #expect(tiny.intermediateSize != tiny.moeIntermediateSize)
    }
}
