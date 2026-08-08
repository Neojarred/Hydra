import Foundation
import HydraCore
import Testing

@testable import HydraFormat

/// `HydraLayout` placing a second architecture.
///
/// The point of these tests is not that Gemma has a layout, but that **the placement rules did
/// not change to accommodate it**. The alignment invariants, the offsets the shaders rely on
/// and the page-aligned file size are the same code path GPT-OSS uses; only the list of tensors
/// differs. If a future architecture needs the rules bent, that should show up here as a
/// failure rather than as a special case somewhere in the format.
@Suite("Gemma 4 resident layout")
struct GemmaLayoutTests {

    private let layout = HydraLayout(config: Gemma4Config.a4b)
    private let config = Gemma4Config.a4b

    @Test("Every tensor is placed once, in order, without overlap")
    func placementIsSound() {
        #expect(layout.resident.count == config.residentTensors.count)

        var previousEnd = 0
        for placement in layout.resident {
            #expect(placement.offset >= previousEnd, "\(placement.sourceName) overlaps")
            #expect(
                placement.offset % HydraLayout.tensorAlignment == 0,
                "\(placement.sourceName) is misaligned — the shaders' wide loads depend on this")
            previousEnd = placement.end
        }
        #expect(layout.residentBytes >= previousEnd)
        #expect(layout.residentBytes % HydraLayout.pageAlignment == 0)
    }

    @Test("Names are unique, so placement(of:) is unambiguous")
    func namesAreUnique() {
        let names = layout.resident.map(\.sourceName)
        #expect(Set(names).count == names.count)
        #expect(layout.placement(of: "model.language_model.embed_tokens.weight") != nil)
    }

    /// The embedding is tied to the output head, so it sits in `resident.bin` and there is no
    /// separate file. Reporting a size here would make the repacker plan a file that must not
    /// exist.
    @Test("There is no separate embedding file")
    func noEmbeddingFile() {
        #expect(layout.embeddingBytes == 0)
        #expect(layout.placement(of: "model.language_model.embed_tokens.weight") != nil)

        // GPT-OSS is the opposite case, and must stay that way.
        let gptOss = HydraLayout(config: GptOssConfig.b20)
        #expect(gptOss.embeddingBytes == GptOssConfig.b20.embeddingBytes)
        #expect(gptOss.placement(of: "model.embed_tokens.weight") == nil)
    }

    /// A layer file holds every expert of one layer, back to back at the blob stride. This is
    /// what `expertOffset` indexes and what the slot cache `pread`s.
    @Test("The expert layer file is sized from the Gemma blob")
    func expertFileSize() {
        #expect(layout.expertLayerFileBytes == 128 * config.expertSlotBytes)
        #expect(layout.expertOffset(0) == 0)
        #expect(layout.expertOffset(127) == 127 * config.expertSlotBytes)
        #expect(layout.expertBlob.strideBytes % HydraLayout.pageAlignment == 0)
    }

    /// Sliding and full layers place a different number of tensors, and different sizes for the
    /// same names. A layout that assumed one geometry would silently overlap them.
    @Test("The two layer geometries produce different placements")
    func geometriesDiffer() {
        func size(_ suffix: String, layer: Int) -> Int? {
            layout.resident.first {
                $0.sourceName == "model.language_model.layers.\(layer).\(suffix)"
            }?.byteCount
        }
        // Layer 0 is sliding, layer 5 is full.
        #expect(size("self_attn.q_proj.weight", layer: 0) == 4096 * 2816 * 2)
        #expect(size("self_attn.q_proj.weight", layer: 5) == 8192 * 2816 * 2)
        #expect(size("self_attn.v_proj.weight", layer: 0) != nil)
        #expect(size("self_attn.v_proj.weight", layer: 5) == nil)
    }

    @Test("The layout agrees with the config on the total")
    func totalsAgree() {
        #expect(layout.residentBytes >= config.residentBytes)
        // Padding is alignment only, so it stays negligible against 4.46 GiB.
        let overhead = Double(layout.residentBytes - config.residentBytes)
        #expect(overhead / Double(config.residentBytes) < 0.001)
    }
}
