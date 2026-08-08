import Foundation
import Testing

@testable import HydraCore

/// The hub contract from D-023, exercised through **both** architectures at once.
///
/// Every test here runs over `allModels`, so adding a third architecture means adding one line
/// and immediately learning whether the seam holds. That is the point of the contract: not that
/// each model works, but that nothing downstream has to ask which one it is.
@Suite("Model descriptor contract")
struct ModelDescriptorTests {

    static let allModels: [any ModelDescriptor] = [
        GptOssConfig.b20, GptOssConfig.b120, Gemma4Config.a4b,
        GptOssConfig.tiny, Gemma4Config.tiny,
    ]

    @Test("Every model reports a coherent structure", arguments: allModels.indices)
    func structureIsCoherent(index: Int) {
        let model = Self.allModels[index]

        #expect(model.layerCount > 0)
        #expect(model.layerTypes.count == model.layerCount)
        #expect(model.expertsPerToken <= model.expertCount)
        #expect(model.fullAttentionLayerCount + model.slidingAttentionLayerCount
            == model.layerCount)

        for layer in 0..<model.layerCount {
            let g = model.attentionGeometry(atLayer: layer)
            #expect(g.attentionHeadCount % g.keyValueHeadCount == 0,
                "\(model.name) layer \(layer): GQA group is not whole")
            #expect(g.queryDim == g.attentionHeadCount * g.headDim)
        }
    }

    /// The resident list is what the layout places, so a duplicate name would make one tensor
    /// silently overwrite another, and an empty one would produce a model with no weights.
    @Test("Resident tensors are named uniquely and sized", arguments: allModels.indices)
    func residentListIsSound(index: Int) {
        let model = Self.allModels[index]
        let tensors = model.residentTensors

        #expect(!tensors.isEmpty)
        #expect(Set(tensors.map(\.name)).count == tensors.count, "\(model.name) has a duplicate")
        #expect(tensors.allSatisfy { $0.byteCount > 0 })
        #expect(model.residentBytes == tensors.reduce(0) { $0 + $1.byteCount })
    }

    /// Three numbers, and the invariant between them. Whatever is inside a blob, a slot must
    /// hold its payload and start on a page boundary.
    @Test("The expert blob contract holds", arguments: allModels.indices)
    func expertBlobIsSound(index: Int) {
        let model = Self.allModels[index]
        let blob = model.expertBlob

        #expect(blob.payloadBytes >= blob.sourceBytes)
        #expect(blob.strideBytes >= blob.payloadBytes)
        #expect(blob.strideBytes % ExpertBlobLayout.pageAlignment == 0)
        #expect(model.expertSlotBytes == blob.strideBytes)
        #expect(model.expertPoolBytes == model.layerCount * model.expertCount * blob.sourceBytes)
    }

    /// The two architectures answer this differently on purpose, and both answers must survive
    /// the abstraction: GPT-OSS keeps its embedding out of the working set, Gemma ties it to
    /// the output head and must hold it resident.
    @Test("Where the embedding lives is a property, not an assumption")
    func embeddingPlacementIsExplicit() {
        #expect(GptOssConfig.b20.embeddingFileBytes > 0)
        #expect(!GptOssConfig.b20.residentTensors.contains { $0.name.contains("embed_tokens") })

        #expect(Gemma4Config.a4b.embeddingFileBytes == 0)
        #expect(Gemma4Config.a4b.residentTensors.contains { $0.name.contains("embed_tokens") })
    }

    @Test("installedBytes accounts for every destination", arguments: allModels.indices)
    func installedBytesIsComplete(index: Int) {
        let model = Self.allModels[index]
        #expect(model.installedBytes
            == model.expertPoolBytes + model.residentBytes + model.embeddingFileBytes)
        // And it still matches what each concrete config computed before the abstraction.
        if let gptOss = model as? GptOssConfig {
            #expect(model.installedBytes == gptOss.installedBytes)
        }
        if let gemma = model as? Gemma4Config {
            #expect(model.installedBytes == gemma.installedBytes)
        }
    }

    /// The dispatch point from D-023: one enum, read once, never re-derived downstream.
    @Test("Architecture is reported, not inferred")
    func architectureIsExplicit() {
        #expect(GptOssConfig.b20.architecture == .gptOss)
        #expect(Gemma4Config.a4b.architecture == .gemma4)
        #expect(ModelArchitecture.allCases.count == 2)
    }

    /// Uniform in GPT-OSS, deliberately not in Gemma. A consumer that read the geometry once
    /// and reused it would be right for one model and silently wrong for the other.
    @Test("Geometry is uniform for GPT-OSS and varies for Gemma")
    func geometryVariesWhereItShould() {
        let gptOss = GptOssConfig.b20
        let first = gptOss.attentionGeometry(atLayer: 0)
        #expect((0..<gptOss.layerCount).allSatisfy {
            gptOss.attentionGeometry(atLayer: $0) == first
        })

        let gemma = Gemma4Config.a4b
        #expect(gemma.attentionGeometry(atLayer: 0) != gemma.attentionGeometry(atLayer: 5))
    }
}
