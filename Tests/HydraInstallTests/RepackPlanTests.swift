import Foundation
import HydraCore
import HydraFormat
import Testing

@testable import HydraInstall

/// Ces tests construisent un checkpoint source **synthétique** ayant exactement la forme
/// du vrai (mêmes clés, mêmes tailles de tenseurs), puis vérifient que le plan de repack
/// le couvre intégralement. Aucun accès réseau : la validité du plan est une propriété
/// structurelle, elle doit être vérifiable hors ligne.
struct RepackPlanTests {

    /// Fabrique un index et des en-têtes safetensors conformes à la configuration donnée.
    /// Répartit les tenseurs sur plusieurs shards, comme le vrai dépôt.
    static func syntheticSource(
        config: GptOssConfig, shardCount: Int = 3
    ) throws -> (index: SafetensorsIndex, headers: [String: SafetensorsHeader], total: Int) {

        let blob = HydraLayout(config: config).expertBlob
        var declarations: [(name: String, dtype: String, shape: [Int], bytes: Int)] = []

        for layer in 0..<config.layerCount {
            for entry in HydraLayout.residentTensorNames(layer: layer) {
                let n = entry.bytes(config)
                declarations.append((entry.name, "BF16", [n / 2], n))
            }
            let E = config.expertCount
            for (suffix, slot) in [
                ("gate_up_proj_blocks", blob.gateUpBlocks),
                ("gate_up_proj_scales", blob.gateUpScales),
                ("gate_up_proj_bias", blob.gateUpBias),
                ("down_proj_blocks", blob.downBlocks),
                ("down_proj_scales", blob.downScales),
                ("down_proj_bias", blob.downBias),
            ] {
                let name = "model.layers.\(layer).mlp.experts.\(suffix)"
                let isBias = suffix.hasSuffix("bias")
                declarations.append(
                    (name, isBias ? "BF16" : "U8", [E, slot.byteCount], slot.byteCount * E))
            }
        }
        declarations.append(("model.norm.weight", "BF16", [config.hiddenSize], 2 * config.hiddenSize))
        declarations.append(
            ("lm_head.weight", "BF16", [config.vocabSize, config.hiddenSize], config.lmHeadBytes))
        declarations.append(
            ("model.embed_tokens.weight", "BF16",
             [config.vocabSize, config.hiddenSize], config.embeddingBytes))

        // Répartition round-robin sur les shards.
        var weightMap: [String: String] = [:]
        var perShard: [String: [(name: String, dtype: String, shape: [Int], bytes: Int)]] = [:]
        for (i, d) in declarations.enumerated() {
            let shard = "model-\(String(format: "%05d", i % shardCount)).safetensors"
            weightMap[d.name] = shard
            perShard[shard, default: []].append(d)
        }

        var headers: [String: SafetensorsHeader] = [:]
        for (shard, tensors) in perShard {
            var json: [String: Any] = [:]
            var cursor = 0
            for t in tensors {
                json[t.name] = [
                    "dtype": t.dtype, "shape": t.shape,
                    "data_offsets": [cursor, cursor + t.bytes],
                ]
                cursor += t.bytes
            }
            let data = try JSONSerialization.data(withJSONObject: json)
            headers[shard] = try SafetensorsHeader(headerJSON: data, headerByteCount: data.count)
        }

        let total = declarations.reduce(0) { $0 + $1.bytes }
        return (SafetensorsIndex(weightMap: weightMap, totalSize: total), headers, total)
    }

    @Test("Le plan couvre exactement le checkpoint source", arguments: [GptOssConfig.b20, .b120])
    func planCoversSourceExactly(config: GptOssConfig) throws {
        let src = try Self.syntheticSource(config: config)
        let plan = try RepackPlan(config: config, weightMap: src.index.weightMap, headers: src.headers)

        let problems = plan.validate(declaredSourceTotal: src.index.totalSize)
        #expect(
            problems.isEmpty,
            Comment(rawValue: problems.map(\.description).joined(separator: " ; ")))

        // La taille du checkpoint synthétique est celle du vrai.
        #expect(src.total == config.installedBytes)
        #expect(plan.totalSourceBytes == config.installedBytes)
    }

    @Test("Chaque tenseur source est lu une fois et une seule")
    func everyTensorReadOnce() throws {
        let config = GptOssConfig.b20
        let src = try Self.syntheticSource(config: config)
        let plan = try RepackPlan(config: config, weightMap: src.index.weightMap, headers: src.headers)

        let planned = Set(plan.operations.map(\.sourceTensor))
        #expect(planned.count == plan.operations.count, "un tenseur est planifié deux fois")
        #expect(planned == Set(src.index.weightMap.keys), "couverture des clés incomplète")
    }

    @Test("Le nombre d'opérations reste faible grâce à l'éparpillement")
    func scatterKeepsRequestCountLow() throws {
        let config = GptOssConfig.b20
        let src = try Self.syntheticSource(config: config)
        let plan = try RepackPlan(config: config, weightMap: src.index.weightMap, headers: src.headers)

        // Sans éparpillement, il faudrait une opération par expert et par sous-tenseur :
        // 24 couches x 32 experts x 6 = 4 608, plus les résidents.
        let naive = config.layerCount * config.expertCount * 6
        #expect(plan.operations.count < naive / 5)
        // 24 x 6 experts + 24 x 13 résidents + norm + head + embed
        #expect(plan.operations.count == 24 * 6 + 24 * 13 + 3)
    }

    @Test("Les opérations sont ordonnées pour une lecture séquentielle")
    func operationsAreSequential() throws {
        let config = GptOssConfig.b20
        let src = try Self.syntheticSource(config: config)
        let plan = try RepackPlan(config: config, weightMap: src.index.weightMap, headers: src.headers)

        for i in 1..<plan.operations.count {
            let a = plan.operations[i - 1], b = plan.operations[i]
            if a.sourceShard == b.sourceShard {
                #expect(a.sourceOffset <= b.sourceOffset, "lecture arrière dans \(a.sourceShard)")
            } else {
                #expect(a.sourceShard < b.sourceShard)
            }
        }
    }

    @Test("Un tenseur de taille inattendue fait échouer le plan, pas le téléchargement")
    func rejectsWrongShape() throws {
        let config = GptOssConfig.b20
        var src = try Self.syntheticSource(config: config)

        // On corrompt la taille de la tête LM dans son shard.
        let shard = src.index.weightMap["lm_head.weight"]!
        var json: [String: Any] = [:]
        for (name, e) in src.headers[shard]!.tensors {
            let bytes = name == "lm_head.weight" ? e.byteCount - 2 : e.byteCount
            json[name] = [
                "dtype": e.dtype.rawValue, "shape": e.shape,
                "data_offsets": [e.dataOffsets.start, e.dataOffsets.start + bytes],
            ]
        }
        let data = try JSONSerialization.data(withJSONObject: json)
        src.headers[shard] = try SafetensorsHeader(headerJSON: data, headerByteCount: data.count)

        #expect(throws: RepackPlan.PlanError.self) {
            _ = try RepackPlan(
                config: config, weightMap: src.index.weightMap, headers: src.headers)
        }
    }

    @Test("Un tenseur absent est détecté avant tout téléchargement")
    func rejectsMissingTensor() throws {
        let config = GptOssConfig.b20
        var src = try Self.syntheticSource(config: config)
        var map = src.index.weightMap
        map.removeValue(forKey: "model.layers.5.mlp.experts.down_proj_scales")
        src = (SafetensorsIndex(weightMap: map, totalSize: src.index.totalSize), src.headers, src.total)

        #expect(throws: RepackPlan.PlanError.self) {
            _ = try RepackPlan(
                config: config, weightMap: src.index.weightMap, headers: src.headers)
        }
    }
}

struct HydraLayoutTests {

    @Test("Le stride d'un blob est aligné sur la page et contient la charge utile")
    func blobStrideIsPageAligned() {
        for config in [GptOssConfig.b20, .b120] {
            let layout = HydraLayout(config: config)
            let blob = layout.expertBlob
            // La charge utile mise en page dépasse la somme brute du remplissage d'alignement,
            // et c'est elle, pas la somme brute, qui dimensionne un slot mémoire.
            #expect(blob.sourceBytes == config.expertBlobBytes)
            #expect(blob.payloadBytes >= blob.sourceBytes)
            #expect(blob.payloadBytes - blob.sourceBytes < ExpertBlobLayout.tensorAlignment)
            #expect(config.expertSlotBytes == blob.strideBytes)
            #expect(blob.strideBytes % ExpertBlobLayout.pageAlignment == 0)
            #expect(blob.strideBytes >= blob.payloadBytes)
            // Le remplissage doit rester marginal : moins d'une page perdue par blob.
            #expect(blob.strideBytes - blob.payloadBytes < ExpertBlobLayout.pageAlignment)
        }
    }

    @Test("Les sous-tenseurs d'un blob ne se chevauchent pas et sont alignés")
    func blobSlotsAreDisjoint() {
        let blob = HydraLayout(config: .b120).expertBlob
        let sorted = blob.slots.sorted { $0.offset < $1.offset }
        for i in 0..<sorted.count {
            #expect(sorted[i].offset % ExpertBlobLayout.tensorAlignment == 0)
            if i > 0 { #expect(sorted[i].offset >= sorted[i - 1].end) }
        }
        #expect(sorted.last!.end <= blob.payloadBytes)
    }

    @Test("Les tenseurs résidents ne se chevauchent pas et sont alignés")
    func residentPlacementsAreDisjoint() {
        let layout = HydraLayout(config: .b120)
        var previousEnd = 0
        for p in layout.resident {
            #expect(p.offset % ExpertBlobLayout.tensorAlignment == 0, Comment(rawValue: p.sourceName))
            #expect(p.offset >= previousEnd, "chevauchement sur \(p.sourceName)")
            previousEnd = p.end
        }
        #expect(layout.residentBytes >= previousEnd)
    }

    @Test("Le fichier d'une couche contient tous ses experts")
    func layerFileHoldsEveryExpert() {
        let config = GptOssConfig.b120
        let layout = HydraLayout(config: config)
        #expect(layout.expertOffset(0) == 0)
        #expect(layout.expertOffset(config.expertCount - 1) + layout.expertBlob.payloadBytes
            <= layout.expertLayerFileBytes)
    }

    @Test("Le surcoût d'alignement du pool reste sous 0,1 %")
    func alignmentOverheadIsNegligible() {
        let config = GptOssConfig.b120
        let layout = HydraLayout(config: config)
        let onDisk = config.layerCount * layout.expertLayerFileBytes
        let overhead = Double(onDisk - config.expertPoolBytes) / Double(config.expertPoolBytes)
        #expect(overhead < 0.001, "surcoût \(overhead * 100) %")
    }
}
