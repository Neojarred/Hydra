import Foundation
import HydraCore
import HydraFormat
import Testing

@testable import HydraInstall

/// These tests build a **synthetic** source checkpoint with exactly the shape of the real
/// one (same keys, same tensor sizes), then check that the repack plan covers it entirely.
/// No network access: the plan's validity is a structural property and must be checkable
/// offline.
struct RepackPlanTests {

    /// Builds an index and safetensors headers conforming to the given configuration.
    /// Spreads the tensors over several shards, as the real repository does.
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

        // Round-robin distribution over the shards.
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

    @Test("The plan covers the source checkpoint exactly", arguments: [GptOssConfig.b20, .b120])
    func planCoversSourceExactly(config: GptOssConfig) throws {
        let src = try Self.syntheticSource(config: config)
        let plan = try RepackPlan(config: config, weightMap: src.index.weightMap, headers: src.headers)

        let problems = plan.validate(declaredSourceTotal: src.index.totalSize)
        #expect(
            problems.isEmpty,
            Comment(rawValue: problems.map(\.description).joined(separator: " ; ")))

        // The synthetic checkpoint's size is that of the real one.
        #expect(src.total == config.installedBytes)
        #expect(plan.totalSourceBytes == config.installedBytes)
    }

    @Test("Every source tensor is read exactly once")
    func everyTensorReadOnce() throws {
        let config = GptOssConfig.b20
        let src = try Self.syntheticSource(config: config)
        let plan = try RepackPlan(config: config, weightMap: src.index.weightMap, headers: src.headers)

        let planned = Set(plan.operations.map(\.sourceTensor))
        #expect(planned.count == plan.operations.count, "a tensor is planned twice")
        #expect(planned == Set(src.index.weightMap.keys), "incomplete key coverage")
    }

    @Test("The operation count stays low thanks to scatter copies")
    func scatterKeepsRequestCountLow() throws {
        let config = GptOssConfig.b20
        let src = try Self.syntheticSource(config: config)
        let plan = try RepackPlan(config: config, weightMap: src.index.weightMap, headers: src.headers)

        // Without scattering, one operation per expert per sub-tensor would be needed:
        // 24 layers x 32 experts x 6 = 4,608, plus the residents.
        let naive = config.layerCount * config.expertCount * 6
        #expect(plan.operations.count < naive / 5)
        // 24 x 6 experts + 24 x 13 residents + norm + head + embed
        #expect(plan.operations.count == 24 * 6 + 24 * 13 + 3)
    }

    @Test("Operations are ordered for a sequential read")
    func operationsAreSequential() throws {
        let config = GptOssConfig.b20
        let src = try Self.syntheticSource(config: config)
        let plan = try RepackPlan(config: config, weightMap: src.index.weightMap, headers: src.headers)

        for i in 1..<plan.operations.count {
            let a = plan.operations[i - 1], b = plan.operations[i]
            if a.sourceShard == b.sourceShard {
                #expect(a.sourceOffset <= b.sourceOffset, "backward read in \(a.sourceShard)")
            } else {
                #expect(a.sourceShard < b.sourceShard)
            }
        }
    }

    @Test("A tensor of unexpected size fails the plan, not the download")
    func rejectsWrongShape() throws {
        let config = GptOssConfig.b20
        var src = try Self.syntheticSource(config: config)

        // We corrupt the LM head's size in its shard.
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

    @Test("A missing tensor is detected before any download")
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

    @Test("A blob's stride is page-aligned and holds the payload")
    func blobStrideIsPageAligned() {
        for config in [GptOssConfig.b20, .b120] {
            let layout = HydraLayout(config: config)
            let blob = layout.expertBlob
            // The laid-out payload exceeds the raw sum because of alignment padding, and it is that,
            // not the raw sum, that sizes a memory slot.
            #expect(blob.sourceBytes == config.expertBlobBytes)
            #expect(blob.payloadBytes >= blob.sourceBytes)
            #expect(blob.payloadBytes - blob.sourceBytes < ExpertBlobLayout.tensorAlignment)
            #expect(config.expertSlotBytes == blob.strideBytes)
            #expect(blob.strideBytes % ExpertBlobLayout.pageAlignment == 0)
            #expect(blob.strideBytes >= blob.payloadBytes)
            // Padding must stay marginal: less than one page lost per blob.
            #expect(blob.strideBytes - blob.payloadBytes < ExpertBlobLayout.pageAlignment)
        }
    }

    @Test("A blob's sub-tensors do not overlap and are aligned")
    func blobSlotsAreDisjoint() {
        let blob = HydraLayout(config: .b120).expertBlob
        let sorted = blob.slots.sorted { $0.offset < $1.offset }
        for i in 0..<sorted.count {
            #expect(sorted[i].offset % ExpertBlobLayout.tensorAlignment == 0)
            if i > 0 { #expect(sorted[i].offset >= sorted[i - 1].end) }
        }
        #expect(sorted.last!.end <= blob.payloadBytes)
    }

    @Test("The resident tensors do not overlap and are aligned")
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

    @Test("A layer's file contains all of its experts")
    func layerFileHoldsEveryExpert() {
        let config = GptOssConfig.b120
        let layout = HydraLayout(config: config)
        #expect(layout.expertOffset(0) == 0)
        #expect(layout.expertOffset(config.expertCount - 1) + layout.expertBlob.payloadBytes
            <= layout.expertLayerFileBytes)
    }

    @Test("The pool's alignment overhead stays under 0.1 %")
    func alignmentOverheadIsNegligible() {
        let config = GptOssConfig.b120
        let layout = HydraLayout(config: config)
        let onDisk = config.layerCount * layout.expertLayerFileBytes
        let overhead = Double(onDisk - config.expertPoolBytes) / Double(config.expertPoolBytes)
        #expect(overhead < 0.001, "overhead \(overhead * 100) %")
    }
}
