import Foundation
import HydraCore
import HydraFormat
import Testing

@testable import HydraInstall

/// Planning a Gemma 4 install from the index alone, before a byte is downloaded.
///
/// The plan is computed from a synthetic checkpoint that mirrors the real one's tensor names
/// and shapes — including the vision tower, whose whole point here is to be *excluded and
/// still accounted for*. Downloading 50 GB to test a byte-range calculation would be absurd;
/// what has to be right is structural.
@Suite("Gemma 4 repack plan")
struct GemmaRepackPlanTests {

    private let config = Gemma4Config.tiny

    /// Builds an index containing exactly what the real checkpoint has: the text model, plus a
    /// vision tower that must be skipped.
    private func syntheticCheckpoint(
        includeVision: Bool = true, includeAudio: Bool = true, includeStray: Bool = false
    ) -> (weightMap: [String: String], headers: [String: SafetensorsHeader], total: Int) {
        var declarations: [(String, String, [Int], Int)] = []

        for placement in HydraLayout(config: config).resident {
            declarations.append(
                (placement.sourceName, "BF16", [placement.byteCount / 2], placement.byteCount))
        }
        let blob = config.expertBlobLayout
        for layer in 0..<config.layerCount {
            let base = "model.language_model.layers.\(layer).experts"
            declarations.append((
                "\(base).gate_up_proj", "BF16",
                [config.expertCount, 2 * config.moeIntermediateSize, config.hiddenSize],
                blob.gateUp.byteCount * config.expertCount))
            declarations.append((
                "\(base).down_proj", "BF16",
                [config.expertCount, config.hiddenSize, config.moeIntermediateSize],
                blob.down.byteCount * config.expertCount))
        }
        if includeVision {
            for layer in 0..<3 {
                declarations.append((
                    "model.vision_tower.encoder.layers.\(layer).self_attn.q_proj.linear.weight",
                    "BF16", [64, 64], 8192))
            }
            declarations.append(
                ("model.embed_vision.embedding_projection.weight", "BF16", [64, 64], 8192))
        }
        if includeAudio {
            // Still excluded: a different encoder, a different input path, and nothing asked
            // for it. Present here so the exclusion machinery keeps being exercised.
            declarations.append((
                "model.audio_tower.encoder.layers.0.self_attn.q_proj.weight",
                "BF16", [32, 32], 2048))
            declarations.append(
                ("model.embed_audio.embedding_projection.weight", "BF16", [32, 32], 2048))
        }
        if includeStray {
            // A tensor matching no rule: the case the coverage guarantee exists to catch.
            declarations.append(("model.something_new.weight", "BF16", [16], 32))
        }

        var weightMap: [String: String] = [:]
        var json: [String: Any] = [:]
        var cursor = 0
        var total = 0
        for (name, dtype, shape, bytes) in declarations {
            weightMap[name] = "model-00001.safetensors"
            json[name] = [
                "dtype": dtype, "shape": shape,
                "data_offsets": [cursor, cursor + bytes],
            ]
            cursor += bytes
            total += bytes
        }
        let data = try! JSONSerialization.data(withJSONObject: json)
        let header = try! SafetensorsHeader(headerJSON: data, headerByteCount: data.count)
        return (weightMap, ["model-00001.safetensors": header], total)
    }

    // MARK: - Construction

    @Test("The plan covers the text model and validates")
    func planIsSound() throws {
        let c = syntheticCheckpoint()
        let plan = try GemmaRepackPlan(
            config: config, weightMap: c.weightMap, headers: c.headers)

        let problems = plan.validate(
            weightMap: c.weightMap, declaredSourceTotal: c.total)
        #expect(problems.isEmpty, "\(problems.map(\.description))")
    }

    /// Experts are fused per layer, so one operation scatters all of them at the blob stride.
    /// This is the mechanism `ScatterCopy` exists for, reused unchanged.
    @Test("Each expert tensor becomes one scattering operation")
    func expertsScatterInOneOperation() throws {
        let c = syntheticCheckpoint()
        let plan = try GemmaRepackPlan(
            config: config, weightMap: c.weightMap, headers: c.headers)

        let expertOps = plan.operations.filter { $0.sourceTensor.contains(".experts.") }
        #expect(expertOps.count == config.layerCount * 2)
        for op in expertOps {
            #expect(op.chunkCount == config.expertCount)
            #expect(op.destinationStride == config.expertSlotBytes)
        }
    }

    /// The embedding is tied to the output head, so it is a resident tensor rather than its own
    /// file. A plan that emitted an `embed.bin` operation would create a file nothing maps.
    @Test("There is no embedding destination")
    func noEmbeddingFile() throws {
        let c = syntheticCheckpoint()
        let plan = try GemmaRepackPlan(
            config: config, weightMap: c.weightMap, headers: c.headers)

        #expect(plan.destinationSizes[.embedding] == nil)
        #expect(!plan.operations.contains { $0.destination == .embedding })
        let embedding = plan.operations.first {
            $0.sourceTensor.hasSuffix("embed_tokens.weight")
        }
        #expect(embedding?.destination == .resident)
    }

    // MARK: - The restated coverage guarantee

    /// The vision tower is **installed**, and that is a deliberate reversal.
    ///
    /// Nothing executes these weights yet. They are installed anyway because they are what a
    /// 50 GB download buys: leaving them out would mean fetching the whole checkpoint a second
    /// time to add image support later. They go to their own file rather than into
    /// `resident.bin`, so a text-only conversation never pays for them.
    @Test("The vision tower is installed into its own file")
    func visionIsInstalled() throws {
        let c = syntheticCheckpoint()
        let plan = try GemmaRepackPlan(
            config: config, weightMap: c.weightMap, headers: c.headers)

        #expect(plan.visionPlacements.count == 4)
        #expect(plan.destinationSizes[.vision] == plan.visionFileBytes)
        #expect(plan.visionFileBytes > 0)
        #expect(plan.operations.contains { $0.destination == .vision })

        // Not in the resident file, which is read on every token and prefaulted at load.
        #expect(!plan.operations.contains {
            $0.destination == .resident && $0.sourceTensor.contains("vision")
        })

        // Placements do not overlap and are ordered, so the file can be mapped and sliced.
        let sorted = plan.visionPlacements.sorted { $0.offset < $1.offset }
        for (a, b) in zip(sorted, sorted.dropFirst()) {
            #expect(a.offset + a.byteCount <= b.offset, "\(a.name) overlaps \(b.name)")
        }
        #expect(sorted.last.map { $0.offset + $0.byteCount } ?? 0 <= plan.visionFileBytes)

        // Shape and dtype are carried from the checkpoint: without them the file is an
        // undifferentiated block no later encoder could read.
        #expect(plan.visionPlacements.allSatisfy { !$0.shape.isEmpty })
        #expect(plan.visionPlacements.allSatisfy { $0.dtype == .bf16 })
    }

    @Test("Audio is still excluded, and recorded rather than dropped")
    func audioIsExcludedAndRecorded() throws {
        let c = syntheticCheckpoint()
        let plan = try GemmaRepackPlan(
            config: config, weightMap: c.weightMap, headers: c.headers)

        #expect(plan.excluded.count == 2)
        #expect(plan.excluded.allSatisfy { $0.reason.contains("not executed") })
        #expect(plan.excluded.allSatisfy { $0.name.contains("audio") })
        #expect(plan.totalExcludedBytes == 2 * 2048)

        // Planned plus excluded is the whole checkpoint: nothing went missing.
        #expect(plan.totalSourceBytes + plan.totalExcludedBytes == c.total)
    }

    /// A checkpoint with no tower at all must not produce an empty file to map.
    @Test("No tower means no vision file")
    func absentTowerMeansNoFile() throws {
        let c = syntheticCheckpoint(includeVision: false)
        let plan = try GemmaRepackPlan(
            config: config, weightMap: c.weightMap, headers: c.headers)

        #expect(plan.visionPlacements.isEmpty)
        #expect(plan.destinationSizes[.vision] == nil)
        #expect(plan.visionTensors.isEmpty)
    }

    /// The property that matters. A tensor matching neither the plan nor an exclusion rule must
    /// fail validation — otherwise a checkpoint that grew a component would install silently
    /// incomplete, which is exactly what the GPT-OSS coverage check prevents.
    @Test("An unrecognized tensor fails validation rather than disappearing")
    func unaccountedTensorIsReported() throws {
        let c = syntheticCheckpoint(includeStray: true)
        let plan = try GemmaRepackPlan(
            config: config, weightMap: c.weightMap, headers: c.headers)

        let problems = plan.validate(weightMap: c.weightMap, declaredSourceTotal: c.total)
        #expect(problems.contains { $0.description.contains("neither planned nor excluded") })
        #expect(problems.contains { $0.description.contains("something_new") })
    }

    @Test("A declared total that does not add up is reported")
    func wrongTotalIsReported() throws {
        let c = syntheticCheckpoint()
        let plan = try GemmaRepackPlan(
            config: config, weightMap: c.weightMap, headers: c.headers)

        let problems = plan.validate(
            weightMap: c.weightMap, declaredSourceTotal: c.total + 4096)
        #expect(problems.contains { $0.description.contains("incomplete coverage") })
    }

    @Test("A missing tensor is refused at planning time")
    func missingTensorIsRefused() {
        var c = syntheticCheckpoint()
        c.weightMap["model.language_model.layers.0.experts.gate_up_proj"] = nil
        #expect(throws: RepackPlan.PlanError.self) {
            try GemmaRepackPlan(config: config, weightMap: c.weightMap, headers: c.headers)
        }
    }

    /// Both classifications match by prefix so that a checkpoint growing a layer stays
    /// covered, where an explicit list would silently start failing the coverage check.
    @Test("Classification matches by prefix, not by an explicit list")
    func classificationIsByPrefix() {
        #expect(GemmaRepackPlan.isVisionTensor(
            "model.vision_tower.encoder.layers.99.brand.new.weight"))
        #expect(GemmaRepackPlan.isVisionTensor("model.embed_vision.anything"))
        #expect(!GemmaRepackPlan.isVisionTensor(
            "model.language_model.layers.0.mlp.gate_proj.weight"))

        #expect(GemmaRepackPlan.exclusionReason(for: "model.audio_tower.anything") != nil)
        #expect(GemmaRepackPlan.exclusionReason(for: "model.embed_audio.anything") != nil)
        #expect(GemmaRepackPlan.exclusionReason(
            for: "model.language_model.layers.0.mlp.gate_proj.weight") == nil)

        // The two classifications must stay disjoint: a tensor both installed and excluded
        // would be counted twice and break the coverage sum.
        #expect(GemmaRepackPlan.exclusionReason(for: "model.vision_tower.x") == nil)
        #expect(!GemmaRepackPlan.isVisionTensor("model.audio_tower.x"))
    }
}
