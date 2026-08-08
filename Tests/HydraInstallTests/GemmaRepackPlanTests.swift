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
        includeVision: Bool = true, includeStray: Bool = false
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

    @Test("Vision tensors are excluded, and recorded rather than dropped")
    func towersAreExcludedAndRecorded() throws {
        let c = syntheticCheckpoint()
        let plan = try GemmaRepackPlan(
            config: config, weightMap: c.weightMap, headers: c.headers)

        #expect(plan.excluded.count == 4)
        #expect(plan.excluded.allSatisfy { $0.reason.contains("not executed") })
        #expect(plan.totalExcludedBytes == 4 * 8192)
        #expect(!plan.operations.contains { $0.sourceTensor.contains("vision") })

        // Planned plus excluded is the whole checkpoint: nothing went missing.
        #expect(plan.totalSourceBytes + plan.totalExcludedBytes == c.total)
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

    /// Exclusion is by prefix so that a checkpoint growing a new vision layer stays covered.
    @Test("Exclusion matches by prefix, not by an explicit list")
    func exclusionIsByPrefix() {
        #expect(GemmaRepackPlan.exclusionReason(
            for: "model.vision_tower.encoder.layers.99.brand.new.weight") != nil)
        #expect(GemmaRepackPlan.exclusionReason(
            for: "model.audio_tower.anything") != nil)
        #expect(GemmaRepackPlan.exclusionReason(
            for: "model.language_model.layers.0.mlp.gate_proj.weight") == nil)
    }
}
