import Foundation
import HydraCore
import HydraFormat
import Testing

@testable import HydraInstall

/// The plan that turns a downloaded MLX Qwen3.5 checkpoint into `.hydra`.
///
/// Built on a synthetic index rather than the real 20 GB one, for the same reason the Gemma
/// tests are: the numbers are the config's own, and what has to be right is structural. The
/// index below has exactly the two namespaces the published repository has, `language_model.`
/// and `vision_tower.`, verified against its `model.safetensors.index.json`.
@Suite("Qwen 3.5 MoE repack plan")
struct QwenRepackPlanTests {

    private let config = Qwen35MoeConfig.tiny

    private func syntheticCheckpoint(
        includeVision: Bool = true, includeStray: Bool = false, gemmaStyleVision: Bool = false
    ) -> (weightMap: [String: String], headers: [String: SafetensorsHeader], total: Int) {
        var declarations: [(String, String, [Int], Int)] = []

        for placement in HydraLayout(model: config).resident {
            declarations.append(
                (placement.sourceName, "BF16", [placement.byteCount / 2], placement.byteCount))
        }

        let blob = config.expertBlobLayout
        for layer in 0..<config.layerCount {
            let stems = config.expertTensors(layer: layer)
            let parts: [(String, Int)] = [
                ("weight", blob.gateLayout.weightBytes),
                ("scales", blob.gateLayout.scaleBytes),
                ("biases", blob.gateLayout.biasBytes),
            ]
            let downParts: [(String, Int)] = [
                ("weight", blob.downLayout.weightBytes),
                ("scales", blob.downLayout.scaleBytes),
                ("biases", blob.downLayout.biasBytes),
            ]
            for (index, stem) in stems.enumerated() {
                for (part, bytes) in (index == 2 ? downParts : parts) {
                    declarations.append((
                        "\(stem.stem).\(part)", "BF16",
                        [config.expertCount, stem.rows, stem.cols],
                        bytes * config.expertCount))
                }
            }
        }

        if includeVision {
            // No `model.` in front, and no `embed_vision.` beside it: this checkpoint's tower
            // sits at the top level under its own name.
            let prefix = gemmaStyleVision ? "model.vision_tower." : "vision_tower."
            for layer in 0..<3 {
                declarations.append((
                    "\(prefix)encoder.layers.\(layer).self_attn.q_proj.weight",
                    "BF16", [64, 64], 8192))
            }
        }
        if includeStray {
            declarations.append(("language_model.something_new.weight", "BF16", [16], 32))
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

    private func makePlan(
        _ c: (weightMap: [String: String], headers: [String: SafetensorsHeader], total: Int)
    ) throws -> QwenRepackPlan {
        try QwenRepackPlan(config: config, weightMap: c.weightMap, headers: c.headers)
    }

    // MARK: - Coverage

    @Test("The plan covers the checkpoint and validates")
    func planIsSound() throws {
        let c = syntheticCheckpoint()
        let plan = try makePlan(c)
        let problems = plan.validate(weightMap: c.weightMap, declaredSourceTotal: c.total)
        #expect(problems.isEmpty, "\(problems.map(\.description))")
    }

    /// A tensor matching no rule is reported rather than dropped.
    ///
    /// This is the guarantee that makes the plan trustworthy against a revision of the
    /// checkpoint: a new namespace fails validation loudly instead of installing a model that
    /// is quietly missing weights.
    @Test("A tensor matching no rule is refused")
    func strayTensorIsCaught() throws {
        let c = syntheticCheckpoint(includeStray: true)
        let plan = try makePlan(c)
        let problems = plan.validate(weightMap: c.weightMap, declaredSourceTotal: c.total)
        #expect(problems.contains { $0.description.contains("something_new") })
    }

    /// Reusing Gemma's `model.vision_tower.` prefix would not raise here, so it is asserted.
    ///
    /// The names differ between the two checkpoints, and a prefix that matches nothing does not
    /// throw: it simply installs no tower. Validation is what turns that into a failure, and
    /// this test is what proves validation still can.
    @Test("A tower under an unexpected prefix fails rather than vanishing")
    func wrongVisionPrefixIsCaught() throws {
        let c = syntheticCheckpoint(gemmaStyleVision: true)
        let plan = try makePlan(c)
        #expect(plan.visionPlacements.isEmpty, "nothing matched, as expected of a wrong prefix")
        let problems = plan.validate(weightMap: c.weightMap, declaredSourceTotal: c.total)
        #expect(
            problems.contains { $0.description.contains("neither planned nor excluded") },
            "an unmatched tower must be reported, not silently skipped")
    }

    // MARK: - What differs from Gemma

    /// Nine operations a layer, not two: this checkpoint is quantized, so each of the three
    /// matrices arrives as a weight, its scales and its biases.
    @Test("Every expert sub-tensor becomes one scattering operation")
    func expertsScatterInOneOperation() throws {
        let plan = try makePlan(syntheticCheckpoint())
        let expertOps = plan.operations.filter { $0.sourceTensor.contains(".switch_mlp.") }
        #expect(expertOps.count == config.layerCount * 9)
        for op in expertOps {
            #expect(op.chunkCount == config.expertCount, "\(op.sourceTensor)")
            #expect(op.destinationStride == config.expertBlobLayout.strideBytes)
        }
    }

    /// The gate and down layouts have the same byte counts, and no test can tell them apart.
    ///
    /// Substituting one for the other in the plan changes nothing measurable, which a
    /// falsification pass found by surviving. It is a property rather than a gap: the weights
    /// are `rows · cols / 2` either way, and the scales are `rows · (cols / group) · 2`, which
    /// is symmetric in the two dimensions whenever both are multiples of the group size, as MLX
    /// requires. Asserted so that a future build which breaks the symmetry tells the reader the
    /// plan's choice of slot has started to matter.
    @Test("The gate and down layouts are indistinguishable by size")
    func gateAndDownAreSameSize() {
        for c in [config, Qwen35MoeConfig.a3bQ4, .a3bQ8] {
            let blob = c.expertBlobLayout
            #expect(blob.gateLayout.weightBytes == blob.downLayout.weightBytes, "\(c.name)")
            #expect(blob.gateLayout.scaleBytes == blob.downLayout.scaleBytes, "\(c.name)")
            #expect(blob.gateLayout.biasBytes == blob.downLayout.biasBytes, "\(c.name)")
        }
    }

    /// The embeddings are not tied, so both tables are installed.
    ///
    /// Gemma reads one table twice and its plan has a single operation. Carrying that assumption
    /// over would leave `lm_head` out of the plan entirely, and validation would catch it only
    /// because of the coverage rule; asserting both are resident says what is meant.
    @Test("Both the embedding table and the output head are resident")
    func embeddingAndHeadAreSeparate() throws {
        let plan = try makePlan(syntheticCheckpoint())
        let embedding = plan.operations.filter {
            $0.sourceTensor.hasPrefix("\(Qwen35MoeConfig.prefix).embed_tokens.")
        }
        let head = plan.operations.filter {
            $0.sourceTensor.hasPrefix("language_model.lm_head.")
        }
        #expect(embedding.count == 3, "weight, scales and biases")
        #expect(head.count == 3)
        #expect(embedding.allSatisfy { $0.destination == .resident })
        #expect(head.allSatisfy { $0.destination == .resident })

        // Two tables, and not the same bytes written twice.
        let offsets = Set((embedding + head).map(\.destinationOffset))
        #expect(offsets.count == 6)
    }

    /// A recurrent layer has no q, k or v projection, and the plan must not demand them.
    @Test("Linear layers contribute a different inventory from full ones")
    func linearLayersHaveNoProjections() throws {
        let plan = try makePlan(syntheticCheckpoint())
        for layer in 0..<config.layerCount {
            let stem = config.layerTensor("", layer: layer)
            let mine = plan.operations.filter { $0.sourceTensor.hasPrefix(stem) }
            let hasSelfAttention = mine.contains { $0.sourceTensor.contains(".self_attn.") }
            let hasLinearAttention = mine.contains { $0.sourceTensor.contains(".linear_attn.") }
            #expect(
                hasSelfAttention == (config.attentionPattern(atLayer: layer) == .full),
                "layer \(layer)")
            #expect(hasLinearAttention == !hasSelfAttention)
        }
        // The ratio the config declares, reached through the plan rather than asserted twice.
        let full = (0..<config.layerCount).count { config.attentionPattern(atLayer: $0) == .full }
        #expect(full == config.layerCount / config.fullAttentionInterval)
    }

    /// The tower goes to its own file, so a text-only conversation never maps those pages.
    @Test("The vision tower is installed into its own file")
    func visionIsInstalled() throws {
        let plan = try makePlan(syntheticCheckpoint())
        #expect(plan.visionPlacements.count == 3)
        #expect(plan.destinationSizes[.vision] != nil)
        #expect(plan.operations.contains { $0.destination == .vision })
        #expect(
            plan.visionPlacements.allSatisfy { $0.name.hasPrefix("vision_tower.") })
    }

    /// A checkpoint without a tower produces no file at all, rather than an empty one.
    @Test("No tower means no vision file")
    func noVisionMeansNoFile() throws {
        let c = syntheticCheckpoint(includeVision: false)
        let plan = try makePlan(c)
        #expect(plan.visionPlacements.isEmpty)
        #expect(plan.destinationSizes[.vision] == nil)
        #expect(plan.validate(weightMap: c.weightMap, declaredSourceTotal: c.total).isEmpty)
    }

    // MARK: - Failures

    @Test("A missing tensor is refused rather than skipped")
    func missingTensorThrows() throws {
        var c = syntheticCheckpoint()
        c.weightMap[config.finalNormTensor] = nil
        #expect(throws: RepackPlan.PlanError.self) { try makePlan(c) }
    }

    /// A tensor whose byte count disagrees with the config is a config that does not describe
    /// this checkpoint, which must fail before 20 GB is copied into the wrong shape.
    @Test("A tensor of the wrong size is refused")
    func wrongSizeThrows() throws {
        var c = syntheticCheckpoint()
        let name = config.expertTensors(layer: 0)[0].stem + ".weight"
        var json: [String: Any] = [:]
        for (tensor, entry) in c.headers["model-00001.safetensors"]!.tensors {
            let bytes = tensor == name ? entry.byteCount / 2 : entry.byteCount
            json[tensor] = [
                "dtype": "BF16", "shape": entry.shape,
                "data_offsets": [entry.dataOffsets.start, entry.dataOffsets.start + bytes],
            ]
        }
        let data = try JSONSerialization.data(withJSONObject: json)
        c.headers["model-00001.safetensors"] = try SafetensorsHeader(
            headerJSON: data, headerByteCount: data.count)
        #expect(throws: RepackPlan.PlanError.self) { try makePlan(c) }
    }

    /// Both published builds go through the same plan; only the widths change.
    @Test("The 8-bit build plans the same operations as the 4-bit one, with more bytes")
    func bothBuildsPlanAlike() throws {
        let eightBit = Qwen35MoeConfig(
            name: "Qwen tiny (8-bit)", layerCount: 8, hiddenSize: 64, vocabSize: 256,
            maxPositionEmbeddings: 512, attentionHeadCount: 4, keyValueHeadCount: 2, headDim: 16,
            linearKeyHeads: 2, linearValueHeads: 4, linearKeyHeadDim: 16, linearValueHeadDim: 16,
            expertCount: 8, expertsPerToken: 2, moeIntermediateSize: 16,
            sharedExpertIntermediateSize: 16, quantBits: 8, groupSize: 16)
        let wide = syntheticCheckpointFor(eightBit)
        let plan = try QwenRepackPlan(
            config: eightBit, weightMap: wide.weightMap, headers: wide.headers)
        let narrow = try makePlan(syntheticCheckpoint())

        #expect(plan.operations.count == narrow.operations.count, "the same tensors")
        #expect(plan.validate(weightMap: wide.weightMap, declaredSourceTotal: wide.total).isEmpty)
        #expect(plan.totalSourceBytes > narrow.totalSourceBytes, "and more of them")
    }

    /// The synthetic index for an arbitrary config, used by the 8-bit case above.
    private func syntheticCheckpointFor(
        _ config: Qwen35MoeConfig
    ) -> (weightMap: [String: String], headers: [String: SafetensorsHeader], total: Int) {
        var declarations: [(String, [Int], Int)] = []
        for placement in HydraLayout(model: config).resident {
            declarations.append(
                (placement.sourceName, [placement.byteCount / 2], placement.byteCount))
        }
        let blob = config.expertBlobLayout
        for layer in 0..<config.layerCount {
            for (index, stem) in config.expertTensors(layer: layer).enumerated() {
                let layout = index == 2 ? blob.downLayout : blob.gateLayout
                for (part, bytes) in [
                    ("weight", layout.weightBytes), ("scales", layout.scaleBytes),
                    ("biases", layout.biasBytes),
                ] {
                    declarations.append((
                        "\(stem.stem).\(part)", [config.expertCount, stem.rows, stem.cols],
                        bytes * config.expertCount))
                }
            }
        }
        for layer in 0..<3 {
            declarations.append((
                "vision_tower.encoder.layers.\(layer).self_attn.q_proj.weight", [64, 64], 8192))
        }

        var weightMap: [String: String] = [:]
        var json: [String: Any] = [:]
        var cursor = 0
        for (name, shape, bytes) in declarations {
            weightMap[name] = "model-00001.safetensors"
            json[name] = [
                "dtype": "BF16", "shape": shape, "data_offsets": [cursor, cursor + bytes],
            ]
            cursor += bytes
        }
        let data = try! JSONSerialization.data(withJSONObject: json)
        let header = try! SafetensorsHeader(headerJSON: data, headerByteCount: data.count)
        return (weightMap, ["model-00001.safetensors": header], cursor)
    }
}
