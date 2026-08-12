import Foundation
import HydraCore
import HydraFormat

/// The plan for converting the MLX 4-bit Gemma 4 checkpoint into `.hydra`.
///
/// The same three artifacts as every other plan, and the same `ScatterCopy` machinery. What
/// differs is entirely in the names and the shapes:
///
/// - **Nine sub-tensors to an expert blob**, not two. The checkpoint keeps `gate_proj`,
///   `up_proj` and `down_proj` apart, and each carries its own `.scales` and `.biases`. The
///   BF16 build's fused `gate_up_proj` is split by the repacker; here there is nothing to
///   split and everything to gather.
/// - **The prefix is inverted**, `language_model.model.…`, and the vision tower is named
///   without one at all.
/// - **Mixed precision**, so a tensor's byte count comes from its own `MLXAffineLayout` and
///   never from a model-wide constant.
///
/// Each of the nine is still fused *across experts*: `U32[128, 704, 352]` is all 128 experts of
/// a layer end to end, which is exactly the shape `ScatterCopy` exists to scatter into blobs.
public struct GemmaMLXRepackPlan: Sendable {

    public let config: Gemma4MLXConfig
    public let layout: HydraLayout
    public let operations: [ScatterCopy]
    public let spans: [RepackPlan.SourceSpan]
    public let destinationSizes: [DestinationFile: Int]
    public let excluded: [(name: String, byteCount: Int, reason: String)]
    public let visionPlacements: [GemmaRepackPlan.VisionPlacement]

    /// The multimodal tower, installed for the same reason as the BF16 build's: the bytes are
    /// what the download buys, and leaving them out means fetching the checkpoint again to add
    /// image support. The names lose the `model.` prefix here.
    public static let visionPrefixes: [String] = ["vision_tower.", "embed_vision."]

    /// Nothing is excluded from this checkpoint, it carries no audio tower, but the rule is
    /// kept so a revision that grows one fails validation rather than disappearing.
    public static let excludedPrefixes: [(prefix: String, reason: String)] = [
        ("audio_tower.", "audio tower, not executed"),
        ("embed_audio.", "audio projection, not executed"),
    ]

    public static func isVisionTensor(_ name: String) -> Bool {
        visionPrefixes.contains { name.hasPrefix($0) }
    }

    public static func exclusionReason(for name: String) -> String? {
        excludedPrefixes.first { name.hasPrefix($0.prefix) }?.reason
    }

    public var totalSourceBytes: Int { operations.reduce(0) { $0 + $1.sourceByteCount } }
    public var totalExcludedBytes: Int { excluded.reduce(0) { $0 + $1.byteCount } }

    public init(
        config: Gemma4MLXConfig,
        weightMap: [String: String],
        headers: [String: SafetensorsHeader]
    ) throws {
        let layout = HydraLayout(model: config)
        self.config = config
        self.layout = layout

        var ops: [ScatterCopy] = []

        func source(_ name: String, expecting bytes: Int) throws -> (shard: String, offset: Int) {
            guard let shard = weightMap[name] else {
                throw RepackPlan.PlanError.missingTensor(name)
            }
            guard let header = headers[shard] else {
                throw RepackPlan.PlanError.unknownShard(shard)
            }
            guard let entry = header.tensors[name], let range = header.fileRange(of: name) else {
                throw RepackPlan.PlanError.missingTensor(name)
            }
            guard entry.byteCount == bytes else {
                throw RepackPlan.PlanError.unexpectedShape(
                    name, expected: bytes, got: entry.byteCount)
            }
            return (shard, range.lowerBound)
        }

        // --- Resident tensors ---
        for placement in layout.resident {
            let s = try source(placement.sourceName, expecting: placement.byteCount)
            ops.append(
                ScatterCopy(
                    sourceTensor: placement.sourceName,
                    sourceShard: s.shard, sourceOffset: s.offset,
                    destination: .resident,
                    destinationOffset: placement.offset,
                    destinationStride: 0,
                    chunkByteCount: placement.byteCount,
                    chunkCount: 1))
        }

        // --- Experts: nine tensors per layer, each fused across the 128 experts ---
        let blob = config.expertBlobLayout
        let experts = config.expertCount
        for layer in 0..<config.layerCount {
            let stems = config.expertTensors(layer: layer)
            let slots: [(part: String, slot: ExpertBlobLayout.Slot, bytes: Int)] = [
                ("weight", blob.gateWeights, blob.gateLayout.weightBytes),
                ("scales", blob.gateScales, blob.gateLayout.scaleBytes),
                ("biases", blob.gateBiases, blob.gateLayout.biasBytes),
                ("weight", blob.upWeights, blob.gateLayout.weightBytes),
                ("scales", blob.upScales, blob.gateLayout.scaleBytes),
                ("biases", blob.upBiases, blob.gateLayout.biasBytes),
                ("weight", blob.downWeights, blob.downLayout.weightBytes),
                ("scales", blob.downScales, blob.downLayout.scaleBytes),
                ("biases", blob.downBiases, blob.downLayout.biasBytes),
            ]
            for (index, entry) in slots.enumerated() {
                let name = "\(stems[index / 3].stem).\(entry.part)"
                let s = try source(name, expecting: entry.bytes * experts)
                ops.append(
                    ScatterCopy(
                        sourceTensor: name,
                        sourceShard: s.shard, sourceOffset: s.offset,
                        destination: .expertLayer(layer),
                        destinationOffset: entry.slot.offset,
                        destinationStride: blob.strideBytes,
                        chunkByteCount: entry.bytes,
                        chunkCount: experts))
            }
        }

        // --- The vision tower, copied verbatim and described by the manifest ---
        var visionCursor = 0
        var placements: [GemmaRepackPlan.VisionPlacement] = []
        for name in weightMap.keys.filter(Self.isVisionTensor).sorted() {
            guard let shard = weightMap[name], let header = headers[shard],
                let entry = header.tensors[name], let range = header.fileRange(of: name)
            else { throw RepackPlan.PlanError.missingTensor(name) }

            visionCursor = alignUp(visionCursor, to: HydraLayout.tensorAlignment)
            placements.append(
                GemmaRepackPlan.VisionPlacement(
                    name: name, offset: visionCursor, byteCount: entry.byteCount,
                    dtype: entry.dtype, shape: entry.shape))
            ops.append(
                ScatterCopy(
                    sourceTensor: name,
                    sourceShard: shard, sourceOffset: range.lowerBound,
                    destination: .vision,
                    destinationOffset: visionCursor,
                    destinationStride: 0,
                    chunkByteCount: entry.byteCount,
                    chunkCount: 1))
            visionCursor += entry.byteCount
        }
        self.visionPlacements = placements
        let visionBytes = placements.isEmpty
            ? 0 : alignUp(visionCursor, to: HydraLayout.pageAlignment)

        // --- What is present but not executed ---
        let planned = Set(ops.map(\.sourceTensor))
        var skipped: [(name: String, byteCount: Int, reason: String)] = []
        for (name, shard) in weightMap where !planned.contains(name) {
            guard let reason = Self.exclusionReason(for: name) else { continue }
            skipped.append((name, headers[shard]?.tensors[name]?.byteCount ?? 0, reason))
        }
        self.excluded = skipped.sorted { $0.name < $1.name }

        ops.sort {
            $0.sourceShard == $1.sourceShard
                ? $0.sourceOffset < $1.sourceOffset
                : $0.sourceShard < $1.sourceShard
        }
        self.operations = ops
        self.spans = RepackPlan.makeSpans(
            operations: ops, maximumBytes: RepackPlan.maximumSpanBytes)

        var sizes: [DestinationFile: Int] = [.resident: layout.residentBytes]
        for layer in 0..<config.layerCount {
            sizes[.expertLayer(layer)] = layout.expertLayerFileBytes
        }
        if visionBytes > 0 { sizes[.vision] = visionBytes }
        self.destinationSizes = sizes
    }

    /// The same restated coverage guarantee as the BF16 plan: every tensor in the index is
    /// planned, installed as vision, or excluded by a recorded rule.
    public func validate(
        weightMap: [String: String], declaredSourceTotal: Int?
    ) -> [RepackPlan.Problem] {
        var problems: [RepackPlan.Problem] = []

        var writes: [DestinationFile: [Range<Int>]] = [:]
        for op in operations {
            guard let size = destinationSizes[op.destination] else {
                problems.append(RepackPlan.Problem(
                    description: "\(op.sourceTensor) targets an unplanned file"))
                continue
            }
            for i in 0..<op.chunkCount {
                let start = op.destinationOffset(ofChunk: i)
                let range = start..<(start + op.chunkByteCount)
                if range.upperBound > size {
                    problems.append(RepackPlan.Problem(
                        description: "\(op.sourceTensor) overruns \(op.destination.path) "
                            + "(\(range.upperBound) > \(size))"))
                    break
                }
                writes[op.destination, default: []].append(range)
            }
        }
        for (file, ranges) in writes {
            let sorted = ranges.sorted { $0.lowerBound < $1.lowerBound }
            for i in 1..<max(sorted.count, 1) where sorted[i].lowerBound < sorted[i - 1].upperBound {
                problems.append(RepackPlan.Problem(
                    description: "overlapping writes in \(file.path): "
                        + "\(sorted[i - 1]) and \(sorted[i])"))
                break
            }
        }

        let planned = Set(operations.map(\.sourceTensor))
        let excludedNames = Set(excluded.map(\.name))
        let unaccounted = weightMap.keys.filter {
            !planned.contains($0) && !excludedNames.contains($0)
        }
        if !unaccounted.isEmpty {
            problems.append(RepackPlan.Problem(
                description: "\(unaccounted.count) tensor(s) neither planned nor excluded, "
                    + "first: \(unaccounted.sorted().prefix(3).joined(separator: ", "))"))
        }

        if let declared = declaredSourceTotal,
            declared != totalSourceBytes + totalExcludedBytes
        {
            problems.append(RepackPlan.Problem(
                description: "incomplete coverage: planned \(totalSourceBytes) + excluded "
                    + "\(totalExcludedBytes) against a declared \(declared)"))
        }
        return problems
    }
}
