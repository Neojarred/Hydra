import Foundation
import HydraCore
import HydraFormat

/// The plan for converting a Gemma 4 checkpoint into `.hydra`.
///
/// The same three artifacts as `RepackPlan` — operations, spans, destination sizes — built
/// from a different set of tensor names. The `ScatterCopy` machinery is unchanged: Gemma
/// stores its experts fused per layer, exactly the shape that mechanism exists to split.
///
/// **What differs is the coverage guarantee.** For GPT-OSS the decisive check is that the
/// planned bytes equal the `total_size` the index declares: if they match, nothing was
/// silently forgotten. Gemma's checkpoint also carries a vision tower and, in other members
/// of the family, an audio tower — which this runtime deliberately does not execute (D-021).
/// Planned bytes are therefore *less* than the declared total, and the naive check would have
/// to be abandoned.
///
/// It is restated instead: **every tensor in the index is either planned or explicitly
/// excluded by a recorded rule, and the two sum to the declared total.** A tensor that
/// matched no rule and no plan would fail validation rather than disappear — which is the
/// property that mattered, and the reason the towers are excluded by name rather than by
/// omission.
public struct GemmaRepackPlan: Sendable {

    public let config: Gemma4Config
    public let layout: HydraLayout
    public let operations: [ScatterCopy]
    public let spans: [RepackPlan.SourceSpan]
    public let destinationSizes: [DestinationFile: Int]

    /// Tensors present in the checkpoint that this runtime does not execute, with their sizes.
    ///
    /// Recorded rather than dropped: the manifest carries them so that adding image support
    /// later is additive, and so that a reader can see what a `.hydra` installation chose not
    /// to contain.
    public let excluded: [(name: String, byteCount: Int, reason: String)]

    /// Prefixes this runtime does not execute, each with the reason it is skipped.
    ///
    /// Matching by prefix rather than listing names is deliberate: a checkpoint that grew a
    /// new vision layer would still be covered, where an explicit list would silently start
    /// failing the coverage check.
    public static let excludedPrefixes: [(prefix: String, reason: String)] = [
        ("model.vision_tower.", "vision tower — not executed (D-021)"),
        ("model.embed_vision.", "vision projection — not executed (D-021)"),
        ("model.audio_tower.", "audio tower — not executed (D-021)"),
        ("model.embed_audio.", "audio projection — not executed (D-021)"),
    ]

    public static func exclusionReason(for name: String) -> String? {
        excludedPrefixes.first { name.hasPrefix($0.prefix) }?.reason
    }

    public var totalSourceBytes: Int { operations.reduce(0) { $0 + $1.sourceByteCount } }
    public var totalExcludedBytes: Int { excluded.reduce(0) { $0 + $1.byteCount } }

    // MARK: - Construction

    public init(
        config: Gemma4Config,
        weightMap: [String: String],
        headers: [String: SafetensorsHeader]
    ) throws {
        let layout = HydraLayout(config: config)
        self.config = config
        self.layout = layout

        var ops: [ScatterCopy] = []

        func source(_ name: String, expecting bytes: Int) throws -> (shard: String, offset: Int) {
            guard let shard = weightMap[name] else { throw RepackPlan.PlanError.missingTensor(name) }
            guard let header = headers[shard] else { throw RepackPlan.PlanError.unknownShard(shard) }
            guard let entry = header.tensors[name], let range = header.fileRange(of: name) else {
                throw RepackPlan.PlanError.missingTensor(name)
            }
            guard entry.byteCount == bytes else {
                throw RepackPlan.PlanError.unexpectedShape(
                    name, expected: bytes, got: entry.byteCount)
            }
            return (shard, range.lowerBound)
        }

        // --- Resident tensors: contiguous copies into resident.bin ---
        //
        // The tied embedding is among them: it is the output head, read on every token, so
        // unlike GPT-OSS there is no separate embedding file.
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

        // --- Experts: one fused tensor per layer, scattered into E blobs ---
        let blob = config.expertBlobLayout
        let experts = config.expertCount
        for layer in 0..<config.layerCount {
            let base = "model.language_model.layers.\(layer).experts"
            for (suffix, slot) in [
                ("gate_up_proj", blob.gateUp),
                ("down_proj", blob.down),
            ] {
                let name = "\(base).\(suffix)"
                let s = try source(name, expecting: slot.byteCount * experts)
                ops.append(
                    ScatterCopy(
                        sourceTensor: name,
                        sourceShard: s.shard, sourceOffset: s.offset,
                        destination: .expertLayer(layer),
                        destinationOffset: slot.offset,
                        destinationStride: blob.strideBytes,
                        chunkByteCount: slot.byteCount,
                        chunkCount: experts))
            }
        }

        // --- What is present but not executed ---
        let planned = Set(ops.map(\.sourceTensor))
        var skipped: [(name: String, byteCount: Int, reason: String)] = []
        for (name, shard) in weightMap where !planned.contains(name) {
            guard let reason = Self.exclusionReason(for: name) else { continue }
            let bytes = headers[shard]?.tensors[name]?.byteCount ?? 0
            skipped.append((name, bytes, reason))
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
        self.destinationSizes = sizes
    }

    // MARK: - Verification

    /// Checks the plan **before** any download, the way `RepackPlan.validate` does.
    ///
    /// - Parameters:
    ///   - weightMap: every tensor the index declares, so that one matching neither the plan
    ///     nor an exclusion rule is reported rather than ignored.
    ///   - declaredSourceTotal: the index's `total_size`.
    public func validate(
        weightMap: [String: String], declaredSourceTotal: Int?
    ) -> [RepackPlan.Problem] {
        var problems: [RepackPlan.Problem] = []

        // No write may run past its file or overlap another.
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

        // The restated coverage guarantee: nothing in the index is unaccounted for.
        let planned = Set(operations.map(\.sourceTensor))
        let excludedNames = Set(excluded.map(\.name))
        let unaccounted = weightMap.keys.filter {
            !planned.contains($0) && !excludedNames.contains($0)
        }
        if !unaccounted.isEmpty {
            problems.append(RepackPlan.Problem(
                description: "\(unaccounted.count) tensor(s) neither planned nor excluded, "
                    + "starting with \(unaccounted.sorted().prefix(3).joined(separator: ", "))"))
        }

        if let declared = declaredSourceTotal {
            let accounted = totalSourceBytes + totalExcludedBytes
            if accounted != declared {
                problems.append(RepackPlan.Problem(
                    description: "incomplete coverage: \(totalSourceBytes) bytes planned plus "
                        + "\(totalExcludedBytes) excluded is \(accounted), against \(declared) "
                        + "declared by the index (gap \(declared - accounted))"))
            }
        }

        return problems
    }
}
