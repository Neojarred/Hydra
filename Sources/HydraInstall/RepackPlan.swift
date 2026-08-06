import Foundation
import HydraCore
import HydraFormat

/// A destination file of a `.hydra` installation.
public enum DestinationFile: Sendable, Hashable {
    case resident
    case embedding
    case expertLayer(Int)

    public var path: String {
        switch self {
        case .resident: return "resident.bin"
        case .embedding: return "embed.bin"
        case .expertLayer(let i): return String(format: "experts/layer_%02d.bin", i)
        }
    }
}

/// A copy from a source range to one or more evenly spaced destinations.
///
/// The `chunkCount == 1` case is an ordinary contiguous copy (a resident tensor).
///
/// The `chunkCount > 1` case is what makes the expert repack efficient. In the source
/// checkpoint, a tensor like `gate_up_proj_blocks` holds **all** of a layer's experts end to
/// end. In `.hydra` they must be spread across blobs interleaved with the other
/// sub-tensors. Rather than issuing one request per expert, we read the source range **once,
/// sequentially**, and scatter each chunk to `destinationOffset + i * destinationStride`.
///
///
/// The direct consequence: about 150 network requests to install the 20B instead of several
/// thousand — while keeping a bounded working buffer.
public struct ScatterCopy: Sendable, Equatable {
    public let sourceTensor: String
    public let sourceShard: String
    /// Absolute offset in the shard, data section included.
    public let sourceOffset: Int
    public let destination: DestinationFile
    public let destinationOffset: Int
    public let destinationStride: Int
    public let chunkByteCount: Int
    public let chunkCount: Int

    public var sourceByteCount: Int { chunkByteCount * chunkCount }
    public var sourceRange: Range<Int> { sourceOffset..<(sourceOffset + sourceByteCount) }

    /// Destination offset of chunk `index`.
    public func destinationOffset(ofChunk index: Int) -> Int {
        destinationOffset + index * destinationStride
    }
}

/// The complete plan for converting a Hugging Face checkpoint to `.hydra`.
///
/// The plan is computed **from the headers alone** — a few tens of KiB of network — and is
/// fully verifiable before committing to a single byte of weight download.
public struct RepackPlan: Sendable {

    public let config: GptOssConfig
    public let layout: HydraLayout
    /// Sorted by (shard, source offset): reading stays sequential on the remote disk, which
    /// avoids reopening connections to go backwards.
    public let operations: [ScatterCopy]

    /// Contiguous regions of the source checkpoint, each covered by consecutive operations. A
    /// region downloads in **a single request**, whose response is routed to several
    /// destinations as the bytes arrive.
    ///
    /// This is what makes installation fast. Since the plan covers the checkpoint exactly with
    /// no gaps, neighbouring tensors within a shard form long contiguous regions: we read the
    /// source file nearly end to end, sequentially, instead of issuing one request per tensor.
    /// Measured on the real repository: 33.5 MB/s with large requests against 5.2 MB/s with
    /// small ones.
    public let spans: [SourceSpan]

    public let destinationSizes: [DestinationFile: Int]

    /// A region's ceiling. It does not bound memory — the response is consumed as it streams —
    /// but it bounds what a network interruption forces us to redo, and gives the resume
    /// journal a useful granularity.
    public static let maximumSpanBytes = 256 * 1024 * 1024

    public struct SourceSpan: Sendable, Equatable {
        public let shard: String
        public let range: Range<Int>
        /// Indices into `operations`, consecutive and ordered. End to end they cover `range`
        /// exactly.
        public let operationIndices: [Int]
    }

    static func makeSpans(operations: [ScatterCopy], maximumBytes: Int) -> [SourceSpan] {
        var spans: [SourceSpan] = []
        var current: (shard: String, start: Int, end: Int, indices: [Int])?

        for (index, op) in operations.enumerated() {
            if var open = current,
                open.shard == op.sourceShard,
                open.end == op.sourceOffset,
                open.end - open.start + op.sourceByteCount <= maximumBytes
            {
                open.end += op.sourceByteCount
                open.indices.append(index)
                current = open
                continue
            }
            if let open = current {
                spans.append(
                    SourceSpan(
                        shard: open.shard, range: open.start..<open.end,
                        operationIndices: open.indices))
            }
            current = (op.sourceShard, op.sourceOffset, op.sourceRange.upperBound, [index])
        }
        if let open = current {
            spans.append(
                SourceSpan(
                    shard: open.shard, range: open.start..<open.end, operationIndices: open.indices))
        }
        return spans
    }

    public var totalSourceBytes: Int {
        operations.reduce(0) { $0 + $1.sourceByteCount }
    }

    public var totalDestinationBytes: Int {
        destinationSizes.values.reduce(0, +)
    }

    public enum PlanError: Error, CustomStringConvertible {
        case missingTensor(String)
        case unexpectedShape(String, expected: Int, got: Int)
        case unknownShard(String)

        public var description: String {
            switch self {
            case .missingTensor(let n):
                return "tensor missing from the source checkpoint: \(n)"
            case let .unexpectedShape(n, e, g):
                return "tensor \(n): \(g) bytes, \(e) expected — incompatible checkpoint"
            case .unknownShard(let s):
                return "missing header for shard \(s)"
            }
        }
    }

    /// Builds the plan from the weight map and each shard's header.
    public init(
        config: GptOssConfig,
        weightMap: [String: String],
        headers: [String: SafetensorsHeader]
    ) throws {
        let layout = HydraLayout(config: config)
        self.config = config
        self.layout = layout

        var ops: [ScatterCopy] = []

        /// Locates a source tensor and checks its size.
        func source(_ name: String, expecting bytes: Int) throws -> (shard: String, offset: Int) {
            guard let shard = weightMap[name] else { throw PlanError.missingTensor(name) }
            guard let header = headers[shard] else { throw PlanError.unknownShard(shard) }
            guard let entry = header.tensors[name], let range = header.fileRange(of: name) else {
                throw PlanError.missingTensor(name)
            }
            guard entry.byteCount == bytes else {
                throw PlanError.unexpectedShape(name, expected: bytes, got: entry.byteCount)
            }
            return (shard, range.lowerBound)
        }

        // --- Resident tensors: contiguous copies into resident.bin ---
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

        // --- Embedding: a dedicated file, mapped but outside the Metal working set ---
        let embed = try source("model.embed_tokens.weight", expecting: config.embeddingBytes)
        ops.append(
            ScatterCopy(
                sourceTensor: "model.embed_tokens.weight",
                sourceShard: embed.shard, sourceOffset: embed.offset,
                destination: .embedding,
                destinationOffset: 0, destinationStride: 0,
                chunkByteCount: config.embeddingBytes, chunkCount: 1))

        // --- Experts: scattering one source tensor into E blobs ---
        let blob = layout.expertBlob
        let E = config.expertCount
        for layer in 0..<config.layerCount {
            let subTensors: [(suffix: String, slot: ExpertBlobLayout.Slot)] = [
                ("gate_up_proj_blocks", blob.gateUpBlocks),
                ("gate_up_proj_scales", blob.gateUpScales),
                ("gate_up_proj_bias", blob.gateUpBias),
                ("down_proj_blocks", blob.downBlocks),
                ("down_proj_scales", blob.downScales),
                ("down_proj_bias", blob.downBias),
            ]
            for (suffix, slot) in subTensors {
                let name = "model.layers.\(layer).mlp.experts.\(suffix)"
                let s = try source(name, expecting: slot.byteCount * E)
                ops.append(
                    ScatterCopy(
                        sourceTensor: name,
                        sourceShard: s.shard, sourceOffset: s.offset,
                        destination: .expertLayer(layer),
                        destinationOffset: slot.offset,
                        destinationStride: blob.strideBytes,
                        chunkByteCount: slot.byteCount,
                        chunkCount: E))
            }
        }

        ops.sort {
            $0.sourceShard == $1.sourceShard
                ? $0.sourceOffset < $1.sourceOffset
                : $0.sourceShard < $1.sourceShard
        }
        self.operations = ops
        self.spans = Self.makeSpans(operations: ops, maximumBytes: Self.maximumSpanBytes)

        var sizes: [DestinationFile: Int] = [
            .resident: layout.residentBytes,
            .embedding: config.embeddingBytes,
        ]
        for layer in 0..<config.layerCount {
            sizes[.expertLayer(layer)] = layout.expertLayerFileBytes
        }
        self.destinationSizes = sizes
    }

    // MARK: - Verification

    public struct Problem: Sendable, CustomStringConvertible {
        public let description: String
    }

    /// Checks that the plan is coherent **before** any download.
    ///
    /// The decisive check is the last one: the sum of source bytes covered must equal the
    /// `total_size` the index declares. If they match, the plan covers the whole checkpoint,
    /// with neither gap nor duplicate — hence no tensor was silently forgotten.
    ///
    public func validate(declaredSourceTotal: Int?) -> [Problem] {
        var problems: [Problem] = []

        // Aucun tenseur source lu deux fois.
        var seen = Set<String>()
        for op in operations where !seen.insert(op.sourceTensor).inserted {
            problems.append(Problem(description: "tenseur source lu deux fois : \(op.sourceTensor)"))
        }

        // No write may run past its file or overlap another.
        var writes: [DestinationFile: [Range<Int>]] = [:]
        for op in operations {
            guard let size = destinationSizes[op.destination] else {
                problems.append(Problem(description: "destination inconnue : \(op.destination.path)"))
                continue
            }
            for i in 0..<op.chunkCount {
                let start = op.destinationOffset(ofChunk: i)
                let range = start..<(start + op.chunkByteCount)
                if range.upperBound > size {
                    problems.append(
                        Problem(description:
                            "\(op.sourceTensor) overruns \(op.destination.path) "
                            + "(\(range.upperBound) > \(size))"))
                    break
                }
                writes[op.destination, default: []].append(range)
            }
        }
        for (file, ranges) in writes {
            let sorted = ranges.sorted { $0.lowerBound < $1.lowerBound }
            for i in 1..<max(sorted.count, 1) where sorted[i].lowerBound < sorted[i - 1].upperBound {
                problems.append(
                    Problem(description:
                        "overlapping writes in \(file.path): "
                        + "\(sorted[i - 1]) and \(sorted[i])"))
                break
            }
        }

        // Does the plan cover the source checkpoint exactly?
        if let declared = declaredSourceTotal, declared != totalSourceBytes {
            problems.append(
                Problem(description:
                    "incomplete coverage: \(totalSourceBytes) bytes planned, "
                    + "\(declared) declared by the index (gap \(declared - totalSourceBytes))"))
        }

        return problems
    }
}
