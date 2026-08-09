import CryptoKit
import Foundation
import HydraCore
import HydraFormat

/// Executes a `RepackPlan` without ever materializing more than one network block in memory.
///
/// **The project's invariant lives here.** The source checkpoint weighs 12.8 GiB for the 20B
/// and 60.8 GiB for the 120B; at no point may the heap hold a meaningful share of it. The
/// mechanism comes down to three rules:
///
/// 1. the plan is cut into **contiguous regions** of the source checkpoint, each downloaded
///    in a single request;
/// 2. the response is **consumed as it streams** — each block the network stack delivers is
///    routed to its destination then released before the next one arrives;
/// 3. nothing accumulates between blocks, save a 32-byte hash state per operation in
///    flight.
///
/// A first version split every range into 4 MiB sub-requests, so that the memory bound would
/// be a property of the splitting rather than of the network stack's behaviour. Measured on
/// the real repository, that choice divided throughput by 6.4. The bound now comes from
/// incremental consumption, and it is checked by the tests rather than guaranteed by
/// construction.
public struct StreamingRepacker: Sendable {

    public let plan: any InstallablePlan
    public let source: ByteRangeSource

    /// Number of operations between two disk synchronizations and two resume points.
    public let checkpointInterval: Int

    /// Called with the **partial** directory, just before the manifest is written.
    ///
    /// Used to drop in whatever is not a weight — the tokenizer, in practice. Doing it before
    /// promotion guarantees that a promoted installation is always complete: a valid `.hydra`
    /// without its tokenizer cannot exist.
    public var auxiliary: (@Sendable (URL) async throws -> Void)?

    public init(
        plan: any InstallablePlan, source: ByteRangeSource, checkpointInterval: Int = 16,
        auxiliary: (@Sendable (URL) async throws -> Void)? = nil
    ) {
        self.plan = plan
        self.source = source
        self.checkpointInterval = checkpointInterval
        self.auxiliary = auxiliary
    }

    public struct Progress: Sendable {
        public let operationsDone: Int
        public let operationsTotal: Int
        public let bytesDone: Int
        public let bytesTotal: Int
        public let currentTensor: String
        /// The largest block received so far. Serves as **experimental proof** of the memory
        /// invariant, not merely as a design argument.
        public let peakPayloadBytes: Int
    }

    public enum RepackError: Error, CustomStringConvertible {
        case journalUnreadable(String)
        case incompleteSpan(shard: String, expected: Int, got: Int)

        public var description: String {
            switch self {
            case .journalUnreadable(let m):
                return "resume journal unreadable: \(m)"
            case let .incompleteSpan(shard, expected, got):
                return "incomplete region on \(shard): \(got) bytes routed, \(expected) expected"
            }
        }
    }

    // MARK: - Resume journal

    /// Records the operations whose data is **durable**. An operation appears only after
    /// `F_FULLFSYNC`, so a resume cannot skip a write left sitting in a disk cache.
    ///
    struct Journal: Codable {
        var sourceDescription: String
        var completed: [Int: String]  // operation index -> sha256 of the source bytes

        static let fileName = "progress.json"

        static func read(from root: URL) throws -> Journal? {
            let url = root.appending(path: fileName)
            guard FileManager.default.fileExists(atPath: url.path) else { return nil }
            do {
                return try JSONDecoder().decode(Journal.self, from: Data(contentsOf: url))
            } catch {
                throw RepackError.journalUnreadable(error.localizedDescription)
            }
        }

        func write(to root: URL) throws {
            let encoder = JSONEncoder()
            encoder.outputFormatting = .sortedKeys
            try encoder.encode(self).write(to: root.appending(path: Self.fileName), options: .atomic)
        }
    }

    // MARK: - Routing a region

    /// Distributes a contiguous region's bytes to its operations' destinations.
    ///
    /// Blocks arrive in order but at arbitrary boundaries: one block may straddle two
    /// operations, and within one operation two expert blobs. The router therefore assumes no
    /// alignment at all; it merely follows a cursor.
    ///
    /// Safety: `consume` is called serially by the source (`URLSession`'s delegate queue, or a
    /// local read loop). No concurrent access occurs.
    final class SpanRouter: @unchecked Sendable {
        private let operations: [ScatterCopy]
        private let indices: [Int]
        private let writer: InstallationWriter

        private var position = 0  // index into `indices`
        private var consumedInOperation = 0
        private var hasher = SHA256()
        private(set) var digests: [Int: String] = [:]
        private(set) var bytesRouted = 0
        private(set) var peakBlock = 0

        init(operations: [ScatterCopy], indices: [Int], writer: InstallationWriter) {
            self.operations = operations
            self.indices = indices
            self.writer = writer
        }

        var currentTensor: String {
            position < indices.count ? operations[indices[position]].sourceTensor : ""
        }

        func consume(_ block: Data) throws {
            peakBlock = max(peakBlock, block.count)
            var offsetInBlock = 0

            while offsetInBlock < block.count {
                guard position < indices.count else { break }
                let op = operations[indices[position]]

                let remainingInOperation = op.sourceByteCount - consumedInOperation
                let take = min(remainingInOperation, block.count - offsetInBlock)
                let piece = block.subdata(
                    in: block.startIndex.advanced(by: offsetInBlock)
                        ..< block.startIndex.advanced(by: offsetInBlock + take))
                hasher.update(data: piece)

                // Scattering: the piece may span several expert blobs.
                var written = 0
                while written < take {
                    let absolute = consumedInOperation + written
                    let chunkIndex = absolute / op.chunkByteCount
                    let offsetInChunk = absolute % op.chunkByteCount
                    let slice = min(op.chunkByteCount - offsetInChunk, take - written)

                    try writer.write(
                        piece.subdata(
                            in: piece.startIndex.advanced(by: written)
                                ..< piece.startIndex.advanced(by: written + slice)),
                        to: op.destination,
                        at: op.destinationOffset(ofChunk: chunkIndex) + offsetInChunk)
                    written += slice
                }

                consumedInOperation += take
                offsetInBlock += take
                bytesRouted += take

                if consumedInOperation == op.sourceByteCount {
                    digests[indices[position]] = hasher.finalize().hexString
                    hasher = SHA256()
                    consumedInOperation = 0
                    position += 1
                }
            }
        }
    }

    // MARK: - Execution

    /// Installs into `<destination>.partial`, then promotes atomically.
    /// Resumes automatically if a coherent partial directory already exists.
    @discardableResult
    public func run(
        destination: URL,
        onProgress: (@Sendable (Progress) -> Void)? = nil
    ) async throws -> HydraManifest {

        let partial = destination.appendingPathExtension("partial")
        let writer = try InstallationWriter(root: partial, sizes: plan.destinationSizes)
        defer { writer.close() }

        var journal = try Journal.read(from: partial)
            ?? Journal(sourceDescription: source.sourceDescription, completed: [:])
        if journal.sourceDescription != source.sourceDescription {
            // The source changed: start over rather than mix two revisions.
            journal = Journal(sourceDescription: source.sourceDescription, completed: [:])
        }

        let totalBytes = plan.totalSourceBytes
        let alreadyDone = plan.operations.enumerated()
            .filter { journal.completed[$0.offset] != nil }
            .reduce(0) { $0 + $1.element.sourceByteCount }

        let counters = Counters(bytesDone: alreadyDone)
        var sinceCheckpoint = 0

        for span in plan.spans {
            try Task.checkCancellation()

            // A region's operations finish in order: resuming at the first unfinished one is
            // enough, and shortens the range to re-download by that much.
            guard let resumeAt = span.operationIndices.firstIndex(where: {
                journal.completed[$0] == nil
            }) else { continue }

            let remaining = Array(span.operationIndices[resumeAt...])
            let start = plan.operations[remaining[0]].sourceOffset
            let router = SpanRouter(
                operations: plan.operations, indices: remaining, writer: writer)

            let operationsDoneBefore = journal.completed.count
            try await source.stream(
                file: span.shard, range: start..<span.range.upperBound
            ) { block in
                try router.consume(block)
                let snapshot = counters.advance(by: block.count, peak: router.peakBlock)
                onProgress?(
                    Progress(
                        operationsDone: operationsDoneBefore, operationsTotal: plan.operations.count,
                        bytesDone: snapshot.bytes, bytesTotal: totalBytes,
                        currentTensor: router.currentTensor, peakPayloadBytes: snapshot.peak))
            }

            guard router.bytesRouted == span.range.upperBound - start else {
                throw RepackError.incompleteSpan(
                    shard: span.shard, expected: span.range.upperBound - start,
                    got: router.bytesRouted)
            }

            for (index, digest) in router.digests { journal.completed[index] = digest }
            sinceCheckpoint += router.digests.count
            if sinceCheckpoint >= checkpointInterval {
                try writer.synchronize()
                try journal.write(to: partial)
                sinceCheckpoint = 0
            }
        }

        // Everything must be durable before the manifest asserts the installation exists.
        try writer.synchronize()
        try journal.write(to: partial)

        let manifest = HydraManifest(
            model: .init(model: plan.model),
            layout: .init(
                expertStrideBytes: plan.layout.expertBlob.strideBytes,
                expertPayloadBytes: plan.layout.expertBlob.payloadBytes,
                residentBytes: plan.layout.residentBytes,
                embeddingBytes: plan.layout.embeddingBytes,
                tensorAlignment: ExpertBlobLayout.tensorAlignment,
                pageAlignment: ExpertBlobLayout.pageAlignment),
            files: Dictionary(
                uniqueKeysWithValues: plan.destinationSizes.map {
                    ($0.key.path, HydraManifest.FileEntry(byteCount: $0.value))
                }),
            sourceDescription: source.sourceDescription,
            sourceTotalBytes: totalBytes,
            tensors: plan.operations.enumerated().map { index, op in
                HydraManifest.TensorDigest(
                    tensor: op.sourceTensor, byteCount: op.sourceByteCount,
                    sha256: journal.completed[index] ?? "")
            },
            // `nil` rather than `[]` for a text-only installation, so a reader can tell "no
            // tower" from "a tower whose description was lost".
            vision: plan.visionTensors.isEmpty ? nil : plan.visionTensors)

        // The tokenizer and metadata are laid down before the manifest: a promoted
        // installation is complete, or does not exist.
        if let auxiliary { try await auxiliary(partial) }

        try manifest.write(to: partial)
        try? FileManager.default.removeItem(at: partial.appending(path: Journal.fileName))
        try writer.promote(to: destination)
        return manifest
    }

    /// Counters shared with the progress callback, which is `@Sendable`.
    final class Counters: @unchecked Sendable {
        private var bytes: Int
        private var peak = 0
        private let lock = NSLock()

        init(bytesDone: Int) { bytes = bytesDone }

        func advance(by count: Int, peak block: Int) -> (bytes: Int, peak: Int) {
            lock.lock()
            defer { lock.unlock() }
            bytes += count
            peak = max(peak, block)
            return (bytes, peak)
        }
    }
}

extension SHA256.Digest {
    var hexString: String { map { String(format: "%02x", $0) }.joined() }
}
