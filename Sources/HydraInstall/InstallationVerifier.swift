import Foundation
import HydraCore
import HydraFormat

/// Verifies that a `.hydra` installation really holds the upstream checkpoint's bytes.
///
/// The manifest's structural check, format, architecture, file sizes, is cheap and done
/// on every load. It proves nothing about the **content**, however: a repacker that wrote
/// everything one byte off would produce files of the right size and a valid
/// manifest.
///
/// This verifier closes that gap by sampling: it draws random windows from the expert blobs
/// and the resident tensors, re-requests the corresponding **source** bytes, and compares.
/// A systematic placement error, an offset, swapped sub-tensors, a wrong stride between
/// experts, is caught on the very first sample.
public struct InstallationVerifier: Sendable {

    public let plan: RepackPlan
    public let source: ByteRangeSource
    public let root: URL

    /// The size of a compared window. Large enough to be meaningful, small enough that the
    /// verification stays memory-bounded like the rest of the project.
    public static let windowBytes = 64 * 1024

    public init(plan: RepackPlan, source: ByteRangeSource, root: URL) {
        self.plan = plan
        self.source = source
        self.root = root
    }

    public struct Mismatch: Sendable, CustomStringConvertible {
        public let tensor: String
        public let chunkIndex: Int
        public let offsetInChunk: Int
        public let firstDifferingByte: Int

        public var description: String {
            "\(tensor) [chunk \(chunkIndex), offset \(offsetInChunk)] "
                + "diverges at byte \(firstDifferingByte)"
        }
    }

    public struct Report: Sendable {
        public let samplesChecked: Int
        public let bytesCompared: Int
        public let mismatches: [Mismatch]
        public var passed: Bool { mismatches.isEmpty }
    }

    public enum VerifyError: Error, CustomStringConvertible {
        case destinationUnreadable(String)

        public var description: String {
            switch self {
            case .destinationUnreadable(let f):
                return "installed file unreadable: \(f)"
            }
        }
    }

    /// Compares `sampleCount` randomly drawn windows. The draw is deterministic so that a
    /// verification failure is reproducible exactly.
    public func spotCheck(sampleCount: Int = 24, seed: UInt64 = 0x5EED) async throws -> Report {
        var rng = SplitMix64(seed: seed)
        var mismatches: [Mismatch] = []
        var bytesCompared = 0

        // One descriptor per destination file, opened on demand.
        var handles: [DestinationFile: FileHandle] = [:]
        defer { for (_, h) in handles { try? h.close() } }

        for _ in 0..<sampleCount {
            let op = plan.operations[Int(rng.next() % UInt64(plan.operations.count))]
            let chunkIndex = Int(rng.next() % UInt64(op.chunkCount))
            let maximumOffset = max(1, op.chunkByteCount - Self.windowBytes)
            let offsetInChunk = Int(rng.next() % UInt64(maximumOffset))
            let length = min(Self.windowBytes, op.chunkByteCount - offsetInChunk)

            // The bytes as they are in the upstream checkpoint.
            let sourceStart = op.sourceOffset + chunkIndex * op.chunkByteCount + offsetInChunk
            let expected = try await source.read(
                file: op.sourceShard, range: sourceStart..<(sourceStart + length))

            // The bytes as they were installed.
            let handle: FileHandle
            if let existing = handles[op.destination] {
                handle = existing
            } else {
                guard let opened = FileHandle(
                    forReadingAtPath: root.appending(path: op.destination.path).path)
                else {
                    throw VerifyError.destinationUnreadable(op.destination.path)
                }
                handles[op.destination] = opened
                handle = opened
            }
            let destinationStart = op.destinationOffset(ofChunk: chunkIndex) + offsetInChunk
            try handle.seek(toOffset: UInt64(destinationStart))
            let actual = try handle.read(upToCount: length) ?? Data()

            bytesCompared += length
            if actual != expected {
                let firstDifference = zip(expected, actual).enumerated()
                    .first { $0.element.0 != $0.element.1 }?.offset ?? min(expected.count, actual.count)
                mismatches.append(
                    Mismatch(
                        tensor: op.sourceTensor, chunkIndex: chunkIndex,
                        offsetInChunk: offsetInChunk, firstDifferingByte: firstDifference))
            }
        }

        return Report(
            samplesChecked: sampleCount, bytesCompared: bytesCompared, mismatches: mismatches)
    }
}

/// A deterministic generator: a verification failure must be reproducible as-is.
struct SplitMix64 {
    private var state: UInt64

    init(seed: UInt64) { state = seed }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
