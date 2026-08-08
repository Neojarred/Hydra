import Darwin
import Foundation
import HydraCore
import HydraFormat

/// Rewrites `resident.bin` at a lower precision, once the download is complete.
///
/// **A post-pass, not a step in the stream.** Quantizing inside `SpanRouter` would have been
/// possible, but it would have cost the thing that makes installation trustworthy: the
/// verifier re-requests source bytes and compares them to what was written. Transformed
/// bytes cannot be compared to their source. Doing it afterwards keeps that check exact over
/// the whole download, and confines the transformation to a step that is deterministic,
/// local, and reproducible by re-running it.
///
/// It costs one extra pass over a file that is 2.27 GiB for the 20B — a few seconds against
/// the tens of minutes the download takes.
///
/// Memory stays bounded the same way the repacker's does: the file is walked in chunks of
/// whole blocks, never held. The LM head alone is 1.08 GiB, so holding even one tensor would
/// have broken the project's invariant.
public struct DenseQuantizer: Sendable {

    public let config: GptOssConfig
    public let precision: PrecisionPolicy

    /// Values converted per chunk. 2 M values is 4 MiB read and 2 MiB written — large enough
    /// that the syscalls disappear, small enough to be invisible in the footprint.
    static let valuesPerChunk = 65536 * Q8.blockSize

    public init(config: GptOssConfig, precision: PrecisionPolicy) {
        self.config = config
        self.precision = precision
    }

    public enum QuantizeError: Error, CustomStringConvertible {
        case openFailed(String, errno: Int32)
        case readFailed(String, at: Int, errno: Int32)
        case writeFailed(String, at: Int, errno: Int32)
        case shortRead(String, at: Int, expected: Int, got: Int)
        case tensorMissing(String)

        public var description: String {
            switch self {
            case let .openFailed(f, e):
                return "cannot open \(f): \(String(cString: strerror(e)))"
            case let .readFailed(f, o, e):
                return "cannot read \(f) at \(o): \(String(cString: strerror(e)))"
            case let .writeFailed(f, o, e):
                return "cannot write \(f) at \(o): \(String(cString: strerror(e)))"
            case let .shortRead(f, o, expected, got):
                return "short read in \(f) at \(o): \(got) bytes of \(expected)"
            case .tensorMissing(let name):
                return "\(name) is absent from the published layout — the two layouts disagree"
            }
        }
    }

    public struct Report: Sendable {
        public let tensorsQuantized: Int
        public let bytesBefore: Int
        public let bytesAfter: Int
        public var savedBytes: Int { bytesBefore - bytesAfter }
    }

    /// Converts `<root>/resident.bin` in place, through a temporary file that replaces it
    /// only once complete. An interrupted run therefore leaves the published file intact.
    @discardableResult
    public func run(root: URL) throws -> Report {
        let published = HydraLayout(config: config, precision: .published)
        let target = HydraLayout(config: config, precision: precision)

        let sourceURL = root.appending(path: "resident.bin")
        let targetURL = root.appending(path: "resident.bin.quantizing")

        let input = open(sourceURL.path, O_RDONLY)
        guard input >= 0 else {
            throw QuantizeError.openFailed("resident.bin", errno: errno)
        }
        defer { close(input) }

        let output = open(targetURL.path, O_RDWR | O_CREAT | O_TRUNC, 0o644)
        guard output >= 0 else {
            throw QuantizeError.openFailed("resident.bin.quantizing", errno: errno)
        }
        defer { close(output) }

        // Sized up front: the alignment gaps between tensors are then zeros we never write,
        // and a full disk fails here rather than halfway through.
        guard ftruncate(output, off_t(target.residentBytes)) == 0 else {
            throw QuantizeError.writeFailed(
                "resident.bin.quantizing", at: target.residentBytes, errno: errno)
        }

        var quantized = 0
        for destination in target.resident {
            guard let source = published.placement(of: destination.sourceName) else {
                throw QuantizeError.tensorMissing(destination.sourceName)
            }
            if destination.precision == .bf16 {
                try copy(
                    from: input, at: source.offset, count: source.byteCount,
                    to: output, at: destination.offset)
            } else {
                try quantize(from: input, source: source, to: output, destination: destination)
                quantized += 1
            }
        }

        // The manifest will vouch for these bytes; they must be on the platter first.
        guard fcntl(output, F_FULLFSYNC) != -1 else {
            throw QuantizeError.writeFailed("resident.bin.quantizing", at: 0, errno: errno)
        }
        try FileManager.default.removeItem(at: sourceURL)
        try FileManager.default.moveItem(at: targetURL, to: sourceURL)

        return Report(
            tensorsQuantized: quantized,
            bytesBefore: published.residentBytes, bytesAfter: target.residentBytes)
    }

    // MARK: - Byte movement

    private func copy(
        from input: Int32, at sourceOffset: Int, count: Int,
        to output: Int32, at destinationOffset: Int
    ) throws {
        var moved = 0
        var buffer = [UInt8](repeating: 0, count: min(count, Self.valuesPerChunk * 2))
        while moved < count {
            let slice = min(buffer.count, count - moved)
            try buffer.withUnsafeMutableBytes { raw in
                let got = pread(input, raw.baseAddress, slice, off_t(sourceOffset + moved))
                guard got >= 0 else {
                    throw QuantizeError.readFailed(
                        "resident.bin", at: sourceOffset + moved, errno: errno)
                }
                guard got == slice else {
                    throw QuantizeError.shortRead(
                        "resident.bin", at: sourceOffset + moved, expected: slice, got: got)
                }
                let put = pwrite(output, raw.baseAddress, slice, off_t(destinationOffset + moved))
                guard put == slice else {
                    throw QuantizeError.writeFailed(
                        "resident.bin.quantizing", at: destinationOffset + moved, errno: errno)
                }
            }
            moved += slice
        }
    }

    /// Converts one tensor from BF16 to Q8.
    ///
    /// Blocks are taken **flat**, not row by row. Every dimension of GPT-OSS has a column
    /// count that is a multiple of 32, so a block never straddles two rows and the flat walk
    /// is exactly the row-wise one — without this pass needing to know a tensor's shape.
    /// `HydraLayout` refuses to quantize anything whose value count is not a whole number of
    /// blocks, which is what makes that assumption safe here.
    private func quantize(
        from input: Int32, source: HydraLayout.TensorPlacement,
        to output: Int32, destination: HydraLayout.TensorPlacement
    ) throws {
        let values = source.byteCount / 2
        var done = 0

        var words = [UInt16](repeating: 0, count: min(values, Self.valuesPerChunk))
        while done < values {
            let count = min(words.count, values - done)
            let sourceOffset = source.offset + done * 2

            try words.withUnsafeMutableBytes { raw in
                let got = pread(input, raw.baseAddress, count * 2, off_t(sourceOffset))
                guard got >= 0 else {
                    throw QuantizeError.readFailed("resident.bin", at: sourceOffset, errno: errno)
                }
                guard got == count * 2 else {
                    throw QuantizeError.shortRead(
                        "resident.bin", at: sourceOffset, expected: count * 2, got: got)
                }
            }

            var floats = [Float](repeating: 0, count: count)
            for i in 0..<count { floats[i] = BF16.toFloat(words[i]) }
            let encoded = try Q8.encode(floats)

            try write(
                encoded.levels, to: output,
                at: destination.offset + done, what: "levels")
            try write(
                encoded.scales, to: output,
                at: destination.scaleOffset + done / Q8.blockSize * 2, what: "scales")

            done += count
        }
    }

    private func write(_ data: Data, to output: Int32, at offset: Int, what: String) throws {
        try data.withUnsafeBytes { raw in
            let put = pwrite(output, raw.baseAddress, raw.count, off_t(offset))
            guard put == raw.count else {
                throw QuantizeError.writeFailed(
                    "resident.bin.quantizing (\(what))", at: offset, errno: errno)
            }
        }
    }
}
