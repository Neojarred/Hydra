import Foundation

/// A byte source addressable by ranges.
///
/// The interface can **only** read a bounded range. It exposes no way to fetch a whole
/// file. That is deliberate: the project's memory invariant is easier to defend when the
/// tool has no ability to violate it.
public protocol ByteRangeSource: Sendable {
    /// Reads exactly `range` in one go. Reserved for **small** reads: headers, indexes,
    /// metadata. Never for weights.
    func read(file: String, range: Range<Int>) async throws -> Data

    /// Streams `range`, handing bytes back in blocks, in order, as they arrive. This is the
    /// weights path: the range may weigh hundreds of megabytes without the heap exceeding one
    /// block's size.
    ///
    /// The implementation guarantees `sink` is called **serially**.
    func stream(
        file: String, range: Range<Int>,
        into sink: @escaping @Sendable (Data) throws -> Void
    ) async throws

    /// A label for logs and the manifest.
    var sourceDescription: String { get }
}

/// A local source, for tests and for repacking an already-downloaded checkpoint.
public struct LocalDirectorySource: ByteRangeSource {

    public let root: URL

    public init(root: URL) {
        self.root = root
    }

    public var sourceDescription: String { "local:\(root.path)" }

    public enum SourceError: Error, CustomStringConvertible {
        case openFailed(String, errno: Int32)
        case shortRead(String, expected: Int, got: Int)

        public var description: String {
            switch self {
            case let .openFailed(f, e):
                return "cannot open \(f): \(String(cString: strerror(e)))"
            case let .shortRead(f, expected, got):
                return "short read on \(f): \(got) bytes, \(expected) expected"
            }
        }
    }

    public func stream(
        file: String, range: Range<Int>,
        into sink: @escaping @Sendable (Data) throws -> Void
    ) async throws {
        let path = root.appending(path: file).path
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else { throw SourceError.openFailed(file, errno: errno) }
        defer { close(fd) }

        // A deliberately small read block: it plays the role of the blocks URLSession
        // delivers, so the tests exercise the same chunking path.
        let blockSize = 64 * 1024
        var position = range.lowerBound
        var buffer = Data(count: blockSize)

        while position < range.upperBound {
            let want = min(blockSize, range.upperBound - position)
            let got: Int = buffer.withUnsafeMutableBytes { raw in
                var total = 0
                while total < want {
                    let n = pread(
                        fd, raw.baseAddress!.advanced(by: total),
                        want - total, off_t(position + total))
                    if n <= 0 { break }
                    total += n
                }
                return total
            }
            guard got == want else {
                throw SourceError.shortRead(file, expected: want, got: got)
            }
            try sink(buffer.prefix(want))
            position += want
        }
    }

    public func read(file: String, range: Range<Int>) async throws -> Data {
        let path = root.appending(path: file).path
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else { throw SourceError.openFailed(file, errno: errno) }
        defer { close(fd) }

        var buffer = Data(count: range.count)
        let got: Int = buffer.withUnsafeMutableBytes { raw in
            var total = 0
            while total < range.count {
                let n = pread(
                    fd, raw.baseAddress!.advanced(by: total),
                    range.count - total, off_t(range.lowerBound + total))
                if n <= 0 { break }
                total += n
            }
            return total
        }
        guard got == range.count else {
            throw SourceError.shortRead(file, expected: range.count, got: got)
        }
        return buffer
    }
}

/// One specific Hugging Face repository, seen as a range source.
public struct HuggingFaceSource: ByteRangeSource {

    public let client: HuggingFaceClient
    public let repo: String
    /// A single session for the whole install: connections are reused from one range to the
    /// next, which avoids paying for the TLS handshake again.
    private let streaming: StreamingHTTPClient

    public init(client: HuggingFaceClient = HuggingFaceClient(), repo: String) {
        self.client = client
        self.repo = repo
        self.streaming = StreamingHTTPClient()
    }

    public var sourceDescription: String {
        "\(client.endpoint.absoluteString)/\(repo)@\(client.revision)"
    }

    public func read(file: String, range: Range<Int>) async throws -> Data {
        try await client.fetchRange(repo: repo, file: file, range: range)
    }

    public func stream(
        file: String, range: Range<Int>,
        into sink: @escaping @Sendable (Data) throws -> Void
    ) async throws {
        try await streaming.stream(
            url: client.fileURL(repo: repo, file: file),
            range: range, userAgent: "hydra/0.1", sink: sink)
    }
}
