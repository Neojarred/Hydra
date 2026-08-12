import Foundation
import HydraFormat

/// A Hugging Face repository's `model.safetensors.index.json`.
public struct SafetensorsIndex: Sendable {
    /// Tensor name → the shard file that contains it.
    public let weightMap: [String: String]
    /// `metadata.total_size`, when present.
    public let totalSize: Int?

    public var shards: Set<String> { Set(weightMap.values) }

    public init(weightMap: [String: String], totalSize: Int?) {
        self.weightMap = weightMap
        self.totalSize = totalSize
    }

    public init(json: Data) throws {
        guard let root = try JSONSerialization.jsonObject(with: json) as? [String: Any],
            let map = root["weight_map"] as? [String: String]
        else {
            throw HuggingFaceClient.ClientError.malformedIndex
        }
        self.weightMap = map
        self.totalSize = (root["metadata"] as? [String: Any])?["total_size"] as? Int
    }
}

/// Read-only access to a Hugging Face repository, by **bounded byte ranges**.
///
/// This client never downloads a whole file. It can do only two things: read a small JSON
/// file, and request a range. That is deliberate, the project's memory invariant is easier
/// to defend if the tool has no ability to violate it.
public struct HuggingFaceClient: Sendable {

    public let endpoint: URL
    public let revision: String
    private let session: URLSession

    public init(
        endpoint: URL = URL(string: "https://huggingface.co")!,
        revision: String = "main",
        session: URLSession = .shared
    ) {
        self.endpoint = endpoint
        self.revision = revision
        self.session = session
    }

    public enum ClientError: Error, CustomStringConvertible {
        case badStatus(Int, url: String)
        case rangeNotHonored(url: String)
        case malformedIndex
        case shortRead(expected: Int, got: Int)

        public var description: String {
            switch self {
            case let .badStatus(code, url):
                return "HTTP \(code) sur \(url)"
            case .rangeNotHonored(let url):
                return "the server ignored the Range request on \(url), unbounded read refused"
            case .malformedIndex:
                return "index safetensors illisible"
            case let .shortRead(e, g):
                return "short read: \(g) bytes received, \(e) expected"
            }
        }
    }

    public func fileURL(repo: String, file: String) -> URL {
        endpoint.appending(path: "\(repo)/resolve/\(revision)/\(file)")
    }

    /// Reads a byte range. The result is **bounded by construction**: if the server ignores the
    /// `Range` header and answers 200, we refuse the response rather than ingest the whole
    /// file.
    public func fetchRange(repo: String, file: String, range: Range<Int>) async throws -> Data {
        let url = fileURL(repo: repo, file: file)
        var request = URLRequest(url: url)
        request.setValue("bytes=\(range.lowerBound)-\(range.upperBound - 1)", forHTTPHeaderField: "Range")
        request.setValue("hydra/0.1", forHTTPHeaderField: "User-Agent")

        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw ClientError.badStatus(-1, url: url.absoluteString)
        }
        guard http.statusCode == 206 else {
            if http.statusCode == 200 { throw ClientError.rangeNotHonored(url: url.absoluteString) }
            throw ClientError.badStatus(http.statusCode, url: url.absoluteString)
        }
        guard data.count == range.count else {
            throw ClientError.shortRead(expected: range.count, got: data.count)
        }
        return data
    }

    /// Reads a small file in full. Reserved for metadata JSON and the tokenizer, never for
    /// weights.
    public func fetchSmallFile(repo: String, file: String) async throws -> Data {
        let url = fileURL(repo: repo, file: file)
        var request = URLRequest(url: url)
        request.setValue("hydra/0.1", forHTTPHeaderField: "User-Agent")
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw ClientError.badStatus(
                (response as? HTTPURLResponse)?.statusCode ?? -1, url: url.absoluteString)
        }
        return data
    }

    public func fetchIndex(repo: String) async throws -> SafetensorsIndex {
        let data = try await fetchSmallFile(repo: repo, file: "model.safetensors.index.json")
        return try SafetensorsIndex(json: data)
    }

    /// Reads a shard's header in two bounded requests: 8 bytes to learn the header's size, then
    /// exactly that size. The data is never touched.
    public func fetchHeader(repo: String, file: String) async throws -> SafetensorsHeader {
        let prefix = try await fetchRange(repo: repo, file: file, range: 0..<8)
        let length = try SafetensorsHeader.headerLength(fromPrefix: prefix)
        let json = try await fetchRange(repo: repo, file: file, range: 8..<(8 + length))
        return try SafetensorsHeader(headerJSON: json, headerByteCount: length)
    }
}
