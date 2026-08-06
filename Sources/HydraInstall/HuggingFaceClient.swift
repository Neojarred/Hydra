import Foundation
import HydraFormat

/// Index `model.safetensors.index.json` d'un dépôt Hugging Face.
public struct SafetensorsIndex: Sendable {
    /// Nom de tenseur → fichier shard qui le contient.
    public let weightMap: [String: String]
    /// `metadata.total_size`, quand il est présent.
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

/// Accès en lecture seule à un dépôt Hugging Face, par **plages d'octets bornées**.
///
/// Ce client ne télécharge jamais un fichier entier. Il ne sait faire que deux choses :
/// lire un petit fichier JSON, et demander une plage. C'est délibéré — l'invariant mémoire
/// du projet est plus facile à défendre si l'outil n'a pas la capacité de le violer.
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
                return "le serveur a ignoré la requête Range sur \(url) — lecture non bornée refusée"
            case .malformedIndex:
                return "index safetensors illisible"
            case let .shortRead(e, g):
                return "lecture courte : \(g) octets reçus, \(e) attendus"
            }
        }
    }

    public func fileURL(repo: String, file: String) -> URL {
        endpoint.appending(path: "\(repo)/resolve/\(revision)/\(file)")
    }

    /// Lit une plage d'octets. Le résultat est **borné par construction** : si le serveur
    /// ignore l'en-tête `Range` et répond 200, on refuse la réponse au lieu d'ingérer le
    /// fichier entier.
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

    /// Lit un petit fichier en entier. Réservé aux JSON de métadonnées et au tokenizer —
    /// jamais aux poids.
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

    /// Lit l'en-tête d'un shard en deux requêtes bornées : 8 octets pour connaître la
    /// taille de l'en-tête, puis exactement cette taille. Les données ne sont pas touchées.
    public func fetchHeader(repo: String, file: String) async throws -> SafetensorsHeader {
        let prefix = try await fetchRange(repo: repo, file: file, range: 0..<8)
        let length = try SafetensorsHeader.headerLength(fromPrefix: prefix)
        let json = try await fetchRange(repo: repo, file: file, range: 8..<(8 + length))
        return try SafetensorsHeader(headerJSON: json, headerByteCount: length)
    }
}
