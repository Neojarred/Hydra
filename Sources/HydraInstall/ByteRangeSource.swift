import Foundation

/// Source d'octets adressable par plages.
///
/// L'interface ne sait **que** lire une plage bornée. Elle n'expose aucun moyen de
/// récupérer un fichier entier. C'est délibéré : l'invariant mémoire du projet est plus
/// facile à défendre quand l'outil n'a pas la capacité de le violer.
public protocol ByteRangeSource: Sendable {
    /// Lit exactement `range` d'un coup. Réservé aux **petites** lectures : en-têtes,
    /// index, métadonnées. Jamais pour des poids.
    func read(file: String, range: Range<Int>) async throws -> Data

    /// Diffuse `range` en remettant les octets par blocs, dans l'ordre, à mesure qu'ils
    /// arrivent. C'est le chemin des poids : la plage peut peser des centaines de mégaoctets
    /// sans que le tas ne dépasse la taille d'un bloc.
    ///
    /// L'implémentation garantit que `sink` est appelé **de façon sérialisée**.
    func stream(
        file: String, range: Range<Int>,
        into sink: @escaping @Sendable (Data) throws -> Void
    ) async throws

    /// Libellé pour les journaux et le manifeste.
    var sourceDescription: String { get }
}

/// Source locale, pour les tests et pour repacker un checkpoint déjà téléchargé.
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
                return "ouverture impossible de \(f) : \(String(cString: strerror(e)))"
            case let .shortRead(f, expected, got):
                return "lecture courte sur \(f) : \(got) octets, \(expected) attendus"
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

        // Bloc de lecture volontairement petit : il joue le rôle des blocs que livre
        // URLSession, pour que les tests exercent le même chemin de découpage.
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

/// Un dépôt Hugging Face précis, vu comme une source de plages.
public struct HuggingFaceSource: ByteRangeSource {

    public let client: HuggingFaceClient
    public let repo: String
    /// Une seule session pour toute l'installation : les connexions sont réutilisées
    /// d'une plage à l'autre, ce qui évite de repayer la poignée TLS.
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
