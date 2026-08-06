import Foundation
import HydraTokenize

/// Récupère les fichiers du tokeniseur et les range dans l'installation `.hydra`.
///
/// Le `tokenizer.json` d'origine pèse 27 Mo et demande plusieurs secondes d'analyse JSON
/// à chaque lancement. On le **compile** donc une fois, à l'installation, vers un format
/// binaire compact — exactement la logique appliquée aux poids : le format d'origine sert
/// à l'échange, pas à l'exécution.
///
/// Le JSON d'origine est conservé à côté : il permet de recompiler après une évolution du
/// format sans retélécharger, et de comparer en cas de doute.
public struct TokenizerInstaller: Sendable {

    public let client: HuggingFaceClient
    public let repo: String

    public init(client: HuggingFaceClient = HuggingFaceClient(), repo: String) {
        self.client = client
        self.repo = repo
    }

    /// Fichiers récupérés. `tokenizer.json` est le seul indispensable ; les autres sont
    /// des métadonnées utiles au rendu de conversation.
    public static let files = [
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "chat_template.jinja", "generation_config.json",
    ]
    public static let requiredFiles = ["tokenizer.json"]

    public enum InstallError: Error, CustomStringConvertible {
        case requiredFileMissing(String)

        public var description: String {
            switch self {
            case .requiredFileMissing(let name):
                return "fichier de tokeniseur indispensable absent du dépôt : \(name)"
            }
        }
    }

    /// Installe dans `<root>/tokenizer/`, puis compile le format compact.
    @discardableResult
    public func install(into root: URL) async throws -> Int {
        let directory = root.appending(path: "tokenizer")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        var written = 0
        for name in Self.files {
            do {
                let data = try await client.fetchSmallFile(repo: repo, file: name)
                try data.write(to: directory.appending(path: name), options: .atomic)
                written += 1
            } catch {
                // Les fichiers facultatifs peuvent manquer selon les dépôts ; seul
                // tokenizer.json bloque l'installation.
                if Self.requiredFiles.contains(name) {
                    throw InstallError.requiredFileMissing(name)
                }
            }
        }

        try TokenizerFile.compile(
            jsonAt: directory.appending(path: "tokenizer.json"),
            to: directory.appending(path: TokenizerFile.compactFileName))
        return written
    }

    /// Charge le tokeniseur d'une installation, en préférant le format compact.
    public static func load(from root: URL) throws -> BPETokenizer {
        let directory = root.appending(path: "tokenizer")
        let compact = directory.appending(path: TokenizerFile.compactFileName)
        if FileManager.default.fileExists(atPath: compact.path) {
            return try TokenizerFile.loadCompact(at: compact)
        }
        return try TokenizerFile.parseJSON(at: directory.appending(path: "tokenizer.json"))
    }

    public static func isInstalled(at root: URL) -> Bool {
        FileManager.default.fileExists(
            atPath: root.appending(path: "tokenizer/\(TokenizerFile.compactFileName)").path)
    }
}
