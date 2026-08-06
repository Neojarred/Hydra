import Foundation
import HydraTokenize

/// Fetches the tokenizer files and places them in the `.hydra` installation.
///
/// The original `tokenizer.json` weighs 27 MB and takes several seconds of JSON parsing on
/// every launch. So we **compile** it once, at install time, into a compact binary format —
/// exactly the logic applied to the weights: the original format is for exchange, not for
/// execution.
///
/// The original JSON is kept alongside: it allows recompiling after a format change without
/// re-downloading, and comparing in case of doubt.
public struct TokenizerInstaller: Sendable {

    public let client: HuggingFaceClient
    public let repo: String

    public init(client: HuggingFaceClient = HuggingFaceClient(), repo: String) {
        self.client = client
        self.repo = repo
    }

    /// The files fetched. `tokenizer.json` is the only indispensable one; the others are
    /// metadata useful for rendering conversations.
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
                return "indispensable tokenizer file missing from the repository: \(name)"
            }
        }
    }

    /// Installs into `<root>/tokenizer/`, then compiles the compact format.
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
                // Optional files may be missing depending on the repository; only
                // tokenizer.json blocks the install.
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

    /// Loads an installation's tokenizer, preferring the compact format.
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
