import Foundation
import HydraCore
import HydraInstall

/// A catalogue **enumerated**, not open.
///
/// Accepting an arbitrary Hugging Face repository would open the door to architectures the
/// runtime cannot execute, Gated DeltaNet, dense attention, other quantization formats.
/// They would then have to be detected and refused cleanly, for no benefit: Hydra is a
/// streaming MoE engine and owns that (D-009, D-016).
///
/// What Gemma changed is the *type* an entry holds. It was a `GptOssConfig`, which made this a
/// list of one architecture's variants with names attached. It is now `any ModelDescriptor`,
/// so an entry describes a model, and the two factories, `RepackPlanFactory` and
/// `ModelRuntime`, turn it into the right plan and the right runner. Adding a model is a row
/// here plus a case in each of those two switches, and nothing else (D-023).
public struct CatalogEntry: Identifiable, Sendable, Equatable {
    public static func == (a: CatalogEntry, b: CatalogEntry) -> Bool { a.id == b.id }

    public let id: String
    public let repository: String
    public let displayName: String
    public let model: any ModelDescriptor
    public let summary: String

    public var installedBytes: Int { model.installedBytes }
    public var architecture: ModelArchitecture { model.architecture }

    public init(
        id: String, repository: String, displayName: String,
        model: any ModelDescriptor, summary: String
    ) {
        self.id = id
        self.repository = repository
        self.displayName = displayName
        self.model = model
        self.summary = summary
    }

    public static let all: [CatalogEntry] = [
        CatalogEntry(
            id: "gpt-oss-20b",
            repository: "openai/gpt-oss-20b",
            displayName: "GPT-OSS 20B",
            model: GptOssConfig.b20,
            summary: "24 layers, 32 experts per layer, 4 active per token."),
        CatalogEntry(
            id: "gpt-oss-120b",
            repository: "openai/gpt-oss-120b",
            displayName: "GPT-OSS 120B",
            model: GptOssConfig.b120,
            summary: "36 layers, 128 experts per layer, 4 active per token."),
        // The **instruction-tuned** repository, which is what `Gemma4Prompt` was transcribed
        // from. The base model tokenizes identically and answers nothing, so the distinction
        // is not cosmetic.
        CatalogEntry(
            id: "gemma-4-26b-a4b",
            repository: "google/gemma-4-26B-A4B-it",
            displayName: "Gemma 4 26B-A4B",
            model: Gemma4Config.a4b,
            summary: "30 layers, 128 experts per layer, 8 active per token, BF16 experts."),
        // The same architecture at 4 bits. Not Google's own conversion, there is no
        // first-party MLX release, but a conversion of their QAT weights, which D-024
        // establishes by measurement rather than by the repository's name.
        CatalogEntry(
            id: "gemma-4-26b-a4b-q4",
            repository: "lmstudio-community/gemma-4-26B-A4B-it-QAT-MLX-4bit",
            displayName: "Gemma 4 26B-A4B (Q4)",
            model: Gemma4MLXConfig.a4b,
            summary: "The same model at 4 bits: a third of the install, a third of the "
                + "bytes read per token."),
    ]

    public static func entry(id: String) -> CatalogEntry? { all.first { $0.id == id } }
}

/// A model's state on this machine.
public enum InstallationState: Sendable, Equatable {
    case absent
    case partial
    case installed(bytes: Int)
    case installing(fraction: Double, throughput: Double)

    public var isInstalled: Bool {
        if case .installed = self { return true }
        return false
    }
    public var isInstalling: Bool {
        if case .installing = self { return true }
        return false
    }
}

/// Installation locations and inspection.
public enum ModelLocations {

    public static func directory() throws -> URL {
        let base = try FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask,
            appropriateFor: nil, create: true)
        let models = base.appending(path: "Hydra/Models")
        try FileManager.default.createDirectory(at: models, withIntermediateDirectories: true)
        return models
    }

    public static func root(for entry: CatalogEntry) throws -> URL {
        try directory().appending(path: "\(entry.id).hydra")
    }

    public static func state(of entry: CatalogEntry) -> InstallationState {
        guard let root = try? root(for: entry) else { return .absent }
        let manager = FileManager.default
        if manager.fileExists(atPath: root.appending(path: "manifest.json").path),
            TokenizerInstaller.isInstalled(at: root)
        {
            return .installed(bytes: measure(root))
        }
        if manager.fileExists(atPath: root.appendingPathExtension("partial").path) {
            return .partial
        }
        return .absent
    }

    /// Free disk space, for the 10 GB warning (D-004).
    public static func availableBytes() -> Int? {
        guard let directory = try? directory(),
            let values = try? directory.resourceValues(
                forKeys: [.volumeAvailableCapacityForImportantUsageKey]),
            let free = values.volumeAvailableCapacityForImportantUsage
        else { return nil }
        return Int(free)
    }

    private static func measure(_ root: URL) -> Int {
        guard let enumerator = FileManager.default.enumerator(
            at: root, includingPropertiesForKeys: [.fileSizeKey])
        else { return 0 }
        var total = 0
        for case let url as URL in enumerator {
            total += (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        }
        return total
    }
}
