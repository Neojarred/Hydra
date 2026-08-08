import Foundation
import HydraCore
import HydraInstall

/// A catalogue **frozen** to OpenAI's two official models.
///
/// Accepting an arbitrary Hugging Face repository would open the door to architectures the
/// runtime cannot execute — Gated DeltaNet, dense attention, other quantization formats.
/// They would then have to be detected and refused cleanly, for no benefit: Hydra is a
/// streaming MoE engine and owns that (D-009, D-016).
public struct CatalogEntry: Identifiable, Sendable, Equatable {
    public static func == (a: CatalogEntry, b: CatalogEntry) -> Bool { a.id == b.id }

    public let id: String
    public let repository: String
    public let displayName: String
    public let config: GptOssConfig
    public let summary: String

    public var installedBytes: Int { config.installedBytes }

    public static let all: [CatalogEntry] = [
        CatalogEntry(
            id: "gpt-oss-20b",
            repository: "openai/gpt-oss-20b",
            displayName: "GPT-OSS 20B",
            config: .b20,
            summary: "24 layers, 32 experts per layer, 4 active per token."),
        CatalogEntry(
            id: "gpt-oss-120b",
            repository: "openai/gpt-oss-120b",
            displayName: "GPT-OSS 120B",
            config: .b120,
            summary: "36 layers, 128 experts per layer, 4 active per token."),
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
