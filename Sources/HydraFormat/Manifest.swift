import Foundation
import HydraCore

/// `manifest.json` — what marks an installation as complete.
///
/// Without a manifest, the runtime treats an installation as partial and refuses to load
/// it. The manifest is written only after all data has been made durable; its presence
/// alone is therefore a guarantee, not a hint.
public struct HydraManifest: Codable, Sendable, Equatable {

    /// The on-disk format version. Any unknown value is rejected rather than
    /// interpreted as best it can be.
    public static let currentFormat = "hydra/1"

    public struct Model: Codable, Sendable, Equatable {
        public let name: String
        public let layerCount: Int
        public let expertCount: Int
        public let expertsPerToken: Int
        public let hiddenSize: Int
        public let intermediateSize: Int?
        public let vocabSize: Int
        public let slidingWindow: Int
        /// How the experts are stored, as the source checkpoint publishes them. Compared
        /// against what the model declares rather than against a constant, so a new format is
        /// a new value and not a new branch (D-023).
        public let expertQuantization: String
        /// Which family this installation belongs to.
        ///
        /// **Optional for a reason**: installations written before Hydra had more than one
        /// architecture carry no such field, and must keep loading. Absent means GPT-OSS,
        /// which is what they are.
        public let architecture: String?

        init(copying other: Model, architecture: String?, intermediateSize: Int?) {
            self.name = other.name
            self.layerCount = other.layerCount
            self.expertCount = other.expertCount
            self.expertsPerToken = other.expertsPerToken
            self.hiddenSize = other.hiddenSize
            self.intermediateSize = intermediateSize
            self.vocabSize = other.vocabSize
            self.slidingWindow = other.slidingWindow
            self.expertQuantization = other.expertQuantization
            self.architecture = architecture
        }

        public init(model: any ModelDescriptor) {
            self.name = model.name
            self.layerCount = model.layerCount
            self.expertCount = model.expertCount
            self.expertsPerToken = model.expertsPerToken
            self.hiddenSize = model.hiddenSize
            self.intermediateSize = nil
            self.vocabSize = model.vocabSize
            self.slidingWindow = model.slidingWindow
            self.expertQuantization = model.expertFormat
            self.architecture = model.architecture.rawValue
        }
    }

    public struct Layout: Codable, Sendable, Equatable {
        public let expertStrideBytes: Int
        public let expertPayloadBytes: Int
        public let residentBytes: Int
        public let embeddingBytes: Int
        public let tensorAlignment: Int
        public let pageAlignment: Int

        public init(
            expertStrideBytes: Int, expertPayloadBytes: Int, residentBytes: Int,
            embeddingBytes: Int, tensorAlignment: Int, pageAlignment: Int
        ) {
            self.expertStrideBytes = expertStrideBytes
            self.expertPayloadBytes = expertPayloadBytes
            self.residentBytes = residentBytes
            self.embeddingBytes = embeddingBytes
            self.tensorAlignment = tensorAlignment
            self.pageAlignment = pageAlignment
        }
    }

    public struct FileEntry: Codable, Sendable, Equatable {
        public let byteCount: Int

        public init(byteCount: Int) { self.byteCount = byteCount }
    }

    /// A digest of a tensor's **source** bytes, computed on the fly during the repack. It
    /// allows checking that a resume has not mixed two revisions of the upstream repository,
    /// without having to re-read the destination files.
    public struct TensorDigest: Codable, Sendable, Equatable {
        public let tensor: String
        public let byteCount: Int
        public let sha256: String

        public init(tensor: String, byteCount: Int, sha256: String) {
            self.tensor = tensor
            self.byteCount = byteCount
            self.sha256 = sha256
        }
    }

    public let format: String
    public let model: Model
    public let layout: Layout
    public let files: [String: FileEntry]
    public let sourceDescription: String
    public let sourceTotalBytes: Int
    public let tensors: [TensorDigest]
    public let createdAt: Date

    public init(
        model: Model, layout: Layout, files: [String: FileEntry],
        sourceDescription: String, sourceTotalBytes: Int,
        tensors: [TensorDigest], createdAt: Date = Date()
    ) {
        self.format = Self.currentFormat
        self.model = model
        self.layout = layout
        self.files = files
        self.sourceDescription = sourceDescription
        self.sourceTotalBytes = sourceTotalBytes
        self.tensors = tensors
        self.createdAt = createdAt
    }

    public enum ValidationError: Error, CustomStringConvertible {
        case unknownFormat(String)
        case unsupportedQuantization(String)
        case architectureMismatch(String)
        case fileMissing(String)
        case fileSizeMismatch(String, expected: Int, got: Int)

        public var description: String {
            switch self {
            case .unknownFormat(let f):
                return "unknown installation format \"\(f)\" — expected \(HydraManifest.currentFormat)"
            case .unsupportedQuantization(let q):
                return "unsupported expert quantization: \(q)"
            case .architectureMismatch(let d):
                return "the installation does not match the expected model: \(d)"
            case .fileMissing(let f):
                return "file missing from the installation: \(f)"
            case let .fileSizeMismatch(f, e, g):
                return "\(f) is \(g) bytes, \(e) expected — incomplete or corrupted installation"
            }
        }
    }

    /// Checks that the installation matches the requested configuration and that every
    /// announced file is present with the right size.
    ///
    /// This is a **structural** check, cheap, performed on every load. Fully hashing files of
    /// several gigabytes is a separate check, done lazily.
    ///
    public func validate(against descriptor: any ModelDescriptor, root: URL) throws {
        guard format == Self.currentFormat else { throw ValidationError.unknownFormat(format) }
        // Compared against what the model declares, never against a constant: a checkpoint
        // published in a new format is a new value here, not a new branch (D-023).
        guard model.expertQuantization == descriptor.expertFormat else {
            throw ValidationError.unsupportedQuantization(
                "\(model.expertQuantization), expected \(descriptor.expertFormat)")
        }
        // Absent means GPT-OSS: installations predate the field, and must keep loading.
        let installed = model.architecture ?? ModelArchitecture.gptOss.rawValue
        guard installed == descriptor.architecture.rawValue else {
            throw ValidationError.architectureMismatch(
                "installed \(installed), expected \(descriptor.architecture.rawValue)")
        }
        var expected = Model(model: descriptor)
        // Legacy manifests carry no architecture and may carry an intermediate size we no
        // longer record; neither is an architecture difference.
        expected = Model(
            copying: expected, architecture: model.architecture,
            intermediateSize: model.intermediateSize)
        guard model == expected else {
            throw ValidationError.architectureMismatch(
                "\(model.name) (\(model.layerCount) layers, \(model.expertCount) experts) "
                + "vs \(expected.name) (\(expected.layerCount), \(expected.expertCount))")
        }
        for (path, entry) in files {
            let url = root.appending(path: path)
            guard let attrs = try? FileManager.default.attributesOfItem(atPath: url.path),
                let size = attrs[.size] as? Int
            else {
                throw ValidationError.fileMissing(path)
            }
            guard size == entry.byteCount else {
                throw ValidationError.fileSizeMismatch(path, expected: entry.byteCount, got: size)
            }
        }
    }

    public static func read(from root: URL) throws -> HydraManifest {
        let data = try Data(contentsOf: root.appending(path: "manifest.json"))
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(HydraManifest.self, from: data)
    }

    public func write(to root: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(self).write(to: root.appending(path: "manifest.json"), options: .atomic)
    }
}
