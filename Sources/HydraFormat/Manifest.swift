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
        public let intermediateSize: Int
        public let vocabSize: Int
        public let slidingWindow: Int
        /// The experts' quantization format. An unexpected value is rejected.
        public let expertQuantization: String

        public init(config: GptOssConfig) {
            self.name = config.name
            self.layerCount = config.layerCount
            self.expertCount = config.expertCount
            self.expertsPerToken = config.expertsPerToken
            self.hiddenSize = config.hiddenSize
            self.intermediateSize = config.intermediateSize
            self.vocabSize = config.vocabSize
            self.slidingWindow = config.slidingWindow
            self.expertQuantization = "mxfp4"
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

    /// The precision the dense weights are stored at, when it is not what was published.
    ///
    /// **Optional on purpose.** Installations made before this existed have no such field,
    /// and must keep loading: absent means `bf16`, which is what they are. Making it
    /// required would have turned a new option into a forced reinstall.
    ///
    /// It sits at the top level rather than inside `Model` because `Model` is compared for
    /// equality against the expected architecture — a storage choice is not an architecture
    /// difference, and putting it there would make every Q8 installation look like the wrong
    /// model.
    public let densePrecision: String?

    /// What the runtime must lay the installation out with. This is the single place the
    /// on-disk choice becomes a decision the loader acts on.
    public var precisionPolicy: PrecisionPolicy {
        PrecisionPolicy(
            dense: densePrecision.flatMap(WeightPrecision.init(rawValue:)) ?? .bf16,
            sourceExperts: WeightPrecision(rawValue: model.expertQuantization) ?? .mxfp4)
    }

    public init(
        model: Model, layout: Layout, files: [String: FileEntry],
        sourceDescription: String, sourceTotalBytes: Int,
        tensors: [TensorDigest], densePrecision: WeightPrecision = .bf16,
        createdAt: Date = Date()
    ) {
        self.format = Self.currentFormat
        self.model = model
        self.layout = layout
        self.files = files
        self.sourceDescription = sourceDescription
        self.sourceTotalBytes = sourceTotalBytes
        self.tensors = tensors
        // Written only when it differs, so an ordinary installation's manifest is
        // byte-identical to what earlier versions produced.
        self.densePrecision = densePrecision == .bf16 ? nil : densePrecision.rawValue
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
    public func validate(against config: GptOssConfig, root: URL) throws {
        guard format == Self.currentFormat else { throw ValidationError.unknownFormat(format) }
        guard model.expertQuantization == "mxfp4" else {
            throw ValidationError.unsupportedQuantization(model.expertQuantization)
        }
        // An unreadable value here would silently lay the file out as BF16 and read
        // quantized bytes as floats — plausible output, entirely wrong.
        if let densePrecision, WeightPrecision(rawValue: densePrecision) == nil {
            throw ValidationError.unsupportedQuantization("dense: \(densePrecision)")
        }
        let expected = Model(config: config)
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
