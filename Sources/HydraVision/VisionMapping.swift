import Foundation
import HydraCore
import HydraFormat
import HydraMetal
import Metal

/// Access to the vision tower's tensors inside `vision.bin`.
///
/// The tower is the one part of an installation whose structure does not come from a
/// `ModelDescriptor`: the installer copied it verbatim and the manifest describes each tensor
/// with a name, an offset, a dtype and a shape. So resolution here goes through the manifest
/// rather than through `HydraLayout`, and the shapes are **checked against the config** rather
/// than assumed.
///
/// That check is the point of this type. Every tensor in the tower is BF16 and most of them are
/// square-ish, so a `qkv.weight` fetched where `proj.weight` was meant has the right dtype and a
/// plausible size, and produces finite nonsense. The names below are the ones the manifest
/// actually carries, listed in `QwenRepackPlan.visionPrefixes`.
public final class VisionMapping: @unchecked Sendable {

    public let config: Qwen35VisionConfig
    public let file: MappedFile
    /// Name to placement, from the manifest.
    private let tensors: [String: HydraManifest.VisionTensor]

    public enum MappingError: Error, CustomStringConvertible {
        case noTower(URL)
        case tensorMissing(String)
        case wrongShape(String, expected: [Int], found: [Int])
        case wrongType(String, expected: String, found: String)

        public var description: String {
            switch self {
            case .noTower(let root):
                return "no vision tower installed at \(root.lastPathComponent): the manifest "
                    + "carries no vision tensors, so this model was installed text-only"
            case .tensorMissing(let name):
                return "the vision tower has no tensor named \(name)"
            case let .wrongShape(name, expected, found):
                return "\(name) is \(found), the config says \(expected)"
            case let .wrongType(name, expected, found):
                return "\(name) is \(found), expected \(expected)"
            }
        }
    }

    public init(
        root: URL, config: Qwen35VisionConfig = .a3b, device: MTLDevice
    ) throws {
        let manifest = try HydraManifest.read(from: root)
        guard let listed = manifest.vision, !listed.isEmpty else {
            throw MappingError.noTower(root)
        }
        self.config = config
        self.tensors = Dictionary(uniqueKeysWithValues: listed.map { ($0.name, $0) })
        self.file = try MappedFile(
            url: root.appending(path: "vision.bin"), device: device)
        try validate()
    }

    // MARK: - Names

    /// The tower's tensor names, in one place.
    ///
    /// Spelled out rather than built from a format string at each use. A misspelling here fails
    /// once, loudly, in `validate()`; a misspelling at a use site fails at the moment a
    /// particular layer runs, which for layer 19 of 27 is a long way into a forward pass.
    public enum Name {
        public static func norm1Weight(_ layer: Int) -> String { "vision_tower.blocks.\(layer).norm1.weight" }
        public static func norm1Bias(_ layer: Int) -> String { "vision_tower.blocks.\(layer).norm1.bias" }
        public static func norm2Weight(_ layer: Int) -> String { "vision_tower.blocks.\(layer).norm2.weight" }
        public static func norm2Bias(_ layer: Int) -> String { "vision_tower.blocks.\(layer).norm2.bias" }
        public static func qkvWeight(_ layer: Int) -> String { "vision_tower.blocks.\(layer).attn.qkv.weight" }
        public static func qkvBias(_ layer: Int) -> String { "vision_tower.blocks.\(layer).attn.qkv.bias" }
        public static func projWeight(_ layer: Int) -> String { "vision_tower.blocks.\(layer).attn.proj.weight" }
        public static func projBias(_ layer: Int) -> String { "vision_tower.blocks.\(layer).attn.proj.bias" }
        public static func fc1Weight(_ layer: Int) -> String { "vision_tower.blocks.\(layer).mlp.linear_fc1.weight" }
        public static func fc1Bias(_ layer: Int) -> String { "vision_tower.blocks.\(layer).mlp.linear_fc1.bias" }
        public static func fc2Weight(_ layer: Int) -> String { "vision_tower.blocks.\(layer).mlp.linear_fc2.weight" }
        public static func fc2Bias(_ layer: Int) -> String { "vision_tower.blocks.\(layer).mlp.linear_fc2.bias" }

        public static let patchWeight = "vision_tower.patch_embed.proj.weight"
        public static let patchBias = "vision_tower.patch_embed.proj.bias"
        public static let positionEmbedding = "vision_tower.pos_embed.weight"

        public static let mergerNormWeight = "vision_tower.merger.norm.weight"
        public static let mergerNormBias = "vision_tower.merger.norm.bias"
        public static let mergerFC1Weight = "vision_tower.merger.linear_fc1.weight"
        public static let mergerFC1Bias = "vision_tower.merger.linear_fc1.bias"
        public static let mergerFC2Weight = "vision_tower.merger.linear_fc2.weight"
        public static let mergerFC2Bias = "vision_tower.merger.linear_fc2.bias"
    }

    // MARK: - Resolution

    public func placement(_ name: String) throws -> HydraManifest.VisionTensor {
        guard let tensor = tensors[name] else { throw MappingError.tensorMissing(name) }
        return tensor
    }

    /// A tensor as a GPU buffer range. The tower is BF16 throughout, so callers read `ushort`.
    public func tensor(_ name: String) throws -> (buffer: MTLBuffer, offset: Int, byteCount: Int) {
        let placement = try placement(name)
        return (file.buffer, placement.offset, placement.byteCount)
    }

    /// A tensor decoded to float on the CPU, for the reference implementation and for tests.
    ///
    /// Deliberately not on the decoding path: this allocates and converts the whole tensor, so
    /// it is for the oracle and for inspection, never for a forward pass.
    public func floats(_ name: String) throws -> [Float] {
        let placement = try placement(name)
        let count = placement.byteCount / 2
        var out = [Float](repeating: 0, count: count)
        file.withBytes { raw in
            for i in 0..<count {
                let bits = raw.loadUnaligned(
                    fromByteOffset: placement.offset + i * 2, as: UInt16.self)
                out[i] = BF16.toFloat(UInt16(littleEndian: bits))
            }
        }
        return out
    }

    // MARK: - Validation

    /// Every tensor the tower needs is present, BF16, and the shape the config implies.
    ///
    /// Run once at load. The alternative is discovering at layer 19 that a name was misspelled,
    /// or worse not discovering it: `norm1.weight` and `norm2.weight` are both `[1152]`, so
    /// swapping them is undetectable at every level except the answer.
    private func validate() throws {
        let hidden = config.hiddenSize
        let inter = config.intermediateSize
        let merged = config.mergedWidth

        var expected: [(String, [Int])] = [
            (Name.patchWeight,
             [hidden, config.temporalPatchSize, config.patchSize, config.patchSize,
              config.inChannels]),
            (Name.patchBias, [hidden]),
            (Name.positionEmbedding, [config.positionEmbeddingCount, hidden]),
            (Name.mergerNormWeight, [hidden]),
            (Name.mergerNormBias, [hidden]),
            (Name.mergerFC1Weight, [merged, merged]),
            (Name.mergerFC1Bias, [merged]),
            (Name.mergerFC2Weight, [config.outHiddenSize, merged]),
            (Name.mergerFC2Bias, [config.outHiddenSize]),
        ]
        for layer in 0..<config.depth {
            expected += [
                (Name.norm1Weight(layer), [hidden]), (Name.norm1Bias(layer), [hidden]),
                (Name.norm2Weight(layer), [hidden]), (Name.norm2Bias(layer), [hidden]),
                (Name.qkvWeight(layer), [3 * hidden, hidden]),
                (Name.qkvBias(layer), [3 * hidden]),
                (Name.projWeight(layer), [hidden, hidden]), (Name.projBias(layer), [hidden]),
                (Name.fc1Weight(layer), [inter, hidden]), (Name.fc1Bias(layer), [inter]),
                (Name.fc2Weight(layer), [hidden, inter]), (Name.fc2Bias(layer), [hidden]),
            ]
        }

        for (name, shape) in expected {
            let tensor = try placement(name)
            guard tensor.dtype == "BF16" else {
                throw MappingError.wrongType(name, expected: "BF16", found: tensor.dtype)
            }
            guard tensor.shape == shape else {
                throw MappingError.wrongShape(name, expected: shape, found: tensor.shape)
            }
            let elements = shape.reduce(1, *)
            guard tensor.byteCount == elements * 2 else {
                throw MappingError.wrongShape(
                    name, expected: [elements * 2], found: [tensor.byteCount])
            }
        }
    }

    /// Bytes the tower occupies, for the memory budget.
    public var byteCount: Int { tensors.values.reduce(0) { $0 + $1.byteCount } }
}
