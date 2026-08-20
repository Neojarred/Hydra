import Foundation
import HydraCore
import HydraFormat
import HydraMetal
import Metal

/// Access to Gemma 4's vision tower inside `vision.bin`.
///
/// Same job as `VisionMapping` does for Qwen and a separate type rather than a shared one,
/// because almost nothing about the two towers lines up: different tensor names, different
/// shapes, a different count per layer, and one tensor that is quantized where every other is
/// BF16. A single mapping parameterized over both would be a switch statement at every call.
public final class Gemma4VisionMapping: @unchecked Sendable {

    public let config: Gemma4VisionConfig
    public let file: MappedFile
    private let tensors: [String: HydraManifest.VisionTensor]

    public enum MappingError: Error, CustomStringConvertible {
        case noTower(URL)
        case tensorMissing(String)
        case wrongShape(String, expected: [Int], found: [Int])

        public var description: String {
            switch self {
            case .noTower(let root):
                return "no vision tower installed at \(root.lastPathComponent)"
            case .tensorMissing(let name):
                return "the vision tower has no tensor named \(name)"
            case let .wrongShape(name, expected, found):
                return "\(name) is \(found), the config says \(expected)"
            }
        }
    }

    public init(root: URL, config: Gemma4VisionConfig = .a4b, device: MTLDevice) throws {
        let manifest = try HydraManifest.read(from: root)
        guard let listed = manifest.vision, !listed.isEmpty else {
            throw MappingError.noTower(root)
        }
        self.config = config
        self.tensors = Dictionary(uniqueKeysWithValues: listed.map { ($0.name, $0) })
        self.file = try MappedFile(url: root.appending(path: "vision.bin"), device: device)
        try validate()
    }

    /// The tower's tensor names.
    ///
    /// Gemma names its blocks `encoder.layers.N` where Qwen names them `blocks.N`, and splits the
    /// attention projection into four tensors where Qwen fuses three into one. Spelled out so a
    /// misspelling fails once, at load, rather than at layer nineteen.
    public enum Name {
        public static func inputNorm(_ l: Int) -> String { "vision_tower.encoder.layers.\(l).input_layernorm.weight" }
        public static func postAttentionNorm(_ l: Int) -> String { "vision_tower.encoder.layers.\(l).post_attention_layernorm.weight" }
        public static func preFeedforwardNorm(_ l: Int) -> String { "vision_tower.encoder.layers.\(l).pre_feedforward_layernorm.weight" }
        public static func postFeedforwardNorm(_ l: Int) -> String { "vision_tower.encoder.layers.\(l).post_feedforward_layernorm.weight" }
        public static func query(_ l: Int) -> String { "vision_tower.encoder.layers.\(l).self_attn.q_proj.linear.weight" }
        public static func key(_ l: Int) -> String { "vision_tower.encoder.layers.\(l).self_attn.k_proj.linear.weight" }
        public static func value(_ l: Int) -> String { "vision_tower.encoder.layers.\(l).self_attn.v_proj.linear.weight" }
        public static func output(_ l: Int) -> String { "vision_tower.encoder.layers.\(l).self_attn.o_proj.linear.weight" }
        public static func queryNorm(_ l: Int) -> String { "vision_tower.encoder.layers.\(l).self_attn.q_norm.weight" }
        public static func keyNorm(_ l: Int) -> String { "vision_tower.encoder.layers.\(l).self_attn.k_norm.weight" }
        public static func gate(_ l: Int) -> String { "vision_tower.encoder.layers.\(l).mlp.gate_proj.linear.weight" }
        public static func up(_ l: Int) -> String { "vision_tower.encoder.layers.\(l).mlp.up_proj.linear.weight" }
        public static func down(_ l: Int) -> String { "vision_tower.encoder.layers.\(l).mlp.down_proj.linear.weight" }

        public static let patchProjection = "vision_tower.patch_embedder.input_proj.weight"
        public static let positionTable = "vision_tower.patch_embedder.position_embedding_table"
        public static let standardizationScale = "vision_tower.std_scale"
        public static let standardizationBias = "vision_tower.std_bias"

        /// The projector into the text model, and the only quantized tensor in the tower.
        public static let projectionWeight = "embed_vision.embedding_projection.weight"
        public static let projectionScales = "embed_vision.embedding_projection.scales"
        public static let projectionBiases = "embed_vision.embedding_projection.biases"
    }

    public func placement(_ name: String) throws -> HydraManifest.VisionTensor {
        guard let tensor = tensors[name] else { throw MappingError.tensorMissing(name) }
        return tensor
    }

    public func tensor(_ name: String) throws -> (buffer: MTLBuffer, offset: Int, byteCount: Int) {
        let placement = try placement(name)
        return (file.buffer, placement.offset, placement.byteCount)
    }

    /// A BF16 tensor decoded to float, for the oracle and for inspection. Never on a hot path.
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

    private func validate() throws {
        let hidden = config.hiddenSize
        let inter = config.intermediateSize

        var expected: [(String, [Int])] = [
            (Name.patchProjection, [hidden, config.patchElements]),
            (Name.positionTable, [2, config.positionEmbeddingSize, hidden]),
            (Name.standardizationScale, [hidden]),
            (Name.standardizationBias, [hidden]),
        ]
        for layer in 0..<config.depth {
            expected += [
                (Name.inputNorm(layer), [hidden]),
                (Name.postAttentionNorm(layer), [hidden]),
                (Name.preFeedforwardNorm(layer), [hidden]),
                (Name.postFeedforwardNorm(layer), [hidden]),
                (Name.query(layer), [hidden, hidden]),
                (Name.key(layer), [hidden, hidden]),
                (Name.value(layer), [hidden, hidden]),
                (Name.output(layer), [hidden, hidden]),
                (Name.queryNorm(layer), [config.headDim]),
                (Name.keyNorm(layer), [config.headDim]),
                (Name.gate(layer), [inter, hidden]),
                (Name.up(layer), [inter, hidden]),
                (Name.down(layer), [hidden, inter]),
            ]
        }
        for (name, shape) in expected {
            let tensor = try placement(name)
            guard tensor.shape == shape else {
                throw MappingError.wrongShape(name, expected: shape, found: tensor.shape)
            }
        }
        // The projector alone is quantized, so its shapes are the packed ones and are checked
        // separately rather than being made to fit the loop above.
        let weight = try placement(Name.projectionWeight)
        guard weight.shape.first == config.outHiddenSize else {
            throw MappingError.wrongShape(
                Name.projectionWeight, expected: [config.outHiddenSize, 0], found: weight.shape)
        }
    }

    public var byteCount: Int { tensors.values.reduce(0) { $0 + $1.byteCount } }
}
