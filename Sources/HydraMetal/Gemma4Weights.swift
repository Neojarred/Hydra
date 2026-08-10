import Foundation
import HydraCore
import HydraFormat
import Metal

/// Where a Gemma layer's weights live, and how they decode.
///
/// **Dispatch happens here, once, when the runner is built** — never inside a decoding step
/// (D-023). `Gemma4LayerRunner` asks this for a projection by role and gets back a value it can
/// encode; it never learns which checkpoint it is running.
///
/// Two conformers, because two checkpoints of the same architecture exist: the BF16 build,
/// whose tensors are plain matrices, and the MLX 4-bit build, whose every projection is a
/// triple of packed values, per-group scales and per-group biases. Everything else about the
/// forward pass — the norms, the residual structure, the router, the two attention geometries —
/// is identical and stays in one place.
public protocol Gemma4Weights: Sendable {

    /// A projection's weights, resolved.
    func projection(
        _ suffix: String, layer: Int, rows: Int, cols: Int
    ) throws -> ForwardEncoder.ProjectionSource

    /// An unquantized tensor — the norms, the router's scales, `layer_scalar`. BF16 in both
    /// checkpoints, because quantizing a per-channel scale saves nothing and costs accuracy.
    func plain(_ suffix: String, layer: Int) throws -> (buffer: MTLBuffer, offset: Int)

    /// The final norm, which has no layer.
    func finalNorm() throws -> (buffer: MTLBuffer, offset: Int)

    /// The tied embedding used as the output head.
    func head(rows: Int, cols: Int) throws -> ForwardEncoder.ProjectionSource

    /// One of an expert's three matrices, inside a blob the slot cache has made resident.
    ///
    /// `blobOffset` is where that blob starts in the layer's buffer. A layer's slots share one
    /// allocation, so this is non-zero for every slot but the first — and ignoring it reads
    /// slot zero's expert whichever one was asked for.
    func expert(
        _ part: Gemma4ExpertPart, blob: MTLBuffer, blobOffset: Int
    ) -> ForwardEncoder.ProjectionSource

    /// One row of the embedding table, dequantized where it needs to be.
    ///
    /// Gemma ties the embedding to the output head, so the same table is read one row at a time
    /// on the way in and in full on the way out. In the MLX build it is quantized like anything
    /// else — 4-bit, group 64 — which the CPU-side row read has to honour or every token starts
    /// from a vector of packed integers.
    func readEmbedding(token: Int, into destination: UnsafeMutableBufferPointer<Float>)
}

/// Which of an expert's three matrices is wanted.
///
/// Named rather than indexed because the BF16 build fuses gate and up into one tensor and the
/// MLX build keeps them apart: `gate` and `up` are two offsets into one matrix there and two
/// separate matrices here, and only a name survives that difference.
public enum Gemma4ExpertPart: Sendable {
    case gate
    case up
    case down
}

// MARK: - The BF16 build

public struct Gemma4BF16Weights: Gemma4Weights {

    private let config: Gemma4Config
    private let mapping: ModelMapping

    public init(config: Gemma4Config, mapping: ModelMapping) {
        self.config = config
        self.mapping = mapping
    }

    private func name(_ suffix: String, layer: Int) -> String {
        "model.language_model.layers.\(layer).\(suffix)"
    }

    public func projection(
        _ suffix: String, layer: Int, rows: Int, cols: Int
    ) throws -> ForwardEncoder.ProjectionSource {
        let tensor = try mapping.residentTensor(name("\(suffix).weight", layer: layer))
        return .bf16(buffer: tensor.buffer, offset: tensor.offset)
    }

    public func plain(_ suffix: String, layer: Int) throws -> (buffer: MTLBuffer, offset: Int) {
        let tensor = try mapping.residentTensor(name(suffix, layer: layer))
        return (tensor.buffer, tensor.offset)
    }

    public func finalNorm() throws -> (buffer: MTLBuffer, offset: Int) {
        let tensor = try mapping.residentTensor("model.language_model.norm.weight")
        return (tensor.buffer, tensor.offset)
    }

    public func head(rows: Int, cols: Int) throws -> ForwardEncoder.ProjectionSource {
        let tensor = try mapping.residentTensor("model.language_model.embed_tokens.weight")
        return .bf16(buffer: tensor.buffer, offset: tensor.offset)
    }

    public func readEmbedding(
        token: Int, into destination: UnsafeMutableBufferPointer<Float>
    ) {
        mapping.readEmbedding(token: token, into: destination)
    }

    public func expert(
        _ part: Gemma4ExpertPart, blob: MTLBuffer, blobOffset: Int
    ) -> ForwardEncoder.ProjectionSource {
        let layout = config.expertBlobLayout
        let inner = config.moeIntermediateSize * config.hiddenSize * 2
        switch part {
        // gate and up are the two halves of one fused `gate_up_proj`.
        case .gate: return .bf16(buffer: blob, offset: blobOffset + layout.gateUp.offset)
        case .up: return .bf16(buffer: blob, offset: blobOffset + layout.gateUp.offset + inner)
        case .down: return .bf16(buffer: blob, offset: blobOffset + layout.down.offset)
        }
    }
}

// MARK: - The MLX 4-bit build

public struct Gemma4MLXWeights: Gemma4Weights {

    private let config: Gemma4MLXConfig
    private let mapping: ModelMapping

    public init(config: Gemma4MLXConfig, mapping: ModelMapping) {
        self.config = config
        self.mapping = mapping
    }

    /// The bit width of a projection, read from the checkpoint's own per-tensor map.
    ///
    /// Not a constant: the dense MLP and the router are 8-bit and everything else is 4, so a
    /// single width would decode 120 tensors at half their intended shape from bytes that are
    /// all present (D-024).
    private func bits(for suffix: String) -> Int {
        suffix.hasPrefix("mlp.") || suffix.hasPrefix("router.proj")
            ? config.denseBits : config.quantBits
    }

    private func triple(
        _ stem: String, bits: Int
    ) throws -> ForwardEncoder.ProjectionSource {
        let words = try mapping.residentTensor("\(stem).weight")
        let scales = try mapping.residentTensor("\(stem).scales")
        let biases = try mapping.residentTensor("\(stem).biases")
        return .mlxAffine(
            words: words.buffer, wordsOffset: words.offset,
            scales: scales.buffer, scalesOffset: scales.offset,
            biases: biases.buffer, biasesOffset: biases.offset,
            bits: bits, groupSize: config.groupSize)
    }

    public func projection(
        _ suffix: String, layer: Int, rows: Int, cols: Int
    ) throws -> ForwardEncoder.ProjectionSource {
        try triple(config.layerTensor(suffix, layer: layer), bits: bits(for: suffix))
    }

    public func plain(_ suffix: String, layer: Int) throws -> (buffer: MTLBuffer, offset: Int) {
        let tensor = try mapping.residentTensor(config.layerTensor(suffix, layer: layer))
        return (tensor.buffer, tensor.offset)
    }

    public func finalNorm() throws -> (buffer: MTLBuffer, offset: Int) {
        let tensor = try mapping.residentTensor(config.finalNormTensor)
        return (tensor.buffer, tensor.offset)
    }

    public func head(rows: Int, cols: Int) throws -> ForwardEncoder.ProjectionSource {
        try triple("\(Gemma4MLXConfig.prefix).embed_tokens", bits: config.quantBits)
    }

    /// One embedding row, unpacked and dequantized on the CPU.
    ///
    /// The row is `hiddenSize` 4-bit values — 352 words at 2816 wide — with a scale and a bias
    /// every 64 columns. Reading it as BF16, which is what the unquantized path does, yields
    /// packed integers reinterpreted as floats: finite, enormous, and nothing to do with the
    /// token.
    public func readEmbedding(
        token: Int, into destination: UnsafeMutableBufferPointer<Float>
    ) {
        let hidden = config.hiddenSize
        precondition(destination.count == hidden)
        let layout = MLXAffineLayout(
            bits: config.quantBits, groupSize: config.groupSize,
            rows: config.vocabSize, cols: hidden)

        let stem = "\(Gemma4MLXConfig.prefix).embed_tokens"
        guard let words = try? mapping.residentTensor("\(stem).weight"),
            let scales = try? mapping.residentTensor("\(stem).scales"),
            let biases = try? mapping.residentTensor("\(stem).biases")
        else { return }

        let perWord = layout.valuesPerWord
        let mask = UInt32((1 << config.quantBits) - 1)
        let wordBase = words.offset + token * layout.wordsPerRow * 4
        let groupBase = scales.offset + token * layout.groupsPerRow * 2
        let biasBase = biases.offset + token * layout.groupsPerRow * 2

        mapping.resident.withBytes { raw in
            for word in 0..<layout.wordsPerRow {
                let packed = UInt32(
                    littleEndian: raw.loadUnaligned(
                        fromByteOffset: wordBase + word * 4, as: UInt32.self))
                for slot in 0..<perWord {
                    let column = word * perWord + slot
                    let group = column / config.groupSize
                    let scale = BF16.toFloat(
                        UInt16(littleEndian: raw.loadUnaligned(
                            fromByteOffset: groupBase + group * 2, as: UInt16.self)))
                    let bias = BF16.toFloat(
                        UInt16(littleEndian: raw.loadUnaligned(
                            fromByteOffset: biasBase + group * 2, as: UInt16.self)))
                    let q = Float((packed >> UInt32(slot * config.quantBits)) & mask)
                    destination[column] = q * scale + bias
                }
            }
        }
    }

    public func expert(
        _ part: Gemma4ExpertPart, blob: MTLBuffer, blobOffset: Int
    ) -> ForwardEncoder.ProjectionSource {
        let layout = config.expertBlobLayout
        let slots: (ExpertBlobLayout.Slot, ExpertBlobLayout.Slot, ExpertBlobLayout.Slot)
        switch part {
        case .gate: slots = (layout.gateWeights, layout.gateScales, layout.gateBiases)
        case .up: slots = (layout.upWeights, layout.upScales, layout.upBiases)
        case .down: slots = (layout.downWeights, layout.downScales, layout.downBiases)
        }
        return .mlxAffine(
            words: blob, wordsOffset: blobOffset + slots.0.offset,
            scales: blob, scalesOffset: blobOffset + slots.1.offset,
            biases: blob, biasesOffset: blobOffset + slots.2.offset,
            bits: config.quantBits, groupSize: config.groupSize)
    }
}

/// Builds the weight source for a descriptor. The Gemma half of `ModelRuntime`'s dispatch.
public enum Gemma4WeightSource {
    public static func make(
        model: any ModelDescriptor, mapping: ModelMapping
    ) -> (any Gemma4Weights)? {
        if let mlx = model as? Gemma4MLXConfig {
            return Gemma4MLXWeights(config: mlx, mapping: mapping)
        }
        if let bf16 = model as? Gemma4Config {
            return Gemma4BF16Weights(config: bf16, mapping: mapping)
        }
        return nil
    }
}
