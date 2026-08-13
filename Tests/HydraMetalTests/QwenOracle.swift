import Foundation
import HydraCore
import HydraFormat
import HydraReference
import Metal
import Testing

@testable import HydraMetal

/// The whole model on the CPU, in double precision, decoded from the installed bytes.
///
/// Deliberately resolves its own tensor names rather than borrowing `Qwen35MoeWeights`: an
/// oracle that asked the implementation where `A_log` lives would agree with it about the wrong
/// tensor, which is the failure this exists to catch.
struct QwenOracleBuilder {

    let config: Qwen35MoeConfig

    // MARK: - Reading the installation back

    /// A BF16 span of `resident.bin`.
    func vector(
        _ raw: UnsafeRawBufferPointer, at offset: Int, count: Int
    ) -> [Double] {
        (0..<count).map {
            Double(BF16.toFloat(UInt16(littleEndian: raw.loadUnaligned(
                fromByteOffset: offset + $0 * 2, as: UInt16.self))))
        }
    }

    /// An MLX affine matrix, decoded row by row into `[rows][cols]`.
    func matrix(
        _ raw: UnsafeRawBufferPointer, words: Int, scales: Int, biases: Int,
        rows: Int, cols: Int, bits: Int
    ) -> [[Double]] {
        let layout = MLXAffineLayout(
            bits: bits, groupSize: config.groupSize, rows: rows, cols: cols)
        return (0..<rows).map { row in
            let w = (0..<layout.wordsPerRow).map { index in
                UInt32(littleEndian: raw.loadUnaligned(
                    fromByteOffset: words + (row * layout.wordsPerRow + index) * 4,
                    as: UInt32.self))
            }
            func group(_ base: Int) -> [Double] {
                vector(raw, at: base + row * layout.groupsPerRow * 2, count: layout.groupsPerRow)
            }
            return MLXAffine.dequantize(
                words: w, scales: group(scales), biases: group(biases),
                bits: bits, groupSize: config.groupSize)
        }
    }

    /// Builds the oracle: every weight of the model, read out of the installation.
    ///
    /// Deliberately resolves its own tensor names rather than borrowing `Qwen35MoeWeights`.
    /// An oracle that asked the implementation where `A_log` lives would agree with it about
    /// the wrong tensor, which is the whole failure this suite exists to catch.
    struct Oracle {
        var embedding: [[Double]]
        var head: [[Double]]
        var finalNorm: [Double]
        var linear: [Int: QwenReferenceLayer]
        var attention: [Int: QwenReferenceAttentionLayer]
        var mixtures: [QwenReferenceMixture]
        var postNorms: [[Double]]
    }

    func buildOracle(
        mapping: ModelMapping, cache: ExpertSlotCache
    ) throws -> Oracle {
        let hidden = config.hiddenSize
        func stem(_ name: String) throws -> (w: Int, s: Int, b: Int) {
            (try mapping.residentTensor("\(name).weight").offset,
             try mapping.residentTensor("\(name).scales").offset,
             try mapping.residentTensor("\(name).biases").offset)
        }
        func plainOffset(_ name: String) throws -> Int {
            try mapping.residentTensor(name).offset
        }
        func layerName(_ suffix: String, _ layer: Int) -> String {
            "\(Qwen35MoeConfig.prefix).layers.\(layer).\(suffix)"
        }

        return try mapping.resident.withBytes { raw -> Oracle in
            func quant(_ name: String, rows: Int, cols: Int, bits: Int) throws -> [[Double]] {
                let s = try stem(name)
                return matrix(
                    raw, words: s.w, scales: s.s, biases: s.b,
                    rows: rows, cols: cols, bits: bits)
            }
            func plain(_ name: String, count: Int) throws -> [Double] {
                vector(raw, at: try plainOffset(name), count: count)
            }

            var linear: [Int: QwenReferenceLayer] = [:]
            var attention: [Int: QwenReferenceAttentionLayer] = [:]
            var mixtures: [QwenReferenceMixture] = []
            var postNorms: [[Double]] = []

            for layer in 0..<config.layerCount {
                let inputNorm = try plain(layerName("input_layernorm.weight", layer), count: hidden)

                if config.attentionPattern(atLayer: layer) == .full {
                    attention[layer] = QwenReferenceAttentionLayer(
                        shape: .init(
                            hiddenSize: hidden, heads: config.attentionHeadCount,
                            keyValueHeads: config.keyValueHeadCount, headDim: config.headDim,
                            ropeTheta: config.ropeTheta,
                            partialRotaryFactor: config.partialRotaryFactor,
                            eps: Double(config.rmsNormEps)),
                        weights: .init(
                            inputNorm: inputNorm,
                            qProj: try quant(
                                layerName("self_attn.q_proj", layer),
                                rows: config.queryProjectionRows, cols: hidden,
                                bits: config.quantBits),
                            kProj: try quant(
                                layerName("self_attn.k_proj", layer),
                                rows: config.keyValueDim, cols: hidden, bits: config.quantBits),
                            vProj: try quant(
                                layerName("self_attn.v_proj", layer),
                                rows: config.keyValueDim, cols: hidden, bits: config.quantBits),
                            oProj: try quant(
                                layerName("self_attn.o_proj", layer),
                                rows: hidden, cols: config.queryDim, bits: config.quantBits),
                            qNorm: try plain(
                                layerName("self_attn.q_norm.weight", layer),
                                count: config.headDim),
                            kNorm: try plain(
                                layerName("self_attn.k_norm.weight", layer),
                                count: config.headDim)))
                } else {
                    let zDim = config.linearValueHeads * config.linearValueHeadDim
                    let convDim = config.linearConvDim
                    let convFlat = try plain(
                        layerName("linear_attn.conv1d.weight", layer),
                        count: convDim * config.linearConvKernel)
                    linear[layer] = QwenReferenceLayer(
                        shape: .init(
                            hiddenSize: hidden, keyHeads: config.linearKeyHeads,
                            valueHeads: config.linearValueHeads,
                            keyDim: config.linearKeyHeadDim,
                            valueDim: config.linearValueHeadDim,
                            convKernel: config.linearConvKernel,
                            eps: Double(config.rmsNormEps)),
                        weights: .init(
                            inputNorm: inputNorm,
                            qkv: try quant(
                                layerName("linear_attn.in_proj_qkv", layer),
                                rows: convDim, cols: hidden, bits: config.quantBits),
                            z: try quant(
                                layerName("linear_attn.in_proj_z", layer),
                                rows: zDim, cols: hidden, bits: config.quantBits),
                            a: try quant(
                                layerName("linear_attn.in_proj_a", layer),
                                rows: config.linearValueHeads, cols: hidden,
                                bits: config.quantBits),
                            b: try quant(
                                layerName("linear_attn.in_proj_b", layer),
                                rows: config.linearValueHeads, cols: hidden,
                                bits: config.quantBits),
                            outProj: try quant(
                                layerName("linear_attn.out_proj", layer),
                                rows: hidden, cols: zDim, bits: config.quantBits),
                            convWeight: (0..<convDim).map { channel in
                                (0..<config.linearConvKernel).map {
                                    convFlat[channel * config.linearConvKernel + $0]
                                }
                            },
                            convBias: nil,
                            logA: try plain(
                                layerName("linear_attn.A_log", layer),
                                count: config.linearValueHeads),
                            dtBias: try plain(
                                layerName("linear_attn.dt_bias", layer),
                                count: config.linearValueHeads),
                            normWeight: try plain(
                                layerName("linear_attn.norm.weight", layer),
                                count: config.linearValueHeadDim)))
                }

                postNorms.append(
                    try plain(layerName("post_attention_layernorm.weight", layer), count: hidden))

                // Every expert of the layer, read through the slot cache, which is also how the
                // GPU sees them.
                let blob = config.expertBlobLayout
                var experts: [QwenReferenceMixture.Expert] = []
                for index in 0..<config.expertCount {
                    let (buffer, base) = try cache.expert(layer: layer, expert: index)
                    let bytes = UnsafeRawBufferPointer(
                        start: buffer.contents(), count: buffer.length)
                    func part(
                        _ w: ExpertBlobLayout.Slot, _ s: ExpertBlobLayout.Slot,
                        _ b: ExpertBlobLayout.Slot, rows: Int, cols: Int
                    ) -> [[Double]] {
                        matrix(
                            bytes, words: base + w.offset, scales: base + s.offset,
                            biases: base + b.offset, rows: rows, cols: cols,
                            bits: config.quantBits)
                    }
                    experts.append(
                        QwenReferenceMixture.Expert(
                            gate: part(
                                blob.gateWeights, blob.gateScales, blob.gateBiases,
                                rows: config.moeIntermediateSize, cols: hidden),
                            up: part(
                                blob.upWeights, blob.upScales, blob.upBiases,
                                rows: config.moeIntermediateSize, cols: hidden),
                            down: part(
                                blob.downWeights, blob.downScales, blob.downBiases,
                                rows: hidden, cols: config.moeIntermediateSize)))
                }

                mixtures.append(
                    QwenReferenceMixture(
                        shape: .init(
                            hiddenSize: hidden, expertCount: config.expertCount,
                            expertsPerToken: config.expertsPerToken,
                            moeIntermediate: config.moeIntermediateSize,
                            sharedIntermediate: config.sharedExpertIntermediateSize),
                        weights: .init(
                            router: try quant(
                                layerName("mlp.gate", layer),
                                rows: config.expertCount, cols: hidden, bits: config.gateBits),
                            sharedGate: try quant(
                                layerName("mlp.shared_expert_gate", layer),
                                rows: 1, cols: hidden, bits: config.gateBits)[0],
                            shared: .init(
                                gate: try quant(
                                    layerName("mlp.shared_expert.gate_proj", layer),
                                    rows: config.sharedExpertIntermediateSize, cols: hidden,
                                    bits: config.quantBits),
                                up: try quant(
                                    layerName("mlp.shared_expert.up_proj", layer),
                                    rows: config.sharedExpertIntermediateSize, cols: hidden,
                                    bits: config.quantBits),
                                down: try quant(
                                    layerName("mlp.shared_expert.down_proj", layer),
                                    rows: hidden, cols: config.sharedExpertIntermediateSize,
                                    bits: config.quantBits)),
                            experts: experts)))
            }

            return Oracle(
                embedding: try quant(
                    config.embeddingStem, rows: config.vocabSize, cols: hidden,
                    bits: config.quantBits),
                head: try quant(
                    config.headStem, rows: config.vocabSize, cols: hidden,
                    bits: config.quantBits),
                finalNorm: try plain(config.finalNormTensor, count: hidden),
                linear: linear, attention: attention,
                mixtures: mixtures, postNorms: postNorms)
        }
    }

    /// The whole model on the CPU, over a sequence, returning the last position's logits.
    func oracleLogits(_ oracle: Oracle, tokens: [Int]) -> [Double] {
        var states = oracle.linear.mapValues { QwenReferenceLayer.State(shape: $0.shape) }
        var caches = oracle.attention.mapValues { _ in QwenReferenceAttentionLayer.Cache() }
        var last = [Double](repeating: 0, count: config.hiddenSize)

        for (position, token) in tokens.enumerated() {
            // No scale on the embedding: this model does not carry Gemma's `sqrt(hidden)`.
            var x = oracle.embedding[token]
            for layer in 0..<config.layerCount {
                if let block = oracle.linear[layer] {
                    x = block.forward(x, state: &states[layer]!)
                } else if let block = oracle.attention[layer] {
                    x = block.forward(x, position: position, cache: &caches[layer]!)
                }
                let normed = Gemma4ReferenceOps.rmsNorm(
                    x, weight: oracle.postNorms[layer], eps: Double(config.rmsNormEps))
                x = zip(x, oracle.mixtures[layer].forward(normed)).map(+)
            }
            last = x
        }

        // The final norm and the untied head. No softcap.
        let normed = Gemma4ReferenceOps.rmsNorm(
            last, weight: oracle.finalNorm, eps: Double(config.rmsNormEps))
        return oracle.head.map { row in zip(row, normed).reduce(0) { $0 + $1.0 * $1.1 } }
    }

}
