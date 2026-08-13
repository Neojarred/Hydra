import Foundation
import HydraCore
import HydraFormat
import HydraReference
import Metal
import Testing

@testable import HydraMetal

/// The full-attention block on the GPU, against the CPU composition.
///
/// Several tokens, so the attention distribution has something to be wrong about: with one
/// token a softmax over one key is 1 however it is scaled, and the scale is the detail Gemma
/// does not share (M-049).
@Suite("Qwen attention block on GPU")
struct QwenAttentionBlockTests {

    private let config = Qwen35MoeConfig(
        layerCount: 4, hiddenSize: 16, fullAttentionInterval: 4,
        attentionHeadCount: 4, keyValueHeadCount: 2, headDim: 8,
        partialRotaryFactor: 0.25, ropeTheta: 10_000,
        linearKeyHeads: 2, linearValueHeads: 4, linearKeyHeadDim: 4, linearValueHeadDim: 4,
        expertCount: 4, expertsPerToken: 2, moeIntermediateSize: 8,
        sharedExpertIntermediateSize: 8, groupSize: 16)

    private func deterministic(_ count: Int, _ seed: UInt64) -> [Double] {
        var state = seed | 1
        return (0..<count).map { _ in
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Double(Int(state >> 33) % 400 - 200) / 1000
        }
    }

    private func bf16(_ context: MetalContext, _ v: [Double]) -> (MTLBuffer, [Double])? {
        let bits = v.map { BF16.fromFloat(Float($0)) }
        let rounded = bits.map { Double(BF16.toFloat($0)) }
        return bits.withUnsafeBytes {
            context.device.makeBuffer(
                bytes: $0.baseAddress!, length: max($0.count, 4), options: .storageModeShared)
        }.map { ($0, rounded) }
    }

    private func floats(_ context: MetalContext, _ v: [Double]) -> MTLBuffer? {
        let f = v.map { Float($0) }
        return f.withUnsafeBytes {
            context.device.makeBuffer(
                bytes: $0.baseAddress!, length: max($0.count, 4), options: .storageModeShared)
        }
    }

    @Test("The GPU block matches the CPU composition across several positions")
    func matchesComposition() throws {
        let context = try MetalContext()
        let encoder = ForwardEncoder(context: context)
        let block = QwenAttentionBlock(config: config, encoder: encoder)
        let scratch = try QwenAttentionBlock.Scratch(config: config, device: context.device)

        let hiddenSize = config.hiddenSize
        let heads = config.attentionHeadCount
        let kvHeads = config.keyValueHeadCount
        let headDim = config.headDim
        let capacity = 8

        func matrixBuffer(_ rows: Int, _ cols: Int, _ seed: UInt64) -> (MTLBuffer, [[Double]])? {
            let flat = deterministic(rows * cols, seed)
            guard let (buffer, rounded) = bf16(context, flat) else { return nil }
            return (buffer, (0..<rows).map { Array(rounded[($0 * cols)..<(($0 + 1) * cols)]) })
        }

        guard let (qBuf, qM) = matrixBuffer(config.queryProjectionRows, hiddenSize, 0x100),
            let (kBuf, kM) = matrixBuffer(config.keyValueDim, hiddenSize, 0x200),
            let (vBuf, vM) = matrixBuffer(config.keyValueDim, hiddenSize, 0x300),
            let (oBuf, oM) = matrixBuffer(hiddenSize, config.queryDim, 0x400),
            let (normBuf, normV) = bf16(context, deterministic(hiddenSize, 0x1).map { $0 + 1 }),
            let (qnBuf, qnV) = bf16(context, deterministic(headDim, 0x500).map { $0 + 1 }),
            let (knBuf, knV) = bf16(context, deterministic(headDim, 0x600).map { $0 + 1 }),
            let hidden = context.device.makeBuffer(
                length: hiddenSize * 4, options: .storageModeShared),
            let keyCache = context.device.makeBuffer(
                length: capacity * config.keyValueDim * 2, options: .storageModeShared),
            let valueCache = context.device.makeBuffer(
                length: capacity * config.keyValueDim * 2, options: .storageModeShared)
        else { return }
        memset(keyCache.contents(), 0, keyCache.length)
        memset(valueCache.contents(), 0, valueCache.length)

        // No learned sinks: an unreachable one a head, as Gemma does.
        let sinkBits = [UInt16](repeating: BF16.fromFloat(-1e30), count: max(heads, 64))
        guard let sinks = sinkBits.withUnsafeBytes({
            context.device.makeBuffer(
                bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
        }) else { return }

        let shape = QwenReferenceAttentionLayer.Shape(
            hiddenSize: hiddenSize, heads: heads, keyValueHeads: kvHeads, headDim: headDim,
            ropeTheta: config.ropeTheta, partialRotaryFactor: config.partialRotaryFactor,
            eps: Double(config.rmsNormEps))
        let reference = QwenReferenceAttentionLayer(
            shape: shape,
            weights: .init(
                inputNorm: normV, qProj: qM, kProj: kM, vProj: vM, oProj: oM,
                qNorm: qnV, kNorm: knV))
        var cache = QwenReferenceAttentionLayer.Cache()

        // The rotary tables, zero past the rotating quarter so the rest does not turn.
        let frequencies = Gemma4ReferenceOps.inverseFrequencies(
            headDim: headDim, theta: config.ropeTheta, rotatingPairs: shape.rotatingPairs)

        for position in 0..<4 {
            let input = deterministic(hiddenSize, 0x9000 + UInt64(position))
            let asFloats = input.map { Float($0) }
            asFloats.withUnsafeBytes {
                hidden.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count)
            }
            guard let cos = floats(context, frequencies.map { Foundation.cos(Double(position) * $0) }),
                let sin = floats(context, frequencies.map { Foundation.sin(Double(position) * $0) }),
                let command = context.commandQueue.makeCommandBuffer()
            else { return }

            try block.encode(
                hidden: hidden, weights: .init(
                    inputNorm: (normBuf, 0),
                    qProj: .bf16(buffer: qBuf, offset: 0), kProj: .bf16(buffer: kBuf, offset: 0),
                    vProj: .bf16(buffer: vBuf, offset: 0), oProj: .bf16(buffer: oBuf, offset: 0),
                    qNorm: (qnBuf, 0), kNorm: (knBuf, 0)),
                scratch: scratch,
                keyCache: keyCache, valueCache: valueCache, sinks: sinks,
                position: position, visibleStart: 0, visibleCount: position + 1, ringSize: 0,
                cos: cos, sin: sin, in: command)
            context.commit(command)
            try context.wait(command)

            let expected = reference.forward(input, position: position, cache: &cache)
            let got = hidden.contents().bindMemory(to: Float.self, capacity: hiddenSize)
            for i in 0..<hiddenSize {
                #expect(
                    abs(Double(got[i]) - expected[i]) < 5e-3,
                    "position \(position), component \(i): the composition diverges")
            }
        }
    }
}
