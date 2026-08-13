import Foundation
import HydraCore
import HydraFormat
import HydraReference
import Metal
import Testing

@testable import HydraMetal

/// The whole linear block on the GPU, against the CPU composition.
///
/// Every kernel underneath is already checked individually. This checks that they are wired
/// together in the right order, reading the right spans, with the state and the window advancing
/// where they should. That is the mistake class the individual tests cannot see, and the one
/// Gemma cost the most time to find.
///
/// Driven with BF16 weights built by hand rather than a quantized checkpoint. The projection
/// kernel is separately checked at both encodings, so what this pins is the composition, and it
/// can be pinned before an installer for this architecture exists.
@Suite("Qwen linear block on GPU")
struct QwenLinearBlockTests {

    private let config = Qwen35MoeConfig(
        layerCount: 4, hiddenSize: 16, fullAttentionInterval: 4,
        linearKeyHeads: 2, linearValueHeads: 4, linearKeyHeadDim: 4, linearValueHeadDim: 4,
        linearConvKernel: 4, expertCount: 4, expertsPerToken: 2,
        moeIntermediateSize: 8, sharedExpertIntermediateSize: 8, groupSize: 16)

    private func deterministic(_ count: Int, _ seed: UInt64) -> [Double] {
        var state = seed | 1
        return (0..<count).map { _ in
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Double(Int(state >> 33) % 400 - 200) / 1000
        }
    }

    /// BF16 on both sides, so the deviation measured is the composition's and not the format's.
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

    @Test("The GPU block matches the CPU composition across a carried sequence")
    func matchesComposition() throws {
        let context = try MetalContext()
        let encoder = ForwardEncoder(context: context)
        let block = QwenLinearBlock(config: config, encoder: encoder)
        let scratch = try QwenLinearBlock.Scratch(config: config, device: context.device)

        let hiddenSize = config.hiddenSize
        let convDim = config.linearConvDim
        let zDim = config.linearValueHeads * config.linearValueHeadDim
        let heads = config.linearValueHeads

        // Every matrix as BF16, and the rounded values fed to the reference so both sides see
        // the same weights.
        func matrixBuffer(_ rows: Int, _ cols: Int, _ seed: UInt64)
            -> (MTLBuffer, [[Double]])?
        {
            let flat = deterministic(rows * cols, seed)
            guard let (buffer, rounded) = bf16(context, flat) else { return nil }
            let matrix = (0..<rows).map { r in Array(rounded[(r * cols)..<((r + 1) * cols)]) }
            return (buffer, matrix)
        }

        guard let (qkvBuf, qkvM) = matrixBuffer(convDim, hiddenSize, 0x100),
            let (zBuf, zM) = matrixBuffer(zDim, hiddenSize, 0x200),
            let (aBuf, aM) = matrixBuffer(heads, hiddenSize, 0x300),
            let (bBuf, bM) = matrixBuffer(heads, hiddenSize, 0x400),
            let (outBuf, outM) = matrixBuffer(hiddenSize, zDim, 0x500),
            let (normBuf, normV) = bf16(context, deterministic(hiddenSize, 0x1).map { $0 + 1 }),
            let (gnormBuf, gnormV) = bf16(
                context, deterministic(config.linearValueHeadDim, 0xA00).map { $0 + 1 }),
            let convWeight = floats(context, deterministic(convDim * config.linearConvKernel, 0x600)),
            let logA = floats(context, deterministic(heads, 0x800)),
            let dtBias = floats(context, deterministic(heads, 0x900)),
            let hidden = context.device.makeBuffer(
                length: hiddenSize * 4, options: .storageModeShared),
            let state = context.device.makeBuffer(
                length: heads * config.linearKeyHeadDim * config.linearValueHeadDim * 4,
                options: .storageModeShared),
            let window = context.device.makeBuffer(
                length: (config.linearConvKernel - 1) * convDim * 4, options: .storageModeShared)
        else { return }
        memset(state.contents(), 0, state.length)
        memset(window.contents(), 0, window.length)

        let weights = QwenLinearBlock.Weights(
            inputNorm: (normBuf, 0),
            qkv: .bf16(buffer: qkvBuf, offset: 0), z: .bf16(buffer: zBuf, offset: 0),
            a: .bf16(buffer: aBuf, offset: 0), b: .bf16(buffer: bBuf, offset: 0),
            outProj: .bf16(buffer: outBuf, offset: 0),
            convWeight: convWeight, convBias: nil,
            logA: logA, dtBias: dtBias, normWeight: (gnormBuf, 0))

        let shape = QwenReferenceLayer.Shape(
            hiddenSize: hiddenSize, keyHeads: config.linearKeyHeads,
            valueHeads: heads, keyDim: config.linearKeyHeadDim,
            valueDim: config.linearValueHeadDim, convKernel: config.linearConvKernel,
            eps: Double(config.rmsNormEps))
        let convFlat = deterministic(convDim * config.linearConvKernel, 0x600)
        let reference = QwenReferenceLayer(
            shape: shape,
            weights: .init(
                inputNorm: normV, qkv: qkvM, z: zM, a: aM, b: bM, outProj: outM,
                convWeight: (0..<convDim).map { c in
                    Array(convFlat[(c * config.linearConvKernel)..<((c + 1) * config.linearConvKernel)])
                },
                convBias: nil,
                logA: deterministic(heads, 0x800), dtBias: deterministic(heads, 0x900),
                normWeight: gnormV))
        var referenceState = QwenReferenceLayer.State(shape: shape)

        var carried = deterministic(hiddenSize, 0xBEEF)
        for token in 0..<5 {
            let input = zip(carried, deterministic(hiddenSize, 0xC000 + UInt64(token))).map(+)
            let asFloats = input.map { Float($0) }
            asFloats.withUnsafeBytes {
                hidden.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count)
            }

            guard let command = context.commandQueue.makeCommandBuffer() else { return }
            try block.encode(
                hidden: hidden, weights: weights, scratch: scratch,
                state: state, stateOffset: 0, window: window, windowOffset: 0, in: command)
            context.commit(command)
            try context.wait(command)

            let expected = reference.forward(input, state: &referenceState)
            let got = hidden.contents().bindMemory(to: Float.self, capacity: hiddenSize)
            for i in 0..<hiddenSize {
                #expect(
                    abs(Double(got[i]) - expected[i]) < 5e-4,
                    "token \(token), component \(i): the composition diverges")
            }
            carried = expected
        }
    }
}
