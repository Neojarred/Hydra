import Foundation
import HydraCore
import HydraFormat
import HydraReference
import Metal
import Testing

@testable import HydraMetal

/// Four layers in the ratio Qwen uses, run end to end against a CPU composition.
///
/// The blocks are each already checked. What only this can catch is the sequencing: which block
/// a layer gets, that both residuals land, and that the two caches advance for the right layers.
/// A runner that gave every layer the attention block would still produce finite text.
@Suite("Qwen layer runner")
struct QwenLayerRunnerTests {

    // Widths are multiples of eight: `bf16_gemv` refuses anything else (M-050).
    private let config = Qwen35MoeConfig(
        layerCount: 4, hiddenSize: 16, fullAttentionInterval: 4,
        attentionHeadCount: 2, keyValueHeadCount: 1, headDim: 8,
        partialRotaryFactor: 0.25, ropeTheta: 10_000,
        linearKeyHeads: 1, linearValueHeads: 2, linearKeyHeadDim: 8, linearValueHeadDim: 8,
        linearConvKernel: 4, expertCount: 4, expertsPerToken: 2,
        moeIntermediateSize: 8, sharedExpertIntermediateSize: 8, groupSize: 16)

    private func deterministic(_ count: Int, _ seed: UInt64) -> [Double] {
        var state = seed | 1
        return (0..<count).map { _ in
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Double(Int(state >> 33) % 300 - 150) / 1000
        }
    }

    private func bf16(_ c: MetalContext, _ v: [Double]) -> (MTLBuffer, [Double])? {
        let bits = v.map { BF16.fromFloat(Float($0)) }
        let rounded = bits.map { Double(BF16.toFloat($0)) }
        return bits.withUnsafeBytes {
            c.device.makeBuffer(
                bytes: $0.baseAddress!, length: max($0.count, 4), options: .storageModeShared)
        }.map { ($0, rounded) }
    }

    /// The layer pattern is what this test exists for.
    @Test("Three recurrent layers to one attending, in that order")
    func layerPattern() throws {
        let context = try MetalContext()
        let runner = QwenLayerRunner(
            config: config, encoder: ForwardEncoder(context: context), context: context)
        #expect(runner.mixerKind(atLayer: 0) == .linear)
        #expect(runner.mixerKind(atLayer: 1) == .linear)
        #expect(runner.mixerKind(atLayer: 2) == .linear)
        #expect(runner.mixerKind(atLayer: 3) == .full)
    }

    /// A full pass over four layers, against the same pass composed on the CPU.
    ///
    /// Both caches advance here: the recurrent state on three layers and the key/value history
    /// on one, and getting either attached to the wrong layer changes the answer without
    /// changing its shape.
    @Test("Four layers on the GPU match four layers on the CPU")
    func fourLayersMatch() throws {
        let context = try MetalContext()
        let encoder = ForwardEncoder(context: context)
        let runner = QwenLayerRunner(config: config, encoder: encoder, context: context)
        let scratch = try QwenLayerRunner.Scratch(config: config, device: context.device)
        let hiddenSize = config.hiddenSize
        let zDim = config.linearValueHeads * config.linearValueHeadDim

        func matrixBuffer(_ r: Int, _ c: Int, _ s: UInt64) -> (MTLBuffer, [[Double]])? {
            guard let (buffer, rounded) = bf16(context, deterministic(r * c, s)) else { return nil }
            return (buffer, (0..<r).map { Array(rounded[($0 * c)..<(($0 + 1) * c)]) })
        }
        func floats(_ v: [Double]) -> MTLBuffer? {
            let f = v.map { Float($0) }
            return f.withUnsafeBytes {
                context.device.makeBuffer(
                    bytes: $0.baseAddress!, length: max($0.count, 4), options: .storageModeShared)
            }
        }
        func makeExpert(_ s: UInt64, inner: Int)
            -> (QwenMixtureBlock.Expert, QwenReferenceMixture.Expert)?
        {
            guard let (g, gm) = matrixBuffer(inner, hiddenSize, s),
                let (u, um) = matrixBuffer(inner, hiddenSize, s &+ 37),
                let (d, dm) = matrixBuffer(hiddenSize, inner, s &+ 73) else { return nil }
            return (
                .init(gate: .bf16(buffer: g, offset: 0), up: .bf16(buffer: u, offset: 0),
                      down: .bf16(buffer: d, offset: 0)),
                .init(gate: gm, up: um, down: dm))
        }

        // Per-layer weights, GPU and CPU built from the same rounded values.
        var gpuLayers: [QwenLayerRunner.LayerWeights] = []
        var cpuLinear: [QwenReferenceLayer] = []
        var cpuAttention: [QwenReferenceAttentionLayer] = []
        var cpuMixtures: [QwenReferenceMixture] = []
        var gpuExpertPool: [[QwenMixtureBlock.Expert]] = []
        var attentionLayerIndex: [Int: Int] = [:]

        for layer in 0..<config.layerCount {
            let seed = UInt64(layer) * 10_000 + 1
            guard let (postNorm, postNormV) = bf16(
                    context, deterministic(hiddenSize, seed &+ 900).map { $0 + 1 }),
                let (routerBuf, routerM) = matrixBuffer(config.expertCount, hiddenSize, seed &+ 100),
                let (sgBuf, sgM) = matrixBuffer(1, hiddenSize, seed &+ 200),
                let shared = makeExpert(seed &+ 300, inner: config.sharedExpertIntermediateSize),
                let (inNorm, inNormV) = bf16(
                    context, deterministic(hiddenSize, seed &+ 800).map { $0 + 1 })
            else { return }

            var gpuExperts: [QwenMixtureBlock.Expert] = []
            var cpuExperts: [QwenReferenceMixture.Expert] = []
            for e in 0..<config.expertCount {
                guard let pair = makeExpert(seed &+ 400 &+ UInt64(e) &* 20,
                                            inner: config.moeIntermediateSize) else { return }
                gpuExperts.append(pair.0)
                cpuExperts.append(pair.1)
            }
            gpuExpertPool.append(gpuExperts)
            cpuMixtures.append(QwenReferenceMixture(
                shape: .init(
                    hiddenSize: hiddenSize, expertCount: config.expertCount,
                    expertsPerToken: config.expertsPerToken,
                    moeIntermediate: config.moeIntermediateSize,
                    sharedIntermediate: config.sharedExpertIntermediateSize),
                weights: .init(
                    router: routerM, sharedGate: sgM[0], shared: shared.1, experts: cpuExperts)))

            let mixture = QwenMixtureBlock.Weights(
                postAttentionNorm: (postNorm, 0), router: .bf16(buffer: routerBuf, offset: 0),
                sharedGate: .bf16(buffer: sgBuf, offset: 0), shared: shared.0)
            _ = postNormV

            if config.attentionPattern(atLayer: layer) == .linear {
                guard let (qkv, qkvM) = matrixBuffer(config.linearConvDim, hiddenSize, seed &+ 10),
                    let (z, zM) = matrixBuffer(zDim, hiddenSize, seed &+ 20),
                    let (a, aM) = matrixBuffer(config.linearValueHeads, hiddenSize, seed &+ 30),
                    let (b, bM) = matrixBuffer(config.linearValueHeads, hiddenSize, seed &+ 40),
                    let (o, oM) = matrixBuffer(hiddenSize, zDim, seed &+ 50),
                    let (gn, gnV) = bf16(
                        context, deterministic(config.linearValueHeadDim, seed &+ 60).map { $0 + 1 }),
                    let (convW, convRounded) = bf16(
                        context,
                        deterministic(config.linearConvDim * config.linearConvKernel, seed &+ 70)),
                    let (logA, logARounded) = bf16(
                        context, deterministic(config.linearValueHeads, seed &+ 80)),
                    let (dtB, dtRounded) = bf16(
                        context, deterministic(config.linearValueHeads, seed &+ 90))
                else { return }
                gpuLayers.append(.init(
                    mixer: .linear(.init(
                        inputNorm: (inNorm, 0), qkv: .bf16(buffer: qkv, offset: 0),
                        z: .bf16(buffer: z, offset: 0), a: .bf16(buffer: a, offset: 0),
                        b: .bf16(buffer: b, offset: 0), outProj: .bf16(buffer: o, offset: 0),
                        convWeight: (convW, 0), convBias: nil,
                        logA: (logA, 0), dtBias: (dtB, 0),
                        normWeight: (gn, 0))),
                    mixture: mixture))
                let convFlat = convRounded
                cpuLinear.append(QwenReferenceLayer(
                    shape: .init(
                        hiddenSize: hiddenSize, keyHeads: config.linearKeyHeads,
                        valueHeads: config.linearValueHeads, keyDim: config.linearKeyHeadDim,
                        valueDim: config.linearValueHeadDim,
                        convKernel: config.linearConvKernel, eps: Double(config.rmsNormEps)),
                    weights: .init(
                        inputNorm: inNormV, qkv: qkvM, z: zM, a: aM, b: bM, outProj: oM,
                        convWeight: (0..<config.linearConvDim).map { c in
                            Array(convFlat[(c * config.linearConvKernel)..<((c + 1) * config.linearConvKernel)])
                        },
                        convBias: nil,
                        logA: logARounded, dtBias: dtRounded,
                        normWeight: gnV)))
            } else {
                guard let (q, qM) = matrixBuffer(config.queryProjectionRows, hiddenSize, seed &+ 10),
                    let (k, kM) = matrixBuffer(config.keyValueDim, hiddenSize, seed &+ 20),
                    let (v, vM) = matrixBuffer(config.keyValueDim, hiddenSize, seed &+ 30),
                    let (o, oM) = matrixBuffer(hiddenSize, config.queryDim, seed &+ 40),
                    let (qn, qnV) = bf16(
                        context, deterministic(config.headDim, seed &+ 50).map { $0 + 1 }),
                    let (kn, knV) = bf16(
                        context, deterministic(config.headDim, seed &+ 60).map { $0 + 1 })
                else { return }
                attentionLayerIndex[layer] = cpuAttention.count
                gpuLayers.append(.init(
                    mixer: .attention(.init(
                        inputNorm: (inNorm, 0), qProj: .bf16(buffer: q, offset: 0),
                        kProj: .bf16(buffer: k, offset: 0), vProj: .bf16(buffer: v, offset: 0),
                        oProj: .bf16(buffer: o, offset: 0), qNorm: (qn, 0), kNorm: (kn, 0))),
                    mixture: mixture))
                cpuAttention.append(QwenReferenceAttentionLayer(
                    shape: .init(
                        hiddenSize: hiddenSize, heads: config.attentionHeadCount,
                        keyValueHeads: config.keyValueHeadCount, headDim: config.headDim,
                        ropeTheta: config.ropeTheta,
                        partialRotaryFactor: config.partialRotaryFactor,
                        eps: Double(config.rmsNormEps)),
                    weights: .init(
                        inputNorm: inNormV, qProj: qM, kProj: kM, vProj: vM, oProj: oM,
                        qNorm: qnV, kNorm: knV)))
            }
        }

        let kv = try KVCache(model: config, contextLength: 32, device: context.device)
        let state = try RecurrentStateCache(
            model: config,
            geometry: .init(
                valueHeads: config.linearValueHeads, keyHeads: config.linearKeyHeads,
                keyDim: config.linearKeyHeadDim, valueDim: config.linearValueHeadDim,
                convDim: config.linearConvDim, convKernel: config.linearConvKernel),
            device: context.device)
        guard let hidden = context.device.makeBuffer(
            length: hiddenSize * 4, options: .storageModeShared) else { return }
        let sinkBits = [UInt16](repeating: BF16.fromFloat(-1e30), count: 64)
        guard let sinks = sinkBits.withUnsafeBytes({
            context.device.makeBuffer(
                bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
        }) else { return }

        var linearStates = cpuLinear.map { QwenReferenceLayer.State(shape: $0.shape) }
        var attentionCaches = cpuAttention.map { _ in QwenReferenceAttentionLayer.Cache() }
        let frequencies = Gemma4ReferenceOps.inverseFrequencies(
            headDim: config.headDim, theta: config.ropeTheta,
            rotatingPairs: Int(Double(config.headDim) * config.partialRotaryFactor) / 2)

        for position in 0..<3 {
            let input = deterministic(hiddenSize, 0x5500 + UInt64(position))
            input.map { Float($0) }.withUnsafeBytes {
                hidden.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count)
            }
            guard let cos = floats(frequencies.map { Foundation.cos(Double(position) * $0) }),
                let sin = floats(frequencies.map { Foundation.sin(Double(position) * $0) })
            else { return }

            for layer in 0..<config.layerCount {
                try runner.encodeLayer(
                    layer, hidden: hidden, weights: gpuLayers[layer], scratch: scratch,
                    kvCache: kv, state: state, position: position,
                    cos: cos, sin: sin, sinks: sinks,
                    commandBuffer: {
                        guard let b = context.commandQueue.makeCommandBuffer() else {
                            throw MetalContext.ContextError.noCommandQueue
                        }
                        return b
                    },
                    fetchExperts: { layerIndex, selected in
                        selected.map { gpuExpertPool[layerIndex][$0] }
                    })
            }
            try kv.advance()
            state.advance()

            // The same four layers on the CPU.
            var x = input
            var linearSlot = 0
            for layer in 0..<config.layerCount {
                if config.attentionPattern(atLayer: layer) == .linear {
                    x = cpuLinear[linearSlot].forward(x, state: &linearStates[linearSlot])
                    linearSlot += 1
                } else {
                    let slot = attentionLayerIndex[layer]!
                    x = cpuAttention[slot].forward(
                        x, position: position, cache: &attentionCaches[slot])
                }
                let normed = Gemma4ReferenceOps.rmsNorm(
                    x, weight: deterministic(hiddenSize, UInt64(layer) * 10_000 + 901)
                        .map { $0 + 1 }.map { Double(BF16.toFloat(BF16.fromFloat(Float($0)))) },
                    eps: Double(config.rmsNormEps))
                x = zip(x, cpuMixtures[layer].forward(normed)).map(+)
            }

            let got = hidden.contents().bindMemory(to: Float.self, capacity: hiddenSize)
            for i in 0..<hiddenSize {
                #expect(
                    abs(Double(got[i]) - x[i]) < 2e-2,
                    "position \(position), component \(i): four layers diverge")
            }
        }
    }
}
