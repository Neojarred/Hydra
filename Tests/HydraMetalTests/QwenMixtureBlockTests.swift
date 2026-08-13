import Foundation
import HydraCore
import HydraFormat
import HydraReference
import Metal
import Testing

@testable import HydraMetal

/// The mixture on the GPU, against the CPU composition.
@Suite("Qwen mixture block on GPU")
struct QwenMixtureBlockTests {

    private let config = Qwen35MoeConfig(
        // Widths are multiples of eight: `bf16_gemv` drops any remainder.
        layerCount: 4, hiddenSize: 16, fullAttentionInterval: 4,
        attentionHeadCount: 2, keyValueHeadCount: 1, headDim: 8,
        linearKeyHeads: 1, linearValueHeads: 2, linearKeyHeadDim: 4, linearValueHeadDim: 4,
        expertCount: 6, expertsPerToken: 3, moeIntermediateSize: 8,
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

    /// A rank whose expert never runs contributes nothing, not whatever was there before.
    ///
    /// The main test writes every rank every token, so the zeroing of the slots is never
    /// exercised by it and the line would rot untested. This encodes only some of the ranks and
    /// checks the result equals the sum of those alone: without the zeroing, the missing rank
    /// would carry the previous token's contribution into this one, which is a difference that
    /// grows with the conversation and never raises.
    @Test("A rank that never runs contributes nothing")
    func unwrittenSlotsAreZero() throws {
        let context = try MetalContext()
        let encoder = ForwardEncoder(context: context)
        let block = QwenMixtureBlock(config: config, encoder: encoder)
        let scratch = try QwenMixtureBlock.Scratch(config: config, device: context.device)
        let hiddenSize = config.hiddenSize

        func matrixBuffer(_ rows: Int, _ cols: Int, _ seed: UInt64) -> MTLBuffer? {
            bf16(context, deterministic(rows * cols, seed))?.0
        }
        func expert(_ seed: UInt64) -> QwenMixtureBlock.Expert? {
            guard let g = matrixBuffer(config.moeIntermediateSize, hiddenSize, seed),
                let u = matrixBuffer(config.moeIntermediateSize, hiddenSize, seed &+ 40),
                let d = matrixBuffer(hiddenSize, config.moeIntermediateSize, seed &+ 80)
            else { return nil }
            return .init(
                gate: .bf16(buffer: g, offset: 0), up: .bf16(buffer: u, offset: 0),
                down: .bf16(buffer: d, offset: 0))
        }
        guard let one = expert(0x2000), let shared = expert(0x9000),
            let router = matrixBuffer(config.expertCount, hiddenSize, 0x100),
            let sg = matrixBuffer(1, hiddenSize, 0x300),
            let norm = bf16(context, deterministic(hiddenSize, 0x1).map { $0 + 1 })?.0,
            let hidden = context.device.makeBuffer(
                length: hiddenSize * 4, options: .storageModeShared)
        else { return }

        let weights = QwenMixtureBlock.Weights(
            postAttentionNorm: (norm, 0), router: .bf16(buffer: router, offset: 0),
            sharedGate: .bf16(buffer: sg, offset: 0), shared: shared)

        func run(ranks: [Int], input: [Double]) throws -> [Float] {
            let asFloats = input.map { Float($0) }
            asFloats.withUnsafeBytes {
                hidden.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count)
            }
            guard let first = context.commandQueue.makeCommandBuffer() else { return [] }
            try block.encodeRouterAndShared(
                hidden: hidden, weights: weights, scratch: scratch, in: first)
            context.commit(first)
            try context.wait(first)

            guard let second = context.commandQueue.makeCommandBuffer() else { return [] }
            for rank in ranks {
                try block.encodeExpert(one, rank: rank, scratch: scratch, in: second)
            }
            try block.encodeCombine(hidden: hidden, scratch: scratch, in: second)
            context.commit(second)
            try context.wait(second)
            let out = hidden.contents().bindMemory(to: Float.self, capacity: hiddenSize)
            return (0..<hiddenSize).map { out[$0] }
        }

        // Fill every slot, then run again filling only rank zero. The two must differ: the
        // second has two fewer contributions. Without the zeroing they would be **equal**,
        // because ranks one and two would still hold the first run's values, and comparing two
        // partial runs to each other cannot see that because they share the same staleness.
        let input = deterministic(hiddenSize, 0x4242)
        let full = try run(ranks: Array(0..<config.expertsPerToken), input: input)
        let partial = try run(ranks: [0], input: input)

        let difference = zip(full, partial).map { abs($0 - $1) }.max() ?? 0
        let why = "a run with fewer experts must differ; equal means the unused slots kept "
            + "the previous run's contributions"
        #expect(difference > 1e-5, "\(why)")
    }

    @Test("The GPU mixture matches the CPU composition, shared branch and all")
    func matchesComposition() throws {
        let context = try MetalContext()
        let encoder = ForwardEncoder(context: context)
        let block = QwenMixtureBlock(config: config, encoder: encoder)
        let scratch = try QwenMixtureBlock.Scratch(config: config, device: context.device)
        let hiddenSize = config.hiddenSize

        func matrixBuffer(_ rows: Int, _ cols: Int, _ seed: UInt64) -> (MTLBuffer, [[Double]])? {
            let flat = deterministic(rows * cols, seed)
            guard let (buffer, rounded) = bf16(context, flat) else { return nil }
            return (buffer, (0..<rows).map { Array(rounded[($0 * cols)..<(($0 + 1) * cols)]) })
        }

        // One expert is three matrices; build the pool and the shared one the same way.
        func makeExpert(_ seed: UInt64, inner: Int)
            -> (QwenMixtureBlock.Expert, QwenReferenceMixture.Expert)?
        {
            guard let (g, gm) = matrixBuffer(inner, hiddenSize, seed),
                let (u, um) = matrixBuffer(inner, hiddenSize, seed &+ 40),
                let (d, dm) = matrixBuffer(hiddenSize, inner, seed &+ 80)
            else { return nil }
            return (
                .init(
                    gate: .bf16(buffer: g, offset: 0), up: .bf16(buffer: u, offset: 0),
                    down: .bf16(buffer: d, offset: 0)),
                .init(gate: gm, up: um, down: dm))
        }

        var gpuExperts: [QwenMixtureBlock.Expert] = []
        var cpuExperts: [QwenReferenceMixture.Expert] = []
        for index in 0..<config.expertCount {
            guard let pair = makeExpert(0x2000 &+ UInt64(index) &* 300, inner: config.moeIntermediateSize)
            else { return }
            gpuExperts.append(pair.0)
            cpuExperts.append(pair.1)
        }
        guard let sharedPair = makeExpert(0x9000, inner: config.sharedExpertIntermediateSize),
            let (routerBuf, routerM) = matrixBuffer(config.expertCount, hiddenSize, 0x100),
            let (sgBuf, sgM) = matrixBuffer(1, hiddenSize, 0x300),
            let (normBuf, normV) = bf16(context, deterministic(hiddenSize, 0x1).map { $0 + 1 }),
            let hidden = context.device.makeBuffer(
                length: hiddenSize * 4, options: .storageModeShared)
        else { return }

        let weights = QwenMixtureBlock.Weights(
            postAttentionNorm: (normBuf, 0),
            router: .bf16(buffer: routerBuf, offset: 0),
            sharedGate: .bf16(buffer: sgBuf, offset: 0), shared: sharedPair.0)

        let reference = QwenReferenceMixture(
            shape: .init(
                hiddenSize: hiddenSize, expertCount: config.expertCount,
                expertsPerToken: config.expertsPerToken,
                moeIntermediate: config.moeIntermediateSize,
                sharedIntermediate: config.sharedExpertIntermediateSize),
            weights: .init(
                router: routerM, sharedGate: sgM[0], shared: sharedPair.1, experts: cpuExperts))

        for token in 0..<3 {
            let input = deterministic(hiddenSize, 0x7700 + UInt64(token))
            let asFloats = input.map { Float($0) }
            asFloats.withUnsafeBytes {
                hidden.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count)
            }

            // The router runs, then the CPU learns which experts were chosen, exactly as the
            // real runner will: nothing can be read from the SSD before that.
            guard let first = context.commandQueue.makeCommandBuffer() else { return }
            try block.encodeRouterAndShared(
                hidden: hidden, weights: weights, scratch: scratch, in: first)
            context.commit(first)
            try context.wait(first)

            let selected = block.selectedExperts(scratch)
            guard let second = context.commandQueue.makeCommandBuffer() else { return }
            for (rank, expert) in selected.enumerated() {
                try block.encodeExpert(
                    gpuExperts[expert], rank: rank, scratch: scratch, in: second)
            }
            try block.encodeCombine(hidden: hidden, scratch: scratch, in: second)
            context.commit(second)
            try context.wait(second)

            // The reference computes the mixture's contribution; the layer adds the residual,
            // and so does the block, so compare against input + contribution.
            let contribution = reference.forward(
                Gemma4ReferenceOps.rmsNorm(input, weight: normV, eps: Double(config.rmsNormEps)))
            let expected = zip(input, contribution).map(+)
            let got = hidden.contents().bindMemory(to: Float.self, capacity: hiddenSize)
            for i in 0..<hiddenSize {
                #expect(
                    abs(Double(got[i]) - expected[i]) < 5e-3,
                    "token \(token), component \(i): the mixture diverges")
            }
        }
    }
}
