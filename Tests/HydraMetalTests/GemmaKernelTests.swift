import Foundation
import HydraCore
import HydraFormat
import HydraReference
import Metal
import Testing

@testable import HydraMetal

/// Gemma 4's kernels, measured against the CPU oracle.
///
/// The same dispositif as `MXFP4KernelTests`: the GPU is compared to `Gemma4ReferenceOps`,
/// which is itself checked against an independent Python transcription. A kernel that is
/// merely plausible passes no test here.
///
/// One of these suites exists to check something that is *not* a kernel: that Gemma's absence
/// of attention sinks can be expressed through the shared attention kernel rather than needing
/// its own. That claim is load-bearing and is therefore measured, not assumed.
@Suite("Gemma 4 kernels")
struct GemmaKernelTests {

    private func makeContext() throws -> MetalContext { try MetalContext() }

    private func buffer(_ context: MetalContext, _ values: [Float]) -> MTLBuffer? {
        let buffer = context.device.makeBuffer(
            length: max(values.count * 4, 256), options: .storageModeShared)
        if let buffer {
            values.withUnsafeBytes {
                buffer.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count)
            }
        }
        return buffer
    }

    private func read(_ buffer: MTLBuffer, count: Int) -> [Float] {
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    private func deterministic(_ count: Int, seed: UInt32) -> [Float] {
        var state = seed
        return (0..<count).map { _ in
            state = state &* 1_103_515_245 &+ 12345
            return Float(state >> 8) / Float(1 << 24) * 2 - 1
        }
    }

    private func worst(_ got: [Float], _ expected: [Double]) -> Double {
        var scale = 0.0
        for value in expected { scale = max(scale, abs(value)) }
        var worst = 0.0
        for (a, b) in zip(got, expected) { worst = max(worst, abs(Double(a) - b) / max(scale, 1e-9)) }
        return worst
    }

    private func run(
        _ context: MetalContext, _ body: (MTLCommandBuffer) throws -> Void
    ) throws {
        guard let commandBuffer = context.commandQueue.makeCommandBuffer() else { return }
        try body(commandBuffer)
        context.commit(commandBuffer)
        commandBuffer.waitUntilCompleted()
    }

    // MARK: - Kernels

    /// `v_norm` and the router's norm have no tensor, so the GPU cannot read a scale it does
    /// not have. A version that quietly used ones would be right only by accident.
    @Test("Unscaled RMSNorm matches the reference")
    func unscaledRmsNorm() throws {
        let context = try makeContext()
        let encoder = ForwardEncoder(context: context)
        let size = 2816
        let input = deterministic(size, seed: 0x1234)

        guard let x = buffer(context, input), let out = buffer(context, [Float](repeating: 0, count: size))
        else { return }

        try run(context) {
            try encoder.rmsNormUnscaled(input: x, output: out, size: size, eps: 1e-6, in: $0)
        }

        let expected = Gemma4ReferenceOps.rmsNorm(
            input.map(Double.init), weight: nil, eps: 1e-6)
        #expect(worst(read(out, count: size), expected) < 1e-5)
    }

    @Test("gelu_pytorch_tanh times up matches the reference")
    func geluMultiply() throws {
        let context = try makeContext()
        let encoder = ForwardEncoder(context: context)
        let size = 2112
        // A range that includes the tails, where a wrong constant shows first.
        let gate = deterministic(size, seed: 0xABCD).map { $0 * 9 }
        let up = deterministic(size, seed: 0x5555)

        guard let g = buffer(context, gate), let u = buffer(context, up),
            let out = buffer(context, [Float](repeating: 0, count: size))
        else { return }

        try run(context) {
            try encoder.geluMultiply(gate: g, up: u, output: out, size: size, in: $0)
        }

        let expected = Gemma4ReferenceOps.mlp(
            gate: gate.map(Double.init), up: up.map(Double.init))
        #expect(worst(read(out, count: size), expected) < 1e-5)
    }

    /// The saturated tail of `tanh`, which is where the kernel actually broke.
    ///
    /// Metal compiles with fast math, so `tanh` goes through `exp(2x)`. Above a gate of about
    /// 10.1 the intermediate overflows to infinity and `inf / inf` is NaN. The test above
    /// scales its gate by **9** — under the threshold by a hair — so every operator test, every
    /// layer test and the whole-model test passed while the first run on real weights returned
    /// 262,144 NaNs. Real Gemma produces gates of 11 at layer 26 of 30.
    ///
    /// The range here is chosen to sit past the cliff on purpose, and finiteness is asserted
    /// separately from accuracy: a NaN compares false against everything, so a test that only
    /// checks closeness can fail for the wrong reason and be read as a tolerance problem.
    @Test("gelu_pytorch_tanh stays finite where tanh saturates")
    func geluSaturates() throws {
        let context = try makeContext()
        let encoder = ForwardEncoder(context: context)
        let size = 2112
        // Well past the overflow threshold, and negative values too: the cubic term flips sign
        // and the same overflow waits on the other side.
        let gate = deterministic(size, seed: 0xABCD).map { $0 * 40 }
        let up = deterministic(size, seed: 0x5555).map { $0 * 4 }
        #expect(gate.contains { $0 > 12 } && gate.contains { $0 < -12 })

        guard let g = buffer(context, gate), let u = buffer(context, up),
            let out = buffer(context, [Float](repeating: 0, count: size))
        else { return }

        try run(context) {
            try encoder.geluMultiply(gate: g, up: u, output: out, size: size, in: $0)
        }

        let got = read(out, count: size)
        #expect(got.allSatisfy { $0.isFinite }, "the saturated tail produced a non-finite value")

        let expected = Gemma4ReferenceOps.mlp(
            gate: gate.map(Double.init), up: up.map(Double.init))
        #expect(worst(got, expected) < 1e-4)
    }

    /// The same overflow, in the one kernel guaranteed to meet large inputs: softcapping exists
    /// precisely because the logits ran away.
    @Test("Softcapping stays finite on logits far outside the cap")
    func softcapSaturates() throws {
        let context = try makeContext()
        let encoder = ForwardEncoder(context: context)
        let size = 4096
        let logits = deterministic(size, seed: 0x99).map { $0 * 4000 }

        guard let out = buffer(context, logits) else { return }
        try run(context) {
            try encoder.softcapLogits(out, size: size, cap: 30, in: $0)
        }

        let got = read(out, count: size)
        #expect(got.allSatisfy { $0.isFinite })
        #expect(got.allSatisfy { abs($0) <= 30 })
        let expected = Gemma4ReferenceOps.softcap(logits.map(Double.init), cap: 30)
        #expect(worst(got, expected) < 1e-4)
    }

    @Test("Logit softcapping matches and bounds the range")
    func softcap() throws {
        let context = try makeContext()
        let encoder = ForwardEncoder(context: context)
        let size = 4096
        let logits = deterministic(size, seed: 0x77).map { $0 * 200 }

        guard let out = buffer(context, logits) else { return }
        try run(context) {
            try encoder.softcapLogits(out, size: size, cap: 30, in: $0)
        }

        let got = read(out, count: size)
        let expected = Gemma4ReferenceOps.softcap(logits.map(Double.init), cap: 30)
        #expect(worst(got, expected) < 1e-5)
        #expect(got.allSatisfy { abs($0) < 30 })
        // The inputs really did exceed the cap, or this proves nothing.
        #expect(logits.contains { abs($0) > 30 })
    }

    // MARK: - The claim that avoided a fourth kernel

    /// Gemma has no attention sinks. `attention_decode` seeds its online softmax on one, and
    /// the claim is that an unreachable sink is indistinguishable from having none: the sink's
    /// term becomes `exp(−1e30 − max) = 0` while the denominator stays at one.
    ///
    /// That claim saved writing a second attention kernel, so it is measured rather than
    /// asserted in a comment.
    @Test("An unreachable sink is equivalent to having no sinks")
    func unreachableSinkEqualsNoSinks() throws {
        let context = try makeContext()
        let encoder = ForwardEncoder(context: context)

        let heads = 4, headDim = 64, keyCount = 24
        let query = deterministic(heads * headDim, seed: 0x2468)
        let keys = (0..<keyCount).map { deterministic(headDim, seed: 0x3000 + UInt32($0)) }
        let values = (0..<keyCount).map { deterministic(headDim, seed: 0x4000 + UInt32($0)) }

        // One key/value head shared by every query head, laid out as the cache expects.
        var keyCache: [Float16] = []
        var valueCache: [Float16] = []
        for i in 0..<keyCount {
            keyCache += keys[i].map(Float16.init)
            valueCache += values[i].map(Float16.init)
        }

        guard let q = buffer(context, query),
            let out = buffer(context, [Float](repeating: 0, count: heads * headDim)),
            let kBuffer = context.device.makeBuffer(
                length: keyCache.count * 2, options: .storageModeShared),
            let vBuffer = context.device.makeBuffer(
                length: valueCache.count * 2, options: .storageModeShared),
            let sinks = context.device.makeBuffer(length: 256, options: .storageModeShared)
        else { return }

        keyCache.withUnsafeBytes { kBuffer.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        valueCache.withUnsafeBytes { vBuffer.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }

        // BF16 for −1e30: an exponent far below anything a score can reach.
        let unreachable = BF16.fromFloat(-1e30)
        let sinkPointer = sinks.contents().bindMemory(to: UInt16.self, capacity: heads)
        for i in 0..<heads { sinkPointer[i] = unreachable }

        try run(context) {
            try encoder.attention(
                query: q, queryOffset: 0, keyCache: kBuffer, valueCache: vBuffer,
                sinks: sinks, sinksOffset: 0, output: out, outputOffset: 0,
                qHeads: heads, kvHeads: 1, headDim: headDim, keyCount: keyCount,
                ringSize: 0, startPosition: 0, smScale: 1.0, in: $0)
        }

        let got = read(out, count: heads * headDim)
        for head in 0..<heads {
            let slice = Array(query[(head * headDim)..<((head + 1) * headDim)])
            let expected = Gemma4ReferenceOps.attention(
                query: slice.map(Double.init),
                keys: keys.map { $0.map(Double.init) },
                values: values.map { $0.map(Double.init) })
            let actual = Array(got[(head * headDim)..<((head + 1) * headDim)])
            #expect(worst(actual, expected) < 2e-3,
                "head \(head): the sink is not neutral, so Gemma needs its own kernel")
        }
    }

    /// The other claim that avoided a kernel: partial rotation is a zero-padded frequency
    /// table, so `rope_apply` needs no knowledge of it. A zero frequency must leave its pair
    /// untouched.
    @Test("A zero frequency leaves its pair exactly untouched")
    func zeroFrequencyIsIdentity() {
        let headDim = 512
        let rotating = 64
        let frequencies = Gemma4ReferenceOps.inverseFrequencies(
            headDim: headDim, theta: 1_000_000, rotatingPairs: rotating)
        let input = deterministic(headDim, seed: 0x9999).map(Double.init)
        let rotated = Gemma4ReferenceOps.applyRoPE(
            input, position: 12_345, frequencies: frequencies)

        let half = headDim / 2
        for i in rotating..<half {
            #expect(rotated[i] == input[i])
            #expect(rotated[i + half] == input[i + half])
        }
        // And the rotated part really did move.
        #expect(rotated[0] != input[0])
    }

    /// The router's chain differs from GPT-OSS's at every step, and the same logits give
    /// different weights. Measured against the oracle, and against GPT-OSS's kernel on the
    /// same input, so "different" is demonstrated rather than asserted.
    @Test("Gemma's router selection matches the reference and differs from GPT-OSS's")
    func routerSelection() throws {
        let context = try makeContext()
        let encoder = ForwardEncoder(context: context)
        let experts = 128, topK = 8

        let logits = deterministic(experts, seed: 0xC0FFEE).map { $0 * 4 }
        let scales = (0..<experts).map { Float(1.0 + 0.01 * Double($0)) }

        guard let logitBuffer = buffer(context, logits),
            let indices = context.device.makeBuffer(length: 256, options: .storageModeShared),
            let weights = context.device.makeBuffer(length: 256, options: .storageModeShared),
            let scaleBuffer = context.device.makeBuffer(
                length: max(experts * 2, 256), options: .storageModeShared)
        else { return }

        let scalePointer = scaleBuffer.contents().bindMemory(to: UInt16.self, capacity: experts)
        for i in 0..<experts { scalePointer[i] = BF16.fromFloat(scales[i]) }

        try run(context) {
            try encoder.gemmaRouterTopK(
                logits: logitBuffer, perExpertScale: scaleBuffer, perExpertScaleOffset: 0,
                indices: indices, weights: weights,
                expertCount: experts, topK: topK, in: $0)
        }

        let gotIndices = Array(UnsafeBufferPointer(
            start: indices.contents().bindMemory(to: UInt32.self, capacity: topK), count: topK))
        let gotWeights = read(weights, count: topK)

        // The oracle uses an identity projection so the logits pass straight through.
        let identity = (0..<experts).map { e in
            (0..<experts).map { $0 == e ? 1.0 : 0.0 }
        }
        let (expectedIndices, expectedWeights) = Gemma4ReferenceOps.router(
            hidden: logits.map(Double.init), projection: identity,
            scale: [Double](repeating: 1, count: experts),
            perExpertScale: scales.map(Double.init), topK: topK, eps: 1e-6)

        // The reference normalizes its input; the kernel receives logits directly, so compare
        // the **selection** exactly and the weights' shape rather than their absolute values.
        #expect(gotIndices.map(Int.init) == expectedIndices,
            "the experts chosen must match, whatever the scaling of the logits")
        #expect(abs(gotWeights.reduce(0, +) - 1.0) > 1e-6,
            "the per-expert scale must break the sum to one")
        #expect(gotWeights.allSatisfy { $0 > 0 && $0.isFinite })

        // GPT-OSS's kernel on the same logits: same experts, different weights.
        guard let gptIndices = context.device.makeBuffer(length: 256, options: .storageModeShared),
            let gptWeights = context.device.makeBuffer(length: 256, options: .storageModeShared)
        else { return }
        try run(context) {
            try encoder.routerTopK(
                logits: logitBuffer, logitsOffset: 0,
                indices: gptIndices, weights: gptWeights,
                expertCount: experts, topK: topK, in: $0)
        }
        let otherWeights = read(gptWeights, count: topK)
        #expect(zip(gotWeights, otherWeights).contains { abs($0 - $1) > 1e-4 },
            "the two routers must not agree, or one of them is wrong")
    }
}
