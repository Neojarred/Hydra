import Foundation
import HydraCore
import HydraFormat
import HydraReference
import Testing

@testable import HydraMetal

/// The chain of trust runs from OpenAI to the GPU: an independent transcription of
/// `gpt_oss/torch/model.py` produces reference vectors; `HydraReference` reproduces them
/// to 1e-12; these tests check that the Metal kernels reproduce `HydraReference`. Every
/// link is verified separately.
struct AttentionKernelTests {

    static func deterministic(_ count: Int, seed: UInt64) -> [Float] {
        var state = seed
        return (0..<count).map { _ in
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            z = z ^ (z >> 31)
            return Float(Double(z % 20000) / 10000.0 - 1.0)
        }
    }

    /// Deviation relative to the vector's magnitude, not to each component: near zero, the
    /// per-component relative deviation measures catastrophic cancellation, not the kernel.
    static func deviation(_ actual: [Float], _ expected: [Double]) -> Double {
        var scale = 0.0
        for value in expected { scale = max(scale, abs(value)) }
        var worst = 0.0
        for (a, e) in zip(actual, expected) {
            worst = max(worst, abs(Double(a) - e) / max(scale, 1e-9))
        }
        return worst
    }

    @Test("RMSNorm: the GPU agrees with the CPU reference")
    func rmsNorm() throws {
        let kernels = AttentionKernels(context: try MetalContext())
        let size = 2880  // dimension réelle de GPT-OSS
        let x = Self.deterministic(size, seed: 11)
        let scale = Self.deterministic(size, seed: 12).map { $0 * 0.5 + 1.0 }

        let gpu = try kernels.rmsNorm(x, scale: scale)
        // The scale goes through a BF16 round trip on the GPU side: the reference must see the
        // same values, otherwise we would measure the quantization and not the kernel.
        let quantized = BF16.decode(BF16.encode(scale)).map(Double.init)
        let cpu = ReferenceOps.rmsNorm(x.map(Double.init), scale: quantized, eps: 1e-5)

        #expect(Self.deviation(gpu, cpu) < 1e-6)
    }

    @Test("RoPE: the GPU agrees, split into halves included")
    func rope() throws {
        let kernels = AttentionKernels(context: try MetalContext())
        let heads = 8, headDim = 64
        let x = Self.deterministic(heads * headDim, seed: 21)

        let (cosTable, sinTable) = ReferenceOps.cosSin(positions: [137])
        let cos = cosTable[0].map(Float.init)
        let sin = sinTable[0].map(Float.init)

        let gpu = try kernels.applyRoPE(x, heads: heads, headDim: headDim, cos: cos, sin: sin)

        for head in 0..<heads {
            let slice = Array(x[(head * headDim)..<((head + 1) * headDim)]).map(Double.init)
            let cpu = ReferenceOps.applyRoPE(
                slice, cos: cos.map(Double.init), sin: sin.map(Double.init))
            let actual = Array(gpu[(head * headDim)..<((head + 1) * headDim)])
            #expect(Self.deviation(actual, cpu) < 1e-6, "head \(head)")
        }
    }

    @Test("SwiGLU: the GPU reproduces the asymmetric clamping and the +1")
    func swiglu() throws {
        let kernels = AttentionKernels(context: try MetalContext())
        var x = Self.deterministic(2 * 2880, seed: 31).map { $0 * 12.0 }
        // Values that cross the thresholds in both directions.
        x[0] = 20; x[1] = -20; x[2] = -20; x[3] = 20

        let gpu = try kernels.swiglu(x)
        let cpu = ReferenceOps.swiglu(x.map(Double.init))

        #expect(gpu.count == x.count / 2)
        #expect(Self.deviation(gpu, cpu) < 1e-6)
    }

    @Test("Attention with sinks: the GPU agrees, in full attention")
    func attentionFull() throws {
        let kernels = AttentionKernels(context: try MetalContext())
        let qHeads = 64, kvHeads = 8, headDim = 64, keyCount = 40  // GQA groupe 8, réel
        let qMult = qHeads / kvHeads

        let query = Self.deterministic(qHeads * headDim, seed: 41)
        let k = Self.deterministic(keyCount * kvHeads * headDim, seed: 42)
        let v = Self.deterministic(keyCount * kvHeads * headDim, seed: 43)
        let sinks = Self.deterministic(qHeads, seed: 44)
        let smScale = 1.0 / Float(headDim).squareRoot()

        let gpu = try kernels.attentionDecode(
            query: query, kCache: k.map(Float16.init), vCache: v.map(Float16.init),
            sinks: sinks, qHeads: qHeads, kvHeads: kvHeads, headDim: headDim,
            keyCount: keyCount, smScale: smScale)

        // Reference: a single query, at the last position, against every key.
        // The cache is FP16 and the sinks BF16 on the GPU side: the reference must see the
        // same values so that the deviation measured is the kernel's.
        let kQuantized = k.map { Double(Float16($0)) }
        let vQuantized = v.map { Double(Float16($0)) }
        let sinkQuantized = BF16.decode(BF16.encode(sinks)).map(Double.init)

        for head in 0..<qHeads {
            let kvHead = head / qMult
            var accumulator = [Double](repeating: 0, count: headDim)
            var logits = [Double](repeating: 0, count: keyCount)
            var peak = sinkQuantized[head]
            for key in 0..<keyCount {
                var dot = 0.0
                for i in 0..<headDim {
                    dot += Double(query[head * headDim + i])
                        * kQuantized[(key * kvHeads + kvHead) * headDim + i]
                }
                logits[key] = dot * Double(smScale)
                peak = max(peak, logits[key])
            }
            var denominator = exp(sinkQuantized[head] - peak)
            for value in logits { denominator += exp(value - peak) }
            for key in 0..<keyCount {
                let weight = exp(logits[key] - peak) / denominator
                for i in 0..<headDim {
                    accumulator[i] += weight * vQuantized[(key * kvHeads + kvHead) * headDim + i]
                }
            }
            let actual = Array(gpu[(head * headDim)..<((head + 1) * headDim)])
            #expect(Self.deviation(actual, accumulator) < 1e-4, "head \(head)")
        }
    }

    /// The sliding-window layers' ring must give exactly the same result as linear storage, as
    /// long as the window has not overflowed.
    @Test("The circular cache equals linear storage before overflow")
    func ringMatchesLinear() throws {
        let kernels = AttentionKernels(context: try MetalContext())
        let qHeads = 8, kvHeads = 2, headDim = 64, keyCount = 20
        let ringSize = 256

        let query = Self.deterministic(qHeads * headDim, seed: 51)
        let k = Self.deterministic(ringSize * kvHeads * headDim, seed: 52).map(Float16.init)
        let v = Self.deterministic(ringSize * kvHeads * headDim, seed: 53).map(Float16.init)
        let sinks = Self.deterministic(qHeads, seed: 54)
        let smScale = 1.0 / Float(headDim).squareRoot()

        let linear = try kernels.attentionDecode(
            query: query, kCache: k, vCache: v, sinks: sinks,
            qHeads: qHeads, kvHeads: kvHeads, headDim: headDim, keyCount: keyCount,
            ringSize: 0, startPosition: 0, smScale: smScale)
        let ring = try kernels.attentionDecode(
            query: query, kCache: k, vCache: v, sinks: sinks,
            qHeads: qHeads, kvHeads: kvHeads, headDim: headDim, keyCount: keyCount,
            ringSize: ringSize, startPosition: 0, smScale: smScale)

        #expect(linear == ring)
    }

    /// A very negative sink must stop weighing: attention becomes classical again and the
    /// weights sum to 1. With a single key, the output is then exactly V.
    @Test("A negligible sink makes attention classical")
    func sinkVanishes() throws {
        let kernels = AttentionKernels(context: try MetalContext())
        let qHeads = 4, kvHeads = 1, headDim = 64
        let query = Self.deterministic(qHeads * headDim, seed: 61)
        let k = Self.deterministic(kvHeads * headDim, seed: 62).map(Float16.init)
        let v = Self.deterministic(kvHeads * headDim, seed: 63).map(Float16.init)

        let negligible = try kernels.attentionDecode(
            query: query, kCache: k, vCache: v,
            sinks: [Float](repeating: -1e9, count: qHeads),
            qHeads: qHeads, kvHeads: kvHeads, headDim: headDim, keyCount: 1, smScale: 0.125)

        for head in 0..<qHeads {
            for i in 0..<headDim {
                #expect(abs(negligible[head * headDim + i] - Float(v[i])) < 1e-3)
            }
        }

        // With a zero sink the mass is shared: the output must move away from V.
        let active = try kernels.attentionDecode(
            query: query, kCache: k, vCache: v,
            sinks: [Float](repeating: 0, count: qHeads),
            qHeads: qHeads, kvHeads: kvHeads, headDim: headDim, keyCount: 1, smScale: 0.125)
        var differs = false
        for i in 0..<headDim where abs(active[i] - negligible[i]) > 1e-4 { differs = true }
        #expect(differs, "the sink has no effect: it is not being taken into account")
    }

    @Test("Router: the GPU selects and weights like the reference")
    func routerTopK() throws {
        let kernels = AttentionKernels(context: try MetalContext())
        for expertCount in [32, 128] {
            let logits = Self.deterministic(expertCount, seed: UInt64(70 + expertCount))
            let (indices, weights) = try kernels.routerTopK(logits, topK: 4)
            let cpu = ReferenceOps.router(logits.map(Double.init), topK: 4)

            #expect(indices == cpu.indices, "\(expertCount) experts")
            #expect(Self.deviation(weights, cpu.weights) < 1e-6)
            #expect(abs(weights.reduce(0, +) - 1.0) < 1e-5)
        }
    }
}
