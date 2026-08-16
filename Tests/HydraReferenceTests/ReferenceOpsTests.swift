import Foundation
import Testing

@testable import HydraReference

/// Every operator is compared against a vector produced by an independent transcription of
/// `gpt_oss/torch/model.py`. Writing it twice is deliberate: these semantics cannot be
/// guessed, and a mistake in any of them never raises an exception, it silently degrades
/// the outputs.
struct ReferenceOpsTests {

    struct Fixtures: Decodable {
        struct RMSNorm: Decodable {
            let input: [[Double]], scale: [Double], eps: Double, expected: [[Double]]
        }
        struct Yarn: Decodable {
            let head_dim: Int, base: Double, initial_context_length: Int
            let scaling_factor: Double, ntk_alpha: Double, ntk_beta: Double
            let concentration: Double, inv_freq: [Double]
            let positions: [Int], cos: [[Double]], sin: [[Double]]
        }
        struct Rope: Decodable {
            let positions: [Int], input: [[[Double]]], expected: [[[Double]]]
        }
        struct Swiglu: Decodable {
            let input: [[Double]], alpha: Double, limit: Double, expected: [[Double]]
        }
        struct Sdpa: Decodable {
            let tokens: Int, kv_heads: Int, q_mult: Int, head_dim: Int
            let sm_scale: Double, sliding_window: Int
            let q: [[[[Double]]]], k: [[[Double]]], v: [[[Double]]]
            let sinks: [Double], expected: [[Double]]
        }
        struct Router: Decodable {
            let top_k: Int, logits: [[Double]], indices: [[Int]], weights: [[Double]]
        }
        let rms_norm: RMSNorm
        let yarn: Yarn
        let rope: Rope
        let swiglu: Swiglu
        let sdpa_full: Sdpa
        let sdpa_sliding: Sdpa
        let router: Router
    }

    static let fixtures: Fixtures = {
        guard let url = Bundle.module.url(
            forResource: "Fixtures/reference_ops", withExtension: "json")
        else {
            fatalError("fixtures absentes, lancer tools/gen_reference_fixtures.py")
        }
        return try! JSONDecoder().decode(Fixtures.self, from: Data(contentsOf: url))
    }()

    /// Both implementations are in double precision and follow the same order of operations:
    /// the expected deviation is on the order of machine epsilon.
    static let tolerance = 1e-12

    static func worstRelative(_ actual: [Double], _ expected: [Double]) -> Double {
        var scale = 0.0
        for value in expected { scale = max(scale, abs(value)) }
        var worst = 0.0
        for (a, e) in zip(actual, expected) {
            worst = max(worst, abs(a - e) / max(scale, 1e-9))
        }
        return worst
    }

    @Test("RMSNorm")
    func rmsNorm() {
        let f = Self.fixtures.rms_norm
        for (row, expected) in zip(f.input, f.expected) {
            let actual = ReferenceOps.rmsNorm(row, scale: f.scale, eps: f.eps)
            #expect(Self.worstRelative(actual, expected) < Self.tolerance)
        }
    }

    @Test("YaRN: concentration and inverse frequencies")
    func yarnParameters() {
        let f = Self.fixtures.yarn
        let parameters = ReferenceOps.YarnParameters(
            headDim: f.head_dim, base: f.base,
            initialContextLength: f.initial_context_length,
            scalingFactor: f.scaling_factor, ntkAlpha: f.ntk_alpha, ntkBeta: f.ntk_beta)
        let (concentration, invFreq) = ReferenceOps.yarn(parameters)

        #expect(abs(concentration - f.concentration) < 1e-15)
        // The concentration is 1.3466 for GPT-OSS: forgetting it breaks nothing visible but
        // shifts all of attention.
        #expect(abs(concentration - 1.34657359) < 1e-8)
        #expect(Self.worstRelative(invFreq, f.inv_freq) < Self.tolerance)
    }

    @Test("YaRN: cos and sin tables, up to 100,000 positions")
    func yarnTables() {
        let f = Self.fixtures.yarn
        let parameters = ReferenceOps.YarnParameters(
            headDim: f.head_dim, base: f.base,
            initialContextLength: f.initial_context_length,
            scalingFactor: f.scaling_factor, ntkAlpha: f.ntk_alpha, ntkBeta: f.ntk_beta)
        let (cos, sin) = ReferenceOps.cosSin(positions: f.positions, parameters: parameters)

        for i in f.positions.indices {
            #expect(Self.worstRelative(cos[i], f.cos[i]) < 1e-10, "cos position \(f.positions[i])")
            #expect(Self.worstRelative(sin[i], f.sin[i]) < 1e-10, "sin position \(f.positions[i])")
        }
    }

    @Test("RoPE splits into two halves")
    func rope() {
        let f = Self.fixtures.rope
        let y = Self.fixtures.yarn
        let parameters = ReferenceOps.YarnParameters(
            headDim: y.head_dim, base: y.base,
            initialContextLength: y.initial_context_length,
            scalingFactor: y.scaling_factor, ntkAlpha: y.ntk_alpha, ntkBeta: y.ntk_beta)
        let (cos, sin) = ReferenceOps.cosSin(positions: f.positions, parameters: parameters)

        for (t, token) in f.input.enumerated() {
            for (h, head) in token.enumerated() {
                let actual = ReferenceOps.applyRoPE(head, cos: cos[t], sin: sin[t])
                #expect(Self.worstRelative(actual, f.expected[t][h]) < 1e-10)
            }
        }
    }

    @Test("SwiGLU: even/odd indices, asymmetric clamping, +1 on the linear branch")
    func swiglu() {
        let f = Self.fixtures.swiglu
        for (row, expected) in zip(f.input, f.expected) {
            let actual = ReferenceOps.swiglu(row, alpha: f.alpha, limit: f.limit)
            #expect(actual.count == row.count / 2)
            #expect(Self.worstRelative(actual, expected) < Self.tolerance)
        }
    }

    /// SwiGLU's three traps, isolated so a regression reads clearly.
    @Test("SwiGLU: every departure from the usual formulation is checked")
    func swigluTraps() {
        // Interleaved split: the input [g, l] gives a single output.
        #expect(ReferenceOps.swiglu([0.0, 0.0]).count == 1)

        // Linear branch with +1: at gate = 0 the output is 0 (swish(0) = 0).
        #expect(abs(ReferenceOps.swiglu([0.0, 5.0])[0]) < 1e-15)

        // Asymmetric clamping: the gate is bounded ONLY from above.
        let clampedHigh = ReferenceOps.swiglu([100.0, 0.0])[0]
        let atLimit = ReferenceOps.swiglu([7.0, 0.0])[0]
        #expect(abs(clampedHigh - atLimit) < 1e-15, "the gate must be clamped at 7")

        let veryNegative = ReferenceOps.swiglu([-100.0, 0.0])[0]
        let minusSeven = ReferenceOps.swiglu([-7.0, 0.0])[0]
        #expect(abs(veryNegative - minusSeven) > 1e-9, "the gate must NOT be clamped from below")

        // The swish uses sigmoid(1.702·x), not sigmoid(x).
        let g = 1.0
        let expected = g * (1.0 / (1.0 + exp(-1.702 * g))) * (0.0 + 1.0)
        #expect(abs(ReferenceOps.swiglu([g, 0.0])[0] - expected) < 1e-15)
    }

    @Test("Attention avec puits, en attention pleine")
    func sdpaFull() {
        let f = Self.fixtures.sdpa_full
        let actual = ReferenceOps.sdpa(
            q: f.q, k: f.k, v: f.v, sinks: f.sinks,
            smScale: f.sm_scale, slidingWindow: f.sliding_window)
        for (a, e) in zip(actual, f.expected) {
            #expect(Self.worstRelative(a, e) < 1e-12)
        }
    }

    @Test("Attention with sinks, in a sliding window")
    func sdpaSliding() {
        let f = Self.fixtures.sdpa_sliding
        #expect(f.sliding_window == 3)
        let actual = ReferenceOps.sdpa(
            q: f.q, k: f.k, v: f.v, sinks: f.sinks,
            smScale: f.sm_scale, slidingWindow: f.sliding_window)
        for (a, e) in zip(actual, f.expected) {
            #expect(Self.worstRelative(a, e) < 1e-12)
        }
    }

    /// The sink contributes no value: it only enlarges the denominator. A very negative sink
    /// must therefore give back exactly classical attention.
    @Test("A negligible sink makes attention classical")
    func sinkVanishes() {
        let f = Self.fixtures.sdpa_full
        let withSinks = ReferenceOps.sdpa(
            q: f.q, k: f.k, v: f.v, sinks: f.sinks, smScale: f.sm_scale)
        let negligible = ReferenceOps.sdpa(
            q: f.q, k: f.k, v: f.v,
            sinks: [Double](repeating: -1e9, count: f.sinks.count), smScale: f.sm_scale)

        // With no sink the weights sum to 1: the first token's output is exactly V.
        for i in 0..<f.head_dim {
            #expect(abs(negligible[0][i] - f.v[0][0][i]) < 1e-9)
        }
        // With the real sinks it differs, so the mechanism does act.
        var differs = false
        for i in 0..<f.head_dim where abs(withSinks[0][i] - negligible[0][i]) > 1e-6 {
            differs = true
        }
        #expect(differs, "the sinks change nothing: they are not being taken into account")
    }

    @Test("Router: softmax over the top-k alone")
    func router() {
        let f = Self.fixtures.router
        for (row, index) in zip(f.logits, f.logits.indices) {
            let (indices, weights) = ReferenceOps.router(row, topK: f.top_k)
            #expect(indices == f.indices[index])
            #expect(Self.worstRelative(weights, f.weights[index]) < Self.tolerance)
            #expect(abs(weights.reduce(0, +) - 1.0) < 1e-12, "the weights must sum to 1")
        }
    }
}
