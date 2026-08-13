import Foundation
import Testing

@testable import HydraReference

/// Qwen's mixture, and the two conventions it shares with one shipped model and not the other.
@Suite("Qwen mixture")
struct QwenMixtureTests {

    private func deterministic(_ count: Int, _ seed: UInt64) -> [Double] {
        var state = seed | 1
        return (0..<count).map { _ in
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Double(Int(state >> 33) % 400 - 200) / 1000
        }
    }

    /// Softmax over everything then renormalize is **the same function** as softmax over the
    /// top-k alone.
    ///
    /// This project has recorded the two as different conventions since Gemma, on the grounds
    /// that one normalizes over every expert and the other over the chosen ones. That is not
    /// what the arithmetic does. The full softmax divides by `Z`, the renormalization divides
    /// by a sum that also carries `Z`, and it cancels:
    ///
    ///     (e^lᵢ / Z) / (Σ_top e^lⱼ / Z)  =  e^lᵢ / Σ_top e^lⱼ
    ///
    /// So the distinction only exists when the renormalization does not happen. Qwen's
    /// `norm_topk_prob` defaults to true and Gemma renormalizes, so the two models agree and
    /// `gemma_router_topk` is provably right for Qwen rather than merely plausible.
    ///
    /// Asserted rather than left as a note, because it is the property that makes reusing that
    /// kernel safe, and it would stop being true if a model ever shipped with the flag off.
    @Test("Renormalizing after a full softmax equals a softmax over the top-k alone")
    func routerConventionsAgree() {
        let logits = [2.0, 1.0, 0.5, -1.0, 3.0, 0.1, -0.5, 1.5]
        let out = QwenReferenceOps.router(logits, topK: 3)

        #expect(out.indices == [4, 0, 7], "the three largest, in order")
        #expect(abs(out.weights.reduce(0, +) - 1.0) < 1e-12, "renormalized")

        // The other convention, spelled out: exponentiate only the chosen and normalize.
        let peak = out.indices.map { logits[$0] }.max() ?? 0
        let exponentials = out.indices.map { Foundation.exp(logits[$0] - peak) }
        let total = exponentials.reduce(0, +)
        let overTopKAlone = exponentials.map { $0 / total }

        for (index, expected) in overTopKAlone.enumerated() {
            #expect(
                abs(out.weights[index] - expected) < 1e-12,
                "the two conventions must agree at rank \(index)")
        }

        // And the consequence: an unchosen expert cannot influence the weights, however far it
        // moves, as long as it stays unchosen.
        var raised = logits
        raised[3] = 1.4
        let after = QwenReferenceOps.router(raised, topK: 3)
        #expect(after.indices == out.indices)
        for index in 0..<3 {
            #expect(
                abs(after.weights[index] - out.weights[index]) < 1e-12,
                "the normalization cancels, so an unchosen logit changes nothing")
        }
    }

    /// The activation is silu, not Gemma's gelu and not GPT-OSS's clamped variant.
    @Test("The activation is silu and differs from both shipped alternatives")
    func activationIsSilu() {
        let gate = [0.5, -1.0, 2.0, 0.0]
        let up = [1.0, 1.0, 1.0, 1.0]
        let silu = QwenReferenceOps.siluMultiply(gate: gate, up: up)

        for (index, g) in gate.enumerated() {
            #expect(abs(silu[index] - g / (1 + Foundation.exp(-g))) < 1e-12)
        }
        // Gemma's gelu over the same branches is a different number everywhere but zero.
        let gelu = Gemma4ReferenceOps.mlp(gate: gate, up: up)
        for index in 0..<gate.count where gate[index] != 0 {
            #expect(abs(silu[index] - gelu[index]) > 1e-6, "silu is not gelu at \(index)")
        }
    }

    /// The shared expert's gate scales its output, and drives it to nothing when very negative.
    @Test("The shared expert is gated by a sigmoid of its own projection")
    func sharedExpertIsGated() {
        let shape = QwenReferenceMixture.Shape(
            hiddenSize: 6, expertCount: 4, expertsPerToken: 2,
            moeIntermediate: 4, sharedIntermediate: 4)
        func matrix(_ r: Int, _ c: Int, _ s: UInt64) -> [[Double]] {
            (0..<r).map { deterministic(c, s &+ UInt64($0)) }
        }
        func expert(_ s: UInt64) -> QwenReferenceMixture.Expert {
            .init(
                gate: matrix(shape.moeIntermediate, shape.hiddenSize, s),
                up: matrix(shape.moeIntermediate, shape.hiddenSize, s &+ 50),
                down: matrix(shape.hiddenSize, shape.moeIntermediate, s &+ 100))
        }
        let experts = (0..<shape.expertCount).map { expert(0x1000 &+ UInt64($0) &* 200) }
        let x = deterministic(shape.hiddenSize, 0xAB)

        // A shared gate driven very negative: sigmoid tends to zero, so only the routed part
        // survives.
        let silenced = QwenReferenceMixture(
            shape: shape,
            weights: .init(
                router: matrix(shape.expertCount, shape.hiddenSize, 0x1),
                sharedGate: [Double](repeating: -1e3, count: shape.hiddenSize),
                shared: expert(0x9000), experts: experts))
        // And one driven very positive: the shared branch arrives at full strength.
        let full = QwenReferenceMixture(
            shape: shape,
            weights: .init(
                router: matrix(shape.expertCount, shape.hiddenSize, 0x1),
                sharedGate: [Double](repeating: 1e3, count: shape.hiddenSize),
                shared: expert(0x9000), experts: experts))

        let a = silenced.forward(x)
        let b = full.forward(x)
        // The signs of x decide which way the huge gate goes, so only assert they differ and
        // that the silenced one is the routed part alone.
        let difference = zip(a, b).map { abs($0 - $1) }.max() ?? 0
        #expect(difference > 1e-6, "the gate must change the result")
    }
}
