import Foundation
import Testing

@testable import HydraReference

/// Qwen's linear attention against an independent Python transcription of the reference.
///
/// The Swift in `QwenReferenceOps` and the Python in `tools/gen_qwen_fixtures.py` were written
/// separately from `modeling_qwen3_5.py`. Agreement between them is the only evidence that the
/// semantics recorded in D-027 were read correctly, and it is what every future Metal kernel
/// will be measured against. If this suite is wrong, everything downstream is wrong and
/// consistent.
///
/// Five details here fail silently rather than loudly, and each has its own test: the norm on q
/// and k is l2 and not RMS, the scale uses the key dimension and lands after that norm, the
/// decay precedes the read, the gate is built from two learned parameters, and the output norm
/// is gated outside its own variance.
@Suite("Qwen linear attention reference")
struct QwenReferenceTests {

    struct Fixtures: Decodable {
        let keyDim: Int
        let valueDim: Int
        let tokens: Int
        let eps: Double
        let logA: Double
        let dtBias: Double
        let query: [[Double]]
        let key: [[Double]]
        let value: [[Double]]
        let a: [Double]
        let b: [Double]
        let expectedDecays: [Double]
        let expectedBetas: [Double]
        let expectedOutputs: [[Double]]
        let expectedFinalState: [[Double]]
        let normWeight: [Double]
        let gate: [Double]
        let expectedGatedNorm: [Double]
        let convChannels: Int
        let convKernel: Int
        let convInput: [[Double]]
        let convWeight: [[Double]]
        let convBias: [Double]
        let expectedConv: [[Double]]
    }

    static let fixtures: Fixtures = {
        guard let url = Bundle.module.url(
            forResource: "Fixtures/qwen_linear_attention", withExtension: "json")
        else { fatalError("the Qwen fixtures are missing: run tools/gen_qwen_fixtures.py") }
        return try! JSONDecoder().decode(Fixtures.self, from: Data(contentsOf: url))
    }()

    private func deviation(_ a: [Double], _ b: [Double]) -> Double {
        precondition(a.count == b.count)
        return zip(a, b).map { Swift.abs($0 - $1) }.max() ?? 0
    }

    /// The recurrence over a whole sequence, state carried.
    ///
    /// Deliberately not a single step. A kernel that decays after reading, or that fails to
    /// carry the state between tokens, agrees on token zero and diverges from token one, so a
    /// one-token fixture would pass on both bugs.
    @Test("The delta rule matches the reference across a sequence")
    func deltaRuleSequence() {
        let f = Self.fixtures
        var state = [[Double]](
            repeating: [Double](repeating: 0, count: f.valueDim), count: f.keyDim)

        for t in 0..<f.tokens {
            let g = QwenReferenceOps.decay(a: f.a[t], logA: f.logA, dtBias: f.dtBias)
            let beta = QwenReferenceOps.sigmoid(f.b[t])
            #expect(Swift.abs(g - f.expectedDecays[t]) < 1e-12, "decay at token \(t)")
            #expect(Swift.abs(beta - f.expectedBetas[t]) < 1e-12, "beta at token \(t)")

            let out = QwenReferenceOps.deltaRuleStep(
                query: f.query[t], key: f.key[t], value: f.value[t],
                decay: g, beta: beta, state: &state, eps: f.eps)
            #expect(
                deviation(out, f.expectedOutputs[t]) < 1e-12,
                "output diverges at token \(t), where the carried state first matters")
        }

        for i in 0..<f.keyDim {
            #expect(
                deviation(state[i], f.expectedFinalState[i]) < 1e-12,
                "the carried state diverges at key row \(i)")
        }
    }

    /// The norm on q and k is l2, not RMS.
    ///
    /// They differ by exactly `sqrt(count)`, and `Gemma4ReferenceOps.rmsNorm` is sitting right
    /// there in the same module. Reusing it would leave every output finite and wrong by that
    /// factor, which is why this is asserted rather than assumed.
    @Test("The q and k norm is l2 and not RMS")
    func l2IsNotRMS() {
        let x = [0.5, -1.5, 2.0, 0.25]
        let l2 = QwenReferenceOps.l2Norm(x, eps: 0)
        let rms = Gemma4ReferenceOps.rmsNorm(x, weight: nil, eps: 0)

        var sum = 0.0
        for v in x { sum += v * v }
        #expect(Swift.abs(l2.reduce(0) { $0 + $1 * $1 } - 1.0) < 1e-12, "l2 gives a unit vector")

        let ratio = rms[0] / l2[0]
        #expect(
            Swift.abs(ratio - Double(x.count).squareRoot()) < 1e-12,
            "the two norms differ by sqrt(count), which is the size of getting this wrong")
    }

    /// The scale uses the key dimension and is applied after the l2 norm.
    ///
    /// Scaling before normalizing cancels: the norm would divide it straight back out. So a
    /// kernel that does it in the wrong order produces a query that is too large by exactly
    /// `sqrt(keyDim)`, and this pins which side of the norm it belongs on.
    @Test("The query scale survives the norm, so it must follow it")
    func scaleFollowsTheNorm() {
        let q = [1.0, 2.0, -0.5, 0.75]
        let scale = 1.0 / Double(q.count).squareRoot()

        // Epsilon off, so the identities below are exact and the tolerance means something.
        let after = QwenReferenceOps.l2Norm(q, eps: 0).map { $0 * scale }
        let before = QwenReferenceOps.l2Norm(q.map { $0 * scale }, eps: 0)

        let magnitudeAfter = after.reduce(0) { $0 + $1 * $1 }.squareRoot()
        let magnitudeBefore = before.reduce(0) { $0 + $1 * $1 }.squareRoot()
        #expect(Swift.abs(magnitudeAfter - scale) < 1e-12)
        #expect(
            Swift.abs(magnitudeBefore - 1.0) < 1e-12,
            "scaling first is undone by the norm, so the order is not a matter of taste")
    }

    /// The decay multiplies the state before `kv_mem` reads it.
    ///
    /// Swapping those two lines is finite and wrong. Here the two orders are computed and
    /// asserted to differ, so a kernel that gets it backwards cannot pass by coincidence.
    @Test("Decaying after the read gives a different answer")
    func decayPrecedesTheRead() {
        let key = [1.0, 0.0]
        let value = [1.0, 1.0]
        let query = [1.0, 0.0]
        let g = 0.5
        let beta = 1.0

        // A state that already holds something, or the decay has nothing to act on.
        var correct = [[0.4, 0.2], [0.1, 0.3]]
        let out = QwenReferenceOps.deltaRuleStep(
            query: query, key: key, value: value, decay: g, beta: beta, state: &correct)

        // The same step with the decay moved after the read.
        var swapped = [[0.4, 0.2], [0.1, 0.3]]
        let k = QwenReferenceOps.l2Norm(key)
        let q = QwenReferenceOps.l2Norm(query).map { $0 / Double(key.count).squareRoot() }
        var memory = [0.0, 0.0]
        for i in 0..<2 { for j in 0..<2 { memory[j] += swapped[i][j] * k[i] } }
        for i in 0..<2 { for j in 0..<2 { swapped[i][j] *= g } }
        let delta = (0..<2).map { (value[$0] - memory[$0]) * beta }
        for i in 0..<2 { for j in 0..<2 { swapped[i][j] += k[i] * delta[j] } }
        var wrong = [0.0, 0.0]
        for i in 0..<2 { for j in 0..<2 { wrong[j] += swapped[i][j] * q[i] } }

        #expect(
            deviation(out, wrong) > 1e-3,
            "the two orders must differ, or this fixture cannot detect the mistake")
    }

    @Test("The gated RMS norm matches, with the gate outside the variance")
    func gatedNorm() {
        let f = Self.fixtures
        let got = QwenReferenceOps.gatedRMSNorm(
            f.expectedOutputs[f.tokens - 1], weight: f.normWeight, gate: f.gate, eps: 1e-6)
        #expect(deviation(got, f.expectedGatedNorm) < 1e-12)
    }

    /// The convolution is causal and depthwise, and its window is carried state.
    ///
    /// Run here one token at a time with an explicit history, which is how decoding will run
    /// it. A kernel that treats the window as sequence-local produces different values for the
    /// first three tokens of every turn, and only those.
    @Test("The causal depthwise convolution matches, fed one token at a time")
    func causalConvolution() {
        let f = Self.fixtures
        var history: [[Double]] = []
        for t in 0..<f.tokens {
            let got = QwenReferenceOps.causalDepthwiseConv(
                input: f.convInput[t], history: history,
                weight: f.convWeight, bias: f.convBias)
            #expect(
                deviation(got, f.expectedConv[t]) < 1e-12,
                "convolution diverges at token \(t)")

            history.append(f.convInput[t])
            if history.count > f.convKernel - 1 { history.removeFirst() }
        }
    }
}
