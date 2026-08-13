import Foundation
import Testing

@testable import HydraReference

/// The composition, not the operators.
///
/// Each operator is already checked against an independent transcription. What this pins is the
/// order they run in and what feeds what, which is where Gemma cost the most time: every
/// operator correct, composed wrongly, and the model answers fluently (D-022).
///
/// Every assertion here is a property that can be stated without a fixture, so it says what the
/// block means rather than what it happened to compute.
@Suite("Qwen linear block composition")
struct QwenReferenceLayerTests {

    private let shape = QwenReferenceLayer.Shape(
        hiddenSize: 8, keyHeads: 2, valueHeads: 4, keyDim: 4, valueDim: 3)

    private func deterministic(_ count: Int, _ seed: UInt64) -> [Double] {
        var state = seed | 1
        return (0..<count).map { _ in
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Double(Int(state >> 33) % 2000 - 1000) / 1000
        }
    }

    private func matrix(_ rows: Int, _ cols: Int, _ seed: UInt64) -> [[Double]] {
        (0..<rows).map { deterministic(cols, seed &+ UInt64($0)) }
    }

    private func makeLayer() -> QwenReferenceLayer {
        let weights = QwenReferenceLayer.Weights(
            inputNorm: deterministic(shape.hiddenSize, 0x1).map { $0 + 1 },
            qkv: matrix(shape.convDim, shape.hiddenSize, 0x100),
            z: matrix(shape.zDim, shape.hiddenSize, 0x200),
            a: matrix(shape.valueHeads, shape.hiddenSize, 0x300),
            b: matrix(shape.valueHeads, shape.hiddenSize, 0x400),
            outProj: matrix(shape.hiddenSize, shape.zDim, 0x500),
            convWeight: matrix(shape.convDim, shape.convKernel, 0x600),
            convBias: deterministic(shape.convDim, 0x700),
            logA: deterministic(shape.valueHeads, 0x800),
            dtBias: deterministic(shape.valueHeads, 0x900),
            normWeight: deterministic(shape.valueDim, 0xA00).map { $0 + 1 })
        return QwenReferenceLayer(shape: shape, weights: weights)
    }

    /// The block carries state, so the same input twice gives different answers.
    ///
    /// This is the property that separates a recurrence from attention, and a block that
    /// silently dropped its state would look perfectly reasonable on any single token.
    @Test("The same token twice gives different outputs, because state is carried")
    func stateIsCarried() {
        let layer = makeLayer()
        var state = QwenReferenceLayer.State(shape: shape)
        let x = deterministic(shape.hiddenSize, 0xBEEF)

        let first = layer.forward(x, state: &state)
        let second = layer.forward(x, state: &state)

        let difference = zip(first, second).map { abs($0 - $1) }.max() ?? 0
        #expect(difference > 1e-6, "the second token must see what the first wrote")
    }

    /// A fresh block reproduces a carried one only from its own beginning.
    @Test("Replaying from a fresh state reproduces the sequence exactly")
    func replayMatches() {
        let layer = makeLayer()
        let tokens = (0..<5).map { deterministic(shape.hiddenSize, 0xC000 + UInt64($0)) }

        var a = QwenReferenceLayer.State(shape: shape)
        let once = tokens.map { layer.forward($0, state: &a) }

        var b = QwenReferenceLayer.State(shape: shape)
        let twice = tokens.map { layer.forward($0, state: &b) }

        for (index, pair) in zip(once, twice).enumerated() {
            let difference = zip(pair.0, pair.1).map { abs($0 - $1) }.max() ?? 0
            #expect(difference == 0, "token \(index) is not reproducible")
        }
    }

    /// The residual is the block's **input**, not its normalized form.
    ///
    /// Adding the normalized value instead is the classic pre-norm slip: the output stays
    /// finite, the scale is subtly wrong, and nothing raises. Detected here by making the norm
    /// weights zero, which makes the whole branch vanish and leaves the residual alone.
    @Test("The residual carries the input, not the normalized input")
    func residualIsTheInput() {
        var weights = makeLayer().weights
        // A zero out-projection kills everything the branch computes, whatever it computed.
        weights.outProj = weights.outProj.map { $0.map { _ in 0.0 } }
        let layer = QwenReferenceLayer(shape: shape, weights: weights)

        var state = QwenReferenceLayer.State(shape: shape)
        let x = deterministic(shape.hiddenSize, 0xD00D)
        let out = layer.forward(x, state: &state)

        for i in 0..<shape.hiddenSize {
            #expect(
                abs(out[i] - x[i]) < 1e-12,
                "with the branch silenced the output is the input, not a normalized copy")
        }
    }

    /// The convolution's window holds what went **into** it, not what came out.
    ///
    /// The reference feeds the window the pre-activation projection. Storing the convolved,
    /// SiLU'd output instead would compound the activation on every token, which grows slowly
    /// and looks like drift rather than like a bug.
    @Test("The window holds the projection, not the convolution's output")
    func windowHoldsPreConvolution() {
        let layer = makeLayer()
        var state = QwenReferenceLayer.State(shape: shape)
        let x = deterministic(shape.hiddenSize, 0xE1)
        _ = layer.forward(x, state: &state)

        let normed = Gemma4ReferenceOps.rmsNorm(
            x, weight: layer.weights.inputNorm, eps: shape.eps)
        let expected = layer.weights.qkv.map { row in
            zip(row, normed).reduce(0) { $0 + $1.0 * $1.1 }
        }
        #expect(state.window.count == 1)
        let difference = zip(state.window[0], expected).map { abs($0 - $1) }.max() ?? 0
        #expect(difference < 1e-12, "the window stored the convolution's output")
    }

    /// The block, composed a second time inside the test, and compared.
    ///
    /// The property tests above pin what the block *means* and missed two things that are pure
    /// bookkeeping: which slice of `z` gates which head, and which key head a value head reads.
    /// Both were injected as deliberate errors and both passed, because no assertion looked at
    /// per-head routing.
    ///
    /// This recomputes the whole block from the operators, independently, which catches any
    /// misrouting by construction rather than by having thought of it. The duplication is the
    /// same discipline as the Python transcription: written twice, compared.
    @Test("The block equals an independent composition of its operators")
    func matchesIndependentComposition() {
        let layer = makeLayer()
        let w = layer.weights
        var state = QwenReferenceLayer.State(shape: shape)
        var window: [[Double]] = []
        var recurrent = (0..<shape.valueHeads).map { _ in
            [[Double]](
                repeating: [Double](repeating: 0, count: shape.valueDim), count: shape.keyDim)
        }

        for token in 0..<4 {
            let x = deterministic(shape.hiddenSize, 0x2200 + UInt64(token))
            let got = layer.forward(x, state: &state)

            // The same block, spelled out.
            let normed = Gemma4ReferenceOps.rmsNorm(x, weight: w.inputNorm, eps: shape.eps)
            func project(_ m: [[Double]], _ v: [Double]) -> [Double] {
                m.map { row in zip(row, v).reduce(0) { $0 + $1.0 * $1.1 } }
            }
            let qkvRaw = project(w.qkv, normed)
            let z = project(w.z, normed)
            let aVals = project(w.a, normed)
            let bVals = project(w.b, normed)

            let qkv = QwenReferenceOps.causalDepthwiseConv(
                input: qkvRaw, history: window, weight: w.convWeight, bias: w.convBias)
            window.append(qkvRaw)
            if window.count > shape.convKernel - 1 { window.removeFirst() }

            let keySpan = shape.keyHeads * shape.keyDim
            var mixed = [Double](repeating: 0, count: shape.zDim)
            for head in 0..<shape.valueHeads {
                // The grouping: value heads share key heads in contiguous blocks, so head 0 and
                // 1 read key head 0 when the factor is two. `head % keyHeads` interleaves them
                // instead and is a different mapping for every head but the first.
                let keyHead = head / (shape.valueHeads / shape.keyHeads)
                let qStart = keyHead * shape.keyDim
                let kStart = keySpan + keyHead * shape.keyDim
                let vStart = 2 * keySpan + head * shape.valueDim
                let out = QwenReferenceOps.deltaRuleStep(
                    query: Array(qkv[qStart..<(qStart + shape.keyDim)]),
                    key: Array(qkv[kStart..<(kStart + shape.keyDim)]),
                    value: Array(qkv[vStart..<(vStart + shape.valueDim)]),
                    decay: QwenReferenceOps.decay(
                        a: aVals[head], logA: w.logA[head], dtBias: w.dtBias[head]),
                    beta: QwenReferenceOps.sigmoid(bVals[head]),
                    state: &recurrent[head], eps: shape.eps)
                let gated = QwenReferenceOps.gatedRMSNorm(
                    out, weight: w.normWeight,
                    // This head's own slice of the gate, not the first head's.
                    gate: Array(z[(head * shape.valueDim)..<((head + 1) * shape.valueDim)]),
                    eps: shape.eps)
                for i in 0..<shape.valueDim { mixed[head * shape.valueDim + i] = gated[i] }
            }
            let expected = zip(x, project(w.outProj, mixed)).map(+)

            let difference = zip(got, expected).map { abs($0 - $1) }.max() ?? 0
            #expect(difference < 1e-12, "token \(token) diverges from the spelled-out block")
        }
    }

    /// The window fills to `kernel - 1` and then slides.
    @Test("The window slides once full")
    func windowSlides() {
        let layer = makeLayer()
        var state = QwenReferenceLayer.State(shape: shape)
        for token in 0..<6 {
            _ = layer.forward(deterministic(shape.hiddenSize, 0xF0 + UInt64(token)), state: &state)
            #expect(state.window.count == min(token + 1, shape.convKernel - 1))
        }
    }
}
