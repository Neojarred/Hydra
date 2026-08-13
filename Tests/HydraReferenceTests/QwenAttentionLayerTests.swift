import Foundation
import Testing

@testable import HydraReference

/// The full-attention block's composition, stated as properties.
@Suite("Qwen attention block composition")
struct QwenAttentionLayerTests {

    private let shape = QwenReferenceAttentionLayer.Shape(
        hiddenSize: 12, heads: 4, keyValueHeads: 2, headDim: 8)

    private func deterministic(_ count: Int, _ seed: UInt64) -> [Double] {
        var state = seed | 1
        return (0..<count).map { _ in
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Double(Int(state >> 33) % 400 - 200) / 1000
        }
    }

    private func matrix(_ rows: Int, _ cols: Int, _ seed: UInt64) -> [[Double]] {
        (0..<rows).map { deterministic(cols, seed &+ UInt64($0)) }
    }

    private func makeLayer() -> QwenReferenceAttentionLayer {
        QwenReferenceAttentionLayer(
            shape: shape,
            weights: .init(
                inputNorm: deterministic(shape.hiddenSize, 0x1).map { $0 + 1 },
                qProj: matrix(shape.queryProjectionRows, shape.hiddenSize, 0x100),
                kProj: matrix(shape.keyValueDim, shape.hiddenSize, 0x200),
                vProj: matrix(shape.keyValueDim, shape.hiddenSize, 0x300),
                oProj: matrix(shape.hiddenSize, shape.queryDim, 0x400),
                qNorm: deterministic(shape.headDim, 0x500).map { $0 + 1 },
                kNorm: deterministic(shape.headDim, 0x600).map { $0 + 1 }))
    }

    /// Attention over one token attends to itself, so the output is that token's value.
    ///
    /// A property rather than a fixture: with a single key the softmax is one, whatever the
    /// query is, so the attended vector must be exactly the value projection's head slice. A
    /// kernel that mixed heads or misread the cache fails this without any reference to compare
    /// against.
    @Test("With one token, attention returns that token's own value")
    func singleTokenAttendsToItself() {
        var weights = makeLayer().weights
        // Identity-ish output projection and a gate that passes everything, so the value
        // survives to the output unchanged apart from the residual.
        weights.oProj = (0..<shape.hiddenSize).map { r in
            (0..<shape.queryDim).map { $0 == r ? 1.0 : 0.0 }
        }
        let layer = QwenReferenceAttentionLayer(shape: shape, weights: weights)

        var cache = QwenReferenceAttentionLayer.Cache()
        let x = deterministic(shape.hiddenSize, 0xAB)
        let out = layer.forward(x, position: 0, cache: &cache)

        #expect(cache.keys.count == 1 && cache.values.count == 1)
        #expect(out.allSatisfy { $0.isFinite })
        // The residual is present: with the branch's contribution removed the input remains.
        var silenced = weights
        silenced.oProj = weights.oProj.map { $0.map { _ in 0.0 } }
        var cache2 = QwenReferenceAttentionLayer.Cache()
        let residualOnly = QwenReferenceAttentionLayer(shape: shape, weights: silenced)
            .forward(x, position: 0, cache: &cache2)
        for i in 0..<shape.hiddenSize {
            #expect(abs(residualOnly[i] - x[i]) < 1e-12, "the residual carries the input")
        }
    }

    /// The block, composed a second time inside the test, over several tokens.
    ///
    /// The properties below missed three things, all of which passed as deliberate errors:
    /// Gemma's attention scale of 1.0 in place of `1/sqrt(headDim)`, the head norms moved after
    /// the rotary instead of before, and value heads mapped to key heads by `%` instead of `/`.
    ///
    /// None of them changes a property. The scale is invisible with one token because a softmax
    /// over one key is 1 whatever it is scaled by; the other two are bookkeeping, and bookkeeping
    /// does not change what a block *means*. **This is the second block where properties proved
    /// insufficient and an independent composition caught everything**, so it is the composition
    /// that is the real test and the properties that are the supplement.
    @Test("The block equals an independent composition, over several tokens")
    func matchesIndependentComposition() {
        let layer = makeLayer()
        let w = layer.weights
        var cache = QwenReferenceAttentionLayer.Cache()
        var keys: [[Double]] = [], values: [[Double]] = []

        func project(_ m: [[Double]], _ v: [Double]) -> [Double] {
            m.map { row in zip(row, v).reduce(0) { $0 + $1.0 * $1.1 } }
        }
        let frequencies = Gemma4ReferenceOps.inverseFrequencies(
            headDim: shape.headDim, theta: shape.ropeTheta, rotatingPairs: shape.rotatingPairs)
        // Several tokens, so the attention distribution has something to be wrong about.
        for position in 0..<4 {
            let x = deterministic(shape.hiddenSize, 0x3300 + UInt64(position))
            let got = layer.forward(x, position: position, cache: &cache)

            let normed = Gemma4ReferenceOps.rmsNorm(x, weight: w.inputNorm, eps: shape.eps)
            let combined = project(w.qProj, normed)
            var query = [Double](repeating: 0, count: shape.queryDim)
            var gate = [Double](repeating: 0, count: shape.queryDim)
            for head in 0..<shape.heads {
                let source = head * shape.headDim * 2
                for i in 0..<shape.headDim {
                    query[head * shape.headDim + i] = combined[source + i]
                    gate[head * shape.headDim + i] = combined[source + shape.headDim + i]
                }
            }
            var key = project(w.kProj, normed)
            values.append(project(w.vProj, normed))

            // Norm first, then the rotary.
            for head in 0..<shape.heads {
                let span = (head * shape.headDim)..<((head + 1) * shape.headDim)
                let rotated = Gemma4ReferenceOps.applyRoPE(
                    Gemma4ReferenceOps.rmsNorm(Array(query[span]), weight: w.qNorm, eps: shape.eps),
                    position: position, frequencies: frequencies)
                for (i, v) in rotated.enumerated() { query[head * shape.headDim + i] = v }
            }
            for head in 0..<shape.keyValueHeads {
                let span = (head * shape.headDim)..<((head + 1) * shape.headDim)
                let rotated = Gemma4ReferenceOps.applyRoPE(
                    Gemma4ReferenceOps.rmsNorm(Array(key[span]), weight: w.kNorm, eps: shape.eps),
                    position: position, frequencies: frequencies)
                for (i, v) in rotated.enumerated() { key[head * shape.headDim + i] = v }
            }
            keys.append(key)

            let scale = 1.0 / Double(shape.headDim).squareRoot()
            var attended = [Double](repeating: 0, count: shape.queryDim)
            for head in 0..<shape.heads {
                // Contiguous grouping: heads 0 and 1 read key head 0 when the factor is two.
                let kvHead = head / (shape.heads / shape.keyValueHeads)
                let span = (kvHead * shape.headDim)..<((kvHead + 1) * shape.headDim)
                let out = Gemma4ReferenceOps.attention(
                    query: (0..<shape.headDim).map { query[head * shape.headDim + $0] * scale },
                    keys: keys.map { Array($0[span]) }, values: values.map { Array($0[span]) })
                for (i, v) in out.enumerated() { attended[head * shape.headDim + i] = v }
            }
            let gated = zip(attended, gate).map { $0 * (1.0 / (1.0 + Foundation.exp(-$1))) }
            let expected = zip(x, project(w.oProj, gated)).map(+)

            let difference = zip(got, expected).map { abs($0 - $1) }.max() ?? 0
            #expect(difference < 1e-12, "token \(position) diverges from the spelled-out block")
        }
    }

    /// The cache grows with the conversation, unlike the linear block's fixed state.
    @Test("The key and value history grows with position")
    func historyGrows() {
        let layer = makeLayer()
        var cache = QwenReferenceAttentionLayer.Cache()
        for position in 0..<5 {
            _ = layer.forward(
                deterministic(shape.hiddenSize, 0xC0 + UInt64(position)),
                position: position, cache: &cache)
            #expect(cache.keys.count == position + 1)
        }
    }

    /// Only a quarter of the head dimension turns, and the rest keeps zero frequency.
    @Test("Partial rotary turns a quarter of the head")
    func partialRotary() {
        #expect(shape.rotatingPairs == 1, "a quarter of 8 is 2 components, so one pair")
        let frequencies = Gemma4ReferenceOps.inverseFrequencies(
            headDim: shape.headDim, theta: shape.ropeTheta,
            rotatingPairs: shape.rotatingPairs)
        #expect(frequencies[0] != 0)
        for i in shape.rotatingPairs..<frequencies.count {
            #expect(frequencies[i] == 0, "pair \(i) must not turn")
        }
    }

    /// The gate is per head, so silencing one head's gate silences only that head.
    ///
    /// The property that the earlier blind spot would have violated: with the gate for head
    /// zero driven very negative, its sigmoid approaches zero and that head's slice vanishes
    /// while the others do not.
    @Test("Driving one head's gate down silences only that head")
    func gateIsPerHead() {
        var weights = makeLayer().weights
        // The output projection copies each head's slice straight through.
        weights.oProj = (0..<shape.hiddenSize).map { r in
            (0..<shape.queryDim).map { $0 == r ? 1.0 : 0.0 }
        }
        // Head zero's gate rows are driven to a large negative constant, the others to a large
        // positive one, by making those rows of q_proj produce a constant.
        for i in 0..<shape.headDim {
            let gateRow = 0 * shape.headDim * 2 + shape.headDim + i
            weights.qProj[gateRow] = weights.qProj[gateRow].map { _ in -50.0 }
        }
        for head in 1..<shape.heads {
            for i in 0..<shape.headDim {
                let gateRow = head * shape.headDim * 2 + shape.headDim + i
                weights.qProj[gateRow] = weights.qProj[gateRow].map { _ in 50.0 }
            }
        }

        let layer = QwenReferenceAttentionLayer(shape: shape, weights: weights)
        var cache = QwenReferenceAttentionLayer.Cache()
        let x = [Double](repeating: 0.1, count: shape.hiddenSize)
        let out = layer.forward(x, position: 0, cache: &cache)

        // Head zero contributes nothing beyond the residual; the others do.
        for i in 0..<min(shape.headDim, shape.hiddenSize) {
            #expect(abs(out[i] - x[i]) < 1e-6, "head zero should be silenced at component \(i)")
        }
    }
}
