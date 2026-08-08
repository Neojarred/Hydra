import Foundation
import HydraCore

/// A complete Gemma 4 decoder layer, in double precision.
///
/// Ground truth for the Metal layer runner, the way `ReferenceLayer` is for GPT-OSS.
/// Deliberately written as plainly as possible — explicit loops, no reuse of the GPT-OSS path —
/// so that a divergence reads as a GPU bug and never as an ambiguity here.
///
/// **The topology is the point.** Every individual operator is checked elsewhere; what this
/// exists to pin down is the two things D-022 found that no amount of care with operators would
/// catch:
///
/// - the mixture-of-experts branch reads the **residual**, the state *before* the dense MLP.
///   The dense and expert paths are therefore parallel branches over the same input, summed —
///   not a chain;
/// - `post_attention_layernorm` is applied **before** the residual add. That is post-norm,
///   where GPT-OSS is pre-norm, and getting it backwards produces a model that still speaks.
public struct Gemma4ReferenceLayer {

    public struct Weights {
        public var inputLayerNorm: [Double]
        public var queryProjection: [[Double]]
        public var keyProjection: [[Double]]
        /// Absent on full-attention layers, where `attention_k_eq_v` makes V reuse the key
        /// projection's output.
        public var valueProjection: [[Double]]?
        public var outputProjection: [[Double]]
        public var queryNorm: [Double]
        public var keyNorm: [Double]
        public var postAttentionLayerNorm: [Double]

        public var preFeedForwardLayerNorm: [Double]
        public var gateProjection: [[Double]]
        public var upProjection: [[Double]]
        public var downProjection: [[Double]]
        public var postFeedForwardLayerNorm1: [Double]

        public var preFeedForwardLayerNorm2: [Double]
        public var routerProjection: [[Double]]
        public var routerScale: [Double]
        public var routerPerExpertScale: [Double]
        public var postFeedForwardLayerNorm2: [Double]

        public var postFeedForwardLayerNorm: [Double]
        public var layerScalar: Double

        public init(
            inputLayerNorm: [Double], queryProjection: [[Double]], keyProjection: [[Double]],
            valueProjection: [[Double]]?, outputProjection: [[Double]],
            queryNorm: [Double], keyNorm: [Double], postAttentionLayerNorm: [Double],
            preFeedForwardLayerNorm: [Double], gateProjection: [[Double]],
            upProjection: [[Double]], downProjection: [[Double]],
            postFeedForwardLayerNorm1: [Double], preFeedForwardLayerNorm2: [Double],
            routerProjection: [[Double]], routerScale: [Double],
            routerPerExpertScale: [Double], postFeedForwardLayerNorm2: [Double],
            postFeedForwardLayerNorm: [Double], layerScalar: Double
        ) {
            self.inputLayerNorm = inputLayerNorm
            self.queryProjection = queryProjection
            self.keyProjection = keyProjection
            self.valueProjection = valueProjection
            self.outputProjection = outputProjection
            self.queryNorm = queryNorm
            self.keyNorm = keyNorm
            self.postAttentionLayerNorm = postAttentionLayerNorm
            self.preFeedForwardLayerNorm = preFeedForwardLayerNorm
            self.gateProjection = gateProjection
            self.upProjection = upProjection
            self.downProjection = downProjection
            self.postFeedForwardLayerNorm1 = postFeedForwardLayerNorm1
            self.preFeedForwardLayerNorm2 = preFeedForwardLayerNorm2
            self.routerProjection = routerProjection
            self.routerScale = routerScale
            self.routerPerExpertScale = routerPerExpertScale
            self.postFeedForwardLayerNorm2 = postFeedForwardLayerNorm2
            self.postFeedForwardLayerNorm = postFeedForwardLayerNorm
            self.layerScalar = layerScalar
        }
    }

    public struct Expert {
        public var gate: [[Double]]
        public var up: [[Double]]
        public var down: [[Double]]

        public init(gate: [[Double]], up: [[Double]], down: [[Double]]) {
            self.gate = gate
            self.up = up
            self.down = down
        }
    }

    public let weights: Weights
    public let experts: [Expert]
    public let heads: Int
    public let headDim: Int
    public let topK: Int
    public let eps: Double

    public init(
        weights: Weights, experts: [Expert], heads: Int, headDim: Int, topK: Int,
        eps: Double = 1e-6
    ) {
        self.weights = weights
        self.experts = experts
        self.heads = heads
        self.headDim = headDim
        self.topK = topK
        self.eps = eps
    }

    private func matvec(_ matrix: [[Double]], _ x: [Double]) -> [Double] {
        matrix.map { row in
            var sum = 0.0
            for i in 0..<x.count { sum += row[i] * x[i] }
            return sum
        }
    }

    /// One decoding step, with the key/value history already accumulated.
    ///
    /// - Parameters:
    ///   - hidden: the incoming residual state.
    ///   - keys, values: the history, this token's entry included as the last element.
    ///   - frequencies: the inverse frequencies for **this layer's** pattern — partial
    ///     rotation is a zero-padded table, not a separate code path.
    ///   - slidingWindow: 0 for a full-attention layer.
    public func forward(
        hidden input: [Double], position: Int,
        keys: [[Double]], values: [[Double]],
        frequencies: [Double], slidingWindow: Int = 0
    ) -> [Double] {
        var hidden = input

        // --- Attention ---
        var residual = hidden
        let normed = Gemma4ReferenceOps.rmsNorm(
            hidden, weight: weights.inputLayerNorm, eps: eps)

        let queryFlat = matvec(weights.queryProjection, normed)
        var attended: [Double] = []
        for head in 0..<heads {
            let slice = Array(queryFlat[(head * headDim)..<((head + 1) * headDim)])
            // q_norm before RoPE, never after.
            let query = Gemma4ReferenceOps.applyRoPE(
                Gemma4ReferenceOps.rmsNorm(slice, weight: weights.queryNorm, eps: eps),
                position: position, frequencies: frequencies)
            attended += Gemma4ReferenceOps.attention(
                query: query, keys: keys, values: values, slidingWindow: slidingWindow)
        }

        // Post-norm: applied to the attention output, **before** the residual add.
        let projected = Gemma4ReferenceOps.rmsNorm(
            matvec(weights.outputProjection, attended),
            weight: weights.postAttentionLayerNorm, eps: eps)
        hidden = zip(residual, projected).map(+)

        // --- Feed-forward: two branches over the same residual ---
        residual = hidden

        let denseInput = Gemma4ReferenceOps.rmsNorm(
            hidden, weight: weights.preFeedForwardLayerNorm, eps: eps)
        let dense = matvec(
            weights.downProjection,
            Gemma4ReferenceOps.mlp(
                gate: matvec(weights.gateProjection, denseInput),
                up: matvec(weights.upProjection, denseInput)))
        let branch1 = Gemma4ReferenceOps.rmsNorm(
            dense, weight: weights.postFeedForwardLayerNorm1, eps: eps)

        // The router reads the **residual**, not the dense branch's output.
        let (indices, routerWeights) = Gemma4ReferenceOps.router(
            hidden: residual, projection: weights.routerProjection,
            scale: weights.routerScale, perExpertScale: weights.routerPerExpertScale,
            topK: topK, eps: eps)
        let expertInput = Gemma4ReferenceOps.rmsNorm(
            residual, weight: weights.preFeedForwardLayerNorm2, eps: eps)

        var mixed = [Double](repeating: 0, count: hidden.count)
        for (expert, weight) in zip(indices, routerWeights) {
            let e = experts[expert]
            let out = matvec(
                e.down,
                Gemma4ReferenceOps.mlp(
                    gate: matvec(e.gate, expertInput), up: matvec(e.up, expertInput)))
            for i in 0..<mixed.count { mixed[i] += weight * out[i] }
        }
        let branch2 = Gemma4ReferenceOps.rmsNorm(
            mixed, weight: weights.postFeedForwardLayerNorm2, eps: eps)

        let combined = Gemma4ReferenceOps.rmsNorm(
            zip(branch1, branch2).map(+),
            weight: weights.postFeedForwardLayerNorm, eps: eps)
        hidden = zip(residual, combined).map(+)

        // A per-layer scalar, applied last.
        return hidden.map { $0 * weights.layerScalar }
    }

    /// Projects and normalizes this token's key and value.
    ///
    /// Separate from `forward` because the runtime writes them into the cache before attending.
    /// The subtlety worth the separate entry point: when `attention_k_eq_v` applies, V reuses
    /// the key **projection's output** — before `k_norm` and before RoPE — and then takes its
    /// own weightless normalization.
    public func keyValue(
        hidden: [Double], position: Int, frequencies: [Double]
    ) -> (key: [Double], value: [Double]) {
        let normed = Gemma4ReferenceOps.rmsNorm(
            hidden, weight: weights.inputLayerNorm, eps: eps)
        let projected = matvec(weights.keyProjection, normed)

        let key = Gemma4ReferenceOps.applyRoPE(
            Gemma4ReferenceOps.rmsNorm(projected, weight: weights.keyNorm, eps: eps),
            position: position, frequencies: frequencies)

        let valueSource = weights.valueProjection.map { matvec($0, normed) } ?? projected
        // v_norm has no weight, and V never goes through RoPE.
        let value = Gemma4ReferenceOps.rmsNorm(valueSource, weight: nil, eps: eps)
        return (key, value)
    }
}
