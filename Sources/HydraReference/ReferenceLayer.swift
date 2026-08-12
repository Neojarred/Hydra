import Foundation

/// A complete GPT-OSS transformer layer, in double precision.
///
/// Ground truth for `LayerRunner`. Deliberately written as plainly as possible, explicit
/// loops, no tricks, so that a divergence from the GPU reads as a GPU bug, never as an
/// ambiguity in the reference.
public enum ReferenceLayer {

    public struct Expert: Sendable {
        /// `[2 × intermediate][hidden]`, rows **interleaved** `[gate₀, up₀, gate₁, …]`.
        public var gateUp: [[Double]]
        public var gateUpBias: [Double]
        /// `[hidden][intermediate]`
        public var down: [[Double]]
        public var downBias: [Double]

        public init(
            gateUp: [[Double]], gateUpBias: [Double], down: [[Double]], downBias: [Double]
        ) {
            self.gateUp = gateUp
            self.gateUpBias = gateUpBias
            self.down = down
            self.downBias = downBias
        }
    }

    public struct Weights: Sendable {
        public var inputNorm: [Double]
        public var queryWeight: [[Double]]  // [qDim][hidden]
        public var queryBias: [Double]
        public var keyWeight: [[Double]]    // [kvDim][hidden]
        public var keyBias: [Double]
        public var valueWeight: [[Double]]
        public var valueBias: [Double]
        public var outputWeight: [[Double]]  // [hidden][qDim]
        public var outputBias: [Double]
        public var sinks: [Double]           // [qHeads]
        public var postNorm: [Double]
        public var routerWeight: [[Double]]  // [experts][hidden]
        public var routerBias: [Double]
        public var experts: [Expert]

        public init(
            inputNorm: [Double], queryWeight: [[Double]], queryBias: [Double],
            keyWeight: [[Double]], keyBias: [Double],
            valueWeight: [[Double]], valueBias: [Double],
            outputWeight: [[Double]], outputBias: [Double], sinks: [Double],
            postNorm: [Double], routerWeight: [[Double]], routerBias: [Double],
            experts: [Expert]
        ) {
            self.inputNorm = inputNorm
            self.queryWeight = queryWeight
            self.queryBias = queryBias
            self.keyWeight = keyWeight
            self.keyBias = keyBias
            self.valueWeight = valueWeight
            self.valueBias = valueBias
            self.outputWeight = outputWeight
            self.outputBias = outputBias
            self.sinks = sinks
            self.postNorm = postNorm
            self.routerWeight = routerWeight
            self.routerBias = routerBias
            self.experts = experts
        }
    }

    public struct Shape: Sendable {
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let headDim: Int
        public let queryHeads: Int
        public let keyValueHeads: Int
        public let expertsPerToken: Int
        public let slidingWindow: Int
        public let rmsNormEps: Double
        public let swigluLimit: Double

        public init(
            hiddenSize: Int, intermediateSize: Int, headDim: Int,
            queryHeads: Int, keyValueHeads: Int, expertsPerToken: Int,
            slidingWindow: Int, rmsNormEps: Double = 1e-5, swigluLimit: Double = 7.0
        ) {
            self.hiddenSize = hiddenSize
            self.intermediateSize = intermediateSize
            self.headDim = headDim
            self.queryHeads = queryHeads
            self.keyValueHeads = keyValueHeads
            self.expertsPerToken = expertsPerToken
            self.slidingWindow = slidingWindow
            self.rmsNormEps = rmsNormEps
            self.swigluLimit = swigluLimit
        }
    }

    /// The key and value history, keys already through RoPE.
    public struct Cache: Sendable {
        public var keys: [[Double]] = []    // [position][kvDim]
        public var values: [[Double]] = []
        public init() {}
    }

    private static func matVec(_ w: [[Double]], _ x: [Double], _ bias: [Double]) -> [Double] {
        var out = [Double](repeating: 0, count: w.count)
        for row in 0..<w.count {
            var sum = 0.0
            let line = w[row]
            for i in 0..<x.count { sum += line[i] * x[i] }
            out[row] = sum + bias[row]
        }
        return out
    }

    /// One decoding step: a single token, at the given position.
    ///
    /// - Parameters:
    ///   - hidden: the incoming residual state, modified and returned.
    ///   - sliding: true for sliding-window layers (even indices).
    public static func decode(
        hidden input: [Double], weights: Weights, shape: Shape,
        cache: inout Cache, position: Int, sliding: Bool,
        rope: (cos: [Double], sin: [Double])
    ) -> [Double] {
        var hidden = input
        let qDim = shape.queryHeads * shape.headDim
        let kvDim = shape.keyValueHeads * shape.headDim
        let qMult = shape.queryHeads / shape.keyValueHeads

        // --- Attention ---
        var normed = ReferenceOps.rmsNorm(hidden, scale: weights.inputNorm, eps: shape.rmsNormEps)
        var query = matVec(weights.queryWeight, normed, weights.queryBias)
        var key = matVec(weights.keyWeight, normed, weights.keyBias)
        let value = matVec(weights.valueWeight, normed, weights.valueBias)

        // RoPE head by head, split into two halves.
        for head in 0..<shape.queryHeads {
            let slice = Array(query[(head * shape.headDim)..<((head + 1) * shape.headDim)])
            let rotated = ReferenceOps.applyRoPE(slice, cos: rope.cos, sin: rope.sin)
            for i in 0..<shape.headDim { query[head * shape.headDim + i] = rotated[i] }
        }
        for head in 0..<shape.keyValueHeads {
            let slice = Array(key[(head * shape.headDim)..<((head + 1) * shape.headDim)])
            let rotated = ReferenceOps.applyRoPE(slice, cos: rope.cos, sin: rope.sin)
            for i in 0..<shape.headDim { key[head * shape.headDim + i] = rotated[i] }
        }

        cache.keys.append(key)
        cache.values.append(value)

        // Visible positions: causal, bounded by the window if it is active.
        let firstVisible = sliding ? max(0, position - shape.slidingWindow + 1) : 0
        let smScale = 1.0 / Double(shape.headDim).squareRoot()

        var attention = [Double](repeating: 0, count: qDim)
        for head in 0..<shape.queryHeads {
            let kvHead = head / qMult
            let sink = weights.sinks[head]

            var logits: [Double] = []
            var peak = sink
            for key in firstVisible...position {
                var dot = 0.0
                for i in 0..<shape.headDim {
                    dot += query[head * shape.headDim + i]
                        * cache.keys[key][kvHead * shape.headDim + i]
                }
                let logit = dot * smScale
                logits.append(logit)
                peak = max(peak, logit)
            }
            // The sink weighs in the denominator, never in the numerator.
            var denominator = exp(sink - peak)
            for logit in logits { denominator += exp(logit - peak) }

            for (index, key) in (firstVisible...position).enumerated() {
                let weight = exp(logits[index] - peak) / denominator
                for i in 0..<shape.headDim {
                    attention[head * shape.headDim + i] +=
                        weight * cache.values[key][kvHead * shape.headDim + i]
                }
            }
        }
        _ = kvDim

        let projected = matVec(weights.outputWeight, attention, weights.outputBias)
        for i in 0..<shape.hiddenSize { hidden[i] += projected[i] }

        // --- Mixture of experts ---
        normed = ReferenceOps.rmsNorm(hidden, scale: weights.postNorm, eps: shape.rmsNormEps)
        let logits = matVec(weights.routerWeight, normed, weights.routerBias)
        let (indices, routerWeights) = ReferenceOps.router(logits, topK: shape.expertsPerToken)

        var mixture = [Double](repeating: 0, count: shape.hiddenSize)
        for (slot, expertIndex) in indices.enumerated() {
            let expert = weights.experts[expertIndex]
            let gateUp = matVec(expert.gateUp, normed, expert.gateUpBias)
            let activated = ReferenceOps.swiglu(gateUp, limit: shape.swigluLimit)
            let down = matVec(expert.down, activated, expert.downBias)
            for i in 0..<shape.hiddenSize { mixture[i] += routerWeights[slot] * down[i] }
        }
        for i in 0..<shape.hiddenSize { hidden[i] += mixture[i] }

        return hidden
    }
}
