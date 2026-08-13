import Foundation
import HydraCore

/// A whole Qwen linear attention block on the CPU, in double precision.
///
/// The operators in `QwenReferenceOps` are each checked against an independent transcription.
/// This checks the thing they are checked *into*: the order they run in, what feeds what, and
/// which tensor the residual reads. Those are a different class of mistake, and the one Gemma
/// cost the most time to find (D-022): every operator correct, composed wrongly, and the model
/// answers.
///
/// The block, from `Qwen3NextDecoderLayer` and `Qwen3_5GatedDeltaNet` (D-027):
///
/// ```
/// residual = x
/// x = input_layernorm(x)
/// qkv  = in_proj_qkv(x)          z = in_proj_z(x)
/// a    = in_proj_a(x)            b = in_proj_b(x)
/// qkv  = silu(causal_conv(qkv))                    <- window is carried state
/// q, k, v = split(qkv)
/// out  = delta_rule(q, k, v, g(a), beta(b))        <- state is carried state
/// out  = gated_rms_norm(out, norm.weight, gate: z)
/// x    = out_proj(out)
/// x    = residual + x
/// ```
public struct QwenReferenceLayer {

    /// The shape of one linear layer, small enough for a fixture.
    public struct Shape: Sendable {
        public let hiddenSize: Int
        public let keyHeads: Int
        public let valueHeads: Int
        public let keyDim: Int
        public let valueDim: Int
        public let convKernel: Int
        public let eps: Double

        public init(
            hiddenSize: Int, keyHeads: Int, valueHeads: Int, keyDim: Int, valueDim: Int,
            convKernel: Int = 4, eps: Double = 1e-6
        ) {
            self.hiddenSize = hiddenSize
            self.keyHeads = keyHeads
            self.valueHeads = valueHeads
            self.keyDim = keyDim
            self.valueDim = valueDim
            self.convKernel = convKernel
            self.eps = eps
        }

        /// q, k and v concatenated, which is what the convolution runs over.
        public var convDim: Int { 2 * keyHeads * keyDim + valueHeads * valueDim }
        public var zDim: Int { valueHeads * valueDim }
    }

    /// Row-major `[rows][cols]` matrices, and the per-head vectors.
    public struct Weights: Sendable {
        public var inputNorm: [Double]
        public var qkv: [[Double]]
        public var z: [[Double]]
        public var a: [[Double]]
        public var b: [[Double]]
        public var outProj: [[Double]]
        public var convWeight: [[Double]]
        public var convBias: [Double]?
        public var logA: [Double]
        public var dtBias: [Double]
        public var normWeight: [Double]

        public init(
            inputNorm: [Double], qkv: [[Double]], z: [[Double]], a: [[Double]], b: [[Double]],
            outProj: [[Double]], convWeight: [[Double]], convBias: [Double]?,
            logA: [Double], dtBias: [Double], normWeight: [Double]
        ) {
            self.inputNorm = inputNorm
            self.qkv = qkv
            self.z = z
            self.a = a
            self.b = b
            self.outProj = outProj
            self.convWeight = convWeight
            self.convBias = convBias
            self.logA = logA
            self.dtBias = dtBias
            self.normWeight = normWeight
        }
    }

    /// Everything the block carries between tokens.
    ///
    /// Both parts are state and both are forgotten just as silently. The window is the one that
    /// is easy to miss, because dropping it is wrong only for the first `kernel - 1` tokens of a
    /// sequence.
    public struct State {
        public var recurrent: [[[Double]]]   // [valueHeads][keyDim][valueDim]
        public var window: [[Double]]        // up to kernel - 1 rows of convDim, oldest first

        public init(shape: Shape) {
            recurrent = (0..<shape.valueHeads).map { _ in
                [[Double]](
                    repeating: [Double](repeating: 0, count: shape.valueDim),
                    count: shape.keyDim)
            }
            window = []
        }
    }

    public let shape: Shape
    public let weights: Weights

    public init(shape: Shape, weights: Weights) {
        self.shape = shape
        self.weights = weights
    }

    private func project(_ matrix: [[Double]], _ x: [Double]) -> [Double] {
        matrix.map { row in zip(row, x).reduce(0) { $0 + $1.0 * $1.1 } }
    }

    /// One token through the block. `state` is advanced.
    public func forward(_ x: [Double], state: inout State) -> [Double] {
        precondition(x.count == shape.hiddenSize)

        // Pre-norm, and the residual is the layer's input.
        let normed = Gemma4ReferenceOps.rmsNorm(x, weight: weights.inputNorm, eps: shape.eps)

        let qkvRaw = project(weights.qkv, normed)
        let z = project(weights.z, normed)
        let aValues = project(weights.a, normed)
        let bValues = project(weights.b, normed)

        // The convolution sees q, k and v together, before they are split.
        let qkv = QwenReferenceOps.causalDepthwiseConv(
            input: qkvRaw, history: state.window,
            weight: weights.convWeight, bias: weights.convBias)
        state.window.append(qkvRaw)
        if state.window.count > shape.convKernel - 1 { state.window.removeFirst() }

        let keySpan = shape.keyHeads * shape.keyDim
        let queries = Array(qkv[0..<keySpan])
        let keys = Array(qkv[keySpan..<(2 * keySpan)])
        let values = Array(qkv[(2 * keySpan)...])

        var mixed = [Double](repeating: 0, count: shape.zDim)
        for head in 0..<shape.valueHeads {
            let keyHead = head / (shape.valueHeads / shape.keyHeads)
            let headOutput = QwenReferenceOps.deltaRuleStep(
                query: Array(queries[(keyHead * shape.keyDim)..<((keyHead + 1) * shape.keyDim)]),
                key: Array(keys[(keyHead * shape.keyDim)..<((keyHead + 1) * shape.keyDim)]),
                value: Array(values[(head * shape.valueDim)..<((head + 1) * shape.valueDim)]),
                decay: QwenReferenceOps.decay(
                    a: aValues[head], logA: weights.logA[head], dtBias: weights.dtBias[head]),
                beta: QwenReferenceOps.sigmoid(bValues[head]),
                state: &state.recurrent[head], eps: shape.eps)

            // The norm is per head, over the value dimension, and gated by this head's slice
            // of z.
            let gated = QwenReferenceOps.gatedRMSNorm(
                headOutput, weight: weights.normWeight,
                gate: Array(z[(head * shape.valueDim)..<((head + 1) * shape.valueDim)]),
                eps: shape.eps)
            for i in 0..<shape.valueDim { mixed[head * shape.valueDim + i] = gated[i] }
        }

        let projected = project(weights.outProj, mixed)
        return zip(x, projected).map(+)
    }
}

/// A whole Qwen **full-attention** block on the CPU.
///
/// The other kind of layer, one in four. Shares the pre-norm residual structure with the linear
/// block and differs entirely inside: a gated query projection, per-head norms before the
/// rotary, a growing key/value history rather than a fixed state, and the ordinary attention
/// scale rather than Gemma's 1.0 (D-027).
public struct QwenReferenceAttentionLayer {

    public struct Shape: Sendable {
        public let hiddenSize: Int
        public let heads: Int
        public let keyValueHeads: Int
        public let headDim: Int
        public let ropeTheta: Double
        public let partialRotaryFactor: Double
        public let eps: Double

        public init(
            hiddenSize: Int, heads: Int, keyValueHeads: Int, headDim: Int,
            ropeTheta: Double = 10_000_000, partialRotaryFactor: Double = 0.25,
            eps: Double = 1e-6
        ) {
            self.hiddenSize = hiddenSize
            self.heads = heads
            self.keyValueHeads = keyValueHeads
            self.headDim = headDim
            self.ropeTheta = ropeTheta
            self.partialRotaryFactor = partialRotaryFactor
            self.eps = eps
        }

        public var queryDim: Int { heads * headDim }
        public var keyValueDim: Int { keyValueHeads * headDim }
        /// `q_proj` emits the gate alongside the query, so it is twice as tall.
        public var queryProjectionRows: Int { queryDim * 2 }
        /// The pairs the rotary actually turns; the rest keep zero frequency.
        public var rotatingPairs: Int { Int(Double(headDim) * partialRotaryFactor) / 2 }
    }

    public struct Weights: Sendable {
        public var inputNorm: [Double]
        public var qProj: [[Double]]
        public var kProj: [[Double]]
        public var vProj: [[Double]]
        public var oProj: [[Double]]
        public var qNorm: [Double]
        public var kNorm: [Double]

        public init(
            inputNorm: [Double], qProj: [[Double]], kProj: [[Double]], vProj: [[Double]],
            oProj: [[Double]], qNorm: [Double], kNorm: [Double]
        ) {
            self.inputNorm = inputNorm
            self.qProj = qProj
            self.kProj = kProj
            self.vProj = vProj
            self.oProj = oProj
            self.qNorm = qNorm
            self.kNorm = kNorm
        }
    }

    /// The history, which grows, unlike the linear block's fixed state.
    public struct Cache {
        public var keys: [[Double]] = []     // [position][keyValueDim]
        public var values: [[Double]] = []
        public init() {}
    }

    public let shape: Shape
    public let weights: Weights

    public init(shape: Shape, weights: Weights) {
        self.shape = shape
        self.weights = weights
    }

    private func project(_ m: [[Double]], _ v: [Double]) -> [Double] {
        m.map { row in zip(row, v).reduce(0) { $0 + $1.0 * $1.1 } }
    }

    public func forward(_ x: [Double], position: Int, cache: inout Cache) -> [Double] {
        let normed = Gemma4ReferenceOps.rmsNorm(x, weight: weights.inputNorm, eps: shape.eps)
        let combined = project(weights.qProj, normed)
        // Per head, and not the tensor halved: head h's query is followed by head h's gate.
        let (queryRaw, gate) = QwenReferenceOps.splitQueryAndGate(
            combined, heads: shape.heads, headDim: shape.headDim)
        var key = project(weights.kProj, normed)
        let value = project(weights.vProj, normed)

        let frequencies = Gemma4ReferenceOps.inverseFrequencies(
            headDim: shape.headDim, theta: shape.ropeTheta, rotatingPairs: shape.rotatingPairs)

        // The norms are per head, over the head dimension, and come **before** the rotary.
        var query = queryRaw
        for head in 0..<shape.heads {
            let span = (head * shape.headDim)..<((head + 1) * shape.headDim)
            let normedHead = Gemma4ReferenceOps.rmsNorm(
                Array(query[span]), weight: weights.qNorm, eps: shape.eps)
            let rotated = Gemma4ReferenceOps.applyRoPE(
                normedHead, position: position, frequencies: frequencies)
            for (i, v) in rotated.enumerated() { query[head * shape.headDim + i] = v }
        }
        for head in 0..<shape.keyValueHeads {
            let span = (head * shape.headDim)..<((head + 1) * shape.headDim)
            let normedHead = Gemma4ReferenceOps.rmsNorm(
                Array(key[span]), weight: weights.kNorm, eps: shape.eps)
            let rotated = Gemma4ReferenceOps.applyRoPE(
                normedHead, position: position, frequencies: frequencies)
            for (i, v) in rotated.enumerated() { key[head * shape.headDim + i] = v }
        }

        cache.keys.append(key)
        cache.values.append(value)

        // `1/sqrt(headDim)`, folded into the query because the reference's attention takes no
        // scale. **Not Gemma's 1.0**: that constant belongs to a model whose query norm absorbs
        // the scale, and borrowing it flattens every distribution here.
        let scale = 1.0 / Double(shape.headDim).squareRoot()
        let group = shape.heads / shape.keyValueHeads
        var attended = [Double](repeating: 0, count: shape.queryDim)
        for head in 0..<shape.heads {
            let kvHead = head / group
            let span = (kvHead * shape.headDim)..<((kvHead + 1) * shape.headDim)
            let out = Gemma4ReferenceOps.attention(
                query: (0..<shape.headDim).map { query[head * shape.headDim + $0] * scale },
                keys: cache.keys.map { Array($0[span]) },
                values: cache.values.map { Array($0[span]) })
            for (i, v) in out.enumerated() { attended[head * shape.headDim + i] = v }
        }

        // The gate multiplies what attention returned, not what it attended with.
        let gated = QwenReferenceOps.applyOutputGate(attended, gate: gate)
        return zip(x, project(weights.oProj, gated)).map(+)
    }
}

/// Qwen's mixture block on the CPU: a shared expert that always runs, and eight routed ones.
///
/// ```
/// shared = down(silu(gate(h)) · up(h))
/// shared = sigmoid(shared_expert_gate(h)) · shared
/// routed = Σ  weight_e · down_e(silu(gate_e(h)) · up_e(h))
/// out    = routed + shared
/// ```
///
/// The shared branch is always active and does not wait on the SSD, which is the structural
/// gain D-026 hoped for: work the GPU can do while the routed experts are read.
public struct QwenReferenceMixture {

    public struct Shape: Sendable {
        public let hiddenSize: Int
        public let expertCount: Int
        public let expertsPerToken: Int
        public let moeIntermediate: Int
        public let sharedIntermediate: Int

        public init(
            hiddenSize: Int, expertCount: Int, expertsPerToken: Int,
            moeIntermediate: Int, sharedIntermediate: Int
        ) {
            self.hiddenSize = hiddenSize
            self.expertCount = expertCount
            self.expertsPerToken = expertsPerToken
            self.moeIntermediate = moeIntermediate
            self.sharedIntermediate = sharedIntermediate
        }
    }

    /// One expert's three matrices.
    public struct Expert: Sendable {
        public var gate: [[Double]]
        public var up: [[Double]]
        public var down: [[Double]]
        public init(gate: [[Double]], up: [[Double]], down: [[Double]]) {
            self.gate = gate
            self.up = up
            self.down = down
        }
    }

    public struct Weights: Sendable {
        public var router: [[Double]]          // [expertCount][hidden]
        public var sharedGate: [Double]        // one row: [hidden]
        public var shared: Expert
        public var experts: [Expert]

        public init(
            router: [[Double]], sharedGate: [Double], shared: Expert, experts: [Expert]
        ) {
            self.router = router
            self.sharedGate = sharedGate
            self.shared = shared
            self.experts = experts
        }
    }

    public let shape: Shape
    public let weights: Weights

    public init(shape: Shape, weights: Weights) {
        self.shape = shape
        self.weights = weights
    }

    private func project(_ m: [[Double]], _ v: [Double]) -> [Double] {
        m.map { row in zip(row, v).reduce(0) { $0 + $1.0 * $1.1 } }
    }

    private func feedForward(_ expert: Expert, _ x: [Double]) -> [Double] {
        project(
            expert.down,
            QwenReferenceOps.siluMultiply(
                gate: project(expert.gate, x), up: project(expert.up, x)))
    }

    /// The mixture's contribution. The caller adds the residual, as the layer does.
    public func forward(_ x: [Double]) -> [Double] {
        let logits = project(weights.router, x)
        let routing = QwenReferenceOps.router(logits, topK: shape.expertsPerToken)

        var out = [Double](repeating: 0, count: shape.hiddenSize)
        for (rank, expert) in routing.indices.enumerated() {
            let contribution = feedForward(weights.experts[expert], x)
            for i in 0..<shape.hiddenSize {
                out[i] += routing.weights[rank] * contribution[i]
            }
        }

        // The shared expert is gated by a sigmoid of its own single-row projection, and the
        // gate is applied to its **output**, not to its input.
        let gateLogit = zip(weights.sharedGate, x).reduce(0) { $0 + $1.0 * $1.1 }
        let scale = QwenReferenceOps.sigmoid(gateLogit)
        let shared = feedForward(weights.shared, x)
        for i in 0..<shape.hiddenSize { out[i] += scale * shared[i] }
        return out
    }
}
