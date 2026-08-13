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
