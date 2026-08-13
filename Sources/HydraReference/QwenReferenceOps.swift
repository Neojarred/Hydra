import Foundation
import HydraCore

/// Reference CPU implementations of Qwen3.5/3.6 MoE's linear attention, in double precision.
///
/// The same arrangement as `Gemma4ReferenceOps`: slow, obviously correct, explicit loops, no
/// tricks, so that a divergence from the GPU reads as a GPU bug and never as an ambiguity here.
///
/// Every detail below is transcribed from `modeling_qwen3_5.py` and recorded in **D-027**. Five
/// of them cannot be guessed, and each is a plausible way to be wrong that produces finite,
/// degraded output with nothing to signal it. They are called out at the operator that carries
/// them.
public enum QwenReferenceOps {

    /// `rsqrt(Σx² + eps) · x`, the l2 normalization the delta rule applies to q and k.
    ///
    /// **Not an RMS norm.** There is no division by the length and no learned weight, so it
    /// differs from `Gemma4ReferenceOps.rmsNorm` by a factor of `sqrt(count)`. Reusing that
    /// operator here is the single easiest way to be wrong, and the result stays finite.
    ///
    /// `eps` is 1e-6 and sits inside the square root, added to the sum rather than to the mean.
    public static func l2Norm(_ x: [Double], eps: Double = 1e-6) -> [Double] {
        var sum = 0.0
        for value in x { sum += value * value }
        let scale = 1.0 / (sum + eps).squareRoot()
        return x.map { $0 * scale }
    }

    /// `log(1 + exp(x))`, computed so that a large `x` does not overflow before the log.
    public static func softplus(_ x: Double) -> Double {
        x > 20 ? x : Foundation.log1p(Foundation.exp(x))
    }

    public static func sigmoid(_ x: Double) -> Double { 1.0 / (1.0 + Foundation.exp(-x)) }

    public static func silu(_ x: Double) -> Double { x * sigmoid(x) }

    /// The per-head decay, `exp(-exp(A_log) · softplus(a + dt_bias))`.
    ///
    /// It is **not** a projection output used directly: `a` is projected, but `A_log` and
    /// `dt_bias` are learned per-head parameters and the composition matters. The result lies
    /// in `(0, 1]` because the exponent is a negative quantity, which is what makes the state
    /// decay rather than grow.
    public static func decay(a: Double, logA: Double, dtBias: Double) -> Double {
        Foundation.exp(-Foundation.exp(logA) * softplus(a + dtBias))
    }

    /// One token of the gated delta rule, for one value head.
    ///
    /// `state` is `[keyDim][valueDim]`, carried across tokens and **mutated in place**. It is
    /// the whole memory of a linear layer: there is no growing key/value cache, which is why a
    /// linear layer costs nothing extra as the context grows, and why it cannot be rewound.
    ///
    /// The order of the first two steps is load-bearing. The decay multiplies the state
    /// *before* `kv_mem` reads it, so the read sees the decayed state and not the previous one.
    /// Swapping them leaves everything finite and wrong.
    ///
    /// - Parameters:
    ///   - query: the head's query, **before** l2 normalization and before the scale.
    ///   - key: the head's key, before l2 normalization.
    ///   - value: the head's value, used as it arrives.
    /// - Returns: the head's output, `[valueDim]`.
    public static func deltaRuleStep(
        query: [Double], key: [Double], value: [Double],
        decay g: Double, beta: Double,
        state: inout [[Double]], eps: Double = 1e-6
    ) -> [Double] {
        let keyDim = key.count
        let valueDim = value.count
        precondition(query.count == keyDim, "query and key share the key head dimension")
        precondition(state.count == keyDim && state[0].count == valueDim)

        // The scale uses the **key** dimension and is applied **after** the l2 norm. Using the
        // value dimension, or scaling before normalizing, changes every output.
        let k = l2Norm(key, eps: eps)
        let q = l2Norm(query, eps: eps).map { $0 / Double(keyDim).squareRoot() }

        // 1. Decay, before anything reads the state.
        for i in 0..<keyDim {
            for j in 0..<valueDim { state[i][j] *= g }
        }

        // 2. What the decayed state already holds for this key: contract over the key
        //    dimension, giving a vector over the value dimension.
        var memory = [Double](repeating: 0, count: valueDim)
        for i in 0..<keyDim {
            let ki = k[i]
            if ki == 0 { continue }
            for j in 0..<valueDim { memory[j] += state[i][j] * ki }
        }

        // 3. The delta rule proper: write only the part of the value the state does not
        //    already predict, scaled by the gate.
        var delta = [Double](repeating: 0, count: valueDim)
        for j in 0..<valueDim { delta[j] = (value[j] - memory[j]) * beta }

        // 4. Accumulate it as an outer product.
        for i in 0..<keyDim {
            let ki = k[i]
            if ki == 0 { continue }
            for j in 0..<valueDim { state[i][j] += ki * delta[j] }
        }

        // 5. Read the updated state with the query.
        var out = [Double](repeating: 0, count: valueDim)
        for i in 0..<keyDim {
            let qi = q[i]
            if qi == 0 { continue }
            for j in 0..<valueDim { out[j] += state[i][j] * qi }
        }
        return out
    }

    /// Splits `q_proj`'s output into the query and its gate, **per head**.
    ///
    /// The projection emits `heads · headDim · 2` values and the reference views them as
    /// `[heads][headDim · 2]` before chunking, so each head's own slice holds its query
    /// followed by its gate. Splitting the whole tensor down the middle instead gives head `h`
    /// the query of head `h` and the gate of head `h - heads/2`, which is finite, plausible and
    /// wrong for every head but the first.
    public static func splitQueryAndGate(
        _ combined: [Double], heads: Int, headDim: Int
    ) -> (query: [Double], gate: [Double]) {
        precondition(combined.count == heads * headDim * 2)
        var query = [Double](repeating: 0, count: heads * headDim)
        var gate = [Double](repeating: 0, count: heads * headDim)
        for head in 0..<heads {
            let source = head * headDim * 2
            for i in 0..<headDim {
                query[head * headDim + i] = combined[source + i]
                gate[head * headDim + i] = combined[source + headDim + i]
            }
        }
        return (query, gate)
    }

    /// `attn_output · sigmoid(gate)`, applied after attention rather than to the query.
    public static func applyOutputGate(_ output: [Double], gate: [Double]) -> [Double] {
        precondition(output.count == gate.count)
        return zip(output, gate).map { $0 * sigmoid($1) }
    }

    /// `silu(gate) · up`, Qwen's SwiGLU, used by both the shared expert and the routed ones.
    ///
    /// **Not Gemma's `gelu_pytorch_tanh` and not GPT-OSS's clamped variant with its `+1`.** All
    /// three are plausible activations over the same two branches, and picking the wrong one
    /// costs a few percent per element, compounding over forty layers into a model that is
    /// merely worse (D-014 against D-027).
    public static func siluMultiply(gate: [Double], up: [Double]) -> [Double] {
        precondition(gate.count == up.count)
        return zip(gate, up).map { silu($0) * $1 }
    }

    /// The router: softmax over **every** expert, take the top-k, then renormalize.
    ///
    /// Read from `Qwen3NextTopKRouter`, with `norm_topk_prob` absent from the published config
    /// and defaulting to true. The same convention Gemma uses; GPT-OSS softmaxes over the top-k
    /// alone, which gives different weights from identical logits and raises nothing.
    public static func router(
        _ logits: [Double], topK: Int
    ) -> (indices: [Int], weights: [Double]) {
        let peak = logits.max() ?? 0
        let exponentials = logits.map { Foundation.exp($0 - peak) }
        let total = exponentials.reduce(0, +)
        let probabilities = exponentials.map { $0 / total }

        var chosen: [Int] = []
        for _ in 0..<topK {
            var best = -Double.infinity
            var bestIndex = 0
            for (index, value) in probabilities.enumerated() where !chosen.contains(index) {
                if value > best { best = value; bestIndex = index }
            }
            chosen.append(bestIndex)
        }
        let selected = chosen.map { probabilities[$0] }
        let sum = selected.reduce(0, +)
        return (chosen, selected.map { $0 / sum })
    }

    /// The gated RMS norm applied to a linear layer's output.
    ///
    /// The learned weight multiplies the **normalized** value, and the gate multiplies
    /// afterwards. The gate is not inside the norm and does not participate in the variance.
    public static func gatedRMSNorm(
        _ x: [Double], weight: [Double], gate: [Double], eps: Double
    ) -> [Double] {
        var sum = 0.0
        for value in x { sum += value * value }
        let inverse = 1.0 / (sum / Double(x.count) + eps).squareRoot()
        var out = [Double](repeating: 0, count: x.count)
        for i in 0..<x.count { out[i] = weight[i] * (x[i] * inverse) * silu(gate[i]) }
        return out
    }

    /// The depthwise causal convolution over the concatenated q, k and v, kernel 4, with SiLU.
    ///
    /// Causal by left padding of `kernel - 1`, so token `t` sees `t-3 … t`, and depthwise so
    /// each of the `conv_dim` channels has its own kernel and no channel mixing happens.
    ///
    /// Its rolling window is per-layer state exactly as the recurrent state is: decoding one
    /// token at a time requires the previous three, and dropping them silently changes the
    /// first tokens of every turn.
    ///
    /// - Parameter history: the previous inputs per channel, oldest first, at most
    ///   `kernel - 1` of them. **Fewer is normal, not exceptional**: the window fills over the
    ///   first tokens of a sequence, and anything older than the start is the left padding.
    ///   Requiring it to be empty or full rejects tokens one and two of every sequence.
    public static func causalDepthwiseConv(
        input: [Double], history: [[Double]], weight: [[Double]], bias: [Double]?
    ) -> [Double] {
        let channels = input.count
        let kernel = weight[0].count
        precondition(weight.count == channels, "one kernel a channel: this is depthwise")
        precondition(history.count < kernel, "the window holds at most kernel - 1 past inputs")

        var out = [Double](repeating: 0, count: channels)
        for c in 0..<channels {
            var sum = bias?[c] ?? 0
            // Taps run oldest to newest; the last tap is the current token.
            for tap in 0..<kernel {
                let age = kernel - 1 - tap
                let index = history.count - age
                // A tap reaching before the start of the sequence is the left padding.
                let value = age == 0 ? input[c] : (index >= 0 ? history[index][c] : 0)
                sum += value * weight[c][tap]
            }
            out[c] = silu(sum)
        }
        return out
    }
}
