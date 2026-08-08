import Foundation
import HydraCore

/// Reference CPU implementations of Gemma 4's operators, in double precision.
///
/// The same arrangement as `ReferenceOps` does for GPT-OSS: slow, obviously correct, and used
/// as ground truth for the Metal kernels. Written deliberately plainly — explicit loops, no
/// tricks — so that a divergence from the GPU reads as a GPU bug and never as an ambiguity
/// here.
///
/// Each operator is checked against a vector produced by an independent Python transcription
/// of `modeling_gemma4.py` (see `tools/gen_gemma_fixtures.py`). The details below are recorded
/// in **D-022**; they cannot be guessed, and getting one wrong yields a model that generates
/// plausible but degraded text without raising an error.
public enum Gemma4ReferenceOps {

    /// `x · (mean(x²) + eps)^-0.5 · w`, the mean in double.
    ///
    /// **The weight is `w`, not `1 + w`.** Gemma 3 used `1 + w` and Gemma 4 does not; carrying
    /// the older habit over is a factor of two on every normalized activation, with nothing to
    /// signal it.
    ///
    /// `weight` is `nil` for the normalizations built `with_scale: false` — `v_norm` and the
    /// router's — which have **no tensor in the checkpoint at all**.
    public static func rmsNorm(_ x: [Double], weight: [Double]?, eps: Double) -> [Double] {
        var sum = 0.0
        for value in x { sum += value * value }
        let scale = Foundation.pow(sum / Double(x.count) + eps, -0.5)
        guard let weight else { return x.map { $0 * scale } }
        return zip(x, weight).map { $0 * scale * $1 }
    }

    /// `gelu_pytorch_tanh`, used by both the dense MLP and the experts.
    ///
    /// Not GPT-OSS's SwiGLU: no clamping, no `+1` on the linear branch, no `sigmoid(1.702·x)`.
    /// Reusing that kernel here would be silently wrong (D-014 against D-022).
    public static func gelu(_ x: Double) -> Double {
        let inner = (2.0 / Double.pi).squareRoot() * (x + 0.044715 * x * x * x)
        return 0.5 * x * (1.0 + tanh(inner))
    }

    /// `down(gelu(gate(x)) · up(x))`, the shape both MLP paths share.
    public static func mlp(gate: [Double], up: [Double]) -> [Double] {
        zip(gate, up).map { gelu($0) * $1 }
    }

    // MARK: - Rotary

    /// Inverse frequencies for one layer type, with the unrotated tail set to **zero**.
    ///
    /// Full-attention layers use `rope_type: "proportional"` with `partial_rotary_factor 0.25`:
    /// the first quarter of the frequency pairs are real and the rest are zero. A zero
    /// frequency gives `cos = 1, sin = 0` — the identity — so partial rotation is expressible
    /// without a kernel that knows about it.
    public static func inverseFrequencies(
        headDim: Int, theta: Double, rotatingPairs: Int
    ) -> [Double] {
        let pairs = headDim / 2
        var out = [Double](repeating: 0, count: pairs)
        for i in 0..<min(rotatingPairs, pairs) {
            out[i] = 1.0 / Foundation.pow(theta, 2.0 * Double(i) / Double(headDim))
        }
        return out
    }

    /// Applies RoPE to one head vector.
    ///
    /// **Split into halves**, as `emb = cat((freqs, freqs))` implies — the same convention
    /// GPT-OSS uses, and the opposite of its SwiGLU's even/odd split.
    public static func applyRoPE(
        _ x: [Double], position: Int, frequencies: [Double]
    ) -> [Double] {
        let half = x.count / 2
        var out = x
        for i in 0..<half {
            let angle = Double(position) * frequencies[i]
            let c = cos(angle), s = sin(angle)
            out[i] = x[i] * c - x[i + half] * s
            out[i + half] = x[i + half] * c + x[i] * s
        }
        return out
    }

    // MARK: - Attention

    /// Scaled dot-product attention with an optional sliding window.
    ///
    /// **The scale is 1.0, not `1/sqrt(headDim)`** — `self.scaling = 1.0` in the source, because
    /// the query norm absorbs it. This is the single easiest thing to get wrong by habit.
    ///
    /// There are **no attention sinks**: that is a GPT-OSS mechanism and Gemma has none, so
    /// nothing is added to the denominator.
    public static func attention(
        query: [Double], keys: [[Double]], values: [[Double]],
        slidingWindow: Int = 0
    ) -> [Double] {
        let first = slidingWindow == 0 ? 0 : max(0, keys.count - slidingWindow)
        var scores: [Double] = []
        for key in keys[first...] {
            var dot = 0.0
            for i in 0..<query.count { dot += query[i] * key[i] }
            scores.append(dot)
        }

        let weights = softmax(scores)
        var out = [Double](repeating: 0, count: values[0].count)
        for (weight, value) in zip(weights, values[first...]) {
            for i in 0..<out.count { out[i] += weight * value[i] }
        }
        return out
    }

    public static func softmax(_ values: [Double]) -> [Double] {
        guard let peak = values.max() else { return [] }
        let exponentials = values.map { Foundation.exp($0 - peak) }
        let total = exponentials.reduce(0, +)
        return exponentials.map { $0 / total }
    }

    // MARK: - Router

    /// Selects experts and weights them.
    ///
    /// The chain differs from GPT-OSS's at every step and the same logits give different
    /// weights: normalize **without a weight**, scale by the learned vector and by
    /// `hiddenSize^-0.5`, softmax over **all** experts, take the top-k, **renormalize them to
    /// sum to one**, then apply the per-expert scale. GPT-OSS softmaxes over the top-k only and
    /// has neither scale.
    ///
    /// On a tie the smaller index wins, so decoding stays reproducible.
    public static func router(
        hidden: [Double], projection: [[Double]], scale: [Double],
        perExpertScale: [Double], topK: Int, eps: Double
    ) -> (indices: [Int], weights: [Double]) {
        let normalized = rmsNorm(hidden, weight: nil, eps: eps)
        let root = Foundation.pow(Double(hidden.count), -0.5)
        let scaled = zip(normalized, scale).map { $0 * $1 * root }

        var logits: [Double] = []
        for row in projection {
            var dot = 0.0
            for i in 0..<scaled.count { dot += scaled[i] * row[i] }
            logits.append(dot)
        }

        let probabilities = softmax(logits)
        let order = probabilities.indices.sorted {
            probabilities[$0] == probabilities[$1] ? $0 < $1 : probabilities[$0] > probabilities[$1]
        }
        let indices = Array(order.prefix(topK))
        var weights = indices.map { probabilities[$0] }
        let total = weights.reduce(0, +)
        weights = weights.map { $0 / total }
        weights = zip(weights, indices).map { $0 * perExpertScale[$1] }
        return (indices, weights)
    }

    // MARK: - Output

    /// `c · tanh(logits / c)`, applied to the logits before sampling.
    public static func softcap(_ logits: [Double], cap: Double) -> [Double] {
        logits.map { cap * tanh($0 / cap) }
    }

    /// Embeddings are multiplied by this on lookup. **Nothing in the checkpoint reveals it** —
    /// it is a constant in the model code — and omitting it mis-scales the entire forward pass.
    public static func embeddingScale(hiddenSize: Int) -> Double {
        Double(hiddenSize).squareRoot()
    }
}
