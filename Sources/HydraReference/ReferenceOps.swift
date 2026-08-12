import Foundation
import HydraCore

/// Reference CPU implementations of GPT-OSS's operators, in double precision.
///
/// They are not for inference, far too slow, but serve as **ground truth** for validating
/// the Metal kernels. It is the same arrangement as for MXFP4: a slow, obviously correct
/// implementation, against which we measure a fast and subtle one's deviation.
///
///
/// Every operator is checked against a vector produced by an independent transcription of
/// `gpt_oss/torch/model.py` (see `tools/gen_reference_fixtures.py`). The details below
/// cannot be guessed, and getting one wrong yields a model that generates plausible but
/// degraded text, never raising an error.
public enum ReferenceOps {

    // MARK: - RMSNorm

    /// `x / sqrt(mean(x²) + eps) * scale`, with the mean computed in double.
    public static func rmsNorm(_ x: [Double], scale: [Double], eps: Double = 1e-5) -> [Double] {
        precondition(x.count == scale.count)
        let meanSquare = x.reduce(0) { $0 + $1 * $1 } / Double(x.count)
        let inverse = 1.0 / (meanSquare + eps).squareRoot()
        return zip(x, scale).map { $0 * inverse * $1 }
    }

    // MARK: - RoPE avec extension YaRN

    /// Alias vers `HydraCore.RoPETables.Parameters`.
    ///
    /// The YaRN computation is **not** reimplemented here: the runtime and the reference share
    /// the same source, in `HydraCore`. The independence that validates it comes from elsewhere
    ///, a Python transcription of OpenAI's code, in the fixtures. Duplicating the
    /// implementation in Swift would gain nothing and create a risk of divergence.
    ///
    public typealias YarnParameters = RoPETables.Parameters

    /// YaRN concentration and inverse frequencies.
    public static func yarn(_ p: YarnParameters) -> (concentration: Double, invFreq: [Double]) {
        let tables = RoPETables(p)
        return (tables.concentration, tables.inverseFrequencies)
    }

    /// cos/sin tables for a set of positions, concentration already applied.
    public static func cosSin(
        positions: [Int], parameters: YarnParameters = .gptOss
    ) -> (cos: [[Double]], sin: [[Double]]) {
        let tables = RoPETables(parameters)
        var cosTable: [[Double]] = []
        var sinTable: [[Double]] = []
        for position in positions {
            let (c, s) = tables.tables(at: position)
            cosTable.append(c)
            sinTable.append(s)
        }
        return (cosTable, sinTable)
    }

    /// Applies RoPE to a head vector.
    ///
    /// **Split into two halves**, not into interleaved pairs: `x1` is the first half of the
    /// components, `x2` the second. This is the opposite of the same model's SwiGLU, which
    /// splits into even and odd indices, two opposing conventions in one architecture, and no
    /// error will flag a swap.
    public static func applyRoPE(_ head: [Double], cos: [Double], sin: [Double]) -> [Double] {
        let half = head.count / 2
        precondition(cos.count == half && sin.count == half)
        var out = [Double](repeating: 0, count: head.count)
        for i in 0..<half {
            let x1 = head[i], x2 = head[half + i]
            out[i] = x1 * cos[i] - x2 * sin[i]
            out[half + i] = x2 * cos[i] + x1 * sin[i]
        }
        return out
    }

    // MARK: - SwiGLU

    /// GPT-OSS's SwiGLU. Three departures from the usual formulation.
    ///
    /// 1. **Split into even and odd indices**, not into two halves: `gate_up_proj`'s rows are
    ///    interleaved `[gate₀, up₀, gate₁, up₁, …]`.
    /// 2. **Asymmetric clamping**: the gate branch is bounded from above only, the linear
    ///    branch on both sides.
    /// 3. **The linear branch gets +1**, and the swish uses `sigmoid(1.702·x)`, not
    ///    `sigmoid(x)`.
    public static func swiglu(_ row: [Double], alpha: Double = 1.702, limit: Double = 7.0)
        -> [Double]
    {
        precondition(row.count % 2 == 0)
        var out = [Double](repeating: 0, count: row.count / 2)
        for i in 0..<out.count {
            let gate = min(row[2 * i], limit)
            let linear = min(max(row[2 * i + 1], -limit), limit)
            let activated = gate * (1.0 / (1.0 + exp(-alpha * gate)))
            out[i] = activated * (linear + 1.0)
        }
        return out
    }

    // MARK: - Attention

    /// Attention with **sinks** and an optional sliding window.
    ///
    /// The sink is a per-head learned logit, added as an **extra column** in the softmax then
    /// removed. It therefore contributes no value to the result: it enlarges the denominator,
    /// which lets a head "look at nothing" by shifting its mass onto the sink. The mechanism
    /// is absent from Gemma 4, hence without precedent in
    /// TurboFieldfare.
    ///
    /// - Parameters:
    ///   - q: `[tokens][kvHeads][qMult][headDim]`
    ///   - k, v: `[tokens][kvHeads][headDim]`
    ///   - sinks: `[kvHeads * qMult]`
    ///   - slidingWindow: 0 for full attention.
    public static func sdpa(
        q: [[[[Double]]]], k: [[[Double]]], v: [[[Double]]],
        sinks: [Double], smScale: Double, slidingWindow: Int = 0
    ) -> [[Double]] {
        let tokens = q.count
        let kvHeads = q[0].count
        let qMult = q[0][0].count
        let headDim = q[0][0][0].count

        var out = [[Double]](
            repeating: [Double](repeating: 0, count: kvHeads * qMult * headDim), count: tokens)

        for h in 0..<kvHeads {
            for m in 0..<qMult {
                let sink = sinks[h * qMult + m]
                for query in 0..<tokens {
                    // Visible positions: causal, and inside the window if it is active.
                    var keys: [Int] = []
                    for key in 0...query {
                        if slidingWindow > 0 && key <= query - slidingWindow { continue }
                        keys.append(key)
                    }

                    var logits = [Double](repeating: 0, count: keys.count)
                    var peak = sink
                    for (index, key) in keys.enumerated() {
                        var dot = 0.0
                        for i in 0..<headDim { dot += q[query][h][m][i] * k[key][h][i] }
                        logits[index] = dot * smScale
                        peak = max(peak, logits[index])
                    }

                    // The sink takes part in the denominator, never in the numerator.
                    var denominator = exp(sink - peak)
                    for value in logits { denominator += exp(value - peak) }

                    let base = (h * qMult + m) * headDim
                    for (index, key) in keys.enumerated() {
                        let weight = exp(logits[index] - peak) / denominator
                        for i in 0..<headDim {
                            out[query][base + i] += weight * v[key][h][i]
                        }
                    }
                }
            }
        }
        return out
    }

    // MARK: - Router

    /// Selects the `topK` best experts then applies the softmax **over those logits alone**,
    /// not over the full distribution. The resulting weights therefore sum to 1 across the
    /// experts retained.
    ///
    /// On a tie the smaller index wins, a convention needed for decoding to stay
    /// reproducible.
    public static func router(_ logits: [Double], topK: Int) -> (indices: [Int], weights: [Double]) {
        let order = logits.indices.sorted {
            logits[$0] == logits[$1] ? $0 < $1 : logits[$0] > logits[$1]
        }
        let selected = Array(order.prefix(topK))
        let values = selected.map { logits[$0] }
        let peak = values.max() ?? 0
        let exponentials = values.map { exp($0 - peak) }
        let total = exponentials.reduce(0, +)
        return (selected, exponentials.map { $0 / total })
    }
}
