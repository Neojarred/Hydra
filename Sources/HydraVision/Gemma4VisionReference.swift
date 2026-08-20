import Foundation
import HydraCore

/// Gemma 4's vision tower in plain Swift, in double precision, as the oracle for the GPU.
///
/// Written from `modeling_gemma4.py` line by line. The pieces that differ from Qwen's tower are
/// the ones worth naming, because each produces a tower that runs and answers wrongly:
///
/// * **four RMSNorms a layer, sandwiched.** Norm before the sublayer and again after it, then the
///   residual. Gemma's text model does the same; Qwen's tower does not, and applying only the
///   leading norm leaves a tower whose activations drift with depth;
/// * **the attention scale is 1**, not `1/sqrt(headDim)`, because q and k are RMS-normalized
///   before the product;
/// * **v is normalized too**, by an RMSNorm with no learned scale, which is why no `v_norm`
///   tensor appears in the manifest and why it is easy to read straight past;
/// * **positions are two lookups summed**, `table[0][x] + table[1][y]`, not one grid resampled;
/// * **the pool is a 3x3 average** followed by a `sqrt(hiddenSize)` scale, then a standardization
///   against learned bias and scale, then a norm-and-project into the text model.
public struct Gemma4VisionReference {

    public let config: Gemma4VisionConfig

    public init(config: Gemma4VisionConfig = .a4b) {
        self.config = config
    }

    /// Everything the tower reads, as flat row-major arrays.
    public protocol Weights {
        func patchProjection() -> [Double]        // [hidden][patchElements], no bias
        /// `[2][positionEmbeddingSize][hidden]`, the x table then the y table.
        func positionTable() -> [Double]

        func inputNorm(_ layer: Int) -> [Double]
        func postAttentionNorm(_ layer: Int) -> [Double]
        func preFeedforwardNorm(_ layer: Int) -> [Double]
        func postFeedforwardNorm(_ layer: Int) -> [Double]
        func queryProjection(_ layer: Int) -> [Double]
        func keyProjection(_ layer: Int) -> [Double]
        func valueProjection(_ layer: Int) -> [Double]
        func outputProjection(_ layer: Int) -> [Double]
        func queryNorm(_ layer: Int) -> [Double]   // [headDim]
        func keyNorm(_ layer: Int) -> [Double]     // [headDim]
        func gateProjection(_ layer: Int) -> [Double]
        func upProjection(_ layer: Int) -> [Double]
        func downProjection(_ layer: Int) -> [Double]

        func standardizationBias() -> [Double]
        func standardizationScale() -> [Double]
        /// `[outHiddenSize][hidden]`, no bias. Quantized in the real checkpoint.
        func projection() -> [Double]
    }

    // MARK: - Pieces

    static func linear(_ x: [Double], weight: [Double], rows: Int, cols: Int) -> [Double] {
        var out = [Double](repeating: 0, count: rows)
        for row in 0..<rows {
            var sum = 0.0
            let base = row * cols
            for column in 0..<cols { sum += weight[base + column] * x[column] }
            out[row] = sum
        }
        return out
    }

    /// Gemma 4's RMSNorm: **a plain multiply by the weight**, not `1 + w`.
    ///
    /// This is the one place the family changed under the same name. Gemma 3's norm initializes
    /// its weight to zeros and multiplies by `1 + w`; Gemma 4 initializes to ones and multiplies
    /// by `w` directly, and every Gemma norm elsewhere in this codebase is the Gemma 3 form. So
    /// the habit is wrong here and reads as correct.
    ///
    /// Settled against the checkpoint rather than the source: this tower's `input_layernorm`
    /// weights average **+2.65** over a range of -0.5 to +8.7, and its `q_norm` is a uniform
    /// 1.008. Weights meant for `1 + w` sit near zero. Using `1 + w` here would inflate every
    /// activation by roughly its own magnitude again, which grows with depth and never fails.
    ///
    /// A `nil` weight is one of the two norms that have none: the value norm inside attention,
    /// and the norm before the projector.
    static func rmsNorm(_ x: [Double], weight: [Double]?, eps: Double) -> [Double] {
        var sum = 0.0
        for value in x { sum += value * value }
        let scale = 1 / (sum / Double(x.count) + eps).squareRoot()
        guard let weight else { return x.map { $0 * scale } }
        return (0..<x.count).map { x[$0] * scale * weight[$0] }
    }

    static func gelu(_ x: Double) -> Double {
        let inner = (2 / Double.pi).squareRoot() * (x + 0.044715 * x * x * x)
        return 0.5 * x * (1 + tanh(inner))
    }

    /// The rotary angles of one patch: half the head turns with x, half with y.
    public func rotaryAngles(x: Int, y: Int) -> [Double] {
        let perAxis = config.rotaryChannelsPerAxis          // 36
        let pairs = perAxis / 2                             // 18
        var out = [Double](repeating: 0, count: perAxis)
        for i in 0..<pairs {
            let inverse = 1 / pow(config.ropeTheta, Double(2 * i) / Double(perAxis))
            out[i] = Double(x) * inverse
            out[pairs + i] = Double(y) * inverse
        }
        return out
    }

    /// `rotate_half` **within each axis's own half of the head**.
    ///
    /// The reference splits the head into `ndim` slices of 36 and applies an ordinary rotary to
    /// each. So component `i` pairs with `i + 18` inside the x half, and `36 + i` with `36 + i +
    /// 18` inside the y half. Pairing across the halves, which is what a single 72-wide
    /// `rotate_half` would do, mixes the two axes together.
    public func applyRotary(_ vector: [Double], angles: [Double]) -> [Double] {
        let perAxis = config.rotaryChannelsPerAxis
        let pairs = perAxis / 2
        var out = vector
        for axis in 0..<2 {
            let base = axis * perAxis
            for i in 0..<pairs {
                let angle = angles[axis * pairs + i]
                let c = cos(angle), s = sin(angle)
                let low = vector[base + i], high = vector[base + pairs + i]
                out[base + i] = low * c - high * s
                out[base + pairs + i] = high * c + low * s
            }
        }
        return out
    }

    /// The learned position of one patch: the x table and the y table, summed.
    public func positionVector(x: Int, y: Int, weights: any Weights) -> [Double] {
        let table = weights.positionTable()
        let hidden = config.hiddenSize
        let stride = config.positionEmbeddingSize * hidden
        var out = [Double](repeating: 0, count: hidden)
        for i in 0..<hidden {
            out[i] = table[x * hidden + i] + table[stride + y * hidden + i]
        }
        return out
    }

    // MARK: - The tower

    /// Runs the tower and returns `[tokenCount][outHiddenSize]`.
    public func forward(
        patches: [Double], gridHeight: Int, gridWidth: Int, weights: any Weights
    ) -> [[Double]] {
        let states = encode(
            patches: patches, gridHeight: gridHeight, gridWidth: gridWidth, weights: weights)
        return pool(
            states, gridHeight: gridHeight, gridWidth: gridWidth, weights: weights,
            eps: Double(config.rmsNormEps))
    }

    /// The patch embedding and the 27 blocks: everything before the pool.
    private func encode(
        patches: [Double], gridHeight: Int, gridWidth: Int, weights: any Weights
    ) -> [[Double]] {
        let hidden = config.hiddenSize
        let count = gridHeight * gridWidth
        let eps = Double(config.rmsNormEps)

        // --- Patch projection, no bias, plus the two-table position ---
        var states: [[Double]] = []
        states.reserveCapacity(count)
        let projection = weights.patchProjection()
        for index in 0..<count {
            let patch = Array(
                patches[(index * config.patchElements)..<((index + 1) * config.patchElements)])
            var embedded = Self.linear(
                patch, weight: projection, rows: hidden, cols: config.patchElements)
            let position = config.patchPosition(atIndex: index, gridWidth: gridWidth)
            let learned = positionVector(x: position.x, y: position.y, weights: weights)
            for i in 0..<hidden { embedded[i] += learned[i] }
            states.append(embedded)
        }

        let angles = (0..<count).map { index -> [Double] in
            let position = config.patchPosition(atIndex: index, gridWidth: gridWidth)
            return rotaryAngles(x: position.x, y: position.y)
        }

        for layer in 0..<config.depth {
            states = block(layer, states: states, angles: angles, weights: weights, eps: eps)
        }
        return states
    }

    private func block(
        _ layer: Int, states: [[Double]], angles: [[Double]], weights: any Weights, eps: Double
    ) -> [[Double]] {
        let hidden = config.hiddenSize
        let heads = config.headCount
        let headDim = config.headDim
        let count = states.count

        // --- Attention: norm, project, normalize q/k/v, turn q and k, attend, project, norm, add
        let normed = states.map {
            Self.rmsNorm($0, weight: weights.inputNorm(layer), eps: eps)
        }
        let qW = weights.queryProjection(layer), kW = weights.keyProjection(layer)
        let vW = weights.valueProjection(layer)
        var queries = [[Double]](), keys = [[Double]](), values = [[Double]]()
        for index in 0..<count {
            var q = Self.linear(normed[index], weight: qW, rows: hidden, cols: hidden)
            var k = Self.linear(normed[index], weight: kW, rows: hidden, cols: hidden)
            var v = Self.linear(normed[index], weight: vW, rows: hidden, cols: hidden)
            for head in 0..<heads {
                let span = (head * headDim)..<((head + 1) * headDim)
                // Normalized per head, then turned. The order matters: turning first would
                // rotate a vector of a different length.
                let qn = Self.rmsNorm(
                    Array(q[span]), weight: weights.queryNorm(layer), eps: eps)
                let kn = Self.rmsNorm(
                    Array(k[span]), weight: weights.keyNorm(layer), eps: eps)
                // The value norm has no learned weight at all.
                let vn = Self.rmsNorm(Array(v[span]), weight: nil, eps: eps)
                q.replaceSubrange(span, with: applyRotary(qn, angles: angles[index]))
                k.replaceSubrange(span, with: applyRotary(kn, angles: angles[index]))
                v.replaceSubrange(span, with: vn)
            }
            queries.append(q); keys.append(k); values.append(v)
        }

        let scale = Double(config.attentionScale)
        var attended = [[Double]](repeating: [Double](repeating: 0, count: hidden), count: count)
        for head in 0..<heads {
            let offset = head * headDim
            for query in 0..<count {
                var logits = [Double](repeating: 0, count: count)
                var peak = -Double.infinity
                for key in 0..<count {
                    var dot = 0.0
                    for i in 0..<headDim {
                        dot += queries[query][offset + i] * keys[key][offset + i]
                    }
                    logits[key] = dot * scale
                    peak = max(peak, logits[key])
                }
                var total = 0.0
                for i in 0..<count { logits[i] = exp(logits[i] - peak); total += logits[i] }
                for key in 0..<count {
                    let weight = logits[key] / total
                    for i in 0..<headDim {
                        attended[query][offset + i] += weight * values[key][offset + i]
                    }
                }
            }
        }

        var out = states
        let oW = weights.outputProjection(layer)
        for index in 0..<count {
            let projected = Self.linear(attended[index], weight: oW, rows: hidden, cols: hidden)
            // **Normed after the sublayer, before the residual.** This is the second half of the
            // sandwich and the piece Qwen's tower has no equivalent of.
            let post = Self.rmsNorm(projected, weight: weights.postAttentionNorm(layer), eps: eps)
            for i in 0..<hidden { out[index][i] += post[i] }
        }

        // --- MLP: norm, GeGLU, norm, add ---
        let gateW = weights.gateProjection(layer), upW = weights.upProjection(layer)
        let downW = weights.downProjection(layer)
        for index in 0..<count {
            let pre = Self.rmsNorm(
                out[index], weight: weights.preFeedforwardNorm(layer), eps: eps)
            let gate = Self.linear(
                pre, weight: gateW, rows: config.intermediateSize, cols: hidden).map(Self.gelu)
            let up = Self.linear(pre, weight: upW, rows: config.intermediateSize, cols: hidden)
            let product = (0..<config.intermediateSize).map { gate[$0] * up[$0] }
            let down = Self.linear(
                product, weight: downW, rows: hidden, cols: config.intermediateSize)
            let post = Self.rmsNorm(down, weight: weights.postFeedforwardNorm(layer), eps: eps)
            for i in 0..<hidden { out[index][i] += post[i] }
        }
        return out
    }

    /// Everything except the projector, which is the tower's one quantized tensor.
    ///
    /// The comparison against the GPU stops here because a synthetic fixture has no quantized
    /// weights to give it. Nothing architectural lives past this point: the projector is one
    /// matrix multiply the MLX kernels are tested on separately.
    public func pooledForTesting(
        patches: [Double], gridHeight: Int, gridWidth: Int, weights: any Weights
    ) -> [[Double]] {
        let states = encode(
            patches: patches, gridHeight: gridHeight, gridWidth: gridWidth, weights: weights)
        return standardize(
            states, gridHeight: gridHeight, gridWidth: gridWidth, weights: weights)
    }

    /// Pool, scale and standardize. The projector follows in `pool`.
    private func standardize(
        _ states: [[Double]], gridHeight: Int, gridWidth: Int, weights: any Weights
    ) -> [[Double]] {
        let hidden = config.hiddenSize
        let k = config.poolingKernelSize
        let pooledHeight = gridHeight / k, pooledWidth = gridWidth / k
        let root = Double(hidden).squareRoot()
        let bias = weights.standardizationBias(), scale = weights.standardizationScale()

        var out: [[Double]] = []
        for blockY in 0..<pooledHeight {
            for blockX in 0..<pooledWidth {
                var pooled = [Double](repeating: 0, count: hidden)
                for dy in 0..<k {
                    for dx in 0..<k {
                        let patch = (blockY * k + dy) * gridWidth + (blockX * k + dx)
                        for i in 0..<hidden { pooled[i] += states[patch][i] }
                    }
                }
                let divisor = Double(k * k)
                for i in 0..<hidden {
                    pooled[i] = (pooled[i] / divisor) * root
                    pooled[i] = (pooled[i] - bias[i]) * scale[i]
                }
                out.append(pooled)
            }
        }
        return out
    }

    /// Pool, scale, standardize, then norm and project into the text model.
    private func pool(
        _ states: [[Double]], gridHeight: Int, gridWidth: Int, weights: any Weights, eps: Double
    ) -> [[Double]] {
        let hidden = config.hiddenSize
        let k = config.poolingKernelSize
        let pooledHeight = gridHeight / k, pooledWidth = gridWidth / k
        let root = Double(hidden).squareRoot()
        let bias = weights.standardizationBias(), scale = weights.standardizationScale()
        let projection = weights.projection()

        var out: [[Double]] = []
        for blockY in 0..<pooledHeight {
            for blockX in 0..<pooledWidth {
                var pooled = [Double](repeating: 0, count: hidden)
                for dy in 0..<k {
                    for dx in 0..<k {
                        let patch = (blockY * k + dy) * gridWidth + (blockX * k + dx)
                        for i in 0..<hidden { pooled[i] += states[patch][i] }
                    }
                }
                // The average, then `sqrt(hiddenSize)`, then the learned standardization. The
                // scale expands the magnitude far enough that the reference keeps it in float32.
                let divisor = Double(k * k)
                for i in 0..<hidden {
                    pooled[i] = (pooled[i] / divisor) * root
                    pooled[i] = (pooled[i] - bias[i]) * scale[i]
                }
                // A norm with no learned weight, then a projection with no bias.
                let normed = Self.rmsNorm(pooled, weight: nil, eps: eps)
                out.append(Self.linear(
                    normed, weight: projection, rows: config.outHiddenSize, cols: hidden))
            }
        }
        return out
    }
}
