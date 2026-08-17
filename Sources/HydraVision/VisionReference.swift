import Foundation
import HydraCore

/// The vision tower in plain Swift, in double precision, as the oracle for the GPU.
///
/// Same role as `HydraReference` on the text side: this is not the implementation that ships,
/// it is the one that decides what "correct" means. It is written from
/// `Qwen3_5MoeVisionModel.forward` line by line rather than from an understanding of ViTs,
/// because the places this can go wrong are all places where an understanding of ViTs would say
/// the wrong thing is fine.
///
/// Three of those are worth naming, because each produces finite, plausible output:
///
/// * the tower has **both** a learned position grid, bilinearly resampled to the image's shape,
///   **and** a 2-D rotary embedding inside attention. Implementing either alone leaves a tower
///   that runs;
/// * attention is **bidirectional**. A causal mask, which every other attention in this codebase
///   uses, gives the first patch no context and the last one everything;
/// * the merger normalizes at width 1152, **before** the four patches are concatenated to 4608.
///   Normalizing after is the reading that the shapes also permit.
///
/// Double precision throughout, and no attempt at speed: one 320x240 image is 300 patches and
/// takes a few seconds. Tests use a tiny tower.
public struct VisionReference {

    public let config: Qwen35VisionConfig

    public init(config: Qwen35VisionConfig = .a3b) {
        self.config = config
    }

    /// Everything the tower reads, as flat row-major arrays of the shapes the manifest records.
    ///
    /// A protocol rather than a struct of arrays so the same code runs against a synthetic tiny
    /// tower and against the real 851 MiB one.
    public protocol Weights {
        func patchWeight() -> [Double]        // [hidden][patchElements]
        func patchBias() -> [Double]          // [hidden]
        func positionEmbedding() -> [Double]  // [positionEmbeddingCount][hidden]

        func norm1Weight(_ layer: Int) -> [Double]
        func norm1Bias(_ layer: Int) -> [Double]
        func norm2Weight(_ layer: Int) -> [Double]
        func norm2Bias(_ layer: Int) -> [Double]
        func qkvWeight(_ layer: Int) -> [Double]   // [3 * hidden][hidden]
        func qkvBias(_ layer: Int) -> [Double]
        func projWeight(_ layer: Int) -> [Double]  // [hidden][hidden]
        func projBias(_ layer: Int) -> [Double]
        func fc1Weight(_ layer: Int) -> [Double]   // [intermediate][hidden]
        func fc1Bias(_ layer: Int) -> [Double]
        func fc2Weight(_ layer: Int) -> [Double]   // [hidden][intermediate]
        func fc2Bias(_ layer: Int) -> [Double]

        func mergerNormWeight() -> [Double]
        func mergerNormBias() -> [Double]
        func mergerFC1Weight() -> [Double]  // [merged][merged]
        func mergerFC1Bias() -> [Double]
        func mergerFC2Weight() -> [Double]  // [out][merged]
        func mergerFC2Bias() -> [Double]
    }

    // MARK: - Pieces

    /// `y = W x + b`, with `W` row-major `[rows][cols]`.
    static func linear(_ x: [Double], weight: [Double], bias: [Double], rows: Int, cols: Int)
        -> [Double]
    {
        var out = [Double](repeating: 0, count: rows)
        for row in 0..<rows {
            var sum = bias.isEmpty ? 0 : bias[row]
            let base = row * cols
            for column in 0..<cols { sum += weight[base + column] * x[column] }
            out[row] = sum
        }
        return out
    }

    /// LayerNorm, **not** RMSNorm: the mean is subtracted.
    ///
    /// Every norm elsewhere in this codebase is an RMSNorm with no bias, so reaching for the
    /// familiar one here is the natural error. It would leave the tower running on activations
    /// whose mean was never removed, which is a scale error that grows with depth rather than a
    /// visible break. The tower's tensors carry a `.bias` beside every `.weight`, which is the
    /// tell.
    static func layerNorm(_ x: [Double], weight: [Double], bias: [Double], eps: Double = 1e-6)
        -> [Double]
    {
        let n = Double(x.count)
        let mean = x.reduce(0, +) / n
        let variance = x.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / n
        let scale = 1 / (variance + eps).squareRoot()
        return (0..<x.count).map { (x[$0] - mean) * scale * weight[$0] + bias[$0] }
    }

    /// `gelu_pytorch_tanh`, the approximation the config names.
    ///
    /// Not interchangeable with the exact erf GELU at the precision this project checks to.
    static func gelu(_ x: Double) -> Double {
        let inner = (2 / Double.pi).squareRoot() * (x + 0.044715 * x * x * x)
        return 0.5 * x * (1 + tanh(inner))
    }

    // MARK: - Positions

    /// The four taps and weights that resample the learned 48x48 grid onto one patch.
    ///
    /// `align_corners` is true for this tower, which is the branch where the endpoints map
    /// exactly onto 0 and `side - 1`. The half-pixel branch is the more common convention and
    /// the wrong one here; it shifts every position by a fraction of a cell, which is a small,
    /// smooth, entirely invisible error.
    func interpolationTaps(index: Int, size: Int) -> [(tap: Int, weight: Double)] {
        let side = config.positionGridSide
        let source = Double(index) * Double(side - 1) / Double(max(size - 1, 1))
        let floor = source.rounded(.down)
        return (0..<2).map { offset in
            let tap = min(max(Int(floor) + offset, 0), side - 1)
            let distance = abs(source - floor - Double(offset))
            return (tap, max(1 - distance, 0))
        }
    }

    /// The learned position vector for one patch, resampled from the square grid.
    func positionVector(y: Int, x: Int, grid: Qwen35VisionConfig.Grid, weights: any Weights)
        -> [Double]
    {
        let table = weights.positionEmbedding()
        let side = config.positionGridSide
        let rows = interpolationTaps(index: y, size: grid.height)
        let columns = interpolationTaps(index: x, size: grid.width)

        var out = [Double](repeating: 0, count: config.hiddenSize)
        for row in rows where row.weight > 0 {
            for column in columns where column.weight > 0 {
                let weight = row.weight * column.weight
                let base = (row.tap * side + column.tap) * config.hiddenSize
                for i in 0..<config.hiddenSize { out[i] += weight * table[base + i] }
            }
        }
        return out
    }

    /// The rotary angles for one patch: 18 frequencies against its row, then 18 against its
    /// column, giving 36 values for a 72-wide head.
    func rotaryAngles(y: Int, x: Int) -> [Double] {
        let half = config.headDim / 2            // 36
        let pairs = half / 2                     // 18
        var out = [Double](repeating: 0, count: half)
        for i in 0..<pairs {
            let inverse = 1 / pow(10000.0, Double(2 * i) / Double(half))
            out[i] = Double(y) * inverse
            out[pairs + i] = Double(x) * inverse
        }
        return out
    }

    /// `q * cos + rotate_half(q) * sin`, over one head.
    ///
    /// `rotate_half` pairs component `i` with `i + 36`, so the first 18 pairs turn with the row
    /// and the last 18 with the column. Pairing `2i` with `2i+1`, the interleaved convention
    /// used by the text model, is the other plausible reading and rotates the wrong components
    /// against each other.
    func applyRotary(_ vector: [Double], angles: [Double]) -> [Double] {
        let half = config.headDim / 2
        var out = [Double](repeating: 0, count: config.headDim)
        for i in 0..<half {
            let c = cos(angles[i]), s = sin(angles[i])
            out[i] = vector[i] * c - vector[i + half] * s
            out[i + half] = vector[i + half] * c + vector[i] * s
        }
        return out
    }

    // MARK: - The tower

    /// Runs the whole tower and returns `[tokenCount][outHiddenSize]`.
    public func forward(
        patches: [Double], grid: Qwen35VisionConfig.Grid, weights: any Weights
    ) -> [[Double]] {
        let hidden = config.hiddenSize
        let count = grid.patchCount

        // --- Patch embedding, plus the resampled learned position ---
        var states: [[Double]] = []
        states.reserveCapacity(count)
        let patchWeight = weights.patchWeight(), patchBias = weights.patchBias()
        for index in 0..<count {
            let patch = Array(patches[(index * config.patchElements)..<((index + 1) * config.patchElements)])
            var embedded = Self.linear(
                patch, weight: patchWeight, bias: patchBias,
                rows: hidden, cols: config.patchElements)
            let position = config.patchPosition(atSequenceIndex: index, grid: grid)
            let learned = positionVector(y: position.y, x: position.x, grid: grid, weights: weights)
            for i in 0..<hidden { embedded[i] += learned[i] }
            states.append(embedded)
        }

        // Rotary angles are a function of the patch's place in the image, so they are computed
        // once and reused by all 27 blocks.
        let angles = (0..<count).map { index -> [Double] in
            let position = config.patchPosition(atSequenceIndex: index, grid: grid)
            return rotaryAngles(y: position.y, x: position.x)
        }

        for layer in 0..<config.depth {
            states = block(layer, states: states, angles: angles, weights: weights)
        }
        return merge(states, weights: weights)
    }

    private func block(
        _ layer: Int, states: [[Double]], angles: [[Double]], weights: any Weights
    ) -> [[Double]] {
        let hidden = config.hiddenSize
        let heads = config.headCount
        let headDim = config.headDim
        let count = states.count

        // --- Attention ---
        let normed = states.map {
            Self.layerNorm($0, weight: weights.norm1Weight(layer), bias: weights.norm1Bias(layer))
        }
        let qkvWeight = weights.qkvWeight(layer), qkvBias = weights.qkvBias(layer)
        var queries = [[Double]](), keys = [[Double]](), values = [[Double]]()
        for index in 0..<count {
            let projected = Self.linear(
                normed[index], weight: qkvWeight, bias: qkvBias, rows: 3 * hidden, cols: hidden)
            // [q | k | v], each 16 heads of 72, in that order.
            var q = Array(projected[0..<hidden])
            var k = Array(projected[hidden..<(2 * hidden)])
            for head in 0..<heads {
                let span = (head * headDim)..<((head + 1) * headDim)
                q.replaceSubrange(span, with: applyRotary(Array(q[span]), angles: angles[index]))
                k.replaceSubrange(span, with: applyRotary(Array(k[span]), angles: angles[index]))
            }
            queries.append(q)
            keys.append(k)
            values.append(Array(projected[(2 * hidden)..<(3 * hidden)]))
        }

        let scale = 1 / Double(headDim).squareRoot()
        var attended = [[Double]](repeating: [Double](repeating: 0, count: hidden), count: count)
        for head in 0..<heads {
            let offset = head * headDim
            for query in 0..<count {
                // **Bidirectional**: every patch sees every patch, including later ones.
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
        let projWeight = weights.projWeight(layer), projBias = weights.projBias(layer)
        for index in 0..<count {
            let projected = Self.linear(
                attended[index], weight: projWeight, bias: projBias, rows: hidden, cols: hidden)
            for i in 0..<hidden { out[index][i] += projected[i] }
        }

        // --- MLP ---
        let fc1Weight = weights.fc1Weight(layer), fc1Bias = weights.fc1Bias(layer)
        let fc2Weight = weights.fc2Weight(layer), fc2Bias = weights.fc2Bias(layer)
        for index in 0..<count {
            let normed = Self.layerNorm(
                out[index], weight: weights.norm2Weight(layer), bias: weights.norm2Bias(layer))
            let up = Self.linear(
                normed, weight: fc1Weight, bias: fc1Bias,
                rows: config.intermediateSize, cols: hidden
            ).map(Self.gelu)
            let down = Self.linear(
                up, weight: fc2Weight, bias: fc2Bias,
                rows: hidden, cols: config.intermediateSize)
            for i in 0..<hidden { out[index][i] += down[i] }
        }
        return out
    }

    /// Four patches into one text token.
    ///
    /// **The norm runs at 1152, before the concatenation.** `use_postshuffle_norm` is false for
    /// this tower, and the shapes permit either reading: the merger's own norm tensors are
    /// `[1152]`, which is the evidence.
    private func merge(_ states: [[Double]], weights: any Weights) -> [[Double]] {
        let unit = config.spatialMergeSize * config.spatialMergeSize
        let normWeight = weights.mergerNormWeight(), normBias = weights.mergerNormBias()
        let fc1Weight = weights.mergerFC1Weight(), fc1Bias = weights.mergerFC1Bias()
        let fc2Weight = weights.mergerFC2Weight(), fc2Bias = weights.mergerFC2Bias()

        var out: [[Double]] = []
        for token in 0..<(states.count / unit) {
            var wide: [Double] = []
            wide.reserveCapacity(config.mergedWidth)
            for slot in 0..<unit {
                wide += Self.layerNorm(
                    states[token * unit + slot], weight: normWeight, bias: normBias)
            }
            let hiddenLayer = Self.linear(
                wide, weight: fc1Weight, bias: fc1Bias,
                rows: config.mergedWidth, cols: config.mergedWidth
            ).map(Self.gelu)
            out.append(Self.linear(
                hiddenLayer, weight: fc2Weight, bias: fc2Bias,
                rows: config.outHiddenSize, cols: config.mergedWidth))
        }
        return out
    }
}
