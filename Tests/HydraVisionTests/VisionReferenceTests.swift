import Foundation
import HydraCore
import Testing

@testable import HydraVision

/// The reference tower, on a tiny synthetic tower whose every weight is known.
///
/// The real tower is 27 blocks of 1152 and 851 MiB; running it in double precision on the CPU
/// is minutes per image, so the oracle is exercised at a size where the arithmetic is checkable
/// and the *structure* is identical: still a patch embedding plus a resampled learned grid,
/// still rotary inside attention, still bidirectional, still a 2x2 merge with its norm before
/// the concatenation.
///
/// These tests are mostly about **distinguishing readings that all produce finite output**. That
/// is the entire risk surface of a vision tower: there is no crash to find.
@Suite("Vision tower reference")
struct VisionReferenceTests {

    /// A tower small enough to reason about, shaped like the real one.
    static var tiny: Qwen35VisionConfig {
        var config = Qwen35VisionConfig()
        config.depth = 2
        // **Head width 72, which is Qwen's own**, so the tests run the specialized kernel that
        // ships rather than the generic fallback. At width 8 they exercised a code path no model
        // uses and left `vision_attention_72` untested.
        config.hiddenSize = 144
        config.headCount = 2            // head dim 72, as the real tower has
        config.intermediateSize = 32   // a multiple of 8: bf16_gemm reads columns in eights
        config.outHiddenSize = 10
        config.patchSize = 2
        config.inChannels = 3
        config.temporalPatchSize = 2    // patch elements 2*2*2*3 = 24
        config.spatialMergeSize = 2
        config.positionEmbeddingCount = 16   // a 4x4 learned grid
        return config
    }

    private let config = tiny

    /// FNV-1a, so the values are the same on every run and in every process.
    ///
    /// `hashValue` is seeded per process and produced a fixture that changed between runs, which
    /// cost this project a day of chasing a margin that moved on its own.
    static func deterministic(_ count: Int, seed: UInt64) -> [Double] {
        var hash = 0xcbf2_9ce4_8422_2325 ^ seed
        return (0..<count).map { _ in
            hash = (hash ^ 0x9E37) &* 0x0000_0100_0000_01B3
            return Double(hash % 2000) / 1000.0 - 1.0
        }
    }

    struct Fixture: VisionReference.Weights {
        let config: Qwen35VisionConfig
        var seed: UInt64 = 1
        /// Scales one block's attention output projection, to make a patch's influence visible.
        func value(_ tag: UInt64, _ count: Int) -> [Double] {
            VisionReferenceTests.deterministic(count, seed: seed &* 1000 &+ tag)
        }

        func patchWeight() -> [Double] { value(1, config.hiddenSize * config.patchElements) }
        func patchBias() -> [Double] { value(2, config.hiddenSize) }
        func positionEmbedding() -> [Double] {
            value(3, config.positionEmbeddingCount * config.hiddenSize)
        }
        func norm1Weight(_ l: Int) -> [Double] { value(UInt64(10 + l), config.hiddenSize) }
        func norm1Bias(_ l: Int) -> [Double] { value(UInt64(20 + l), config.hiddenSize) }
        func norm2Weight(_ l: Int) -> [Double] { value(UInt64(30 + l), config.hiddenSize) }
        func norm2Bias(_ l: Int) -> [Double] { value(UInt64(40 + l), config.hiddenSize) }
        func qkvWeight(_ l: Int) -> [Double] {
            value(UInt64(50 + l), 3 * config.hiddenSize * config.hiddenSize)
        }
        func qkvBias(_ l: Int) -> [Double] { value(UInt64(60 + l), 3 * config.hiddenSize) }
        func projWeight(_ l: Int) -> [Double] {
            value(UInt64(70 + l), config.hiddenSize * config.hiddenSize)
        }
        func projBias(_ l: Int) -> [Double] { value(UInt64(80 + l), config.hiddenSize) }
        func fc1Weight(_ l: Int) -> [Double] {
            value(UInt64(90 + l), config.intermediateSize * config.hiddenSize)
        }
        func fc1Bias(_ l: Int) -> [Double] { value(UInt64(100 + l), config.intermediateSize) }
        func fc2Weight(_ l: Int) -> [Double] {
            value(UInt64(110 + l), config.hiddenSize * config.intermediateSize)
        }
        func fc2Bias(_ l: Int) -> [Double] { value(UInt64(120 + l), config.hiddenSize) }
        func mergerNormWeight() -> [Double] { value(130, config.hiddenSize) }
        func mergerNormBias() -> [Double] { value(131, config.hiddenSize) }
        func mergerFC1Weight() -> [Double] { value(132, config.mergedWidth * config.mergedWidth) }
        func mergerFC1Bias() -> [Double] { value(133, config.mergedWidth) }
        func mergerFC2Weight() -> [Double] {
            value(134, config.outHiddenSize * config.mergedWidth)
        }
        func mergerFC2Bias() -> [Double] { value(135, config.outHiddenSize) }
    }

    private func run(grid: Qwen35VisionConfig.Grid, patches: [Double]? = nil) -> [[Double]] {
        let reference = VisionReference(config: config)
        let values = patches
            ?? Self.deterministic(grid.patchCount * config.patchElements, seed: 7)
        return reference.forward(
            patches: values, grid: grid, weights: Fixture(config: config))
    }

    @Test("The tower produces one vector a merged token, and they are not degenerate")
    func shapeAndSpread() {
        let grid = Qwen35VisionConfig.Grid(temporal: 1, height: 4, width: 6)
        let out = run(grid: grid)

        #expect(out.count == config.tokenCount(for: grid), "4x6 patches merge to 6 tokens")
        #expect(out.allSatisfy { $0.count == config.outHiddenSize })
        #expect(out.allSatisfy { $0.allSatisfy(\.isFinite) })

        // Finiteness proves nothing on its own: a tower whose attention returned zeros would be
        // finite too. Distinct tokens is the cheapest evidence that patches reached the output.
        let first = out[0], last = out[out.count - 1]
        let difference = zip(first, last).map { abs($0 - $1) }.max() ?? 0
        #expect(difference > 1e-9, "every merged token is identical, so nothing reached them")
    }

    /// Attention is bidirectional, which is the single reading most likely to be got wrong here
    /// because every other attention in this codebase is causal.
    ///
    /// Changing the **last** patch must change the **first** token. Under a causal mask it
    /// cannot: the first token would see only itself.
    @Test("A later patch changes an earlier token")
    func attentionIsBidirectional() {
        let grid = Qwen35VisionConfig.Grid(temporal: 1, height: 4, width: 6)
        var patches = Self.deterministic(grid.patchCount * config.patchElements, seed: 7)
        let baseline = run(grid: grid, patches: patches)

        // Perturb the final patch only.
        let last = (grid.patchCount - 1) * config.patchElements
        for i in last..<(last + config.patchElements) { patches[i] += 0.5 }
        let perturbed = run(grid: grid, patches: patches)

        let firstTokenChange = zip(baseline[0], perturbed[0]).map { abs($0 - $1) }.max() ?? 0
        #expect(
            firstTokenChange > 1e-9,
            "the last patch did not reach the first token, so attention is masked")
    }

    /// The learned grid is resampled with `align_corners`, so the corners land exactly.
    ///
    /// The half-pixel convention is the more common one and is wrong here. It is also the most
    /// invisible error in the file: it shifts every position by a fraction of a cell, smoothly,
    /// with no discontinuity anywhere to notice.
    @Test("Position interpolation puts the corners exactly on the grid's corners")
    func interpolationAlignsCorners() {
        let reference = VisionReference(config: config)
        let side = config.positionGridSide          // 4

        // The first and last rows of an image map onto the first and last rows of the table,
        // with all the weight on one tap.
        let first = reference.interpolationTaps(index: 0, size: 8)
        #expect(first[0].tap == 0 && abs(first[0].weight - 1) < 1e-12)
        let last = reference.interpolationTaps(index: 7, size: 8)
        let dominant = last.max { $0.weight < $1.weight }!
        #expect(dominant.tap == side - 1 && abs(dominant.weight - 1) < 1e-12)

        // And the weights always form a partition of unity, wherever they land.
        for size in [1, 2, 3, 5, 8, 13] {
            for index in 0..<size {
                let taps = reference.interpolationTaps(index: index, size: size)
                let total = taps.reduce(0) { $0 + $1.weight }
                #expect(abs(total - 1) < 1e-12, "size \(size), index \(index) sums to \(total)")
                #expect(taps.allSatisfy { $0.tap >= 0 && $0.tap < side })
            }
        }
    }

    /// A grid the same size as the table is resampled to itself, exactly.
    @Test("An image matching the learned grid resamples to identity")
    func identityResample() {
        let reference = VisionReference(config: config)
        for index in 0..<config.positionGridSide {
            let taps = reference.interpolationTaps(index: index, size: config.positionGridSide)
            let dominant = taps.max { $0.weight < $1.weight }!
            #expect(dominant.tap == index, "index \(index) resampled to \(dominant.tap)")
            #expect(abs(dominant.weight - 1) < 1e-12)
        }
    }

    /// Rotary turns the row against the first half of the frequencies and the column against
    /// the second, and a patch at the origin is not turned at all.
    @Test("The rotary at the origin is the identity")
    func rotaryAtOrigin() {
        let reference = VisionReference(config: config)
        let vector = Self.deterministic(config.headDim, seed: 99)
        let turned = reference.applyRotary(vector, angles: reference.rotaryAngles(y: 0, x: 0))
        #expect(zip(vector, turned).allSatisfy { abs($0 - $1) < 1e-12 })

        // And away from it, it is not.
        let moved = reference.applyRotary(vector, angles: reference.rotaryAngles(y: 3, x: 2))
        #expect(zip(vector, moved).contains { abs($0 - $1) > 1e-6 })
    }

    /// **Which components rotate against each other**, which is the convention this tower does
    /// not share with the text model beside it.
    ///
    /// The tower uses `rotate_half`: component `i` turns against `i + 36`. The text model uses
    /// the interleaved form, `2i` against `2i + 1`, and Qwen's own text config says
    /// `mrope_interleaved: true`. Both are rotations, both preserve norms, both produce entirely
    /// finite towers, and swapping them mixes the wrong pairs of channels.
    ///
    /// A basis vector settles it: with `rotate_half`, energy put into component 0 appears at
    /// component `half`; interleaved, it would appear at component 1. This test was added
    /// because the four falsification runs found that nothing else in this suite could tell the
    /// two apart.
    @Test("The rotary pairs each component with the one half a head away")
    func rotaryPairsAcrossTheHalf() {
        let reference = VisionReference(config: config)
        let half = config.headDim / 2

        var basis = [Double](repeating: 0, count: config.headDim)
        basis[0] = 1
        // A row of 1 with the first frequency 1.0 gives a turn of one radian on component 0.
        let angles = reference.rotaryAngles(y: 1, x: 0)
        let turned = reference.applyRotary(basis, angles: angles)

        #expect(abs(turned[0] - cos(angles[0])) < 1e-12)
        #expect(
            abs(turned[half] - sin(angles[0])) < 1e-12,
            "component 0's partner is not \(half) away, so this is not rotate_half")
        #expect(abs(turned[1]) < 1e-12, "component 1 moved, which is the interleaved convention")

        // Norms are preserved either way, which is exactly why norm is not the check.
        let before = basis.reduce(0) { $0 + $1 * $1 }
        let after = turned.reduce(0) { $0 + $1 * $1 }
        #expect(abs(before - after) < 1e-12)
    }

    /// Row and column occupy different halves of the angle vector, so a patch at `(a, b)` and
    /// one at `(b, a)` are turned differently unless `a == b`.
    @Test("Rows and columns are not interchangeable")
    func rowsAndColumnsDiffer() {
        let reference = VisionReference(config: config)
        let one = reference.rotaryAngles(y: 1, x: 5)
        let other = reference.rotaryAngles(y: 5, x: 1)
        #expect(zip(one, other).contains { abs($0 - $1) > 1e-9 })

        let same = reference.rotaryAngles(y: 3, x: 3)
        let half = config.headDim / 2
        #expect(
            (0..<(half / 2)).allSatisfy { abs(same[$0] - same[half / 2 + $0]) < 1e-12 },
            "at (3,3) the row and column halves must agree")
    }

    /// LayerNorm removes the mean; RMSNorm does not. Every other norm in this codebase is an
    /// RMSNorm, so this states the difference rather than leaving it to be remembered.
    @Test("The norm is a LayerNorm, mean removed and bias added")
    func normIsLayerNorm() {
        let x: [Double] = [1, 2, 3, 4]
        let ones = [Double](repeating: 1, count: 4)
        let zeros = [Double](repeating: 0, count: 4)
        let normed = VisionReference.layerNorm(x, weight: ones, bias: zeros)

        #expect(abs(normed.reduce(0, +)) < 1e-9, "the mean was not removed")
        let variance = normed.reduce(0) { $0 + $1 * $1 } / 4
        #expect(abs(variance - 1) < 1e-6)

        // The bias is added after scaling, not before.
        let biased = VisionReference.layerNorm(x, weight: ones, bias: [10, 10, 10, 10])
        #expect(zip(normed, biased).allSatisfy { abs(($0 + 10) - $1) < 1e-9 })
    }

    /// The tanh approximation, not the exact erf form: the config names
    /// `gelu_pytorch_tanh` and the two differ by more than this project's tolerances.
    @Test("GELU is the tanh approximation")
    func geluIsTheTanhForm() {
        #expect(abs(VisionReference.gelu(0)) < 1e-12)
        #expect(abs(VisionReference.gelu(1) - 0.841192) < 1e-5)
        #expect(abs(VisionReference.gelu(-1) + 0.158808) < 1e-5)
        // Large positive input passes through; large negative is suppressed.
        #expect(abs(VisionReference.gelu(10) - 10) < 1e-6)
        #expect(abs(VisionReference.gelu(-10)) < 1e-6)
    }
}
