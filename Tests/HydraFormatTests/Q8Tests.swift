import Foundation
import Testing

@testable import HydraFormat

/// Symmetric 8-bit quantization of the dense weights.
///
/// The property that matters is not "the code round-trips" but **the size of the error it
/// introduces**: D-020 gates the whole idea on quality, and a quantizer that silently loses
/// more than its format allows would make that measurement meaningless. Every test here
/// therefore states a bound rather than an equality, except where exactness is required.
struct Q8Tests {

    /// A deterministic distribution shaped like real weights: centred on zero, mostly small,
    /// with the occasional outlier that sets the block's scale.
    private func syntheticRow(count: Int, seed: UInt32 = 0x2545_F491) -> [Float] {
        var state = seed
        var out = [Float](repeating: 0, count: count)
        for i in 0..<count {
            state = state &* 1_103_515_245 &+ 12345
            let uniform = Float(state >> 8) / Float(1 << 24) * 2 - 1
            out[i] = uniform * uniform * uniform * 0.4
            if i % 97 == 0 { out[i] *= 12 }  // the outlier that stretches the scale
        }
        return out
    }

    private func worstRelative(_ original: [Float], _ restored: [Float]) -> Float {
        var worst: Float = 0
        var start = 0
        while start + Q8.blockSize <= original.count {
            var magnitude: Float = 0
            for i in start..<(start + Q8.blockSize) { magnitude = max(magnitude, abs(original[i])) }
            if magnitude > 0 {
                for i in start..<(start + Q8.blockSize) {
                    worst = max(worst, abs(restored[i] - original[i]) / magnitude)
                }
            }
            start += Q8.blockSize
        }
        return worst
    }

    // MARK: - Format

    @Test("A block is 34 bytes for 32 weights")
    func blockSize() {
        #expect(Q8.blockSize == 32)
        #expect(Q8.bytesPerBlock == 34)
        #expect(Q8.encodedByteCount(values: 2880) == 2880 / 32 * 34)
        // 8.5 bits per weight against BF16's 16: the 47 % that D-020 is after.
        #expect(Double(Q8.encodedByteCount(values: 2880)) / Double(2880 * 2) < 0.54)
    }

    @Test("The block size matches MXFP4's")
    func sharesTheMXFP4Block() {
        #expect(Q8.blockSize == MXFP4.blockSize)
    }

    @Test("A misaligned count is refused rather than truncated")
    func refusesPartialBlocks() {
        #expect(throws: Q8.CodingError.self) { try Q8.encode([Float](repeating: 1, count: 40)) }
    }

    // MARK: - Accuracy

    /// The bound the format allows: a level covers 1/127 of the block's magnitude, so a
    /// rounding to nearest cannot exceed half of that. The BF16 scale adds its own rounding,
    /// hence the margin.
    @Test("The round trip stays inside the format's error bound")
    func roundTripIsBounded() throws {
        let row = syntheticRow(count: 2880)
        let restored = try Q8.decode(try Q8.encode(row))

        #expect(restored.count == row.count)
        let worst = worstRelative(row, restored)
        #expect(worst < 0.008, "worst relative deviation \(worst), above the 1/127 the format allows")
    }

    /// A single weight matters less than the whole vector: what a GEMV propagates is the
    /// accumulated sum, not each term. This is the measure `Bench.deviation` already uses.
    @Test("The error on a dot product stays under a thousandth")
    func dotProductBarelyMoves() throws {
        let weights = syntheticRow(count: 2880)
        let input = syntheticRow(count: 2880, seed: 0x9E37_79B9)
        let restored = try Q8.decode(try Q8.encode(weights))

        var exact = 0.0, approximate = 0.0, scale = 0.0
        for i in 0..<weights.count {
            exact += Double(weights[i]) * Double(input[i])
            approximate += Double(restored[i]) * Double(input[i])
            scale += abs(Double(weights[i]) * Double(input[i]))
        }
        let relative = abs(approximate - exact) / max(scale, 1e-9)
        #expect(relative < 1e-3, "relative deviation \(relative) on the dot product")
    }

    // MARK: - Exactness where it is required

    @Test("An all-zero block decodes to exact zeros")
    func zeroBlockStaysZero() throws {
        let restored = try Q8.decode(try Q8.encode([Float](repeating: 0, count: 64)))
        #expect(restored.allSatisfy { $0 == 0 })
    }

    /// The trap the clamp exists for: rounding the scale to BF16 can push it just below
    /// `magnitude / 127`, sending the extreme weight to 128 — which wraps to -128 in `Int8`
    /// and flips the sign of the largest weight in the block.
    @Test("The extreme value does not wrap")
    func extremeValueDoesNotWrap() {
        // 0.3 has no exact BF16 representation, so the scale really is rounded here.
        for magnitude in [Float(0.3), 1.7, 6.1, 0.017, 1e-4, 1e4] {
            var block = [Float](repeating: magnitude / 400, count: Q8.blockSize)
            block[7] = magnitude
            block[19] = -magnitude
            let (levels, scaleBits) = Q8.encodeBlock(block[...])
            #expect(levels[7] > 0, "the largest weight wrapped at magnitude \(magnitude)")
            #expect(levels[19] < 0, "the smallest weight wrapped at magnitude \(magnitude)")
            #expect(levels[7] <= 127 && levels[19] >= -127)

            let scale = BF16.toFloat(scaleBits)
            let restored = Float(levels[7]) * scale
            #expect(abs(restored - magnitude) / magnitude < 0.01)
        }
    }

    @Test("Quantization is symmetric")
    func oppositeWeightsGiveOppositeLevels() {
        let row = syntheticRow(count: Q8.blockSize)
        let (positive, scaleA) = Q8.encodeBlock(row[...])
        let (negative, scaleB) = Q8.encodeBlock(row.map { -$0 }[...])
        #expect(scaleA == scaleB, "negating a block must not change its scale")
        for i in 0..<Q8.blockSize {
            #expect(positive[i] == -negative[i], "asymmetry at index \(i)")
        }
    }

    // MARK: - Simulation

    /// `simulateInPlace` is what D-020's gate runs on. It must introduce the same error as
    /// the real thing — plus the second rounding back to BF16, never less.
    @Test("The simulation reproduces the quantization error")
    func simulationMatchesTheRealThing() throws {
        let row = syntheticRow(count: 2880)
        let asBF16 = row.map { BF16.fromFloat($0) }
        let reference = asBF16.map { BF16.toFloat($0) }

        var words = asBF16
        let reported: Float = words.withUnsafeMutableBufferPointer { Q8.simulateInPlace($0) }
        let simulated = words.map { BF16.toFloat($0) }

        let measured = worstRelative(reference, simulated)
        #expect(abs(reported - measured) < 1e-6, "the reported deviation does not match the values")
        #expect(measured < 0.012, "the simulation loses more than the format allows: \(measured)")

        // Conservative: never better than true Q8 on the same values.
        let real = try Q8.decode(try Q8.encode(reference))
        #expect(measured >= worstRelative(reference, real) - 1e-6)
    }

    @Test("The simulation leaves a trailing partial block untouched")
    func simulationLeavesTheTailAlone() {
        var words = syntheticRow(count: Q8.blockSize + 5).map { BF16.fromFloat($0) }
        let tail = Array(words.suffix(5))
        words.withUnsafeMutableBufferPointer { _ = Q8.simulateInPlace($0) }
        #expect(Array(words.suffix(5)) == tail)
    }
}
