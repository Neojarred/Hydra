import Foundation
import Testing

@testable import HydraReference

/// MLX's affine decoder, pinned on the two conventions that cannot be guessed.
///
/// Both were settled empirically rather than read off a specification, by decoding a real
/// tensor from the published checkpoint four ways and comparing each against the same tensor in
/// Google's QAT weights:
///
/// | packing | bias | relative error |
/// |---|---|---:|
/// | low-order first | applied | **0.078** |
/// | low-order first | ignored | 1.854 |
/// | high-order first | applied | 1.148 |
/// | high-order first | ignored | 2.138 |
///
/// 0.078 is what 4-bit quantization at group 64 should cost. The others are noise. The same
/// experiment against the *non-QAT* BF16 weights gives 0.43 for the correct convention, which
/// is how we know this build descends from the QAT checkpoint and not from the original.
@Suite("MLX affine dequantization")
struct MLXAffineTests {

    /// The first value in a word occupies the **least** significant bits.
    ///
    /// Getting this backwards leaves every byte present and every weight in the wrong column,
    /// each group of eight reversed. Nothing about the file's size or checksum changes.
    @Test("Values unpack low-order first")
    func unpackingIsLowOrderFirst() {
        // 0x76543210: nibbles 0,1,2,…,7 reading from the low end.
        #expect(MLXAffine.unpack(words: [0x7654_3210], bits: 4) == [0, 1, 2, 3, 4, 5, 6, 7])
        // At 8 bits: bytes 0x10, 0x32, 0x54, 0x76.
        #expect(MLXAffine.unpack(words: [0x7654_3210], bits: 8) == [0x10, 0x32, 0x54, 0x76])

        #expect(MLXAffine.unpack(words: [0xFFFF_FFFF], bits: 4) == Array(repeating: 15, count: 8))
        #expect(MLXAffine.unpack(words: [0], bits: 4) == Array(repeating: 0, count: 8))
    }

    /// `q · scale + bias`, with the bias applied, the difference from every symmetric format
    /// the project already decodes.
    @Test("Dequantization is affine, not symmetric")
    func dequantizationAppliesTheBias() {
        // One group of 8, so the arithmetic is checkable by hand.
        let words: [UInt32] = [0x7654_3210]
        let got = MLXAffine.dequantize(
            words: words, scales: [0.5], biases: [-4.0], bits: 4, groupSize: 8)

        #expect(got == [-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5])

        // Dropping the bias shifts every weight by a per-group constant, a model that still
        // speaks, which is why this is asserted rather than assumed.
        let symmetric = zip(got, MLXAffine.unpack(words: words, bits: 4)).map {
            $0 - Double($1) * 0.5
        }
        #expect(symmetric.allSatisfy { abs($0 - (-4.0)) < 1e-12 })
    }

    /// Each group carries its own scale and bias, indexed along the row.
    @Test("Groups run along the row")
    func groupsAreIndexedByColumn() {
        let words: [UInt32] = [0x0000_0000, 0x1111_1111]  // eight 0s, then eight 1s
        let got = MLXAffine.dequantize(
            words: words, scales: [1.0, 10.0], biases: [0.0, 100.0],
            bits: 4, groupSize: 8)

        #expect(Array(got[0..<8]) == Array(repeating: 0.0, count: 8))
        #expect(Array(got[8..<16]) == Array(repeating: 110.0, count: 8))
    }

    /// The matrix-vector product a kernel has to reproduce.
    @Test("A quantized matvec matches a dequantize-then-multiply")
    func matvecMatchesDequantized() {
        let rows = 3, cols = 16, bits = 4, group = 8
        let wordsPerRow = cols / 8
        var words: [UInt32] = []
        var scales: [Double] = []
        var biases: [Double] = []
        for row in 0..<rows {
            words += [UInt32(0x7654_3210 &+ UInt32(row)), UInt32(0x0123_4567 &- UInt32(row))]
            scales += [0.25 + Double(row) * 0.1, 0.5]
            biases += [-1.0, Double(row) * 0.5]
        }
        let x = (0..<cols).map { Double($0 % 5) - 2.0 }

        let got = MLXAffine.matvec(
            words: words, scales: scales, biases: biases,
            rows: rows, cols: cols, bits: bits, groupSize: group, x: x)

        for row in 0..<rows {
            let decoded = MLXAffine.dequantize(
                words: Array(words[(row * wordsPerRow)..<((row + 1) * wordsPerRow)]),
                scales: Array(scales[(row * 2)..<((row + 1) * 2)]),
                biases: Array(biases[(row * 2)..<((row + 1) * 2)]),
                bits: bits, groupSize: group)
            let expected = zip(decoded, x).reduce(0) { $0 + $1.0 * $1.1 }
            #expect(abs(got[row] - expected) < 1e-12)
        }
    }
}
