import Foundation
import Testing

@testable import HydraCore

/// The byte arithmetic of MLX's affine quantization, pinned against the published checkpoint.
///
/// Every number here was read from `model-00001-of-00003.safetensors`'s own header, not derived
/// from the format description. That matters because the whole install is built on these sizes:
/// a wrong words-per-row makes every expert blob the wrong length, and the repacker's coverage
/// check would then report a mismatch it cannot explain.
@Suite("MLX affine layout")
struct MLXAffineLayoutTests {

    /// An expert's `gate_proj`, as the checkpoint declares it:
    /// `U32[128, 704, 352]` with `BF16[128, 704, 44]` scales and biases.
    @Test("A 4-bit expert projection matches the published shapes")
    func expertGateMatchesCheckpoint() {
        let layout = MLXAffineLayout(bits: 4, groupSize: 64, rows: 704, cols: 2816)

        #expect(layout.valuesPerWord == 8)
        #expect(layout.wordsPerRow == 352, "352 words × 8 values = 2816 columns")
        #expect(layout.groupsPerRow == 44, "2816 columns / 64 = 44 groups")

        #expect(layout.weightBytes == 704 * 352 * 4)
        #expect(layout.scaleBytes == 704 * 44 * 2)
        #expect(layout.biasBytes == layout.scaleBytes)
    }

    /// `down_proj` runs the other way — `U32[128, 2816, 88]`, `BF16[128, 2816, 11]` — so a
    /// layout that assumed a square matrix or reused the hidden size would pass the test above
    /// and fail here.
    @Test("A 4-bit down projection matches the published shapes")
    func expertDownMatchesCheckpoint() {
        let layout = MLXAffineLayout(bits: 4, groupSize: 64, rows: 2816, cols: 704)
        #expect(layout.wordsPerRow == 88, "88 words × 8 values = 704 columns")
        #expect(layout.groupsPerRow == 11, "704 columns / 64 = 11 groups")
    }

    /// The dense MLP is 8 bits, and the *same* declared shape as a 4-bit tensor of half the
    /// width — `U32[2112, 704]` for 2816 columns. Reading it at 4 bits would silently halve
    /// the row and produce a matrix of the wrong shape from bytes that are all present.
    @Test("An 8-bit dense projection packs four values to the word")
    func denseMLPIsEightBit() {
        let layout = MLXAffineLayout(bits: 8, groupSize: 64, rows: 2112, cols: 2816)
        #expect(layout.valuesPerWord == 4)
        #expect(layout.wordsPerRow == 704, "704 words × 4 values = 2816 columns")
        #expect(layout.groupsPerRow == 44)

        // The trap, stated: a `U32[2112, 704]` header says nothing about the bit width, and
        // the same 704 words mean 2816 columns at 8 bits and 5632 at 4. Only the config's
        // per-tensor quantization map distinguishes them, which is why it has to be read
        // rather than assumed uniform.
        let asFourBit = MLXAffineLayout(bits: 4, groupSize: 64, rows: 2112, cols: 5632)
        #expect(asFourBit.wordsPerRow == layout.wordsPerRow, "the declared shape is identical")
        #expect(asFourBit.cols == 2 * layout.cols, "and the matrix is twice as wide")
    }

    /// The overhead is what separates this from the formats already supported.
    @Test("Affine costs a quarter-bit more than MXFP4 for the bias")
    func overheadIsTheBias() {
        let four = MLXAffineLayout(bits: 4, groupSize: 64, rows: 704, cols: 2816)
        #expect(abs(four.bitsPerWeight - 4.5) < 1e-9)

        let eight = MLXAffineLayout(bits: 8, groupSize: 64, rows: 2112, cols: 2816)
        #expect(abs(eight.bitsPerWeight - 8.5) < 1e-9)

        // MXFP4 carries a scale and no bias: 4.25 bits. The extra quarter-bit here buys the
        // asymmetry, which is why the two cannot share a decoder.
        #expect(four.bitsPerWeight > 4.25)
    }

    /// One expert against the BF16 build, which is the number the whole exercise is for.
    @Test("An MLX expert is 3.55x smaller than the BF16 one")
    func expertBlobShrinks() {
        let hidden = 2816
        let inner = 704
        let gate = MLXAffineLayout(bits: 4, groupSize: 64, rows: inner, cols: hidden)
        let up = gate
        let down = MLXAffineLayout(bits: 4, groupSize: 64, rows: hidden, cols: inner)
        let quantized = gate.totalBytes + up.totalBytes + down.totalBytes

        // BF16 stores the same three matrices at two bytes a weight.
        let bf16 = 3 * inner * hidden * 2
        #expect(quantized == 3_345_408, "an expert is \(quantized) B")
        #expect(bf16 == 11_894_784)

        let ratio = Double(bf16) / Double(quantized)
        #expect(ratio > 3.5 && ratio < 3.6, "ratio \(ratio)")

        // Thirty layers of 128 experts: what actually streams from SSD.
        let pool = 30 * 128 * quantized
        #expect(pool < 13_000_000_000, "expert pool \(pool) B")
    }
}
