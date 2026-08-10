import Foundation

/// The byte arithmetic of an MLX affine-quantized tensor.
///
/// MLX stores a quantized matrix as **three** tensors rather than one: the packed values, a
/// scale per group, and a **bias** per group. Dequantization is `value · scale + bias`, which
/// is what "affine" means and what separates this from every format the project has handled so
/// far — MXFP4 and `q4_0` are both symmetric, scale-only. A reader that assumes symmetry gets
/// weights shifted by the bias, which is a model that still speaks.
///
/// Two details are worth stating because they are easy to get backwards:
///
/// - **Values are packed into 32-bit words, low-order first.** At 4 bits that is 8 values per
///   word, at 8 bits 4. The packing is by *word*, not by byte, so a row's length in words is
///   `cols / valuesPerWord` and never `cols / 2`.
/// - **Groups run along the row.** `group_size` counts columns, so a row of 2816 at group 64
///   carries 44 scales and 44 biases. Scales are indexed by `(row, column / groupSize)`.
///
/// Verified against the published checkpoint's own header rather than inferred: an expert's
/// `gate_proj` is `U32[128, 704, 352]` with `BF16[128, 704, 44]` scales, and 352 × 8 = 2816 =
/// the hidden size, 2816 / 64 = 44.
public struct MLXAffineLayout: Sendable, Equatable {

    /// Bits per value. The published Gemma 4 QAT conversion uses 4 for the experts, attention
    /// and embeddings, and 8 for every layer's dense MLP and router.
    public let bits: Int
    /// Columns sharing one scale and one bias.
    public let groupSize: Int
    public let rows: Int
    public let cols: Int

    public init(bits: Int, groupSize: Int, rows: Int, cols: Int) {
        precondition(bits == 4 || bits == 8, "only 4- and 8-bit affine tensors are published")
        precondition(32 % bits == 0, "values must pack whole into a 32-bit word")
        precondition(cols % groupSize == 0, "a row must divide into whole groups")
        precondition(
            groupSize % (32 / bits) == 0, "a group must divide into whole words")
        self.bits = bits
        self.groupSize = groupSize
        self.rows = rows
        self.cols = cols
    }

    /// Values packed into one 32-bit word: 8 at 4 bits, 4 at 8 bits.
    public var valuesPerWord: Int { 32 / bits }
    /// 32-bit words in one row.
    public var wordsPerRow: Int { cols / valuesPerWord }
    /// Scales — and biases — in one row.
    public var groupsPerRow: Int { cols / groupSize }

    public var weightBytes: Int { rows * wordsPerRow * 4 }
    /// BF16, as the checkpoint stores them.
    public var scaleBytes: Int { rows * groupsPerRow * 2 }
    public var biasBytes: Int { scaleBytes }

    /// Everything the tensor occupies, the three parts together.
    public var totalBytes: Int { weightBytes + scaleBytes + biasBytes }

    /// Bits actually spent per weight, the quantization's overhead included.
    ///
    /// 4.5 at 4 bits and group 64, against MXFP4's 4.25 — the extra quarter-bit is the bias
    /// that MXFP4 does not carry.
    public var bitsPerWeight: Double {
        Double(totalBytes * 8) / Double(rows * cols)
    }
}
