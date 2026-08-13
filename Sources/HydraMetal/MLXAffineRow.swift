import Foundation
import HydraCore
import HydraFormat
import Metal

/// Reads one row of an MLX affine-quantized matrix on the CPU.
///
/// Written once and shared, because both quantized checkpoints need exactly this and for the
/// same reason: the embedding table is read a row at a time on the way in, by the CPU, before
/// anything reaches the GPU. Reading it as BF16, which is what the unquantized path does, yields
/// packed integers reinterpreted as floats: finite, enormous, and nothing to do with the token.
///
/// A second transcription of `q · scale + bias` is a second chance to get the group arithmetic
/// wrong, so there is one.
public enum MLXAffineRow {

    /// Dequantizes row `index` into `destination`, which must be `layout.cols` long.
    ///
    /// - Parameters:
    ///   - words, scales, biases: byte offsets of the three tensors within `bytes`.
    public static func read(
        row index: Int, into destination: UnsafeMutableBufferPointer<Float>,
        bytes: UnsafeRawBufferPointer, layout: MLXAffineLayout,
        words: Int, scales: Int, biases: Int, bits: Int, groupSize: Int
    ) {
        precondition(destination.count == layout.cols)
        let perWord = layout.valuesPerWord
        let mask = UInt32((1 << bits) - 1)
        let wordBase = words + index * layout.wordsPerRow * 4
        let groupBase = scales + index * layout.groupsPerRow * 2
        let biasBase = biases + index * layout.groupsPerRow * 2

        for word in 0..<layout.wordsPerRow {
            let packed = UInt32(
                littleEndian: bytes.loadUnaligned(
                    fromByteOffset: wordBase + word * 4, as: UInt32.self))
            for slot in 0..<perWord {
                let column = word * perWord + slot
                // The last word of a row can run past the columns when the width is not a
                // multiple of the packing, and writing those would be out of bounds.
                if column >= layout.cols { break }
                let group = column / groupSize
                let scale = BF16.toFloat(
                    UInt16(littleEndian: bytes.loadUnaligned(
                        fromByteOffset: groupBase + group * 2, as: UInt16.self)))
                let bias = BF16.toFloat(
                    UInt16(littleEndian: bytes.loadUnaligned(
                        fromByteOffset: biasBase + group * 2, as: UInt16.self)))
                let q = Float((packed >> UInt32(slot * bits)) & mask)
                destination[column] = q * scale + bias
            }
        }
    }
}
