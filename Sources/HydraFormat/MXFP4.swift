import Foundation

/// Decoding of the OCP Microscaling MXFP4 format, as used by GPT-OSS's experts.
///
/// Layout verified against the real safetensors headers and against the reference
/// d'OpenAI (`gpt_oss/torch/weights.py`), voir docs/01-DECISIONS.md, D-011 :
///
///   - a block covers **32 consecutive values of the last dimension**;
///   - it is stored in **16 bytes**: two E2M1 FP4 values per `UInt8`;
///   - the **low nibble carries the even index**, the high nibble the odd one;
///   - one **E8M0 scale byte** per block, applied as `value * 2^(scale - 127)`.
///
/// That is 4.25 bits per weight. Getting the nibble order wrong produces a model that
/// generates plausible but degraded text, with no error raised, hence the systematic
/// numerical validation of this path.
public enum MXFP4 {

    /// The number of values one block covers.
    public static let blockSize = 32

    /// Packed value bytes per block (32 values × 4 bits).
    public static let packedBytesPerBlock = 16

    /// Scale bytes per block.
    public static let scaleBytesPerBlock = 1

    /// The E2M1 table: 1 sign bit, 2 exponent bits, 1 mantissa bit.
    /// The index is the raw nibble, 0 to 15.
    public static let valueTable: [Float] = [
        +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ]

    /// The table pre-multiplied by the scale, to avoid one multiplication per value.
    /// (16 entries, reused as-is for each block.)
    @inline(__always)
    private static func scaleFactor(_ scaleByte: UInt8) -> Float {
        // Matching the reference: `exp = scale.int() - 127`, then `ldexp`.
        // The byte 0xFF encodes a NaN in the OCP specification; the reference produces an
        // infinite factor here. We reproduce that rather than diverge from it.
        exp2(Float(Int(scaleByte) - 127))
    }

    public enum DecodeError: Error, CustomStringConvertible {
        case packedSizeMismatch(expected: Int, got: Int)
        case scaleSizeMismatch(expected: Int, got: Int)
        case outputSizeMismatch(expected: Int, got: Int)

        public var description: String {
            switch self {
            case let .packedSizeMismatch(e, g):
                return "MXFP4: \(g) packed bytes supplied, \(e) expected"
            case let .scaleSizeMismatch(e, g):
                return "MXFP4: \(g) scale bytes supplied, \(e) expected"
            case let .outputSizeMismatch(e, g):
                return "MXFP4: output buffer of \(g) values, \(e) expected"
            }
        }
    }

    /// Decodes `blockCount` consecutive blocks.
    ///
    /// - Parameters:
    ///   - packed: `blockCount * 16` bytes of packed FP4 values.
    ///   - scales: `blockCount` octets d'exposants E8M0.
    ///   - output: tampon de `blockCount * 32` `Float`.
    public static func decode(
        packed: UnsafeRawBufferPointer,
        scales: UnsafeRawBufferPointer,
        blockCount: Int,
        into output: UnsafeMutableBufferPointer<Float>
    ) throws {
        guard packed.count == blockCount * packedBytesPerBlock else {
            throw DecodeError.packedSizeMismatch(
                expected: blockCount * packedBytesPerBlock, got: packed.count)
        }
        guard scales.count == blockCount * scaleBytesPerBlock else {
            throw DecodeError.scaleSizeMismatch(expected: blockCount, got: scales.count)
        }
        guard output.count == blockCount * blockSize else {
            throw DecodeError.outputSizeMismatch(
                expected: blockCount * blockSize, got: output.count)
        }

        valueTable.withUnsafeBufferPointer { table in
            for block in 0..<blockCount {
                let factor = scaleFactor(scales[block])
                let src = block * packedBytesPerBlock
                let dst = block * blockSize

                for i in 0..<packedBytesPerBlock {
                    let byte = packed[src + i]
                    // Nibble bas -> index pair, nibble haut -> index impair.
                    output[dst + 2 * i] = table[Int(byte & 0x0F)] * factor
                    output[dst + 2 * i + 1] = table[Int(byte >> 4)] * factor
                }
            }
        }
    }

    /// A convenience variant over `Data`. It allocates the output array; reserved for tests and
    /// tooling. The inference path uses the pointer version, which does not
    /// allocate.
    public static func decode(packed: Data, scales: Data) throws -> [Float] {
        let blockCount = scales.count
        var out = [Float](repeating: 0, count: blockCount * blockSize)
        try packed.withUnsafeBytes { p in
            try scales.withUnsafeBytes { s in
                try out.withUnsafeMutableBufferPointer { o in
                    try decode(packed: p, scales: s, blockCount: blockCount, into: o)
                }
            }
        }
        return out
    }

    /// The bytes occupied by an MXFP4 matrix of `rows` rows and `cols` columns, with blocks cut
    /// along the columns.
    public static func byteSize(rows: Int, cols: Int) -> (blocks: Int, scales: Int) {
        precondition(cols % blockSize == 0, "cols must be a multiple of \(blockSize)")
        let blocksPerRow = cols / blockSize
        return (rows * blocksPerRow * packedBytesPerBlock, rows * blocksPerRow)
    }
}
