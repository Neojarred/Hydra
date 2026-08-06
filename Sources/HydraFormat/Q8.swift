import Foundation

/// Symmetric 8-bit quantization, in blocks of 32 values along a row.
///
/// This is the candidate for the dense weights — attention projections and the LM head —
/// which D-020 identifies as 66 % of the bytes read per token while the experts, the point
/// of the project, are only 34 %.
///
/// **Layout: 32 `Int8` then one BF16 scale**, i.e. 34 bytes per 32 weights, 8.5 bits each
/// against BF16's 16. The block size is deliberately MXFP4's: `ExpertBlobLayout`, the
/// decoders and the shaders' loop structure carry over instead of being reinvented.
///
/// The scale is BF16 rather than E8M0. A power-of-two scale wastes up to a factor of two on
/// the range, which is tolerable when only four bits carry the value and wasteful when
/// eight do — it would cost about one of the eight.
///
/// **Symmetric, not affine.** No zero point: `value = level × scale`. A weight distribution
/// centred on zero gains nothing from an offset, and the asymmetric form would cost one
/// subtraction per weight in the hot loop plus a second constant per block.
public enum Q8 {

    /// Values per block, matching `MXFP4Layout.blockSize`.
    public static let blockSize = 32
    /// Levels, then the scale.
    public static let bytesPerBlock = blockSize + 2

    /// The number of bytes `count` weights occupy once quantized.
    public static func encodedByteCount(values count: Int) -> Int {
        precondition(count % blockSize == 0, "Q8 quantizes whole blocks of \(blockSize)")
        return count / blockSize * bytesPerBlock
    }

    public enum CodingError: Error, CustomStringConvertible {
        case misalignedCount(Int)
        case truncated(expected: Int, got: Int)

        public var description: String {
            switch self {
            case .misalignedCount(let n):
                return "\(n) values is not a multiple of the \(Q8.blockSize)-value block"
            case let .truncated(expected, got):
                return "truncated Q8 payload: \(got) bytes, \(expected) expected"
            }
        }
    }

    // MARK: - Encoding

    /// Quantizes one block of exactly `blockSize` values.
    ///
    /// The scale is derived from the block's largest magnitude, then **rounded to BF16
    /// before** being used to compute the levels. Deriving the levels from a scale the
    /// decoder will never see would build in an error the round-trip could not explain.
    ///
    /// A block that is entirely zero gets a zero scale: every level is then zero, and
    /// decoding gives back exact zeros rather than the noise a fabricated scale would leave.
    public static func encodeBlock(_ values: ArraySlice<Float>) -> (levels: [Int8], scale: UInt16) {
        precondition(values.count == blockSize)

        var magnitude: Float = 0
        for value in values where value.isFinite {
            magnitude = max(magnitude, abs(value))
        }
        guard magnitude > 0 else { return ([Int8](repeating: 0, count: blockSize), 0) }

        // 127, not 128: the range stays symmetric, so a weight and its opposite quantize to
        // opposite levels. Using -128 would make the encoding lopsided for no gain.
        let scaleBits = BF16.fromFloat(magnitude / 127)
        let scale = BF16.toFloat(scaleBits)
        guard scale > 0 else { return ([Int8](repeating: 0, count: blockSize), 0) }

        var levels = [Int8](repeating: 0, count: blockSize)
        for (index, value) in values.enumerated() {
            // Rounding to BF16 can push the scale slightly below `magnitude / 127`, which
            // would send the extreme weight to 128 and wrap. The clamp is not defensive
            // programming, it is load-bearing.
            let level = (value / scale).rounded(.toNearestOrEven)
            levels[index] = Int8(max(-127, min(127, level.isFinite ? level : 0)))
        }
        return (levels, scaleBits)
    }

    /// Quantizes a whole row. `values.count` must be a multiple of `blockSize`.
    public static func encode(_ values: [Float]) throws -> Data {
        guard values.count % blockSize == 0 else {
            throw CodingError.misalignedCount(values.count)
        }
        var out = Data(capacity: encodedByteCount(values: values.count))
        var start = values.startIndex
        while start < values.endIndex {
            let (levels, scale) = encodeBlock(values[start..<(start + blockSize)])
            levels.withUnsafeBufferPointer { out.append(UnsafeRawBufferPointer($0).bindMemory(to: UInt8.self)) }
            out.append(UInt8(scale & 0xFF))
            out.append(UInt8(scale >> 8))
            start += blockSize
        }
        return out
    }

    // MARK: - Decoding

    /// Decodes a payload produced by `encode`.
    public static func decode(_ data: Data) throws -> [Float] {
        guard data.count % bytesPerBlock == 0 else {
            throw CodingError.truncated(
                expected: (data.count / bytesPerBlock + 1) * bytesPerBlock, got: data.count)
        }
        let blocks = data.count / bytesPerBlock
        var out = [Float](repeating: 0, count: blocks * blockSize)
        data.withUnsafeBytes { raw in
            for block in 0..<blocks {
                let base = block * bytesPerBlock
                let scale = BF16.toFloat(
                    UInt16(raw[base + blockSize]) | (UInt16(raw[base + blockSize + 1]) << 8))
                for i in 0..<blockSize {
                    out[block * blockSize + i] = Float(Int8(bitPattern: raw[base + i])) * scale
                }
            }
        }
        return out
    }

    // MARK: - Simulation

    /// Quantizes then immediately dequantizes a BF16 buffer **in place**.
    ///
    /// This is what makes D-020's quality gate cheap: the values the existing `bf16_gemv`
    /// then reads are exactly those a real Q8 kernel would produce, so the effect on the
    /// logits can be measured **before** writing a kernel, a disk format, or a repacker
    /// path. Only the throughput is not measured — the bytes read are unchanged.
    ///
    /// The result is very slightly **worse** than true Q8, because the dequantized value is
    /// rounded a second time to land back in BF16. The gate is therefore conservative: a
    /// verdict of "no measurable loss" here holds a fortiori for the real thing.
    ///
    /// - Returns: the largest relative deviation introduced, against the block's magnitude.
    @discardableResult
    public static func simulateInPlace(_ words: UnsafeMutableBufferPointer<UInt16>) -> Float {
        var worst: Float = 0
        var start = 0
        while start + blockSize <= words.count {
            var block = [Float](repeating: 0, count: blockSize)
            for i in 0..<blockSize { block[i] = BF16.toFloat(words[start + i]) }

            let (levels, scaleBits) = encodeBlock(block[...])
            let scale = BF16.toFloat(scaleBits)
            var magnitude: Float = 0
            for value in block { magnitude = max(magnitude, abs(value)) }

            for i in 0..<blockSize {
                let restored = Float(levels[i]) * scale
                words[start + i] = BF16.fromFloat(restored)
                if magnitude > 0 {
                    worst = max(worst, abs(BF16.toFloat(words[start + i]) - block[i]) / magnitude)
                }
            }
            start += blockSize
        }
        // A trailing partial block is left in BF16 rather than quantized on a truncated
        // range. None of GPT-OSS's dimensions produce one; a future model might.
        return worst
    }
}
