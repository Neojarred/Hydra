import Foundation

/// bfloat16 conversion, the format of **every** unquantized weight in GPT-OSS:
/// attention, routers, norms, sinks, expert biases, embedding and LM head.
///
/// **BF16 is not Float16.** `Float16` is IEEE binary16 (1 sign bit, 5 exponent, 10
/// mantissa); bfloat16 is the top half of a `Float32` (1, 8, 7). Both are two bytes, which
/// makes the confusion easy, and silent: reading one as the other produces neither an
/// error nor a NaN, only plausible, wrong values.
/// 1,0 en BF16 se lit 1,875 en Float16.
///
/// This trap was found by a test on the Metal kernel, not by review. Hence centralizing it
/// here: nowhere else in the project should reinterpret two bytes by hand.
///
public enum BF16 {

    /// An exact, lossless conversion: the 16 bits become the top half of a Float32. No
    /// rounding, no special case, NaNs and infinities carry over.
    @inline(__always)
    public static func toFloat(_ bits: UInt16) -> Float {
        Float(bitPattern: UInt32(bits) << 16)
    }

    /// The reverse, rounding to nearest with ties to even, the same as the reference
    /// libraries use, so that round trips are stable.
    @inline(__always)
    public static func fromFloat(_ value: Float) -> UInt16 {
        let bits = value.bitPattern
        if value.isNaN { return UInt16(bits >> 16) | 0x0040 }
        let rounding: UInt32 = 0x7FFF &+ ((bits >> 16) & 1)
        return UInt16(truncatingIfNeeded: (bits &+ rounding) >> 16)
    }

    /// Decodes a contiguous BF16 buffer. The bytes are little-endian, as in the safetensors
    /// and in our `.hydra` files.
    public static func decode(_ data: Data) -> [Float] {
        let count = data.count / 2
        var out = [Float](repeating: 0, count: count)
        data.withUnsafeBytes { raw in
            for i in 0..<count {
                let bits = raw.loadUnaligned(fromByteOffset: i * 2, as: UInt16.self)
                out[i] = toFloat(UInt16(littleEndian: bits))
            }
        }
        return out
    }

    /// Encodes a sequence of `Float` as little-endian BF16.
    public static func encode(_ values: [Float]) -> Data {
        var out = Data(capacity: values.count * 2)
        for value in values {
            withUnsafeBytes(of: fromFloat(value).littleEndian) { out.append(contentsOf: $0) }
        }
        return out
    }
}
