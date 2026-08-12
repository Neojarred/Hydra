import Foundation
import Testing

@testable import HydraFormat

/// BF16 is the format of every unquantized weight in GPT-OSS. Confusing it with `Float16`
/// raises no error and produces plausible but wrong values, which is exactly what
/// happened in the first version of the Metal kernel.
struct BF16Tests {

    @Test("Conversion to Float is exact")
    func exactValues() {
        // 0x3F80 = 1.0 in BF16 (the top half of the Float32 0x3F800000).
        #expect(BF16.toFloat(0x3F80) == 1.0)
        #expect(BF16.toFloat(0x0000) == 0.0)
        #expect(BF16.toFloat(0xBF80) == -1.0)
        #expect(BF16.toFloat(0x4000) == 2.0)
        #expect(BF16.toFloat(0x3F00) == 0.5)
        #expect(BF16.toFloat(0x4049) == 3.140625)  // π approximated in BF16
    }

    /// The test that documents the trap: the same bytes, two interpretations.
    @Test("BF16 and Float16 do not coincide")
    func bf16IsNotFloat16() {
        let bits: UInt16 = 0x3F80
        #expect(BF16.toFloat(bits) == 1.0)
        #expect(Float(Float16(bitPattern: bits)) == 1.875)
        // 0.875 of difference on a value of 1.0: enough to break everything, too little
        // to raise an alarm.
        #expect(abs(Float(Float16(bitPattern: bits)) - BF16.toFloat(bits)) == 0.875)
    }

    @Test("The round trip is exact on representable values")
    func roundTripIsExactWhenRepresentable() {
        // BF16 has only 8 mantissa bits: only values whose mantissa fits in 7 explicit
        // bits come back unchanged.
        for value: Float in [0, 1, -1, 0.5, 2, 256, -0.125, 3.140625, 0.00390625] {
            let encoded = BF16.fromFloat(value)
            #expect(BF16.toFloat(encoded) == value, "\(value)")
        }
    }

    /// What BF16 actually costs, and why it is acceptable here: 8 mantissa bits give a
    /// relative precision of 2⁻⁸, i.e. 0.39 %. That is the precision at which GPT-OSS
    /// stores its attention and LM head, not an approximation we introduce.
    ///
    @Test("Precision loss stays bounded by 2⁻⁸ in relative terms")
    func precisionLossIsBounded() {
        let bound = Double(exp2(-8.0))
        for value: Float in [1e-8, 1e20, 0.1, -12345.678, 6.02e23, 1.0 / 3.0] {
            let restored = BF16.toFloat(BF16.fromFloat(value))
            let relative = abs(Double(restored) - Double(value)) / Double(abs(value))
            #expect(relative <= bound, "\(value): relative deviation \(relative)")
        }
    }

    @Test("Special values carry over unchanged")
    func specialValues() {
        #expect(BF16.toFloat(0x7F80) == .infinity)
        #expect(BF16.toFloat(0xFF80) == -.infinity)
        #expect(BF16.toFloat(0x7FC0).isNaN)
        #expect(BF16.fromFloat(.infinity) == 0x7F80)
        #expect(BF16.toFloat(BF16.fromFloat(.nan)).isNaN)
    }

    @Test("Decoding a buffer respects little-endianness")
    func decodesLittleEndianBuffer() {
        // 1.0 then -2.0, little-endian as in the safetensors.
        let data = Data([0x80, 0x3F, 0x00, 0xC0])
        let values = BF16.decode(data)
        #expect(values == [1.0, -2.0])
    }

    @Test("Rounding goes to nearest even")
    func roundsToNearestEven() {
        // A Float32 exactly between two BF16s must land on the even mantissa.
        let midpoint = Float(bitPattern: 0x3F80_8000)  // entre 1.0 et 1.0078125
        let encoded = BF16.fromFloat(midpoint)
        #expect(encoded == 0x3F80, "arrondi obtenu : 0x\(String(encoded, radix: 16))")
    }
}
