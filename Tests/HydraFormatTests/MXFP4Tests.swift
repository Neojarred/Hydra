import Foundation
import Testing

@testable import HydraFormat

/// Milestone 1.2 — the MXFP4 decoder must agree with an independent reference
/// implementation. The phase plan's criterion is a relative error < 1e-6; since every
/// value involved is exactly representable in float32, we require **bit-for-bit equality**
/// in practice, which is strictly stronger.
struct MXFP4Tests {

    private static func fixture(_ name: String) throws -> Data {
        let url = try #require(
            Bundle.module.url(forResource: "Fixtures/\(name)", withExtension: nil),
            "fixture « \(name) » introuvable — lancer tools/gen_mxfp4_fixture.py")
        return try Data(contentsOf: url)
    }

    @Test("Decoding matches the reference vector, bit for bit")
    func matchesReferenceVector() throws {
        let packed = try Self.fixture("mxfp4_packed.bin")
        let scales = try Self.fixture("mxfp4_scales.bin")
        let expectedBlob = try Self.fixture("mxfp4_expected.f32")

        let expected = expectedBlob.withUnsafeBytes { raw in
            (0..<(raw.count / 4)).map { raw.loadUnaligned(fromByteOffset: $0 * 4, as: Float.self) }
        }

        let decoded = try MXFP4.decode(packed: packed, scales: scales)

        #expect(decoded.count == expected.count)
        #expect(decoded.count == scales.count * MXFP4.blockSize)

        var worstRelative = 0.0
        var mismatches = 0
        for i in 0..<min(decoded.count, expected.count) {
            // `-0.0 == 0.0` holds, which is the intended behaviour: the E2M1 table
            // distinguishes +0 from -0 but their numeric value is identical.
            if decoded[i] != expected[i] {
                mismatches += 1
                let denom = max(abs(Double(expected[i])), Double.leastNormalMagnitude)
                worstRelative = max(worstRelative, abs(Double(decoded[i] - expected[i])) / denom)
            }
        }
        #expect(mismatches == 0, "\(mismatches) diverging values, worst relative deviation \(worstRelative)")
        #expect(worstRelative < 1e-6)
    }

    @Test("The low nibble carries the even index")
    func nibbleOrder() throws {
        // 0x71 : nibble bas = 1 -> +0.5, nibble haut = 7 -> +6.0.
        // Scale 127 -> factor 1.
        var packed = Data(repeating: 0, count: MXFP4.packedBytesPerBlock)
        packed[0] = 0x71
        let scales = Data([127])

        let out = try MXFP4.decode(packed: packed, scales: scales)
        #expect(out[0] == 0.5)
        #expect(out[1] == 6.0)
        // Reversing the order would give exactly the opposite: this test would fail.
        #expect(out[0] != 6.0)
    }

    @Test("The whole E2M1 table is reachable")
    func fullTable() throws {
        // Bytes 0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE: the 16 nibbles in order.
        var packed = Data(repeating: 0, count: MXFP4.packedBytesPerBlock)
        for i in 0..<8 { packed[i] = UInt8(i * 2) | UInt8((i * 2 + 1) << 4) }
        let out = try MXFP4.decode(packed: packed, scales: Data([127]))

        let expected: [Float] = [
            0, 0.5, 1, 1.5, 2, 3, 4, 6,
            -0, -0.5, -1, -1.5, -2, -3, -4, -6,
        ]
        for i in 0..<16 { #expect(out[i] == expected[i], "index \(i)") }
    }

    @Test("The E8M0 scale is an exact power of two")
    func scaleIsPowerOfTwo() throws {
        var packed = Data(repeating: 0, count: MXFP4.packedBytesPerBlock)
        packed[0] = 0x02  // nibble bas = 2 -> +1.0

        for exponent in [-40, -8, -1, 0, 1, 8, 40] {
            let scaleByte = UInt8(127 + exponent)
            let out = try MXFP4.decode(packed: packed, scales: Data([scaleByte]))
            #expect(out[0] == exp2(Float(exponent)), "exposant \(exponent)")
        }
    }

    @Test("Inconsistent sizes are rejected")
    func rejectsSizeMismatch() {
        let packed = Data(repeating: 0, count: MXFP4.packedBytesPerBlock)
        // Two scales for a single block of values.
        #expect(throws: MXFP4.DecodeError.self) {
            _ = try packed.withUnsafeBytes { p in
                try Data([0, 0]).withUnsafeBytes { s in
                    var out = [Float](repeating: 0, count: 64)
                    try out.withUnsafeMutableBufferPointer { o in
                        try MXFP4.decode(packed: p, scales: s, blockCount: 2, into: o)
                    }
                }
            }
        }
    }

    @Test("The size calculation matches the real safetensors headers")
    func byteSizeMatchesRealCheckpoint() {
        // gpt-oss-20b, couche 0 : gate_up_proj_blocks [32, 5760, 90, 16]
        //                         gate_up_proj_scales [32, 5760, 90]
        // i.e., for one expert, 5760 rows of 2880 columns.
        let (blocks, scales) = MXFP4.byteSize(rows: 5760, cols: 2880)
        #expect(blocks == 8_294_400)
        #expect(scales == 518_400)

        // down_proj_blocks [32, 2880, 90, 16]: 2880 rows of 2880 columns.
        let (dBlocks, dScales) = MXFP4.byteSize(rows: 2880, cols: 2880)
        #expect(dBlocks == 4_147_200)
        #expect(dScales == 259_200)
    }
}
