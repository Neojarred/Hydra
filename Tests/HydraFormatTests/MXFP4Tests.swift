import Foundation
import Testing

@testable import HydraFormat

/// Jalon 1.2 — le décodeur MXFP4 doit concorder avec une implémentation de référence
/// indépendante. Le critère du plan de phases est une erreur relative < 1e-6 ; comme
/// toutes les valeurs en jeu sont exactement représentables en float32, on exige en
/// pratique l'**égalité bit à bit**, ce qui est strictement plus fort.
struct MXFP4Tests {

    private static func fixture(_ name: String) throws -> Data {
        let url = try #require(
            Bundle.module.url(forResource: "Fixtures/\(name)", withExtension: nil),
            "fixture « \(name) » introuvable — lancer tools/gen_mxfp4_fixture.py")
        return try Data(contentsOf: url)
    }

    @Test("Décodage conforme au vecteur de référence, bit à bit")
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
            // `-0.0 == 0.0` est vrai, ce qui est le comportement voulu : la table E2M1
            // distingue +0 et -0 mais leur valeur numérique est identique.
            if decoded[i] != expected[i] {
                mismatches += 1
                let denom = max(abs(Double(expected[i])), Double.leastNormalMagnitude)
                worstRelative = max(worstRelative, abs(Double(decoded[i] - expected[i])) / denom)
            }
        }
        #expect(mismatches == 0, "\(mismatches) valeurs divergentes, pire écart relatif \(worstRelative)")
        #expect(worstRelative < 1e-6)
    }

    @Test("Le nibble bas porte l'index pair")
    func nibbleOrder() throws {
        // 0x71 : nibble bas = 1 -> +0.5, nibble haut = 7 -> +6.0.
        // Échelle 127 -> facteur 1.
        var packed = Data(repeating: 0, count: MXFP4.packedBytesPerBlock)
        packed[0] = 0x71
        let scales = Data([127])

        let out = try MXFP4.decode(packed: packed, scales: scales)
        #expect(out[0] == 0.5)
        #expect(out[1] == 6.0)
        // Inverser l'ordre donnerait exactement l'inverse : ce test échouerait.
        #expect(out[0] != 6.0)
    }

    @Test("Toute la table E2M1 est atteignable")
    func fullTable() throws {
        // Octets 0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE : les 16 nibbles en ordre.
        var packed = Data(repeating: 0, count: MXFP4.packedBytesPerBlock)
        for i in 0..<8 { packed[i] = UInt8(i * 2) | UInt8((i * 2 + 1) << 4) }
        let out = try MXFP4.decode(packed: packed, scales: Data([127]))

        let expected: [Float] = [
            0, 0.5, 1, 1.5, 2, 3, 4, 6,
            -0, -0.5, -1, -1.5, -2, -3, -4, -6,
        ]
        for i in 0..<16 { #expect(out[i] == expected[i], "index \(i)") }
    }

    @Test("L'échelle E8M0 est une puissance de deux exacte")
    func scaleIsPowerOfTwo() throws {
        var packed = Data(repeating: 0, count: MXFP4.packedBytesPerBlock)
        packed[0] = 0x02  // nibble bas = 2 -> +1.0

        for exponent in [-40, -8, -1, 0, 1, 8, 40] {
            let scaleByte = UInt8(127 + exponent)
            let out = try MXFP4.decode(packed: packed, scales: Data([scaleByte]))
            #expect(out[0] == exp2(Float(exponent)), "exposant \(exponent)")
        }
    }

    @Test("Les tailles incohérentes sont rejetées")
    func rejectsSizeMismatch() {
        let packed = Data(repeating: 0, count: MXFP4.packedBytesPerBlock)
        // Deux échelles pour un seul bloc de valeurs.
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

    @Test("Le calcul de taille correspond aux en-têtes safetensors réels")
    func byteSizeMatchesRealCheckpoint() {
        // gpt-oss-20b, couche 0 : gate_up_proj_blocks [32, 5760, 90, 16]
        //                         gate_up_proj_scales [32, 5760, 90]
        // soit, pour un expert, 5760 lignes de 2880 colonnes.
        let (blocks, scales) = MXFP4.byteSize(rows: 5760, cols: 2880)
        #expect(blocks == 8_294_400)
        #expect(scales == 518_400)

        // down_proj_blocks [32, 2880, 90, 16] : 2880 lignes de 2880 colonnes.
        let (dBlocks, dScales) = MXFP4.byteSize(rows: 2880, cols: 2880)
        #expect(dBlocks == 4_147_200)
        #expect(dScales == 259_200)
    }
}
