import Foundation
import Testing

@testable import HydraFormat

/// BF16 est le format de tous les poids non quantifiés de GPT-OSS. Le confondre avec
/// `Float16` ne lève aucune erreur et produit des valeurs plausibles mais fausses — c'est
/// exactement ce qui est arrivé dans la première version du noyau Metal.
struct BF16Tests {

    @Test("La conversion vers Float est exacte")
    func exactValues() {
        // 0x3F80 = 1.0 en BF16 (moitié haute du Float32 0x3F800000).
        #expect(BF16.toFloat(0x3F80) == 1.0)
        #expect(BF16.toFloat(0x0000) == 0.0)
        #expect(BF16.toFloat(0xBF80) == -1.0)
        #expect(BF16.toFloat(0x4000) == 2.0)
        #expect(BF16.toFloat(0x3F00) == 0.5)
        #expect(BF16.toFloat(0x4049) == 3.140625)  // approximation de π en BF16
    }

    /// Le test qui documente le piège : les mêmes octets, deux interprétations.
    @Test("BF16 et Float16 ne coïncident pas")
    func bf16IsNotFloat16() {
        let bits: UInt16 = 0x3F80
        #expect(BF16.toFloat(bits) == 1.0)
        #expect(Float(Float16(bitPattern: bits)) == 1.875)
        // 0,875 d'écart sur une valeur de 1,0 : assez pour tout casser, trop peu
        // pour lever une alerte.
        #expect(abs(Float(Float16(bitPattern: bits)) - BF16.toFloat(bits)) == 0.875)
    }

    @Test("L'aller-retour est exact sur les valeurs représentables")
    func roundTripIsExactWhenRepresentable() {
        // BF16 n'a que 8 bits de mantisse : seules les valeurs dont la mantisse tient
        // sur 7 bits explicites reviennent à l'identique.
        for value: Float in [0, 1, -1, 0.5, 2, 256, -0.125, 3.140625, 0.00390625] {
            let encoded = BF16.fromFloat(value)
            #expect(BF16.toFloat(encoded) == value, "\(value)")
        }
    }

    /// Ce que coûte réellement le BF16, et pourquoi c'est acceptable ici : 8 bits de
    /// mantisse donnent une précision relative de 2⁻⁸, soit 0,39 %. C'est la précision
    /// avec laquelle GPT-OSS stocke son attention et sa tête LM — pas une approximation
    /// que nous introduisons.
    @Test("La perte de précision reste bornée à 2⁻⁸ en relatif")
    func precisionLossIsBounded() {
        let bound = Double(exp2(-8.0))
        for value: Float in [1e-8, 1e20, 0.1, -12345.678, 6.02e23, 1.0 / 3.0] {
            let restored = BF16.toFloat(BF16.fromFloat(value))
            let relative = abs(Double(restored) - Double(value)) / Double(abs(value))
            #expect(relative <= bound, "\(value) : écart relatif \(relative)")
        }
    }

    @Test("Les valeurs spéciales se transposent telles quelles")
    func specialValues() {
        #expect(BF16.toFloat(0x7F80) == .infinity)
        #expect(BF16.toFloat(0xFF80) == -.infinity)
        #expect(BF16.toFloat(0x7FC0).isNaN)
        #expect(BF16.fromFloat(.infinity) == 0x7F80)
        #expect(BF16.toFloat(BF16.fromFloat(.nan)).isNaN)
    }

    @Test("Le décodage d'un tampon respecte le petit-boutisme")
    func decodesLittleEndianBuffer() {
        // 1.0 puis -2.0, en petit-boutiste comme dans les safetensors.
        let data = Data([0x80, 0x3F, 0x00, 0xC0])
        let values = BF16.decode(data)
        #expect(values == [1.0, -2.0])
    }

    @Test("L'arrondi va au plus proche pair")
    func roundsToNearestEven() {
        // Un Float32 pile entre deux BF16 doit tomber sur la mantisse paire.
        let midpoint = Float(bitPattern: 0x3F80_8000)  // entre 1.0 et 1.0078125
        let encoded = BF16.fromFloat(midpoint)
        #expect(encoded == 0x3F80, "arrondi obtenu : 0x\(String(encoded, radix: 16))")
    }
}
