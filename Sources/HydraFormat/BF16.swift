import Foundation

/// Conversion bfloat16, le format de **tous** les poids non quantifiés de GPT-OSS :
/// attention, routeurs, normes, sinks, biais d'experts, embedding et tête LM.
///
/// **BF16 n'est pas Float16.** `Float16` est l'IEEE binary16 (1 bit de signe, 5 d'exposant,
/// 10 de mantisse) ; bfloat16 est la moitié haute d'un `Float32` (1, 8, 7). Les deux font
/// deux octets, ce qui rend la confusion facile — et silencieuse : interpréter l'un pour
/// l'autre ne produit ni erreur ni NaN, seulement des valeurs plausibles et fausses.
/// 1,0 en BF16 se lit 1,875 en Float16.
///
/// Ce piège a été trouvé par un test sur le noyau Metal, pas par relecture. D'où sa
/// centralisation ici : aucun autre endroit du projet ne doit réinterpréter deux octets
/// à la main.
public enum BF16 {

    /// Conversion exacte et sans perte : les 16 bits deviennent la moitié haute d'un
    /// Float32. Aucun arrondi, aucun cas particulier — NaN et infinis se transposent.
    @inline(__always)
    public static func toFloat(_ bits: UInt16) -> Float {
        Float(bitPattern: UInt32(bits) << 16)
    }

    /// Sens inverse, arrondi au plus proche avec égalité vers le pair — le même que celui
    /// des bibliothèques de référence, pour que les allers-retours soient stables.
    @inline(__always)
    public static func fromFloat(_ value: Float) -> UInt16 {
        let bits = value.bitPattern
        if value.isNaN { return UInt16(bits >> 16) | 0x0040 }
        let rounding: UInt32 = 0x7FFF &+ ((bits >> 16) & 1)
        return UInt16(truncatingIfNeeded: (bits &+ rounding) >> 16)
    }

    /// Décode un tampon BF16 contigu. Les octets sont en petit-boutiste, comme dans les
    /// safetensors et dans nos fichiers `.hydra`.
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

    /// Encode une suite de `Float` en BF16 petit-boutiste.
    public static func encode(_ values: [Float]) -> Data {
        var out = Data(capacity: values.count * 2)
        for value in values {
            withUnsafeBytes(of: fromFloat(value).littleEndian) { out.append(contentsOf: $0) }
        }
        return out
    }
}
