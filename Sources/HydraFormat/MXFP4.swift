import Foundation

/// Décodage du format OCP Microscaling MXFP4, tel qu'utilisé par les experts de GPT-OSS.
///
/// Layout vérifié sur les en-têtes safetensors réels et sur l'implémentation de référence
/// d'OpenAI (`gpt_oss/torch/weights.py`) — voir docs/01-DECISIONS.md, D-011 :
///
///   - un bloc couvre **32 valeurs consécutives de la dernière dimension** ;
///   - il est stocké sur **16 octets** : deux valeurs FP4 E2M1 par `UInt8` ;
///   - le **nibble bas porte l'index pair**, le nibble haut l'index impair ;
///   - un **octet d'échelle E8M0** par bloc, appliqué par `valeur * 2^(échelle - 127)`.
///
/// Soit 4,25 bits par poids. Se tromper sur l'ordre des nibbles produit un modèle qui
/// génère du texte plausible mais dégradé, sans lever d'erreur — d'où la validation
/// numérique systématique de ce chemin.
public enum MXFP4 {

    /// Nombre de valeurs couvertes par un bloc.
    public static let blockSize = 32

    /// Octets de valeurs packées par bloc (32 valeurs × 4 bits).
    public static let packedBytesPerBlock = 16

    /// Octets d'échelle par bloc.
    public static let scaleBytesPerBlock = 1

    /// Table E2M1 : 1 bit de signe, 2 bits d'exposant, 1 bit de mantisse.
    /// L'index est le nibble brut, de 0 à 15.
    public static let valueTable: [Float] = [
        +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ]

    /// Table pré-multipliée par la valeur d'échelle, pour éviter une multiplication par valeur.
    /// (16 entrées, réutilisée telle quelle pour chaque bloc.)
    @inline(__always)
    private static func scaleFactor(_ scaleByte: UInt8) -> Float {
        // Conforme à la référence : `exp = scale.int() - 127`, puis `ldexp`.
        // L'octet 0xFF encode un NaN dans la spécification OCP ; la référence produit ici
        // un facteur infini. On reproduit ce comportement plutôt que de diverger d'elle.
        exp2(Float(Int(scaleByte) - 127))
    }

    public enum DecodeError: Error, CustomStringConvertible {
        case packedSizeMismatch(expected: Int, got: Int)
        case scaleSizeMismatch(expected: Int, got: Int)
        case outputSizeMismatch(expected: Int, got: Int)

        public var description: String {
            switch self {
            case let .packedSizeMismatch(e, g):
                return "MXFP4 : \(g) octets packés fournis, \(e) attendus"
            case let .scaleSizeMismatch(e, g):
                return "MXFP4 : \(g) octets d'échelle fournis, \(e) attendus"
            case let .outputSizeMismatch(e, g):
                return "MXFP4 : tampon de sortie de \(g) valeurs, \(e) attendues"
            }
        }
    }

    /// Décode `blockCount` blocs consécutifs.
    ///
    /// - Parameters:
    ///   - packed: `blockCount * 16` octets de valeurs FP4 packées.
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

    /// Variante confortable sur `Data`. Alloue le tableau de sortie ; réservée aux tests
    /// et à l'outillage. Le chemin d'inférence utilise la version sur pointeurs, qui
    /// n'alloue pas.
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

    /// Octets occupés par une matrice MXFP4 de `rows` lignes et `cols` colonnes,
    /// blocs découpés le long des colonnes.
    public static func byteSize(rows: Int, cols: Int) -> (blocks: Int, scales: Int) {
        precondition(cols % blockSize == 0, "cols doit être un multiple de \(blockSize)")
        let blocksPerRow = cols / blockSize
        return (rows * blocksPerRow * packedBytesPerBlock, rows * blocksPerRow)
    }
}
