import Foundation

/// Correspondance octet ↔ caractère du pré-tokeniseur « ByteLevel », héritée de GPT-2.
///
/// Un tokeniseur BPE travaille sur du texte, pas sur des octets. Pour pouvoir représenter
/// n'importe quelle séquence d'octets — y compris invalide en UTF-8 — GPT-2 associe
/// chaque octet à un caractère Unicode **imprimable et unique**. Les octets déjà
/// imprimables en Latin-1 se représentent eux-mêmes ; les autres sont décalés dans la
/// plage U+0100 et suivantes.
///
/// C'est ce qui explique les `Ġ` du vocabulaire : l'espace (0x20) n'est pas imprimable au
/// sens de cette table, il devient donc U+0120. Un vocabulaire lu sans cette conversion
/// paraît illisible et ne concorde avec rien.
public enum ByteLevel {

    /// Octet → caractère.
    public static let encodeTable: [Character] = {
        var bytes: [Int] = []
        bytes.append(contentsOf: Int(Character("!").asciiValue!)...Int(Character("~").asciiValue!))
        bytes.append(contentsOf: 0xA1...0xAC)
        bytes.append(contentsOf: 0xAE...0xFF)

        var codes = bytes
        var extra = 0
        for byte in 0..<256 where !bytes.contains(byte) {
            bytes.append(byte)
            codes.append(256 + extra)
            extra += 1
        }

        var table = [Character](repeating: " ", count: 256)
        for (byte, code) in zip(bytes, codes) {
            table[byte] = Character(UnicodeScalar(code)!)
        }
        return table
    }()

    /// Caractère → octet.
    public static let decodeTable: [Character: UInt8] = {
        var table: [Character: UInt8] = [:]
        for byte in 0..<256 { table[encodeTable[byte]] = UInt8(byte) }
        return table
    }()

    /// Représente une suite d'octets par la chaîne de caractères correspondante.
    public static func encode<S: Sequence>(_ bytes: S) -> String where S.Element == UInt8 {
        String(bytes.map { encodeTable[Int($0)] })
    }

    /// Reconstitue les octets d'une chaîne produite par `encode`.
    /// Un caractère absent de la table signale un vocabulaire corrompu ; on l'ignore
    /// plutôt que d'échouer, pour qu'un token exotique ne fasse pas tomber le décodage.
    public static func decode(_ text: String) -> [UInt8] {
        text.compactMap { decodeTable[$0] }
    }
}
