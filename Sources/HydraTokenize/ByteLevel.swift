import Foundation

/// The byte ↔ character mapping of the "ByteLevel" pre-tokenizer, inherited from GPT-2.
///
/// A BPE tokenizer works on text, not on bytes. To be able to represent any byte sequence,
/// including one invalid in UTF-8, GPT-2 maps every byte to a **printable and unique**
/// Unicode character. Bytes already printable in Latin-1 stand for themselves; the others
/// are shifted into the U+0100 range and beyond.
///
///
/// This is what explains the vocabulary's `Ġ`: the space (0x20) is not printable in this
/// table's sense, so it becomes U+0120. A vocabulary read without this conversion looks
/// unreadable and matches nothing.
public enum ByteLevel {

    /// Byte → character.
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

    /// Character → byte.
    public static let decodeTable: [Character: UInt8] = {
        var table: [Character: UInt8] = [:]
        for byte in 0..<256 { table[encodeTable[byte]] = UInt8(byte) }
        return table
    }()

    /// Represents a byte sequence as the corresponding character string.
    public static func encode<S: Sequence>(_ bytes: S) -> String where S.Element == UInt8 {
        String(bytes.map { encodeTable[Int($0)] })
    }

    /// Reconstructs the bytes of a string produced by `encode`.
    /// A character missing from the table signals a corrupt vocabulary; we skip it rather than
    /// fail, so that one exotic token does not bring decoding down.
    public static func decode(_ text: String) -> [UInt8] {
        text.compactMap { decodeTable[$0] }
    }
}
