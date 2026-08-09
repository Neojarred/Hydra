import Foundation
import HydraCore

/// Loading a `tokenizer.json` in the `tokenizers` library's format.
///
/// GPT-OSS's file weighs 27 MB: close to 200,000 vocabulary entries and 446,000 merges.
/// Walking it with `JSONSerialization` on every launch would cost several seconds; so we
/// convert it **once** into a compact binary format, stored in the `.hydra` installation
/// beside the weights.
///
/// It is the same logic as the weight repack: the original format is for exchange, not for
/// execution.
public enum TokenizerFile {

    public enum LoadError: Error, CustomStringConvertible {
        case unreadable(String)
        case unsupported(String)
        case corruptBinary(String)

        public var description: String {
            switch self {
            case .unreadable(let path): return "tokenizer illisible : \(path)"
            case .unsupported(let detail): return "unsupported tokenizer: \(detail)"
            case .corruptBinary(let detail): return "corrupt compiled tokenizer: \(detail)"
            }
        }
    }

    /// Lit un `tokenizer.json` d'origine.
    public static func parseJSON(at url: URL, architecture: ModelArchitecture) throws -> BPETokenizer {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw LoadError.unreadable(url.lastPathComponent)
        }
        guard let model = root["model"] as? [String: Any],
            (model["type"] as? String) == "BPE",
            let vocabulary = model["vocab"] as? [String: Int]
        else {
            throw LoadError.unsupported("only the BPE type is handled")
        }

        // Merges are either pairs or "left right" strings.
        var merges: [(String, String)] = []
        if let pairs = model["merges"] as? [[String]] {
            merges = pairs.compactMap { $0.count == 2 ? ($0[0], $0[1]) : nil }
        } else if let lines = model["merges"] as? [String] {
            merges = lines.compactMap { line in
                let parts = line.split(separator: " ", maxSplits: 1).map(String.init)
                return parts.count == 2 ? (parts[0], parts[1]) : nil
            }
        }

        var special: [String: Int] = [:]
        for entry in (root["added_tokens"] as? [[String: Any]]) ?? [] {
            guard let content = entry["content"] as? String, let id = entry["id"] as? Int
            else { continue }
            special[content] = id
        }

        return try BPETokenizer(
            vocabulary: vocabulary, merges: merges, specialTokens: special,
            conventions: .for(architecture))
    }

    // MARK: - Compact format

    /// `hydratok/1` — vocabulary and merges, with no JSON re-parsing.
    ///
    /// ```
    /// magic      8 B   "HYDRATOK"
    /// version    4 B
    /// nVocab     4 B   then, per entry: id (4 B), length (4 B), UTF-8 bytes
    /// nMerges    4 B   then, per merge: lengths (4+4 B) then both sides
    /// nSpecial   4 B   same layout as the vocabulary
    /// ```
    static let magic = "HYDRATOK"
    static let version: UInt32 = 1
    public static let compactFileName = "tokenizer.hydratok"

    public static func compile(jsonAt source: URL, to destination: URL) throws {
        let data = try Data(contentsOf: source, options: .mappedIfSafe)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let model = root["model"] as? [String: Any],
            let vocabulary = model["vocab"] as? [String: Int]
        else {
            throw LoadError.unreadable(source.lastPathComponent)
        }

        var merges: [(String, String)] = []
        if let pairs = model["merges"] as? [[String]] {
            merges = pairs.compactMap { $0.count == 2 ? ($0[0], $0[1]) : nil }
        } else if let lines = model["merges"] as? [String] {
            merges = lines.compactMap { line in
                let parts = line.split(separator: " ", maxSplits: 1).map(String.init)
                return parts.count == 2 ? (parts[0], parts[1]) : nil
            }
        }

        var special: [String: Int] = [:]
        for entry in (root["added_tokens"] as? [[String: Any]]) ?? [] {
            guard let content = entry["content"] as? String, let id = entry["id"] as? Int
            else { continue }
            special[content] = id
        }

        var out = Data()
        out.reserveCapacity(8 * 1024 * 1024)
        out.append(contentsOf: Array(magic.utf8))
        appendUInt32(&out, version)

        func appendEntries(_ entries: [String: Int]) {
            appendUInt32(&out, UInt32(entries.count))
            for (piece, id) in entries {
                appendUInt32(&out, UInt32(id))
                let bytes = Array(piece.utf8)
                appendUInt32(&out, UInt32(bytes.count))
                out.append(contentsOf: bytes)
            }
        }

        appendEntries(vocabulary)
        appendUInt32(&out, UInt32(merges.count))
        for (left, right) in merges {
            let l = Array(left.utf8), r = Array(right.utf8)
            appendUInt32(&out, UInt32(l.count))
            appendUInt32(&out, UInt32(r.count))
            out.append(contentsOf: l)
            out.append(contentsOf: r)
        }
        appendEntries(special)

        try out.write(to: destination, options: .atomic)
    }

    public static func loadCompact(at url: URL, architecture: ModelArchitecture) throws -> BPETokenizer {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        var cursor = 0

        func readUInt32() throws -> Int {
            guard cursor + 4 <= data.count else { throw LoadError.corruptBinary("premature end") }
            let value = data.withUnsafeBytes {
                UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: cursor, as: UInt32.self))
            }
            cursor += 4
            return Int(value)
        }
        func readString(_ length: Int) throws -> String {
            guard cursor + length <= data.count else {
                throw LoadError.corruptBinary("truncated string")
            }
            let text = String(decoding: data[(data.startIndex + cursor)..<(data.startIndex + cursor + length)], as: UTF8.self)
            cursor += length
            return text
        }

        guard data.count > 12, String(decoding: data.prefix(8), as: UTF8.self) == magic else {
            throw LoadError.corruptBinary("signature absente")
        }
        cursor = 8
        let fileVersion = try readUInt32()
        guard fileVersion == Int(version) else {
            throw LoadError.corruptBinary("version \(fileVersion) inconnue")
        }

        func readEntries() throws -> [String: Int] {
            let count = try readUInt32()
            var entries = [String: Int](minimumCapacity: count)
            for _ in 0..<count {
                let id = try readUInt32()
                let length = try readUInt32()
                entries[try readString(length)] = id
            }
            return entries
        }

        let vocabulary = try readEntries()
        let mergeCount = try readUInt32()
        var merges: [(String, String)] = []
        merges.reserveCapacity(mergeCount)
        for _ in 0..<mergeCount {
            let leftLength = try readUInt32()
            let rightLength = try readUInt32()
            merges.append((try readString(leftLength), try readString(rightLength)))
        }
        let special = try readEntries()

        return try BPETokenizer(
            vocabulary: vocabulary, merges: merges, specialTokens: special,
            conventions: .for(architecture))
    }

    private static func appendUInt32(_ data: inout Data, _ value: UInt32) {
        withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
    }
}
