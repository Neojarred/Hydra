import Foundation

/// Tokeniseur BPE byte-level, compatible `o200k_harmony`.
///
/// Three peculiarities of GPT-OSS's file shape the implementation:
///
/// - **`ignore_merges = true`**: if a pre-split piece appears verbatim in the vocabulary it
///   is emitted directly, **without applying the merges**. Skipping this rule gives a
///   different split on many common words.
/// - **regular-expression pre-splitting** before any merging: merges never cross a piece
///   boundary.
/// - **special tokens** (`<|start|>`, `<|message|>`, `<|channel|>`…) recognized before BPE:
///   they are what carries the Harmony format, and they must never be merged with ordinary
///   text.
public final class BPETokenizer: @unchecked Sendable {

    /// o200k's pre-split pattern, taken verbatim from the tokenizer file.
    public static let pretokenPattern =
        #"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"#
        + #"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"#
        + #"|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"#

    /// Byte-level encoded string → identifier.
    private let vocabulary: [String: Int]
    /// Identifier → raw bytes, pre-decoded once and for all.
    private let pieces: [[UInt8]]
    /// Mergeable pair → rank. The lowest rank is applied first.
    private let mergeRanks: [Pair: Int]
    private let regex: NSRegularExpression

    public let specialTokens: [String: Int]
    private let specialByID: [Int: String]
    private let specialRegex: NSRegularExpression?

    public var count: Int { pieces.count }

    struct Pair: Hashable {
        let left: String
        let right: String
    }

    public enum TokenizerError: Error, CustomStringConvertible {
        case malformedFile(String)
        case unknownToken(String)

        public var description: String {
            switch self {
            case .malformedFile(let detail): return "unreadable tokenizer file: \(detail)"
            case .unknownToken(let token): return "jeton inconnu : \(token)"
            }
        }
    }

    public init(
        vocabulary: [String: Int], merges: [(String, String)], specialTokens: [String: Int]
    ) throws {
        self.vocabulary = vocabulary
        self.specialTokens = specialTokens
        self.specialByID = Dictionary(
            uniqueKeysWithValues: specialTokens.map { ($0.value, $0.key) })

        var ranks: [Pair: Int] = [:]
        ranks.reserveCapacity(merges.count)
        for (rank, merge) in merges.enumerated() {
            ranks[Pair(left: merge.0, right: merge.1)] = rank
        }
        self.mergeRanks = ranks

        // The reverse table, pre-decoded: decoding is on generation's hot path and must not
        // redo the byte-level conversion for every token.
        let highest = max(
            vocabulary.values.max() ?? 0, specialTokens.values.max() ?? 0)
        var table = [[UInt8]](repeating: [], count: highest + 1)
        for (piece, id) in vocabulary { table[id] = ByteLevel.decode(piece) }
        for (piece, id) in specialTokens { table[id] = Array(piece.utf8) }
        self.pieces = table

        self.regex = try NSRegularExpression(pattern: Self.pretokenPattern)

        if specialTokens.isEmpty {
            self.specialRegex = nil
        } else {
            let alternatives = specialTokens.keys
                .sorted { $0.count > $1.count }  // le plus long gagne
                .map { NSRegularExpression.escapedPattern(for: $0) }
                .joined(separator: "|")
            self.specialRegex = try NSRegularExpression(pattern: alternatives)
        }
    }

    // MARK: - Encoding

    /// Encodes text.
    ///
    /// - Parameter allowSpecial: if true, sequences like `<|start|>` present in the text are
    ///   recognized as special tokens. Leave it **false** for user-supplied text: otherwise a
    ///   message containing `<|end|>` could cut the conversation short or usurp a role.
    ///
    public func encode(_ text: String, allowSpecial: Bool = false) -> [Int] {
        guard allowSpecial, let specialRegex else {
            return encodeOrdinary(text)
        }

        var out: [Int] = []
        let full = text as NSString
        var cursor = 0
        specialRegex.enumerateMatches(
            in: text, range: NSRange(location: 0, length: full.length)
        ) { match, _, _ in
            guard let match else { return }
            if match.range.location > cursor {
                let before = full.substring(
                    with: NSRange(location: cursor, length: match.range.location - cursor))
                out.append(contentsOf: encodeOrdinary(before))
            }
            let token = full.substring(with: match.range)
            if let id = specialTokens[token] { out.append(id) }
            cursor = match.range.location + match.range.length
        }
        if cursor < full.length {
            out.append(contentsOf: encodeOrdinary(
                full.substring(with: NSRange(location: cursor, length: full.length - cursor))))
        }
        return out
    }

    private func encodeOrdinary(_ text: String) -> [Int] {
        guard !text.isEmpty else { return [] }
        var out: [Int] = []
        let full = text as NSString

        regex.enumerateMatches(
            in: text, range: NSRange(location: 0, length: full.length)
        ) { match, _, _ in
            guard let match, match.range.length > 0 else { return }
            let piece = full.substring(with: match.range)
            let encoded = ByteLevel.encode(Array(piece.utf8))
            out.append(contentsOf: applyMerges(encoded))
        }
        return out
    }

    /// Applies the BPE merges to a piece already converted to byte-level.
    private func applyMerges(_ piece: String) -> [Int] {
        // `ignore_merges`: a piece present verbatim in the vocabulary is emitted directly.
        // Without this rule, "hello" would be split even though it exists.
        if let id = vocabulary[piece] { return [id] }

        var symbols = piece.map { String($0) }
        guard symbols.count > 1 else {
            return vocabulary[piece].map { [$0] } ?? []
        }

        while symbols.count > 1 {
            var bestRank = Int.max
            var bestIndex = -1
            for index in 0..<(symbols.count - 1) {
                let pair = Pair(left: symbols[index], right: symbols[index + 1])
                if let rank = mergeRanks[pair], rank < bestRank {
                    bestRank = rank
                    bestIndex = index
                }
            }
            guard bestIndex >= 0 else { break }
            symbols[bestIndex] += symbols[bestIndex + 1]
            symbols.remove(at: bestIndex + 1)
        }

        // A symbol missing from the vocabulary should not exist — byte-level guarantees every
        // byte has a representation. We skip it rather than fail.
        return symbols.compactMap { vocabulary[$0] }
    }

    // MARK: - Decoding

    /// Reconstructs the text of a sequence of identifiers.
    ///
    /// Decoding goes through **bytes** before UTF-8: a multi-byte character may be spread over
    /// several tokens, and decoding token by token would produce replacement characters in the
    /// middle of accented words or emoji.
    public func decode(_ ids: [Int], skipSpecial: Bool = false) -> String {
        var bytes: [UInt8] = []
        bytes.reserveCapacity(ids.count * 4)
        for id in ids {
            guard id >= 0, id < pieces.count else { continue }
            if skipSpecial, specialByID[id] != nil { continue }
            bytes.append(contentsOf: pieces[id])
        }
        return String(decoding: bytes, as: UTF8.self)
    }

    /// A token's raw bytes, for safe incremental decoding.
    public func bytes(for id: Int) -> [UInt8] {
        guard id >= 0, id < pieces.count else { return [] }
        return pieces[id]
    }

    public func isSpecial(_ id: Int) -> Bool { specialByID[id] != nil }
    public func name(of id: Int) -> String? { specialByID[id] }
    public func id(of token: String) -> Int? { specialTokens[token] ?? vocabulary[token] }
}
