import Foundation
import HydraCore

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
    private let regex: NSRegularExpression?
    public let conventions: Conventions

    public let specialTokens: [String: Int]
    private let specialByID: [Int: String]
    private let specialRegex: NSRegularExpression?

    public var count: Int { pieces.count }

    struct Pair: Hashable {
        let left: String
        let right: String
    }

    /// The conventions surrounding the merges, which differ per model while the algorithm does
    /// not (D-023).
    ///
    /// These are read from `tokenizer.json`, never assumed. Every one of them changes which
    /// identifiers come out for the same text, and a mismatch produces a model that reads
    /// subtly the wrong words — no error, just worse answers.
    public struct Conventions: Sendable, Equatable {

        /// How text becomes the string form the vocabulary is keyed by.
        public enum Encoding: Sendable, Equatable {
            /// GPT-2 byte-level: every byte maps to a printable Unicode character, so any byte
            /// sequence is representable and the vocabulary is full of `Ġ`.
            case byteLevel
            /// SentencePiece-style: a space becomes `U+2581`, and any byte with no piece of
            /// its own falls back to a `<0xNN>` token. Gemma 4's arrangement.
            case metaSpaceWithByteFallback
        }

        public var encoding: Encoding
        /// Whether a piece already present in the vocabulary skips the merges entirely.
        /// **True for GPT-OSS, false for Gemma** — the same text tokenizes differently.
        public var ignoreMerges: Bool
        /// The pattern used to pre-split, or `nil` when the model does not pre-split.
        public var preTokenizerPattern: String?

        public static let gptOss = Conventions(
            encoding: .byteLevel, ignoreMerges: true, preTokenizerPattern: BPETokenizer.pretokenPattern)

        /// Gemma 4: the normalizer replaces spaces before anything else runs, so its declared
        /// `Split(" ")` pre-tokenizer matches nothing and the merges see the whole string.
        public static let gemma4 = Conventions(
            encoding: .metaSpaceWithByteFallback, ignoreMerges: false, preTokenizerPattern: nil)

        /// The conventions a model's vocabulary is keyed by.
        ///
        /// **There is deliberately no default.** A convenience initializer used to supply
        /// `.gptOss`, so loading Gemma's vocabulary produced a tokenizer that ran GPT-2
        /// byte-level encoding over it: every space became `Ġ` instead of `▁`, `▁capital`
        /// never formed, and the model was fed a sequence no Gemma has ever seen. It answered
        /// — fluently, with punctuation — which is exactly the failure D-023 says must be made
        /// impossible to reach by omission rather than caught by inspection.
        public static func `for`(_ architecture: ModelArchitecture) -> Conventions {
            switch architecture {
            case .gptOss: return .gptOss
            case .gemma4: return .gemma4
            }
        }

        public init(encoding: Encoding, ignoreMerges: Bool, preTokenizerPattern: String?) {
            self.encoding = encoding
            self.ignoreMerges = ignoreMerges
            self.preTokenizerPattern = preTokenizerPattern
        }
    }

    /// `U+2581 LOWER ONE EIGHTH BLOCK`, SentencePiece's stand-in for a space.
    public static let metaSpace: Character = "\u{2581}"

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
        vocabulary: [String: Int], merges: [(String, String)], specialTokens: [String: Int],
        conventions: Conventions
    ) throws {
        self.conventions = conventions
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
        for (piece, id) in vocabulary {
            table[id] = Self.bytes(of: piece, conventions: conventions)
        }
        for (piece, id) in specialTokens { table[id] = Array(piece.utf8) }
        self.pieces = table

        self.regex = try conventions.preTokenizerPattern.map {
            try NSRegularExpression(pattern: $0)
        }

        if specialTokens.isEmpty {
            self.specialRegex = nil
        } else {
            let alternatives = specialTokens.keys
                .sorted { $0.count > $1.count }  // longest wins
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

        // A model that does not pre-split hands the whole run to the merges. Gemma is that
        // case: its normalizer replaces spaces before its declared `Split(" ")` ever runs, so
        // the pre-tokenizer matches nothing.
        guard let regex else { return applyMerges(normalized(text)) }

        var out: [Int] = []
        let full = text as NSString
        regex.enumerateMatches(
            in: text, range: NSRange(location: 0, length: full.length)
        ) { match, _, _ in
            guard let match, match.range.length > 0 else { return }
            let piece = full.substring(with: match.range)
            out.append(contentsOf: applyMerges(normalized(piece)))
        }
        return out
    }

    /// Puts text into the form the vocabulary is keyed by.
    private func normalized(_ text: String) -> String {
        switch conventions.encoding {
        case .byteLevel:
            return ByteLevel.encode(Array(text.utf8))
        case .metaSpaceWithByteFallback:
            return text.replacingOccurrences(of: " ", with: String(Self.metaSpace))
        }
    }

    /// The starting symbols of the merge loop.
    ///
    /// Byte-level guarantees every character has a piece. Byte fallback does not: a character
    /// absent from the vocabulary is replaced by one `<0xNN>` token per UTF-8 byte, which is
    /// what makes an unseen script encodable at all rather than silently dropped.
    private func initialSymbols(_ piece: String) -> [String] {
        guard conventions.encoding == .metaSpaceWithByteFallback else {
            return piece.map { String($0) }
        }
        var out: [String] = []
        for character in piece {
            let single = String(character)
            if vocabulary[single] != nil {
                out.append(single)
            } else {
                for byte in single.utf8 {
                    out.append(String(format: "<0x%02X>", byte))
                }
            }
        }
        return out
    }

    /// Applies the BPE merges to a piece already converted to byte-level.
    private func applyMerges(_ piece: String) -> [Int] {
        // `ignore_merges`: a piece present verbatim in the vocabulary is emitted directly.
        // True for GPT-OSS — without it, "hello" would be split even though it exists — and
        // **false for Gemma**, where taking the shortcut gives different identifiers for the
        // same text.
        if conventions.ignoreMerges, let id = vocabulary[piece] { return [id] }

        var symbols = initialSymbols(piece)
        guard symbols.count > 1 else {
            return symbols.first.flatMap { vocabulary[$0] }.map { [$0] } ?? []
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

    /// The bytes one vocabulary entry stands for.
    ///
    /// This *is* the decoder each model declares. GPT-OSS reverses the byte-level mapping.
    /// Gemma applies `Replace(U+2581 -> " ")` then `ByteFallback`: a `<0xNN>` entry is the raw
    /// byte NN, which is how the vocabulary represents bytes it has no piece for.
    static func bytes(of piece: String, conventions: Conventions) -> [UInt8] {
        switch conventions.encoding {
        case .byteLevel:
            return ByteLevel.decode(piece)
        case .metaSpaceWithByteFallback:
            if let byte = byteFallbackValue(piece) { return [byte] }
            return Array(piece.replacingOccurrences(
                of: String(metaSpace), with: " ").utf8)
        }
    }

    /// The byte a `<0xNN>` entry stands for, or `nil` for an ordinary piece.
    static func byteFallbackValue(_ piece: String) -> UInt8? {
        guard piece.count == 6, piece.hasPrefix("<0x"), piece.hasSuffix(">") else { return nil }
        return UInt8(piece.dropFirst(3).dropLast(), radix: 16)
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
