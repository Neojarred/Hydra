import Foundation
import HydraCore
import Testing

@testable import HydraTokenize

/// Deux niveaux de validation.
///
/// 1. A **synthetic** tokenizer, tiny and fully controlled, checks the algorithm: merge
///    order, the `ignore_merges` rule, special tokens,
///    conversion byte-level.
/// 2. The **real** `o200k_harmony`, if installed, checks that the implementation agrees
///    with the real vocabulary: known entries, round trips over varied text, full byte
///    coverage.
struct BPETokenizerTests {

    // MARK: - Table byte-level

    @Test("The byte-level table covers all 256 bytes, without collision")
    func byteLevelIsBijective() {
        #expect(ByteLevel.encodeTable.count == 256)
        #expect(Set(ByteLevel.encodeTable).count == 256, "two bytes share a character")
        for byte in 0..<256 {
            #expect(ByteLevel.decodeTable[ByteLevel.encodeTable[byte]] == UInt8(byte))
        }
    }

    @Test("Space becomes Ġ, as in the published vocabulary")
    func spaceMapsToKnownCharacter() {
        // This is what explains the "Ġ" in the vocabulary: without that conversion, a
        // vocabulary read from disk looks unreadable and matches nothing.
        #expect(ByteLevel.encode([0x20]) == "Ġ")
        #expect(ByteLevel.encode([0x0A]) == "Ċ")
        #expect(ByteLevel.encode(Array("hello".utf8)) == "hello")
        #expect(ByteLevel.encode(Array(" hello".utf8)) == "Ġhello")
    }

    @Test("Tout octet fait l'aller-retour, y compris hors UTF-8 valide")
    func byteLevelRoundTrip() {
        let bytes: [UInt8] = (0..<256).map(UInt8.init)
        #expect(ByteLevel.decode(ByteLevel.encode(bytes)) == bytes)
    }

    // MARK: - Synthetic tokenizer

    /// A toy vocabulary: the isolated bytes, a few merges, and one whole word reachable only
    /// through the `ignore_merges` rule.
    static func makeSynthetic() throws -> BPETokenizer {
        var vocabulary: [String: Int] = [:]
        var next = 0
        for byte in 0..<256 {
            vocabulary[String(ByteLevel.encodeTable[byte])] = next
            next += 1
        }
        // Merges, in rank order.
        let merges: [(String, String)] = [
            ("a", "b"),      // rang 0
            ("ab", "c"),     // rang 1
            ("Ġ", "a"),      // rang 2
        ]
        for (left, right) in merges {
            vocabulary[left + right] = next
            next += 1
        }
        // An entry present as such but **unreachable by merging**: only the ignore_merges
        // rule allows emitting it as one token.
        vocabulary["abcd"] = next
        next += 1

        return try BPETokenizer(
            vocabulary: vocabulary, merges: merges,
            specialTokens: ["<|start|>": 9000, "<|end|>": 9001],
            conventions: .gptOss)
    }

    @Test("Merges apply in increasing rank order")
    func mergesApplyByRank() throws {
        let tokenizer = try Self.makeSynthetic()
        // "abc": first a+b (rank 0), then ab+c (rank 1) — a single token.
        #expect(tokenizer.encode("abc").count == 1)
        #expect(tokenizer.decode(tokenizer.encode("abc")) == "abc")
    }

    @Test("ignore_merges: an entry present as such is emitted directly")
    func ignoreMergesTakesPrecedence() throws {
        let tokenizer = try Self.makeSynthetic()
        // "abcd" exists in the vocabulary. Without ignore_merges it would give "abc" + "d".
        let ids = tokenizer.encode("abcd")
        #expect(ids.count == 1, "got \(ids.count) tokens: ignore_merges is not being applied")
        #expect(tokenizer.decode(ids) == "abcd")
    }

    @Test("Special tokens are recognized only when allowed")
    func specialTokensAreOptIn() throws {
        let tokenizer = try Self.makeSynthetic()
        let text = "a<|end|>b"

        let withSpecial = tokenizer.encode(text, allowSpecial: true)
        #expect(withSpecial.contains(9001))

        // By default, a user message containing "<|end|>" must not be able to interrupt
        // the conversation nor impersonate a role.
        let withoutSpecial = tokenizer.encode(text, allowSpecial: false)
        #expect(!withoutSpecial.contains(9001))
        #expect(tokenizer.decode(withoutSpecial) == text)
    }

    @Test("Decoding goes through bytes, not through tokens")
    func decodeJoinsBytes() throws {
        let tokenizer = try Self.makeSynthetic()
        // A multi-byte character split across several tokens must recompose.
        for text in ["é", "→", "🙂", "café crème", "日本語"] {
            #expect(tokenizer.decode(tokenizer.encode(text)) == text, Comment(rawValue: text))
        }
    }

    // MARK: - The real tokenizer, if installed

    nonisolated(unsafe) static let installedTokenizer: BPETokenizer? = {
        guard let base = try? FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask,
            appropriateFor: nil, create: false)
        else { return nil }
        let compact = base.appending(
            path: "Hydra/Models/gpt-oss-20b.hydra/tokenizer/\(TokenizerFile.compactFileName)")
        guard FileManager.default.fileExists(atPath: compact.path) else { return nil }
        return try? TokenizerFile.loadCompact(at: compact, architecture: .gptOss)
    }()

    @Test("o200k: known vocabulary entries give a single token")
    func realTokenizerKnownPieces() throws {
        guard let tokenizer = Self.installedTokenizer else { return }
        // Identifiers read directly from the published tokenizer.json.
        let expected: [(String, Int)] = [
            ("hello", 24912),
            (" hello", 40617),
            (" the", 290),
            ("the", 3086),
            (",", 11),
            (" (", 350),
            ("\n\n", 279),
            (" Bonjour", 141691),
        ]
        for (text, id) in expected {
            let ids = tokenizer.encode(text)
            #expect(ids == [id], "« \(text) » → \(ids), attendu [\(id)]")
        }
    }

    @Test("o200k: exact round trip over varied text")
    func realTokenizerRoundTrip() throws {
        guard let tokenizer = Self.installedTokenizer else { return }
        let corpus = [
            "Hello, world!",
            "Hello, how are you doing today?",
            "Les élèves ont déjà terminé l'exercice n°42.",
            "func main() { print(\"hello\") }",
            "    indentation\n\ttabulation\n\n\ndouble breaks",
            "emoji 🙂🚀 and symbols ∑∫≈",
            "日本語のテキスト",
            "1234567890 3.14159 -1e-9",
            "<|start|>this is not a special token<|end|>",
            "",
        ]
        for text in corpus {
            let ids = tokenizer.encode(text)
            #expect(tokenizer.decode(ids) == text, "round trip broken on \"\(text)\"")
        }
    }

    @Test("o200k: the tokens cover exactly the input bytes")
    func realTokenizerCoversInput() throws {
        guard let tokenizer = Self.installedTokenizer else { return }
        let text = "The quick brown fox jumps over the lazy dog — 42 times."
        let ids = tokenizer.encode(text)

        var rebuilt: [UInt8] = []
        for id in ids { rebuilt.append(contentsOf: tokenizer.bytes(for: id)) }
        #expect(rebuilt == Array(text.utf8), "the segmentation loses or duplicates bytes")
        #expect(!ids.isEmpty)
    }

    @Test("o200k: the Harmony tokens are present and distinct")
    func realTokenizerHarmonySpecials() throws {
        guard let tokenizer = Self.installedTokenizer else { return }
        let required = [
            "<|start|>", "<|end|>", "<|message|>", "<|channel|>",
            "<|return|>", "<|call|>", "<|constrain|>", "<|endoftext|>",
        ]
        var seen = Set<Int>()
        for name in required {
            let id = try #require(tokenizer.specialTokens[name], "jeton \(name) absent")
            #expect(seen.insert(id).inserted, "two special tokens share the identifier \(id)")
            #expect(tokenizer.isSpecial(id))
        }
        #expect(tokenizer.specialTokens["<|start|>"] == 200006)
        #expect(tokenizer.specialTokens["<|message|>"] == 200008)
        #expect(tokenizer.specialTokens["<|return|>"] == 200002)
    }

    @Test("o200k: a round trip of the vocabulary itself stays stable")
    func realTokenizerVocabularyIsStable() throws {
        guard let tokenizer = Self.installedTokenizer else { return }
        // A sample of identifiers spread across the whole vocabulary: decoding then re-encoding
        // must return the same identifier, provided the piece constitutes a complete
        // pre-token.
        var checked = 0
        var stable = 0
        for id in stride(from: 100, to: 199_000, by: 4_999) {
            let text = tokenizer.decode([id])
            guard !text.isEmpty, tokenizer.decode(tokenizer.encode(text)) == text else { continue }
            checked += 1
            if tokenizer.encode(text) == [id] { stable += 1 }
        }
        #expect(checked > 20, "sample too small: \(checked)")
        // Not every piece is a complete pre-token; we require a large majority, not all of
        // them.
        #expect(Double(stable) / Double(checked) > 0.75,
                "seulement \(stable)/\(checked) identifiants stables")
    }
}
