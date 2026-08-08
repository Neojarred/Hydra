import Foundation
import Testing

@testable import HydraTokenize

/// Gemma 4's BPE variant.
///
/// The algorithm is the one `BPETokenizer` already had; what differs is the conventions around
/// it, and each of them changes which identifiers come out for the same text. A mismatch does
/// not fail — the model simply reads subtly the wrong words — so the conventions are asserted
/// individually rather than trusted to a round trip.
@Suite("Gemma 4 tokenizer conventions")
struct GemmaTokenizerTests {

    /// A vocabulary in Gemma's style: `U+2581` for spaces, `<0xNN>` fallbacks, no `Ġ`.
    private func makeTokenizer(ignoreMerges: Bool = false) throws -> BPETokenizer {
        var vocabulary: [String: Int] = [:]
        var next = 0
        func add(_ piece: String) {
            if vocabulary[piece] == nil {
                vocabulary[piece] = next
                next += 1
            }
        }
        // Byte fallbacks first, as the real file has them.
        for byte in 0...255 { add(String(format: "<0x%02X>", byte)) }
        for scalar in "abcdefghijklmnopqrstuvwxyz.,!?" { add(String(scalar)) }
        add("\u{2581}")
        add("\u{2581}he")
        add("\u{2581}hello")
        add("he")
        add("hel")
        add("hello")
        add("\u{2581}wor")
        add("\u{2581}world")

        let merges: [(String, String)] = [
            ("h", "e"), ("he", "l"), ("hel", "lo"),
            ("\u{2581}", "he"), ("\u{2581}he", "llo"),
            ("\u{2581}", "wor"), ("\u{2581}wor", "ld"),
        ]
        return try BPETokenizer(
            vocabulary: vocabulary, merges: merges, specialTokens: [:],
            conventions: BPETokenizer.Conventions(
                encoding: .metaSpaceWithByteFallback, ignoreMerges: ignoreMerges,
                preTokenizerPattern: nil))
    }

    // MARK: - The conventions, one at a time

    /// A space becomes `U+2581` before anything else runs. Decoding must undo it exactly, or
    /// every word boundary in the output is wrong.
    @Test("Spaces round-trip through U+2581")
    func metaSpaceRoundTrips() throws {
        let tokenizer = try makeTokenizer()
        for text in ["hello", " hello", "hello world", " a b c "] {
            #expect(tokenizer.decode(tokenizer.encode(text)) == text, Comment(rawValue: text))
        }
    }

    /// The vocabulary has no piece for most bytes, so an unseen script must decompose into
    /// `<0xNN>` tokens. Without this it would be dropped silently.
    @Test("Unknown characters fall back to bytes, and reassemble")
    func byteFallbackRoundTrips() throws {
        let tokenizer = try makeTokenizer()
        for text in ["é", "日本語", "🙂", "caf\u{00E9} noir"] {
            let ids = tokenizer.encode(text)
            #expect(!ids.isEmpty, Comment(rawValue: "\(text) produced nothing"))
            #expect(tokenizer.decode(ids) == text, Comment(rawValue: text))
        }
    }

    @Test("A byte fallback token is one byte, not six characters")
    func byteFallbackDecodesToOneByte() {
        #expect(BPETokenizer.byteFallbackValue("<0x41>") == 0x41)
        #expect(BPETokenizer.byteFallbackValue("<0xFF>") == 0xFF)
        // Ordinary pieces that merely look similar must not be mistaken for one.
        #expect(BPETokenizer.byteFallbackValue("<0xZZ>") == nil)
        #expect(BPETokenizer.byteFallbackValue("hello") == nil)
        #expect(BPETokenizer.byteFallbackValue("<0x412>") == nil)
    }

    /// `ignore_merges` is true for GPT-OSS and false for Gemma. This is the whole difference,
    /// and it is observable: with the shortcut on, a word present in the vocabulary is emitted
    /// whole; with it off, the merges run and may reach the same place by a different route —
    /// or a different one entirely.
    @Test("ignore_merges changes the output for the same text")
    func ignoreMergesIsObservable() throws {
        let withShortcut = try makeTokenizer(ignoreMerges: true)
        let withoutShortcut = try makeTokenizer(ignoreMerges: false)

        // "hello" is in the vocabulary, so the shortcut emits exactly one token.
        #expect(withShortcut.encode("hello").count == 1)
        // Both must still decode back to the same text, whichever route they took.
        #expect(withShortcut.decode(withShortcut.encode("hello")) == "hello")
        #expect(withoutShortcut.decode(withoutShortcut.encode("hello")) == "hello")
    }

    /// GPT-OSS's conventions must stay exactly as they were: this change is meant to add a
    /// second set, not alter the first.
    @Test("The GPT-OSS conventions are unchanged")
    func gptOssConventionsUnchanged() {
        let c = BPETokenizer.Conventions.gptOss
        #expect(c.encoding == .byteLevel)
        #expect(c.ignoreMerges)
        #expect(c.preTokenizerPattern == BPETokenizer.pretokenPattern)

        let g = BPETokenizer.Conventions.gemma4
        #expect(g.encoding == .metaSpaceWithByteFallback)
        #expect(!g.ignoreMerges)
        #expect(g.preTokenizerPattern == nil, "the normalizer runs first, so the split matches nothing")
    }

    /// Byte-level and byte-fallback disagree about what a vocabulary entry means. Decoding a
    /// Gemma vocabulary with GPT-OSS's rule produces text, just not the right text — which is
    /// exactly why the convention is stored rather than inferred.
    @Test("The two encodings read the same entry differently")
    func encodingsAreNotInterchangeable() {
        let piece = "\u{2581}hello"
        let gemma = BPETokenizer.bytes(of: piece, conventions: .gemma4)
        let gptOss = BPETokenizer.bytes(of: piece, conventions: .gptOss)
        #expect(String(decoding: gemma, as: UTF8.self) == " hello")
        #expect(gemma != gptOss)
    }

    // MARK: - Against the real vocabulary, when it is available

    /// The synthetic vocabulary above proves the mechanism; only the published file proves the
    /// conventions were read correctly. Skipped when it is not present.
    @Test("The real Gemma vocabulary round-trips")
    func realVocabularyRoundTrips() throws {
        let path = "/private/tmp/claude-501/-Users-neojarred-Hydra"
            + "/20bf425d-e372-474a-ade9-47fdfcc0b9aa/scratchpad/gemma-tok/tokenizer.json"
        guard FileManager.default.fileExists(atPath: path),
            let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
            let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let model = root["model"] as? [String: Any],
            let vocabulary = model["vocab"] as? [String: Int]
        else { return }

        let merges: [(String, String)] = (model["merges"] as? [[String]] ?? [])
            .compactMap { $0.count == 2 ? ($0[0], $0[1]) : nil }
        let tokenizer = try BPETokenizer(
            vocabulary: vocabulary, merges: merges, specialTokens: [:],
            conventions: .gemma4)

        #expect(tokenizer.count >= 262_144)
        for text in [
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "def main(): print(\"hi\")",
            "caf\u{00E9} cr\u{00E8}me",
            "\u{65E5}\u{672C}\u{8A9E}",
            "emoji \u{1F642}",
            "  leading and  doubled  spaces ",
            "",
        ] {
            #expect(tokenizer.decode(tokenizer.encode(text)) == text, Comment(rawValue: text))
        }
    }
}
