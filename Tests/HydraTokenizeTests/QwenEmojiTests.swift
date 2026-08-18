import Foundation
import Testing

@testable import HydraTokenize

/// Characters that byte-level BPE splits across tokens, which is most of the interesting ones.
///
/// A four-byte emoji is routinely emitted as several tokens carrying two or three bytes each.
/// Decoding each token on its own turns every fragment into `U+FFFD`, and the bytes are then
/// unrecoverable: the user sees black diamonds where the model wrote a symbol. This is what the
/// user actually reported, and it affected Qwen alone because Qwen's parser was the one
/// accumulating a `String` where Harmony and Gemma accumulate bytes.
@Suite("Qwen streams multi-byte characters intact")
struct QwenEmojiTests {

    /// A tokenizer that splits every string into single bytes, which is the worst case and a
    /// real one: it is exactly what byte-level BPE does to a character it has no merge for.
    private func byteTokenizer() throws -> (BPETokenizer, (String) -> [Int]) {
        // A vocabulary of the 256 byte-level tokens and nothing else, so every character is
        // split as far as it can be. That is the worst case and also a real one: it is exactly
        // what byte-level BPE does to a character it has no merge for.
        var vocabulary: [String: Int] = [:]
        for value in 0...255 {
            vocabulary[String(ByteLevel.encodeTable[value])] = value
        }
        let tokenizer = try BPETokenizer(
            vocabulary: vocabulary, merges: [], specialTokens: [:],
            conventions: BPETokenizer.Conventions(
                encoding: .byteLevel, ignoreMerges: false, preTokenizerPattern: nil))
        return (tokenizer, { text in Array(text.utf8).map { Int($0) } })
    }

    private func stream(_ text: String) throws -> String {
        let (tokenizer, encode) = try byteTokenizer()
        let parser = QwenParser(tokenizer: tokenizer, inThought: false)
        var out = ""
        for token in encode(text) {
            for event in parser.consume(token) {
                if case .answer(let fragment) = event { out += fragment }
            }
        }
        // Whatever is still held back, released as the end of a turn would.
        for event in parser.consume(encode("\n").first ?? 10) {
            if case .answer(let fragment) = event { out += fragment }
        }
        return out
    }

    @Test("An emoji split into single bytes survives the stream", arguments: [
        "check ✅ done",
        "picture 🖼️ here",
        "a 🇫🇷 flag",
        "family 👨‍👩‍👧‍👦 group",
        "maths ∑ and accents éàü",
        "🌍🌎🌏",
    ])
    func multiByteSurvives(text: String) throws {
        let streamed = try stream(text)
        #expect(
            !streamed.contains("\u{FFFD}"),
            "\(text) came back with replacement characters: \(streamed)")
        #expect(streamed.hasPrefix(text) || text.hasPrefix(streamed.trimmingCharacters(in: .newlines)),
                "\(text) came back as \(streamed)")
    }

    /// The failure this replaces, stated directly: decoding a fragment alone destroys it.
    @Test("Decoding a token at a time is what corrupted them")
    func perTokenDecodingCorrupts() throws {
        let (tokenizer, encode) = try byteTokenizer()
        var naive = ""
        for token in encode("✅") { naive += tokenizer.decode([token]) }
        #expect(
            naive.contains("\u{FFFD}"),
            "the byte-at-a-time tokenizer no longer splits, so this suite proves nothing")
        #expect(!(try stream("✅")).contains("\u{FFFD}"))
    }
}
