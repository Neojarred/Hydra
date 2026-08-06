import Foundation
import Testing

@testable import HydraTokenize

/// Harmony is not an ordinary chat template: the model splits its output across channels,
/// and the interface has to tell them apart. A mistake here breaks nothing visibly — it
/// just makes the reasoning appear inside the answer, or loses the answer.
struct HarmonyTests {

    static func makeTokenizer() throws -> BPETokenizer {
        var vocabulary: [String: Int] = [:]
        for byte in 0..<256 { vocabulary[String(ByteLevel.encodeTable[byte])] = byte }
        var next = 256
        for piece in ["analysis", "final", "commentary", "assistant", "Bonjour", "Ġmonde"] {
            vocabulary[piece] = next
            next += 1
        }
        return try BPETokenizer(
            vocabulary: vocabulary, merges: [],
            specialTokens: [
                "<|start|>": 200006, "<|end|>": 200007, "<|message|>": 200008,
                "<|channel|>": 200005, "<|return|>": 200002, "<|call|>": 200012,
                "<|endoftext|>": 199999,
            ])
    }

    // MARK: - Rendu

    @Test("The prompt follows the official template structure")
    func renderMatchesTemplate() {
        let renderer = Harmony.Renderer(currentDate: "2026-08-05", reasoningEffort: .low)
        let text = renderer.render(turns: [.user("Salut")])

        #expect(text.hasPrefix("<|start|>system<|message|>"))
        #expect(text.contains("Knowledge cutoff: 2024-06"))
        #expect(text.contains("Current date: 2026-08-05"))
        #expect(text.contains("Reasoning: low"))
        #expect(text.contains("# Valid channels: analysis, commentary, final."))
        #expect(text.contains("<|start|>user<|message|>Salut<|end|>"))
        // The prompt ends on the generation cue, with no channel: it is up to the model to
        // choose whether it starts with analysis or with final.
        #expect(text.hasSuffix("<|start|>assistant"))
    }

    @Test("Developer instructions are rendered only when present")
    func developerMessageIsOptional() {
        let without = Harmony.Renderer().render(turns: [.user("a")])
        #expect(!without.contains("<|start|>developer"))

        let with = Harmony.Renderer(instructions: "Answer in verse.").render(turns: [.user("a")])
        #expect(with.contains("<|start|>developer<|message|># Instructions\n\nAnswer in verse."))
    }

    /// The official template is explicit: past turns' reasoning is never fed back at inference
    /// time. Reintroducing it would make the model drift.
    @Test("The history keeps only the final channel")
    func historyKeepsOnlyFinalChannel() {
        let text = Harmony.Renderer().render(turns: [
            .user("Question 1"), .assistant("Answer 1"), .user("Question 2"),
        ])
        #expect(text.contains("<|start|>assistant<|channel|>final<|message|>Answer 1<|end|>"))
        #expect(!text.contains("analysis<|message|>Answer 1"))
    }

    // MARK: - Parsing the output

    /// Reproduces a typical model output and checks that the channels are separated.
    @Test("The parser separates reasoning from answer")
    func parserSeparatesChannels() throws {
        let tokenizer = try Self.makeTokenizer()
        let parser = Harmony.Parser(tokenizer: tokenizer)
        var session = Harmony.Parser.Session()

        let output = "<|channel|>analysis<|message|>Bonjour<|end|>"
            + "<|start|>assistant<|channel|>final<|message|>Bonjour<|return|>"
        for token in tokenizer.encode(output, allowSpecial: true) {
            _ = parser.consume(token, session: &session)
        }

        #expect(session.isFinished)
        #expect(session.analysisText == "Bonjour")
        #expect(session.finalText == "Bonjour")
    }

    /// The bug that motivated this test: after `<|start|>`, the model writes the role name.
    /// Untreated, "assistant" appeared at the head of every displayed answer.
    @Test("The role name is never counted as content")
    func roleNameIsNotContent() throws {
        let tokenizer = try Self.makeTokenizer()
        let parser = Harmony.Parser(tokenizer: tokenizer)
        var session = Harmony.Parser.Session()

        let output = "<|start|>assistant<|channel|>final<|message|>Bonjour<|return|>"
        for token in tokenizer.encode(output, allowSpecial: true) {
            _ = parser.consume(token, session: &session)
        }
        #expect(session.finalText == "Bonjour", "obtenu « \(session.finalText) »")
    }

    @Test("Every stop token ends the generation")
    func stopTokensFinish() throws {
        let tokenizer = try Self.makeTokenizer()
        for stop in Harmony.stopTokenNames {
            let parser = Harmony.Parser(tokenizer: tokenizer)
            var session = Harmony.Parser.Session()
            for token in tokenizer.encode(
                "<|channel|>final<|message|>Bonjour" + stop, allowSpecial: true)
            {
                _ = parser.consume(token, session: &session)
            }
            #expect(session.isFinished, "\(stop) does not end the generation")
            #expect(session.finalText == "Bonjour")
        }
    }

    /// An accented character or an emoji can be split across several tokens. Decoding token by
    /// token would produce replacement characters mid-word.
    @Test("Multi-byte text recomposes without replacement characters")
    func multibyteTextIsReassembled() throws {
        let tokenizer = try Self.makeTokenizer()
        let parser = Harmony.Parser(tokenizer: tokenizer)
        var session = Harmony.Parser.Session()

        // Deliberately mixed scripts: each of these spans several tokens, which is the
        // whole point of the test.
        let content = "éàü — 🙂 日本語"
        for token in tokenizer.encode(
            "<|channel|>final<|message|>" + content + "<|return|>", allowSpecial: true)
        {
            _ = parser.consume(token, session: &session)
        }
        #expect(session.finalText == content)
        #expect(!session.finalText.contains("\u{FFFD}"), "replacement character present")
    }

    @Test("The fragments emitted reconstruct the final text exactly")
    func emittedFragmentsMatchFinalText() throws {
        let tokenizer = try Self.makeTokenizer()
        let parser = Harmony.Parser(tokenizer: tokenizer)
        var session = Harmony.Parser.Session()

        var streamed = ""
        for token in tokenizer.encode(
            "<|channel|>final<|message|>Hello world éàü<|return|>", allowSpecial: true)
        {
            for event in parser.consume(token, session: &session) {
                if case .text(.final, let fragment) = event { streamed += fragment }
            }
        }
        // Streaming display must give the same result as accumulating.
        #expect(streamed == session.finalText)
    }
}
