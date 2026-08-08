import Foundation
import Testing

@testable import HydraTokenize

/// Gemma 4's conversation format, checked against the published `chat_template.jinja`.
///
/// A prompt the model has never seen does not fail — it answers worse. So the structure is
/// asserted marker by marker rather than by eye, and the two details most likely to be got
/// wrong from experience with Gemma 2 and 3 each have their own test.
@Suite("Gemma 4 prompt format")
struct Gemma4PromptTests {

    /// A tokenizer carrying the real marker identifiers, so the parser is exercised against
    /// the numbers the published file assigns.
    private func makeTokenizer() throws -> BPETokenizer {
        var vocabulary: [String: Int] = [:]
        var next = 300
        func add(_ piece: String) {
            vocabulary[piece] = next
            next += 1
        }
        for scalar in "abcdefghijklmnopqrstuvwxyz .,!?" { add(String(scalar)) }
        add("\u{2581}")
        add("thought")
        add("\n")
        for byte in 0...255 { vocabulary[String(format: "<0x%02X>", byte)] = 1000 + byte }

        var special: [String: Int] = [:]
        for marker in Gemma4Prompt.Marker.allCases { special[marker.rawValue] = marker.publishedID }

        return try BPETokenizer(
            vocabulary: vocabulary, merges: [], specialTokens: special,
            conventions: .gemma4)
    }

    // MARK: - Rendering

    /// The assistant's role is written `model`. Writing `assistant` is the single likeliest
    /// mistake, and the template never uses that word as a role.
    @Test("The assistant's role is written model")
    func assistantIsCalledModel() {
        let rendered = Gemma4Prompt.Renderer().render(turns: [
            .user("hello"), .model("hi"),
        ])
        #expect(rendered.contains("<|turn>model\n"))
        #expect(!rendered.contains("assistant"))
    }

    /// With thinking off, the generation prompt carries an **already-closed** thought channel.
    /// That is the suppression mechanism; omitting it asks a thinking model to think.
    @Test("Thinking off pre-closes the thought channel")
    func thinkingOffClosesTheChannel() {
        let off = Gemma4Prompt.Renderer(thinking: false).render(turns: [.user("hi")])
        #expect(off.hasSuffix("<|turn>model\n<|channel>thought\n<channel|>"))
        #expect(!off.contains("<|think|>"))

        let on = Gemma4Prompt.Renderer(thinking: true).render(turns: [.user("hi")])
        #expect(on.hasSuffix("<|turn>model\n"))
        #expect(on.contains("<|think|>"))
    }

    @Test("The prompt opens with the beginning-of-sequence marker")
    func startsWithBos() {
        let rendered = Gemma4Prompt.Renderer().render(turns: [.user("hi")])
        #expect(rendered.hasPrefix("<bos>"))
    }

    /// A system turn appears only when there is something to put in it — either instructions
    /// or the thinking marker. An empty one would be a turn the template never emits.
    @Test("The system turn appears only when it has content")
    func systemTurnIsConditional() {
        #expect(!Gemma4Prompt.Renderer().render(turns: [.user("hi")]).contains("<|turn>system"))
        #expect(!Gemma4Prompt.Renderer(instructions: "   ")
            .render(turns: [.user("hi")]).contains("<|turn>system"))

        let withInstructions = Gemma4Prompt.Renderer(instructions: "Be terse.")
            .render(turns: [.user("hi")])
        #expect(withInstructions.contains("<|turn>system\nBe terse.<turn|>\n"))

        // Thinking alone justifies the turn, with no instructions.
        let thinkingOnly = Gemma4Prompt.Renderer(thinking: true).render(turns: [.user("hi")])
        #expect(thinkingOnly.contains("<|turn>system\n<|think|>\n<turn|>\n"))
    }

    @Test("Every turn is closed")
    func turnsAreBalanced() {
        let rendered = Gemma4Prompt.Renderer(instructions: "Be terse.").render(turns: [
            .user("one"), .model("two"), .user("three"),
        ])
        let opens = rendered.components(separatedBy: "<|turn>").count - 1
        let closes = rendered.components(separatedBy: "<turn|>").count - 1
        // One more open than close: the trailing generation prompt is deliberately unclosed.
        #expect(opens == closes + 1)
    }

    // MARK: - Parsing

    @Test("Plain text arrives as the answer")
    func parsesPlainText() throws {
        let tokenizer = try makeTokenizer()
        let parser = Gemma4Prompt.Parser(tokenizer: tokenizer)
        var session = Gemma4Prompt.Parser.Session()

        var events: [Gemma4Prompt.Parser.Event] = []
        for token in tokenizer.encode("hi there") {
            events += parser.consume(token, session: &session)
        }
        #expect(session.answer == "hi there")
        #expect(session.reasoning.isEmpty)
        #expect(!events.isEmpty)
    }

    /// Reasoning must land in its own field, and the channel's **name** must not leak into it.
    /// Without that, every reply would begin with the word "thought".
    @Test("Thought content is separated, and the channel name does not leak")
    func separatesReasoning() throws {
        let tokenizer = try makeTokenizer()
        let parser = Gemma4Prompt.Parser(tokenizer: tokenizer)
        var session = Gemma4Prompt.Parser.Session()

        var tokens = [Gemma4Prompt.Marker.channelOpen.publishedID]
        tokens += tokenizer.encode("thought\n")
        tokens += tokenizer.encode("let me see")
        tokens.append(Gemma4Prompt.Marker.channelClose.publishedID)
        tokens += tokenizer.encode("the answer")

        for token in tokens { _ = parser.consume(token, session: &session) }

        #expect(session.reasoning == "let me see")
        #expect(session.answer == "the answer")
        #expect(!session.reasoning.contains("thought"))
    }

    @Test("A turn close ends the generation")
    func stopsOnTurnClose() throws {
        let tokenizer = try makeTokenizer()
        let parser = Gemma4Prompt.Parser(tokenizer: tokenizer)
        var session = Gemma4Prompt.Parser.Session()

        for token in tokenizer.encode("done") { _ = parser.consume(token, session: &session) }
        let events = parser.consume(Gemma4Prompt.Marker.turnClose.publishedID, session: &session)

        #expect(events.contains(.stopped))
        #expect(session.isFinished)
        #expect(session.answer == "done")
        // Nothing is consumed after the stop.
        #expect(parser.consume(tokenizer.encode("more")[0], session: &session).isEmpty)
    }

    @Test("Both stop markers are recognized")
    func stopTokensAreFound() throws {
        let tokenizer = try makeTokenizer()
        let stops = Gemma4Prompt.stopTokens(in: tokenizer)
        #expect(stops.contains(Gemma4Prompt.Marker.turnClose.publishedID))
        #expect(stops.contains(Gemma4Prompt.Marker.endOfSequence.publishedID))
        #expect(stops.count == 2)
    }

    /// The identifiers the published tokenizer assigns. Recorded so that a checkpoint using
    /// different ones is a visible mismatch rather than a silent misparse.
    @Test("The marker identifiers match the published tokenizer")
    func markerIdentifiersMatchThePublishedFile() {
        #expect(Gemma4Prompt.Marker.turnOpen.publishedID == 105)
        #expect(Gemma4Prompt.Marker.turnClose.publishedID == 106)
        #expect(Gemma4Prompt.Marker.think.publishedID == 98)
        #expect(Gemma4Prompt.Marker.channelOpen.publishedID == 100)
        #expect(Gemma4Prompt.Marker.channelClose.publishedID == 101)
        #expect(Gemma4Prompt.Marker.beginningOfSequence.publishedID == 2)
        #expect(Gemma4Prompt.Marker.endOfSequence.publishedID == 1)
    }
}
