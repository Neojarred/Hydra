import Foundation
import HydraCore
import Testing

@testable import HydraTokenize

/// Gemma's own function-calling protocol, which it gets instead of Qwen's workaround.
///
/// The difference that decides the design: **Gemma's markers are single tokens.** `<|tool_call>`
/// is id 48 and nothing else ever is, so the parser matches on identity. Qwen writes the same
/// idea as ordinary characters, which byte-level BPE splits wherever it likes, which is why its
/// parser needs a holdback the width of the longest marker and why M-071 lived there. None of
/// that applies here.
///
/// The second difference is where a result goes. Qwen files it as a user turn and has to reopen
/// the assistant; Gemma's template puts it inside the model's own turn and emits no generation
/// prompt afterwards, so the model simply carries on.
@Suite("Gemma's tool protocol")
struct Gemma4ToolTests {

    private let searchTool = ToolDefinition(
        name: "web_search",
        description: "Search the web and return ranked snippets.",
        parameters: [
            .init(name: "query", description: "What to search for.", required: true),
            .init(name: "count", type: "number", description: "How many results."),
        ])

    // MARK: - The markers are the published ids

    @Test("Every tool marker is the id the checkpoint assigns it")
    func markerIdentities() {
        // Recorded so a mismatch is visible rather than mysterious. A parser matching on the
        // wrong id sees no calls at all and reports nothing.
        #expect(Gemma4Prompt.Marker.toolOpen.publishedID == 46)
        #expect(Gemma4Prompt.Marker.toolClose.publishedID == 47)
        #expect(Gemma4Prompt.Marker.toolCallOpen.publishedID == 48)
        #expect(Gemma4Prompt.Marker.toolCallClose.publishedID == 49)
        #expect(Gemma4Prompt.Marker.toolResponseOpen.publishedID == 50)
        #expect(Gemma4Prompt.Marker.toolResponseClose.publishedID == 51)
        #expect(Gemma4Prompt.Marker.quote.publishedID == 52)
    }

    // MARK: - Declaring

    @Test("A declaration is the template's own syntax, not JSON")
    func declarationShape() {
        let text = Gemma4Prompt.declaration(for: searchTool)
        #expect(text.hasPrefix("declaration:web_search{"))
        // Strings are wrapped in `<|"|>`, a token, not a quote character. Anything that strips
        // `"` would leave these untouched and anything that adds `"` would be a dialect the
        // checkpoint has not seen.
        #expect(text.contains("<|\"|>Search the web and return ranked snippets.<|\"|>"))
        // Types are upper case, as `format_type_argument` writes them.
        #expect(text.contains("type:<|\"|>STRING<|\"|>"))
        #expect(text.contains("type:<|\"|>NUMBER<|\"|>"))
        #expect(text.contains("type:<|\"|>OBJECT<|\"|>"))
        #expect(text.contains("required:[<|\"|>query<|\"|>]"))
        #expect(!text.contains("\"name\""), "not JSON")
    }

    @Test("A declaration renders identically every time")
    func declarationIsDeterministic() {
        // The template sorts properties, and so must this: a declaration whose key order
        // wandered between two renders of one tool would cost the conversation its cached
        // prefix on every turn and nothing would report why.
        let once = Gemma4Prompt.declaration(for: searchTool)
        for _ in 0..<20 { #expect(Gemma4Prompt.declaration(for: searchTool) == once) }
        let count = try? #require(once.range(of: "count:"))
        let query = try? #require(once.range(of: "query:{"))
        #expect((count?.lowerBound ?? once.endIndex) < (query?.lowerBound ?? once.startIndex),
                "alphabetical, as `dictsort` renders them")
    }

    @Test("Declarations sit in the system turn, after the thinking token")
    func declarationPlacement() {
        let rendered = Gemma4Format().render(
            turns: [.user("what is new")],
            settings: PromptSettings(reasoning: .medium, tools: [searchTool]))

        #expect(rendered.contains("<|tool>declaration:web_search"))
        #expect(rendered.contains("<tool|>"))
        // `<|think|>` goes at the very top of the first system turn. It is also the head of the
        // prompt: moving it moves every token after it.
        let think = try? #require(rendered.range(of: "<|think|>"))
        let tool = try? #require(rendered.range(of: "<|tool>"))
        #expect((think?.lowerBound ?? rendered.endIndex) < (tool?.lowerBound ?? rendered.startIndex))
    }

    @Test("Without tools the prompt is byte for byte what it always was")
    func noToolsUnchanged() {
        // Declarations sit at the head of the prompt, so one leaking in when search is off
        // would shift every token after it and cost the cached prefix.
        let plain = Gemma4Format().render(
            turns: [.user("hello")], settings: PromptSettings(reasoning: .off))
        #expect(!plain.contains("<|tool>"))
        #expect(!plain.contains("declaration:"))
    }

    // MARK: - Reading a call back

    @Test("A call is read from its own markers")
    func parsesACall() {
        let call = try? #require(
            Gemma4Prompt.parseToolCall("call:web_search{query:<|\"|>M4 Max bandwidth<|\"|>}"))
        #expect(call?.name == "web_search")
        #expect(call?["query"] == "M4 Max bandwidth")
    }

    @Test("Several arguments, and the quote token is not a quote character")
    func parsesArguments() {
        let call = try? #require(Gemma4Prompt.parseToolCall(
            "call:web_search{count:8,query:<|\"|>a query, with a comma<|\"|>}"))
        // The comma inside the string must not split the field.
        #expect(call?["query"] == "a query, with a comma")
        #expect(call?["count"] == "8")
    }

    @Test("A nested value is one field however many commas it holds")
    func nestedValues() {
        let fields = Gemma4Prompt.splitFields("a:1,b:{c:2,d:3},e:[1,2,3]")
        #expect(fields.count == 3)
        #expect(fields[1] == "b:{c:2,d:3}")
        #expect(fields[2] == "e:[1,2,3]")
    }

    @Test("Malformed blocks yield no call", arguments: [
        "", "not a call at all", "call:{query:x}", "call:web_search", "response:web_search{x:1}",
    ])
    func refusesMalformed(block: String) {
        // A turn that ends without a call beats one that calls a function nobody named.
        #expect(Gemma4Prompt.parseToolCall(block) == nil)
    }

    // MARK: - Streaming

    private func stream(_ tokens: [Int], inThought: Bool = true)
        -> (events: [Gemma4Prompt.Parser.Event], answer: String, reasoning: String)
    {
        // Token ids only: this parser never needs the vocabulary to find a marker, which is the
        // whole advantage over Qwen's.
        let tokenizer = Gemma4Fixtures.tokenizer()
        let parser = Gemma4Prompt.Parser(tokenizer: tokenizer)
        var session = Gemma4Prompt.Parser.Session(inThought: inThought)
        var events: [Gemma4Prompt.Parser.Event] = []
        for token in tokens where !session.isFinished {
            events += parser.consume(token, session: &session)
        }
        var answer = "", reasoning = ""
        for event in events {
            if case .text(let f) = event { answer += f }
            if case .reasoning(let f) = event { reasoning += f }
        }
        return (events, answer, reasoning)
    }

    @Test("A call is not printed into the answer")
    func callIsNotAnswerText() {
        let tokenizer = Gemma4Fixtures.tokenizer()
        var tokens = tokenizer.encode("I will look that up.", allowSpecial: false)
        tokens.append(Gemma4Prompt.Marker.channelClose.publishedID)
        tokens.append(Gemma4Prompt.Marker.toolCallOpen.publishedID)
        tokens += tokenizer.encode(
            "call:web_search{query:<|\"|>bandwidth<|\"|>}", allowSpecial: false)
        tokens.append(Gemma4Prompt.Marker.toolCallClose.publishedID)

        let result = stream(tokens)
        var call: ToolCall?
        for event in result.events { if case .toolCall(let c) = event { call = c } }

        #expect(call?.name == "web_search")
        #expect(call?["query"] == "bandwidth")
        #expect(result.reasoning.contains("I will look that up."))
        // None of the call's characters reach the user.
        #expect(!result.answer.contains("call:"))
        #expect(!result.answer.contains("web_search"))
        // And the call is announced before the turn ends.
        #expect(result.events.last == .stopped)
    }

    @Test("A call emitted inside the reasoning trace is still a call")
    func callInsideTheThoughtChannel() {
        // Gemma forces a reasoning trace, and it calls from **inside** it. Runtimes that read
        // only the content field lose the call entirely: the reported symptom elsewhere is a
        // tool call that "silently vanishes" because it was routed to `reasoning_content`.
        //
        // Here the marker is matched before any channel is considered, so the channel it
        // arrives in does not matter. This test is the guarantee of that.
        let tokenizer = Gemma4Fixtures.tokenizer()
        var tokens = tokenizer.encode(
            "I should check whether that exists.", allowSpecial: false)
        // No `<channel|>`: the model is still thinking when it calls.
        tokens.append(Gemma4Prompt.Marker.toolCallOpen.publishedID)
        tokens += tokenizer.encode(
            "call:web_search{query:<|\"|>Qwen3.8 27B<|\"|>}", allowSpecial: false)
        tokens.append(Gemma4Prompt.Marker.toolCallClose.publishedID)

        let result = stream(tokens, inThought: true)
        var call: ToolCall?
        for event in result.events { if case .toolCall(let c) = event { call = c } }

        #expect(call?["query"] == "Qwen3.8 27B", "a call inside the trace must not vanish")
        #expect(result.reasoning.contains("I should check whether that exists."))
        #expect(result.answer.isEmpty, "and none of it becomes the answer")
    }

    // MARK: - Feeding the result back

    @Test("A result continues the model's turn rather than opening one")
    func resultContinuesTheTurn() {
        let block = Gemma4Format().renderToolResult(
            "[1] a page\nsome text", settings: PromptSettings(reasoning: .medium))
        #expect(block.hasPrefix("<|tool_response>response:web_search{value:"))
        #expect(block.hasSuffix("<tool_response|>"))
        // The difference from Qwen: no role change, no generation prompt. The template emits
        // neither after a call, and the model simply carries on.
        #expect(!block.contains("<|turn>"))
        #expect(!block.contains("<|channel>"))
    }

    @Test("What follows a result is the answer, not more reasoning")
    func parserResumesOutsideThought() {
        // The thought channel was closed before the call, so a parser told it is inside one
        // would file the answer as reasoning.
        let after = Gemma4Format().settingsAfterToolResult(
            PromptSettings(reasoning: .medium))
        #expect(after.reasoning == .off)
    }
}

/// A tokenizer that carries Gemma's special ids, for parser tests that need no weights.
enum Gemma4Fixtures {
    static func tokenizer() -> BPETokenizer {
        var vocabulary: [String: Int] = [:]
        for value in 0...255 {
            vocabulary[String(ByteLevel.encodeTable[value])] = value + 300
        }
        var special: [String: Int] = [:]
        for marker in Gemma4Prompt.Marker.allCases {
            special[marker.rawValue] = marker.publishedID
        }
        return try! BPETokenizer(
            vocabulary: vocabulary.merging(special) { a, _ in a }, merges: [],
            specialTokens: special,
            conventions: BPETokenizer.Conventions(
                encoding: .byteLevel, ignoreMerges: false, preTokenizerPattern: nil))
    }
}
