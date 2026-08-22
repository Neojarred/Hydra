import Foundation
import HydraCore
import Testing

@testable import HydraTokenize

/// Qwen's tool protocol: declaring a function, reading the call back, feeding the result in.
///
/// Everything here streams through a tokenizer that emits **one byte at a time**, because that
/// is the case the project has already been bitten by (M-071). Qwen's tool markers are plain
/// text rather than special tokens, so `</tool_call>` is not one token the parser can match on:
/// it is twelve bytes that arrive whenever byte-level BPE feels like splitting them. A parser
/// that matched on token identity would pass a test written with whole markers and fail on the
/// first real generation.
@Suite("Qwen tool calls")
struct QwenToolCallTests {

    // MARK: - The worst-case stream

    /// A vocabulary of the 256 byte tokens and nothing else, so every marker is split as far as
    /// it can be. The same instrument `QwenEmojiTests` uses, for the same reason.
    private func byteTokenizer() throws -> (BPETokenizer, (String) -> [Int]) {
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

    private func stream(
        _ text: String, inThought: Bool = false
    ) throws -> (events: [PromptEvent], answer: String, reasoning: String) {
        let (tokenizer, encode) = try byteTokenizer()
        let parser = QwenParser(tokenizer: tokenizer, inThought: inThought)
        var events: [PromptEvent] = []
        for token in encode(text) where !parser.isFinished {
            events += parser.consume(token)
        }
        var answer = "", reasoning = ""
        for event in events {
            if case .answer(let fragment) = event { answer += fragment }
            if case .reasoning(let fragment) = event { reasoning += fragment }
        }
        return (events, answer, reasoning)
    }

    private func call(in events: [PromptEvent]) -> ToolCall? {
        for event in events { if case .toolCall(let call) = event { return call } }
        return nil
    }

    // MARK: - Reading a call back

    @Test("A call split into single bytes is read whole")
    func callSurvivesByteSplitting() throws {
        let result = try stream("""
            <tool_call>
            <function=web_search>
            <parameter=query>
            M4 Max memory bandwidth
            </parameter>
            </function>
            </tool_call><|im_end|>
            """)

        let call = try #require(self.call(in: result.events))
        #expect(call.name == "web_search")
        #expect(call["query"] == "M4 Max memory bandwidth")
        // Nothing of the call reaches the user's answer. Before the parser knew the markers,
        // every one of these characters would have been printed into the message.
        #expect(result.answer.isEmpty)
        #expect(!result.answer.contains("tool_call"))
        #expect(!result.answer.contains("parameter"))
    }

    @Test("The call ends the turn, and says so after announcing itself")
    func callIsFollowedByStopped() throws {
        let result = try stream("""
            <tool_call>
            <function=web_search>
            <parameter=query>
            anything
            </parameter>
            </function>
            </tool_call>
            """)
        // Qwen has no stop token for a call: `</tool_call>` both ends the generation and is the
        // call. Order matters — a consumer that saw `.stopped` first and stopped reading would
        // finish the turn holding a request it never noticed.
        #expect(result.events.count >= 2)
        #expect(result.events[result.events.count - 2] == .toolCall(
            ToolCall(name: "web_search", arguments: ["query": "anything"])))
        #expect(result.events.last == .stopped)
    }

    @Test("Words before the call are the answer, and reach the user")
    func preambleIsAnswerText() throws {
        // The template invites exactly this: reasoning in natural language *before* the call.
        let result = try stream("""
            Let me look that up.
            <tool_call>
            <function=web_search>
            <parameter=query>
            bandwidth
            </parameter>
            </function>
            </tool_call>
            """)
        #expect(result.answer.contains("Let me look that up."))
        #expect(!result.answer.contains("<tool_call>"))
        #expect(self.call(in: result.events) != nil)
    }

    @Test("A call after a thinking block is still a call")
    func callAfterThinking() throws {
        let result = try stream("""
            I should search for this.
            </think>
            <tool_call>
            <function=web_search>
            <parameter=query>
            unified memory
            </parameter>
            </function>
            </tool_call>
            """, inThought: true)
        #expect(result.reasoning.contains("I should search for this."))
        #expect(self.call(in: result.events)?["query"] == "unified memory")
    }

    @Test("Several parameters, in any order")
    func severalParameters() throws {
        let result = try stream("""
            <tool_call>
            <function=web_search>
            <parameter=query>
            apple silicon
            </parameter>
            <parameter=count>
            8
            </parameter>
            </function>
            </tool_call>
            """)
        let call = try #require(self.call(in: result.events))
        #expect(call.arguments == ["query": "apple silicon", "count": "8"])
    }

    @Test("A value spanning several lines keeps its newlines")
    func multiLineValue() throws {
        // The template's own example advertises multi-line values, so trimming to one line
        // would silently truncate a legitimate argument.
        let result = try stream("""
            <tool_call>
            <function=note>
            <parameter=body>
            first line
            second line
            </parameter>
            </function>
            </tool_call>
            """)
        #expect(self.call(in: result.events)?["body"] == "first line\nsecond line")
    }

    @Test("An argument carrying multi-byte characters survives the buffer")
    func multiByteArgument() throws {
        let result = try stream("""
            <tool_call>
            <function=web_search>
            <parameter=query>
            température à Paris 🌡️
            </parameter>
            </function>
            </tool_call>
            """)
        let query = try #require(self.call(in: result.events)?["query"])
        #expect(query == "température à Paris 🌡️")
        #expect(!query.contains("\u{FFFD}"))
    }

    // MARK: - Refusing what cannot be read

    @Test("A call cut off by the end of the turn is discarded, not guessed at")
    func truncatedCallIsDiscarded() throws {
        let result = try stream("""
            <tool_call>
            <function=web_search>
            <parameter=query>
            half a quer<|im_end|>
            """)
        // A truncated call is worse than no call: running a search for "half a quer" spends a
        // credit and 15 seconds of prefill on a question nobody asked.
        #expect(self.call(in: result.events) == nil)
        #expect(result.events.last == .stopped)
    }

    @Test("A block with no function name yields no call")
    func namelessBlockIsDiscarded() throws {
        let result = try stream("<tool_call>\nnonsense\n</tool_call>")
        #expect(self.call(in: result.events) == nil)
        #expect(result.events.last == .stopped)
    }

    @Test("parseToolCall refuses malformed blocks", arguments: [
        "", "no markers at all", "<function=>\n<parameter=q>\nv\n</parameter>",
        "<parameter=q>\nv\n</parameter>",
    ])
    func parseRefusesMalformed(block: String) {
        #expect(QwenFormat.parseToolCall(block) == nil)
    }

    @Test("An unterminated parameter does not swallow the rest")
    func unterminatedParameter() {
        let call = try? #require(
            QwenFormat.parseToolCall("<function=f>\n<parameter=a>\nvalue\n</parameter>\n"
                + "<parameter=b>\ndangling"))
        #expect(call?.name == "f")
        // `a` is whole and kept; `b` never closed and is dropped rather than run to the end of
        // the block.
        #expect(call?.arguments == ["a": "value"])
    }

    // MARK: - Declaring the tools

    private var searchTool: ToolDefinition {
        ToolDefinition(
            name: "web_search",
            description: "Search the web and return ranked snippets.",
            parameters: [
                .init(name: "query", description: "What to search for.", required: true),
                .init(name: "count", type: "number", description: "How many results."),
            ])
    }

    @Test("Without tools, the prompt is what it has always been")
    func noToolsRendersUnchanged() {
        // The regression that matters most. Every format declares tools at the *head* of the
        // prompt, so a declaration that leaks in when none was asked for shifts every token
        // after it and costs the conversation its whole cached prefix — 90 seconds on a long
        // chat, for a feature that is switched off.
        let format = QwenFormat()
        let turns: [ChatTurn] = [.user("hello"), .assistant("hi"), .user("again")]
        let plain = format.render(
            turns: turns, settings: PromptSettings(reasoning: .medium))

        let expected =
            "<|im_start|>user\nhello<|im_end|>\n"
            + "<|im_start|>assistant\nhi<|im_end|>\n"
            + "<|im_start|>user\nagain<|im_end|>\n"
            + "<|im_start|>assistant\n<think>\n"
        #expect(plain == expected)
        #expect(!plain.contains("# Tools"))
    }

    @Test("With tools, the declaration leads a system turn")
    func toolsRenderInTheSystemTurn() {
        let rendered = QwenFormat().render(
            turns: [.user("hello")],
            settings: PromptSettings(reasoning: .medium, tools: [searchTool]))

        #expect(rendered.hasPrefix("<|im_start|>system\n# Tools\n"))
        #expect(rendered.contains("<tools>"))
        #expect(rendered.contains("</tools>"))
        #expect(rendered.contains("\"name\":\"web_search\""))
        // The template's own reminder, which is the text the checkpoint was trained against.
        #expect(rendered.contains("<IMPORTANT>"))
        #expect(rendered.contains("</tool_call> XML tags"))
    }

    // MARK: - Telling the model what year it is

    @Test("With tools, the model is told the date and that its memory is older")
    func toolsCarryTheDate() {
        let rendered = QwenFormat().render(
            turns: [.user("what is new")],
            settings: PromptSettings(
                reasoning: .medium, tools: [searchTool], today: "2026-08-21"))

        // Qwen's template states no date, where Harmony's does. Without this the model argues
        // with retrieved evidence from inside 2024 and spends its whole budget doing it.
        #expect(rendered.contains("Today's date is 2026-08-21."))
        #expect(rendered.contains("the result is right"))
        // In the system turn, before the conversation starts.
        let date = try? #require(rendered.range(of: "Today's date is"))
        let firstUser = try? #require(rendered.range(of: "<|im_start|>user"))
        #expect((date?.lowerBound ?? rendered.endIndex)
            < (firstUser?.lowerBound ?? rendered.startIndex))
    }

    @Test("Searching states the date even with no tools declared")
    func searchingCarriesTheDateWithoutTools() {
        // The regression this exists for: the situation note was rendered only alongside a tool
        // declaration, so the day the model stopped being offered tools it also stopped being
        // told what year it was — and went straight back to answering from its training cutoff
        // and calling the search results fabricated.
        let rendered = QwenFormat().render(
            turns: [.user("what is new")],
            settings: PromptSettings(reasoning: .medium, searching: true, today: "2026-08-21"))

        #expect(rendered.contains("Today's date is 2026-08-21."))
        #expect(rendered.contains("never dismiss them as fabricated"))
        #expect(!rendered.contains("# Tools"), "no declaration, just the situation")
    }

    @Test("Without tools, no date is stated and nothing changes at midnight")
    func noDateWithoutTools() {
        // A line that changes daily costs every conversation its cached prefix once a day.
        // With search off there is nothing to gain from paying that.
        let rendered = QwenFormat().render(
            turns: [.user("hello")], settings: PromptSettings(reasoning: .medium))
        #expect(!rendered.contains("Today's date"))
    }

    @Test("The date is the day, in UTC, formatted as Harmony formats it")
    func dateFormat() {
        // 23:30 UTC, deliberately: in this machine's own zone that instant is already the
        // next day, so a formatter that forgot to pin UTC would report the 18th and this is
        // the only kind of instant that catches it.
        let stamp = QwenFormat.today(Date(timeIntervalSince1970: 1_787_009_400))
        #expect(stamp.wholeMatch(of: /\d{4}-\d{2}-\d{2}/) != nil)
        #expect(stamp == "2026-08-17")
    }

    @Test("Instructions still follow, after the declaration and the date")
    func instructionsComeLast() {
        let rendered = QwenFormat().render(
            turns: [.user("hello")],
            settings: PromptSettings(
                reasoning: .medium, instructions: "Be brief.", tools: [searchTool],
                today: "2026-08-21"))
        let date = try? #require(rendered.range(of: "Today's date is"))
        let brief = try? #require(rendered.range(of: "Be brief."))
        #expect((date?.lowerBound ?? rendered.endIndex)
            < (brief?.lowerBound ?? rendered.startIndex))
    }

    @Test("Instructions and tools share one system turn, declaration first")
    func instructionsFollowTheDeclaration() {
        let rendered = QwenFormat().render(
            turns: [.user("hello")],
            settings: PromptSettings(
                reasoning: .medium, instructions: "Be brief.", tools: [searchTool]))

        let system = try? #require(rendered.range(of: "<|im_end|>"))
        let head = String(rendered[rendered.startIndex..<(system?.lowerBound ?? rendered.endIndex)])
        #expect(head.contains("# Tools"))
        #expect(head.contains("Be brief."))
        // One system turn, not two: the template folds them together, and two would be a
        // dialect the model has not seen.
        #expect(rendered.components(separatedBy: "<|im_start|>system").count == 2)
    }

    @Test("The schema renders identically every time")
    func schemaIsDeterministic() {
        // The prompt is diffed against the previous turn's tokens to decide what to prefill.
        // A schema whose key order wandered between two renders of the same tool would cost a
        // full re-prefill and nothing would report why.
        let once = QwenFormat.schema(for: searchTool)
        for _ in 0..<20 { #expect(QwenFormat.schema(for: searchTool) == once) }
        #expect(once.contains("\"required\":[\"query\"]"))
        #expect(once.contains("\"type\":\"number\""))
    }

    @Test("Supporting a dialect and being offered the tool are different questions")
    func supportIsStated() {
        // Qwen can render and parse a call and is deliberately never offered one: its turn is
        // split instead, because it cannot be trusted with the deliberation that precedes a
        // decision to search (M-077).
        #expect(QwenFormat().supportsTools)
        #expect(!QwenFormat().declaresTools)

        // Gemma is offered it. Its dialect is token-delimited and its checkpoint is trained on
        // one, so it decides for itself.
        #expect(Gemma4Format().supportsTools)
        #expect(Gemma4Format().declaresTools)

        // GPT-OSS has neither yet.
        #expect(!HarmonyFormat().supportsTools)
        #expect(!HarmonyFormat().declaresTools)
    }

    // MARK: - Feeding the result back

    @Test("A tool result closes the turn, files the response, and reopens the assistant")
    func toolResultRendering() {
        let rendered = QwenFormat().renderToolResult(
            "[1] a page\nsome text", settings: PromptSettings(reasoning: .medium))

        #expect(rendered == "<|im_end|>\n<|im_start|>user\n<tool_response>\n"
            + "[1] a page\nsome text\n</tool_response><|im_end|>\n"
            + "<|im_start|>assistant\n<think>\n")
    }

    @Test("The reopened assistant turn is where the model resumes")
    func toolResultReopensTheAssistant() {
        // The model is mid-turn: it called a function and is about to be handed the answer.
        // A block that forgot to reopen the assistant would leave it writing as the user.
        // The generation prompt that resumes after a tool result is the same one that starts a
        // turn, thinking block included: the template makes no distinction, and a parser told
        // it is inside a block the prompt never opened files the answer as reasoning.
        #expect(QwenFormat().renderToolResult(
            "x", settings: PromptSettings(reasoning: .medium))
            .hasSuffix("<|im_start|>assistant\n<think>\n"))
        #expect(QwenFormat().renderToolResult(
            "x", settings: PromptSettings(reasoning: .off))
            .hasSuffix("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
    }
}
