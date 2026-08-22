import Foundation
import HydraCore
import HydraSearch
import Testing

@testable import HydraTokenize

/// The two-pass search turn, which is what actually ships.
///
/// The tool-call protocol next door is retained but unreachable: nothing declares tools, so the
/// model is never asked whether to search. It is asked for a query, without thinking, and then
/// handed the results. That decision is the whole of M-077 — deliberation is where this model
/// fails, and a turn that has already been given the facts has nothing to deliberate about.
///
/// These tests pin the shape the engine depends on. The engine rewinds the recurrence to the
/// exact position it prefilled the conversation to, and that only works if the conversation and
/// its generation prompt are rendered separately and join back byte for byte.
@Suite("Qwen's two-pass search turn")
struct QwenSearchPassTests {

    private let format = QwenFormat()
    private let turns: [ChatTurn] = [.user("what changed in the new release")]

    private func settings(
        _ reasoning: ReasoningLevel = .medium, searching: Bool = true
    ) -> PromptSettings {
        PromptSettings(reasoning: reasoning, searching: searching, today: "2026-08-22")
    }

    // MARK: - The seam the rewind depends on

    @Test("The open conversation plus the generation prompt is the whole prompt")
    func openPlusTailIsRender() {
        // The engine prefills `renderOpen`, notes the position, and continues from it twice.
        // If these two did not compose exactly, the second continuation would resume at a
        // position describing different text and every token after it would be wrong — silently,
        // because nothing checks.
        for reasoning in ReasoningLevel.allCases {
            let whole = format.render(turns: turns, settings: settings(reasoning))
            let open = format.renderOpen(turns: turns, settings: settings(reasoning))
            #expect(open + format.generationPrompt(settings(reasoning)) == whole, "\(reasoning)")
            #expect(!open.hasSuffix("<think>\n"), "the open form must not open thinking")
        }
    }

    @Test("The open conversation ends after the user's turn")
    func openEndsAtTheUserTurn() {
        let open = format.renderOpen(turns: turns, settings: settings())
        #expect(open.hasSuffix("<|im_end|>\n"))
        #expect(!open.contains("<|im_start|>assistant"))
    }

    // MARK: - The query pass

    @Test("The query request closes thinking whatever the conversation asked for")
    func queryRequestNeverThinks() {
        // Not a detail. Measured over the same prompt and seeds, the answer phase degenerates
        // with thinking on and never with it off; the query pass is the one generation that has
        // never degenerated, and its being unthinking is why.
        let request = format.renderQueryRequest()
        #expect(request.hasSuffix("<think>\n\n</think>\n\n"))
        #expect(request.contains("<|im_start|>assistant\n"))
    }

    @Test("The query request asks for the query alone, and for the question's own words")
    func queryRequestWording() {
        let request = format.renderQueryRequest()
        #expect(request.contains("nothing else"))
        // The failure this exists for: asked to reason first, the model replaced the unfamiliar
        // names in the question with familiar ones and searched for those instead.
        #expect(request.contains("exact names, versions and numbers"))
    }

    // MARK: - Feeding the results back

    @Test("Results arrive as a user turn and reopen the assistant")
    func resultsShape() {
        let rendered = format.renderSearchResults("[1] a page\nsome text", settings: settings())
        #expect(rendered.hasPrefix("<|im_start|>user\n<tool_response>\n"))
        #expect(rendered.contains("[1] a page\nsome text"))
        #expect(rendered.contains("</tool_response>"))
        #expect(rendered.hasSuffix("<|im_start|>assistant\n<think>\n"))
    }

    @Test("The answer instruction sits outside the results, not inside them")
    func instructionIsNotPageText() {
        let rendered = format.renderSearchResults("page text", settings: settings())
        let closing = try? #require(rendered.range(of: "</tool_response>"))
        let instruction = try? #require(rendered.range(of: "Now answer my question"))
        // Inside the block it would read as more retrieved text, which the block itself tells
        // the model not to take instruction from.
        #expect((closing?.upperBound ?? rendered.endIndex)
            <= (instruction?.lowerBound ?? rendered.startIndex))
    }

    @Test("The answer instruction carries both halves of what it is for")
    func instructionCoversBothFailures() {
        let text = QwenFormat.answerInstruction
        // Handed a result whose URL was `huggingface.co/Qwen/Qwen3.8-27B`, the model wrote that
        // no such model existed and cited that result in the next sentence.
        #expect(text.contains("your training predates them"))
        // And, once told to report what the sources say, it refused any question no single
        // source answered outright — which is most comparisons.
        #expect(text.contains("Combine what several of them say"))
        #expect(text.contains("Do not refuse"))
    }

    @Test("A searching turn answers without thinking, and says so in the prompt")
    func searchingTurnsClosesThinking() {
        // The engine forces `.off` for a searching turn; this is the rendered consequence.
        let rendered = format.renderSearchResults("x", settings: settings(.off))
        #expect(rendered.hasSuffix("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
    }

    // MARK: - Reading the query back

    @Test("The query is taken from what the model wrote", arguments: [
        ("Qwen3.8 27B benchmarks", "Qwen3.8 27B benchmarks"),
        ("  Qwen3.8 27B benchmarks  ", "Qwen3.8 27B benchmarks"),
        ("\"Qwen3.8 27B benchmarks\"", "Qwen3.8 27B benchmarks"),
        ("`Qwen3.8 27B benchmarks`", "Qwen3.8 27B benchmarks"),
        ("query: Qwen3.8 27B benchmarks", "Qwen3.8 27B benchmarks"),
        ("Search query: Qwen3.8 27B benchmarks", "Qwen3.8 27B benchmarks"),
        // The instruction asks for one line and mostly gets one; when it does not, the query
        // is still the first line and the rest is the model ignoring the instruction.
        ("Qwen3.8 27B benchmarks\n\nThis should find the official card.",
         "Qwen3.8 27B benchmarks"),
        ("", ""),
        ("   \n  ", ""),
    ])
    func queryCleaning(raw: String, expected: String) {
        #expect(WebSearchTool.cleanQuery(raw) == expected)
    }

    @Test("Nested decoration is unwrapped, not left half-stripped")
    func nestedQuotes() {
        #expect(WebSearchTool.cleanQuery("\"`a query`\"") == "a query")
        #expect(WebSearchTool.cleanQuery("query: \"a query\"") == "a query")
    }

    @Test("A quote that is not a wrapper is left alone")
    func unmatchedQuotes() {
        // `26" MacBook` is a query, not a quoted string, and stripping one end would corrupt it.
        #expect(WebSearchTool.cleanQuery("what is a 26\" display") == "what is a 26\" display")
    }

    // MARK: - What the turn costs when it is off

    @Test("Not searching renders exactly what it rendered before search existed")
    func notSearchingIsUnchanged() {
        let plain = format.render(
            turns: [.user("hello")], settings: PromptSettings(reasoning: .medium))
        #expect(plain == "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n")
        #expect(!plain.contains("Today's date"))
        #expect(!plain.contains("# Tools"))
    }
}
