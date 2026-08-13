import Foundation
import HydraCore
import Testing

@testable import HydraTokenize

/// Qwen's prompt format, against the published `chat_template.jinja`.
@Suite("Qwen prompt format")
struct QwenFormatTests {

    private let format = QwenFormat()

    @Test("Turns are bracketed and the assistant turn is opened for the model")
    func rendersTurns() {
        let out = format.render(
            turns: [.init(role: .user, content: "hello")],
            settings: PromptSettings(reasoning: .medium, instructions: "be brief"))

        #expect(out.contains("<|im_start|>system\nbe brief<|im_end|>\n"))
        #expect(out.contains("<|im_start|>user\nhello<|im_end|>\n"))
        #expect(
            out.hasSuffix("<|im_start|>assistant\n<think>\n"),
            "the prompt opens the thinking block itself, so the model continues inside it")
    }

    /// Thinking off is an empty block, not a missing one.
    ///
    /// The template pre-fills `<think>\n\n</think>\n\n` so the tags stay well formed and the
    /// model writes its answer directly. Omitting them instead leaves the model to open a block
    /// nobody closes.
    @Test("Thinking off pre-fills an empty block")
    func thinkingOffIsAnEmptyBlock() {
        let out = format.render(
            turns: [.init(role: .user, content: "hi")],
            settings: PromptSettings(reasoning: .off))
        #expect(out.hasSuffix("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
    }

    /// History is replayed without its reasoning, as the template does.
    @Test("A replayed assistant turn drops its thinking")
    func historyDropsThinking() {
        let out = format.render(
            turns: [
                .init(role: .user, content: "q"),
                .init(role: .assistant, content: "<think>\nworking\n</think>\n\nthe answer"),
                .init(role: .user, content: "again"),
            ],
            settings: PromptSettings(reasoning: .medium))
        #expect(out.contains("<|im_start|>assistant\nthe answer<|im_end|>\n"))
        #expect(!out.contains("working"), "discarded working must not be replayed as content")
    }

    /// The parser starts inside the block, because the prompt opened it.
    @Test("Reasoning before the closing tag is reasoning, and after it is the answer")
    func parserSplitsOnTheClosingTag() throws {
        let tokenizer = try makeTokenizer()
        let parser = format.makeParser(
            tokenizer: tokenizer, settings: PromptSettings(reasoning: .medium))

        var reasoning = "", answer = ""
        for token in tokenizer.encode("weighing it </think> so: yes<|im_end|>", allowSpecial: true) {
            for event in parser.consume(token) {
                switch event {
                case .reasoning(let text): reasoning += text
                case .answer(let text): answer += text
                case .stopped: break
                }
            }
        }
        #expect(reasoning.contains("weighing it"))
        #expect(answer.contains("yes"))
        #expect(!answer.contains("weighing"), "the thinking must not reach the answer")
        #expect(parser.isFinished)
    }

    /// With thinking off the parser starts outside, because the prompt already closed the block.
    @Test("With thinking off the first text is the answer")
    func parserStartsOutsideWhenThinkingIsOff() throws {
        let tokenizer = try makeTokenizer()
        let parser = format.makeParser(
            tokenizer: tokenizer, settings: PromptSettings(reasoning: .off))

        var answer = "", reasoning = ""
        for token in tokenizer.encode("straight to it<|im_end|>", allowSpecial: true) {
            for event in parser.consume(token) {
                switch event {
                case .answer(let text): answer += text
                case .reasoning(let text): reasoning += text
                case .stopped: break
                }
            }
        }
        #expect(answer.contains("straight to it"))
        #expect(reasoning.isEmpty, "nothing was reasoning: the prompt closed the block")
    }

    /// A tiny vocabulary that spells the markers, enough to exercise the parser.
    private func makeTokenizer() throws -> BPETokenizer {
        var vocabulary: [String: Int] = [:]
        var next = 0
        for marker in QwenFormat.Marker.allCases {
            vocabulary[marker.rawValue] = next
            next += 1
        }
        // Byte-level pieces for the rest, so any text encodes.
        for byte in 0..<256 {
            let piece = ByteLevel.encode([UInt8(byte)])
            if vocabulary[piece] == nil { vocabulary[piece] = next; next += 1 }
        }
        var special: [String: Int] = [:]
        for marker in QwenFormat.Marker.allCases { special[marker.rawValue] = vocabulary[marker.rawValue]! }
        return try BPETokenizer(
            vocabulary: vocabulary, merges: [], specialTokens: special,
            conventions: .qwen)
    }
}
