import Foundation
import HydraCore

/// Qwen3.5/3.6's prompt format, transcribed from `chat_template.jinja` in the published
/// checkpoint rather than from the family it belongs to.
///
/// Turns are bracketed by `<|im_start|>role\n` and `<|im_end|>\n`. Reasoning lives in `<think>`
/// tags inside the assistant turn.
///
/// **The generation prompt opens the thinking block itself**, exactly as Gemma's opens its
/// thought channel (D-023): the template emits `<|im_start|>assistant\n<think>\n` and the model
/// continues from inside it. So the opening tag never reaches the parser, and a parser that
/// waits to see one files the whole answer as reasoning.
///
/// Thinking is a switch and not a level. The template expresses "off" by pre-filling an
/// **empty** block, `<think>\n\n</think>\n\n`, so the model writes its answer directly and the
/// tags are still present and well-formed. There is no equivalent of GPT-OSS's low, medium and
/// high, and offering one would be inventing behaviour the checkpoint does not have.
public struct QwenFormat: ConversationFormat {

    public init() {}

    public var name: String { "qwen-3.5" }

    /// `<|im_end|>` plus the newline and the next `<|im_start|>role`.
    public var reservedStopTokens: Int { 8 }

    public enum Marker: String, CaseIterable, Sendable {
        case start = "<|im_start|>"
        case end = "<|im_end|>"
        case thinkOpen = "<think>"
        case thinkClose = "</think>"
    }

    public func render(turns: [ChatTurn], settings: PromptSettings) -> String {
        var out = ""
        let instructions = settings.instructions?.trimmingCharacters(in: .whitespacesAndNewlines)
        if let instructions, !instructions.isEmpty {
            out += "\(Marker.start.rawValue)system\n\(instructions)\(Marker.end.rawValue)\n"
        }

        for turn in turns {
            let role = turn.role == .user ? "user" : "assistant"
            // Historical reasoning is dropped, which is what the template does for every turn
            // before the last query unless `preserve_thinking` is set. Replaying it would feed
            // the model its own discarded working as though it were the answer.
            let content = turn.role == .user ? turn.content : Self.strippingThinking(turn.content)
            out += "\(Marker.start.rawValue)\(role)\n\(content)\(Marker.end.rawValue)\n"
        }

        out += "\(Marker.start.rawValue)assistant\n"
        out += settings.reasoning == .off
            ? "\(Marker.thinkOpen.rawValue)\n\n\(Marker.thinkClose.rawValue)\n\n"
            : "\(Marker.thinkOpen.rawValue)\n"
        return out
    }

    /// Removes a `<think>…</think>` block, as the template does when replaying history.
    static func strippingThinking(_ content: String) -> String {
        guard let close = content.range(of: Marker.thinkClose.rawValue) else { return content }
        return String(content[close.upperBound...])
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    public func makeParser(
        tokenizer: BPETokenizer, settings: PromptSettings
    ) -> any ConversationParser {
        // Thinking on means the prompt already opened the block, so the parser starts inside it.
        // Off means the prompt opened *and closed* an empty one, so it starts outside.
        QwenParser(
            tokenizer: tokenizer, inThought: settings.reasoning != .off)
    }
}

/// Streams Qwen's output, separating reasoning from the answer.
final class QwenParser: ConversationParser {

    private let tokenizer: BPETokenizer
    private var inThought: Bool
    private var pending = ""
    private(set) var isFinished = false

    init(tokenizer: BPETokenizer, inThought: Bool) {
        self.tokenizer = tokenizer
        self.inThought = inThought
    }

    /// The longest marker, so a partial one is never emitted as text.
    ///
    /// A marker can straddle two tokens. Flushing eagerly would print `</thi` into the answer
    /// and then swallow the rest, which reads as the model producing stray characters rather
    /// than as a parser bug.
    private static let longestMarker = QwenFormat.Marker.allCases
        .map(\.rawValue.count).max() ?? 0

    func consume(_ token: Int) -> [PromptEvent] {
        guard !isFinished else { return [] }
        pending += tokenizer.decode([token])
        var events: [PromptEvent] = []

        while true {
            if let range = pending.range(of: QwenFormat.Marker.end.rawValue) {
                let text = String(pending[pending.startIndex..<range.lowerBound])
                if !text.isEmpty { events.append(inThought ? .reasoning(text) : .answer(text)) }
                pending = ""
                isFinished = true
                events.append(.stopped)
                return events
            }
            if inThought, let range = pending.range(of: QwenFormat.Marker.thinkClose.rawValue) {
                let text = String(pending[pending.startIndex..<range.lowerBound])
                if !text.isEmpty { events.append(.reasoning(text)) }
                pending = String(pending[range.upperBound...])
                inThought = false
                continue
            }
            break
        }

        // Hold back enough to recognize a marker split across tokens.
        if pending.count > Self.longestMarker {
            let keep = pending.index(pending.endIndex, offsetBy: -Self.longestMarker)
            let text = String(pending[pending.startIndex..<keep])
            if !text.isEmpty { events.append(inThought ? .reasoning(text) : .answer(text)) }
            pending = String(pending[keep...])
        }
        return events
    }
}
