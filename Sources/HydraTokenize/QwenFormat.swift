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

    /// What an image looks like in the rendered prompt.
    ///
    /// One `<|image_pad|>` a picture, not one a token. The published processor expands the pad
    /// into as many copies as the image has tokens and then swaps their embeddings out; here the
    /// tower's output is spliced in at the pad's position instead, so the count never has to be
    /// written into the string at all. The two brackets are real tokens either way and the model
    /// expects them.
    public enum Vision: String, Sendable {
        case start = "<|vision_start|>"
        case pad = "<|image_pad|>"
        case end = "<|vision_end|>"

        public static var placeholder: String {
            "\(Vision.start.rawValue)\(Vision.pad.rawValue)\(Vision.end.rawValue)"
        }
    }

    /// Splits a tokenized prompt at each image pad, so the runner can be handed text runs and
    /// images in order.
    ///
    /// The pad token is dropped rather than kept: it is a placeholder for embeddings that are
    /// arriving from somewhere else, and leaving it in would feed the model a spare token whose
    /// embedding means "an image goes here" in the middle of the image it introduces.
    ///
    /// Returns `nil` if the number of pads does not match the number of images, which is a
    /// programming error rather than a user one and must not be papered over: a prompt with two
    /// pads and one image would otherwise splice the same picture twice.
    public static func split(
        tokens: [Int], atImagePad pad: Int, images: Int
    ) -> [PromptPiece]? {
        var pieces: [PromptPiece] = []
        var run: [Int] = []
        var seen = 0
        for token in tokens {
            if token == pad {
                if !run.isEmpty { pieces.append(.text(run)); run = [] }
                pieces.append(.image(index: seen))
                seen += 1
            } else {
                run.append(token)
            }
        }
        if !run.isEmpty { pieces.append(.text(run)) }
        return seen == images ? pieces : nil
    }

    /// A run of the prompt, before the runner turns it into embeddings.
    public enum PromptPiece: Sendable, Equatable {
        case text([Int])
        /// The nth image of the message, in the order they were attached.
        case image(index: Int)
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
            // Images lead the turn, before the words, which is where the published template puts
            // them: a question almost always refers back to the picture it follows.
            let pictures = String(
                repeating: Vision.placeholder, count: turn.role == .user ? turn.images : 0)
            out += "\(Marker.start.rawValue)\(role)\n\(pictures)\(content)"
                + "\(Marker.end.rawValue)\n"
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
    /// Bytes not yet decoded, and the text decoded from them but not yet emitted.
    ///
    /// **Bytes, not a string.** This parser used to do `pending += tokenizer.decode([token])`,
    /// decoding each token's bytes on their own. Byte-level BPE splits a character across
    /// tokens whenever it feels like it, and every emoji Qwen writes is four bytes, so each
    /// fragment decoded alone became replacement characters and the bytes were gone: `U+FFFD`
    /// is not reversible. Harmony and Gemma's parsers have always accumulated bytes and decoded
    /// at a character boundary; this one was the exception, which is why only Qwen produced the
    /// black diamonds.
    private var pendingBytes: [UInt8] = []
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

    /// How many leading bytes form whole characters.
    ///
    /// Anything after that is the start of a character whose remaining bytes are in the next
    /// token, and it waits there rather than being decoded into a replacement character.
    private static func completeUTF8Prefix(_ bytes: [UInt8]) -> Int {
        var end = bytes.count
        var scanned = 0
        while end > 0, scanned < 4 {
            let byte = bytes[end - 1]
            if byte & 0b1100_0000 == 0b1000_0000 {
                end -= 1
                scanned += 1
                continue
            }
            let length: Int
            if byte & 0b1000_0000 == 0 { length = 1 }
            else if byte & 0b1110_0000 == 0b1100_0000 { length = 2 }
            else if byte & 0b1111_0000 == 0b1110_0000 { length = 3 }
            else { length = 4 }
            return end - 1 + length <= bytes.count ? bytes.count : end - 1
        }
        return end == bytes.count ? bytes.count : end
    }

    func consume(_ token: Int) -> [PromptEvent] {
        guard !isFinished else { return [] }
        // A special token is a whole marker and never part of a character, so it decodes on its
        // own; anything else joins the byte buffer.
        if tokenizer.isSpecial(token) {
            pending += tokenizer.decode([token])
        } else {
            pendingBytes += tokenizer.bytes(for: token)
            let boundary = Self.completeUTF8Prefix(pendingBytes)
            if boundary > 0 {
                pending += String(decoding: pendingBytes[..<boundary], as: UTF8.self)
                pendingBytes.removeFirst(boundary)
            }
        }
        var events: [PromptEvent] = []

        while true {
            if let range = pending.range(of: QwenFormat.Marker.end.rawValue) {
                let text = String(pending[pending.startIndex..<range.lowerBound])
                if !text.isEmpty { events.append(inThought ? .reasoning(text) : .answer(text)) }
                pending = ""
                pendingBytes = []
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
