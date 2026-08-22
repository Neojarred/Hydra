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

    /// Every literal the parser must recognize, and nothing else.
    ///
    /// `longestMarker` is derived from this list, and the holdback that keeps a marker split
    /// across two tokens from being printed as text is derived from that. Adding a marker here
    /// widens the holdback automatically; writing one as a string literal somewhere else does
    /// not, which is how `</thi` ends up in an answer.
    ///
    /// The tool markers are **plain text, not special tokens**. Qwen's template writes them as
    /// ordinary characters, so they tokenize like any other text and straddle token boundaries
    /// freely — exactly the case M-071 was about.
    public enum Marker: String, CaseIterable, Sendable {
        case start = "<|im_start|>"
        case end = "<|im_end|>"
        case thinkOpen = "<think>"
        case thinkClose = "</think>"
        case toolCallOpen = "<tool_call>"
        case toolCallClose = "</tool_call>"
        case functionOpen = "<function="
        case parameterOpen = "<parameter="
        case parameterClose = "</parameter>"
        case toolResponseOpen = "<tool_response>"
        case toolResponseClose = "</tool_response>"
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

    public var supportsTools: Bool { true }

    public func render(turns: [ChatTurn], settings: PromptSettings) -> String {
        var out = ""
        let instructions = settings.instructions?.trimmingCharacters(in: .whitespacesAndNewlines)
        // With tools, the template folds the declaration and the user's instructions into one
        // system turn, the declaration first. Without them, the turn is what it always was:
        // this branch must render byte for byte what it rendered before tools existed, or every
        // conversation saved before today loses its cached prefix on its next turn.
        var system: [String] = []
        if !settings.tools.isEmpty { system.append(Self.toolDeclaration(settings.tools)) }
        if settings.searching || !settings.tools.isEmpty {
            system.append(Self.situation(today: settings.today ?? Self.today()))
        }
        if let instructions, !instructions.isEmpty { system.append(instructions) }
        if !system.isEmpty {
            out += "\(Marker.start.rawValue)system\n"
                + system.joined(separator: "\n\n") + "\(Marker.end.rawValue)\n"
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

        out += generationPrompt(settings)
        return out
    }

    /// The conversation, with **no** generation prompt: the prompt ready to be continued.
    ///
    /// Search needs the turn opened twice from the same point — once to ask the model for a
    /// query, once to hand it the results — and the recurrence's checkpoint sits exactly here,
    /// at the end of a `prefill`. Rendering the tail separately is what lets the second pass
    /// rewind to it instead of reprocessing the conversation.
    public func renderOpen(turns: [ChatTurn], settings: PromptSettings) -> String {
        let full = render(turns: turns, settings: settings)
        let tail = generationPrompt(settings)
        return full.hasSuffix(tail) ? String(full.dropLast(tail.count)) : full
    }

    public func generationPrompt(_ settings: PromptSettings) -> String {
        "\(Marker.start.rawValue)assistant\n" + Self.thinkingOpener(settings)
    }

    /// Asks for a search query and nothing else, **without thinking**.
    ///
    /// Thinking is off here for a measured reason, and it is the whole point of splitting the
    /// turn: on this model, deliberation is the failure surface. Six seeds, 1,200 tokens, the
    /// same prompt — thinking on degenerates on half of them and thinking off on none of them
    /// (M-077). A query is a dozen words, it needs no reasoning, and generating it in the mode
    /// that does not loop costs nothing.
    ///
    /// It is also where the model used to talk itself out of searching: given the choice, it
    /// spent a thousand tokens arguing about whether the thing in the question existed. It no
    /// longer has the choice. It is asked for a query, so it writes one.
    public func renderQueryRequest() -> String {
        Marker.start.rawValue + "user\n" + Self.queryInstruction + Marker.end.rawValue + "\n"
            + Marker.start.rawValue + "assistant\n"
            // Thinking closed explicitly, whatever the conversation's own setting is.
            + Marker.thinkOpen.rawValue + "\n\n" + Marker.thinkClose.rawValue + "\n\n"
    }

    static let queryInstruction = """
        Write the single best web search query for my question above. \
        Reply with the query itself and nothing else: no explanation, no quotes, no label. \
        Use the exact names, versions and numbers from my question, even if they are unfamiliar \
        to you — they are what I am asking about.
        """

    /// Hands the results back and opens the answer, from the same point the query was asked.
    public func renderSearchResults(_ text: String, settings: PromptSettings) -> String {
        Marker.start.rawValue + "user\n"
            + Marker.toolResponseOpen.rawValue + "\n" + text + "\n"
            + Marker.toolResponseClose.rawValue + "\n\n" + Self.answerInstruction
            + Marker.end.rawValue + "\n"
            + generationPrompt(settings)
    }

    /// The instruction that closes the results, immediately before the model answers.
    ///
    /// Deliberately **after** the results and not only in the system turn. The same words a
    /// thousand tokens earlier did not hold: handed a search result whose URL was literally
    /// `huggingface.co/Qwen/Qwen3.8-27B`, the model wrote "there is no evidence that a model
    /// named Qwen3.8 27B exists" and then cited that result in the next sentence. A prior that
    /// strong is not talked out of anything by a preamble it read before the evidence.
    ///
    /// It also **reframes the task**. Asked to answer, the model first decides whether the
    /// question is legitimate, and that decision is where it goes wrong. Asked to report what
    /// the sources say, there is nothing to adjudicate: the sources say what they say. The
    /// judgement it is bad at is the one removed.
    ///
    /// Outside the `<tool_response>` block, because inside it this would read as text from a
    /// web page, which the same paragraph tells the model to distrust as instruction.
    static let answerInstruction = """
        Now answer my question using these results. They were retrieved today and they are \
        genuine. Do not decide whether the models, products or versions they name are real: \
        they are, and your training predates them.

        No single result will answer the whole question. Combine what several of them say. \
        Where they cover only part of it, answer that part and name what is missing. Do not \
        refuse to answer because no source addresses the question directly, and do not say a \
        thing is absent from the results without first checking every one of them for it.
        """

    /// The `# Tools` system block, transcribed from `chat_template.jinja` rather than
    /// paraphrased.
    ///
    /// The wording is the checkpoint's own, down to the `<IMPORTANT>` reminder: this is the text
    /// Qwen was trained against, and an improved version of it is a dialect the model half
    /// recognizes. It costs about 250 tokens plus the schema, once, at the head of the prompt.
    static func toolDeclaration(_ tools: [ToolDefinition]) -> String {
        var out = "# Tools\n\nYou have access to the following functions:\n\n<tools>"
        for tool in tools { out += "\n" + schema(for: tool) }
        out += "\n</tools>"
        out += """
            \n\nIf you choose to call a function ONLY reply in the following format with NO \
            suffix:\n\n<tool_call>\n<function=example_function_name>\n\
            <parameter=example_parameter_1>\nvalue_1\n</parameter>\n\
            <parameter=example_parameter_2>\nThis is the value for the second parameter\n\
            that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n\
            <IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an \
            inner <function=...></function> block must be nested within \
            <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n\
            - You may provide optional reasoning for your function call in natural language \
            BEFORE the function call, but NOT after\n- If there is no function call \
            available, answer the question like normal with your current knowledge and do not \
            tell the user about function calls\n</IMPORTANT>
            """
        return out
    }

    /// What day it is, and what that means for what the model thinks it knows.
    ///
    /// **Qwen's template never states the date.** Harmony's does, which is why this only ever
    /// showed up here. Measured on the release this was written against: asked about a model
    /// published after its cutoff, Qwen 3.6 spent its entire reasoning budget arguing the thing
    /// could not exist — "as of early-mid 2024", in its own words — and then distrusted the
    /// search results that said otherwise. A model with no idea what year it is will defend its
    /// training data against evidence, and no amount of retrieved text fixes that.
    ///
    /// It sits in the system turn's free-text slot, which is exactly where the template puts a
    /// user's own system message, so nothing here invents a shape the checkpoint has not seen.
    /// It is rendered **only alongside tools**: without them the prompt must stay byte for byte
    /// what it was, and a line that changes at midnight would otherwise cost every conversation
    /// its cached prefix once a day for no benefit.
    static func situation(today: String) -> String {
        """
        Today's date is \(today). Your training data ends well before this, so anything you \
        believe about recent releases, versions, prices or events is probably out of date and \
        may be missing entirely.

        This application searches the web for you and puts the results in the conversation \
        before you answer. Those results are real pages retrieved today by the application, \
        not by the user and not invented: treat them as genuine even when they describe \
        something you have never heard of, and never dismiss them as fabricated. If a result \
        and your memory disagree about anything recent, the result is right. If a result \
        describes a product or version you do not recognise, it exists and your training simply \
        predates it. Answer from the results and say what they say.
        """
    }

    /// UTC, and formatted the way Harmony formats it, so the two models are told the same thing
    /// in the same words.
    static func today(_ now: Date = Date()) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd"
        formatter.timeZone = TimeZone(identifier: "UTC")
        formatter.locale = Locale(identifier: "en_US_POSIX")
        return formatter.string(from: now)
    }

    /// One tool as the JSON the template's `tool | tojson` would produce.
    ///
    /// Sorted keys, deliberately. The prompt is compared against the previous turn's token for
    /// token to decide what needs prefilling, so a declaration whose key order varies between
    /// two renders of the same tool would silently cost a full re-prefill.
    static func schema(for tool: ToolDefinition) -> String {
        var properties: [String: Any] = [:]
        for parameter in tool.parameters {
            properties[parameter.name] = [
                "type": parameter.type, "description": parameter.description,
            ]
        }
        let object: [String: Any] = [
            "type": "function",
            "function": [
                "name": tool.name,
                "description": tool.description,
                "parameters": [
                    "type": "object",
                    "properties": properties,
                    "required": tool.parameters.filter(\.required).map(\.name),
                ],
            ],
        ]
        guard let data = try? JSONSerialization.data(
            withJSONObject: object, options: [.sortedKeys, .withoutEscapingSlashes]),
            let text = String(data: data, encoding: .utf8)
        else { return "{}" }
        return text
    }

    /// A tool's output, and the markers that hand the turn back to the assistant.
    ///
    /// The template files a tool result as a **user** turn wrapping `<tool_response>`, which is
    /// why nothing here needed a new role. The assistant header that follows reopens thinking
    /// the same way `render` does, because the model is resuming mid-turn and a prompt that
    /// forgets to reopen the block leaves the parser filing the answer as reasoning.
    public func renderToolResult(_ text: String, settings: PromptSettings) -> String {
        Marker.end.rawValue + "\n"
            + Marker.start.rawValue + "user\n"
            + Marker.toolResponseOpen.rawValue + "\n" + text + "\n"
            + Marker.toolResponseClose.rawValue
            + Marker.end.rawValue + "\n"
            + Marker.start.rawValue + "assistant\n"
            + Self.thinkingOpener(settings)
    }

    /// What the template emits after `<|im_start|>assistant`, thinking on or off.
    ///
    /// Shared by `render` and `renderToolResult` because the checkpoint makes no distinction:
    /// a generation prompt is a generation prompt, and the one that resumes after a tool result
    /// opens the block exactly as the one that starts a turn does.
    static func thinkingOpener(_ settings: PromptSettings) -> String {
        settings.reasoning == .off
            ? "\(Marker.thinkOpen.rawValue)\n\n\(Marker.thinkClose.rawValue)\n\n"
            : "\(Marker.thinkOpen.rawValue)\n"
    }

    /// Reads a `<tool_call>` block into a call.
    ///
    /// Given the text **between** the markers, so the streaming parser can hold the block until
    /// it is whole and then hand it here in one piece. Returns `nil` on anything malformed: a
    /// call that cannot be read is a turn that should end without one, never a call with a
    /// guessed name.
    public static func parseToolCall(_ block: String) -> ToolCall? {
        guard let open = block.range(of: Marker.functionOpen.rawValue),
            let nameEnd = block[open.upperBound...].firstIndex(of: ">")
        else { return nil }
        let name = String(block[open.upperBound..<nameEnd])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else { return nil }

        var arguments: [String: String] = [:]
        var cursor = block.index(after: nameEnd)
        while let key = block.range(
            of: Marker.parameterOpen.rawValue, range: cursor..<block.endIndex)
        {
            guard let keyEnd = block[key.upperBound...].firstIndex(of: ">") else { break }
            let name = String(block[key.upperBound..<keyEnd])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let valueStart = block.index(after: keyEnd)
            guard let close = block.range(
                of: Marker.parameterClose.rawValue, range: valueStart..<block.endIndex)
            else { break }
            if !name.isEmpty {
                arguments[name] = String(block[valueStart..<close.lowerBound])
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            }
            cursor = close.upperBound
        }
        return ToolCall(name: name, arguments: arguments)
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
    /// Buffering a `<tool_call>` block rather than emitting text.
    private var inToolCall = false
    private(set) var isFinished = false

    init(tokenizer: BPETokenizer, inThought: Bool) {
        self.tokenizer = tokenizer
        self.inThought = inThought
    }

    /// Empties the tool-call buffer once the block is whole.
    ///
    /// A call arrives in fragments like any other text, and half of one is not a call: the
    /// parser holds the whole `<tool_call>…</tool_call>` and reads it in one piece rather than
    /// tracking `<function=` and `<parameter=` across token boundaries by hand. The block is
    /// short — a query and a count — so the buffering is bounded by the format itself.
    ///
    /// Returns no events while it is still filling, which is the caller's signal to emit
    /// nothing: text inside a call is the call's, not the answer's.
    private func drainToolCall() -> [PromptEvent] {
        if let close = pending.range(of: QwenFormat.Marker.toolCallClose.rawValue) {
            let block = String(pending[pending.startIndex..<close.lowerBound])
            finish()
            guard let call = QwenFormat.parseToolCall(block) else {
                // A block we could not read ends the turn without a call, which the engine
                // treats as an ordinary finish: a turn that says nothing beats one that calls
                // a function nobody named.
                return [.stopped]
            }
            return [.toolCall(call), .stopped]
        }
        // The turn ended mid-call. A truncated call is worse than no call, so what arrived is
        // discarded rather than parsed for whatever can be salvaged.
        if pending.range(of: QwenFormat.Marker.end.rawValue) != nil {
            finish()
            return [.stopped]
        }
        return []
    }

    private func finish() {
        pending = ""
        pendingBytes = []
        inToolCall = false
        isFinished = true
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

        if inToolCall { return drainToolCall() }

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
            // The template invites the model to explain itself in words before the call and
            // forbids it after, so whatever precedes the marker is ordinary answer text and is
            // emitted as such: "I will look that up" is worth showing while the search runs.
            if let range = pending.range(of: QwenFormat.Marker.toolCallOpen.rawValue) {
                let text = String(pending[pending.startIndex..<range.lowerBound])
                if !text.isEmpty { events.append(inThought ? .reasoning(text) : .answer(text)) }
                pending = String(pending[range.upperBound...])
                inToolCall = true
                return events + drainToolCall()
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
