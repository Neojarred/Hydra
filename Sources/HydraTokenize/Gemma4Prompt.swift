import Foundation
import HydraCore

/// Gemma 4's conversation format, transcribed from the published `chat_template.jinja` of
/// `google/gemma-4-26B-A4B-it`.
///
/// **Not the `<start_of_turn>` format of Gemma 2 and 3.** Gemma 4 introduced paired markers,
/// `<|turn>role … <turn|>`, and a separate thought channel. Carrying the older format over
/// would produce a prompt the model has never seen, which degrades answers without failing.
///
/// The shape, for a conversation without tools:
///
/// ```
/// <bos>
/// <|turn>system
/// {instructions}<turn|>
/// <|turn>user
/// {text}<turn|>
/// <|turn>model
/// {answer}<turn|>
/// <|turn>model
/// ```
///
/// Two details are easy to miss and both come straight from the template:
///
/// - the assistant's role is written **`model`**, not `assistant`;
/// - when thinking is **disabled**, the generation prompt emits an already-closed thought
///   channel, `<|channel>thought\n<channel|>`. That is not decoration: pre-closing the channel
///   is how the template suppresses reasoning. Omitting it asks a thinking model to think.
public enum Gemma4Prompt {

    /// The markers, with the identifiers the published tokenizer assigns them. The names are
    /// resolved through the tokenizer rather than hardcoded; the values are recorded here so a
    /// mismatch is visible rather than mysterious.
    public enum Marker: String, CaseIterable, Sendable {
        case beginningOfSequence = "<bos>"
        case endOfSequence = "<eos>"
        case turnOpen = "<|turn>"
        case turnClose = "<turn|>"
        case think = "<|think|>"
        case channelOpen = "<|channel>"
        case channelClose = "<channel|>"
        /// The tool markers, which are **single special tokens** and not text.
        ///
        /// The difference from Qwen matters more than it looks. Qwen writes `<tool_call>` as
        /// ordinary characters, so it arrives split across token boundaries at the tokenizer's
        /// convenience and its parser needs a holdback the width of the longest marker to avoid
        /// printing half of one. Gemma's are ids: 48 is `<|tool_call>` and nothing else ever is,
        /// so the parser matches on identity and the whole class of straddling bug does not
        /// exist here.
        case toolOpen = "<|tool>"
        case toolClose = "<tool|>"
        case toolCallOpen = "<|tool_call>"
        case toolCallClose = "<tool_call|>"
        case toolResponseOpen = "<|tool_response>"
        case toolResponseClose = "<tool_response|>"
        /// The template's own string delimiter, used inside declarations and arguments.
        case quote = "<|\"|>"

        /// What `google/gemma-4-26B-A4B` assigns, for reference and for tests.
        public var publishedID: Int {
            switch self {
            case .beginningOfSequence: return 2
            case .endOfSequence: return 1
            case .think: return 98
            case .channelOpen: return 100
            case .channelClose: return 101
            case .turnOpen: return 105
            case .turnClose: return 106
            case .toolOpen: return 46
            case .toolClose: return 47
            case .toolCallOpen: return 48
            case .toolCallClose: return 49
            case .toolResponseOpen: return 50
            case .toolResponseClose: return 51
            case .quote: return 52
            }
        }
    }

    public enum Role: String, Sendable {
        case system
        case user
        /// The template writes `model`, never `assistant`.
        case model
    }

    public struct Turn: Sendable, Equatable {
        public let role: Role
        public let content: String
        /// Pictures attached to this turn, rendered as placeholders ahead of the words.
        public let images: Int

        public init(role: Role, content: String, images: Int = 0) {
            self.role = role
            self.content = content
            self.images = images
        }

        public static func user(_ text: String, images: Int = 0) -> Turn {
            Turn(role: .user, content: text, images: images)
        }
        public static func model(_ text: String) -> Turn { Turn(role: .model, content: text) }
    }

    /// How an image appears in Gemma's prompt.
    ///
    /// **One `<|image|>` a picture and nothing around it.** Gemma's own chat template emits
    /// exactly that; the processor then expands it into one copy per soft token and the model
    /// swaps their embeddings. Here the tower's output is spliced at the placeholder's position,
    /// so the count never enters the string, and the count is not fixed anyway: it varies with
    /// the image's shape.
    ///
    /// The checkpoint also carries `<|image>` and `<image|>` brackets, 255999 and 258882, which
    /// the template does not use. They are left alone rather than added on the theory that a
    /// bracket must be needed.
    public enum Vision: String, Sendable {
        case placeholder = "<|image|>"
    }

    /// A run of the prompt, before the runner turns it into embeddings.
    public enum PromptPiece: Sendable, Equatable {
        case text([Int])
        case image(index: Int)
    }

    /// Splits a tokenized prompt at each image placeholder.
    ///
    /// The placeholder is dropped: it stands in for embeddings arriving from the tower, and
    /// keeping it would feed the model a spare token meaning "an image goes here" inside the
    /// image it introduces. A count mismatch returns `nil` rather than splicing one picture
    /// twice.
    public static func split(
        tokens: [Int], atPlaceholder placeholder: Int, images: Int
    ) -> [PromptPiece]? {
        var pieces: [PromptPiece] = []
        var run: [Int] = []
        var seen = 0
        for token in tokens {
            if token == placeholder {
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

    // MARK: - Rendering

    public struct Renderer: Sendable {
        /// Whether the model is invited to reason. When false the generation prompt closes the
        /// thought channel immediately, which is the template's way of suppressing it.
        public var thinking: Bool
        /// Functions the model may call. Empty renders byte for byte what it always did.
        public var tools: [ToolDefinition] = []
        /// The date to state, `yyyy-MM-dd`, when this turn may read the web.
        ///
        /// Gemma's template states no date, exactly as Qwen's does not. Without it the model
        /// answers from its cutoff and says so: asked about a model published after it, one run
        /// opened "As of early 2025" while reading results retrieved today. The same fix, in
        /// the other renderer, because it was written for Qwen and never carried across.
        public var today: String?
        public var instructions: String?

        public init(
            thinking: Bool = false, instructions: String? = nil,
            tools: [ToolDefinition] = [], today: String? = nil
        ) {
            self.thinking = thinking
            self.instructions = instructions
            self.tools = tools
            self.today = today
        }

        /// Renders the prompt, ready to encode with `allowSpecial: true`.
        ///
        /// Turn text is inserted **as-is**; it is the encoding that must refuse to interpret
        /// markers a user typed. The split is the same one Harmony uses: the renderer knows
        /// nothing of tokens, the encoder nothing of the format.
        public func render(turns: [Turn]) -> String {
            var out = Marker.beginningOfSequence.rawValue

            let trimmed = instructions?.trimmingCharacters(in: .whitespacesAndNewlines)
            if thinking || !(trimmed ?? "").isEmpty || !tools.isEmpty {
                out += "\(Marker.turnOpen.rawValue)\(Role.system.rawValue)\n"
                // `<|think|>` goes at the very top of the first system turn, before anything
                // else. That is the template's own ordering, and it is also the head of the
                // prompt: turning thinking on or off moves every token after it.
                if thinking { out += "\(Marker.think.rawValue)\n" }
                if let trimmed, !trimmed.isEmpty { out += trimmed }
                if let today, !tools.isEmpty {
                    if trimmed?.isEmpty == false { out += "\n\n" }
                    out += Gemma4Prompt.situation(today: today)
                }
                // Declarations follow the instructions, each in its own `<|tool>` block.
                for tool in tools {
                    out += Marker.toolOpen.rawValue
                        + Gemma4Prompt.declaration(for: tool)
                        + Marker.toolClose.rawValue
                }
                out += "\(Marker.turnClose.rawValue)\n"
            }

            for turn in turns {
                out += "\(Marker.turnOpen.rawValue)\(turn.role.rawValue)\n"
                // Images lead the turn, which is where the template puts them.
                if turn.role == .user, turn.images > 0 {
                    out += String(repeating: Vision.placeholder.rawValue, count: turn.images)
                }
                // Past reasoning is never fed back: the template strips it from model turns,
                // exactly as Harmony drops the analysis channel.
                out += turn.content.trimmingCharacters(in: .whitespacesAndNewlines)
                out += Marker.turnClose.rawValue + "\n"
            }

            out += "\(Marker.turnOpen.rawValue)\(Role.model.rawValue)\n"
            // **The thought channel is opened by the prompt either way.**
            //
            // Closed immediately when thinking is off, that is the template's way of
            // suppressing reasoning, and it is what the published `chat_template.jinja`
            // writes.
            //
            // Left open when thinking is on, which the template does *not* write: it stops at
            // `<|turn>model` and expects the model to produce `<|channel>thought` itself.
            // Measured, it does not. Asked what follows a bare `<|channel>`, the model's best
            // token is « eyes» at a logit of 5.1, a flat, unconfident distribution with
            // "thought" nowhere near it. Seed the header and the same question gives «The» at
            // 26.7. So we write it, which is also what the template's own tool-response branch
            // does, and the reasoning arrives inside a channel instead of leaking into the
            // answer.
            out += "\(Marker.channelOpen.rawValue)thought\n"
            if !thinking { out += Marker.channelClose.rawValue }
            return out
        }
    }

    // MARK: - Tools

    /// What day it is, and what that means for what the model believes.
    ///
    /// The same words Qwen is given, for the same reason and with the same evidence behind
    /// them: a model that does not know the year defends its training data against the pages
    /// in front of it.
    public static func situation(today: String) -> String {
        """
        Today's date is \(today). Your training data ends well before this, so anything you \
        believe about recent releases, versions, prices or events is probably out of date and \
        may be missing entirely. When a search result and your own memory disagree about \
        something recent, the result is right. If a result describes a product or version you \
        do not recognise, it exists and your training simply predates it.
        """
    }

    /// UTC, formatted as Harmony and Qwen format it, so all three are told the same thing in
    /// the same words.
    public static func today(_ now: Date = Date()) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd"
        formatter.timeZone = TimeZone(identifier: "UTC")
        formatter.locale = Locale(identifier: "en_US_POSIX")
        return formatter.string(from: now)
    }

    /// One function, in the shape `format_function_declaration` writes.
    ///
    /// Not JSON. Gemma's declarations are a bespoke key/value syntax whose strings are wrapped
    /// in `<|"|>`, a real token rather than a quote character, and whose type names are upper
    /// case. Transcribed rather than approximated: this is the text the checkpoint was trained
    /// against, and a declaration in a dialect it half-recognizes is the failure that raises
    /// nothing (D-023).
    public static func declaration(for tool: ToolDefinition) -> String {
        func quoted(_ text: String) -> String {
            Marker.quote.rawValue + text + Marker.quote.rawValue
        }
        // `dictsort` in the template, so the properties are alphabetical. A declaration whose
        // order wandered between two renders of the same tool would cost the conversation its
        // cached prefix and nothing would say why.
        let sorted = tool.parameters.sorted { $0.name < $1.name }
        let properties = sorted.map { parameter in
            "\(parameter.name):{description:\(quoted(parameter.description)),"
                + "type:\(quoted(parameter.type.uppercased()))}"
        }.joined(separator: ",")
        let required = sorted.filter(\.required).map { quoted($0.name) }.joined(separator: ",")

        var out = "declaration:\(tool.name){description:\(quoted(tool.description))"
        out += ",parameters:{properties:{\(properties)}"
        if !required.isEmpty { out += ",required:[\(required)]" }
        out += ",type:\(quoted("OBJECT"))}}"
        return out
    }

    /// A tool's result, in the block the template writes, continuing the model's own turn.
    ///
    /// **No new turn.** Gemma's template places the response immediately after the call inside
    /// the same `<|turn>model`, and `add_generation_prompt` deliberately emits nothing when the
    /// previous thing was a call or a response. The model simply carries on. Qwen files the
    /// same thing as a user turn and has to reopen the assistant; this does not.
    public static func toolResponse(name: String, value: String) -> String {
        Marker.toolResponseOpen.rawValue
            + "response:\(name){value:\(value)}"
            + Marker.toolResponseClose.rawValue
    }

    /// Reads a `call:name{key:value,...}` block into a call.
    ///
    /// Given the text between the two markers, so the streaming parser can hand it over whole.
    /// Returns `nil` on anything it cannot read: a turn that ends without a call beats one that
    /// calls a function nobody named.
    public static func parseToolCall(_ block: String) -> ToolCall? {
        let body = block.trimmingCharacters(in: .whitespacesAndNewlines)
        guard body.hasPrefix("call:"), let brace = body.firstIndex(of: "{"),
            body.hasSuffix("}")
        else { return nil }
        let name = String(body[body.index(body.startIndex, offsetBy: 5)..<brace])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else { return nil }

        let inner = String(body[body.index(after: brace)..<body.index(before: body.endIndex)])
        var arguments: [String: String] = [:]
        for field in Self.splitFields(inner) {
            guard let colon = field.firstIndex(of: ":") else { continue }
            let key = String(field[field.startIndex..<colon])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            var value = String(field[field.index(after: colon)...])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            // Strings arrive wrapped in the `<|"|>` token, which is not a quote character and
            // will not be stripped by anything that thinks it is.
            let quote = Marker.quote.rawValue
            if value.hasPrefix(quote), value.hasSuffix(quote), value.count > quote.count * 2 {
                value = String(value.dropFirst(quote.count).dropLast(quote.count))
            }
            if !key.isEmpty { arguments[key] = value }
        }
        return ToolCall(name: name, arguments: arguments)
    }

    /// Splits `a:1,b:{c:2},d:[<|"|>x, y<|"|>]` on the commas that separate fields, not the ones
    /// inside a value.
    ///
    /// Depth **and** quoting. Tracking only braces was enough until a test asked for a query
    /// with a comma in it, which is an ordinary thing to search for: `a query, with a comma`
    /// came back as `a query` and the rest became a field named nothing. The quote is `<|"|>`,
    /// a five-character token rather than a quote character, so nothing that scans for `"`
    /// would have found it.
    static func splitFields(_ text: String) -> [String] {
        let quote = Array(Marker.quote.rawValue)
        let characters = Array(text)
        var fields: [String] = []
        var current: [Character] = []
        var depth = 0
        var quoted = false
        var index = 0

        while index < characters.count {
            // A quote token opens or closes a string, and is kept: the caller strips a matched
            // pair, and an unmatched one is evidence the block was malformed.
            if index + quote.count <= characters.count,
                Array(characters[index..<(index + quote.count)]) == quote
            {
                quoted.toggle()
                current.append(contentsOf: quote)
                index += quote.count
                continue
            }
            let character = characters[index]
            index += 1
            if !quoted {
                switch character {
                case "{", "[": depth += 1
                case "}", "]": depth -= 1
                case "," where depth == 0:
                    fields.append(String(current))
                    current.removeAll(keepingCapacity: true)
                    continue
                default: break
                }
            }
            current.append(character)
        }
        if !current.isEmpty { fields.append(String(current)) }
        return fields
    }

    // MARK: - Parsing

    /// Interprets the model's output token by token.
    ///
    /// Byte-wise and incremental, for the reason Harmony's parser is: a multi-byte character
    /// can be spread over several tokens, so a token is never decoded to text on its own.
    ///
    /// Reasoning arrives inside `<|channel>thought … <channel|>` and is kept separate from the
    /// answer, visible to the interface if it wants it, never mistaken for the reply.
    public struct Parser: Sendable {
        private let tokenizer: BPETokenizer
        private let markers: [Int: Marker]
        private let stopIDs: Set<Int>

        public init(tokenizer: BPETokenizer) {
            self.tokenizer = tokenizer
            var found: [Int: Marker] = [:]
            for marker in Marker.allCases {
                if let id = tokenizer.id(of: marker.rawValue) { found[id] = marker }
            }
            self.markers = found
            self.stopIDs = Set(
                [Marker.turnClose, .endOfSequence].compactMap { tokenizer.id(of: $0.rawValue) })
        }

        public enum Event: Sendable, Equatable {
            /// The model asked for a function to be run. Emitted before the `.stopped` that
            /// follows it, as Qwen's is: a consumer that stopped reading at `.stopped` would
            /// finish the turn holding a request it never noticed.
            case toolCall(ToolCall)
            case text(String)
            case reasoning(String)
            case stopped
        }

        public struct Session: Sendable {
            var pending: [UInt8] = []
            var inThought = false
            /// Set while consuming the channel's name, which is content the user never sees.
            var readingChannelName = false
            /// The name accumulated so far.
            ///
            /// Buffered across tokens rather than read from one: "thought" is not a single
            /// piece in every vocabulary, and checking one token for the terminating newline
            /// silently produced an empty name, so the thought channel never opened and the
            /// reasoning went into the answer.
            var channelName: [UInt8] = []

            /// How far the parser will look for the newline that ends a channel name.
            ///
            /// Generous next to the only name the template writes, `thought`, and small
            /// next to a generation. It exists so that a malformed channel costs a few
            /// discarded characters rather than the entire turn.
            static let channelNameLimit = 64
            /// Settable within the module so the parser can advance it; read-only outside,
            /// because the session is the caller's record of what happened and nothing else
            /// should rewrite it.
            public internal(set) var isFinished = false
            /// Accumulating a `<|tool_call>` block rather than emitting text.
            var inToolCall = false
            var callBytes: [UInt8] = []
            public internal(set) var answer = ""
            public internal(set) var reasoning = ""

            public init() {}

            /// The state the prompt left the parser in.
            ///
            /// The renderer opens `<|channel>thought` itself, so those tokens are in the
            /// prompt and the parser never sees them. Without being told, it would start
            /// outside any channel and file the whole of the reasoning as the answer.
            public init(inThought: Bool) {
                self.inThought = inThought
            }
        }

        public func consume(_ token: Int, session: inout Session) -> [Event] {
            guard !session.isFinished else { return [] }

            if stopIDs.contains(token) {
                var events = flush(&session)
                session.isFinished = true
                events.append(.stopped)
                return events
            }

            if let marker = markers[token] {
                var events = flush(&session)
                switch marker {
                case .channelOpen:
                    session.readingChannelName = true
                    session.channelName.removeAll(keepingCapacity: true)
                case .channelClose:
                    session.inThought = false
                    session.readingChannelName = false
                case .toolCallOpen:
                    // Everything until the closing marker belongs to the call, not the answer.
                    session.inToolCall = true
                    session.callBytes.removeAll(keepingCapacity: true)
                case .toolCallClose:
                    session.inToolCall = false
                    let block = String(decoding: session.callBytes, as: UTF8.self)
                    session.callBytes.removeAll(keepingCapacity: true)
                    // A call that cannot be read ends the turn without one. Running a search
                    // for a half-parsed name is worse than not running one.
                    if let call = Gemma4Prompt.parseToolCall(block) {
                        events.append(.toolCall(call))
                    }
                    session.isFinished = true
                    events.append(.stopped)
                case .think, .turnOpen, .beginningOfSequence, .turnClose, .endOfSequence,
                    .toolOpen, .toolClose, .toolResponseOpen, .toolResponseClose, .quote:
                    break
                }
                return events
            }

            let bytes = tokenizer.bytes(for: token)

            if session.inToolCall {
                session.callBytes.append(contentsOf: bytes)
                return []
            }

            // The channel's name follows `<|channel>` and is not content. It ends at the first
            // newline; without this, "thought" appears at the head of the reasoning.
            if session.readingChannelName {
                guard let newline = bytes.firstIndex(of: 0x0A) else {
                    session.channelName.append(contentsOf: bytes)
                    // **The name must terminate.** The template only ever writes
                    // `<|channel>thought\n`, but the model is sampling, not reciting: it can
                    // emit `<|channel>` and then never a newline. Without this bound the
                    // parser accumulates for the rest of the generation and returns no events
                    // at all, the interface sits on "thinking" while hundreds of tokens are
                    // produced and discarded, then ends with nothing. Observed, not
                    // theorised.
                    //
                    // Past the bound we give up on naming the channel and treat what was
                    // swallowed as ordinary content, which is the reading that loses least.
                    if session.channelName.count > Session.channelNameLimit {
                        session.readingChannelName = false
                        session.inThought = false
                        session.pending.append(contentsOf: session.channelName)
                        session.channelName.removeAll(keepingCapacity: true)
                        return flush(&session)
                    }
                    return []
                }
                session.channelName.append(contentsOf: bytes[..<newline])
                let name = String(decoding: session.channelName, as: UTF8.self)
                session.inThought = name.contains("thought")
                session.readingChannelName = false
                session.channelName.removeAll(keepingCapacity: true)
                session.pending.append(contentsOf: bytes[(newline + 1)...])
                return flush(&session)
            }

            session.pending.append(contentsOf: bytes)
            return flush(&session)
        }

        /// Emits only the complete UTF-8 prefix of what is buffered: otherwise an accent split
        /// across two tokens would surface as a replacement character.
        private func flush(_ session: inout Session) -> [Event] {
            let boundary = completeUTF8Prefix(session.pending)
            guard boundary > 0 else { return [] }
            let fragment = String(decoding: session.pending[..<boundary], as: UTF8.self)
            session.pending.removeFirst(boundary)
            guard !fragment.isEmpty else { return [] }

            if session.inThought {
                session.reasoning += fragment
                return [.reasoning(fragment)]
            }
            session.answer += fragment
            return [.text(fragment)]
        }

        private func completeUTF8Prefix(_ bytes: [UInt8]) -> Int {
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
    }

    /// Tokens that end a generation.
    public static func stopTokens(in tokenizer: BPETokenizer) -> Set<Int> {
        Set([Marker.turnClose, .endOfSequence].compactMap { tokenizer.id(of: $0.rawValue) })
    }
}
