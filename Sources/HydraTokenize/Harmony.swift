import Foundation

/// The **Harmony** conversation format, mandatory for GPT-OSS.
///
/// This is not a plain chat template. The model was trained to spread its answers across
/// **channels**: `analysis` carries the reasoning, `final` the answer meant for the user,
/// `commentary` the tool calls. Without the format the model does not know where to write
/// what, and the output is unusable.
///
/// The exact structure, taken from the published `chat_template.jinja`:
///
/// ```
/// <|start|>system<|message|>{identity}
/// Knowledge cutoff: 2024-06
/// Current date: {YYYY-MM-DD}
///
/// Reasoning: {low|medium|high}
///
/// # Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>
/// <|start|>developer<|message|># Instructions\n\n{instructions}\n\n<|end|>
/// <|start|>user<|message|>{text}<|end|>
/// <|start|>assistant<|channel|>final<|message|>{answer}<|end|>
/// <|start|>assistant
/// ```
///
/// **Reasoning from previous turns is never fed back** at inference: only the `final`
/// channel is kept in the history. The official template is explicit about this.
public enum Harmony {

    public enum ReasoningEffort: String, Sendable, CaseIterable {
        case low, medium, high
    }

    public enum Channel: String, Sendable {
        case analysis, commentary, final
    }

    public struct Turn: Sendable {
        public enum Role: String, Sendable { case user, assistant }
        public let role: Role
        public let content: String

        public init(role: Role, content: String) {
            self.role = role
            self.content = content
        }

        public static func user(_ text: String) -> Turn { Turn(role: .user, content: text) }
        public static func assistant(_ text: String) -> Turn { Turn(role: .assistant, content: text) }
    }

    // MARK: - Rendu

    public struct Renderer: Sendable {
        public var modelIdentity: String
        public var knowledgeCutoff: String
        public var currentDate: String
        public var reasoningEffort: ReasoningEffort
        /// The developer user's instructions, rendered in the `developer` message.
        public var instructions: String?

        public init(
            modelIdentity: String = "You are ChatGPT, a large language model trained by OpenAI.",
            knowledgeCutoff: String = "2024-06",
            currentDate: String? = nil,
            reasoningEffort: ReasoningEffort = .medium,
            instructions: String? = nil
        ) {
            self.modelIdentity = modelIdentity
            self.knowledgeCutoff = knowledgeCutoff
            if let currentDate {
                self.currentDate = currentDate
            } else {
                let formatter = DateFormatter()
                formatter.dateFormat = "yyyy-MM-dd"
                formatter.timeZone = TimeZone(identifier: "UTC")
                self.currentDate = formatter.string(from: Date())
            }
            self.reasoningEffort = reasoningEffort
            self.instructions = instructions
        }

        public func systemMessage() -> String {
            var text = modelIdentity + "\n"
            text += "Knowledge cutoff: \(knowledgeCutoff)\n"
            text += "Current date: \(currentDate)\n\n"
            text += "Reasoning: \(reasoningEffort.rawValue)\n\n"
            text += "# Valid channels: analysis, commentary, final. "
            text += "Channel must be included for every message."
            return text
        }

        /// Renders the complete prompt, ready to encode with `allowSpecial: true`.
        ///
        /// A turn's text is inserted **as-is**; it is the encoding that must refuse to interpret
        /// any markers it contains (`allowSpecial: false` for user content). Splitting the
        /// responsibilities is deliberate: the renderer knows nothing of tokens, the encoder
        /// nothing of the format.
        public func render(turns: [Turn]) -> String {
            var out = "<|start|>system<|message|>" + systemMessage() + "<|end|>"

            if let instructions, !instructions.isEmpty {
                out += "<|start|>developer<|message|># Instructions\n\n"
                out += instructions + "\n\n<|end|>"
            }

            for turn in turns {
                switch turn.role {
                case .user:
                    out += "<|start|>user<|message|>" + turn.content + "<|end|>"
                case .assistant:
                    // Reasoning from past turns is deliberately omitted.
                    out += "<|start|>assistant<|channel|>final<|message|>"
                    out += turn.content + "<|end|>"
                }
            }

            out += "<|start|>assistant"
            return out
        }
    }

    // MARK: - Control tokens

    /// Tokens that end a generation. `<|return|>` marks a turn's normal end; `<|call|>` signals
    /// a tool call; `<|endoftext|>` acts as a backstop.
    public static let stopTokenNames = ["<|return|>", "<|endoftext|>", "<|call|>"]

    public static func stopTokens(in tokenizer: BPETokenizer) -> Set<Int> {
        Set(stopTokenNames.compactMap { tokenizer.specialTokens[$0] })
    }

    // MARK: - Incremental output parsing

    /// Interprets the model's output token by token.
    ///
    /// The model typically produces:
    /// ```
    /// <|channel|>analysis<|message|>…raisonnement…<|end|>
    /// <|start|>assistant<|channel|>final<|message|>…answer…<|return|>
    /// ```
    /// The interface generally shows only `final`, but `analysis` must stay reachable — it is
    /// the chain of reasoning, and hiding it without capturing it would amount to losing it.
    ///
    ///
    /// Parsing is **incremental and byte-wise**: an accented character or an emoji may be spread
    /// over several tokens, so we never try to decode a token in isolation into text.
    ///
    public struct Parser: Sendable {

        public enum Event: Sendable, Equatable {
            /// Text produced on a channel. May arrive in fragments.
            case text(channel: Channel, String)
            /// A channel has just ended.
            case channelEnded(Channel)
            /// A stop token was encountered.
            case stopped(String)
        }

        enum State {
            /// After `<|start|>`: the model writes the role's name ("assistant"), which is not
            /// content. We absorb it up to `<|channel|>` or `<|message|>`.
            case readingRole
            /// Reading the channel's name after `<|channel|>`.
            case readingChannelName
            /// Accumulating a message's content.
            case readingContent(Channel)
        }

        private let tokenizer: BPETokenizer
        private let stops: Set<Int>

        public init(tokenizer: BPETokenizer) {
            self.tokenizer = tokenizer
            self.stops = Harmony.stopTokens(in: tokenizer)
        }

        /// The parse's mutable state, kept apart from the parser so it stays `Sendable`.
        public struct Session: Sendable {
            var state: State = .readingRole
            var pendingBytes: [UInt8] = []
            var channelNameBytes: [UInt8] = []
            /// Text accumulated per channel. The reasoning stays reachable even when the interface
            /// does not show it — hiding it without capturing it would lose it.
            public internal(set) var channels: [Channel: String] = [:]
            public internal(set) var isFinished = false

            public init() {}

            public var finalText: String { channels[.final] ?? "" }
            public var analysisText: String { channels[.analysis] ?? "" }
        }

        /// Consumes a token and returns the events produced.
        public func consume(_ token: Int, session: inout Session) -> [Event] {
            guard !session.isFinished else { return [] }

            if stops.contains(token) {
                let name = tokenizer.name(of: token) ?? "<|stop|>"
                var events = flush(&session)
                session.isFinished = true
                events.append(.stopped(name))
                return events
            }

            if let special = tokenizer.name(of: token) {
                return handleSpecial(special, session: &session)
            }

            let bytes = tokenizer.bytes(for: token)
            switch session.state {
            case .readingChannelName:
                session.channelNameBytes.append(contentsOf: bytes)
                return []
            case .readingContent(let channel):
                session.pendingBytes.append(contentsOf: bytes)
                // We only emit bytes that form complete UTF-8: otherwise an accent split across two
                // tokens would produce a replacement character.
                guard let text = decodeComplete(&session.pendingBytes), !text.isEmpty else {
                    return []
                }
                session.channels[channel, default: ""] += text
                return [.text(channel: channel, text)]
            case .readingRole:
                // The role's name is written by the model after `<|start|>`; it is not part of
                // the answer. Without this case, "assistant" appeared at the head of every
                // message displayed.
                return []
            }
        }

        private func handleSpecial(_ name: String, session: inout Session) -> [Event] {
            switch name {
            case "<|channel|>":
                session.channelNameBytes = []
                session.state = .readingChannelName
                return []
            case "<|message|>":
                let raw = String(decoding: session.channelNameBytes, as: UTF8.self)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                let channel = Channel(rawValue: raw) ?? .final
                session.channelNameBytes = []
                session.state = .readingContent(channel)
                return []
            case "<|end|>", "<|start|>":
                let events = flush(&session)
                session.state = .readingRole
                return events
            default:
                return []
            }
        }

        private func flush(_ session: inout Session) -> [Event] {
            var events: [Event] = []
            if case .readingContent(let channel) = session.state {
                if !session.pendingBytes.isEmpty {
                    let text = String(decoding: session.pendingBytes, as: UTF8.self)
                    session.pendingBytes = []
                    if !text.isEmpty {
                        session.channels[channel, default: ""] += text
                        events.append(.text(channel: channel, text))
                    }
                }
                events.append(.channelEnded(channel))
            }
            return events
        }

        /// Extracts the complete UTF-8 prefix of the pending bytes, and removes it from the buffer.
        private func decodeComplete(_ buffer: inout [UInt8]) -> String? {
            guard !buffer.isEmpty else { return nil }
            // Walk back to a character boundary: a continuation byte is 10xxxxxx, a lead byte
            // begins a sequence.
            var end = buffer.count
            var back = 0
            while end > 0, back < 4 {
                let byte = buffer[end - 1]
                if byte & 0b1100_0000 != 0b1000_0000 {
                    // Lead byte: the sequence is complete if its length fits.
                    let needed: Int
                    if byte & 0b1000_0000 == 0 { needed = 1 }
                    else if byte & 0b1110_0000 == 0b1100_0000 { needed = 2 }
                    else if byte & 0b1111_0000 == 0b1110_0000 { needed = 3 }
                    else { needed = 4 }
                    if back + 1 >= needed { end = buffer.count }
                    else { end -= 1 }
                    break
                }
                end -= 1
                back += 1
            }
            guard end > 0 else { return nil }
            let complete = Array(buffer[0..<end])
            buffer.removeFirst(end)
            return String(decoding: complete, as: UTF8.self)
        }
    }
}
