import Foundation

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

        public init(role: Role, content: String) {
            self.role = role
            self.content = content
        }

        public static func user(_ text: String) -> Turn { Turn(role: .user, content: text) }
        public static func model(_ text: String) -> Turn { Turn(role: .model, content: text) }
    }

    // MARK: - Rendering

    public struct Renderer: Sendable {
        /// Whether the model is invited to reason. When false the generation prompt closes the
        /// thought channel immediately, which is the template's way of suppressing it.
        public var thinking: Bool
        public var instructions: String?

        public init(thinking: Bool = false, instructions: String? = nil) {
            self.thinking = thinking
            self.instructions = instructions
        }

        /// Renders the prompt, ready to encode with `allowSpecial: true`.
        ///
        /// Turn text is inserted **as-is**; it is the encoding that must refuse to interpret
        /// markers a user typed. The split is the same one Harmony uses: the renderer knows
        /// nothing of tokens, the encoder nothing of the format.
        public func render(turns: [Turn]) -> String {
            var out = Marker.beginningOfSequence.rawValue

            let trimmed = instructions?.trimmingCharacters(in: .whitespacesAndNewlines)
            if thinking || !(trimmed ?? "").isEmpty {
                out += "\(Marker.turnOpen.rawValue)\(Role.system.rawValue)\n"
                if thinking { out += "\(Marker.think.rawValue)\n" }
                if let trimmed, !trimmed.isEmpty { out += trimmed }
                out += "\(Marker.turnClose.rawValue)\n"
            }

            for turn in turns {
                out += "\(Marker.turnOpen.rawValue)\(turn.role.rawValue)\n"
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
                case .think, .turnOpen, .beginningOfSequence, .turnClose, .endOfSequence:
                    break
                }
                return events
            }

            let bytes = tokenizer.bytes(for: token)

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
