import Foundation
import HydraSearch
import HydraTokenize

/// One generation. An assistant message can hold several: regenerating, or editing the
/// question above it, adds a variant instead of overwriting the previous one.
public struct Variant: Identifiable, Codable, Sendable, Equatable {
    public var id: UUID = UUID()
    public var text: String = ""
    public var reasoning: String = ""
    /// Tokens produced, time to first token, throughput.
    public var outputTokens: Int?
    public var timeToFirstToken: Double?
    /// Prompt tokens that were not already in the cache. What the wait scales with.
    public var newPromptTokens: Int?
    public var tokensPerSecond: Double?
    /// Searches this answer ran, in order.
    ///
    /// Optional so every conversation saved before search existed decodes unchanged, which is
    /// how images were added and for the same reason.
    public var searches: [Search]?

    public init(text: String = "", reasoning: String = "") {
        self.text = text
        self.reasoning = reasoning
    }
}

/// One search an answer ran, kept so the sources survive a relaunch.
///
/// The sources are stored and the **snippets are not**. What the model read was worth 900
/// tokens of prompt; what the user needs afterwards is where it came from. Keeping the text
/// would triple the size of `conversations.json` to preserve something nothing displays.
public struct Search: Identifiable, Codable, Sendable, Equatable {
    public var id: UUID = UUID()
    public var query: String
    public var sources: [Source]
    /// Results the provider returned that did not fit the token budget. Shown, not hidden:
    /// the user is entitled to know the answer was written from five pages and not eight.
    public var dropped: Int
    /// What the block cost to read, which is most of what the turn's wait was.
    public var tokens: Int

    public struct Source: Identifiable, Codable, Sendable, Equatable {
        public var id: UUID = UUID()
        public var title: String
        public var url: String

        public init(title: String, url: String) {
            self.title = title
            self.url = url
        }
    }

    public init(query: String, sources: [Source], dropped: Int, tokens: Int) {
        self.query = query
        self.sources = sources
        self.dropped = dropped
        self.tokens = tokens
    }
}

/// A displayed message, with its reasoning kept separate.
///
/// The `analysis` channel is **stored but separate**: GPT-OSS's official template never
/// feeds it back into the history at inference time, so it must not mix with the answer.
/// Hiding it without keeping it would amount to losing it.
public struct Message: Identifiable, Codable, Sendable, Equatable {
    public enum Role: String, Codable, Sendable { case user, assistant }

    public var id: UUID
    public var role: Role
    public var date: Date
    /// Always at least one variant. User messages have exactly one.
    public var variants: [Variant]
    public var activeVariant: Int
    /// Files attached by the user, inserted into the prompt.
    public var attachments: [Attachment]

    public struct Attachment: Identifiable, Codable, Sendable, Equatable {
        public var id: UUID = UUID()
        public var name: String
        /// The text of a document. Empty for a picture, whose content is its pixels.
        public var content: String
        public var byteCount: Int
        /// Where a picture lives on disk, and what it will cost.
        ///
        /// Optional so that every conversation saved before images existed decodes unchanged,
        /// and so a document and a picture stay one type: the chip, the byte count and the
        /// removal all work the same way and only the rendering differs.
        public var image: Image?

        public struct Image: Codable, Sendable, Equatable {
            public var path: String
            /// Text tokens the picture will occupy, from the header alone. Shown on the chip so
            /// the cost is visible before anything is decoded, let alone generated.
            public var tokens: Int
            public var pixelWidth: Int
            public var pixelHeight: Int

            public init(path: String, tokens: Int, pixelWidth: Int, pixelHeight: Int) {
                self.path = path
                self.tokens = tokens
                self.pixelWidth = pixelWidth
                self.pixelHeight = pixelHeight
            }
        }

        public init(
            id: UUID = UUID(), name: String, content: String, byteCount: Int,
            image: Image? = nil
        ) {
            self.id = id
            self.name = name
            self.content = content
            self.byteCount = byteCount
            self.image = image
        }

        public var isImage: Bool { image != nil }
    }

    public init(
        id: UUID = UUID(), role: Role, text: String, reasoning: String = "",
        date: Date = Date(), attachments: [Attachment] = []
    ) {
        self.id = id
        self.role = role
        self.date = date
        self.variants = [Variant(text: text, reasoning: reasoning)]
        self.activeVariant = 0
        self.attachments = attachments
    }

    public var current: Variant {
        get { variants[min(activeVariant, variants.count - 1)] }
        set { variants[min(activeVariant, variants.count - 1)] = newValue }
    }

    public var text: String {
        get { current.text }
        set { current.text = newValue }
    }
    public var reasoning: String { current.reasoning }
    public var hasSeveralVariants: Bool { variants.count > 1 }

    /// The text passed to the model: the message and its attachments.
    public var promptText: String {
        guard !attachments.isEmpty else { return text }
        var out = ""
        for attachment in attachments where !attachment.isImage {
            out += "--- \(attachment.name) ---\n\(attachment.content)\n\n"
        }
        return out + text
    }
}

/// Sampling settings, per conversation.
public struct GenerationSettings: Codable, Sendable, Equatable {
    /// Whether the sliders are being ignored in favour of what the loaded model publishes.
    ///
    /// Optional so that a conversation saved before this existed decodes to `nil` and is
    /// treated as following the model. That is deliberate: the stored 0.7 / 0.9 below was never
    /// anybody's choice, it was this app's default for every model, and on Qwen it produced
    /// visible repetition (M-069). Migrating those conversations is the point, not a
    /// side effect. Moving either slider sets this to `false` and the values are honoured again.
    public var usesModelDefaults: Bool?

    /// True unless the user has explicitly moved a slider.
    public var followsModel: Bool { usesModelDefaults ?? true }

    /// The fallback when the user has taken the wheel.
    ///
    /// These were chosen for GPT-OSS, which at 1.0 / 1.0 on a short prompt can wander into
    /// another language, and were then applied to every model that followed it. That is the
    /// mistake `followsModel` exists to undo: a recommendation for one model is not a house
    /// style, and each model publishes its own in `ModelDescriptor.samplingDefaults`.
    public var temperature: Double = 0.7
    public var topP: Double = 0.9
    public var reasoningEffort: String = "medium"
    /// The token budget for one turn, **reasoning included**.
    ///
    /// It used to be 1024, which is enough for an answer but not for a hard question:
    /// GPT-OSS at medium reasoning readily spends more than a thousand tokens in the
    /// analysis channel, exhausted its budget before writing anything, and stopped just
    /// after it had finished thinking. The user saw the reasoning end, then nothing.
    ///
    /// The real budget is bounded by the remaining context anyway: the engine applies
    /// whichever of the two limits binds first.
    public var maximumTokens: Int = 4096
    /// Extra instructions, placed wherever the active format puts them, Harmony's
    /// `developer` message, Gemma's leading `system` turn.
    public var instructions: String = ""

    /// Whether this conversation may search the web.
    ///
    /// **Off unless asked**, and optional so conversations saved before it decode to `nil` and
    /// stay off. Hydra's premise is that nothing leaves the machine; a search sends the query
    /// to a third party, and that is a trade the user makes knowingly or not at all.
    ///
    /// Per conversation rather than global because the declaration sits at the head of the
    /// prompt: flipping it mid-chat changes the twentieth token and costs the whole cached
    /// prefix, which on a long conversation is a minute and a half for a checkbox.
    public var searchesWeb: Bool?
    public var usesWebSearch: Bool { searchesWeb ?? false }

    /// The ceiling on one search's results, in tokens.
    ///
    /// A thousand is a measured choice, not a round number: eight snippets land at roughly 900
    /// tokens, which is 17 seconds of prefill on Qwen and just under what an image already
    /// costs. Raising it buys sources and spends seconds, one for one.
    public var searchTokenBudget: Int = 1000

    public init() {}

    public var reasoning: ReasoningLevel {
        ReasoningLevel(rawValue: reasoningEffort) ?? .medium
    }

    public var prompt: PromptSettings {
        PromptSettings(
            reasoning: reasoning,
            instructions: instructions.isEmpty ? nil : instructions,
            // The engine clears this again if the loaded model has no dialect for it or no
            // client behind it: this says what the conversation asked for, not what it gets.
            searching: usesWebSearch)
    }
}

public struct Conversation: Identifiable, Codable, Sendable, Equatable {
    public var id: UUID
    public var title: String
    public var modelID: String
    public var messages: [Message]
    public var settings: GenerationSettings
    public var createdAt: Date
    public var updatedAt: Date
    /// Tokens taken by the last prompt sent, for the context gauge.
    public var contextUsed: Int = 0

    public init(
        id: UUID = UUID(), title: String = "New conversation",
        modelID: String = CatalogEntry.all[0].id,
        messages: [Message] = [], settings: GenerationSettings = GenerationSettings()
    ) {
        self.id = id
        self.title = title
        self.modelID = modelID
        self.messages = messages
        self.settings = settings
        self.createdAt = Date()
        self.updatedAt = Date()
    }

    /// A title derived from the first message, truncated cleanly on a word.
    public mutating func retitleFromFirstMessage() {
        guard title == "New conversation",
            let first = messages.first(where: { $0.role == .user })
        else { return }
        let flat = first.text.replacingOccurrences(of: "\n", with: " ")
            .trimmingCharacters(in: .whitespaces)
        if flat.isEmpty { return }
        if flat.count <= 42 {
            title = flat
            return
        }
        let cut = flat.prefix(42)
        title = (cut.lastIndex(of: " ").map { String(cut[..<$0]) } ?? String(cut)) + "…"
    }

    /// Past turns, up to but excluding `limit`, in neither format's spelling.
    /// Only the answer enters the history, both official templates are explicit that
    /// reasoning from earlier turns is not replayed.
    /// The pictures attached to this conversation, in the order the turns present them.
    ///
    /// Flat rather than per turn, because that is the order the prompt's image pads appear in
    /// and therefore the order the runner wants them spliced.
    public func imagePaths(upTo limit: Int? = nil) -> [String] {
        let slice = limit.map { Array(messages.prefix($0)) } ?? messages
        return slice.filter { $0.role == .user }
            .flatMap { $0.attachments.compactMap { $0.image?.path } }
    }

    public func turns(upTo limit: Int? = nil) -> [ChatTurn] {
        let slice = limit.map { Array(messages.prefix($0)) } ?? messages
        return slice.compactMap { message in
            switch message.role {
            case .user:
                var turn = ChatTurn.user(message.promptText)
                turn.images = message.attachments.count { $0.isImage }
                return turn
            case .assistant:
                return message.text.isEmpty ? nil : .assistant(message.text)
            }
        }
    }
}

/// Conversation persistence, as JSON under Application Support.
///
/// Atomic and deferred writes: a conversation changes on every token produced, and
/// rewriting the file at that rate would cost more than the generation itself.
public final class ConversationStore: @unchecked Sendable {

    private let url: URL
    private let queue = DispatchQueue(label: "hydra.conversations")
    private var pending: [Conversation]?

    public init() throws {
        let base = try FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask,
            appropriateFor: nil, create: true)
        let directory = base.appending(path: "Hydra")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        self.url = directory.appending(path: "conversations.json")
    }

    public func load() -> [Conversation] {
        guard let data = try? Data(contentsOf: url) else { return [] }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return (try? decoder.decode([Conversation].self, from: data)) ?? []
    }

    /// Schedules a save. Calls close together coalesce into a single write.
    public func save(_ conversations: [Conversation]) {
        queue.async { [self] in
            let alreadyScheduled = pending != nil
            pending = conversations
            guard !alreadyScheduled else { return }
            queue.asyncAfter(deadline: .now() + 0.4) { [self] in
                guard let snapshot = pending else { return }
                pending = nil
                write(snapshot)
            }
        }
    }

    /// Forces an immediate write, used when the application closes.
    public func flush(_ conversations: [Conversation]) {
        queue.sync {
            pending = nil
            write(conversations)
        }
    }

    private func write(_ conversations: [Conversation]) {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try? encoder.encode(conversations).write(to: url, options: .atomic)
    }
}
