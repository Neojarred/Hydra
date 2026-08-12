import Foundation
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

    public init(text: String = "", reasoning: String = "") {
        self.text = text
        self.reasoning = reasoning
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
        public var content: String
        public var byteCount: Int
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
        for attachment in attachments {
            out += "--- \(attachment.name) ---\n\(attachment.content)\n\n"
        }
        return out + text
    }
}

/// Sampling settings, per conversation.
public struct GenerationSettings: Codable, Sendable, Equatable {
    /// OpenAI recommends 1.0 with `top_p` 1.0 for GPT-OSS — the raw distribution. On very
    /// short prompts that makes the 20B frankly unstable: it can wander into another
    /// language. So we adopt a slightly tighter default, and OpenAI's recommendation stays
    /// one slider away.
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
    /// Extra instructions, placed wherever the active format puts them — Harmony's
    /// `developer` message, Gemma's leading `system` turn.
    public var instructions: String = ""

    public init() {}

    public var reasoning: ReasoningLevel {
        ReasoningLevel(rawValue: reasoningEffort) ?? .medium
    }

    public var prompt: PromptSettings {
        PromptSettings(
            reasoning: reasoning,
            instructions: instructions.isEmpty ? nil : instructions)
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
    /// Only the answer enters the history — both official templates are explicit that
    /// reasoning from earlier turns is not replayed.
    public func turns(upTo limit: Int? = nil) -> [ChatTurn] {
        let slice = limit.map { Array(messages.prefix($0)) } ?? messages
        return slice.compactMap { message in
            switch message.role {
            case .user: return .user(message.promptText)
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

    /// Forces an immediate write — used when the application closes.
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
