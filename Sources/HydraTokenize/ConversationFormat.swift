import Foundation
import HydraCore

/// How a conversation becomes a prompt, and how the tokens coming back become text.
///
/// The third and last thing that genuinely differs per model (D-023), after the install plan
/// and the runner. It differs in the way that rule cares about: get the markers wrong and
/// nothing raises an error, the model reads a prompt in a dialect it half-recognizes and
/// answers plausibly worse, which is the failure mode no crash will ever surface.
///
/// The two formats already existed and are unchanged. What is new is a neutral vocabulary
/// between them and the generation loop, so that loop stops naming Harmony in a comment about
/// end markers and then relying on Harmony's parser three lines later.
public protocol ConversationFormat: Sendable {

    /// For diagnostics and tests. Never branched on by the generation loop.
    var name: String { get }

    /// Tokens to keep in reserve at the end of the context for the format's closing markers.
    ///
    /// A number rather than a shared constant because the two formats close differently, and
    /// running out of room mid-marker truncates a generation in a way that looks like the
    /// model stopping early.
    var reservedStopTokens: Int { get }

    /// The full prompt, ready to encode with `allowSpecial: true`.
    func render(turns: [ChatTurn], settings: PromptSettings) -> String

    /// A parser holding its own streaming state.
    ///
    /// A class, where both underlying parsers use a `Session` passed `inout`. That works well
    /// inside one format and not at all across an existential, so the adapter owns the session
    /// and the loop just feeds it tokens.
    ///
    /// It takes the same settings the prompt was rendered with, because a prompt can leave the
    /// parser mid-channel: Gemma's opens `<|channel>thought` itself, so those tokens never
    /// reach the parser and it would otherwise file the reasoning as the answer.
    func makeParser(
        tokenizer: BPETokenizer, settings: PromptSettings
    ) -> any ConversationParser
}

/// The one place a prompt format is chosen, alongside `RepackPlanFactory` and `ModelRuntime`.
public enum ConversationFormats {
    public static func format(for architecture: ModelArchitecture) -> any ConversationFormat {
        switch architecture {
        case .gptOss: return HarmonyFormat()
        case .gemma4: return Gemma4Format()
        case .qwen35Moe: return QwenFormat()
        }
    }
}

/// One turn, in neither format's spelling.
public struct ChatTurn: Sendable, Equatable {
    public enum Role: String, Sendable {
        case user
        /// Harmony writes `assistant`, Gemma writes `model`. Neither name leaks out here.
        case assistant
    }

    public var role: Role
    public var content: String
    /// Pictures attached to this turn, which a multimodal format renders as placeholders ahead
    /// of the words. Zero for every text-only model, which ignores it entirely.
    public var images: Int

    public init(role: Role, content: String, images: Int = 0) {
        self.role = role
        self.content = content
        self.images = images
    }

    public static func user(_ text: String) -> ChatTurn { ChatTurn(role: .user, content: text) }
    public static func assistant(_ text: String) -> ChatTurn {
        ChatTurn(role: .assistant, content: text)
    }
}

/// How hard the model is asked to think, in terms both formats can honour.
///
/// `off` has no Harmony equivalent, GPT-OSS always reasons, and the effort levels only say
/// how much, so it maps to `low` there and to a closed thought channel in Gemma. That
/// asymmetry is the reason this is a shared enum rather than one format's type reused.
public enum ReasoningLevel: String, Sendable, CaseIterable {
    case off, low, medium, high
}

public struct PromptSettings: Sendable {
    public var reasoning: ReasoningLevel
    public var instructions: String?

    public init(reasoning: ReasoningLevel = .medium, instructions: String? = nil) {
        self.reasoning = reasoning
        self.instructions = instructions
    }
}

/// What a decoded token turned out to be.
///
/// Reduced to what the caller acts on. Harmony's `commentary` channel and its `channelEnded`
/// events have no consumer in the generation loop and are dropped by the adapter rather than
/// carried through as cases nobody switches on.
public enum PromptEvent: Sendable, Equatable {
    case answer(String)
    case reasoning(String)
    case stopped
}

public protocol ConversationParser: AnyObject {
    var isFinished: Bool { get }
    func consume(_ token: Int) -> [PromptEvent]
}

// MARK: - Harmony

public struct HarmonyFormat: ConversationFormat {

    public init() {}

    public var name: String { "harmony" }
    /// `<|end|>` and the start of the next message.
    public var reservedStopTokens: Int { 8 }

    public func render(turns: [ChatTurn], settings: PromptSettings) -> String {
        let instructions = settings.instructions?.trimmingCharacters(
            in: .whitespacesAndNewlines)
        let renderer = Harmony.Renderer(
            reasoningEffort: Self.effort(settings.reasoning),
            instructions: (instructions?.isEmpty ?? true) ? nil : instructions)
        return renderer.render(
            turns: turns.map {
                Harmony.Turn(
                    role: $0.role == .user ? .user : .assistant, content: $0.content)
            })
    }

    /// GPT-OSS has no way to be told not to reason, so `off` asks for as little as the format
    /// allows rather than pretending the channel can be closed.
    static func effort(_ level: ReasoningLevel) -> Harmony.ReasoningEffort {
        switch level {
        case .off, .low: return .low
        case .medium: return .medium
        case .high: return .high
        }
    }

    /// Harmony writes its own channel headers, so the settings say nothing the parser needs.
    public func makeParser(
        tokenizer: BPETokenizer, settings: PromptSettings
    ) -> any ConversationParser {
        HarmonyParserAdapter(tokenizer: tokenizer)
    }
}

final class HarmonyParserAdapter: ConversationParser {
    private let parser: Harmony.Parser
    private var session = Harmony.Parser.Session()

    init(tokenizer: BPETokenizer) {
        self.parser = Harmony.Parser(tokenizer: tokenizer)
    }

    var isFinished: Bool { session.isFinished }

    func consume(_ token: Int) -> [PromptEvent] {
        parser.consume(token, session: &session).compactMap { event in
            switch event {
            case let .text(channel, fragment):
                switch channel {
                case .final: return .answer(fragment)
                case .analysis: return .reasoning(fragment)
                // Tool-call scaffolding, which nothing downstream renders.
                case .commentary: return nil
                }
            case .stopped: return .stopped
            case .channelEnded: return nil
            }
        }
    }
}

// MARK: - Gemma 4

public struct Gemma4Format: ConversationFormat {

    public init() {}

    public var name: String { "gemma-4" }
    /// `<|turn>` and `<|endofsequence|>`.
    public var reservedStopTokens: Int { 4 }

    public func render(turns: [ChatTurn], settings: PromptSettings) -> String {
        let renderer = Gemma4Prompt.Renderer(
            thinking: settings.reasoning != .off, instructions: settings.instructions)
        return renderer.render(
            turns: turns.map {
                Gemma4Prompt.Turn(
                    role: $0.role == .user ? .user : .model, content: $0.content)
            })
    }

    public func makeParser(
        tokenizer: BPETokenizer, settings: PromptSettings
    ) -> any ConversationParser {
        Gemma4ParserAdapter(tokenizer: tokenizer, inThought: settings.reasoning != .off)
    }
}

final class Gemma4ParserAdapter: ConversationParser {
    private let parser: Gemma4Prompt.Parser
    private var session: Gemma4Prompt.Parser.Session

    init(tokenizer: BPETokenizer, inThought: Bool) {
        self.parser = Gemma4Prompt.Parser(tokenizer: tokenizer)
        self.session = Gemma4Prompt.Parser.Session(inThought: inThought)
    }

    var isFinished: Bool { session.isFinished }

    func consume(_ token: Int) -> [PromptEvent] {
        parser.consume(token, session: &session).map { event in
            switch event {
            case .text(let fragment): return .answer(fragment)
            case .reasoning(let fragment): return .reasoning(fragment)
            case .stopped: return .stopped
            }
        }
    }
}
