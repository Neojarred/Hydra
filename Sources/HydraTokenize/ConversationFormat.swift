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

    /// Whether the model is *offered* the tool and decides for itself, rather than having its
    /// turn split around a search it never asked for.
    ///
    /// Separate from `supportsTools`, which asks whether the dialect exists at all. Qwen's does
    /// and is deliberately not used: it can render and parse a call, and cannot be trusted with
    /// the deliberation that precedes one.
    var declaresTools: Bool { get }

    /// Whether this format's checkpoint was trained to call functions in a dialect we render
    /// and parse.
    ///
    /// A stated capability rather than an assumption, because the failure it prevents is the
    /// silent one: declaring tools to a model whose template we have not implemented produces
    /// a prompt it half-recognizes and a call we cannot read back, and nothing raises.
    var supportsTools: Bool { get }

    /// The text to append after a tool call: the result, and the markers that hand the turn
    /// back to the assistant.
    ///
    /// Text rather than turns because the engine appends this to a conversation already in the
    /// KV cache. Re-rendering the history instead would mean re-encoding the call the model
    /// just wrote, and byte-level BPE does not promise the same token ids across that join.
    ///
    /// It takes the settings because it ends with a generation prompt, and a generation prompt
    /// carries the reasoning state: Qwen's opens the thinking block itself. Rendering the tail
    /// without it hands the parser a stream whose first words are an answer while the parser is
    /// still waiting to leave a block that was never opened.
    func renderToolResult(_ text: String, settings: PromptSettings) -> String

    /// The conversation with no generation prompt, ready to be continued more than once.
    func renderOpen(turns: [ChatTurn], settings: PromptSettings) -> String
    /// The tail that hands the turn to the assistant.
    func generationPrompt(_ settings: PromptSettings) -> String
    /// A continuation that asks for a search query and nothing else, thinking closed.
    func renderQueryRequest() -> String
    /// A continuation that supplies results and opens the answer.
    func renderSearchResults(_ text: String, settings: PromptSettings) -> String

    /// The settings a parser should be built with after a tool result has been appended.
    ///
    /// Qwen's result reopens the assistant turn and its thought block, so the parser resumes
    /// exactly as it started. Gemma's continues a turn whose thought channel the model already
    /// closed before calling, so a parser told it is inside one would file the answer as
    /// reasoning.
    func settingsAfterToolResult(_ settings: PromptSettings) -> PromptSettings

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

extension ConversationFormat {
    public var supportsTools: Bool { false }
    public var declaresTools: Bool { false }
    public func settingsAfterToolResult(_ settings: PromptSettings) -> PromptSettings {
        settings
    }
    /// Unreachable while `supportsTools` is false, for the same reason `renderToolResult` is.
    public func renderOpen(turns: [ChatTurn], settings: PromptSettings) -> String {
        render(turns: turns, settings: settings)
    }
    public func generationPrompt(_ settings: PromptSettings) -> String { "" }
    public func renderQueryRequest() -> String { "" }
    public func renderSearchResults(_ text: String, settings: PromptSettings) -> String { "" }
    /// Unreachable while `supportsTools` is false, and empty rather than a `fatalError` because
    /// a wrong prompt is recoverable and a crash in the generation queue is not.
    public func renderToolResult(_ text: String, settings: PromptSettings) -> String { "" }
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

    /// Functions the model may call, rendered wherever the active format declares them.
    ///
    /// Empty by default, and empty must render **byte for byte** what was rendered before
    /// tools existed: every format puts its declaration at the head of the prompt, so a
    /// declaration that appears when it should not moves every token after it and costs a
    /// conversation its entire cached prefix.
    public var tools: [ToolDefinition]

    /// Whether this turn will be handed web results.
    ///
    /// Separate from `tools`, which is now always empty: the model is no longer asked to decide
    /// whether to search, but the prompt still has to tell it what year it is and that the
    /// results it is about to read are genuine. Attaching that to `tools` meant it vanished the
    /// day tools did, and the model went straight back to answering from 2024.
    public var searching: Bool

    /// The date to tell the model, as `yyyy-MM-dd`, or `nil` for today.
    ///
    /// Injected only so tests can pin it. Production leaves it `nil`: a model reading dated web
    /// pages needs the real date, and a fixed one would be a lie that survives into shipping.
    public var today: String?

    public init(
        reasoning: ReasoningLevel = .medium, instructions: String? = nil,
        tools: [ToolDefinition] = [], searching: Bool = false, today: String? = nil
    ) {
        self.reasoning = reasoning
        self.instructions = instructions
        self.tools = tools
        self.searching = searching
        self.today = today
    }
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

    /// Gemma is offered the **official** protocol, not the workaround Qwen needed.
    ///
    /// Its markers are single tokens, its template defines a real function-calling dialect, and
    /// nothing measured here shows it degenerating the way Qwen does when it reasons. So it is
    /// told what it can call and left to decide, which is cheaper than searching on every turn
    /// and is what the checkpoint was trained for. If it turns out to share Qwen's problem, the
    /// two-pass machinery is next door and already built.
    public var supportsTools: Bool { true }
    public var declaresTools: Bool { true }

    public func settingsAfterToolResult(_ settings: PromptSettings) -> PromptSettings {
        // The thought channel closed before the call; what follows the result is the answer.
        var after = settings
        after.reasoning = .off
        return after
    }

    /// The result continues the model's own turn: no role change and no generation prompt,
    /// because the template emits neither after a call.
    public func renderToolResult(_ text: String, settings: PromptSettings) -> String {
        Gemma4Prompt.toolResponse(name: WebSearchToolName, value: text)
    }

    public func render(turns: [ChatTurn], settings: PromptSettings) -> String {
        let renderer = Gemma4Prompt.Renderer(
            thinking: settings.reasoning != .off, instructions: settings.instructions,
            tools: settings.tools,
            today: settings.tools.isEmpty ? nil : (settings.today ?? Gemma4Prompt.today()))
        return renderer.render(
            turns: turns.map {
                Gemma4Prompt.Turn(
                    role: $0.role == .user ? .user : .model, content: $0.content,
                    images: $0.images)
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
            case .toolCall(let call): return .toolCall(call)
            case .stopped: return .stopped
            }
        }
    }
}
