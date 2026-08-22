import Foundation

/// The one function Hydra offers.
///
/// Named here rather than in the search target so a prompt format can render a result against
/// it without a format depending on an HTTP client.
public let WebSearchToolName = "web_search"

/// A function the model may call.
///
/// Neutral, like `ChatTurn`: the three checkpoints declare tools in three unrelated dialects,
/// and none of their spellings leaks out here.
public struct ToolDefinition: Sendable, Equatable {

    public struct Parameter: Sendable, Equatable {
        public var name: String
        /// A JSON Schema type name, which is what all three formats' declarations bottom out in.
        public var type: String
        public var description: String
        public var required: Bool

        public init(
            name: String, type: String = "string", description: String, required: Bool = false
        ) {
            self.name = name
            self.type = type
            self.description = description
            self.required = required
        }
    }

    public var name: String
    public var description: String
    public var parameters: [Parameter]

    public init(name: String, description: String, parameters: [Parameter]) {
        self.name = name
        self.description = description
        self.parameters = parameters
    }
}

/// A call the model asked for.
///
/// Arguments are strings because Qwen's protocol has no other type: it writes
/// `<parameter=topn>8</parameter>` and the value is whatever characters lie between the
/// markers. Converting is the caller's job, at the point where it knows what it wanted.
public struct ToolCall: Sendable, Equatable {
    public var name: String
    public var arguments: [String: String]

    public init(name: String, arguments: [String: String]) {
        self.name = name
        self.arguments = arguments
    }

    public subscript(_ key: String) -> String? { arguments[key] }
}

/// What a decoded token turned out to be.
///
/// Reduced to what the caller acts on. Harmony's `commentary` channel and its `channelEnded`
/// events have no consumer in the generation loop and are dropped by the adapter rather than
/// carried through as cases nobody switches on.
public enum PromptEvent: Sendable, Equatable {
    case answer(String)
    case reasoning(String)
    /// The model has asked for a function to be run.
    ///
    /// Always emitted **before** the `.stopped` that follows it, never instead of it: Qwen has
    /// no separate stop token for a call, so `</tool_call>` both ends the generation and is the
    /// call. A consumer that saw only `.stopped` would finish the turn holding a request it
    /// never noticed.
    case toolCall(ToolCall)
    case stopped
}
