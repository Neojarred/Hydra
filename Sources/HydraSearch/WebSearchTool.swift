import Foundation
import HydraCore

/// The one tool Hydra offers a model, and how a call to it becomes a query.
///
/// It lives beside the client rather than beside the prompt formats because the name, the
/// description and the arguments are one decision with the thing that executes them: a
/// description that promises a page and a client that returns a snippet is a mismatch nothing
/// would report.
///
/// **One function, not a namespace.** GPT-OSS ships a trained `browser` namespace with
/// `search`, `open` and `find` over a cursor and a viewport, and it is tempting to mirror it.
/// That is the full-page rung, and on this machine a page is 50 to 100 seconds of prefill
/// (`docs/03-WEB-SEARCH.md`). A single search that returns ranked snippets is the rung that
/// fits, and offering only that is what keeps a model from asking for the one that does not.
/// **Currently unused by the shipping path.** Kept deliberately.
///
/// The model is no longer offered this as a tool: it is asked for a query directly and the
/// search is run for it (M-077). The declaration below is what a model would be shown if it
/// were, and is retained against Qwen 3.8, whose bounded `reasoning_effort` may make
/// model-decided search workable where 3.6's unbounded deliberation was not.
///
/// `cleanQuery` and `failureNote` *are* live and used by both the engine and the CLI.
public enum WebSearchTool {

    public static let name = "web_search"

    /// What the model is told the tool does.
    ///
    /// **Directive, not advisory.** The first version of this said "use it when the answer
    /// depends on recent events; answer from your own knowledge otherwise", and on a thinking
    /// model that is an invitation to deliberate. Measured: asked about a model released after
    /// its cutoff, Qwen spent its whole reasoning budget arguing with itself about whether the
    /// thing existed — "maybe Phi-3.8? maybe Llama 3.8?" for a thousand tokens — and never
    /// answered. It was doing what the description asked.
    ///
    /// So the order is stated instead: search first, reason afterwards. Deciding from memory
    /// whether something exists is the one judgement a model with a stale cutoff cannot make,
    /// and it is exactly the judgement the old wording requested.
    ///
    /// It also says **snippets** and says so twice, because a model that believes whole pages
    /// are coming asks a broader question and then complains it cannot see the rest.
    public static var definition: ToolDefinition {
        ToolDefinition(
            name: name,
            description: """
                Search the web and return short ranked snippets from several pages, with their \
                titles and links. Returns extracts, not whole pages. Search first, then reason: \
                if the question mentions a product, model, version, price, date or event that \
                you are not certain about, call this straight away rather than working out from \
                memory whether it exists. Your knowledge of such things is older than today and \
                may simply be missing them.
                """,
            parameters: [
                .init(
                    name: "query",
                    description: "The search terms, as they would be typed into a search engine.",
                    required: true)
            ])
    }

    /// The query a call asks for, or `nil` if the call is not one this tool can answer.
    ///
    /// Both failures are the model's to make and neither is exceptional: it can name a function
    /// that does not exist, and it can call this one with no query. Returning `nil` lets the
    /// engine tell it so and carry on, which is cheaper than ending the turn.
    public static func query(
        from call: ToolCall, maximumResults: Int = 8, snippetsPerResult: Int = 1
    ) -> SearchQuery? {
        guard call.name == name else { return nil }
        let text = (call["query"] ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return nil }
        return SearchQuery(
            text: text, maximumResults: maximumResults, snippetsPerResult: snippetsPerResult)
    }

    /// The query the model wrote, stripped of the decoration it puts around one anyway.
    ///
    /// One copy. It lived twice — once in the engine, once in the CLI, with a comment saying
    /// the second mirrored the first — and two copies of a parser drift until the command-line
    /// tool stops being a faithful instrument for the application it is used to debug.
    public static func cleanQuery(_ raw: String) -> String {
        var line = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        // First line only: the instruction asks for the query alone and mostly gets it.
        if let newline = line.firstIndex(of: "\n") { line = String(line[..<newline]) }
        line = line.trimmingCharacters(in: .whitespacesAndNewlines)
        // Labels the instruction asked it not to write, and it sometimes writes anyway.
        for prefix in ["query:", "search query:", "search:", "q:"] where
            line.lowercased().hasPrefix(prefix)
        {
            line = String(line.dropFirst(prefix.count)).trimmingCharacters(in: .whitespaces)
        }
        // Matched wrapping quotes and stray markdown, repeatedly: `"`a query`"` happens.
        while let first = line.first, let last = line.last, line.count > 1,
            (first == "\"" && last == "\"") || (first == "'" && last == "'")
                || (first == "`" && last == "`")
        {
            line = String(line.dropFirst().dropLast()).trimmingCharacters(in: .whitespaces)
        }
        return line
    }

    /// What the model is handed when a call could not be run.
    ///
    /// A failed search must not end the turn. The model has a question in front of it and its
    /// own weights to answer from, and a note saying the search failed is enough for it to say
    /// so honestly. Ending the generation instead would turn a flaky network into a blank reply.
    public static func failureNote(_ reason: String) -> String {
        "The search could not be run: \(reason). Answer from your own knowledge, and say that "
            + "you could not check the web."
    }
}
