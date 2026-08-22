import Foundation
import HydraCore
import HydraInstall
import HydraSearch
import HydraTokenize

/// `hydra search` — the search path with no model attached.
///
/// It exists to answer one question before any protocol work is done: **what does a block of
/// snippets actually cost in tokens, measured with the tokenizer that will read it?**
/// `docs/03-WEB-SEARCH.md` estimates 700 tokens and 18 seconds, and the whole design rests on
/// that estimate. An estimate that decides a design should be measured before it is believed.
///
/// So this prints the block, its cost, where the cost went, and what the cost implies in
/// seconds of prefill on each model. It never loads a checkpoint: only the tokenizer, which is
/// a few megabytes of JSON.
enum Search {

    struct Options {
        var query: String = ""
        var model: String = "qwen-q4"
        var results: Int = 8
        var chunks: Int = 1
        var budget: Int = 1000
        var withoutURLs = false
        var showsBlock = true
        var showsJSON = false
        /// A saved response to render instead of calling the API.
        ///
        /// Not a test hook: on a 1,000-a-month allowance, tuning the block's shape against a
        /// live endpoint spends credits to re-measure text that has not changed. Capture one
        /// response, then sweep `--budget` and `--chunks` over it for free.
        var from: String?
        /// Where to record the provider's response verbatim.
        ///
        /// The web is not a fixed input. Two identical queries a minute apart came back 990 and
        /// 996 tokens, which is enough to move a generation onto a different trajectory — so an
        /// A/B of two prompts against a live endpoint is not an A/B at all. Recording a response
        /// once and replaying it is what makes the comparison mean something.
        var save: String?
    }

    /// Prefill rates, measured (`docs/02-MEASUREMENTS.md`, and the table in
    /// `docs/03-WEB-SEARCH.md` §1). Reported so a token count reads as a wait, which is the
    /// only unit a user experiences.
    static let prefillRates: [(String, Double)] = [
        ("Qwen 3.6", 53), ("Gemma 4", 40),
    ]

    static func run(_ options: Options) async throws {
        guard !options.query.isEmpty else {
            print("usage: hydra search \"<query>\" [options]")
            return
        }

        // The tokenizer of the model that will read the block. Not a detail: the same block
        // measures differently under each vocabulary, and the budget belongs to whichever one
        // is loaded.
        let (model, _, slug) = modelNamed(options.model)
        let root = try defaultModelDirectory().appending(path: "\(slug).hydra")
        let tokenizer = try TokenizerInstaller.load(
            from: root, architecture: model.architecture)
        let count: (String) -> Int = { tokenizer.encode($0, allowSpecial: false).count }

        let response: SearchResponse
        if let path = options.from {
            let data = try Data(contentsOf: URL(fileURLWithPath: path))
            response = try TavilyClient.decode(data, query: options.query)
            print("rendering \(path), no request made")
            print("  \(response.results.count) usable results")
        } else {
            let client = try TavilyClient.fromEnvironment()
            let query = SearchQuery(
                text: options.query, maximumResults: options.results,
                snippetsPerResult: options.chunks)

            print("searching \(client.name): \"\(options.query)\"")
            print("  asking for \(options.results) results, \(options.chunks) chunks each")

            let started = Date()
            let (raw, decoded) = try await client.searchRecording(query)
            response = decoded
            let elapsed = Date().timeIntervalSince(started)
            if let path = options.save {
                try raw.write(to: URL(fileURLWithPath: path))
                print("  recorded to \(path)")
            }

            print(String(
                format: "  %d results in %.2f s round trip", response.results.count, elapsed))
            if let provider = response.providerSeconds {
                print(String(format: "  (%.2f s of that was Tavily's own)", provider))
            }
        }
        print()

        if options.showsJSON {
            for (index, result) in response.results.enumerated() {
                print("  [\(index + 1)] \(result.title)")
                print("      \(result.url)")
                if let score = result.score { print(String(format: "      score %.4f", score)) }
                print("      \(result.snippet.count) chars, "
                    + "\(count(result.snippet)) tokens")
            }
            print()
        }

        let rendered = SearchBlock.render(
            response, budget: options.budget, includesURLs: !options.withoutURLs, count: count)

        if options.showsBlock {
            print(String(repeating: "-", count: 78))
            print(rendered.text, terminator: "")
            print(String(repeating: "-", count: 78))
            print()
        }

        report(rendered, response: response, options: options, count: count)
    }

    /// Where the tokens went.
    ///
    /// Broken down rather than totalled, because the parts are the levers: the URLs are pure
    /// overhead the model rarely needs in full, the preamble is a fixed toll, and only the
    /// snippets are the thing we are actually paying for. A total tells you the feature is
    /// slow; a breakdown tells you what to cut.
    static func report(
        _ rendered: SearchBlock.Rendered, response: SearchResponse, options: Options,
        count: (String) -> Int
    ) {
        // Measured on what is **rendered**, not on the raw result. The block trims a URL to
        // its host, caps a title and drops structural chrome, and a breakdown that priced the
        // untrimmed forms would report a cost the model never pays — a report about the wrong
        // artifact, which is the fault this file exists to avoid.
        let included = response.results.prefix(rendered.included)
        let urls = included.map { count(SearchBlock.host(of: $0.url) + "\n") }.reduce(0, +)
        let titles = included.map { count(SearchBlock.trimTitle($0.title)) }.reduce(0, +)
        let snippets = included.map { count(SearchBlock.clean($0.snippet)) }.reduce(0, +)
        let preamble = count(SearchBlock.preamble)
        let scaffolding = max(0, rendered.tokens - urls - titles - snippets - preamble)

        print("token budget \(options.budget)")
        print("  block            \(rendered.tokens) tokens")
        print("    preamble       \(preamble)")
        print("    titles         \(titles)")
        if !options.withoutURLs {
            print("    urls           \(urls)"
                + (rendered.included > 0
                    ? "  (\(urls / max(1, rendered.included)) a result)" : ""))
        }
        print("    snippets       \(snippets)")
        print("    scaffolding    \(scaffolding)")
        print("  results included \(rendered.included) of \(response.results.count)"
            + (rendered.truncated ? ", the last one cut to fit" : ""))
        if rendered.dropped > 0 {
            print("  results dropped  \(rendered.dropped)  (ranked below what fit)")
        }
        print()

        print("what that costs to read, at measured prefill rates")
        for (name, rate) in prefillRates {
            print("  " + pad(name, 12)
                + String(format: "%5.1f s", Double(rendered.tokens) / rate))
        }
        print()
        print("  an image, for comparison, is bounded at 1024 tokens")
    }
}
