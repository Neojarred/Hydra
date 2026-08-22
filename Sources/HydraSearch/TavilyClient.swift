import Foundation

/// Tavily, the shipped default.
///
/// Chosen on a token argument rather than an index-quality one: its `chunks_per_source`
/// bounds the returned text **server-side**, at 500 characters a chunk, so the prompt budget is
/// a request parameter instead of something recovered afterwards by trimming HTML. That deletes
/// the extraction problem for the snippet rung entirely, and extraction was the part of
/// `docs/03-WEB-SEARCH.md` with no cheap answer.
///
/// The free tier is 1,000 credits a month with no card, and a `basic` search is one credit with
/// its chunks included; `advanced` is two. We ask for `basic` because doubling the price to
/// re-rank pages we are only going to read 500 characters of is not obviously worth it, and
/// because the CLI exists to test that assumption rather than to assume it.
public struct TavilyClient: WebSearch {

    public static let keyVariable = "TAVILY_API_KEY"

    public var name: String { "Tavily" }

    private let key: String
    private let endpoint: URL
    private let session: URLSession

    public init(
        key: String,
        endpoint: URL = URL(string: "https://api.tavily.com/search")!,
        session: URLSession = .shared
    ) {
        self.key = key
        self.endpoint = endpoint
        self.session = session
    }

    /// Reads the key from the environment, for the CLI and for tests.
    ///
    /// The application asks the user for it instead and keeps it in the keychain: an API key
    /// belongs to a person, and no allowance in this market survives being shared between
    /// everyone who downloads a DMG.
    public static func fromEnvironment(
        session: URLSession = .shared
    ) throws -> TavilyClient {
        guard let key = ProcessInfo.processInfo.environment[keyVariable], !key.isEmpty else {
            throw SearchError.missingKey(provider: "Tavily", variable: keyVariable)
        }
        return TavilyClient(key: key, session: session)
    }

    public func search(_ query: SearchQuery) async throws -> SearchResponse {
        try Self.decode(try await rawSearch(query), query: query.text)
    }

    private func rawSearch(_ query: SearchQuery) async throws -> Data {
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("hydra/0.5", forHTTPHeaderField: "User-Agent")

        // `include_answer` and `include_raw_content` are named here, set to false, rather than
        // left to their defaults. Both are one word away from changing what this product is:
        // the first has Tavily's model write the answer our model is supposed to write, the
        // second unbounds the token cost the whole design exists to bound. Defaults drift;
        // an explicit false is a decision the next reader can see.
        let body: [String: Any] = [
            "query": query.text,
            "search_depth": "basic",
            "max_results": query.maximumResults,
            "chunks_per_source": query.snippetsPerResult,
            "include_answer": false,
            "include_raw_content": false,
            "include_images": false,
        ]
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw SearchError.badStatus(-1, provider: name, message: nil)
        }

        switch http.statusCode {
        case 200:
            break
        case 401, 403:
            throw SearchError.unauthorized(provider: name)
        // 432 is Tavily's own "plan limit reached"; 429 is ordinary rate limiting. Both mean
        // "no more searches for you right now" to a user, and only one of them is temporary.
        case 429, 432:
            throw SearchError.allowanceExhausted(provider: name)
        default:
            throw SearchError.badStatus(
                http.statusCode, provider: name, message: Self.errorMessage(in: data))
        }

        return data
    }

    /// The same search, with the provider's bytes handed back for recording.
    ///
    /// Used by the measurement tools: replaying a recorded response is the only way to compare
    /// two prompts without the web moving underneath the comparison.
    public func searchRecording(
        _ query: SearchQuery
    ) async throws -> (raw: Data, response: SearchResponse) {
        let raw = try await rawSearch(query)
        return (raw, try Self.decode(raw, query: query.text))
    }

    // MARK: - Decoding

    /// Tavily's wire shape, kept private: nothing outside this file knows the vendor's
    /// spelling.
    private struct Payload: Decodable {
        struct Result: Decodable {
            let title: String?
            let url: String?
            let content: String?
            let score: Double?
            let publishedDate: String?
        }
        let query: String?
        let results: [Result]?
        let responseTime: Double?
    }

    public static func decode(_ data: Data, query: String) throws -> SearchResponse {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let payload: Payload
        do {
            payload = try decoder.decode(Payload.self, from: data)
        } catch {
            throw SearchError.malformedResponse(provider: "Tavily", detail: "\(error)")
        }

        // A result without a URL is not citable and a result without text is not readable.
        // Dropping them here keeps every later stage free of optionals that only this vendor's
        // schema made possible.
        let results: [SearchResult] = (payload.results ?? []).compactMap { raw in
            guard let url = raw.url, !url.isEmpty else { return nil }
            let snippet = (raw.content ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            guard !snippet.isEmpty else { return nil }
            return SearchResult(
                title: (raw.title ?? "").trimmingCharacters(in: .whitespacesAndNewlines),
                url: url,
                snippet: snippet,
                score: raw.score,
                publishedDate: raw.publishedDate)
        }

        return SearchResponse(
            query: payload.query ?? query, results: results,
            providerSeconds: payload.responseTime)
    }

    /// Best effort: Tavily puts its explanation under `detail`, sometimes as a string and
    /// sometimes as an object with an `error`. Neither shape is worth a type.
    static func errorMessage(in data: Data) -> String? {
        guard let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        if let detail = root["detail"] as? String { return detail }
        if let detail = root["detail"] as? [String: Any],
            let message = detail["error"] as? String
        {
            return message
        }
        return root["error"] as? String
    }
}
