import Foundation

/// Web search, in no provider's spelling.
///
/// The fourth thing that genuinely differs per vendor, after the install plan, the runner and
/// the prompt format, and it differs in the same way: get it wrong and nothing raises, the
/// model simply reads worse text and answers plausibly worse.
///
/// The protocol is deliberately narrow. A provider takes a query and returns ranked results
/// carrying a **snippet**, never a page. That is not a simplification to be relaxed later: on
/// this machine a page costs 50 to 100 seconds of prefill and a snippet block costs 15, so the
/// shape of this protocol *is* the feature's design (`docs/03-WEB-SEARCH.md`).
public protocol WebSearch: Sendable {

    /// For diagnostics and for the interface, which must name the third party the query is
    /// being sent to. Never branched on.
    var name: String { get }

    func search(_ query: SearchQuery) async throws -> SearchResponse
}

/// What is asked of a provider.
///
/// Only what every provider can honour. Vendor-specific knobs stay inside the vendor's client,
/// where they can be tuned against that vendor's billing without the caller learning about it.
public struct SearchQuery: Sendable, Equatable {

    public var text: String

    /// How many pages to rank. The token cost is roughly linear in this, so it is the
    /// main lever the caller has before the budget starts dropping results on the floor.
    public var maximumResults: Int

    /// How many excerpts to take from each page.
    ///
    /// More chunks is more of the page and a better chance the answer is actually present;
    /// it is also directly more tokens.
    ///
    /// **One**, measured rather than assumed. On a real factual query at a 1,000-token budget,
    /// one chunk a source fits five results and two chunks fits three, for the same ~640
    /// tokens of text. Five sources that each state the answer beat three that state it at
    /// greater length: the redundancy is the point, since a small model reading the same figure
    /// from four independent pages is the cheapest defence against it inventing a fifth.
    public var snippetsPerResult: Int

    public init(text: String, maximumResults: Int = 8, snippetsPerResult: Int = 1) {
        self.text = text
        self.maximumResults = maximumResults
        self.snippetsPerResult = snippetsPerResult
    }
}

/// One ranked page.
public struct SearchResult: Sendable, Equatable {

    public var title: String
    public var url: String

    /// The provider's extracted text for this page. **A snippet, not the page**: nothing in
    /// this type is ever a full document, and no caller should have to check.
    public var snippet: String

    /// The provider's own relevance score, kept for ordering and for diagnostics. Not
    /// comparable between providers and never shown to the model.
    public var score: Double?

    public var publishedDate: String?

    public init(
        title: String, url: String, snippet: String, score: Double? = nil,
        publishedDate: String? = nil
    ) {
        self.title = title
        self.url = url
        self.snippet = snippet
        self.score = score
        self.publishedDate = publishedDate
    }
}

public struct SearchResponse: Sendable, Equatable {

    public var query: String
    public var results: [SearchResult]

    /// What the provider says it spent. **Not** the wait: the network is on top of it, and
    /// both are noise next to the prefill that follows.
    public var providerSeconds: Double?

    public init(query: String, results: [SearchResult], providerSeconds: Double? = nil) {
        self.query = query
        self.results = results
        self.providerSeconds = providerSeconds
    }
}

/// Why a search failed, in terms the interface can put in front of a user.
///
/// `allowanceExhausted` is a case rather than a `badStatus(429)` on purpose. Every free tier in
/// this market fails closed when the month's credits are gone, and "HTTP 429" in a chat window
/// is the kind of seam this project has been bitten at before: the user needs to be told their
/// allowance ran out, not shown a status code.
public enum SearchError: Error, CustomStringConvertible, Equatable {

    case missingKey(provider: String, variable: String)
    case allowanceExhausted(provider: String)
    case unauthorized(provider: String)
    case badStatus(Int, provider: String, message: String?)
    case malformedResponse(provider: String, detail: String)

    public var description: String {
        switch self {
        case let .missingKey(provider, variable):
            return "no \(provider) API key: set \(variable)"
        case let .allowanceExhausted(provider):
            return "your \(provider) search allowance is used up for this month"
        case let .unauthorized(provider):
            return "\(provider) rejected the API key"
        case let .badStatus(code, provider, message):
            return message.map { "\(provider) returned HTTP \(code): \($0)" }
                ?? "\(provider) returned HTTP \(code)"
        case let .malformedResponse(provider, detail):
            return "\(provider) returned something unreadable: \(detail)"
        }
    }
}
