import Foundation
import HydraSearch
import HydraTokenize
import Testing

@testable import HydraApp

/// The engine's own search turn, against the real checkpoint and the real endpoint.
///
/// Everything else about search was verified through `hydra chat`, and the CLI is a **different
/// implementation of the same design**: it renders the same three pieces, but the loop around
/// them, the rewind, the cancellation checks and the event stream are the engine's own code.
/// A design proven in the instrument and never run in the application is a design nobody has
/// shipped.
///
/// Skipped unless `HYDRA_LIVE_SEARCH=1`, because it needs a GPU, a 18 GiB checkpoint, a network
/// and an API credit, and none of those belong in a suite that has to pass on any machine.
/// Run it deliberately:
///
///     HYDRA_LIVE_SEARCH=1 TAVILY_API_KEY=… swift test --filter LiveSearchTests
@Suite("The engine's search turn, live", .enabled(if: ProcessInfo.processInfo
    .environment["HYDRA_LIVE_SEARCH"] == "1"))
struct LiveSearchTests {

    /// Everything one turn reported, in the order the interface would have seen it.
    struct Transcript {
        var events: [String] = []
        var text = ""
        var reasoning = ""
        var query: String?
        var sources: [String] = []
        var searchTokens = 0
        var dropped = 0
        var failure: String?
        var newPromptTokens = 0
        var contextUsed = 0
    }

    private func run(
        _ question: String, model id: String = "qwen-3-6-35b-a3b-q4",
        searching: Bool = true, seconds: TimeInterval = 420
    ) throws -> Transcript {
        let key = try #require(
            ProcessInfo.processInfo.environment["TAVILY_API_KEY"],
            "TAVILY_API_KEY is required for a live search test")
        let entry = try #require(CatalogEntry.all.first { $0.id == id })

        let engine = InferenceEngine()
        engine.webSearch = TavilyClient(key: key)

        let loaded = DispatchSemaphore(value: 0)
        let loadFailure = Box<String?>(nil)
        engine.load(entry: entry, contextLength: 8192, slotsPerLayer: nil, progress: { _ in }) {
            if case .failure(let error) = $0 { loadFailure.value = error.description }
            loaded.signal()
        }
        #expect(loaded.wait(timeout: .now() + 300) == .success, "the model did not load in time")
        #expect(loadFailure.value == nil, "load failed: \(loadFailure.value ?? "")")
        defer { engine.unload() }

        var settings = GenerationSettings()
        settings.searchesWeb = searching
        settings.maximumTokens = 700
        settings.reasoningEffort = "medium"   // deliberately on: search must override it

        let box = Box(Transcript())
        let done = DispatchSemaphore(value: 0)
        engine.generate(turns: [.user(question)], settings: settings) { event in
            var transcript = box.value
            switch event {
            case .started(let prompt, let new, _):
                transcript.events.append("started")
                transcript.newPromptTokens = new
                _ = prompt
            case .firstToken: transcript.events.append("firstToken")
            case .reasoning(let f): transcript.reasoning += f
            case .text(let f): transcript.text += f
            case .searching(let query, _):
                transcript.events.append("searching")
                transcript.query = query
            case .searched(let outcome):
                transcript.events.append("searched")
                transcript.sources = outcome.sources.map(\.url)
                transcript.searchTokens = outcome.tokens
                transcript.dropped = outcome.dropped
            case .searchFailed(let reason):
                transcript.events.append("searchFailed")
                transcript.failure = reason
            case .finished(_, _, let used):
                transcript.events.append("finished")
                transcript.contextUsed = used
                box.value = transcript
                done.signal()
                return
            case .failed(let reason):
                transcript.events.append("failed")
                transcript.failure = reason
                box.value = transcript
                done.signal()
                return
            }
            box.value = transcript
        }
        #expect(done.wait(timeout: .now() + seconds) == .success, "the turn did not finish")
        return box.value
    }

    // MARK: -

    @Test("A searching turn queries, reads, and answers")
    func searchingTurn() throws {
        let transcript = try run(
            "Give me key details about the new Qwen3.8 27B model and how it compares to "
                + "qwen3.6 27B")

        #expect(transcript.failure == nil, "the turn failed: \(transcript.failure ?? "")")
        // The order is the contract the interface is written against.
        #expect(transcript.events.contains("searching"))
        #expect(transcript.events.contains("searched"))
        #expect(transcript.events.last == "finished")

        let query = try #require(transcript.query)
        #expect(!query.isEmpty)
        #expect(query.count < 200, "a query, not an essay: \(query)")

        #expect(!transcript.sources.isEmpty, "sources are what the interface lists")
        #expect(transcript.searchTokens > 100)
        #expect(transcript.searchTokens <= 1000, "the budget is a budget")

        // The turn answered. This is the failure that started all of it: 2,894 tokens produced
        // and two characters of text.
        #expect(transcript.text.count > 200, "no answer: \(transcript.text)")
        // And it answered without thinking, whatever the conversation asked for.
        #expect(transcript.reasoning.isEmpty, "a searching turn must not think")

        // The gauge counts the results it read, not just the prompt and the answer.
        #expect(transcript.contextUsed > transcript.newPromptTokens + transcript.searchTokens)
    }

    @Test("A turn that does not search is unchanged, and still thinks")
    func nonSearchingTurn() throws {
        let transcript = try run(
            "Explain in two sentences why a mixture-of-experts model saves memory.",
            searching: false)

        #expect(transcript.failure == nil)
        #expect(!transcript.events.contains("searching"))
        #expect(!transcript.events.contains("searched"))
        #expect(transcript.text.count > 100)
        // Thinking is only overridden for a searching turn. Taking it away everywhere would be
        // a much larger change than the one that was measured.
        #expect(!transcript.reasoning.isEmpty, "a normal turn still reasons")
    }

    @Test("Q8 runs the same path", .enabled(if: FileManager.default.fileExists(
        atPath: (try? ModelLocations.directory()
            .appending(path: "qwen-3-6-35b-a3b-q8.hydra").path) ?? "/nonexistent")))
    func quantizationEight() throws {
        // The reported failures included Q8, and every measurement since was Q4. A different
        // quantization is a different model as far as this behaviour is concerned.
        let transcript = try run(
            "What is the memory bandwidth of the Apple M4 Max, in GB/s?",
            model: "qwen-3-6-35b-a3b-q8", seconds: 600)

        #expect(transcript.failure == nil, "the turn failed: \(transcript.failure ?? "")")
        #expect(transcript.events.contains("searched"))
        #expect(transcript.text.count > 100, "no answer: \(transcript.text)")
        #expect(transcript.reasoning.isEmpty)
        #expect(transcript.text.contains("546"), "the answer is in the snippets")
    }

    /// A value written on the inference queue and read on the test's thread.
    private final class Box<T>: @unchecked Sendable {
        private var storage: T
        private let lock = NSLock()
        init(_ value: T) { storage = value }
        var value: T {
            get { lock.lock(); defer { lock.unlock() }; return storage }
            set { lock.lock(); storage = newValue; lock.unlock() }
        }
    }
}
