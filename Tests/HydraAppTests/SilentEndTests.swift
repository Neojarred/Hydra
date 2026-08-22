import Foundation
import Testing

@testable import HydraApp

/// A turn that produces no answer must say why.
///
/// The failure this guards against is silence, which no assertion elsewhere can catch: the
/// engine finished cleanly, the metrics were right, the reasoning was stored, and the user was
/// shown a fold of thinking followed by nothing. Measured on a real searching turn — 2,894
/// tokens produced, two characters of answer.
@Suite("A turn that answers nothing explains itself")
struct SilentEndTests {

    @Test("Running out of context says so, and says what to do")
    func contextExhausted() {
        let note = InferenceEngine.silentEndNote(
            produced: 300, budget: 300, room: 300, rounds: 0, maximumTokens: 4096)
        #expect(note.contains("300 tokens of context"))
        #expect(note.contains("context length"))
    }

    @Test("Running out of the token budget blames the budget, not the context")
    func budgetExhausted() {
        let note = InferenceEngine.silentEndNote(
            produced: 4096, budget: 4096, room: 8000, rounds: 0, maximumTokens: 4096)
        #expect(note.contains("reasoning used up"))
        #expect(!note.contains("context"))
    }

    @Test("Stopping below the budget is the case that used to say nothing at all")
    func modelJustStopped() {
        let note = InferenceEngine.silentEndNote(
            produced: 2894, budget: 4096, room: 7832, rounds: 0, maximumTokens: 4096)
        #expect(!note.isEmpty)
        #expect(note.contains("2894"))
        #expect(note.contains("without writing an"))
    }

    @Test("A turn that searched says so, since that is where the wait went")
    func mentionsTheSearch() {
        let once = InferenceEngine.silentEndNote(
            produced: 2894, budget: 4096, room: 7832, rounds: 1, maximumTokens: 4096)
        #expect(once.contains("searched the web once"))

        let twice = InferenceEngine.silentEndNote(
            produced: 2894, budget: 4096, room: 7832, rounds: 2, maximumTokens: 4096)
        #expect(twice.contains("searched the web 2 times"))

        let never = InferenceEngine.silentEndNote(
            produced: 2894, budget: 4096, room: 7832, rounds: 0, maximumTokens: 4096)
        #expect(!never.contains("searched"))
    }
}
