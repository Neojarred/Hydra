import Testing

@testable import HydraCore

@Suite("N-gram drafting")
struct NGramDrafterTests {

    @Test("A recent repetition is proposed")
    func proposesFromRepetition() {
        // "the sky is blue" … then "the sky" → we expect "is blue".
        let history = [10, 20, 30, 40, 50, 99, 10, 20]
        let drafter = NGramDrafter(patternLengths: [2], draftLength: 3)
        #expect(drafter.propose(history: history) == [30, 40, 50])
    }

    @Test("The longest pattern wins")
    func longestPatternWins() {
        // Two possible repetitions: "20" alone leads to two places, "10 20" to only one.
        let history = [1, 20, 77, 10, 20, 30, 40, 10, 20]
        let drafter = NGramDrafter(patternLengths: [2, 1], draftLength: 2)
        #expect(drafter.propose(history: history) == [30, 40])
    }

    @Test("With no repetition, no proposal")
    func silentWithoutRepetition() {
        let drafter = NGramDrafter(patternLengths: [3, 2], draftLength: 4)
        #expect(drafter.propose(history: [1, 2, 3, 4, 5]).isEmpty)
    }

    @Test("A history that is too short does nothing")
    func shortHistory() {
        let drafter = NGramDrafter()
        #expect(drafter.propose(history: []).isEmpty)
        #expect(drafter.propose(history: [7]).isEmpty)
    }

    @Test("The most recent repetition is preferred")
    func prefersMostRecent() {
        let history = [5, 6, 100, 200, 5, 6, 300, 400, 5, 6]
        let drafter = NGramDrafter(patternLengths: [2], draftLength: 2)
        #expect(drafter.propose(history: history) == [300, 400])
    }

    /// The proposal is bounded by the end of the history, not by the pattern's position: the
    /// continuation of an older repetition may overlap recent tokens, and that is legitimate.
    @Test("The proposal stays within the bounds of the history")
    func neverOverruns() {
        let history = [1, 2, 3, 1, 2]
        let drafter = NGramDrafter(patternLengths: [2], draftLength: 10)
        let draft = drafter.propose(history: history)
        #expect(draft == [3, 1, 2])
        #expect(draft.count <= history.count)
    }
}
