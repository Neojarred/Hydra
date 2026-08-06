import Testing

@testable import HydraCore

@Suite("Tirage par n-grammes")
struct NGramDrafterTests {

    @Test("Une reprise récente est proposée")
    func proposesFromRepetition() {
        // « le ciel est bleu » … puis « le ciel » → on attend « est bleu ».
        let history = [10, 20, 30, 40, 50, 99, 10, 20]
        let drafter = NGramDrafter(patternLengths: [2], draftLength: 3)
        #expect(drafter.propose(history: history) == [30, 40, 50])
    }

    @Test("Le motif le plus long prime")
    func longestPatternWins() {
        // Deux reprises possibles : « 20 » seul mène à deux endroits, « 10 20 » à un seul.
        let history = [1, 20, 77, 10, 20, 30, 40, 10, 20]
        let drafter = NGramDrafter(patternLengths: [2, 1], draftLength: 2)
        #expect(drafter.propose(history: history) == [30, 40])
    }

    @Test("Sans reprise, aucune proposition")
    func silentWithoutRepetition() {
        let drafter = NGramDrafter(patternLengths: [3, 2], draftLength: 4)
        #expect(drafter.propose(history: [1, 2, 3, 4, 5]).isEmpty)
    }

    @Test("Un historique trop court ne fait rien")
    func shortHistory() {
        let drafter = NGramDrafter()
        #expect(drafter.propose(history: []).isEmpty)
        #expect(drafter.propose(history: [7]).isEmpty)
    }

    @Test("La reprise la plus récente est préférée")
    func prefersMostRecent() {
        let history = [5, 6, 100, 200, 5, 6, 300, 400, 5, 6]
        let drafter = NGramDrafter(patternLengths: [2], draftLength: 2)
        #expect(drafter.propose(history: history) == [300, 400])
    }

    /// La proposition est bornée par la fin de l'historique, pas par la position du motif :
    /// la suite d'une reprise ancienne peut recouvrir des jetons récents, et c'est légitime.
    @Test("La proposition reste dans les bornes de l'historique")
    func neverOverruns() {
        let history = [1, 2, 3, 1, 2]
        let drafter = NGramDrafter(patternLengths: [2], draftLength: 10)
        let draft = drafter.propose(history: history)
        #expect(draft == [3, 1, 2])
        #expect(draft.count <= history.count)
    }
}
