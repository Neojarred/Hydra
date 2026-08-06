import Foundation
import HydraCore
import Testing

@testable import HydraMetal

/// Décodage spéculatif.
///
/// La promesse est forte : produire les mêmes jetons, plus vite. Un vérificateur faux ne
/// planterait pas — il produirait du texte plausible et faux, sans que rien ne le signale.
/// Ces tests comparent donc la suite complète des jetons, pas une distribution approchée.
@Suite("Décodage spéculatif")
struct SpeculativeTests {

    private func makeRunner() throws -> (ModelRunner, URL) {
        let config = GptOssConfig.tiny
        let temporary = FileManager.default.temporaryDirectory
            .appending(path: "hydra-spec-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
        let root = try LayerRunnerTests.installTinyModel(at: temporary)
        let context = try MetalContext()
        let mapping = try ModelMapping(root: root, config: config, device: context.device)
        let cache = ExpertSlotCache(
            root: root, config: config,
            slotsPerLayer: config.expertsPerToken, device: context.device)
        let runner = try ModelRunner(
            config: config, context: context, mapping: mapping,
            expertCache: cache, contextLength: 128, prefillChunk: 16)
        return (runner, temporary)
    }

    /// Génère `count` jetons, avec ou sans brouillons, et rend la suite produite.
    ///
    /// Le brouillon est volontairement **faux par endroits** : c'est le mélange
    /// d'acceptations et de rejets qui exerce le rembobinage.
    private func generate(
        _ runner: ModelRunner, prompt: [Int], count: Int,
        sampling: ModelRunner.Sampling, drafts: [[Int]]
    ) throws -> [Int] {
        runner.reset()
        // Le tirage doit repartir du même état, sinon on compare deux suites
        // pseudo-aléatoires différentes et non deux chemins de décodage.
        runner.resetSampling()
        var distribution = try runner.prefill(tokens: prompt)
        var produced: [Int] = []
        var round = 0
        while produced.count < count {
            let draft = round < drafts.count ? drafts[round] : []
            let result = try runner.step(
                from: distribution, draft: draft, sampling: sampling)
            produced += result.tokens
            distribution = result.next
            round += 1
        }
        return Array(produced.prefix(count))
    }

    @Test("Un brouillon ne change pas la suite produite, en glouton")
    func greedyMatchesReference() throws {
        let (runner, temporary) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: temporary) }

        let prompt = (0..<10).map { ($0 * 7 + 3) % GptOssConfig.tiny.vocabSize }
        let greedy = ModelRunner.Sampling(temperature: 0, topP: 1)

        let reference = try generate(
            runner, prompt: prompt, count: 24, sampling: greedy, drafts: [])

        // On rejoue en proposant la vraie suite par tranches : tout doit être accepté.
        var perfect: [[Int]] = []
        var index = 0
        while index < reference.count {
            perfect.append(Array(reference[index..<min(index + 4, reference.count)]))
            index += 4
        }
        let withPerfectDrafts = try generate(
            runner, prompt: prompt, count: 24, sampling: greedy, drafts: perfect)
        #expect(withPerfectDrafts == reference, "un brouillon juste doit être transparent")

        // Puis avec des brouillons partiellement faux, pour exercer le rejet.
        let wrong = perfect.map { block -> [Int] in
            guard block.count > 1 else { return block }
            var copy = block
            copy[1] = (copy[1] + 1) % GptOssConfig.tiny.vocabSize
            return copy
        }
        let withWrongDrafts = try generate(
            runner, prompt: prompt, count: 24, sampling: greedy, drafts: wrong)
        #expect(withWrongDrafts == reference, "un brouillon faux doit être sans effet")

        // Et avec des brouillons entièrement faux : chemin de repli sans dépense.
        let garbage = (0..<12).map { round in
            (0..<4).map { ($0 * 31 + round * 17 + 5) % GptOssConfig.tiny.vocabSize }
        }
        let withGarbage = try generate(
            runner, prompt: prompt, count: 24, sampling: greedy, drafts: garbage)
        #expect(withGarbage == reference, "un brouillon absurde doit être sans effet")
    }

    /// Le tirage stochastique est le cas délicat : chaque jeton émis doit consommer
    /// exactement un tirage, sinon les suites divergent malgré une graine identique.
    @Test("Un brouillon ne change pas la suite produite, en échantillonnage")
    func sampledMatchesReference() throws {
        let (runner, temporary) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: temporary) }

        let prompt = (0..<10).map { ($0 * 5 + 11) % GptOssConfig.tiny.vocabSize }
        let sampling = ModelRunner.Sampling(temperature: 0.8, topP: 0.9, seed: 12345)

        let reference = try generate(
            runner, prompt: prompt, count: 20, sampling: sampling, drafts: [])

        var blocks: [[Int]] = []
        var index = 0
        while index < reference.count {
            blocks.append(Array(reference[index..<min(index + 3, reference.count)]))
            index += 3
        }
        let speculated = try generate(
            runner, prompt: prompt, count: 20, sampling: sampling, drafts: blocks)
        #expect(speculated == reference, "la suite tirée doit être identique à graine égale")
    }

    @Test("La passe groupée rend un jeu de logits par position")
    func verifyReturnsPerPositionLogits() throws {
        let (runner, temporary) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: temporary) }

        let prompt = (0..<6).map { ($0 * 3 + 1) % GptOssConfig.tiny.vocabSize }
        runner.reset()
        _ = try runner.prefill(tokens: prompt)

        let batch = [4, 9, 2]
        let logits = try runner.verify(tokens: batch)
        #expect(logits.count == batch.count)
        for row in logits {
            #expect(row.count == GptOssConfig.tiny.vocabSize)
            #expect(row.allSatisfy { $0.isFinite }, "des logits non finis signalent un bug")
        }
        #expect(runner.position == prompt.count + batch.count)
    }
}
