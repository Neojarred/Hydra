import Foundation
import HydraCore
import HydraFormat
import Testing

@testable import HydraMetal

/// Réutilisation du cache KV d'un tour de conversation au suivant.
///
/// Deux propriétés doivent tenir, et leur violation serait **silencieuse** : le modèle
/// continuerait de répondre, simplement à partir d'un contexte faux.
///
/// 1. Donner un stockage linéaire aux couches à fenêtre glissante ne doit pas élargir leur
///    attention. Stockage et fenêtre sont deux notions distinctes ; les confondre rendrait
///    ces couches pleines.
/// 2. Rembobiner puis reprendre doit donner exactement le même état qu'un calcul complet.
@Suite("Réutilisation du cache KV")
struct KVReuseTests {

    private func makeModel() throws -> (URL, MetalContext, ModelMapping, URL) {
        let config = GptOssConfig.tiny
        let temporary = FileManager.default.temporaryDirectory
            .appending(path: "hydra-kv-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
        let root = try LayerRunnerTests.installTinyModel(at: temporary)
        let context = try MetalContext()
        let mapping = try ModelMapping(root: root, config: config, device: context.device)
        return (root, context, mapping, temporary)
    }

    private func makeRunner(
        root: URL, context: MetalContext, mapping: ModelMapping, contextLength: Int
    ) throws -> ModelRunner {
        let config = GptOssConfig.tiny
        let cache = ExpertSlotCache(
            root: root, config: config,
            slotsPerLayer: config.expertsPerToken, device: context.device)
        return try ModelRunner(
            config: config, context: context, mapping: mapping,
            expertCache: cache, contextLength: contextLength, prefillChunk: 16)
    }

    private func divergence(_ a: [Float], _ b: [Float]) -> Float {
        var scale: Float = 0
        for value in a { scale = max(scale, abs(value)) }
        var worst: Float = 0
        for (x, y) in zip(a, b) { worst = max(worst, abs(x - y) / max(scale, 1e-6)) }
        return worst
    }

    /// L'invite dépasse la fenêtre glissante (8) : si le stockage linéaire élargissait
    /// l'attention, les jetons au-delà de la fenêtre entreraient dans le calcul et l'écart
    /// serait massif.
    @Test("Le stockage linéaire fenêtre comme l'anneau")
    func linearStorageKeepsWindow() throws {
        let (root, context, mapping, temporary) = try makeModel()
        defer { try? FileManager.default.removeItem(at: temporary) }

        let prompt = (0..<20).map { ($0 * 7 + 3) % GptOssConfig.tiny.vocabSize }

        // Contexte court : couches glissantes en stockage linéaire.
        let linear = try makeRunner(
            root: root, context: context, mapping: mapping, contextLength: 64)
        #expect(linear.canRewind, "un contexte court doit donner un cache rembobinable")
        let linearOutput = Array(try linear.prefill(tokens: prompt))

        // Contexte au-delà du seuil : les couches glissantes reprennent leur anneau.
        let ringed = try makeRunner(
            root: root, context: context, mapping: mapping,
            contextLength: KVCache.linearWindowLimit + 1)
        #expect(!ringed.canRewind, "un anneau ne doit pas se déclarer rembobinable")
        let ringedOutput = Array(try ringed.prefill(tokens: prompt))

        let worst = divergence(ringedOutput, linearOutput)
        #expect(worst < 2e-3, "écart relatif \(worst) entre anneau et stockage linéaire")
    }

    /// Le scénario réel : un tour, une réponse, puis un tour de suite dont l'invite
    /// prolonge la précédente.
    @Test("Rembobiner puis reprendre égale un calcul complet")
    func rewindMatchesFullPrefill() throws {
        let (root, context, mapping, temporary) = try makeModel()
        defer { try? FileManager.default.removeItem(at: temporary) }

        let vocab = GptOssConfig.tiny.vocabSize
        let firstTurn = (0..<12).map { ($0 * 7 + 3) % vocab }
        let generated = (0..<9).map { ($0 * 13 + 5) % vocab }
        let secondTurn = (0..<6).map { ($0 * 5 + 11) % vocab }

        // Ce que le modèle a réellement traversé au premier tour, réponse comprise.
        let fed = firstTurn + generated
        let followUp = fed + secondTurn

        // Chemin réutilisé : on rejoue le premier tour, puis on reprend au préfixe commun.
        let reused = try makeRunner(
            root: root, context: context, mapping: mapping, contextLength: 64)
        _ = try reused.prefill(tokens: firstTurn)
        for token in generated { _ = try reused.forward(token: token, needsLogits: false) }
        #expect(reused.position == fed.count)

        let common = commonPrefixLength(fed, followUp)
        #expect(common == fed.count, "l'invite de suite doit prolonger ce qui a été traité")
        reused.rewind(to: common)
        #expect(reused.position == common)
        let reusedOutput = Array(try reused.prefill(tokens: Array(followUp.dropFirst(common))))

        // Chemin neuf : tout est recalculé depuis zéro.
        let fresh = try makeRunner(
            root: root, context: context, mapping: mapping, contextLength: 64)
        let freshOutput = Array(try fresh.prefill(tokens: followUp))

        let worst = divergence(freshOutput, reusedOutput)
        #expect(worst < 2e-3, "écart relatif \(worst) entre reprise et calcul complet")

        let bestFresh = freshOutput.firstIndex(of: freshOutput.max()!)
        let bestReused = reusedOutput.firstIndex(of: reusedOutput.max()!)
        #expect(bestFresh == bestReused, "le jeton glouton diffère")
    }

    /// Une invite qui **diverge** du cache doit repartir du point de divergence, pas
    /// réutiliser des clés qui ne lui correspondent plus. C'est le cas d'un message modifié.
    @Test("Une invite divergente reprend au bon endroit")
    func divergentPromptRestartsAtDivergence() throws {
        let (root, context, mapping, temporary) = try makeModel()
        defer { try? FileManager.default.removeItem(at: temporary) }

        let vocab = GptOssConfig.tiny.vocabSize
        let original = (0..<14).map { ($0 * 7 + 3) % vocab }
        // Même début, fin différente — un message édité au huitième jeton.
        var edited = Array(original[0..<8])
        edited += (0..<6).map { ($0 * 3 + 41) % vocab }

        let common = commonPrefixLength(original, edited)
        #expect(common == 8, "le préfixe commun doit s'arrêter à la divergence")

        let reused = try makeRunner(
            root: root, context: context, mapping: mapping, contextLength: 64)
        _ = try reused.prefill(tokens: original)
        reused.rewind(to: common)
        let reusedOutput = Array(try reused.prefill(tokens: Array(edited.dropFirst(common))))

        let fresh = try makeRunner(
            root: root, context: context, mapping: mapping, contextLength: 64)
        let freshOutput = Array(try fresh.prefill(tokens: edited))

        let worst = divergence(freshOutput, reusedOutput)
        #expect(worst < 2e-3, "écart relatif \(worst) après reprise sur invite modifiée")
    }
}
