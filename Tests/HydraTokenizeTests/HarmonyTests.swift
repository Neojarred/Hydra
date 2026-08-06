import Foundation
import Testing

@testable import HydraTokenize

/// Harmony n'est pas un gabarit de chat ordinaire : le modèle répartit sa sortie sur des
/// canaux, et l'interface doit les distinguer. Une erreur ici ne casse rien visiblement —
/// elle fait juste apparaître le raisonnement dans la réponse, ou perdre la réponse.
struct HarmonyTests {

    static func makeTokenizer() throws -> BPETokenizer {
        var vocabulary: [String: Int] = [:]
        for byte in 0..<256 { vocabulary[String(ByteLevel.encodeTable[byte])] = byte }
        var next = 256
        for piece in ["analysis", "final", "commentary", "assistant", "Bonjour", "Ġmonde"] {
            vocabulary[piece] = next
            next += 1
        }
        return try BPETokenizer(
            vocabulary: vocabulary, merges: [],
            specialTokens: [
                "<|start|>": 200006, "<|end|>": 200007, "<|message|>": 200008,
                "<|channel|>": 200005, "<|return|>": 200002, "<|call|>": 200012,
                "<|endoftext|>": 199999,
            ])
    }

    // MARK: - Rendu

    @Test("L'invite suit la structure du gabarit officiel")
    func renderMatchesTemplate() {
        let renderer = Harmony.Renderer(currentDate: "2026-08-05", reasoningEffort: .low)
        let text = renderer.render(turns: [.user("Salut")])

        #expect(text.hasPrefix("<|start|>system<|message|>"))
        #expect(text.contains("Knowledge cutoff: 2024-06"))
        #expect(text.contains("Current date: 2026-08-05"))
        #expect(text.contains("Reasoning: low"))
        #expect(text.contains("# Valid channels: analysis, commentary, final."))
        #expect(text.contains("<|start|>user<|message|>Salut<|end|>"))
        // L'invite se termine sur l'amorce de génération, sans canal : c'est au modèle
        // de choisir s'il commence par analysis ou par final.
        #expect(text.hasSuffix("<|start|>assistant"))
    }

    @Test("Les consignes développeur ne sont rendues que si elles existent")
    func developerMessageIsOptional() {
        let without = Harmony.Renderer().render(turns: [.user("a")])
        #expect(!without.contains("<|start|>developer"))

        let with = Harmony.Renderer(instructions: "Réponds en vers.").render(turns: [.user("a")])
        #expect(with.contains("<|start|>developer<|message|># Instructions\n\nRéponds en vers."))
    }

    /// Le gabarit officiel est explicite : le raisonnement des tours passés n'est jamais
    /// réinjecté en inférence. Le réintroduire ferait dériver le modèle.
    @Test("L'historique ne conserve que le canal final")
    func historyKeepsOnlyFinalChannel() {
        let text = Harmony.Renderer().render(turns: [
            .user("Question 1"), .assistant("Réponse 1"), .user("Question 2"),
        ])
        #expect(text.contains("<|start|>assistant<|channel|>final<|message|>Réponse 1<|end|>"))
        #expect(!text.contains("analysis<|message|>Réponse 1"))
    }

    // MARK: - Analyse de la sortie

    /// Reproduit une sortie typique du modèle et vérifie la séparation des canaux.
    @Test("Le parseur sépare raisonnement et réponse")
    func parserSeparatesChannels() throws {
        let tokenizer = try Self.makeTokenizer()
        let parser = Harmony.Parser(tokenizer: tokenizer)
        var session = Harmony.Parser.Session()

        let output = "<|channel|>analysis<|message|>Bonjour<|end|>"
            + "<|start|>assistant<|channel|>final<|message|>Bonjour<|return|>"
        for token in tokenizer.encode(output, allowSpecial: true) {
            _ = parser.consume(token, session: &session)
        }

        #expect(session.isFinished)
        #expect(session.analysisText == "Bonjour")
        #expect(session.finalText == "Bonjour")
    }

    /// Le bug qui a motivé ce test : après `<|start|>`, le modèle écrit le nom du rôle.
    /// Sans traitement, « assistant » apparaissait en tête de chaque réponse affichée.
    @Test("Le nom du rôle n'est jamais compté comme du contenu")
    func roleNameIsNotContent() throws {
        let tokenizer = try Self.makeTokenizer()
        let parser = Harmony.Parser(tokenizer: tokenizer)
        var session = Harmony.Parser.Session()

        let output = "<|start|>assistant<|channel|>final<|message|>Bonjour<|return|>"
        for token in tokenizer.encode(output, allowSpecial: true) {
            _ = parser.consume(token, session: &session)
        }
        #expect(session.finalText == "Bonjour", "obtenu « \(session.finalText) »")
    }

    @Test("Chaque jeton d'arrêt termine la génération")
    func stopTokensFinish() throws {
        let tokenizer = try Self.makeTokenizer()
        for stop in Harmony.stopTokenNames {
            let parser = Harmony.Parser(tokenizer: tokenizer)
            var session = Harmony.Parser.Session()
            for token in tokenizer.encode(
                "<|channel|>final<|message|>Bonjour" + stop, allowSpecial: true)
            {
                _ = parser.consume(token, session: &session)
            }
            #expect(session.isFinished, "\(stop) ne termine pas la génération")
            #expect(session.finalText == "Bonjour")
        }
    }

    /// Un caractère accentué ou un emoji peut être réparti sur plusieurs jetons. Décoder
    /// jeton par jeton produirait des caractères de remplacement en plein mot.
    @Test("Le texte multi-octets se recompose sans caractère de remplacement")
    func multibyteTextIsReassembled() throws {
        let tokenizer = try Self.makeTokenizer()
        let parser = Harmony.Parser(tokenizer: tokenizer)
        var session = Harmony.Parser.Session()

        let content = "éàü — 🙂 日本語"
        for token in tokenizer.encode(
            "<|channel|>final<|message|>" + content + "<|return|>", allowSpecial: true)
        {
            _ = parser.consume(token, session: &session)
        }
        #expect(session.finalText == content)
        #expect(!session.finalText.contains("\u{FFFD}"), "caractère de remplacement présent")
    }

    @Test("Les fragments émis reconstituent exactement le texte final")
    func emittedFragmentsMatchFinalText() throws {
        let tokenizer = try Self.makeTokenizer()
        let parser = Harmony.Parser(tokenizer: tokenizer)
        var session = Harmony.Parser.Session()

        var streamed = ""
        for token in tokenizer.encode(
            "<|channel|>final<|message|>Bonjour le monde éàü<|return|>", allowSpecial: true)
        {
            for event in parser.consume(token, session: &session) {
                if case .text(.final, let fragment) = event { streamed += fragment }
            }
        }
        // L'affichage au fil de l'eau doit donner le même résultat que l'accumulation.
        #expect(streamed == session.finalText)
    }
}
