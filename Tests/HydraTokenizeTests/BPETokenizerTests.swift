import Foundation
import Testing

@testable import HydraTokenize

/// Deux niveaux de validation.
///
/// 1. Un tokeniseur **synthétique**, minuscule et entièrement maîtrisé, vérifie
///    l'algorithme : ordre des fusions, règle `ignore_merges`, jetons spéciaux,
///    conversion byte-level.
/// 2. Le **vrai** `o200k_harmony`, s'il est installé, vérifie que l'implémentation
///    concorde avec le vocabulaire réel : entrées connues, aller-retour sur du texte
///    varié, couverture intégrale des octets.
struct BPETokenizerTests {

    // MARK: - Table byte-level

    @Test("La table byte-level couvre les 256 octets, sans collision")
    func byteLevelIsBijective() {
        #expect(ByteLevel.encodeTable.count == 256)
        #expect(Set(ByteLevel.encodeTable).count == 256, "deux octets partagent un caractère")
        for byte in 0..<256 {
            #expect(ByteLevel.decodeTable[ByteLevel.encodeTable[byte]] == UInt8(byte))
        }
    }

    @Test("L'espace devient Ġ, comme dans le vocabulaire publié")
    func spaceMapsToKnownCharacter() {
        // C'est ce qui explique les « Ġ » du vocabulaire : sans cette conversion, un
        // vocabulaire lu paraît illisible et ne concorde avec rien.
        #expect(ByteLevel.encode([0x20]) == "Ġ")
        #expect(ByteLevel.encode([0x0A]) == "Ċ")
        #expect(ByteLevel.encode(Array("hello".utf8)) == "hello")
        #expect(ByteLevel.encode(Array(" hello".utf8)) == "Ġhello")
    }

    @Test("Tout octet fait l'aller-retour, y compris hors UTF-8 valide")
    func byteLevelRoundTrip() {
        let bytes: [UInt8] = (0..<256).map(UInt8.init)
        #expect(ByteLevel.decode(ByteLevel.encode(bytes)) == bytes)
    }

    // MARK: - Tokeniseur synthétique

    /// Vocabulaire jouet : les octets isolés, quelques fusions, et un mot complet qui
    /// n'est atteignable que par la règle `ignore_merges`.
    static func makeSynthetic() throws -> BPETokenizer {
        var vocabulary: [String: Int] = [:]
        var next = 0
        for byte in 0..<256 {
            vocabulary[String(ByteLevel.encodeTable[byte])] = next
            next += 1
        }
        // Fusions, dans l'ordre des rangs.
        let merges: [(String, String)] = [
            ("a", "b"),      // rang 0
            ("ab", "c"),     // rang 1
            ("Ġ", "a"),      // rang 2
        ]
        for (left, right) in merges {
            vocabulary[left + right] = next
            next += 1
        }
        // Entrée présente telle quelle mais **inatteignable par fusion** : seule la règle
        // ignore_merges permet de l'émettre en un jeton.
        vocabulary["abcd"] = next
        next += 1

        return try BPETokenizer(
            vocabulary: vocabulary, merges: merges,
            specialTokens: ["<|start|>": 9000, "<|end|>": 9001])
    }

    @Test("Les fusions s'appliquent par rang croissant")
    func mergesApplyByRank() throws {
        let tokenizer = try Self.makeSynthetic()
        // « abc » : d'abord a+b (rang 0), puis ab+c (rang 1) — un seul jeton.
        #expect(tokenizer.encode("abc").count == 1)
        #expect(tokenizer.decode(tokenizer.encode("abc")) == "abc")
    }

    @Test("ignore_merges : une entrée présente telle quelle est émise directement")
    func ignoreMergesTakesPrecedence() throws {
        let tokenizer = try Self.makeSynthetic()
        // « abcd » existe au vocabulaire. Sans ignore_merges il donnerait « abc » + « d ».
        let ids = tokenizer.encode("abcd")
        #expect(ids.count == 1, "obtenu \(ids.count) jetons : ignore_merges n'est pas appliqué")
        #expect(tokenizer.decode(ids) == "abcd")
    }

    @Test("Les jetons spéciaux ne sont reconnus que si on l'autorise")
    func specialTokensAreOptIn() throws {
        let tokenizer = try Self.makeSynthetic()
        let text = "a<|end|>b"

        let withSpecial = tokenizer.encode(text, allowSpecial: true)
        #expect(withSpecial.contains(9001))

        // Par défaut, un message utilisateur contenant « <|end|> » ne doit pas pouvoir
        // interrompre la conversation ni usurper un rôle.
        let withoutSpecial = tokenizer.encode(text, allowSpecial: false)
        #expect(!withoutSpecial.contains(9001))
        #expect(tokenizer.decode(withoutSpecial) == text)
    }

    @Test("Le décodage passe par les octets, pas par les jetons")
    func decodeJoinsBytes() throws {
        let tokenizer = try Self.makeSynthetic()
        // Un caractère multi-octets réparti sur plusieurs jetons doit se recomposer.
        for text in ["é", "→", "🙂", "café crème", "日本語"] {
            #expect(tokenizer.decode(tokenizer.encode(text)) == text, Comment(rawValue: text))
        }
    }

    // MARK: - Le vrai tokeniseur, s'il est installé

    nonisolated(unsafe) static let installedTokenizer: BPETokenizer? = {
        guard let base = try? FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask,
            appropriateFor: nil, create: false)
        else { return nil }
        let compact = base.appending(
            path: "Hydra/Models/gpt-oss-20b.hydra/tokenizer/\(TokenizerFile.compactFileName)")
        guard FileManager.default.fileExists(atPath: compact.path) else { return nil }
        return try? TokenizerFile.loadCompact(at: compact)
    }()

    @Test("o200k : les entrées connues du vocabulaire donnent un seul jeton")
    func realTokenizerKnownPieces() throws {
        guard let tokenizer = Self.installedTokenizer else { return }
        // Identifiants relevés directement dans le tokenizer.json publié.
        let expected: [(String, Int)] = [
            ("hello", 24912),
            (" hello", 40617),
            (" the", 290),
            ("the", 3086),
            (",", 11),
            (" (", 350),
            ("\n\n", 279),
            (" Bonjour", 141691),
        ]
        for (text, id) in expected {
            let ids = tokenizer.encode(text)
            #expect(ids == [id], "« \(text) » → \(ids), attendu [\(id)]")
        }
    }

    @Test("o200k : aller-retour exact sur du texte varié")
    func realTokenizerRoundTrip() throws {
        guard let tokenizer = Self.installedTokenizer else { return }
        let corpus = [
            "Hello, world!",
            "Bonjour, comment ça va aujourd'hui ?",
            "Les élèves ont déjà terminé l'exercice n°42.",
            "func main() { print(\"salut\") }",
            "    indentation\n\ttabulation\n\n\ndoubles sauts",
            "emoji 🙂🚀 et symboles ∑∫≈",
            "日本語のテキスト",
            "1234567890 3.14159 -1e-9",
            "<|start|>ceci n'est pas un jeton spécial<|end|>",
            "",
        ]
        for text in corpus {
            let ids = tokenizer.encode(text)
            #expect(tokenizer.decode(ids) == text, "aller-retour cassé sur « \(text) »")
        }
    }

    @Test("o200k : les jetons couvrent exactement les octets de l'entrée")
    func realTokenizerCoversInput() throws {
        guard let tokenizer = Self.installedTokenizer else { return }
        let text = "Le renard brun rapide saute par-dessus le chien paresseux — 42 fois."
        let ids = tokenizer.encode(text)

        var rebuilt: [UInt8] = []
        for id in ids { rebuilt.append(contentsOf: tokenizer.bytes(for: id)) }
        #expect(rebuilt == Array(text.utf8), "la segmentation perd ou duplique des octets")
        #expect(!ids.isEmpty)
    }

    @Test("o200k : les jetons Harmony sont présents et distincts")
    func realTokenizerHarmonySpecials() throws {
        guard let tokenizer = Self.installedTokenizer else { return }
        let required = [
            "<|start|>", "<|end|>", "<|message|>", "<|channel|>",
            "<|return|>", "<|call|>", "<|constrain|>", "<|endoftext|>",
        ]
        var seen = Set<Int>()
        for name in required {
            let id = try #require(tokenizer.specialTokens[name], "jeton \(name) absent")
            #expect(seen.insert(id).inserted, "deux jetons spéciaux partagent l'identifiant \(id)")
            #expect(tokenizer.isSpecial(id))
        }
        #expect(tokenizer.specialTokens["<|start|>"] == 200006)
        #expect(tokenizer.specialTokens["<|message|>"] == 200008)
        #expect(tokenizer.specialTokens["<|return|>"] == 200002)
    }

    @Test("o200k : un aller-retour du vocabulaire lui-même reste stable")
    func realTokenizerVocabularyIsStable() throws {
        guard let tokenizer = Self.installedTokenizer else { return }
        // Un échantillon d'identifiants répartis sur tout le vocabulaire : décoder puis
        // réencoder doit rendre le même identifiant, dès lors que le morceau constitue
        // un pré-jeton complet.
        var checked = 0
        var stable = 0
        for id in stride(from: 100, to: 199_000, by: 4_999) {
            let text = tokenizer.decode([id])
            guard !text.isEmpty, tokenizer.decode(tokenizer.encode(text)) == text else { continue }
            checked += 1
            if tokenizer.encode(text) == [id] { stable += 1 }
        }
        #expect(checked > 20, "échantillon trop petit : \(checked)")
        // Tous les morceaux ne sont pas des pré-jetons complets ; on exige une large
        // majorité, pas la totalité.
        #expect(Double(stable) / Double(checked) > 0.75,
                "seulement \(stable)/\(checked) identifiants stables")
    }
}
