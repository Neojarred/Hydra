import Foundation

/// Tokeniseur BPE byte-level, compatible `o200k_harmony`.
///
/// Trois particularités du fichier de GPT-OSS conditionnent l'implémentation :
///
/// - **`ignore_merges = true`** : si un morceau pré-découpé figure tel quel dans le
///   vocabulaire, il est émis directement, **sans appliquer les fusions**. Sauter cette
///   règle donne un découpage différent sur de nombreux mots courants.
/// - **pré-découpage par expression régulière** avant toute fusion : les fusions ne
///   traversent jamais une frontière de morceau.
/// - **jetons spéciaux** (`<|start|>`, `<|message|>`, `<|channel|>`…) reconnus avant le
///   BPE : ce sont eux qui portent le format Harmony, et ils ne doivent jamais être
///   fusionnés avec du texte ordinaire.
public final class BPETokenizer: @unchecked Sendable {

    /// Motif de pré-découpage d'o200k, repris tel quel du fichier de tokeniseur.
    public static let pretokenPattern =
        #"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"#
        + #"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"#
        + #"|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"#

    /// Chaîne encodée en byte-level → identifiant.
    private let vocabulary: [String: Int]
    /// Identifiant → octets bruts, pré-décodés une fois pour toutes.
    private let pieces: [[UInt8]]
    /// Paire fusionnable → rang. Le rang le plus faible est appliqué en premier.
    private let mergeRanks: [Pair: Int]
    private let regex: NSRegularExpression

    public let specialTokens: [String: Int]
    private let specialByID: [Int: String]
    private let specialRegex: NSRegularExpression?

    public var count: Int { pieces.count }

    struct Pair: Hashable {
        let left: String
        let right: String
    }

    public enum TokenizerError: Error, CustomStringConvertible {
        case malformedFile(String)
        case unknownToken(String)

        public var description: String {
            switch self {
            case .malformedFile(let detail): return "fichier de tokeniseur illisible : \(detail)"
            case .unknownToken(let token): return "jeton inconnu : \(token)"
            }
        }
    }

    public init(
        vocabulary: [String: Int], merges: [(String, String)], specialTokens: [String: Int]
    ) throws {
        self.vocabulary = vocabulary
        self.specialTokens = specialTokens
        self.specialByID = Dictionary(
            uniqueKeysWithValues: specialTokens.map { ($0.value, $0.key) })

        var ranks: [Pair: Int] = [:]
        ranks.reserveCapacity(merges.count)
        for (rank, merge) in merges.enumerated() {
            ranks[Pair(left: merge.0, right: merge.1)] = rank
        }
        self.mergeRanks = ranks

        // Table inverse, pré-décodée : le décodage est sur le chemin chaud de la
        // génération, il ne doit pas refaire la conversion byte-level à chaque token.
        let highest = max(
            vocabulary.values.max() ?? 0, specialTokens.values.max() ?? 0)
        var table = [[UInt8]](repeating: [], count: highest + 1)
        for (piece, id) in vocabulary { table[id] = ByteLevel.decode(piece) }
        for (piece, id) in specialTokens { table[id] = Array(piece.utf8) }
        self.pieces = table

        self.regex = try NSRegularExpression(pattern: Self.pretokenPattern)

        if specialTokens.isEmpty {
            self.specialRegex = nil
        } else {
            let alternatives = specialTokens.keys
                .sorted { $0.count > $1.count }  // le plus long gagne
                .map { NSRegularExpression.escapedPattern(for: $0) }
                .joined(separator: "|")
            self.specialRegex = try NSRegularExpression(pattern: alternatives)
        }
    }

    // MARK: - Encodage

    /// Encode du texte.
    ///
    /// - Parameter allowSpecial: si vrai, les séquences comme `<|start|>` présentes dans
    ///   le texte sont reconnues comme jetons spéciaux. À laisser **faux** pour du texte
    ///   fourni par l'utilisateur : sinon un message contenant `<|end|>` pourrait
    ///   interrompre la conversation ou usurper un rôle.
    public func encode(_ text: String, allowSpecial: Bool = false) -> [Int] {
        guard allowSpecial, let specialRegex else {
            return encodeOrdinary(text)
        }

        var out: [Int] = []
        let full = text as NSString
        var cursor = 0
        specialRegex.enumerateMatches(
            in: text, range: NSRange(location: 0, length: full.length)
        ) { match, _, _ in
            guard let match else { return }
            if match.range.location > cursor {
                let before = full.substring(
                    with: NSRange(location: cursor, length: match.range.location - cursor))
                out.append(contentsOf: encodeOrdinary(before))
            }
            let token = full.substring(with: match.range)
            if let id = specialTokens[token] { out.append(id) }
            cursor = match.range.location + match.range.length
        }
        if cursor < full.length {
            out.append(contentsOf: encodeOrdinary(
                full.substring(with: NSRange(location: cursor, length: full.length - cursor))))
        }
        return out
    }

    private func encodeOrdinary(_ text: String) -> [Int] {
        guard !text.isEmpty else { return [] }
        var out: [Int] = []
        let full = text as NSString

        regex.enumerateMatches(
            in: text, range: NSRange(location: 0, length: full.length)
        ) { match, _, _ in
            guard let match, match.range.length > 0 else { return }
            let piece = full.substring(with: match.range)
            let encoded = ByteLevel.encode(Array(piece.utf8))
            out.append(contentsOf: applyMerges(encoded))
        }
        return out
    }

    /// Applique les fusions BPE à un morceau déjà converti en byte-level.
    private func applyMerges(_ piece: String) -> [Int] {
        // `ignore_merges` : un morceau présent tel quel dans le vocabulaire est émis
        // directement. Sans cette règle, « hello » se découperait alors qu'il existe.
        if let id = vocabulary[piece] { return [id] }

        var symbols = piece.map { String($0) }
        guard symbols.count > 1 else {
            return vocabulary[piece].map { [$0] } ?? []
        }

        while symbols.count > 1 {
            var bestRank = Int.max
            var bestIndex = -1
            for index in 0..<(symbols.count - 1) {
                let pair = Pair(left: symbols[index], right: symbols[index + 1])
                if let rank = mergeRanks[pair], rank < bestRank {
                    bestRank = rank
                    bestIndex = index
                }
            }
            guard bestIndex >= 0 else { break }
            symbols[bestIndex] += symbols[bestIndex + 1]
            symbols.remove(at: bestIndex + 1)
        }

        // Un symbole absent du vocabulaire ne devrait pas exister — le byte-level garantit
        // que tout octet a sa représentation. On l'ignore plutôt que d'échouer.
        return symbols.compactMap { vocabulary[$0] }
    }

    // MARK: - Décodage

    /// Reconstitue le texte d'une suite d'identifiants.
    ///
    /// Le décodage passe par les **octets** avant l'UTF-8 : un caractère multi-octets peut
    /// être réparti sur plusieurs jetons, et décoder jeton par jeton produirait des
    /// caractères de remplacement au milieu des mots accentués ou des emoji.
    public func decode(_ ids: [Int], skipSpecial: Bool = false) -> String {
        var bytes: [UInt8] = []
        bytes.reserveCapacity(ids.count * 4)
        for id in ids {
            guard id >= 0, id < pieces.count else { continue }
            if skipSpecial, specialByID[id] != nil { continue }
            bytes.append(contentsOf: pieces[id])
        }
        return String(decoding: bytes, as: UTF8.self)
    }

    /// Octets bruts d'un jeton, pour un décodage incrémental sûr.
    public func bytes(for id: Int) -> [UInt8] {
        guard id >= 0, id < pieces.count else { return [] }
        return pieces[id]
    }

    public func isSpecial(_ id: Int) -> Bool { specialByID[id] != nil }
    public func name(of id: Int) -> String? { specialByID[id] }
    public func id(of token: String) -> Int? { specialTokens[token] ?? vocabulary[token] }
}
