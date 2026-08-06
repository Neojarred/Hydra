/// Propose des jetons candidats en cherchant dans ce qui a déjà été écrit.
///
/// Le décodage spéculatif a besoin d'une source de brouillons **bon marché** : le gain
/// vient de vérifier plusieurs jetons en une passe, donc le brouillon doit coûter
/// nettement moins que la vérification. Un second modèle ne convient pas ici — le 20B
/// n'est que 4,5 fois moins cher que le 120B, il en faudrait dix.
///
/// Cette source-ci ne coûte rien : on cherche la dernière occurrence des `n` derniers
/// jetons dans l'historique, et on propose ce qui suivait. Elle est muette en conversation
/// ouverte et très efficace dès que la réponse reprend le contexte — résumé, réécriture,
/// code, questions sur un document joint.
///
/// Un brouillon faux ne coûte que la vérification, qui aurait eu lieu de toute façon ;
/// il ne peut donc jamais changer la sortie, seulement le temps mis à l'obtenir.
public struct NGramDrafter: Sendable {

    /// Longueurs de motif essayées, de la plus spécifique à la plus permissive.
    ///
    /// Un motif long se trompe rarement mais trouve rarement ; un motif court trouve
    /// souvent et se trompe souvent. On prend la première correspondance en partant du
    /// plus long : c'est le meilleur compromis sans coût de recherche notable.
    public let patternLengths: [Int]
    /// Nombre de jetons proposés par tentative.
    public let draftLength: Int

    public init(patternLengths: [Int] = [3, 2], draftLength: Int = 4) {
        self.patternLengths = patternLengths
        self.draftLength = draftLength
    }

    /// Jetons proposés pour la suite de `history`, ou vide si rien ne correspond.
    ///
    /// La recherche part de la fin : la reprise la plus récente est la plus probable.
    public func propose(history: [Int]) -> [Int] {
        guard !history.isEmpty else { return [] }

        for length in patternLengths where history.count > length {
            let pattern = Array(history.suffix(length))
            // On s'arrête avant la fin : le motif final est celui qu'on cherche à
            // prolonger, pas un précédent utilisable.
            var start = history.count - length - 1
            while start >= 0 {
                if Array(history[start..<(start + length)]) == pattern {
                    let from = start + length
                    guard from < history.count else { break }
                    let upper = min(from + draftLength, history.count)
                    let draft = Array(history[from..<upper])
                    if !draft.isEmpty { return draft }
                }
                start -= 1
            }
        }
        return []
    }
}
