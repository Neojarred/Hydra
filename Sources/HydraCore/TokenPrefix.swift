/// Longueur du plus long préfixe commun à deux suites de jetons.
///
/// Sert à décider ce qu'un cache KV peut conserver d'un tour de conversation au suivant.
/// Vit ici plutôt que dans le moteur pour que le runtime puisse en être testé sans
/// dépendre de l'application.
public func commonPrefixLength(_ a: [Int], _ b: [Int]) -> Int {
    var index = 0
    let limit = min(a.count, b.count)
    while index < limit && a[index] == b[index] { index += 1 }
    return index
}
