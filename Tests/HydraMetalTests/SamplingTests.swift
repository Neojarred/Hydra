import Testing
@testable import HydraMetal

/// La sélection partielle doit rendre **exactement** le même noyau qu'un tri complet.
///
/// C'est une optimisation qui touche au tirage des jetons : si elle décale le noyau ne
/// serait-ce que d'un rang, le modèle change de comportement sans que rien ne le signale.
@Suite("Échantillonnage top-p")
struct SamplingTests {

    /// Générateur reproductible : un test d'échantillonnage qui varie d'une exécution à
    /// l'autre ne prouve rien.
    private func pseudoRandom(count: Int, seed: UInt64) -> [Float] {
        var state = seed | 1
        return (0..<count).map { _ in
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return Float(Double((z ^ (z >> 31)) % 1_000_000) / 1_000_000.0)
        }
    }

    @Test("Le tas rend exactement les k plus grandes, dans l'ordre")
    func heapMatchesFullSort() {
        for (count, k) in [(1000, 16), (1000, 256), (50_000, 256), (201_088, 1024)] {
            let values = pseudoRandom(count: count, seed: UInt64(count &* 31 &+ k))
            let selected = values.withUnsafeBufferPointer {
                ModelRunner.largestIndices($0, count: k)
            }
            let expected = Array(
                Array(0..<count).sorted { values[$0] > values[$1] }.prefix(k))

            #expect(selected.count == k, "k=\(k) sur \(count) entrées")
            #expect(selected.map { values[$0] } == expected.map { values[$0] },
                    "les valeurs retenues doivent être les mêmes, dans le même ordre")
        }
    }

    @Test("Une demande plus large que le vocabulaire rend tout")
    func heapLargerThanInput() {
        let values = pseudoRandom(count: 10, seed: 7)
        let selected = values.withUnsafeBufferPointer {
            ModelRunner.largestIndices($0, count: 50)
        }
        #expect(Set(selected) == Set(0..<10))
    }

    @Test("Les valeurs égales ne font pas boucler la sélection")
    func heapWithTies() {
        let values = [Float](repeating: 0.25, count: 5000)
        let selected = values.withUnsafeBufferPointer {
            ModelRunner.largestIndices($0, count: 64)
        }
        #expect(selected.count == 64)
        #expect(Set(selected).count == 64, "aucun indice dupliqué")
    }
}
