import Testing
@testable import HydraMetal

/// Partial selection must return **exactly** the same nucleus as a full sort.
///
/// This is an optimization that touches token sampling: if it shifts the nucleus by even
/// one rank, the model changes behaviour with nothing to signal it.
@Suite("Top-p sampling")
struct SamplingTests {

    /// A reproducible generator: a sampling test that varies from one run to the next proves
    /// nothing.
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

    @Test("The heap returns exactly the k largest, in order")
    func heapMatchesFullSort() {
        for (count, k) in [(1000, 16), (1000, 256), (50_000, 256), (201_088, 1024)] {
            let values = pseudoRandom(count: count, seed: UInt64(count &* 31 &+ k))
            let selected = values.withUnsafeBufferPointer {
                ModelRunner.largestIndices($0, count: k)
            }
            let expected = Array(
                Array(0..<count).sorted { values[$0] > values[$1] }.prefix(k))

            #expect(selected.count == k, "k=\(k) over \(count) entries")
            #expect(selected.map { values[$0] } == expected.map { values[$0] },
                    "the retained values must be the same, in the same order")
        }
    }

    @Test("A request wider than the vocabulary returns everything")
    func heapLargerThanInput() {
        let values = pseudoRandom(count: 10, seed: 7)
        let selected = values.withUnsafeBufferPointer {
            ModelRunner.largestIndices($0, count: 50)
        }
        #expect(Set(selected) == Set(0..<10))
    }

    @Test("Equal values do not make the selection loop")
    func heapWithTies() {
        let values = [Float](repeating: 0.25, count: 5000)
        let selected = values.withUnsafeBufferPointer {
            ModelRunner.largestIndices($0, count: 64)
        }
        #expect(selected.count == 64)
        #expect(Set(selected).count == 64, "no duplicated index")
    }
}
