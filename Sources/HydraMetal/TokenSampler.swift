import Foundation

/// Drawing a token from a distribution, independently of which model produced it.
///
/// Lifted out of `ModelRunner` unchanged when Gemma needed the same behaviour. The alternative
/// was a second copy of the nucleus logic, which is precisely the kind of duplication that
/// lets two architectures drift apart on something the user would experience as "Gemma is
/// worse at high temperature" rather than as a bug.
///
/// The state is a single 64-bit word: a SplitMix64 sequence seeded from `Sampling.seed` on
/// first use. It is a value type, so each runner owns its own stream, resetting one runner's
/// sampler cannot perturb another's.
public struct TokenSampler {

    public init() {}

    /// The whole of the sampler's mutable state.
    private var state: UInt64 = 0

    /// Restarts the pseudo-random sequence.
    ///
    /// Deliberately distinct from resetting a runner's caches: two successive generations on
    /// the same prompt must be able to differ, otherwise "regenerate" would always return the
    /// same text. Used to make a measurement reproducible.
    public mutating func reset() {
        state = 0
    }

    /// The most probable token: argmax, with the first index winning a tie.
    ///
    /// Tie-breaking is part of the contract, not an implementation detail. Two runs of the
    /// same model on the same prompt must agree token for token, and a distribution with two
    /// equal maxima is common enough near the start of a generation that "whichever we happen
    /// to see last" would make greedy decoding non-reproducible.
    public static func greedyToken(from distribution: UnsafeBufferPointer<Float>) -> Int {
        var bestIndex = 0
        var bestValue = -Float.greatestFiniteMagnitude
        for (index, value) in distribution.enumerated() where value > bestValue {
            bestValue = value
            bestIndex = index
        }
        return bestIndex
    }

    /// Draws a token from the distribution.
    ///
    /// `temperature = 0` switches to greedy decoding, which makes the run reproducible,
    /// indispensable for checking that a change in cache size does not alter outputs.
    public mutating func sample(
        from distribution: UnsafeBufferPointer<Float>,
        using sampling: ModelRunner.Sampling
    ) -> Int {
        guard sampling.temperature > 0 else { return Self.greedyToken(from: distribution) }

        // Nothing is allocated at the size of the vocabulary.
        //
        // The previous version built two arrays of 201,088 entries on every token, the
        // probabilities and the ordering, 2.4 MiB to allocate, fill and discard, before
        // sorting the lot to keep about thirty. Here the sum is computed in one pass with no
        // storage, and only the retained candidates are materialized.
        // The maximum falls out of the selection itself: the largest logit is the first
        // candidate. One full pass over the vocabulary saved.
        let candidates = sampling.topP < 1.0
            ? Self.largestIndices(distribution, count: 64) : []
        var peak = -Float.greatestFiniteMagnitude
        if let best = candidates.first {
            peak = distribution[best]
        } else {
            for value in distribution { peak = max(peak, value) }
        }

        let inverseTemperature = 1 / sampling.temperature
        var total: Float = 0
        for value in distribution { total += exp((value - peak) * inverseTemperature) }

        if state == 0 { state = sampling.seed | 1 }
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        z = z ^ (z >> 31)
        let uniform = Float(Double(z % 1_000_000) / 1_000_000.0)

        guard sampling.topP < 1.0 else {
            // Without truncation a simple walk suffices: the enumeration order does not
            // change the distribution being sampled.
            let target = uniform * total
            var cumulative: Float = 0
            for (index, value) in distribution.enumerated() {
                cumulative += exp((value - peak) * inverseTemperature)
                if cumulative >= target { return index }
            }
            return distribution.count - 1
        }

        // The top-p nucleus holds only a handful of tokens. We extract a small batch,
        // widened only if the target mass is not reached.
        var limit = 64
        var nucleus: [Int] = []
        var mass: Float = 0
        var pool = candidates
        while true {
            nucleus = []
            mass = 0
            var reached = false
            for index in pool {
                nucleus.append(index)
                mass += exp((distribution[index] - peak) * inverseTemperature)
                if mass / total >= sampling.topP { reached = true; break }
            }
            if reached || limit >= distribution.count { break }
            limit = min(limit * 4, distribution.count)
            pool = Self.largestIndices(distribution, count: limit)
        }

        let target = uniform * mass
        var cumulative: Float = 0
        for index in nucleus {
            cumulative += exp((distribution[index] - peak) * inverseTemperature)
            if cumulative >= target { return index }
        }
        return nucleus.last ?? 0
    }

    /// The `count` largest values, in descending order.
    ///
    /// A min-heap of size `count`: a single pass over the vocabulary, and the only
    /// allocation is the heap itself, a few dozen entries, not two hundred thousand.
    public static func largestIndices(
        _ values: UnsafeBufferPointer<Float>, count: Int
    ) -> [Int] {
        let size = min(count, values.count)
        guard size > 0 else { return [] }

        var heapValues = [Float](repeating: 0, count: size)
        var heapIndices = [Int](repeating: 0, count: size)
        var filled = 0

        func siftDown(_ start: Int) {
            var parent = start
            while true {
                let left = 2 * parent + 1
                guard left < filled else { return }
                var smallest = left
                let right = left + 1
                if right < filled, heapValues[right] < heapValues[left] { smallest = right }
                if heapValues[parent] <= heapValues[smallest] { return }
                heapValues.swapAt(parent, smallest)
                heapIndices.swapAt(parent, smallest)
                parent = smallest
            }
        }

        for index in 0..<values.count {
            let value = values[index]
            if filled < size {
                heapValues[filled] = value
                heapIndices[filled] = index
                filled += 1
                // Sift up from the inserted leaf.
                var child = filled - 1
                while child > 0 {
                    let parent = (child - 1) / 2
                    if heapValues[parent] <= heapValues[child] { break }
                    heapValues.swapAt(parent, child)
                    heapIndices.swapAt(parent, child)
                    child = parent
                }
            } else if value > heapValues[0] {
                heapValues[0] = value
                heapIndices[0] = index
                siftDown(0)
            }
        }

        return heapIndices[0..<filled].sorted { values[$0] > values[$1] }
    }
}
