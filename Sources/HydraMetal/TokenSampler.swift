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

    /// The whole of the sampler's random state.
    private var state: UInt64 = 0

    /// Tokens emitted since the last reset, and how many times each, for the presence penalty.
    ///
    /// The sampler records what it returns rather than being told, so no caller threads a
    /// history through. It holds only what this sampler produced: penalizing the prompt's
    /// tokens too would push the model away from the words of the question it was asked, which
    /// is a different and worse failure than the one being fixed.
    ///
    /// Counted rather than a set, so a bounded window can evict correctly, and so a frequency
    /// penalty has what it needs if one is ever wanted.
    private var emitted: [Int: Int] = [:]
    /// Emission order, kept only when a window bounds the penalty.
    private var order: [Int] = []

    /// Restarts the pseudo-random sequence.
    ///
    /// Deliberately distinct from resetting a runner's caches: two successive generations on
    /// the same prompt must be able to differ, otherwise "regenerate" would always return the
    /// same text. Used to make a measurement reproducible.
    public mutating func reset() {
        state = 0
        beginGeneration()
    }

    /// Forgets what has been emitted, without disturbing the pseudo-random sequence.
    ///
    /// **Two different lifetimes live in this type and they are not the same one.** The random
    /// stream must survive from one generation to the next, or "regenerate" would return the
    /// same text every time, which is why `reset()` is deliberately not called between turns.
    /// The presence penalty's history must *not* survive: it describes one answer.
    ///
    /// Conflating them is a bug this file shipped. The penalty was added to a sampler whose
    /// `reset()` nothing ever calls, so `emitted` accumulated for the whole life of the loaded
    /// model. By the fifth turn of a conversation every ordinary word, "the", "is", the subject
    /// of the conversation itself, carried a permanent 1.5 penalty, and the model was pushed off
    /// its own vocabulary a little further with every answer.
    ///
    /// It never showed in testing because every measurement runs one generation in a fresh
    /// process, where the history starts empty and the fault cannot appear. It only shows in a
    /// long conversation, which is the only place a user ever is.
    public mutating func beginGeneration() {
        emitted.removeAll(keepingCapacity: true)
        order.removeAll(keepingCapacity: true)
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
        let token = draw(from: distribution, using: sampling)
        guard sampling.presencePenalty > 0 || sampling.frequencyPenalty > 0 else { return token }
        emitted[token, default: 0] += 1
        if sampling.repeatWindow > 0 {
            order.append(token)
            while order.count > sampling.repeatWindow {
                let evicted = order.removeFirst()
                if let count = emitted[evicted] {
                    if count <= 1 { emitted.removeValue(forKey: evicted) }
                    else { emitted[evicted] = count - 1 }
                }
            }
        }
        return token
    }

    /// The penalty owed by a token, given what has already been emitted.
    ///
    /// Flat, not proportional to the count: that is what "presence" means, and it is what the
    /// model cards specify. A count-proportional term is a *frequency* penalty and a different
    /// parameter, which none of the three models here asks for.
    private func penalty(_ index: Int, _ sampling: ModelRunner.Sampling) -> Float {
        guard let count = emitted[index] else { return 0 }
        // Flat, then count-proportional. `presence` is what the model cards specify and is paid
        // once; `frequency` escalates and is what a stable loop needs to be charged for, since
        // a penalty a loop has already paid cannot break it.
        return sampling.presencePenalty
            + sampling.frequencyPenalty * Float(count)
    }

    private mutating func draw(
        from distribution: UnsafeBufferPointer<Float>,
        using sampling: ModelRunner.Sampling
    ) -> Int {
        guard sampling.temperature > 0 else {
            // Greedy still owes the penalty, otherwise "temperature 0" would be the one mode
            // that loops, which is the opposite of what a user asking for determinism wants.
            guard sampling.presencePenalty > 0, !emitted.isEmpty else {
                return Self.greedyToken(from: distribution)
            }
            var bestIndex = 0
            var bestValue = -Float.greatestFiniteMagnitude
            for index in 0..<distribution.count {
                let value = distribution[index] - penalty(index, sampling)
                if value > bestValue { bestValue = value; bestIndex = index }
            }
            return bestIndex
        }

        // --- Truncation, in the order HuggingFace applies it: penalty, top-k, then top-p ---
        //
        // The order matters and is not a convention. Top-p is defined over whatever survives
        // top-k, so a nucleus computed against the full vocabulary is a different set from the
        // one the model's own `generation_config.json` describes.
        //
        // Nothing is allocated at the size of the vocabulary. The candidates are pulled with a
        // bounded heap in one pass, and only those are materialized.
        let penalized = (sampling.presencePenalty > 0 || sampling.frequencyPenalty > 0)
            && !emitted.isEmpty
        let wanted = sampling.topK > 0 ? sampling.topK : 64

        // A penalty can only lower a logit, so it can promote a token by at most as many ranks
        // as there are penalized tokens. Taking that many extra candidates makes the penalized
        // top-k exact rather than approximate, which matters because the whole point of the
        // penalty is to let a token that was *not* winning win.
        func pool(_ count: Int) -> [Int] {
            let widened = penalized ? count + emitted.count : count
            let raw = Self.largestIndices(distribution, count: widened)
            guard penalized else { return raw }
            return Array(
                raw.sorted { a, b in
                    let va = distribution[a] - penalty(a, sampling)
                    let vb = distribution[b] - penalty(b, sampling)
                    return va == vb ? a < b : va > vb
                }.prefix(count))
        }

        func weight(_ index: Int, _ peak: Float, _ inverseTemperature: Float) -> Float {
            exp((distribution[index] - penalty(index, sampling) - peak) * inverseTemperature)
        }

        if state == 0 { state = sampling.seed | 1 }
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        z = z ^ (z >> 31)
        let uniform = Float(Double(z % 1_000_000) / 1_000_000.0)
        let inverseTemperature = 1 / sampling.temperature

        // --- Untruncated: no top-k, no top-p, no penalty. The path GPT-OSS asks for ---
        if sampling.topK <= 0, sampling.topP >= 1.0, !penalized {
            var peak = -Float.greatestFiniteMagnitude
            for value in distribution { peak = max(peak, value) }
            var total: Float = 0
            for value in distribution { total += exp((value - peak) * inverseTemperature) }
            let target = uniform * total
            var cumulative: Float = 0
            for (index, value) in distribution.enumerated() {
                cumulative += exp((value - peak) * inverseTemperature)
                if cumulative >= target { return index }
            }
            return distribution.count - 1
        }

        // --- Truncated: build the candidate set, widening until top-p is satisfied ---
        var limit = wanted
        var candidates = pool(limit)
        guard let best = candidates.first else { return 0 }
        let peak = distribution[best] - penalty(best, sampling)

        var nucleus: [Int] = []
        var mass: Float = 0
        while true {
            // Top-p is relative to the mass that survived top-k, so the denominator is the
            // candidate set and not the vocabulary.
            var available: Float = 0
            for index in candidates { available += weight(index, peak, inverseTemperature) }

            nucleus = []
            mass = 0
            var reached = sampling.topP >= 1.0
            for index in candidates {
                nucleus.append(index)
                mass += weight(index, peak, inverseTemperature)
                if !reached, mass / available >= sampling.topP { reached = true; break }
            }
            // A fixed top-k is the whole answer: there is nothing to widen to.
            if reached || sampling.topK > 0 || limit >= distribution.count { break }
            limit = min(limit * 4, distribution.count)
            candidates = pool(limit)
        }

        let target = uniform * mass
        var cumulative: Float = 0
        for index in nucleus {
            cumulative += weight(index, peak, inverseTemperature)
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
