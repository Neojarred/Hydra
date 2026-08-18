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
                TokenSampler.largestIndices($0, count: k)
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
            TokenSampler.largestIndices($0, count: 50)
        }
        #expect(Set(selected) == Set(0..<10))
    }

    @Test("Equal values do not make the selection loop")
    func heapWithTies() {
        let values = [Float](repeating: 0.25, count: 5000)
        let selected = values.withUnsafeBufferPointer {
            TokenSampler.largestIndices($0, count: 64)
        }
        #expect(selected.count == 64)
        #expect(Set(selected).count == 64, "no duplicated index")
    }
}

/// Top-k and the presence penalty, the two controls the sampler did not have until M-069.
///
/// Neither is decoration. Qwen's model card asks for `top_k=20, presence_penalty=1.5`; the app
/// ran neither, and the visible consequence was the model emitting one word ninety times in a
/// row. The measurement that followed is worth carrying into these tests: **top-k alone did not
/// fix it and the presence penalty did**, so the penalty is the part that has to keep working.
@Suite("Top-k and the presence penalty")
struct TruncationTests {

    /// A distribution with a clear ranking: token `i` has logit `-i`, so 0 is the most likely.
    private func ramp(_ count: Int) -> [Float] {
        (0..<count).map { -Float($0) }
    }

    private func draw(
        _ values: [Float], _ sampling: ModelRunner.Sampling, count: Int = 400,
        sampler: inout TokenSampler
    ) -> [Int] {
        values.withUnsafeBufferPointer { buffer in
            (0..<count).map { _ in sampler.sample(from: buffer, using: sampling) }
        }
    }

    @Test("Top-k never draws a token outside the k most probable")
    func topKTruncates() {
        var sampler = TokenSampler()
        // A high temperature flattens the distribution, so without top-k the tail would be
        // drawn constantly. That is what makes this a test of the truncation and not of the
        // distribution's shape.
        let sampling = ModelRunner.Sampling(temperature: 4.0, topP: 1.0, topK: 5)
        let drawn = Set(draw(ramp(500), sampling, sampler: &sampler))
        #expect(drawn.allSatisfy { $0 < 5 }, "drew from outside the top 5: \(drawn.sorted())")
        #expect(drawn.count > 1, "top-k collapsed to a single token, so it proves nothing")
    }

    @Test("Without top-k the tail is reachable")
    func withoutTopKTheTailIsReachable() {
        var sampler = TokenSampler()
        let sampling = ModelRunner.Sampling(temperature: 4.0, topP: 1.0, topK: 0)
        let drawn = Set(draw(ramp(500), sampling, sampler: &sampler))
        #expect(drawn.contains { $0 >= 5 }, "the control never left the top 5 either")
    }

    /// The property the loop fix rests on: a token already emitted stops being the argmax.
    ///
    /// Greedy removes the randomness, so the sequence is arithmetic. The logits are one apart
    /// and the penalty is ten, so each emission puts its token permanently below every token
    /// that has not been emitted, and the decoder walks. A sampler that recorded nothing, or
    /// applied the penalty after truncation, would return 0 forever, which is the observed
    /// failure.
    @Test("The presence penalty walks a greedy decoder off a repeated token")
    func penaltyBreaksAGreedyLoop() {
        var sampler = TokenSampler()
        let sampling = ModelRunner.Sampling(temperature: 0, presencePenalty: 10)
        let drawn = draw(ramp(50), sampling, count: 4, sampler: &sampler)
        #expect(drawn == [0, 1, 2, 3], "greedy repeated itself: \(drawn)")
    }

    /// The penalty is **flat**, not cumulative: that is what "presence" means.
    ///
    /// Written down because the arithmetic is genuinely surprising and this test was first
    /// asserted the other way. With logits one apart and a penalty of 1.5, token 0 drops to
    /// -1.5 on its first emission and stays there however often it is drawn again, so once the
    /// alternatives fall past -1.5 it legitimately wins again: `0, 1, 0, 0`. A frequency
    /// penalty would give `0, 1, 2, 3`, and a frequency penalty is a different parameter that
    /// none of the three models here asks for.
    @Test("The penalty does not deepen with repetition")
    func penaltyIsFlatNotCumulative() {
        var sampler = TokenSampler()
        let sampling = ModelRunner.Sampling(temperature: 0, presencePenalty: 1.5)
        let drawn = draw(ramp(50), sampling, count: 4, sampler: &sampler)
        #expect(drawn == [0, 1, 0, 0], "the penalty is accumulating: \(drawn)")
    }

    @Test("With no penalty a greedy decoder does repeat, which is the control")
    func withoutPenaltyGreedyRepeats() {
        var sampler = TokenSampler()
        let drawn = draw(ramp(50), .greedy, count: 4, sampler: &sampler)
        #expect(drawn == [0, 0, 0, 0], "the control did not repeat, so the test above is void")
    }

    /// The penalty must survive truncation, which is the ordering bug worth guarding.
    ///
    /// Applying the penalty *after* top-k would leave the repeated token in the candidate set
    /// and at the top of it, so the loop would continue while every unit test on the penalty
    /// alone still passed.
    @Test("A penalized token loses to one that top-k would otherwise have excluded")
    func penaltyIsAppliedBeforeTruncation() {
        var sampler = TokenSampler()
        let sampling = ModelRunner.Sampling(
            temperature: 0, topP: 0.95, topK: 2, presencePenalty: 10)
        // Two draws: the first takes 0, the second must not, even though top-k = 2 keeps 0 and 1
        // and 0 is far ahead on the raw logits.
        let drawn = draw(ramp(50), sampling, count: 2, sampler: &sampler)
        #expect(drawn[0] == 0)
        #expect(drawn[1] != 0, "the penalty was applied after truncation, so it did nothing")
    }

    /// The penalty's history belongs to one answer, and the random stream belongs to the
    /// session. Conflating them is the bug this pins.
    ///
    /// `reset()` restarts the random stream, so it is deliberately never called between turns:
    /// a regeneration that reused the stream would return the same text. The presence penalty
    /// was added to that same sampler and cleared only there, so it accumulated for the life of
    /// the loaded model and every ordinary word picked up a permanent penalty.
    ///
    /// Every measurement missed it, because a measurement runs one generation in a fresh
    /// process where the history starts empty. It appears only across turns, which is the only
    /// place a user ever is.
    @Test("Beginning an answer forgets the penalty but not the random stream")
    func beginGenerationClearsOnlyTheHistory() {
        var sampler = TokenSampler()
        let greedyPenalised = ModelRunner.Sampling(temperature: 0, presencePenalty: 10)

        // A first answer walks off the tokens it has used.
        let first = draw(ramp(50), greedyPenalised, count: 3, sampler: &sampler)
        #expect(first == [0, 1, 2])

        // A second answer starts from a clean history, so it walks from the top again.
        sampler.beginGeneration()
        let second = draw(ramp(50), greedyPenalised, count: 3, sampler: &sampler)
        #expect(second == [0, 1, 2], "the penalty outlived the answer it belonged to: \(second)")

        // And the random stream is untouched, so two sampled answers still differ.
        var streamed = TokenSampler()
        let sampling = ModelRunner.Sampling(temperature: 1.0, topP: 0.95, topK: 20)
        let a = draw(ramp(500), sampling, count: 24, sampler: &streamed)
        streamed.beginGeneration()
        let b = draw(ramp(500), sampling, count: 24, sampler: &streamed)
        #expect(a != b, "beginning an answer restarted the random stream, so regenerate repeats")
    }

    @Test("Resetting forgets what was emitted")
    func resetClearsTheHistory() {
        var sampler = TokenSampler()
        let sampling = ModelRunner.Sampling(temperature: 0, presencePenalty: 10)
        _ = draw(ramp(50), sampling, count: 3, sampler: &sampler)
        sampler.reset()
        let after = draw(ramp(50), sampling, count: 1, sampler: &sampler)
        #expect(after == [0], "the penalty outlived the generation it belonged to")
    }

    /// A bounded window forgets, so a long answer is not pushed off its own vocabulary.
    @Test("A repeat window expires the penalty")
    func windowExpires() {
        var sampler = TokenSampler()
        let sampling = ModelRunner.Sampling(
            temperature: 0, presencePenalty: 10, repeatWindow: 2)
        // 0 is emitted, then penalized for two more draws, then free again.
        let drawn = draw(ramp(50), sampling, count: 4, sampler: &sampler)
        #expect(drawn == [0, 1, 2, 0], "the window did not expire: \(drawn)")
    }
}
