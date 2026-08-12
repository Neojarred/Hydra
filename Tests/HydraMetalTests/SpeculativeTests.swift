import Foundation
import HydraCore
import Testing

@testable import HydraMetal

/// Speculative decoding.
///
/// The promise is strong: produce the same tokens, faster. A faulty verifier would not
/// crash, it would produce plausible, wrong text with nothing to signal it. These tests
/// therefore compare the full token sequence, not an approximate distribution.
@Suite("Speculative decoding")
struct SpeculativeTests {

    private func makeRunner() async throws -> (ModelRunner, URL) {
        let config = GptOssConfig.tiny
        let temporary = FileManager.default.temporaryDirectory
            .appending(path: "hydra-spec-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
        let root = try await LayerRunnerTests.installTinyModel(at: temporary)
        let context = try MetalContext()
        let mapping = try ModelMapping(root: root, model: config, device: context.device)
        let cache = ExpertSlotCache(
root: root, model: config,
            slotsPerLayer: config.expertsPerToken, device: context.device)
        let runner = try ModelRunner(
            config: config, context: context, mapping: mapping,
            expertCache: cache, contextLength: 128, prefillChunk: 16)
        return (runner, temporary)
    }

    /// Generates `count` tokens, with or without drafts, and returns the sequence produced.
    ///
    /// The draft is deliberately **wrong in places**: it is the mix of acceptances and
    /// rejections that exercises the rewind.
    private func generate(
        _ runner: ModelRunner, prompt: [Int], count: Int,
        sampling: ModelRunner.Sampling, drafts: [[Int]]
    ) throws -> [Int] {
        runner.reset()
        // Sampling must restart from the same state, otherwise we compare two different
        // pseudo-random sequences rather than two decoding paths.
        runner.resetSampling()
        var distribution = try runner.prefill(tokens: prompt)
        var produced: [Int] = []
        var round = 0
        while produced.count < count {
            let draft = round < drafts.count ? drafts[round] : []
            let result = try runner.step(
                from: distribution, draft: draft, sampling: sampling)
            produced += result.tokens
            distribution = result.next
            round += 1
        }
        return Array(produced.prefix(count))
    }

    @Test("A draft does not change the sequence produced, greedy")
    func greedyMatchesReference() async throws {
        let (runner, temporary) = try await makeRunner()
        defer { try? FileManager.default.removeItem(at: temporary) }

        let prompt = (0..<10).map { ($0 * 7 + 3) % GptOssConfig.tiny.vocabSize }
        let greedy = ModelRunner.Sampling(temperature: 0, topP: 1)

        let reference = try generate(
            runner, prompt: prompt, count: 24, sampling: greedy, drafts: [])

        // Replay proposing the true sequence in slices: everything must be accepted.
        var perfect: [[Int]] = []
        var index = 0
        while index < reference.count {
            perfect.append(Array(reference[index..<min(index + 4, reference.count)]))
            index += 4
        }
        let withPerfectDrafts = try generate(
            runner, prompt: prompt, count: 24, sampling: greedy, drafts: perfect)
        #expect(withPerfectDrafts == reference, "a correct draft must be transparent")

        // Then with partly wrong drafts, to exercise rejection.
        let wrong = perfect.map { block -> [Int] in
            guard block.count > 1 else { return block }
            var copy = block
            copy[1] = (copy[1] + 1) % GptOssConfig.tiny.vocabSize
            return copy
        }
        let withWrongDrafts = try generate(
            runner, prompt: prompt, count: 24, sampling: greedy, drafts: wrong)
        #expect(withWrongDrafts == reference, "a wrong draft must have no effect")

        // And with entirely wrong drafts: the fallback path, spending nothing.
        let garbage = (0..<12).map { round in
            (0..<4).map { ($0 * 31 + round * 17 + 5) % GptOssConfig.tiny.vocabSize }
        }
        let withGarbage = try generate(
            runner, prompt: prompt, count: 24, sampling: greedy, drafts: garbage)
        #expect(withGarbage == reference, "a nonsensical draft must have no effect")
    }

    /// Stochastic sampling is the delicate case: every emitted token must consume exactly one
    /// draw, otherwise the sequences diverge despite an identical seed.
    @Test("A draft does not change the sequence produced, sampled")
    func sampledMatchesReference() async throws {
        let (runner, temporary) = try await makeRunner()
        defer { try? FileManager.default.removeItem(at: temporary) }

        let prompt = (0..<10).map { ($0 * 5 + 11) % GptOssConfig.tiny.vocabSize }
        let sampling = ModelRunner.Sampling(temperature: 0.8, topP: 0.9, seed: 12345)

        let reference = try generate(
            runner, prompt: prompt, count: 20, sampling: sampling, drafts: [])

        var blocks: [[Int]] = []
        var index = 0
        while index < reference.count {
            blocks.append(Array(reference[index..<min(index + 3, reference.count)]))
            index += 3
        }
        let speculated = try generate(
            runner, prompt: prompt, count: 20, sampling: sampling, drafts: blocks)
        #expect(speculated == reference, "the sampled sequence must be identical at equal seed")
    }

    @Test("The batched pass returns one set of logits per position")
    func verifyReturnsPerPositionLogits() async throws {
        let (runner, temporary) = try await makeRunner()
        defer { try? FileManager.default.removeItem(at: temporary) }

        let prompt = (0..<6).map { ($0 * 3 + 1) % GptOssConfig.tiny.vocabSize }
        runner.reset()
        _ = try runner.prefill(tokens: prompt)

        let batch = [4, 9, 2]
        let logits = try runner.verify(tokens: batch)
        #expect(logits.count == batch.count)
        for row in logits {
            #expect(row.count == GptOssConfig.tiny.vocabSize)
            #expect(row.allSatisfy { $0.isFinite }, "non-finite logits signal a bug")
        }
        #expect(runner.position == prompt.count + batch.count)
    }
}
