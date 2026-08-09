import Foundation
import HydraCore
import HydraFormat
import Testing

@testable import HydraMetal

/// Reusing the KV cache from one conversation turn to the next.
///
/// Two properties must hold, and violating them would be **silent**: the model would go on
/// answering, simply from a false context.
///
/// 1. Giving sliding-window layers linear storage must not widen their attention. Storage
///    and window are two distinct notions; conflating them would turn those layers into
///    full-attention ones.
/// 2. Rewinding then resuming must give exactly the same state as a full computation.
@Suite("KV cache reuse")
struct KVReuseTests {

    private func makeModel() async throws -> (URL, MetalContext, ModelMapping, URL) {
        let config = GptOssConfig.tiny
        let temporary = FileManager.default.temporaryDirectory
            .appending(path: "hydra-kv-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
        let root = try await LayerRunnerTests.installTinyModel(at: temporary)
        let context = try MetalContext()
        let mapping = try ModelMapping(root: root, model: config, device: context.device)
        return (root, context, mapping, temporary)
    }

    private func makeRunner(
        root: URL, context: MetalContext, mapping: ModelMapping, contextLength: Int
    ) throws -> ModelRunner {
        let config = GptOssConfig.tiny
        let cache = ExpertSlotCache(
root: root, model: config,
            slotsPerLayer: config.expertsPerToken, device: context.device)
        return try ModelRunner(
            config: config, context: context, mapping: mapping,
            expertCache: cache, contextLength: contextLength, prefillChunk: 16)
    }

    private func divergence(_ a: [Float], _ b: [Float]) -> Float {
        var scale: Float = 0
        for value in a { scale = max(scale, abs(value)) }
        var worst: Float = 0
        for (x, y) in zip(a, b) { worst = max(worst, abs(x - y) / max(scale, 1e-6)) }
        return worst
    }

    /// The prompt exceeds the sliding window (8): if linear storage widened attention, tokens
    /// beyond the window would enter the computation and the deviation would be massive.
    ///
    @Test("Linear storage windows exactly like the ring")
    func linearStorageKeepsWindow() async throws {
        let (root, context, mapping, temporary) = try await makeModel()
        defer { try? FileManager.default.removeItem(at: temporary) }

        let prompt = (0..<20).map { ($0 * 7 + 3) % GptOssConfig.tiny.vocabSize }

        // Short context: sliding layers on linear storage.
        let linear = try makeRunner(
            root: root, context: context, mapping: mapping, contextLength: 64)
        #expect(linear.canRewind, "a short context must give a rewindable cache")
        let linearOutput = Array(try linear.prefill(tokens: prompt))

        // Context beyond the threshold: sliding layers take their ring back.
        let ringed = try makeRunner(
            root: root, context: context, mapping: mapping,
            contextLength: KVCache.linearWindowLimit + 1)
        #expect(!ringed.canRewind, "a ring must not declare itself rewindable")
        let ringedOutput = Array(try ringed.prefill(tokens: prompt))

        let worst = divergence(ringedOutput, linearOutput)
        #expect(worst < 2e-3, "relative deviation \(worst) between ring and linear storage")
    }

    /// The real scenario: one turn, one answer, then a follow-up turn whose prompt extends the
    /// previous one.
    @Test("Rewinding then resuming equals a full computation")
    func rewindMatchesFullPrefill() async throws {
        let (root, context, mapping, temporary) = try await makeModel()
        defer { try? FileManager.default.removeItem(at: temporary) }

        let vocab = GptOssConfig.tiny.vocabSize
        let firstTurn = (0..<12).map { ($0 * 7 + 3) % vocab }
        let generated = (0..<9).map { ($0 * 13 + 5) % vocab }
        let secondTurn = (0..<6).map { ($0 * 5 + 11) % vocab }

        // What the model actually went through on the first turn, answer included.
        let fed = firstTurn + generated
        let followUp = fed + secondTurn

        // Reused path: replay the first turn, then resume at the common prefix.
        let reused = try makeRunner(
            root: root, context: context, mapping: mapping, contextLength: 64)
        _ = try reused.prefill(tokens: firstTurn)
        for token in generated { _ = try reused.forward(token: token, needsLogits: false) }
        #expect(reused.position == fed.count)

        let common = commonPrefixLength(fed, followUp)
        #expect(common == fed.count, "the follow-up prompt must extend what was processed")
        reused.rewind(to: common)
        #expect(reused.position == common)
        let reusedOutput = Array(try reused.prefill(tokens: Array(followUp.dropFirst(common))))

        // Fresh path: everything recomputed from scratch.
        let fresh = try makeRunner(
            root: root, context: context, mapping: mapping, contextLength: 64)
        let freshOutput = Array(try fresh.prefill(tokens: followUp))

        let worst = divergence(freshOutput, reusedOutput)
        #expect(worst < 2e-3, "relative deviation \(worst) between resumption and full computation")

        let bestFresh = freshOutput.firstIndex(of: freshOutput.max()!)
        let bestReused = reusedOutput.firstIndex(of: reusedOutput.max()!)
        #expect(bestFresh == bestReused, "the greedy token differs")
    }

    /// A prompt that **diverges** from the cache must restart at the point of divergence, not
    /// reuse keys that no longer correspond to it. This is the edited-message case.
    @Test("A divergent prompt resumes at the right place")
    func divergentPromptRestartsAtDivergence() async throws {
        let (root, context, mapping, temporary) = try await makeModel()
        defer { try? FileManager.default.removeItem(at: temporary) }

        let vocab = GptOssConfig.tiny.vocabSize
        let original = (0..<14).map { ($0 * 7 + 3) % vocab }
        // Same beginning, different ending — a message edited at the eighth token.
        var edited = Array(original[0..<8])
        edited += (0..<6).map { ($0 * 3 + 41) % vocab }

        let common = commonPrefixLength(original, edited)
        #expect(common == 8, "the common prefix must stop at the divergence")

        let reused = try makeRunner(
            root: root, context: context, mapping: mapping, contextLength: 64)
        _ = try reused.prefill(tokens: original)
        reused.rewind(to: common)
        let reusedOutput = Array(try reused.prefill(tokens: Array(edited.dropFirst(common))))

        let fresh = try makeRunner(
            root: root, context: context, mapping: mapping, contextLength: 64)
        let freshOutput = Array(try fresh.prefill(tokens: edited))

        let worst = divergence(freshOutput, reusedOutput)
        #expect(worst < 2e-3, "relative deviation \(worst) after resuming on an edited prompt")
    }
}
