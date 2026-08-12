import Foundation
import HydraCore
import HydraFormat
import Testing

@testable import HydraMetal

/// The dispatch seam of D-023, exercised through the protocol rather than the concrete type.
///
/// The point is not that `ModelRunner` works, other suites cover that, but that **a caller
/// can drive a model without knowing which one it is**. Every test here holds
/// `any TextModelRunner` and never mentions `ModelRunner`, so if a member ever stops being
/// reachable through the protocol, this stops compiling rather than quietly forcing the next
/// caller to branch.
@Suite("Model runner dispatch")
struct TextModelRunnerTests {

    private func makeRunner() async throws -> (any TextModelRunner, URL) {
        let config = GptOssConfig.tiny
        let temporary = FileManager.default.temporaryDirectory
            .appending(path: "hydra-dispatch-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
        let root = try await LayerRunnerTests.installTinyModel(at: temporary)
        let context = try MetalContext()
        let mapping = try ModelMapping(root: root, model: config, device: context.device)
        let cache = ExpertSlotCache(
            root: root, model: config, slotsPerLayer: config.expertsPerToken,
            device: context.device)
        let runner = try ModelRunner(
            config: config, context: context, mapping: mapping,
            expertCache: cache, contextLength: 64, prefillChunk: 16)
        return (runner, temporary)
    }

    /// A whole turn driven through the protocol: prompt, distribution, decode, stop.
    @Test("A caller can prefill and decode without naming the architecture")
    func drivesAModelBlind() async throws {
        let (runner, temporary) = try await makeRunner()
        defer { try? FileManager.default.removeItem(at: temporary) }

        let prompt = (0..<8).map { ($0 * 7 + 3) % GptOssConfig.tiny.vocabSize }
        var distribution = try runner.prefill(tokens: prompt)
        #expect(runner.position == prompt.count)

        var produced: [Int] = []
        for _ in 0..<6 {
            let token = runner.greedyToken(from: distribution)
            produced.append(token)
            distribution = try runner.forward(token: token, needsLogits: true)
        }
        #expect(produced.count == 6)
        #expect(runner.position == prompt.count + 6)
        #expect(distribution.allSatisfy { $0.isFinite })
    }

    /// Rewinding is what makes a conversation turn's work reusable, and a caller must be able
    /// to ask for it without knowing whose cache it is.
    @Test("Rewinding is reachable through the protocol")
    func rewindsBlind() async throws {
        let (runner, temporary) = try await makeRunner()
        defer { try? FileManager.default.removeItem(at: temporary) }

        let prompt = (0..<10).map { ($0 * 5 + 1) % GptOssConfig.tiny.vocabSize }
        _ = try runner.prefill(tokens: prompt)
        #expect(runner.canRewind)

        runner.rewind(to: 4)
        #expect(runner.position == 4)
        runner.reset()
        #expect(runner.position == 0)
    }

    /// Speculative decoding goes through the protocol too, so a second architecture inherits
    /// it rather than reimplementing it.
    @Test("Batched verification is reachable through the protocol")
    func verifiesBlind() async throws {
        let (runner, temporary) = try await makeRunner()
        defer { try? FileManager.default.removeItem(at: temporary) }

        _ = try runner.prefill(tokens: [1, 2, 3, 4])
        let logits = try runner.verify(tokens: [5, 6, 7])
        #expect(logits.count == 3)
        #expect(logits.allSatisfy { $0.count == GptOssConfig.tiny.vocabSize })
    }

    /// The one branch D-023 permits: choosing a prompt format. It must be answered by the
    /// runner rather than inferred from a config the caller had to keep.
    @Test("The runner reports its own architecture")
    func reportsItsArchitecture() async throws {
        let (runner, temporary) = try await makeRunner()
        defer { try? FileManager.default.removeItem(at: temporary) }
        #expect(runner.architecture == .gptOss)
    }

    /// Memory reporting must survive the abstraction: it is what the app shows and what D-012
    /// is argued on.
    @Test("Reserved memory is reported through the protocol")
    func reportsMemoryBlind() async throws {
        let (runner, temporary) = try await makeRunner()
        defer { try? FileManager.default.removeItem(at: temporary) }
        #expect(runner.reservedBytes > 0)
        #expect(runner.kvCache.byteCount > 0)
        #expect(runner.expertCache.reservedBytes > 0)
    }
}
