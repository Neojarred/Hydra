import Foundation
import HydraCore
import HydraFormat
import HydraInstall
import HydraReference
import Metal
import Testing

@testable import HydraMetal

/// The whole Qwen model against a CPU oracle built from the installed bytes.
///
/// `QwenModelTests` asks whether the model behaves like a model: finite, spread, sensitive to
/// its prefix, the same through both entry points. Ten deliberate wiring faults were injected
/// against it and **eight survived**, because every one of them preserves all four properties.
/// A head that reads the embedding table, `A_log` resolved from `dt_bias`, an expert's `up`
/// served where its `gate` belongs: each gives a model that runs and answers.
///
/// So this suite compares against an answer computed independently, in double precision, by
/// dequantizing the very bytes the installation contains. A disagreement is a wiring fault and
/// never a quantization loss, because both sides start from the same integers.
@Suite("Qwen 3.5 MoE against a CPU oracle")
struct QwenOracleTests {

    private let config = Qwen35MoeConfig.tiny
    private let fixture = QwenFixture()
    private var oracle: QwenOracleBuilder { QwenOracleBuilder(config: config) }

    // MARK: - The test

    /// Four tokens through the whole model, GPU against oracle.
    ///
    /// Four rather than one, because a single token exercises neither the recurrent state's
    /// carry, the convolution window's fill, nor the attention history.
    @Test("The model agrees with the oracle over a sequence")
    func matchesOracle() async throws {
        let root = try fixture.temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let installed = try await fixture.install(at: root, config: config)

        let context = try MetalContext()
        let mapping = try ModelMapping(root: installed, model: config, device: context.device)
        let cache = ExpertSlotCache(
            root: installed, model: config, slotsPerLayer: config.expertCount,
            device: context.device)
        let runner = try ModelRuntime.makeRunner(
            model: config, context: context, mapping: mapping,
            expertCache: cache, contextLength: 32)

        let tokens = [3, 17, 42, 8]
        let got = Array(try runner.prefill(tokens: tokens))

        let built = try oracle.buildOracle(mapping: mapping, cache: cache)
        let expected = oracle.oracleLogits(built, tokens: tokens)

        #expect(got.count == expected.count)
        var worst = 0.0
        var at = 0
        for i in 0..<expected.count where abs(Double(got[i]) - expected[i]) > worst {
            worst = abs(Double(got[i]) - expected[i])
            at = i
        }
        // Float32 on the GPU against double on the CPU, through eight layers and a 256-wide
        // head. The tolerance is on the accumulated arithmetic, not on the weights: both sides
        // decoded the same integers.
        #expect(worst < 5e-3, "logit \(at) differs by \(worst)")
    }

    /// The embedding row the runner reads is the row the checkpoint holds.
    @Test("An embedding row matches the oracle's decoding")
    func embeddingRowMatches() async throws {
        let root = try fixture.temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let installed = try await fixture.install(at: root, config: config)

        let context = try MetalContext()
        let mapping = try ModelMapping(root: installed, model: config, device: context.device)
        let weights = Qwen35MoeWeights(config: config, mapping: mapping)

        var row = [Float](repeating: 0, count: config.hiddenSize)
        row.withUnsafeMutableBufferPointer { weights.readEmbedding(token: 11, into: $0) }

        let expected: [Double] = try mapping.resident.withBytes { raw in
            let s = (
                try mapping.residentTensor("\(config.embeddingStem).weight").offset,
                try mapping.residentTensor("\(config.embeddingStem).scales").offset,
                try mapping.residentTensor("\(config.embeddingStem).biases").offset)
            return oracle.matrix(
                raw, words: s.0, scales: s.1, biases: s.2,
                rows: config.vocabSize, cols: config.hiddenSize, bits: config.quantBits)[11]
        }

        for i in 0..<config.hiddenSize {
            #expect(abs(Double(row[i]) - expected[i]) < 1e-6, "column \(i)")
        }
        #expect(row.contains { $0 != 0 }, "and not the all-zero row a missing tensor gives")
    }
}
