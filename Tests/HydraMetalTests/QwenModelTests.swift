import Foundation
import HydraCore
import HydraFormat
import HydraInstall
import Metal
import Testing

@testable import HydraMetal

/// The whole Qwen model, from a synthetic checkpoint through the real install plan to logits.
///
/// Every piece below this has been checked against the CPU reference: the delta rule, the
/// convolution, the gated norm, the attention block, the mixture, the layer order. None of that
/// proves the pieces were **wired** to each other, and the failures wiring produces are exactly
/// the ones nothing else catches. A linear layer handed the attention layer's norm, `A_log`
/// resolved from `dt_bias`, the head reading the embedding table because Gemma ties them: each
/// produces a model that runs and returns finite numbers.
///
/// The tiny config keeps the architecture and shrinks the size: still three linear layers to
/// one attending, still a shared expert beside routed ones, still an untied head.
@Suite("Qwen 3.5 MoE model on GPU")
struct QwenModelTests {

    private let config = Qwen35MoeConfig.tiny
    private let fixture = QwenFixture()

    private func makeRunner(at installed: URL) throws -> (MetalContext, any TextModelRunner) {
        let context = try MetalContext()
        let mapping = try ModelMapping(root: installed, model: config, device: context.device)
        let cache = ExpertSlotCache(
            root: installed, model: config, slotsPerLayer: config.expertsPerToken,
            device: context.device)
        return (
            context,
            try ModelRuntime.makeRunner(
                model: config, context: context, mapping: mapping,
                expertCache: cache, contextLength: 64)
        )
    }

    // MARK: - The tests

    @Test("A synthetic Qwen checkpoint installs")
    func installs() async throws {
        let root = try fixture.temporary()
        defer { try? FileManager.default.removeItem(at: root) }

        let installed = try await fixture.install(at: root, config: config)
        let manifest = try HydraManifest.read(from: installed)
        #expect(manifest.model.architecture == ModelArchitecture.qwen35Moe.rawValue)
        try manifest.validate(against: config, root: installed)
        #expect(manifest.vision?.isEmpty == false, "the tower is installed (D-021)")
    }

    /// The model runs, and the result is a distribution rather than debris.
    ///
    /// Finiteness alone proves nothing here: every wiring fault this test exists to catch
    /// produces finite numbers. The spread is what says the head saw signal.
    @Test("The model produces a usable distribution")
    func producesADistribution() async throws {
        let root = try fixture.temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let installed = try await fixture.install(at: root, config: config)
        let (_, runner) = try makeRunner(at: installed)

        let logits = try runner.prefill(tokens: [1, 2, 3, 4, 5])
        #expect(logits.count == config.vocabSize)
        #expect(logits.allSatisfy { $0.isFinite })

        let minimum = logits.min() ?? 0
        let maximum = logits.max() ?? 0
        let zeros = logits.reduce(into: 0) { count, value in if value == 0 { count += 1 } }
        let diagnosis = zeros == logits.count
            ? "all zero, so the head's work did not run"
            : "not all zero, so the head ran on a degenerate input"
        let report = "the distribution is flat: min \(minimum), max \(maximum), "
            + "\(zeros) of \(logits.count) exactly zero, \(diagnosis)"
        #expect(maximum - minimum > 1e-4, "\(report)")
    }

    /// Position matters, which is the cheapest evidence that the recurrent state is carried.
    ///
    /// Three quarters of the layers here have no cache at all: their entire memory of the
    /// prompt is the state buffer the delta rule mutates. A runner that allocated it, ran every
    /// kernel and never carried it between tokens would produce a perfectly finite, perfectly
    /// spread distribution that ignored everything before the current token. This asks whether
    /// the same token after a different prefix predicts something different.
    @Test("The prefix changes the prediction")
    func contextIsCarried() async throws {
        let root = try fixture.temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let installed = try await fixture.install(at: root, config: config)

        func distribution(after prefix: [Int]) throws -> [Float] {
            let (_, runner) = try makeRunner(at: installed)
            return Array(try runner.prefill(tokens: prefix + [7]))
        }

        let short = try distribution(after: [3])
        let long = try distribution(after: [3, 11, 42, 8, 19, 5])
        let difference = zip(short, long).map { abs($0 - $1) }.max() ?? 0
        #expect(difference > 1e-5, "the same token after a different prefix predicts the same")
        
    }

    /// Decoding one token at a time equals asking for the same tokens as a prompt.
    ///
    /// The two paths differ only in whether logits are computed at each position, and they must
    /// not differ in anything else. This is what would catch a `forward` that advances the
    /// recurrent state twice, or a `prefill` that skips the last token's state update.
    @Test("Prefill and step-by-step decoding agree")
    func prefillMatchesDecoding() async throws {
        let root = try fixture.temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let installed = try await fixture.install(at: root, config: config)
        let tokens = [4, 9, 2, 14]

        let (_, batched) = try makeRunner(at: installed)
        let fromPrefill = Array(try batched.prefill(tokens: tokens))

        let (_, stepped) = try makeRunner(at: installed)
        var last: [Float] = []
        for token in tokens { last = Array(try stepped.forward(token: token, needsLogits: true)) }

        let difference = zip(fromPrefill, last).map { abs($0 - $1) }.max() ?? .infinity
        #expect(difference < 1e-4, "the two paths diverge by \(difference)")
        #expect(batched.position == stepped.position)
    }

    /// Chunked prefill must equal feeding the tokens one at a time, across a chunk boundary.
    ///
    /// The chunked path reorders the work: a layer's experts are read once for the whole chunk
    /// instead of once a token, the recurrence's token loop moves inside its kernel, and the
    /// projections run batched. None of that is allowed to change the answer, and each of them
    /// can change it quietly.
    ///
    /// The chunk is set to four against ten tokens, so the run crosses two boundaries. That is
    /// the part with no equivalent in Gemma: the recurrent state and the convolution window are
    /// carried from one chunk to the next, and a run that fitted in one chunk would not test it.
    @Test("Chunked prefill matches token-by-token decoding across chunk boundaries")
    func chunkedPrefillMatchesDecoding() async throws {
        let root = try fixture.temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let installed = try await fixture.install(at: root, config: config)
        let tokens = [4, 9, 2, 14, 7, 21, 3, 18, 5, 11]

        let context = try MetalContext()
        func runner(chunk: Int) throws -> Qwen35MoeRunner {
            let mapping = try ModelMapping(
                root: installed, model: config, device: context.device)
            let cache = ExpertSlotCache(
                root: installed, model: config, slotsPerLayer: config.expertsPerToken,
                device: context.device)
            return try Qwen35MoeRunner(
                config: config, context: context, mapping: mapping,
                expertCache: cache, contextLength: 64, prefillChunk: chunk)
        }

        let batched = try runner(chunk: 4)
        // The chunk actually took effect. Without this the test claims to cross two boundaries
        // and silently does not: the parameter was ignored for a day and this passed anyway,
        // because a single chunk and three chunks agree by construction.
        #expect(batched.prefillChunkTokens == 4, "the requested chunk was ignored")
        #expect(tokens.count > 2 * batched.prefillChunkTokens, "and the run crosses two")
        let fromPrefill = Array(try batched.prefill(tokens: tokens))

        let stepped = try runner(chunk: 4)
        var last: [Float] = []
        for token in tokens { last = Array(try stepped.forward(token: token, needsLogits: true)) }

        var worst: Float = 0
        var at = 0
        for i in 0..<min(fromPrefill.count, last.count)
        where abs(fromPrefill[i] - last[i]) > worst {
            worst = abs(fromPrefill[i] - last[i])
            at = i
        }
        #expect(fromPrefill.count == last.count)
        #expect(worst < 2e-3, "logit \(at) differs by \(worst)")
        #expect(batched.position == stepped.position)
    }

    /// A follow-up turn resumes from the last turn boundary, not from nothing.
    ///
    /// This is the property a conversation lives on. The engine asks for the longest prefix it
    /// can resume from; asking `canRewind` instead is all-or-nothing, and for a recurrent model
    /// the answer is almost always "no" because the common prefix lands a few tokens past the
    /// checkpoint. A thousand-token conversation would then reprocess itself every turn.
    @Test("A later turn resumes from the previous turn's boundary")
    func reusesThePreviousTurn() async throws {
        let root = try fixture.temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let installed = try await fixture.install(at: root, config: config)
        let (_, runner) = try makeRunner(at: installed)

        // Turn one: a prompt, then a few generated tokens.
        _ = try runner.prefill(tokens: [1, 2, 3, 4, 5, 6])
        for token in [7, 8, 9] { _ = try runner.forward(token: token, needsLogits: false) }
        #expect(runner.position == 9)

        // Turn two shares all nine tokens. The exact position is reachable because it is the
        // current one, so nothing is reprocessed.
        #expect(runner.reusablePrefix(atMost: 9) == 9)

        // And if the shared prefix stops short of the current position, as it does whenever a
        // turn's rendered history differs by a token from what was generated, the answer is the
        // checkpoint at 6 rather than zero.
        #expect(runner.reusablePrefix(atMost: 8) == 6, "snaps down to the turn boundary")
        #expect(runner.reusablePrefix(atMost: 7) == 6)
        #expect(runner.reusablePrefix(atMost: 5) == 0, "nothing was checkpointed below the turn")

        // Resuming there leaves the runner where it says it does.
        runner.rewind(to: runner.reusablePrefix(atMost: 8))
        #expect(runner.position == 6)
    }

    /// The recurrent state cannot be rewound to a position it did not checkpoint, and the
    /// runner must refuse rather than answer from a state describing tokens the caller believes
    /// it discarded.
    @Test("Rewinding is offered only where both memories can go")
    func rewindRespectsBothMemories() async throws {
        let root = try fixture.temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let installed = try await fixture.install(at: root, config: config)
        let (_, runner) = try makeRunner(at: installed)

        _ = try runner.prefill(tokens: [1, 2, 3, 4])
        #expect(runner.canRewind(to: 4), "the turn boundary was checkpointed")
        #expect(!runner.canRewind(to: 2), "no checkpoint there, and no way to un-sum a decay")

        // Appending needs no rewind, so the current position is always reachable. That is the
        // common case and the reason this limitation does not affect ordinary conversation:
        // it applies to editing history, not to continuing it.
        _ = try runner.forward(token: 5, needsLogits: false)
        #expect(runner.position == 5)
        #expect(runner.canRewind(to: 5), "resuming where we are restores nothing")
        #expect(runner.canRewind(to: 4), "and the turn's checkpoint survives the decode")
        #expect(!runner.canRewind(to: 3), "but nothing between them was ever saved")

        runner.rewind(to: 2)
        #expect(runner.position == 5, "a refused rewind leaves the runner alone")
    }
}

/// Prefilling a prompt that carries an image.
///
/// The image path reaches the model through two seams and no others: the per-token embedding
/// closure the chunked prefill already had, and the rotary's three position axes. Everything
/// else, the recurrence, the mixture, the attention, the head, cannot tell an image token from
/// a text one. These tests pin that.
@Suite("Qwen multimodal prefill")
struct QwenMultimodalPrefillTests {

    private let config = Qwen35MoeConfig.tiny
    private let fixture = QwenFixture()

    private func makeRunner(at installed: URL) throws -> Qwen35MoeRunner {
        let context = try MetalContext()
        let mapping = try ModelMapping(root: installed, model: config, device: context.device)
        let cache = ExpertSlotCache(
            root: installed, model: config, slotsPerLayer: config.expertsPerToken,
            device: context.device)
        return try Qwen35MoeRunner(
            config: config, context: context, mapping: mapping,
            expertCache: cache, contextLength: 256, prefillChunk: 4)
    }

    /// A prompt of nothing but text must take the ordinary path, bit for bit.
    ///
    /// The multimodal entry point is the one the app will call for every message, image or not,
    /// so a text-only prompt going through it has to be indistinguishable. It short-circuits to
    /// `prefill(tokens:)` for exactly this reason, and this is what holds that in place.
    @Test("A text-only prompt is identical through either entry point")
    func textOnlyIsIdentical() async throws {
        let root = try fixture.temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let installed = try await fixture.install(at: root, config: config)
        let tokens = [4, 9, 2, 14, 7, 21, 3]

        let plain = Array(try makeRunner(at: installed).prefill(tokens: tokens))
        let viaElements = Array(
            try makeRunner(at: installed).prefill(elements: [.text(tokens)]))

        #expect(plain.count == viaElements.count)
        let worst = zip(plain, viaElements).map { abs($0 - $1) }.max() ?? .infinity
        #expect(worst == 0, "the two entry points differ by \(worst)")
    }

    /// An image's embeddings reach the model, and the model's answer depends on them.
    ///
    /// Finiteness is not the check: a runner that silently dropped the image would be perfectly
    /// finite. Changing the image must change the logits, and it must do so **through the
    /// image**, so the same token count with different values is the comparison.
    @Test("The image's embeddings change what the model predicts")
    func imageReachesTheModel() async throws {
        let root = try fixture.temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let installed = try await fixture.install(at: root, config: config)

        let tokens = config.hiddenSize
        func run(scale: Float) throws -> [Float] {
            let values = (0..<(6 * tokens)).map { Float($0 % 13) * 0.01 * scale - 0.05 }
            return Array(try makeRunner(at: installed).prefill(elements: [
                .text([1, 2, 3]),
                .image(embeddings: values, frames: 1, height: 2, width: 3),
                .text([4, 5]),
            ]))
        }

        let one = try run(scale: 1)
        let other = try run(scale: -3)
        #expect(one.allSatisfy { $0.isFinite })
        #expect(one.count == config.vocabSize)

        let difference = zip(one, other).map { abs($0 - $1) }.max() ?? 0
        #expect(difference > 1e-6, "the image was ignored: changing it changed nothing")
    }

    /// An image straddling a chunk boundary is the normal case, not an edge one.
    ///
    /// A chunk is 512 tokens in production and an image is a thousand, so every real image is
    /// split across chunks. Here the chunk is 4 and the image is 6, so it spans a boundary, and
    /// the answer must not depend on where that boundary falls.
    @Test("An image split across chunks gives the same answer as one that is not")
    func imageAcrossAChunkBoundary() async throws {
        let root = try fixture.temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let installed = try await fixture.install(at: root, config: config)
        let values = (0..<(6 * config.hiddenSize)).map { Float($0 % 11) * 0.01 - 0.05 }

        let context = try MetalContext()
        func runner(chunk: Int) throws -> Qwen35MoeRunner {
            let mapping = try ModelMapping(
                root: installed, model: config, device: context.device)
            let cache = ExpertSlotCache(
                root: installed, model: config, slotsPerLayer: config.expertsPerToken,
                device: context.device)
            return try Qwen35MoeRunner(
                config: config, context: context, mapping: mapping,
                expertCache: cache, contextLength: 256, prefillChunk: chunk)
        }
        let elements: [Qwen35MoeRunner.PromptElement] = [
            .text([1, 2, 3]),
            .image(embeddings: values, frames: 1, height: 2, width: 3),
            .text([4, 5]),
        ]

        let split = Array(try runner(chunk: 4).prefill(elements: elements))
        let whole = Array(try runner(chunk: 64).prefill(elements: elements))
        let worst = zip(split, whole).map { abs($0 - $1) }.max() ?? .infinity
        #expect(worst < 2e-3, "the chunk boundary changed the answer by \(worst)")
    }

    /// **The image's shape reaches the rotary**, not just its contents.
    ///
    /// Added because replacing the three position axes with plain linear positions failed none
    /// of the tests above: every one of them compares two runs against each other, and both arms
    /// then share the same wrong positions. Nothing could see it.
    ///
    /// A 2x3 grid and a 3x2 grid hold the same six tokens with the same embeddings, and differ
    /// only in which row and column each one sits at. Under mRoPE that is a different rotary and
    /// a different answer; under linear positions both are 3, 4, 5, 6, 7, 8 and the two runs are
    /// identical. So this separates them and nothing else does.
    @Test("Transposing an image's grid changes the answer")
    func gridShapeReachesTheRotary() async throws {
        let root = try fixture.temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let installed = try await fixture.install(at: root, config: config)
        let values = (0..<(6 * config.hiddenSize)).map { Float($0 % 7) * 0.02 - 0.06 }

        func run(height: Int, width: Int) throws -> [Float] {
            Array(try makeRunner(at: installed).prefill(elements: [
                .text([1, 2, 3]),
                .image(embeddings: values, frames: 1, height: height, width: width),
            ]))
        }
        let tall = try run(height: 3, width: 2)
        let wide = try run(height: 2, width: 3)

        let difference = zip(tall, wide).map { abs($0 - $1) }.max() ?? 0
        #expect(
            difference > 1e-6,
            "a transposed grid gave the same answer, so rows and columns never reached the rotary")
    }

    /// The position counter advances by the image's longer side, so the model's own idea of
    /// where it is afterwards matches what mRoPE says.
    @Test("Position accounting survives an image")
    func positionsAdvanceCorrectly() async throws {
        let root = try fixture.temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let installed = try await fixture.install(at: root, config: config)
        let runner = try makeRunner(at: installed)
        let values = [Float](repeating: 0.01, count: 6 * config.hiddenSize)

        _ = try runner.prefill(elements: [
            .text([1, 2, 3]),
            .image(embeddings: values, frames: 1, height: 2, width: 3),
        ])
        // The cache holds one slot a token: three text and six image.
        #expect(runner.position == 9, "the cache advanced by \(runner.position), expected 9")
    }
}
