import Foundation
import HydraCore
import Testing

@testable import HydraReference

/// The Swift oracle measured against an independent Python transcription of
/// `modeling_gemma4.py`.
///
/// This is the same dispositif as `ReferenceOpsTests`: two implementations written from the
/// same source but not from each other, compared on frozen vectors. It is what turns D-022's
/// ten traps from a document into failing tests — a wrong RMSNorm or an attention scale of
/// `1/sqrt(headDim)` produces plausible text, so nothing but a comparison catches it.
@Suite("Gemma 4 reference operators")
struct Gemma4ReferenceTests {

    // MARK: - Fixtures

    /// Loaded per instance rather than into a static: `[String: Any]` is not `Sendable`, and
    /// the file is 42 KB — cheaper to re-read than to make thread-safe.
    private let fixtures: [String: Any] = {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appending(path: "Fixtures/gemma4_operators.json")
        guard let data = try? Data(contentsOf: url),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return [:] }
        return object
    }()

    private func fixture(_ name: String) -> [String: Any] {
        fixtures[name] as? [String: Any] ?? [:]
    }
    private func doubles(_ box: [String: Any], _ key: String) -> [Double] {
        (box[key] as? [Any])?.compactMap { ($0 as? NSNumber)?.doubleValue } ?? []
    }
    private func matrix(_ box: [String: Any], _ key: String) -> [[Double]] {
        (box[key] as? [Any])?.compactMap { row in
            (row as? [Any])?.compactMap { ($0 as? NSNumber)?.doubleValue }
        } ?? []
    }
    private func number(_ box: [String: Any], _ key: String) -> Double {
        (box[key] as? NSNumber)?.doubleValue ?? .nan
    }

    private func worst(_ a: [Double], _ b: [Double]) -> Double {
        guard a.count == b.count, !a.isEmpty else { return .infinity }
        var scale = 0.0
        for value in b { scale = max(scale, abs(value)) }
        var worst = 0.0
        for (x, y) in zip(a, b) { worst = max(worst, abs(x - y) / max(scale, 1e-12)) }
        return worst
    }

    @Test("The fixtures are present")
    func fixturesExist() {
        #expect(!fixtures.isEmpty,
            "run: python3 tools/gen_gemma_fixtures.py Tests/HydraReferenceTests/Fixtures")
        #expect(fixtures.count == 11)
    }

    // MARK: - Normalization

    /// The trap that would cost the most: Gemma 3 scaled by `1 + w` and Gemma 4 scales by `w`.
    /// Both produce output; only one is the model.
    @Test("RMSNorm scales by w, not 1 + w")
    func rmsNormMatches() {
        let f = fixture("rms_norm")
        let got = Gemma4ReferenceOps.rmsNorm(
            doubles(f, "input"), weight: doubles(f, "weight"), eps: number(f, "eps"))
        #expect(worst(got, doubles(f, "expected")) < 1e-12)

        // And the 1 + w reading must be visibly different, or the test proves nothing.
        let wrong = Gemma4ReferenceOps.rmsNorm(
            doubles(f, "input"), weight: doubles(f, "weight").map { 1 + $0 },
            eps: number(f, "eps"))
        #expect(worst(wrong, doubles(f, "expected")) > 0.1)
    }

    /// `v_norm` and the router's norm are built `with_scale: false` and have no tensor. An
    /// implementation that looked for a weight would fail to load; one that skipped the
    /// operation entirely would load and be wrong.
    @Test("The weightless normalization matches")
    func unscaledRmsNormMatches() {
        let f = fixture("rms_norm_unscaled")
        let got = Gemma4ReferenceOps.rmsNorm(
            doubles(f, "input"), weight: nil, eps: number(f, "eps"))
        #expect(worst(got, doubles(f, "expected")) < 1e-12)
    }

    // MARK: - Activation

    @Test("gelu_pytorch_tanh matches the reference")
    func geluMatches() {
        let f = fixture("gelu_tanh")
        let input = doubles(f, "input")
        let got = input.map(Gemma4ReferenceOps.gelu)
        #expect(worst(got, doubles(f, "expected")) < 1e-12)
        #expect(Gemma4ReferenceOps.gelu(0) == 0)
    }

    /// What separates the two models' feed-forward paths is **not** the activation curve.
    ///
    /// `x · sigmoid(1.702x)` sits within 0.2 % of `gelu_pytorch_tanh` — that closeness is the
    /// reason 1.702 was chosen, and a test asserting they differ would be asserting something
    /// false. The real divergence is structural: GPT-OSS clamps its gate from above, clamps
    /// the linear branch on both sides, and adds **one** to it (D-014). Gemma does none of
    /// that, and applying GPT-OSS's kernel here would be wrong for those reasons rather than
    /// for the curve.
    @Test("The activations are close; the structure is what differs")
    func activationClosenessIsNotTheDifference() {
        let input = doubles(fixture("gelu_tanh"), "input")
        let gemma = input.map(Gemma4ReferenceOps.gelu)
        let quick = input.map { $0 / (1 + Foundation.exp(-1.702 * $0)) }
        #expect(worst(quick, gemma) < 0.01, "these curves are near-identical by design")

        // The structural difference, on a value the clamp actually bites.
        let gate = 20.0, up = -20.0
        let gemmaOut = Gemma4ReferenceOps.gelu(gate) * up
        let gptOss = min(gate, 7.0) / (1 + Foundation.exp(-1.702 * min(gate, 7.0)))
            * (max(min(up, 7.0), -7.0) + 1)
        #expect(abs(gemmaOut - gptOss) > 1.0, "clamping and the +1 are the real divergence")
    }

    // MARK: - Rotary

    @Test("RoPE matches on a sliding layer")
    func ropeSlidingMatches() {
        let f = fixture("rope_sliding")
        let frequencies = Gemma4ReferenceOps.inverseFrequencies(
            headDim: Int(number(f, "headDim")), theta: number(f, "theta"),
            rotatingPairs: Int(number(f, "rotatingPairs")))
        #expect(worst(frequencies, doubles(f, "inverseFrequencies")) < 1e-12)

        let got = Gemma4ReferenceOps.applyRoPE(
            doubles(f, "input"), position: Int(number(f, "position")), frequencies: frequencies)
        #expect(worst(got, doubles(f, "expected")) < 1e-12)
    }

    /// The full layers' partial rotation, and the property that makes it free: the unrotated
    /// tail carries a **zero** inverse frequency, which is the identity.
    @Test("Partial rotation is zero frequencies, and leaves its tail untouched")
    func ropePartialMatches() {
        let f = fixture("rope_full_partial")
        let headDim = Int(number(f, "headDim"))
        let rotating = Int(number(f, "rotatingPairs"))
        let frequencies = Gemma4ReferenceOps.inverseFrequencies(
            headDim: headDim, theta: number(f, "theta"), rotatingPairs: rotating)

        #expect(worst(frequencies, doubles(f, "inverseFrequencies")) < 1e-12)
        #expect(frequencies.count { $0 == 0 } == headDim / 2 - rotating)

        let input = doubles(f, "input")
        let got = Gemma4ReferenceOps.applyRoPE(
            input, position: Int(number(f, "position")), frequencies: frequencies)
        #expect(worst(got, doubles(f, "expected")) < 1e-12)

        // The unrotated pairs must come through exactly, not merely close.
        let half = headDim / 2
        for i in rotating..<half {
            #expect(got[i] == input[i])
            #expect(got[i + half] == input[i + half])
        }
    }

    // MARK: - Attention

    /// `scaling = 1.0`. A `1/sqrt(headDim)` reflex changes every score and still produces text.
    @Test("Attention uses a scale of 1.0")
    func attentionMatches() {
        let f = fixture("attention_full")
        let query = doubles(f, "query")
        let keys = matrix(f, "keys")
        let values = matrix(f, "values")
        let got = Gemma4ReferenceOps.attention(query: query, keys: keys, values: values)
        #expect(worst(got, doubles(f, "expected")) < 1e-12)

        let scaled = Gemma4ReferenceOps.attention(
            query: query.map { $0 / Double(query.count).squareRoot() },
            keys: keys, values: values)
        #expect(worst(scaled, doubles(f, "expected")) > 1e-4,
            "a 1/sqrt(headDim) scale must be visibly different, or this test proves nothing")
    }

    @Test("The sliding window bounds what attention reaches")
    func windowedAttentionMatches() {
        let f = fixture("attention_windowed")
        let got = Gemma4ReferenceOps.attention(
            query: doubles(f, "query"), keys: matrix(f, "keys"), values: matrix(f, "values"),
            slidingWindow: Int(number(f, "window")))
        #expect(worst(got, doubles(f, "expected")) < 1e-12)
    }

    // MARK: - Router

    /// Softmax over all experts, then top-k, then renormalize, then the per-expert scale.
    /// GPT-OSS softmaxes over the top-k only — same logits, different weights.
    @Test("The router matches, and differs from GPT-OSS's")
    func routerMatches() {
        let f = fixture("router")
        let (indices, weights) = Gemma4ReferenceOps.router(
            hidden: doubles(f, "hidden"), projection: matrix(f, "projection"),
            scale: doubles(f, "scale"), perExpertScale: doubles(f, "perExpertScale"),
            topK: Int(number(f, "topK")), eps: 1e-6)

        let expectedIndices = (f["expectedIndices"] as? [Any])?
            .compactMap { ($0 as? NSNumber)?.intValue } ?? []
        #expect(indices == expectedIndices)
        #expect(worst(weights, doubles(f, "expectedWeights")) < 1e-12)

        // Without the per-expert scale the weights would sum to one. They do not.
        #expect(abs(weights.reduce(0, +) - 1.0) > 1e-6)
    }

    // MARK: - Output

    @Test("Logit softcapping matches and is bounded by the cap")
    func softcapMatches() {
        let f = fixture("softcap")
        let cap = number(f, "cap")
        let got = Gemma4ReferenceOps.softcap(doubles(f, "input"), cap: cap)
        #expect(worst(got, doubles(f, "expected")) < 1e-12)
        #expect(got.allSatisfy { abs($0) < cap })
    }

    @Test("The embedding scale is sqrt(hiddenSize)")
    func embeddingScaleMatches() {
        let f = fixture("embedding_scale")
        let got = Gemma4ReferenceOps.embeddingScale(hiddenSize: Int(number(f, "hiddenSize")))
        #expect(abs(got - number(f, "expected")) < 1e-12)
        #expect(abs(Double(Gemma4Config.a4b.embeddingScale) - got) < 1e-3)
    }
}

/// The whole decoder layer, where the topology matters more than any single operator.
///
/// Each operator above is already checked. What this suite exists for is the two structural
/// findings of D-022 that no amount of care with operators would catch: the expert branch reads
/// the **residual** rather than the dense MLP's output, and `post_attention_layernorm` is
/// applied **before** the residual add. Both produce a model that still speaks.
@Suite("Gemma 4 reference layer")
struct Gemma4ReferenceLayerTests {

    private let fixtures: [String: Any] = {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appending(path: "Fixtures/gemma4_operators.json")
        guard let data = try? Data(contentsOf: url),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return [:] }
        return object
    }()

    private func vector(_ box: [String: Any], _ key: String) -> [Double] {
        (box[key] as? [Any])?.compactMap { ($0 as? NSNumber)?.doubleValue } ?? []
    }
    private func matrix(_ box: [String: Any], _ key: String) -> [[Double]] {
        (box[key] as? [Any])?.compactMap { row in
            (row as? [Any])?.compactMap { ($0 as? NSNumber)?.doubleValue }
        } ?? []
    }

    private func makeLayer() -> (Gemma4ReferenceLayer, [String: Any])? {
        guard let f = fixtures["layer"] as? [String: Any],
            let w = f["weights"] as? [String: Any],
            let rawExperts = f["experts"] as? [[String: Any]]
        else { return nil }

        let weights = Gemma4ReferenceLayer.Weights(
            inputLayerNorm: vector(w, "input_layernorm"),
            queryProjection: matrix(w, "q_proj"), keyProjection: matrix(w, "k_proj"),
            valueProjection: matrix(w, "v_proj"), outputProjection: matrix(w, "o_proj"),
            queryNorm: vector(w, "q_norm"), keyNorm: vector(w, "k_norm"),
            postAttentionLayerNorm: vector(w, "post_attention_layernorm"),
            preFeedForwardLayerNorm: vector(w, "pre_feedforward_layernorm"),
            gateProjection: matrix(w, "gate_proj"), upProjection: matrix(w, "up_proj"),
            downProjection: matrix(w, "down_proj"),
            postFeedForwardLayerNorm1: vector(w, "post_feedforward_layernorm_1"),
            preFeedForwardLayerNorm2: vector(w, "pre_feedforward_layernorm_2"),
            routerProjection: matrix(w, "router_proj"), routerScale: vector(w, "router_scale"),
            routerPerExpertScale: vector(w, "router_per_expert_scale"),
            postFeedForwardLayerNorm2: vector(w, "post_feedforward_layernorm_2"),
            postFeedForwardLayerNorm: vector(w, "post_feedforward_layernorm"),
            layerScalar: (w["layer_scalar"] as? NSNumber)?.doubleValue ?? 1)

        let experts = rawExperts.map {
            Gemma4ReferenceLayer.Expert(
                gate: matrix($0, "gate"), up: matrix($0, "up"), down: matrix($0, "down"))
        }
        let layer = Gemma4ReferenceLayer(
            weights: weights, experts: experts,
            heads: (f["heads"] as? NSNumber)?.intValue ?? 0,
            keyValueHeads: (f["keyValueHeads"] as? NSNumber)?.intValue ?? 1,
            headDim: (f["headDim"] as? NSNumber)?.intValue ?? 0,
            topK: (f["topK"] as? NSNumber)?.intValue ?? 0,
            eps: (f["eps"] as? NSNumber)?.doubleValue ?? 1e-6)
        return (layer, f)
    }

    private func worst(_ a: [Double], _ b: [Double]) -> Double {
        guard a.count == b.count, !a.isEmpty else { return .infinity }
        var scale = 0.0
        for value in b { scale = max(scale, abs(value)) }
        var worst = 0.0
        for (x, y) in zip(a, b) { worst = max(worst, abs(x - y) / max(scale, 1e-12)) }
        return worst
    }

    @Test("A whole layer matches the reference")
    func layerMatches() throws {
        guard let (layer, f) = makeLayer() else {
            Issue.record("fixture missing — run tools/gen_gemma_fixtures.py")
            return
        }
        let position = (f["position"] as? NSNumber)?.intValue ?? 0
        let frequencies = vector(f, "inverseFrequencies")
        let input = vector(f, "input")

        let kv = layer.keyValue(hidden: input, position: position, frequencies: frequencies)
        let got = layer.forward(
            hidden: input, position: position, keys: [kv.key], values: [kv.value],
            frequencies: frequencies)

        #expect(worst(got, vector(f, "expected")) < 1e-12)
    }

    /// If the expert branch chained off the dense MLP instead of reading the residual, the
    /// layer would still produce a vector. This proves the two are parallel.
    @Test("The expert branch reads the residual, not the MLP's output")
    func branchesAreParallel() throws {
        guard let (layer, f) = makeLayer() else { return }
        let position = (f["position"] as? NSNumber)?.intValue ?? 0
        let frequencies = vector(f, "inverseFrequencies")
        let input = vector(f, "input")
        let kv = layer.keyValue(hidden: input, position: position, frequencies: frequencies)
        let correct = layer.forward(
            hidden: input, position: position, keys: [kv.key], values: [kv.value],
            frequencies: frequencies)

        // Zeroing the dense branch's output norm must leave the expert branch intact. If the
        // two were chained, the expert path would collapse with it.
        var chained = layer.weights
        chained.postFeedForwardLayerNorm1 = [Double](
            repeating: 0, count: layer.weights.postFeedForwardLayerNorm1.count)
        let withoutDense = Gemma4ReferenceLayer(
            weights: chained, experts: layer.experts, heads: layer.heads,
            keyValueHeads: layer.keyValueHeads, headDim: layer.headDim,
            topK: layer.topK, eps: layer.eps
        ).forward(
            hidden: input, position: position, keys: [kv.key], values: [kv.value],
            frequencies: frequencies)

        #expect(worst(withoutDense, correct) > 1e-6, "the branches must be separable")
        #expect(withoutDense.allSatisfy { $0.isFinite && $0 != 0 },
            "the expert branch must survive the dense branch being silenced")
    }

    /// `attention_k_eq_v` on full layers: V reuses the key projection's output, before k_norm
    /// and before RoPE, then takes its own weightless normalization.
    @Test("Without a value projection, V follows the key projection")
    func keyEqualsValue() throws {
        guard let (layer, f) = makeLayer() else { return }
        let position = (f["position"] as? NSNumber)?.intValue ?? 0
        let frequencies = vector(f, "inverseFrequencies")
        let input = vector(f, "input")

        var shared = layer.weights
        shared.valueProjection = nil
        let sharedLayer = Gemma4ReferenceLayer(
            weights: shared, experts: layer.experts, heads: layer.heads,
            keyValueHeads: layer.keyValueHeads, headDim: layer.headDim,
            topK: layer.topK, eps: layer.eps)

        let kv = sharedLayer.keyValue(
            hidden: input, position: position, frequencies: frequencies)
        // V is the un-RoPE'd, weightlessly normalized key projection — never equal to K.
        #expect(worst(kv.value, kv.key) > 1e-6)
        #expect(kv.value.count == kv.key.count)
    }
}
