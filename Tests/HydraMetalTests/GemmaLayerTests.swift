import Foundation
import HydraCore
import HydraFormat
import HydraInstall
import HydraReference
import Metal
import Testing

@testable import HydraMetal

/// A whole Gemma 4 layer on the GPU, against the CPU oracle.
///
/// This is the test the rest of the Gemma work exists to make possible. Every operator and
/// every kernel has been checked in isolation; what none of them can catch is the **wiring** —
/// a norm applied on the wrong side of a residual, the expert branch fed the MLP's output
/// instead of the residual, V taken after `k_norm` instead of before. All of those produce a
/// layer that runs, returns finite numbers, and is wrong.
///
/// The weights come from a real installation produced by the real repacker, not from buffers
/// filled in place, so a layout error is in scope too.
@Suite("Gemma 4 layer on GPU")
struct GemmaLayerTests {

    private let config = Gemma4Config.tiny

    // MARK: - A miniature installation

    /// Writes a synthetic Gemma checkpoint and repacks it, the way `installTinyModel` does for
    /// GPT-OSS. Values are small and well conditioned: random bytes read as BF16 give
    /// infinities, and a comparison against infinity proves nothing.
    private func install(at root: URL) async throws -> URL {
        var declarations: [(name: String, shape: [Int], bytes: Int)] = []
        for tensor in config.residentTensors {
            declarations.append((tensor.name, [tensor.byteCount / 2], tensor.byteCount))
        }
        let blob = config.expertBlobLayout
        for layer in 0..<config.layerCount {
            let base = "model.language_model.layers.\(layer).experts"
            declarations.append((
                "\(base).gate_up_proj",
                [config.expertCount, 2 * config.moeIntermediateSize, config.hiddenSize],
                blob.gateUp.byteCount * config.expertCount))
            declarations.append((
                "\(base).down_proj",
                [config.expertCount, config.hiddenSize, config.moeIntermediateSize],
                blob.down.byteCount * config.expertCount))
        }

        let sourceRoot = root.appending(path: "source")
        try FileManager.default.createDirectory(at: sourceRoot, withIntermediateDirectories: true)

        var header: [String: Any] = [:]
        var cursor = 0
        for t in declarations {
            header[t.name] = [
                "dtype": "BF16", "shape": t.shape,
                "data_offsets": [cursor, cursor + t.bytes],
            ]
            cursor += t.bytes
        }
        var headerJSON = try JSONSerialization.data(
            withJSONObject: header, options: [.sortedKeys])
        while headerJSON.count % 8 != 0 { headerJSON.append(0x20) }

        var file = Data()
        withUnsafeBytes(of: UInt64(headerJSON.count).littleEndian) { file.append(contentsOf: $0) }
        file.append(headerJSON)
        for t in declarations {
            file.append(
                LayerRunnerTests.syntheticBytes(
                    tensor: t.name, byteCount: t.bytes, kind: .bf16))
        }
        try file.write(to: sourceRoot.appending(path: "shard-0.safetensors"))

        let weightMap = Dictionary(
            uniqueKeysWithValues: declarations.map { ($0.name, "shard-0.safetensors") })
        let headers = [
            "shard-0.safetensors": try SafetensorsHeader.read(
                contentsOf: sourceRoot.appending(path: "shard-0.safetensors"))
        ]
        let plan = try GemmaRepackPlan(
            config: config, weightMap: weightMap, headers: headers)
        let destination = root.appending(path: "tiny-gemma.hydra")
        _ = try await StreamingRepacker(
            plan: plan, source: LocalDirectorySource(root: sourceRoot)
        ).run(destination: destination)
        return destination
    }

    // MARK: - Reading the installed weights back

    private func vector(_ mapping: ModelMapping, _ name: String) throws -> [Double] {
        let placement = try mapping.residentTensor(name)
        return mapping.resident.withBytes { raw in
            (0..<(placement.byteCount / 2)).map { i in
                let o = placement.offset + i * 2
                return Double(BF16.toFloat(UInt16(raw[o]) | (UInt16(raw[o + 1]) << 8)))
            }
        }
    }

    private func matrix(
        _ mapping: ModelMapping, _ name: String, rows: Int, cols: Int
    ) throws -> [[Double]] {
        let flat = try vector(mapping, name)
        return (0..<rows).map { r in Array(flat[(r * cols)..<((r + 1) * cols)]) }
    }

    /// One expert's three matrices, read from its slot in the layer file.
    private func expert(
        _ cache: ExpertSlotCache, layer: Int, index: Int
    ) throws -> Gemma4ReferenceLayer.Expert {
        let blob = config.expertBlobLayout
        let (buffer, _) = try cache.expert(layer: layer, expert: index)
        let raw = UnsafeRawBufferPointer(start: buffer.contents(), count: buffer.length)
        func read(_ offset: Int, rows: Int, cols: Int) -> [[Double]] {
            (0..<rows).map { r in
                (0..<cols).map { c in
                    let o = offset + (r * cols + c) * 2
                    return Double(BF16.toFloat(UInt16(raw[o]) | (UInt16(raw[o + 1]) << 8)))
                }
            }
        }
        let inner = config.moeIntermediateSize
        let h = config.hiddenSize
        return Gemma4ReferenceLayer.Expert(
            gate: read(blob.gateUp.offset, rows: inner, cols: h),
            up: read(blob.gateUp.offset + inner * h * 2, rows: inner, cols: h),
            down: read(blob.down.offset, rows: h, cols: inner))
    }

    private func weights(
        _ mapping: ModelMapping, layer: Int
    ) throws -> Gemma4ReferenceLayer.Weights {
        let l = "model.language_model.layers.\(layer)"
        let g = config.attentionGeometry(atLayer: layer)
        let h = config.hiddenSize
        return Gemma4ReferenceLayer.Weights(
            inputLayerNorm: try vector(mapping, "\(l).input_layernorm.weight"),
            queryProjection: try matrix(
                mapping, "\(l).self_attn.q_proj.weight", rows: g.queryDim, cols: h),
            keyProjection: try matrix(
                mapping, "\(l).self_attn.k_proj.weight", rows: g.keyValueDim, cols: h),
            valueProjection: config.hasValueProjection(atLayer: layer)
                ? try matrix(mapping, "\(l).self_attn.v_proj.weight",
                             rows: g.keyValueDim, cols: h)
                : nil,
            outputProjection: try matrix(
                mapping, "\(l).self_attn.o_proj.weight", rows: h, cols: g.queryDim),
            queryNorm: try vector(mapping, "\(l).self_attn.q_norm.weight"),
            keyNorm: try vector(mapping, "\(l).self_attn.k_norm.weight"),
            postAttentionLayerNorm: try vector(mapping, "\(l).post_attention_layernorm.weight"),
            preFeedForwardLayerNorm: try vector(mapping, "\(l).pre_feedforward_layernorm.weight"),
            gateProjection: try matrix(
                mapping, "\(l).mlp.gate_proj.weight", rows: config.intermediateSize, cols: h),
            upProjection: try matrix(
                mapping, "\(l).mlp.up_proj.weight", rows: config.intermediateSize, cols: h),
            downProjection: try matrix(
                mapping, "\(l).mlp.down_proj.weight", rows: h, cols: config.intermediateSize),
            postFeedForwardLayerNorm1: try vector(
                mapping, "\(l).post_feedforward_layernorm_1.weight"),
            preFeedForwardLayerNorm2: try vector(
                mapping, "\(l).pre_feedforward_layernorm_2.weight"),
            routerProjection: try matrix(
                mapping, "\(l).router.proj.weight", rows: config.expertCount, cols: h),
            routerScale: try vector(mapping, "\(l).router.scale"),
            routerPerExpertScale: try vector(mapping, "\(l).router.per_expert_scale"),
            postFeedForwardLayerNorm2: try vector(
                mapping, "\(l).post_feedforward_layernorm_2.weight"),
            postFeedForwardLayerNorm: try vector(
                mapping, "\(l).post_feedforward_layernorm.weight"),
            layerScalar: try vector(mapping, "\(l).layer_scalar")[0])
    }

    // MARK: - The comparison

    private func divergence(_ got: [Float], _ expected: [Double]) -> Double {
        var scale = 0.0
        for value in expected { scale = max(scale, abs(value)) }
        var worst = 0.0
        for (a, b) in zip(got, expected) {
            worst = max(worst, abs(Double(a) - b) / max(scale, 1e-9))
        }
        return worst
    }

    /// Runs layer `layer` on the GPU and against the oracle, from the same installed weights.
    private func compareLayer(_ layer: Int) async throws {
        let temporary = FileManager.default.temporaryDirectory
            .appending(path: "hydra-gemma-layer-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temporary) }

        let root = try await install(at: temporary)
        let context = try MetalContext()
        let mapping = try ModelMapping(root: root, model: config, device: context.device)
        let cache = ExpertSlotCache(
            root: root, model: config, slotsPerLayer: config.expertCount,
            device: context.device)
        let encoder = ForwardEncoder(context: context)
        let scratch = try Gemma4DecodeScratch(config: config, device: context.device)
        let kvCache = try KVCache(model: config, contextLength: 32, device: context.device)
        let runner = Gemma4LayerRunner(config: config, encoder: encoder, mapping: mapping)

        // A deterministic hidden state, and the tables for this layer's pattern.
        let hidden = (0..<config.hiddenSize).map {
            Float(sin(Double($0) * 0.37) * 0.4)
        }
        let hiddenPointer = scratch.hidden.contents().bindMemory(
            to: Float.self, capacity: config.hiddenSize)
        for i in 0..<config.hiddenSize { hiddenPointer[i] = hidden[i] }

        let position = 0
        let tables = Gemma4RoPETables(config: config, layer: layer).tables(at: position)
        let cosPointer = scratch.cosTable.contents().bindMemory(
            to: Float.self, capacity: tables.cos.count)
        let sinPointer = scratch.sinTable.contents().bindMemory(
            to: Float.self, capacity: tables.sin.count)
        for i in 0..<tables.cos.count {
            cosPointer[i] = Float(tables.cos[i])
            sinPointer[i] = Float(tables.sin[i])
        }

        // --- GPU ---
        guard let first = context.commandQueue.makeCommandBuffer() else { return }
        try runner.encodeAttentionAndRouter(
            layer: layer, position: position, scratch: scratch, kvCache: kvCache, in: first)
        let perExpert = try mapping.residentTensor(
            "model.language_model.layers.\(layer).router.per_expert_scale")
        try encoder.gemmaRouterTopK(
            logits: scratch.routerLogits,
            perExpertScale: perExpert.buffer, perExpertScaleOffset: perExpert.offset,
            indices: scratch.routerIndices, weights: scratch.routerWeights,
            expertCount: config.expertCount, topK: config.expertsPerToken, in: first)
        first.commit()
        await first.completed()

        let selectedPointer = scratch.routerIndices.contents().bindMemory(
            to: UInt32.self, capacity: config.expertsPerToken)
        let selected = (0..<config.expertsPerToken).map { Int(selectedPointer[$0]) }
        try cache.load(layer: layer, experts: selected)

        guard let second = context.commandQueue.makeCommandBuffer() else { return }
        try runner.encodeMixtureStart(scratch: scratch, in: second)
        for (slot, index) in selected.enumerated() {
            let (buffer, _) = try cache.expert(layer: layer, expert: index, pin: true)
            try runner.encodeSingleExpert(
                buffer: buffer, weightIndex: slot, scratch: scratch, in: second)
        }
        try runner.encodeCombineBranches(
            layer: layer, count: selected.count, scratch: scratch, in: second)
        second.commit()
        await second.completed()
        cache.release(layer: layer)

        let got = Array(UnsafeBufferPointer(
            start: scratch.hidden.contents().bindMemory(
                to: Float.self, capacity: config.hiddenSize),
            count: config.hiddenSize))

        // --- CPU oracle, from the same installed bytes ---
        let reference = Gemma4ReferenceLayer(
            weights: try weights(mapping, layer: layer),
            experts: try (0..<config.expertCount).map {
                try expert(cache, layer: layer, index: $0)
            },
            heads: config.attentionHeadCount,
            keyValueHeads: config.attentionGeometry(atLayer: layer).keyValueHeadCount,
            headDim: config.attentionGeometry(atLayer: layer).headDim,
            topK: config.expertsPerToken, eps: Double(config.rmsNormEps))

        let frequencies = Gemma4RoPETables(config: config, layer: layer).inverseFrequencies
        let kv = reference.keyValue(
            hidden: hidden.map(Double.init), position: position, frequencies: frequencies)
        let expected = reference.forward(
            hidden: hidden.map(Double.init), position: position,
            keys: [kv.key], values: [kv.value], frequencies: frequencies)

        #expect(got.allSatisfy { $0.isFinite }, "layer \(layer) produced non-finite values")
        let worst = divergence(got, expected)
        #expect(worst < 5e-2, "layer \(layer): relative deviation \(worst) against the oracle")
    }

    /// A sliding layer: 4 heads of 16, its own value projection, full rotation.
    @Test("A sliding layer matches the oracle")
    func slidingLayerMatches() async throws {
        try await compareLayer(0)
    }

    /// A full-attention layer, which is a different geometry, a different theta, partial
    /// rotation, and no value projection of its own.
    @Test("A full-attention layer matches the oracle")
    func fullLayerMatches() async throws {
        try await compareLayer(5)
    }
}
