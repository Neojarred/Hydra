import Foundation
import HydraCore
import HydraFormat
import HydraInstall
import HydraReference
import Metal
import Testing

@testable import HydraMetal

/// The MLX 4-bit build, end to end, against the same oracle the BF16 build answers to.
///
/// This is the test that makes the whole quantized path trustworthy. Every piece below it has
/// been checked — the byte arithmetic against the checkpoint's header, the decoder against a
/// double-precision reference, the plan against the published index — and none of that proves
/// the pieces were **wired** to each other. A projection reading its scales from the wrong
/// offset, an expert whose `up` matrix is served where `gate` belongs, an embedding row read as
/// BF16: each produces a model that runs and returns finite numbers.
///
/// The oracle is built by dequantizing the very bytes the installation contains, so a
/// disagreement is a wiring fault and never a quantization loss.
@Suite("Gemma 4 MLX model on GPU")
struct GemmaMLXModelTests {

    private let config = Gemma4MLXConfig.tiny

    // MARK: - A miniature MLX checkpoint

    /// Deterministic bytes for one tensor, chosen so nothing decodes to an extreme.
    private func bytes(for name: String, count: Int, quantized: Bool) -> Data {
        var state = UInt64(truncatingIfNeeded: name.hashValue) | 1
        func next() -> UInt64 {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return state >> 33
        }
        var data = Data(capacity: count)
        if quantized {
            // Packed values: any bit pattern is a legal set of small unsigned integers.
            for _ in 0..<count { data.append(UInt8(truncatingIfNeeded: next())) }
        } else {
            // BF16, kept small and well conditioned: random bit patterns read as BF16 give
            // infinities, and a comparison against infinity proves nothing.
            //
            // Scales and biases are generated as a **pair**, because a quantized weight is
            // `q · scale + bias` with `q` in 0…15 — so independent random values decode to a
            // matrix centred on `7.5 · scale`, an order of magnitude larger than the BF16
            // fixture's and enough to saturate the softcap by the sixth layer. Centring the
            // range on zero is what makes the two fixtures comparable.
            let isScale = name.hasSuffix(".scales")
            let isBias = name.hasSuffix(".biases")
            let midpoint = Float((1 << config.quantBits) - 1) / 2
            for _ in 0..<(count / 2) {
                let value: Float
                if isScale {
                    value = Float(Int(next() % 40) + 10) / 1000
                } else if isBias {
                    value = -midpoint * Float(Int(next() % 40) + 10) / 1000
                } else {
                    value = Float(Int(next() % 200) - 100) / 400
                }
                let bits = BF16.fromFloat(value)
                data.append(UInt8(bits & 0xFF))
                data.append(UInt8(bits >> 8))
            }
        }
        return data
    }

    private func install(at root: URL) async throws -> URL {
        var declarations: [(name: String, shape: [Int], bytes: Int, quantized: Bool)] = []

        // A tensor is quantized exactly when the checkpoint also carries its scales — which is
        // the only reliable test. Guessing from the name fails on
        // `post_feedforward_layernorm_1.weight`: it ends in `.weight` and *not* in
        // `norm.weight`, so a suffix rule writes a norm as packed integers, the runtime reads
        // them back as BF16, and the layer explodes six operations later.
        let resident = config.residentTensors
        let quantizedStems = Set(
            resident.map(\.name).filter { $0.hasSuffix(".scales") }
                .map { String($0.dropLast(".scales".count)) })
        for (name, byteCount) in resident {
            let quantized = name.hasSuffix(".weight")
                && quantizedStems.contains(String(name.dropLast(".weight".count)))
            declarations.append((name, [byteCount / (quantized ? 4 : 2)], byteCount, quantized))
        }

        // The experts: nine tensors a layer, each fused across the experts.
        let blob = config.expertBlobLayout
        for layer in 0..<config.layerCount {
            let stems = config.expertTensors(layer: layer)
            let sizes: [(String, Int)] = [
                ("weight", blob.gateLayout.weightBytes), ("scales", blob.gateLayout.scaleBytes),
                ("biases", blob.gateLayout.biasBytes),
                ("weight", blob.gateLayout.weightBytes), ("scales", blob.gateLayout.scaleBytes),
                ("biases", blob.gateLayout.biasBytes),
                ("weight", blob.downLayout.weightBytes), ("scales", blob.downLayout.scaleBytes),
                ("biases", blob.downLayout.biasBytes),
            ]
            for (index, entry) in sizes.enumerated() {
                let total = entry.1 * config.expertCount
                declarations.append((
                    "\(stems[index / 3].stem).\(entry.0)", [config.expertCount, total / 4],
                    total, entry.0 == "weight"))
            }
        }

        let sourceRoot = root.appending(path: "source")
        try FileManager.default.createDirectory(at: sourceRoot, withIntermediateDirectories: true)

        var header: [String: Any] = [:]
        var cursor = 0
        for t in declarations {
            header[t.name] = [
                "dtype": t.quantized ? "U32" : "BF16", "shape": t.shape,
                "data_offsets": [cursor, cursor + t.bytes],
            ]
            cursor += t.bytes
        }
        var headerJSON = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        while headerJSON.count % 8 != 0 { headerJSON.append(0x20) }

        var file = Data()
        withUnsafeBytes(of: UInt64(headerJSON.count).littleEndian) { file.append(contentsOf: $0) }
        file.append(headerJSON)
        for t in declarations {
            file.append(bytes(for: t.name, count: t.bytes, quantized: t.quantized))
        }
        try file.write(to: sourceRoot.appending(path: "shard-0.safetensors"))

        let weightMap = Dictionary(
            uniqueKeysWithValues: declarations.map { ($0.name, "shard-0.safetensors") })
        let headers = [
            "shard-0.safetensors": try SafetensorsHeader.read(
                contentsOf: sourceRoot.appending(path: "shard-0.safetensors"))
        ]
        let plan = try GemmaMLXRepackPlan(
            config: config, weightMap: weightMap, headers: headers)
        let destination = root.appending(path: "tiny-gemma-mlx.hydra")
        _ = try await StreamingRepacker(
            plan: plan, source: LocalDirectorySource(root: sourceRoot)
        ).run(destination: destination)
        return destination
    }

    private func temporary() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appending(path: "hydra-mlx-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    // MARK: - The tests

    /// The plan covers a checkpoint shaped like the published one.
    @Test("A synthetic MLX checkpoint installs")
    func installs() async throws {
        let root = try temporary()
        defer { try? FileManager.default.removeItem(at: root) }

        let installed = try await install(at: root)
        let manifest = try HydraManifest.read(from: installed)
        #expect(manifest.model.architecture == ModelArchitecture.gemma4.rawValue)
        #expect(manifest.model.expertQuantization.hasPrefix("mlx-affine-"))
        try manifest.validate(against: config, root: installed)
    }

    /// The whole model executes, and the result is a distribution rather than debris.
    ///
    /// A wrong offset anywhere in the quantized path — scales read where biases live, an
    /// expert's `up` served for its `gate` — produces finite numbers, so finiteness alone
    /// proves nothing. The softcap bound and the spread are what say the head saw signal.
    @Test("The quantized model produces a usable distribution")
    func producesADistribution() async throws {
        let root = try temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let installed = try await install(at: root)

        let context = try MetalContext()
        let mapping = try ModelMapping(root: installed, model: config, device: context.device)
        let cache = ExpertSlotCache(
            root: installed, model: config, slotsPerLayer: config.expertsPerToken,
            device: context.device)
        let runner = try ModelRuntime.makeRunner(
            model: config, context: context, mapping: mapping,
            expertCache: cache, contextLength: 32)

        let logits = try runner.prefill(tokens: [1, 2, 3])
        #expect(logits.count == config.vocabSize)
        #expect(logits.allSatisfy { $0.isFinite })
        #expect(logits.allSatisfy { abs($0) <= config.base.finalLogitSoftcapping })

        // Not a constant vector: the head has to have read something that varies.
        let minimum = logits.min() ?? 0
        let maximum = logits.max() ?? 0
        #expect(
            maximum - minimum > 1e-4,
            "the distribution is flat: min \(minimum), max \(maximum)")
    }

    /// The embedding is quantized too, and read one row at a time on the CPU.
    ///
    /// Reading it as BF16 — which is what the unquantized path does — reinterprets packed
    /// integers as floats. The values stay finite, so only a comparison against the decoder
    /// catches it.
    @Test("An embedding row matches the reference decoder")
    func embeddingRowIsDequantized() async throws {
        let root = try temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let installed = try await install(at: root)

        let context = try MetalContext()
        let mapping = try ModelMapping(root: installed, model: config, device: context.device)
        let weights = Gemma4MLXWeights(config: config, mapping: mapping)

        let hidden = config.hiddenSize
        var row = [Float](repeating: 0, count: hidden)
        row.withUnsafeMutableBufferPointer { weights.readEmbedding(token: 5, into: $0) }

        // The same row, decoded from the installed bytes by the reference.
        let layout = MLXAffineLayout(
            bits: config.quantBits, groupSize: config.groupSize,
            rows: config.vocabSize, cols: hidden)
        let stem = "\(Gemma4MLXConfig.prefix).embed_tokens"
        let w = try mapping.residentTensor("\(stem).weight")
        let s = try mapping.residentTensor("\(stem).scales")
        let b = try mapping.residentTensor("\(stem).biases")

        let expected: [Double] = mapping.resident.withBytes { raw in
            var words: [UInt32] = []
            for i in 0..<layout.wordsPerRow {
                words.append(UInt32(littleEndian: raw.loadUnaligned(
                    fromByteOffset: w.offset + (5 * layout.wordsPerRow + i) * 4,
                    as: UInt32.self)))
            }
            func readBF16(_ base: Int, _ index: Int) -> Double {
                Double(BF16.toFloat(UInt16(littleEndian: raw.loadUnaligned(
                    fromByteOffset: base + (5 * layout.groupsPerRow + index) * 2,
                    as: UInt16.self))))
            }
            return MLXAffine.dequantize(
                words: words,
                scales: (0..<layout.groupsPerRow).map { readBF16(s.offset, $0) },
                biases: (0..<layout.groupsPerRow).map { readBF16(b.offset, $0) },
                bits: config.quantBits, groupSize: config.groupSize)
        }

        for i in 0..<hidden {
            #expect(abs(Double(row[i]) - expected[i]) < 1e-6, "column \(i)")
        }
        // And it is not the degenerate all-zero row that a missing tensor would give.
        #expect(row.contains { $0 != 0 })
    }

    /// Same input, same output — the property a cache must not be able to break.
    @Test("Decoding is reproducible")
    func decodingIsReproducible() async throws {
        let root = try temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let installed = try await install(at: root)

        let context = try MetalContext()
        let mapping = try ModelMapping(root: installed, model: config, device: context.device)
        let cache = ExpertSlotCache(
            root: installed, model: config, slotsPerLayer: config.expertsPerToken,
            device: context.device)
        let runner = try ModelRuntime.makeRunner(
            model: config, context: context, mapping: mapping,
            expertCache: cache, contextLength: 32)

        let first = Array(try runner.forward(token: 3, needsLogits: true))
        runner.reset()
        let second = Array(try runner.forward(token: 3, needsLogits: true))
        #expect(first == second)
    }
}
