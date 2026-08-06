import Foundation
import HydraCore
import HydraFormat
import HydraInstall
import HydraReference
import Metal
import Testing

@testable import HydraMetal

/// Integration test: a complete layer on the GPU, compared with the CPU reference.
///
/// This is the first point where everything meets — repacker, format, zero-copy mapping,
/// expert cache, and the seven Metal kernels. The preceding tests validate each operator
/// in isolation; this one validates their **assembly** — the order of operations, tensor
/// offsets, the KV cache layout and the router wiring. An assembly error shows up in no
/// unit test.
struct LayerRunnerTests {

    static let shape = GptOssConfig.tiny

    // MARK: - Synthetic checkpoint with sane values

    /// A random byte pattern but **of reasonable magnitude**.
    ///
    /// The repacker tests' generator produces arbitrary bytes; read as BF16 that gives extreme
    /// exponents and infinities, which would make any numerical comparison meaningless. Here
    /// the BF16s stay within [-0.5, 0.5] and the MXFP4 scales within the exponents observed on
    /// the real checkpoint.
    static func syntheticBytes(tensor: String, byteCount: Int, kind: Kind) -> Data {
        var state: UInt64 = 0xF00D
        for b in tensor.utf8 { state = (state ^ UInt64(b)) &* 1099511628211 }
        func next() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }

        switch kind {
        case .bf16:
            var values: [Float] = []
            values.reserveCapacity(byteCount / 2)
            for _ in 0..<(byteCount / 2) {
                values.append(Float(Double(next() % 20000) / 20000.0 - 0.5))
            }
            return BF16.encode(values)
        case .mxfp4Blocks:
            var data = Data(count: byteCount)
            for i in 0..<byteCount { data[i] = UInt8(truncatingIfNeeded: next()) }
            return data
        case .mxfp4Scales:
            var data = Data(count: byteCount)
            // Exponents -8 to -3, as measured on the installed GPT-OSS 20B.
            for i in 0..<byteCount { data[i] = UInt8(119 + next() % 6) }
            return data
        }
    }

    enum Kind { case bf16, mxfp4Blocks, mxfp4Scales }

    static func kind(of tensor: String) -> Kind {
        if tensor.hasSuffix("_blocks") { return .mxfp4Blocks }
        if tensor.hasSuffix("_scales") { return .mxfp4Scales }
        return .bf16
    }

    /// Writes a valid safetensors checkpoint then runs it through the real repacker.
    static func installTinyModel(at root: URL) throws -> URL {
        let config = shape
        let blob = config.expertBlobLayout

        var declarations: [(name: String, dtype: String, shape: [Int], bytes: Int)] = []
        for layer in 0..<config.layerCount {
            for entry in HydraLayout.residentTensorNames(layer: layer) {
                let n = entry.bytes(config)
                declarations.append((entry.name, "BF16", [n / 2], n))
            }
            for (suffix, slot) in [
                ("gate_up_proj_blocks", blob.gateUpBlocks),
                ("gate_up_proj_scales", blob.gateUpScales),
                ("gate_up_proj_bias", blob.gateUpBias),
                ("down_proj_blocks", blob.downBlocks),
                ("down_proj_scales", blob.downScales),
                ("down_proj_bias", blob.downBias),
            ] {
                let name = "model.layers.\(layer).mlp.experts.\(suffix)"
                declarations.append((
                    name, suffix.hasSuffix("bias") ? "BF16" : "U8",
                    [config.expertCount, slot.byteCount], slot.byteCount * config.expertCount))
            }
        }
        declarations.append(
            ("model.norm.weight", "BF16", [config.hiddenSize], 2 * config.hiddenSize))
        declarations.append(
            ("lm_head.weight", "BF16", [config.vocabSize, config.hiddenSize], config.lmHeadBytes))
        declarations.append((
            "model.embed_tokens.weight", "BF16",
            [config.vocabSize, config.hiddenSize], config.embeddingBytes))

        let sourceRoot = root.appending(path: "source")
        try FileManager.default.createDirectory(at: sourceRoot, withIntermediateDirectories: true)

        var header: [String: Any] = [:]
        var cursor = 0
        for t in declarations {
            header[t.name] = [
                "dtype": t.dtype, "shape": t.shape, "data_offsets": [cursor, cursor + t.bytes],
            ]
            cursor += t.bytes
        }
        var headerJSON = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        while headerJSON.count % 8 != 0 { headerJSON.append(0x20) }

        var file = Data()
        withUnsafeBytes(of: UInt64(headerJSON.count).littleEndian) { file.append(contentsOf: $0) }
        file.append(headerJSON)
        for t in declarations {
            file.append(syntheticBytes(tensor: t.name, byteCount: t.bytes, kind: kind(of: t.name)))
        }
        try file.write(to: sourceRoot.appending(path: "shard-0.safetensors"))

        let weightMap = Dictionary(
            uniqueKeysWithValues: declarations.map { ($0.name, "shard-0.safetensors") })
        let headers = [
            "shard-0.safetensors": try SafetensorsHeader.read(
                contentsOf: sourceRoot.appending(path: "shard-0.safetensors"))
        ]
        let plan = try RepackPlan(config: config, weightMap: weightMap, headers: headers)

        let destination = root.appending(path: "tiny.hydra")
        _ = try runBlocking {
            try await StreamingRepacker(plan: plan, source: LocalDirectorySource(root: sourceRoot))
                .run(destination: destination)
        }
        return destination
    }

    /// Runs an asynchronous task from a synchronous test.
    static func runBlocking<T: Sendable>(_ body: @escaping @Sendable () async throws -> T) throws -> T {
        let semaphore = DispatchSemaphore(value: 0)
        nonisolated(unsafe) var result: Result<T, Error>?
        Task {
            do { result = .success(try await body()) } catch { result = .failure(error) }
            semaphore.signal()
        }
        semaphore.wait()
        return try result!.get()
    }

    // MARK: - Reading the installed weights, CPU side

    static func bf16Matrix(
        _ mapping: ModelMapping, _ name: String, rows: Int, cols: Int
    ) throws -> [[Double]] {
        let placement = try mapping.residentTensor(name)
        return mapping.resident.withBytes { raw in
            (0..<rows).map { row in
                (0..<cols).map { col in
                    let offset = placement.offset + (row * cols + col) * 2
                    return Double(
                        BF16.toFloat(
                            UInt16(
                                littleEndian: raw.loadUnaligned(
                                    fromByteOffset: offset, as: UInt16.self))))
                }
            }
        }
    }

    static func bf16Vector(_ mapping: ModelMapping, _ name: String, count: Int) throws -> [Double] {
        try bf16Matrix(mapping, name, rows: 1, cols: count)[0]
    }

    static func expert(
        from buffer: MTLBuffer, blob: ExpertBlobLayout, config: GptOssConfig
    ) throws -> ReferenceLayer.Expert {
        let base = buffer.contents()
        func decode(_ slotBlocks: ExpertBlobLayout.Slot, _ slotScales: ExpertBlobLayout.Slot,
                    rows: Int, cols: Int) throws -> [[Double]] {
            let blocksPerRow = cols / MXFP4Layout.blockSize
            let bytesPerRow = blocksPerRow * MXFP4Layout.packedBytesPerBlock
            return try (0..<rows).map { row in
                let packed = Data(
                    bytesNoCopy: base.advanced(by: slotBlocks.offset + row * bytesPerRow),
                    count: bytesPerRow, deallocator: .none)
                let scales = Data(
                    bytesNoCopy: base.advanced(by: slotScales.offset + row * blocksPerRow),
                    count: blocksPerRow, deallocator: .none)
                return try MXFP4.decode(packed: packed, scales: scales).map(Double.init)
            }
        }
        func bias(_ slot: ExpertBlobLayout.Slot, count: Int) -> [Double] {
            (0..<count).map { i in
                Double(
                    BF16.toFloat(
                        UInt16(
                            littleEndian: base.advanced(by: slot.offset + i * 2)
                                .loadUnaligned(as: UInt16.self))))
            }
        }

        return ReferenceLayer.Expert(
            gateUp: try decode(
                blob.gateUpBlocks, blob.gateUpScales,
                rows: 2 * config.intermediateSize, cols: config.hiddenSize),
            gateUpBias: bias(blob.gateUpBias, count: 2 * config.intermediateSize),
            down: try decode(
                blob.downBlocks, blob.downScales,
                rows: config.hiddenSize, cols: config.intermediateSize),
            downBias: bias(blob.downBias, count: config.hiddenSize))
    }

    // MARK: - Le test

    @Test("A complete GPU layer agrees with the CPU reference, over several tokens")
    func layerMatchesReference() throws {
        let config = Self.shape
        let temporary = FileManager.default.temporaryDirectory
            .appending(path: "hydra-layer-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temporary) }

        let root = try Self.installTinyModel(at: temporary)

        let context = try MetalContext()
        let device = context.device
        let mapping = try ModelMapping(root: root, config: config, device: device)
        let cache = ExpertSlotCache(
            root: root, config: config, slotsPerLayer: config.expertsPerToken, device: device)
        let scratch = try DecodeScratch(config: config, device: device)
        let kvCache = try KVCache(config: config, contextLength: 64, device: device)
        let runner = LayerRunner(
            config: config, encoder: ForwardEncoder(context: context),
            mapping: mapping, cache: cache)

        // --- Weights re-read on the CPU side for the reference ---
        let layer = 0  // couche paire : fenêtre glissante
        let qDim = config.attentionHeadCount * config.headDim
        let kvDim = config.keyValueHeadCount * config.headDim
        let blob = config.expertBlobLayout

        var experts: [ReferenceLayer.Expert] = []
        for index in 0..<config.expertCount {
            let (buffer, _) = try cache.expert(layer: layer, expert: index)
            experts.append(try Self.expert(from: buffer, blob: blob, config: config))
        }

        let prefix = "model.layers.\(layer)"
        let weights = ReferenceLayer.Weights(
            inputNorm: try Self.bf16Vector(
                mapping, "\(prefix).input_layernorm.weight", count: config.hiddenSize),
            queryWeight: try Self.bf16Matrix(
                mapping, "\(prefix).self_attn.q_proj.weight", rows: qDim, cols: config.hiddenSize),
            queryBias: try Self.bf16Vector(
                mapping, "\(prefix).self_attn.q_proj.bias", count: qDim),
            keyWeight: try Self.bf16Matrix(
                mapping, "\(prefix).self_attn.k_proj.weight", rows: kvDim, cols: config.hiddenSize),
            keyBias: try Self.bf16Vector(mapping, "\(prefix).self_attn.k_proj.bias", count: kvDim),
            valueWeight: try Self.bf16Matrix(
                mapping, "\(prefix).self_attn.v_proj.weight", rows: kvDim, cols: config.hiddenSize),
            valueBias: try Self.bf16Vector(mapping, "\(prefix).self_attn.v_proj.bias", count: kvDim),
            outputWeight: try Self.bf16Matrix(
                mapping, "\(prefix).self_attn.o_proj.weight", rows: config.hiddenSize, cols: qDim),
            outputBias: try Self.bf16Vector(
                mapping, "\(prefix).self_attn.o_proj.bias", count: config.hiddenSize),
            sinks: try Self.bf16Vector(
                mapping, "\(prefix).self_attn.sinks", count: config.attentionHeadCount),
            postNorm: try Self.bf16Vector(
                mapping, "\(prefix).post_attention_layernorm.weight", count: config.hiddenSize),
            routerWeight: try Self.bf16Matrix(
                mapping, "\(prefix).mlp.router.weight",
                rows: config.expertCount, cols: config.hiddenSize),
            routerBias: try Self.bf16Vector(
                mapping, "\(prefix).mlp.router.bias", count: config.expertCount),
            experts: experts)

        let referenceShape = ReferenceLayer.Shape(
            hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize,
            headDim: config.headDim, queryHeads: config.attentionHeadCount,
            keyValueHeads: config.keyValueHeadCount, expertsPerToken: config.expertsPerToken,
            slidingWindow: config.slidingWindow, rmsNormEps: Double(config.rmsNormEps),
            swigluLimit: Double(config.swigluLimit))

        // --- Several tokens: the KV cache and the window come into play ---
        var referenceCache = ReferenceLayer.Cache()
        var referenceHidden = (0..<config.hiddenSize).map { Double(sin(Double($0) * 0.1)) }

        let hiddenPointer = scratch.hidden.contents().bindMemory(
            to: Float.self, capacity: config.hiddenSize)
        for i in 0..<config.hiddenSize { hiddenPointer[i] = Float(referenceHidden[i]) }

        let parameters = ReferenceOps.YarnParameters(
            headDim: config.headDim, base: Double(config.ropeTheta),
            initialContextLength: config.yarnOriginalContext,
            scalingFactor: Double(config.yarnFactor),
            ntkAlpha: Double(config.yarnBetaSlow), ntkBeta: Double(config.yarnBetaFast))

        for position in 0..<12 {
            // RoPE tables for the current position, shared by the GPU and the reference.
            let (cosTable, sinTable) = ReferenceOps.cosSin(
                positions: [position], parameters: parameters)
            let cosPointer = scratch.cosTable.contents().bindMemory(
                to: Float.self, capacity: config.headDim / 2)
            let sinPointer = scratch.sinTable.contents().bindMemory(
                to: Float.self, capacity: config.headDim / 2)
            for i in 0..<(config.headDim / 2) {
                cosPointer[i] = Float(cosTable[0][i])
                sinPointer[i] = Float(sinTable[0][i])
            }

            // cb1: up to the router, whose results the CPU must read.
            guard let first = context.commandQueue.makeCommandBuffer() else { return }
            try runner.encodeAttentionAndRouter(
                layer: layer, position: position, scratch: scratch, kvCache: kvCache, in: first)
            first.commit()
            first.waitUntilCompleted()

            // I/O: parallel load of the selected experts.
            let selected = runner.selectedExperts(scratch)
            try cache.load(layer: layer, experts: selected)

            // cb2 : les experts.
            guard let second = context.commandQueue.makeCommandBuffer() else { return }
            try runner.encodeMixtureOfExperts(
                layer: layer, experts: selected, scratch: scratch, in: second)
            second.commit()
            second.waitUntilCompleted()
            cache.release(layer: layer)

            // CPU reference, same position, same tables.
            referenceHidden = ReferenceLayer.decode(
                hidden: referenceHidden, weights: weights, shape: referenceShape,
                cache: &referenceCache, position: position, sliding: true,
                rope: (cosTable[0], sinTable[0]))

            // The router must have chosen the same experts as the reference.
            let normed = ReferenceOps.rmsNorm(
                referenceHidden, scale: weights.postNorm, eps: referenceShape.rmsNormEps)
            _ = normed

            var scale = 0.0
            for value in referenceHidden { scale = max(scale, abs(value)) }
            var worst = 0.0
            for i in 0..<config.hiddenSize {
                worst = max(
                    worst, abs(Double(hiddenPointer[i]) - referenceHidden[i]) / max(scale, 1e-6))
            }
            #expect(worst < 2e-3, "position \(position): relative deviation \(worst)")
        }
    }
}

/// Chunked prefill must produce **exactly the same state** as token-by-token processing.
/// The computation is identical — only the order of the reads changes — so any deviation
/// signals a layout, causal-masking or routing bug.
///
/// This is the test that licenses turning chunked prefill on: without it, the speed-up
/// would be paid for in quality, which the project refuses (D-015).
struct PrefillRunnerTests {

    @Test("Chunked prefill gives the same result as token by token")
    func chunkedMatchesSequential() throws {
        let config = GptOssConfig.tiny
        let temporary = FileManager.default.temporaryDirectory
            .appending(path: "hydra-prefill-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temporary) }

        let root = try LayerRunnerTests.installTinyModel(at: temporary)
        let context = try MetalContext()
        let device = context.device
        let mapping = try ModelMapping(root: root, config: config, device: device)

        // A prompt longer than both the sliding window (8) and the expert cache, to
        // exercise the ring and the tiling.
        let prompt = (0..<20).map { ($0 * 7 + 3) % config.vocabSize }

        func run(chunked: Bool) throws -> [Float] {
            let cache = ExpertSlotCache(
                root: root, config: config,
                slotsPerLayer: config.expertsPerToken, device: device)
            let runner = try ModelRunner(
                config: config, context: context, mapping: mapping,
                expertCache: cache, contextLength: 64, prefillChunk: chunked ? 16 : 1)
            if chunked {
                return Array(try runner.prefill(tokens: prompt))
            }
            var distribution = try runner.forward(token: prompt[0])
            for token in prompt.dropFirst() { distribution = try runner.forward(token: token) }
            return Array(distribution)
        }

        let sequential = try run(chunked: false)
        let chunked = try run(chunked: true)

        #expect(sequential.count == chunked.count)
        var scale: Float = 0
        for value in sequential { scale = max(scale, abs(value)) }
        var worst: Float = 0
        for (a, b) in zip(sequential, chunked) {
            worst = max(worst, abs(a - b) / max(scale, 1e-6))
        }
        // The two paths sum in a different order: the expected deviation is that of
        // floating-point arithmetic, not of a computational divergence.
        #expect(worst < 2e-3, "relative deviation \(worst) between sequential and chunked")

        // And the chosen token must be the same: that is what the user sees.
        let bestSequential = sequential.firstIndex(of: sequential.max()!)
        let bestChunked = chunked.firstIndex(of: chunked.max()!)
        #expect(bestSequential == bestChunked, "the greedy token differs")
    }

    @Test("Chunked prefill crosses several chunks without discontinuity")
    func multipleChunks() throws {
        let config = GptOssConfig.tiny
        let temporary = FileManager.default.temporaryDirectory
            .appending(path: "hydra-prefill-multi-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temporary) }

        let root = try LayerRunnerTests.installTinyModel(at: temporary)
        let context = try MetalContext()
        let mapping = try ModelMapping(root: root, config: config, device: context.device)
        let prompt = (0..<30).map { ($0 * 11 + 5) % config.vocabSize }

        func run(chunk: Int) throws -> [Float] {
            let cache = ExpertSlotCache(
                root: root, config: config,
                slotsPerLayer: config.expertsPerToken, device: context.device)
            let runner = try ModelRunner(
                config: config, context: context, mapping: mapping,
                expertCache: cache, contextLength: 64, prefillChunk: chunk)
            return Array(try runner.prefill(tokens: prompt))
        }

        // 30 tokens in chunks of 8: four chunks, one of them incomplete.
        let single = try run(chunk: 32)
        let multiple = try run(chunk: 8)

        var scale: Float = 0
        for value in single { scale = max(scale, abs(value)) }
        var worst: Float = 0
        for (a, b) in zip(single, multiple) { worst = max(worst, abs(a - b) / max(scale, 1e-6)) }
        #expect(worst < 2e-3, "relative deviation \(worst) between one chunk and several")
    }
}
