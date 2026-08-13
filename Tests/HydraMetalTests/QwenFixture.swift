import Foundation
import HydraCore
import HydraFormat
import HydraInstall
import Testing

@testable import HydraMetal

/// A miniature Qwen checkpoint, installed through the real plan.
///
/// Shared by the behavioural suite and the oracle suite, so that both answer questions about
/// the same installation. Two fixtures drifting apart is how a model comes to pass one suite
/// and fail the other for reasons that have nothing to do with the model.
struct QwenFixture {

    /// Deterministic bytes for one tensor, chosen so nothing decodes to an extreme.
    func bytes(for name: String, count: Int, quantized: Bool, config: Qwen35MoeConfig) -> Data {
        var state = UInt64(truncatingIfNeeded: name.hashValue) | 1
        func next() -> UInt64 {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return state >> 33
        }
        var data = Data(capacity: count)
        if quantized {
            for _ in 0..<count { data.append(UInt8(truncatingIfNeeded: next())) }
            return data
        }

        // BF16, kept small and well conditioned. Scales and biases are generated as a pair, so
        // that `q · scale + bias` with `q` in 0…15 is centred on zero rather than on
        // `7.5 · scale`, which would saturate every activation by the third layer.
        let isScale = name.hasSuffix(".scales")
        let isBias = name.hasSuffix(".biases")
        let midpoint = Float((1 << config.quantBits) - 1) / 2
        // `A_log` feeds `exp(-exp(A_log) · softplus(…))`. Left in the same range as everything
        // else it reaches `exp(exp(0.25))`, which decays the state to nothing in two tokens and
        // makes the linear layers indistinguishable from zeros.
        let isDecay = name.hasSuffix("A_log")
        // A norm weight multiplies a normalized value, so centring it on zero would cancel the
        // layer. Centred on one, as trained norms are.
        let isNorm = name.hasSuffix("norm.weight") || name.hasSuffix("layernorm.weight")
        for _ in 0..<(count / 2) {
            let value: Float
            if isScale {
                value = Float(Int(next() % 40) + 10) / 1000
            } else if isBias {
                value = -midpoint * Float(Int(next() % 40) + 10) / 1000
            } else if isDecay {
                value = -Float(Int(next() % 200) + 100) / 100
            } else if isNorm {
                value = 1 + Float(Int(next() % 100) - 50) / 500
            } else {
                value = Float(Int(next() % 200) - 100) / 400
            }
            let bits = BF16.fromFloat(value)
            data.append(UInt8(bits & 0xFF))
            data.append(UInt8(bits >> 8))
        }
        return data
    }

    func install(at root: URL, config: Qwen35MoeConfig) async throws -> URL {
        var declarations: [(name: String, shape: [Int], bytes: Int, quantized: Bool)] = []

        // A tensor is quantized exactly when the checkpoint also carries its scales. Guessing
        // from the name fails: `linear_attn.conv1d.weight` ends in `.weight` and is BF16.
        let resident = config.residentTensors
        let quantizedStems = Set(
            resident.map(\.name).filter { $0.hasSuffix(".scales") }
                .map { String($0.dropLast(".scales".count)) })
        for (name, byteCount) in resident {
            let quantized = name.hasSuffix(".weight")
                && quantizedStems.contains(String(name.dropLast(".weight".count)))
            declarations.append((name, [byteCount / (quantized ? 4 : 2)], byteCount, quantized))
        }

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

        // A small vision tower, because the real checkpoint has one and the plan installs it.
        for layer in 0..<2 {
            declarations.append((
                "vision_tower.encoder.layers.\(layer).self_attn.q_proj.weight",
                [16, 16], 512, false))
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
            file.append(bytes(for: t.name, count: t.bytes, quantized: t.quantized, config: config))
        }
        try file.write(to: sourceRoot.appending(path: "shard-0.safetensors"))

        let weightMap = Dictionary(
            uniqueKeysWithValues: declarations.map { ($0.name, "shard-0.safetensors") })
        let headers = [
            "shard-0.safetensors": try SafetensorsHeader.read(
                contentsOf: sourceRoot.appending(path: "shard-0.safetensors"))
        ]
        let plan = try QwenRepackPlan(config: config, weightMap: weightMap, headers: headers)
        #expect(
            plan.validate(weightMap: weightMap, declaredSourceTotal: cursor).isEmpty,
            "the fixture must be a checkpoint the plan fully covers")

        let destination = root.appending(path: "tiny-qwen.hydra")
        _ = try await StreamingRepacker(
            plan: plan, source: LocalDirectorySource(root: sourceRoot)
        ).run(destination: destination)
        return destination
    }

    func temporary() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appending(path: "hydra-qwen-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

}
