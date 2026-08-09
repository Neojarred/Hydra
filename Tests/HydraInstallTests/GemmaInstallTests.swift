import Foundation
import HydraCore
import HydraFormat
import Testing

@testable import HydraInstall

/// Installing a Gemma 4 model end to end, through the same repacker GPT-OSS uses.
///
/// The plan was already checked in isolation; what this proves is that **the repacker needed no
/// knowledge of the architecture to execute it**. If installing a second model had required a
/// branch inside `StreamingRepacker`, D-023's seam would be in the wrong place — and this suite
/// is what would have said so.
///
/// A miniature checkpoint rather than the real 50 GB one: what has to be right here is
/// structural — that every byte lands where the layout says, and that the manifest describes
/// what is actually on disk.
@Suite("Gemma 4 installation")
struct GemmaInstallTests {

    private let config = Gemma4Config.tiny

    /// Writes a synthetic Gemma checkpoint, vision tower included, and repacks it.
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
        // A vision tower, which must be skipped and still accounted for.
        for layer in 0..<2 {
            declarations.append((
                "model.vision_tower.encoder.layers.\(layer).self_attn.q_proj.linear.weight",
                [32, 32], 2048))
        }
        declarations.append(
            ("model.embed_vision.embedding_projection.weight", [32, 32], 2048))

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
        // Deterministic content, distinct per tensor, so a misplaced byte is detectable.
        for t in declarations {
            var state = UInt32(truncatingIfNeeded: t.name.hashValue) | 1
            var bytes = [UInt8](repeating: 0, count: t.bytes)
            for i in 0..<t.bytes {
                state = state &* 1_103_515_245 &+ 12345
                bytes[i] = UInt8(truncatingIfNeeded: state >> 16)
            }
            file.append(contentsOf: bytes)
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

        let problems = plan.validate(weightMap: weightMap, declaredSourceTotal: cursor)
        #expect(problems.isEmpty, "\(problems.map(\.description))")

        let destination = root.appending(path: "tiny-gemma.hydra")
        _ = try await StreamingRepacker(
            plan: plan, source: LocalDirectorySource(root: sourceRoot)
        ).run(destination: destination)
        return destination
    }

    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appending(path: "hydra-gemma-install-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    /// The repacker executed a Gemma plan without knowing it was one.
    @Test("A Gemma model installs through the shared repacker")
    func installsThroughTheSharedRepacker() async throws {
        let temporary = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: temporary) }

        let root = try await install(at: temporary)
        let manifest = try HydraManifest.read(from: root)

        #expect(manifest.model.architecture == ModelArchitecture.gemma4.rawValue)
        // Recorded from the source, not hardcoded — which is what makes the QAT variant a
        // value rather than a branch.
        #expect(manifest.model.expertQuantization == "bf16")
        try manifest.validate(against: config, root: root)
    }

    /// The files the layout describes, and no others. An `embed.bin` here would be a file
    /// nothing maps, because Gemma's embedding is tied to the output head.
    @Test("The installation contains exactly the planned files")
    func filesMatchTheLayout() async throws {
        let temporary = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: temporary) }

        let root = try await install(at: temporary)
        let layout = HydraLayout(config: config)

        let resident = try Data(contentsOf: root.appending(path: "resident.bin"))
        #expect(resident.count == layout.residentBytes)
        #expect(!FileManager.default.fileExists(atPath: root.appending(path: "embed.bin").path))

        for layer in 0..<config.layerCount {
            let file = root.appending(path: String(format: "experts/layer_%02d.bin", layer))
            let data = try Data(contentsOf: file)
            #expect(data.count == layout.expertLayerFileBytes)
        }
    }

    /// The check that matters most: the bytes on disk are the bytes from the source, at the
    /// offsets the layout claims. A plan that scattered experts to the wrong stride would
    /// produce files of exactly the right size and entirely wrong contents.
    @Test("Every resident tensor lands where the layout says")
    func bytesLandWhereTheLayoutSays() async throws {
        let temporary = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: temporary) }

        let root = try await install(at: temporary)
        let layout = HydraLayout(config: config)

        let installed = try Data(contentsOf: root.appending(path: "resident.bin"))
        let sourceFile = try Data(
            contentsOf: temporary.appending(path: "source/shard-0.safetensors"))
        let header = try SafetensorsHeader.read(
            contentsOf: temporary.appending(path: "source/shard-0.safetensors"))

        for placement in layout.resident {
            guard let range = header.fileRange(of: placement.sourceName) else {
                Issue.record("\(placement.sourceName) missing from the source")
                continue
            }
            let expected = sourceFile[range]
            let got = installed[placement.offset..<placement.end]
            #expect(Array(got) == Array(expected), "\(placement.sourceName) is misplaced")
        }
    }

    /// The vision tower is present in the source and absent from the installation, and the
    /// plan says so rather than staying silent about it.
    @Test("The vision tower is excluded and recorded")
    func visionIsExcludedAndRecorded() async throws {
        let temporary = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: temporary) }

        _ = try await install(at: temporary)
        // The plan is rebuilt here rather than returned, to check the same source twice.
        let sourceRoot = temporary.appending(path: "source")
        let header = try SafetensorsHeader.read(
            contentsOf: sourceRoot.appending(path: "shard-0.safetensors"))
        let weightMap = Dictionary(
            uniqueKeysWithValues: header.tensors.keys.map { ($0, "shard-0.safetensors") })
        let plan = try GemmaRepackPlan(
            config: config, weightMap: weightMap, headers: ["shard-0.safetensors": header])

        #expect(plan.excluded.count == 3)
        #expect(plan.totalExcludedBytes == 3 * 2048)
        #expect(!plan.operations.contains { $0.sourceTensor.contains("vision") })
    }
}
