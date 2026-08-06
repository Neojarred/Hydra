import Foundation
import HydraCore
import HydraFormat
import Testing

@testable import HydraInstall

/// End-to-end repack over a synthetic checkpoint written to disk in the **real safetensors
/// format**, then read back byte by byte.
///
/// This is the test that validates the scatter logic: in the source, a layer's experts sit
/// end to end; in `.hydra` they are interleaved with the other sub-tensors. A single byte
/// of offset error would produce a model that loads without error and generates noise —
/// exactly the kind of bug one does not want to discover while writing kernels.
///
/// The progress callback is `@Sendable`: it cannot mutate a captured variable. This small
/// box serializes the observations made from inside the repacker.
final class Observed<Value>: @unchecked Sendable {
    private var storage: Value
    private let lock = NSLock()

    init(_ initial: Value) { storage = initial }

    var value: Value {
        lock.lock(); defer { lock.unlock() }
        return storage
    }

    func update(_ transform: (inout Value) -> Void) {
        lock.lock(); defer { lock.unlock() }
        transform(&storage)
    }
}

/// A source that fails once a byte budget is exceeded.
///
/// Interrupting an install by cancelling a task depends on the scheduler: on a miniature
/// model, the install sometimes finished before the cancellation took effect, and the test
/// failed at random. A failure triggered by the volume transferred is reproducible exactly.
///
struct FailingSource: ByteRangeSource {
    let inner: LocalDirectorySource
    let budget: Int
    let transferred = Counter()

    struct Interrupted: Error {}

    var sourceDescription: String { inner.sourceDescription }

    func read(file: String, range: Range<Int>) async throws -> Data {
        try await inner.read(file: file, range: range)
    }

    func stream(
        file: String, range: Range<Int>,
        into sink: @escaping @Sendable (Data) throws -> Void
    ) async throws {
        try await inner.stream(file: file, range: range) { block in
            if transferred.value >= budget { throw Interrupted() }
            transferred.add(block.count)
            try sink(block)
        }
    }
}

struct StreamingRepackerTests {

    /// A deterministic pattern, unique to each byte of the checkpoint: any permutation,
    /// truncation or offset makes the verification fail.
    static func expectedByte(tensor: String, index: Int) -> UInt8 {
        var h: UInt64 = 1469598103934665603
        for b in tensor.utf8 { h = (h ^ UInt64(b)) &* 1099511628211 }
        h = (h ^ UInt64(truncatingIfNeeded: index)) &* 1099511628211
        return UInt8(truncatingIfNeeded: h >> 13)
    }

    struct Declaration {
        let name: String
        let dtype: String
        let shape: [Int]
        let bytes: Int
    }

    static func declarations(for config: GptOssConfig) -> [Declaration] {
        let blob = config.expertBlobLayout
        var out: [Declaration] = []
        for layer in 0..<config.layerCount {
            for entry in HydraLayout.residentTensorNames(layer: layer) {
                let n = entry.bytes(config)
                out.append(Declaration(name: entry.name, dtype: "BF16", shape: [n / 2], bytes: n))
            }
            for (suffix, slot) in [
                ("gate_up_proj_blocks", blob.gateUpBlocks),
                ("gate_up_proj_scales", blob.gateUpScales),
                ("gate_up_proj_bias", blob.gateUpBias),
                ("down_proj_blocks", blob.downBlocks),
                ("down_proj_scales", blob.downScales),
                ("down_proj_bias", blob.downBias),
            ] {
                out.append(
                    Declaration(
                        name: "model.layers.\(layer).mlp.experts.\(suffix)",
                        dtype: suffix.hasSuffix("bias") ? "BF16" : "U8",
                        shape: [config.expertCount, slot.byteCount],
                        bytes: slot.byteCount * config.expertCount))
            }
        }
        out.append(
            Declaration(name: "model.norm.weight", dtype: "BF16",
                        shape: [config.hiddenSize], bytes: 2 * config.hiddenSize))
        out.append(
            Declaration(name: "lm_head.weight", dtype: "BF16",
                        shape: [config.vocabSize, config.hiddenSize], bytes: config.lmHeadBytes))
        out.append(
            Declaration(name: "model.embed_tokens.weight", dtype: "BF16",
                        shape: [config.vocabSize, config.hiddenSize], bytes: config.embeddingBytes))
        return out
    }

    /// Writes a valid safetensors checkpoint, spread over several shards.
    static func writeSyntheticCheckpoint(
        config: GptOssConfig, to root: URL, shardCount: Int = 3
    ) throws -> SafetensorsIndex {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)

        var weightMap: [String: String] = [:]
        var perShard: [String: [Declaration]] = [:]
        for (i, d) in declarations(for: config).enumerated() {
            let shard = "shard-\(i % shardCount).safetensors"
            weightMap[d.name] = shard
            perShard[shard, default: []].append(d)
        }

        for (shard, tensors) in perShard {
            var header: [String: Any] = [:]
            var cursor = 0
            for t in tensors {
                header[t.name] = [
                    "dtype": t.dtype, "shape": t.shape,
                    "data_offsets": [cursor, cursor + t.bytes],
                ]
                cursor += t.bytes
            }
            var headerJSON = try JSONSerialization.data(
                withJSONObject: header, options: [.sortedKeys])
            // The safetensors header must be 8-byte aligned: we pad it with spaces.
            while headerJSON.count % 8 != 0 { headerJSON.append(0x20) }

            var file = Data()
            withUnsafeBytes(of: UInt64(headerJSON.count).littleEndian) { file.append(contentsOf: $0) }
            file.append(headerJSON)
            for t in tensors {
                var payload = Data(count: t.bytes)
                for i in 0..<t.bytes { payload[i] = expectedByte(tensor: t.name, index: i) }
                file.append(payload)
            }
            try file.write(to: root.appending(path: shard))
        }

        let total = declarations(for: config).reduce(0) { $0 + $1.bytes }
        return SafetensorsIndex(weightMap: weightMap, totalSize: total)
    }

    static func makeTemporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appending(path: "hydra-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    static func loadHeaders(root: URL, index: SafetensorsIndex) throws -> [String: SafetensorsHeader] {
        var headers: [String: SafetensorsHeader] = [:]
        for shard in index.shards {
            headers[shard] = try SafetensorsHeader.read(contentsOf: root.appending(path: shard))
        }
        return headers
    }

    // MARK: - Le test central

    @Test("Full repack: every byte lands in the right place")
    func repackPlacesEveryByteCorrectly() async throws {
        let config = GptOssConfig.tiny
        let temp = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: temp) }

        let sourceRoot = temp.appending(path: "source")
        let index = try Self.writeSyntheticCheckpoint(config: config, to: sourceRoot)
        let headers = try Self.loadHeaders(root: sourceRoot, index: index)
        let plan = try RepackPlan(config: config, weightMap: index.weightMap, headers: headers)
        #expect(plan.validate(declaredSourceTotal: index.totalSize).isEmpty)

        let destination = temp.appending(path: "tiny.hydra")
        // A deliberately tiny buffer, to force the chunking to cross expert blob
        // boundaries.
        let repacker = StreamingRepacker(
            plan: plan, source: LocalDirectorySource(root: sourceRoot), checkpointInterval: 3)

        let peak = Observed(0)
        let manifest = try await repacker.run(destination: destination) { progress in
            peak.update { $0 = max($0, progress.peakPayloadBytes) }
        }

        // The local source delivers 64 KiB blocks, like URLSession.
        #expect(peak.value <= 64 * 1024,
                Comment(rawValue: "memory invariant violated: peak of \(peak.value) bytes"))
        try manifest.validate(against: config, root: destination)

        // --- Checking the experts, sub-tensor by sub-tensor ---
        let blob = config.expertBlobLayout
        for layer in 0..<config.layerCount {
            let data = try Data(
                contentsOf: destination.appending(path: "experts/layer_\(String(format: "%02d", layer)).bin"),
                options: .mappedIfSafe)
            for (suffix, slot) in [
                ("gate_up_proj_blocks", blob.gateUpBlocks),
                ("gate_up_proj_scales", blob.gateUpScales),
                ("gate_up_proj_bias", blob.gateUpBias),
                ("down_proj_blocks", blob.downBlocks),
                ("down_proj_scales", blob.downScales),
                ("down_proj_bias", blob.downBias),
            ] {
                let tensor = "model.layers.\(layer).mlp.experts.\(suffix)"
                for expert in 0..<config.expertCount {
                    let base = expert * blob.strideBytes + slot.offset
                    for i in stride(from: 0, to: slot.byteCount, by: 7) {
                        let expected = Self.expectedByte(
                            tensor: tensor, index: expert * slot.byteCount + i)
                        #expect(
                            data[base + i] == expected,
                            Comment(rawValue: "\(tensor) expert \(expert) octet \(i)"))
                    }
                }
            }
        }

        // --- Checking the residents ---
        let resident = try Data(
            contentsOf: destination.appending(path: "resident.bin"), options: .mappedIfSafe)
        for placement in plan.layout.resident {
            for i in stride(from: 0, to: placement.byteCount, by: 13) {
                let expected = Self.expectedByte(tensor: placement.sourceName, index: i)
                #expect(
                    resident[placement.offset + i] == expected,
                    Comment(rawValue: "\(placement.sourceName) octet \(i)"))
            }
        }

        // --- Checking the embedding ---
        let embed = try Data(
            contentsOf: destination.appending(path: "embed.bin"), options: .mappedIfSafe)
        #expect(embed.count == config.embeddingBytes)
        for i in stride(from: 0, to: embed.count, by: 11) {
            #expect(embed[i] == Self.expectedByte(tensor: "model.embed_tokens.weight", index: i))
        }
    }

    /// The invariant must not depend on region size: a region of several hundred megabytes is
    /// consumed with the same footprint as a small one.
    @Test("The footprint does not depend on the size of the regions downloaded")
    func footprintIsIndependentOfSpanSize() async throws {
        let config = GptOssConfig.tiny
        let temp = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: temp) }
        let sourceRoot = temp.appending(path: "source")
        let index = try Self.writeSyntheticCheckpoint(config: config, to: sourceRoot)
        let headers = try Self.loadHeaders(root: sourceRoot, index: index)
        let plan = try RepackPlan(config: config, weightMap: index.weightMap, headers: headers)

        // Since the plan covers the whole checkpoint with no gaps, the regions must be far
        // fewer than the operations: that is what gives the throughput.
        #expect(plan.spans.count < plan.operations.count / 4,
                Comment(rawValue: "\(plan.spans.count) regions for \(plan.operations.count) operations"))
        #expect(plan.spans.reduce(0) { $0 + $1.range.count } == plan.totalSourceBytes)

        let peak = Observed(0)
        _ = try await StreamingRepacker(
            plan: plan, source: LocalDirectorySource(root: sourceRoot)
        ).run(destination: temp.appending(path: "m.hydra")) { progress in
            peak.update { $0 = max($0, progress.peakPayloadBytes) }
        }
        #expect(peak.value <= 64 * 1024)
    }

    @Test("An install exists under its final name only once complete")
    func installationIsAtomic() async throws {
        let config = GptOssConfig.tiny
        let temp = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: temp) }
        let sourceRoot = temp.appending(path: "source")
        let index = try Self.writeSyntheticCheckpoint(config: config, to: sourceRoot)
        let headers = try Self.loadHeaders(root: sourceRoot, index: index)
        let plan = try RepackPlan(config: config, weightMap: index.weightMap, headers: headers)
        let destination = temp.appending(path: "tiny.hydra")

        let sawFinalDirectoryEarly = Observed(false)
        _ = try await StreamingRepacker(
            plan: plan, source: LocalDirectorySource(root: sourceRoot)
        ).run(destination: destination) { progress in
            if progress.bytesDone < progress.bytesTotal,
                FileManager.default.fileExists(atPath: destination.path) {
                sawFinalDirectoryEarly.update { $0 = true }
            }
        }

        #expect(!sawFinalDirectoryEarly.value, "the final directory appeared before completion")
        #expect(FileManager.default.fileExists(atPath: destination.path))
        #expect(!FileManager.default.fileExists(
            atPath: destination.appendingPathExtension("partial").path))
    }

    @Test("A manifest from another architecture is rejected")
    func manifestRejectsWrongArchitecture() async throws {
        let config = GptOssConfig.tiny
        let temp = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: temp) }
        let sourceRoot = temp.appending(path: "source")
        let index = try Self.writeSyntheticCheckpoint(config: config, to: sourceRoot)
        let headers = try Self.loadHeaders(root: sourceRoot, index: index)
        let plan = try RepackPlan(config: config, weightMap: index.weightMap, headers: headers)
        let destination = temp.appending(path: "tiny.hydra")

        let manifest = try await StreamingRepacker(
            plan: plan, source: LocalDirectorySource(root: sourceRoot)
        ).run(destination: destination)

        let other = GptOssConfig(
            name: "autre", layerCount: 4, expertCount: 8, expertsPerToken: 2,
            hiddenSize: 64, intermediateSize: 64, headDim: 16,
            attentionHeadCount: 4, keyValueHeadCount: 2, vocabSize: 128, slidingWindow: 8)
        #expect(throws: HydraManifest.ValidationError.self) {
            try manifest.validate(against: other, root: destination)
        }
    }

    @Test("A resume reconstructs exactly the same install")
    func resumeProducesIdenticalResult() async throws {
        let config = GptOssConfig.tiny
        let temp = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: temp) }
        let sourceRoot = temp.appending(path: "source")
        let index = try Self.writeSyntheticCheckpoint(config: config, to: sourceRoot)
        let headers = try Self.loadHeaders(root: sourceRoot, index: index)
        let plan = try RepackPlan(config: config, weightMap: index.weightMap, headers: headers)
        let source = LocalDirectorySource(root: sourceRoot)

        // Reference: an install carried out in one go.
        let reference = temp.appending(path: "reference.hydra")
        _ = try await StreamingRepacker(plan: plan, source: source).run(destination: reference)

        // Interrupted deterministically at a third of the transfer.
        let interrupted = temp.appending(path: "resumed.hydra")
        let failing = FailingSource(inner: source, budget: plan.totalSourceBytes / 3)
        do {
            _ = try await StreamingRepacker(
                plan: plan, source: failing, checkpointInterval: 1
            ).run(destination: interrupted)
            Issue.record("the install should have failed")
        } catch {
            // Attendu.
        }

        // Nothing was promoted: the partial remains for the resume.
        #expect(!FileManager.default.fileExists(atPath: interrupted.path))
        #expect(FileManager.default.fileExists(
            atPath: interrupted.appendingPathExtension("partial").path))

        // Resume with a healthy source.
        _ = try await StreamingRepacker(plan: plan, source: source).run(destination: interrupted)

        for file in plan.destinationSizes.keys {
            let a = try Data(contentsOf: reference.appending(path: file.path))
            let b = try Data(contentsOf: interrupted.appending(path: file.path))
            #expect(a == b, Comment(rawValue: "divergence sur \(file.path)"))
        }
    }
}
