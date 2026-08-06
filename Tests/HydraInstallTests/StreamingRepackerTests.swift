import Foundation
import HydraCore
import HydraFormat
import Testing

@testable import HydraInstall

/// Repack de bout en bout sur un checkpoint synthétique écrit sur disque au **vrai format
/// safetensors**, puis relecture octet par octet.
///
/// C'est le test qui valide la logique d'éparpillement : dans la source, les experts d'une
/// couche sont bout à bout ; dans `.hydra`, ils sont entrelacés avec les autres
/// sous-tenseurs. Une erreur d'un seul octet de décalage produirait un modèle qui charge
/// sans erreur et génère du bruit — exactement le type de bug qu'on ne veut pas découvrir
/// au moment des noyaux.
/// Le rappel de progression est `@Sendable` : il ne peut pas muter une variable capturée.
/// Cette petite boîte sérialise les observations faites depuis le repacker.
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

/// Source qui échoue une fois un budget d'octets dépassé.
///
/// Interrompre une installation en annulant une tâche dépend de l'ordonnanceur : sur un
/// modèle miniature, l'installation se terminait parfois avant que l'annulation ne prenne
/// effet, et le test échouait au hasard. Une panne déclenchée par le volume transféré est
/// reproductible à l'identique.
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

    /// Motif déterministe et propre à chaque octet du checkpoint : n'importe quelle
    /// permutation, troncature ou décalage rend la vérification fausse.
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

    /// Écrit un checkpoint safetensors valide, réparti sur plusieurs shards.
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
            // L'en-tête safetensors doit être aligné sur 8 octets : on complète par des espaces.
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

    @Test("Repack complet : chaque octet arrive à la bonne place")
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
        // Tampon volontairement minuscule, pour forcer le découpage à traverser
        // les frontières de blobs d'experts.
        let repacker = StreamingRepacker(
            plan: plan, source: LocalDirectorySource(root: sourceRoot), checkpointInterval: 3)

        let peak = Observed(0)
        let manifest = try await repacker.run(destination: destination) { progress in
            peak.update { $0 = max($0, progress.peakPayloadBytes) }
        }

        // La source locale livre des blocs de 64 kio, comme URLSession.
        #expect(peak.value <= 64 * 1024,
                Comment(rawValue: "invariant mémoire violé : pic de \(peak.value) octets"))
        try manifest.validate(against: config, root: destination)

        // --- Vérification des experts, sous-tenseur par sous-tenseur ---
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

        // --- Vérification des résidents ---
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

        // --- Vérification de l'embedding ---
        let embed = try Data(
            contentsOf: destination.appending(path: "embed.bin"), options: .mappedIfSafe)
        #expect(embed.count == config.embeddingBytes)
        for i in stride(from: 0, to: embed.count, by: 11) {
            #expect(embed[i] == Self.expectedByte(tensor: "model.embed_tokens.weight", index: i))
        }
    }

    /// L'invariant ne doit pas dépendre de la taille des régions : une région de plusieurs
    /// centaines de mégaoctets se consomme avec la même empreinte qu'une petite.
    @Test("L'empreinte ne dépend pas de la taille des régions téléchargées")
    func footprintIsIndependentOfSpanSize() async throws {
        let config = GptOssConfig.tiny
        let temp = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: temp) }
        let sourceRoot = temp.appending(path: "source")
        let index = try Self.writeSyntheticCheckpoint(config: config, to: sourceRoot)
        let headers = try Self.loadHeaders(root: sourceRoot, index: index)
        let plan = try RepackPlan(config: config, weightMap: index.weightMap, headers: headers)

        // Le plan couvrant tout le checkpoint sans trou, les régions doivent être bien
        // moins nombreuses que les opérations : c'est ce qui fait le débit.
        #expect(plan.spans.count < plan.operations.count / 4,
                Comment(rawValue: "\(plan.spans.count) régions pour \(plan.operations.count) opérations"))
        #expect(plan.spans.reduce(0) { $0 + $1.range.count } == plan.totalSourceBytes)

        let peak = Observed(0)
        _ = try await StreamingRepacker(
            plan: plan, source: LocalDirectorySource(root: sourceRoot)
        ).run(destination: temp.appending(path: "m.hydra")) { progress in
            peak.update { $0 = max($0, progress.peakPayloadBytes) }
        }
        #expect(peak.value <= 64 * 1024)
    }

    @Test("Une installation n'existe sous son nom final qu'une fois complète")
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

        #expect(!sawFinalDirectoryEarly.value, "le répertoire final est apparu avant la fin")
        #expect(FileManager.default.fileExists(atPath: destination.path))
        #expect(!FileManager.default.fileExists(
            atPath: destination.appendingPathExtension("partial").path))
    }

    @Test("Un manifeste d'une autre architecture est rejeté")
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

    @Test("Une reprise reconstitue exactement la même installation")
    func resumeProducesIdenticalResult() async throws {
        let config = GptOssConfig.tiny
        let temp = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: temp) }
        let sourceRoot = temp.appending(path: "source")
        let index = try Self.writeSyntheticCheckpoint(config: config, to: sourceRoot)
        let headers = try Self.loadHeaders(root: sourceRoot, index: index)
        let plan = try RepackPlan(config: config, weightMap: index.weightMap, headers: headers)
        let source = LocalDirectorySource(root: sourceRoot)

        // Référence : installation menée d'un trait.
        let reference = temp.appending(path: "reference.hydra")
        _ = try await StreamingRepacker(plan: plan, source: source).run(destination: reference)

        // Interrompue de façon déterministe au tiers du transfert.
        let interrupted = temp.appending(path: "resumed.hydra")
        let failing = FailingSource(inner: source, budget: plan.totalSourceBytes / 3)
        do {
            _ = try await StreamingRepacker(
                plan: plan, source: failing, checkpointInterval: 1
            ).run(destination: interrupted)
            Issue.record("l'installation aurait dû échouer")
        } catch {
            // Attendu.
        }

        // Rien n'a été promu : le partiel subsiste pour la reprise.
        #expect(!FileManager.default.fileExists(atPath: interrupted.path))
        #expect(FileManager.default.fileExists(
            atPath: interrupted.appendingPathExtension("partial").path))

        // Reprise avec une source saine.
        _ = try await StreamingRepacker(plan: plan, source: source).run(destination: interrupted)

        for file in plan.destinationSizes.keys {
            let a = try Data(contentsOf: reference.appending(path: file.path))
            let b = try Data(contentsOf: interrupted.appending(path: file.path))
            #expect(a == b, Comment(rawValue: "divergence sur \(file.path)"))
        }
    }
}
