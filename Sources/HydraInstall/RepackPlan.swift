import Foundation
import HydraCore
import HydraFormat

/// Fichier destination d'une installation `.hydra`.
public enum DestinationFile: Sendable, Hashable {
    case resident
    case embedding
    case expertLayer(Int)

    public var path: String {
        switch self {
        case .resident: return "resident.bin"
        case .embedding: return "embed.bin"
        case .expertLayer(let i): return String(format: "experts/layer_%02d.bin", i)
        }
    }
}

/// Copie d'une plage source vers une ou plusieurs destinations régulièrement espacées.
///
/// Le cas `chunkCount == 1` est une copie contiguë ordinaire (un tenseur résident).
///
/// Le cas `chunkCount > 1` est ce qui rend le repack des experts efficace. Dans le
/// checkpoint source, un tenseur comme `gate_up_proj_blocks` contient **tous** les experts
/// d'une couche bout à bout. Dans `.hydra`, ils doivent être répartis dans des blobs
/// entrelacés avec les autres sous-tenseurs. Plutôt que d'émettre une requête par expert,
/// on lit la plage source **une seule fois, séquentiellement**, et on éparpille chaque
/// morceau à `destinationOffset + i * destinationStride`.
///
/// Conséquence directe : environ 150 requêtes réseau pour installer le 20B, au lieu de
/// plusieurs milliers — tout en gardant un tampon de travail borné.
public struct ScatterCopy: Sendable, Equatable {
    public let sourceTensor: String
    public let sourceShard: String
    /// Décalage absolu dans le shard, section de données comprise.
    public let sourceOffset: Int
    public let destination: DestinationFile
    public let destinationOffset: Int
    public let destinationStride: Int
    public let chunkByteCount: Int
    public let chunkCount: Int

    public var sourceByteCount: Int { chunkByteCount * chunkCount }
    public var sourceRange: Range<Int> { sourceOffset..<(sourceOffset + sourceByteCount) }

    /// Décalage destination du morceau `index`.
    public func destinationOffset(ofChunk index: Int) -> Int {
        destinationOffset + index * destinationStride
    }
}

/// Plan complet de conversion d'un checkpoint Hugging Face vers `.hydra`.
///
/// Le plan est calculé **à partir des seuls en-têtes** — quelques dizaines de kio de
/// réseau — et entièrement vérifiable avant d'engager le moindre téléchargement de poids.
public struct RepackPlan: Sendable {

    public let config: GptOssConfig
    public let layout: HydraLayout
    /// Triées par (shard, décalage source) : la lecture reste séquentielle sur le disque
    /// distant, ce qui évite de rouvrir des connexions en arrière.
    public let operations: [ScatterCopy]

    /// Régions contiguës du checkpoint source, chacune couverte par des opérations
    /// consécutives. Une région se télécharge en **une seule requête**, dont la réponse
    /// est routée vers plusieurs destinations au fil de l'arrivée des octets.
    ///
    /// C'est ce qui rend l'installation rapide. Le plan couvrant exactement le checkpoint
    /// sans trou, les tenseurs voisins d'un shard forment de longues régions contiguës :
    /// on lit le fichier source presque de bout en bout, séquentiellement, au lieu
    /// d'émettre une requête par tenseur. Mesuré sur le vrai dépôt : 33,5 Mo/s en grandes
    /// requêtes contre 5,2 Mo/s en petites.
    public let spans: [SourceSpan]

    public let destinationSizes: [DestinationFile: Int]

    /// Plafond d'une région. Ne borne pas la mémoire — la réponse est consommée au fil de
    /// l'eau — mais borne ce qu'une interruption réseau oblige à refaire, et donne au
    /// journal de reprise une granularité utile.
    public static let maximumSpanBytes = 256 * 1024 * 1024

    public struct SourceSpan: Sendable, Equatable {
        public let shard: String
        public let range: Range<Int>
        /// Indices dans `operations`, consécutifs et ordonnés. Mis bout à bout, ils
        /// couvrent exactement `range`.
        public let operationIndices: [Int]
    }

    static func makeSpans(operations: [ScatterCopy], maximumBytes: Int) -> [SourceSpan] {
        var spans: [SourceSpan] = []
        var current: (shard: String, start: Int, end: Int, indices: [Int])?

        for (index, op) in operations.enumerated() {
            if var open = current,
                open.shard == op.sourceShard,
                open.end == op.sourceOffset,
                open.end - open.start + op.sourceByteCount <= maximumBytes
            {
                open.end += op.sourceByteCount
                open.indices.append(index)
                current = open
                continue
            }
            if let open = current {
                spans.append(
                    SourceSpan(
                        shard: open.shard, range: open.start..<open.end,
                        operationIndices: open.indices))
            }
            current = (op.sourceShard, op.sourceOffset, op.sourceRange.upperBound, [index])
        }
        if let open = current {
            spans.append(
                SourceSpan(
                    shard: open.shard, range: open.start..<open.end, operationIndices: open.indices))
        }
        return spans
    }

    public var totalSourceBytes: Int {
        operations.reduce(0) { $0 + $1.sourceByteCount }
    }

    public var totalDestinationBytes: Int {
        destinationSizes.values.reduce(0, +)
    }

    public enum PlanError: Error, CustomStringConvertible {
        case missingTensor(String)
        case unexpectedShape(String, expected: Int, got: Int)
        case unknownShard(String)

        public var description: String {
            switch self {
            case .missingTensor(let n):
                return "tenseur absent du checkpoint source : \(n)"
            case let .unexpectedShape(n, e, g):
                return "tenseur \(n) : \(g) octets, \(e) attendus — checkpoint incompatible"
            case .unknownShard(let s):
                return "en-tête manquant pour le shard \(s)"
            }
        }
    }

    /// Construit le plan à partir de la carte des poids et des en-têtes de chaque shard.
    public init(
        config: GptOssConfig,
        weightMap: [String: String],
        headers: [String: SafetensorsHeader]
    ) throws {
        let layout = HydraLayout(config: config)
        self.config = config
        self.layout = layout

        var ops: [ScatterCopy] = []

        /// Localise un tenseur source et vérifie sa taille.
        func source(_ name: String, expecting bytes: Int) throws -> (shard: String, offset: Int) {
            guard let shard = weightMap[name] else { throw PlanError.missingTensor(name) }
            guard let header = headers[shard] else { throw PlanError.unknownShard(shard) }
            guard let entry = header.tensors[name], let range = header.fileRange(of: name) else {
                throw PlanError.missingTensor(name)
            }
            guard entry.byteCount == bytes else {
                throw PlanError.unexpectedShape(name, expected: bytes, got: entry.byteCount)
            }
            return (shard, range.lowerBound)
        }

        // --- Tenseurs résidents : copies contiguës vers resident.bin ---
        for placement in layout.resident {
            let s = try source(placement.sourceName, expecting: placement.byteCount)
            ops.append(
                ScatterCopy(
                    sourceTensor: placement.sourceName,
                    sourceShard: s.shard, sourceOffset: s.offset,
                    destination: .resident,
                    destinationOffset: placement.offset,
                    destinationStride: 0,
                    chunkByteCount: placement.byteCount,
                    chunkCount: 1))
        }

        // --- Embedding : fichier dédié, mappé mais hors working set Metal ---
        let embed = try source("model.embed_tokens.weight", expecting: config.embeddingBytes)
        ops.append(
            ScatterCopy(
                sourceTensor: "model.embed_tokens.weight",
                sourceShard: embed.shard, sourceOffset: embed.offset,
                destination: .embedding,
                destinationOffset: 0, destinationStride: 0,
                chunkByteCount: config.embeddingBytes, chunkCount: 1))

        // --- Experts : éparpillement d'un tenseur source vers E blobs ---
        let blob = layout.expertBlob
        let E = config.expertCount
        for layer in 0..<config.layerCount {
            let subTensors: [(suffix: String, slot: ExpertBlobLayout.Slot)] = [
                ("gate_up_proj_blocks", blob.gateUpBlocks),
                ("gate_up_proj_scales", blob.gateUpScales),
                ("gate_up_proj_bias", blob.gateUpBias),
                ("down_proj_blocks", blob.downBlocks),
                ("down_proj_scales", blob.downScales),
                ("down_proj_bias", blob.downBias),
            ]
            for (suffix, slot) in subTensors {
                let name = "model.layers.\(layer).mlp.experts.\(suffix)"
                let s = try source(name, expecting: slot.byteCount * E)
                ops.append(
                    ScatterCopy(
                        sourceTensor: name,
                        sourceShard: s.shard, sourceOffset: s.offset,
                        destination: .expertLayer(layer),
                        destinationOffset: slot.offset,
                        destinationStride: blob.strideBytes,
                        chunkByteCount: slot.byteCount,
                        chunkCount: E))
            }
        }

        ops.sort {
            $0.sourceShard == $1.sourceShard
                ? $0.sourceOffset < $1.sourceOffset
                : $0.sourceShard < $1.sourceShard
        }
        self.operations = ops
        self.spans = Self.makeSpans(operations: ops, maximumBytes: Self.maximumSpanBytes)

        var sizes: [DestinationFile: Int] = [
            .resident: layout.residentBytes,
            .embedding: config.embeddingBytes,
        ]
        for layer in 0..<config.layerCount {
            sizes[.expertLayer(layer)] = layout.expertLayerFileBytes
        }
        self.destinationSizes = sizes
    }

    // MARK: - Vérification

    public struct Problem: Sendable, CustomStringConvertible {
        public let description: String
    }

    /// Vérifie que le plan est cohérent **avant** tout téléchargement.
    ///
    /// Le contrôle décisif est le dernier : la somme des octets sources couverts doit
    /// égaler le `total_size` déclaré par l'index. S'il y a égalité, c'est que le plan
    /// couvre le checkpoint entier, sans trou ni doublon — donc qu'aucun tenseur n'a été
    /// oublié en silence.
    public func validate(declaredSourceTotal: Int?) -> [Problem] {
        var problems: [Problem] = []

        // Aucun tenseur source lu deux fois.
        var seen = Set<String>()
        for op in operations where !seen.insert(op.sourceTensor).inserted {
            problems.append(Problem(description: "tenseur source lu deux fois : \(op.sourceTensor)"))
        }

        // Aucune écriture ne doit sortir de son fichier ni chevaucher une autre.
        var writes: [DestinationFile: [Range<Int>]] = [:]
        for op in operations {
            guard let size = destinationSizes[op.destination] else {
                problems.append(Problem(description: "destination inconnue : \(op.destination.path)"))
                continue
            }
            for i in 0..<op.chunkCount {
                let start = op.destinationOffset(ofChunk: i)
                let range = start..<(start + op.chunkByteCount)
                if range.upperBound > size {
                    problems.append(
                        Problem(description:
                            "\(op.sourceTensor) déborde de \(op.destination.path) "
                            + "(\(range.upperBound) > \(size))"))
                    break
                }
                writes[op.destination, default: []].append(range)
            }
        }
        for (file, ranges) in writes {
            let sorted = ranges.sorted { $0.lowerBound < $1.lowerBound }
            for i in 1..<max(sorted.count, 1) where sorted[i].lowerBound < sorted[i - 1].upperBound {
                problems.append(
                    Problem(description:
                        "chevauchement d'écriture dans \(file.path) : "
                        + "\(sorted[i - 1]) et \(sorted[i])"))
                break
            }
        }

        // Le plan couvre-t-il exactement le checkpoint source ?
        if let declared = declaredSourceTotal, declared != totalSourceBytes {
            problems.append(
                Problem(description:
                    "couverture incomplète : \(totalSourceBytes) octets planifiés, "
                    + "\(declared) déclarés par l'index (écart \(declared - totalSourceBytes))"))
        }

        return problems
    }
}
