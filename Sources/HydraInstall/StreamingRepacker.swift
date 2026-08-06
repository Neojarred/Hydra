import CryptoKit
import Foundation
import HydraCore
import HydraFormat

/// Exécute un `RepackPlan` sans jamais matérialiser plus qu'un bloc réseau en mémoire.
///
/// **L'invariant du projet est ici.** Le checkpoint source pèse 12,8 Gio pour le 20B et
/// 60,8 Gio pour le 120B ; le tas ne doit à aucun moment en contenir une part
/// significative. La mécanique tient en trois règles :
///
/// 1. le plan est découpé en **régions contiguës** du checkpoint source, téléchargées
///    chacune en une seule requête ;
/// 2. la réponse est **consommée au fil de l'eau** — chaque bloc livré par la pile réseau
///    est routé vers sa destination puis relâché avant l'arrivée du suivant ;
/// 3. rien n'est accumulé entre les blocs, hormis un état de hachage de 32 octets par
///    opération en cours.
///
/// Une première version découpait chaque plage en sous-requêtes de 4 Mio, pour que la
/// borne mémoire soit une propriété du découpage plutôt que du comportement de la pile
/// réseau. Mesuré sur le vrai dépôt, ce choix divisait le débit par 6,4. La borne vient
/// désormais de la consommation incrémentale, et elle est vérifiée par les tests plutôt
/// que par construction.
public struct StreamingRepacker: Sendable {

    public let plan: RepackPlan
    public let source: ByteRangeSource

    /// Nombre d'opérations entre deux synchronisations disque et deux points de reprise.
    public let checkpointInterval: Int

    /// Appelé avec le répertoire **partiel**, juste avant l'écriture du manifeste.
    ///
    /// Sert à déposer ce qui n'est pas un poids — le tokeniseur, en pratique. Le faire
    /// avant la promotion garantit qu'une installation promue est toujours complète :
    /// il ne peut pas exister de `.hydra` valide sans son tokeniseur.
    public var auxiliary: (@Sendable (URL) async throws -> Void)?

    public init(
        plan: RepackPlan, source: ByteRangeSource, checkpointInterval: Int = 16,
        auxiliary: (@Sendable (URL) async throws -> Void)? = nil
    ) {
        self.plan = plan
        self.source = source
        self.checkpointInterval = checkpointInterval
        self.auxiliary = auxiliary
    }

    public struct Progress: Sendable {
        public let operationsDone: Int
        public let operationsTotal: Int
        public let bytesDone: Int
        public let bytesTotal: Int
        public let currentTensor: String
        /// Plus gros bloc reçu depuis le début. Sert de **preuve expérimentale** de
        /// l'invariant mémoire, pas seulement d'argument de conception.
        public let peakPayloadBytes: Int
    }

    public enum RepackError: Error, CustomStringConvertible {
        case journalUnreadable(String)
        case incompleteSpan(shard: String, expected: Int, got: Int)

        public var description: String {
            switch self {
            case .journalUnreadable(let m):
                return "journal de reprise illisible : \(m)"
            case let .incompleteSpan(shard, expected, got):
                return "région incomplète sur \(shard) : \(got) octets routés, \(expected) attendus"
            }
        }
    }

    // MARK: - Journal de reprise

    /// Enregistre les opérations dont les données sont **durables**. Une opération n'y
    /// figure qu'après `F_FULLFSYNC` : une reprise ne peut donc pas sauter une écriture
    /// restée dans un cache disque.
    struct Journal: Codable {
        var sourceDescription: String
        var completed: [Int: String]  // index d'opération -> sha256 des octets sources

        static let fileName = "progress.json"

        static func read(from root: URL) throws -> Journal? {
            let url = root.appending(path: fileName)
            guard FileManager.default.fileExists(atPath: url.path) else { return nil }
            do {
                return try JSONDecoder().decode(Journal.self, from: Data(contentsOf: url))
            } catch {
                throw RepackError.journalUnreadable(error.localizedDescription)
            }
        }

        func write(to root: URL) throws {
            let encoder = JSONEncoder()
            encoder.outputFormatting = .sortedKeys
            try encoder.encode(self).write(to: root.appending(path: Self.fileName), options: .atomic)
        }
    }

    // MARK: - Routage d'une région

    /// Distribue les octets d'une région contiguë vers les destinations de ses opérations.
    ///
    /// Les blocs arrivent dans l'ordre mais à des frontières arbitraires : un bloc peut
    /// chevaucher deux opérations, et à l'intérieur d'une opération deux blobs d'experts.
    /// Le routeur ne suppose donc aucun alignement, il ne suit qu'un curseur.
    ///
    /// Sûreté : `consume` est appelé de façon sérialisée par la source (file de délégation
    /// d'`URLSession`, ou boucle de lecture locale). Aucun accès concurrent n'a lieu.
    final class SpanRouter: @unchecked Sendable {
        private let operations: [ScatterCopy]
        private let indices: [Int]
        private let writer: InstallationWriter

        private var position = 0  // index dans `indices`
        private var consumedInOperation = 0
        private var hasher = SHA256()
        private(set) var digests: [Int: String] = [:]
        private(set) var bytesRouted = 0
        private(set) var peakBlock = 0

        init(operations: [ScatterCopy], indices: [Int], writer: InstallationWriter) {
            self.operations = operations
            self.indices = indices
            self.writer = writer
        }

        var currentTensor: String {
            position < indices.count ? operations[indices[position]].sourceTensor : ""
        }

        func consume(_ block: Data) throws {
            peakBlock = max(peakBlock, block.count)
            var offsetInBlock = 0

            while offsetInBlock < block.count {
                guard position < indices.count else { break }
                let op = operations[indices[position]]

                let remainingInOperation = op.sourceByteCount - consumedInOperation
                let take = min(remainingInOperation, block.count - offsetInBlock)
                let piece = block.subdata(
                    in: block.startIndex.advanced(by: offsetInBlock)
                        ..< block.startIndex.advanced(by: offsetInBlock + take))
                hasher.update(data: piece)

                // Éparpillement : le morceau peut recouvrir plusieurs blobs d'experts.
                var written = 0
                while written < take {
                    let absolute = consumedInOperation + written
                    let chunkIndex = absolute / op.chunkByteCount
                    let offsetInChunk = absolute % op.chunkByteCount
                    let slice = min(op.chunkByteCount - offsetInChunk, take - written)

                    try writer.write(
                        piece.subdata(
                            in: piece.startIndex.advanced(by: written)
                                ..< piece.startIndex.advanced(by: written + slice)),
                        to: op.destination,
                        at: op.destinationOffset(ofChunk: chunkIndex) + offsetInChunk)
                    written += slice
                }

                consumedInOperation += take
                offsetInBlock += take
                bytesRouted += take

                if consumedInOperation == op.sourceByteCount {
                    digests[indices[position]] = hasher.finalize().hexString
                    hasher = SHA256()
                    consumedInOperation = 0
                    position += 1
                }
            }
        }
    }

    // MARK: - Exécution

    /// Installe dans `<destination>.partial`, puis promeut atomiquement.
    /// Reprend automatiquement si un répertoire partiel cohérent existe déjà.
    @discardableResult
    public func run(
        destination: URL,
        onProgress: (@Sendable (Progress) -> Void)? = nil
    ) async throws -> HydraManifest {

        let partial = destination.appendingPathExtension("partial")
        let writer = try InstallationWriter(root: partial, sizes: plan.destinationSizes)
        defer { writer.close() }

        var journal = try Journal.read(from: partial)
            ?? Journal(sourceDescription: source.sourceDescription, completed: [:])
        if journal.sourceDescription != source.sourceDescription {
            // La source a changé : on repart de zéro plutôt que de mélanger deux révisions.
            journal = Journal(sourceDescription: source.sourceDescription, completed: [:])
        }

        let totalBytes = plan.totalSourceBytes
        let alreadyDone = plan.operations.enumerated()
            .filter { journal.completed[$0.offset] != nil }
            .reduce(0) { $0 + $1.element.sourceByteCount }

        let counters = Counters(bytesDone: alreadyDone)
        var sinceCheckpoint = 0

        for span in plan.spans {
            try Task.checkCancellation()

            // Les opérations d'une région se terminent dans l'ordre : reprendre à la
            // première inachevée suffit, et raccourcit d'autant la plage à retélécharger.
            guard let resumeAt = span.operationIndices.firstIndex(where: {
                journal.completed[$0] == nil
            }) else { continue }

            let remaining = Array(span.operationIndices[resumeAt...])
            let start = plan.operations[remaining[0]].sourceOffset
            let router = SpanRouter(
                operations: plan.operations, indices: remaining, writer: writer)

            let operationsDoneBefore = journal.completed.count
            try await source.stream(
                file: span.shard, range: start..<span.range.upperBound
            ) { block in
                try router.consume(block)
                let snapshot = counters.advance(by: block.count, peak: router.peakBlock)
                onProgress?(
                    Progress(
                        operationsDone: operationsDoneBefore, operationsTotal: plan.operations.count,
                        bytesDone: snapshot.bytes, bytesTotal: totalBytes,
                        currentTensor: router.currentTensor, peakPayloadBytes: snapshot.peak))
            }

            guard router.bytesRouted == span.range.upperBound - start else {
                throw RepackError.incompleteSpan(
                    shard: span.shard, expected: span.range.upperBound - start,
                    got: router.bytesRouted)
            }

            for (index, digest) in router.digests { journal.completed[index] = digest }
            sinceCheckpoint += router.digests.count
            if sinceCheckpoint >= checkpointInterval {
                try writer.synchronize()
                try journal.write(to: partial)
                sinceCheckpoint = 0
            }
        }

        // Tout doit être durable avant que le manifeste n'affirme que l'installation existe.
        try writer.synchronize()
        try journal.write(to: partial)

        let manifest = HydraManifest(
            model: .init(config: plan.config),
            layout: .init(
                expertStrideBytes: plan.layout.expertBlob.strideBytes,
                expertPayloadBytes: plan.layout.expertBlob.payloadBytes,
                residentBytes: plan.layout.residentBytes,
                embeddingBytes: plan.config.embeddingBytes,
                tensorAlignment: ExpertBlobLayout.tensorAlignment,
                pageAlignment: ExpertBlobLayout.pageAlignment),
            files: Dictionary(
                uniqueKeysWithValues: plan.destinationSizes.map {
                    ($0.key.path, HydraManifest.FileEntry(byteCount: $0.value))
                }),
            sourceDescription: source.sourceDescription,
            sourceTotalBytes: totalBytes,
            tensors: plan.operations.enumerated().map { index, op in
                HydraManifest.TensorDigest(
                    tensor: op.sourceTensor, byteCount: op.sourceByteCount,
                    sha256: journal.completed[index] ?? "")
            })

        // Le tokeniseur et les métadonnées sont déposés avant le manifeste : une
        // installation promue est complète, ou n'existe pas.
        if let auxiliary { try await auxiliary(partial) }

        try manifest.write(to: partial)
        try? FileManager.default.removeItem(at: partial.appending(path: Journal.fileName))
        try writer.promote(to: destination)
        return manifest
    }

    /// Compteurs partagés avec le rappel de progression, qui est `@Sendable`.
    final class Counters: @unchecked Sendable {
        private var bytes: Int
        private var peak = 0
        private let lock = NSLock()

        init(bytesDone: Int) { bytes = bytesDone }

        func advance(by count: Int, peak block: Int) -> (bytes: Int, peak: Int) {
            lock.lock()
            defer { lock.unlock() }
            bytes += count
            peak = max(peak, block)
            return (bytes, peak)
        }
    }
}

extension SHA256.Digest {
    var hexString: String { map { String(format: "%02x", $0) }.joined() }
}
