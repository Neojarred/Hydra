import Foundation
import HydraCore
import HydraFormat

/// Vérifie qu'une installation `.hydra` contient bien les octets du checkpoint amont.
///
/// La vérification structurelle du manifeste — format, architecture, tailles de fichiers —
/// est peu coûteuse et faite à chaque chargement. Elle ne prouve pourtant rien sur le
/// **contenu** : un repacker qui écrirait tout à un octet de décalage produirait des
/// fichiers de la bonne taille et un manifeste valide.
///
/// Ce vérificateur ferme cette brèche par échantillonnage : il tire des fenêtres au hasard
/// dans les blobs d'experts et les tenseurs résidents, redemande les octets **sources**
/// correspondants, et compare. Une erreur systématique de placement — décalage, inversion
/// de sous-tenseurs, mauvais pas entre experts — est détectée dès le premier échantillon.
public struct InstallationVerifier: Sendable {

    public let plan: RepackPlan
    public let source: ByteRangeSource
    public let root: URL

    /// Taille d'une fenêtre comparée. Assez grande pour être significative, assez petite
    /// pour que la vérification reste bornée en mémoire comme le reste du projet.
    public static let windowBytes = 64 * 1024

    public init(plan: RepackPlan, source: ByteRangeSource, root: URL) {
        self.plan = plan
        self.source = source
        self.root = root
    }

    public struct Mismatch: Sendable, CustomStringConvertible {
        public let tensor: String
        public let chunkIndex: Int
        public let offsetInChunk: Int
        public let firstDifferingByte: Int

        public var description: String {
            "\(tensor) [morceau \(chunkIndex), offset \(offsetInChunk)] "
                + "diverge à l'octet \(firstDifferingByte)"
        }
    }

    public struct Report: Sendable {
        public let samplesChecked: Int
        public let bytesCompared: Int
        public let mismatches: [Mismatch]
        public var passed: Bool { mismatches.isEmpty }
    }

    public enum VerifyError: Error, CustomStringConvertible {
        case destinationUnreadable(String)

        public var description: String {
            switch self {
            case .destinationUnreadable(let f):
                return "fichier installé illisible : \(f)"
            }
        }
    }

    /// Compare `sampleCount` fenêtres tirées au hasard. Le tirage est déterministe pour
    /// que l'échec d'une vérification soit reproductible à l'identique.
    public func spotCheck(sampleCount: Int = 24, seed: UInt64 = 0x5EED) async throws -> Report {
        var rng = SplitMix64(seed: seed)
        var mismatches: [Mismatch] = []
        var bytesCompared = 0

        // Un descripteur par fichier destination, ouvert à la demande.
        var handles: [DestinationFile: FileHandle] = [:]
        defer { for (_, h) in handles { try? h.close() } }

        for _ in 0..<sampleCount {
            let op = plan.operations[Int(rng.next() % UInt64(plan.operations.count))]
            let chunkIndex = Int(rng.next() % UInt64(op.chunkCount))
            let maximumOffset = max(1, op.chunkByteCount - Self.windowBytes)
            let offsetInChunk = Int(rng.next() % UInt64(maximumOffset))
            let length = min(Self.windowBytes, op.chunkByteCount - offsetInChunk)

            // Octets tels qu'ils sont dans le checkpoint amont.
            let sourceStart = op.sourceOffset + chunkIndex * op.chunkByteCount + offsetInChunk
            let expected = try await source.read(
                file: op.sourceShard, range: sourceStart..<(sourceStart + length))

            // Octets tels qu'ils ont été installés.
            let handle: FileHandle
            if let existing = handles[op.destination] {
                handle = existing
            } else {
                guard let opened = FileHandle(
                    forReadingAtPath: root.appending(path: op.destination.path).path)
                else {
                    throw VerifyError.destinationUnreadable(op.destination.path)
                }
                handles[op.destination] = opened
                handle = opened
            }
            let destinationStart = op.destinationOffset(ofChunk: chunkIndex) + offsetInChunk
            try handle.seek(toOffset: UInt64(destinationStart))
            let actual = try handle.read(upToCount: length) ?? Data()

            bytesCompared += length
            if actual != expected {
                let firstDifference = zip(expected, actual).enumerated()
                    .first { $0.element.0 != $0.element.1 }?.offset ?? min(expected.count, actual.count)
                mismatches.append(
                    Mismatch(
                        tensor: op.sourceTensor, chunkIndex: chunkIndex,
                        offsetInChunk: offsetInChunk, firstDifferingByte: firstDifference))
            }
        }

        return Report(
            samplesChecked: sampleCount, bytesCompared: bytesCompared, mismatches: mismatches)
    }
}

/// Générateur déterministe : un échec de vérification doit être reproductible tel quel.
struct SplitMix64 {
    private var state: UInt64

    init(seed: UInt64) { state = seed }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
