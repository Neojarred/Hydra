import Darwin
import Foundation
import HydraCore
import HydraFormat
import Metal

/// Fichier `.hydra` mappé en lecture seule et exposé à Metal **sans copie**.
///
/// `mmap` donne un pointeur aligné sur la page ; `makeBuffer(bytesNoCopy:)` l'enveloppe
/// dans un `MTLBuffer` qui pointe sur ces mêmes pages. Aucun octet ne transite par le tas
/// Swift, et le noyau reste libre de recycler les pages non touchées.
///
/// C'est la raison pour laquelle le format aligne ses fichiers sur 16 Kio : `bytesNoCopy`
/// exige une adresse **et** une longueur multiples de la taille de page. Un fichier mal
/// aligné obligerait à copier.
public final class MappedFile: @unchecked Sendable {

    public let url: URL
    public let byteCount: Int
    public let buffer: MTLBuffer

    private let base: UnsafeMutableRawPointer
    private let mappedLength: Int

    public enum MappingError: Error, CustomStringConvertible {
        case openFailed(String, errno: Int32)
        case statFailed(String, errno: Int32)
        case mapFailed(String, errno: Int32)
        case notPageAligned(String, byteCount: Int, pageSize: Int)
        case bufferCreationFailed(String, bytes: Int)
        case tooLargeForSingleBuffer(String, bytes: Int, limit: Int)

        public var description: String {
            switch self {
            case let .openFailed(f, e):
                return "ouverture impossible de \(f) : \(String(cString: strerror(e)))"
            case let .statFailed(f, e):
                return "stat impossible sur \(f) : \(String(cString: strerror(e)))"
            case let .mapFailed(f, e):
                return "mmap impossible sur \(f) : \(String(cString: strerror(e)))"
            case let .notPageAligned(f, bytes, page):
                return "\(f) fait \(bytes) octets, pas un multiple de la page (\(page)) — "
                    + "l'enveloppe Metal sans copie est impossible"
            case let .bufferCreationFailed(f, bytes):
                return "MTLBuffer sans copie impossible pour \(f) (\(bytes) octets)"
            case let .tooLargeForSingleBuffer(f, bytes, limit):
                return "\(f) fait \(bytes) octets, au-delà de maxBufferLength (\(limit))"
            }
        }
    }

    public static var pageSize: Int { Int(sysconf(_SC_PAGESIZE)) }

    public init(url: URL, device: MTLDevice) throws {
        self.url = url
        let name = url.lastPathComponent

        let descriptor = open(url.path, O_RDONLY)
        guard descriptor >= 0 else { throw MappingError.openFailed(name, errno: errno) }
        defer { close(descriptor) }  // le mappage survit à la fermeture du descripteur

        var info = stat()
        guard fstat(descriptor, &info) == 0 else {
            throw MappingError.statFailed(name, errno: errno)
        }
        let size = Int(info.st_size)
        let page = Self.pageSize

        guard size % page == 0 else {
            throw MappingError.notPageAligned(name, byteCount: size, pageSize: page)
        }
        guard size <= device.maxBufferLength else {
            throw MappingError.tooLargeForSingleBuffer(
                name, bytes: size, limit: device.maxBufferLength)
        }

        guard let pointer = mmap(nil, size, PROT_READ, MAP_PRIVATE, descriptor, 0),
            pointer != MAP_FAILED
        else {
            throw MappingError.mapFailed(name, errno: errno)
        }

        // Les poids sont parcourus dans l'ordre du fichier au chargement, puis relus par
        // le GPU ; l'indication séquentielle aide le noyau à anticiper sans jamais forcer
        // la résidence.
        madvise(pointer, size, MADV_SEQUENTIAL)

        guard let buffer = device.makeBuffer(
            bytesNoCopy: pointer, length: size, options: .storageModeShared, deallocator: nil)
        else {
            munmap(pointer, size)
            throw MappingError.bufferCreationFailed(name, bytes: size)
        }

        self.base = pointer
        self.mappedLength = size
        self.byteCount = size
        self.buffer = buffer
    }

    /// Force la résidence des pages, séquentiellement.
    ///
    /// Sans cela, la première passe du modèle les fait entrer une par une, à la demande :
    /// mesuré, cela ajoutait **1,7 s** au premier prefill du 20B — bien plus que le calcul
    /// lui-même. Une lecture séquentielle laisse le noyau grouper les lectures.
    ///
    /// Ce n'est pas contraire à l'invariant du projet : ces pages sont adossées à un
    /// fichier et restent reprenables sous pression mémoire. On choisit *quand* les faire
    /// entrer, pas *si* elles entrent.
    @discardableResult
    public func prefault() -> Int {
        madvise(base, mappedLength, MADV_WILLNEED)
        // `madvise` n'est qu'une indication : on touche réellement une valeur par page
        // pour garantir la résidence. Le compilateur ne peut pas éliminer l'accumulation.
        var checksum = 0
        let page = Self.pageSize
        let pointer = base.assumingMemoryBound(to: UInt8.self)
        for offset in stride(from: 0, to: byteCount, by: page) {
            checksum &+= Int(pointer[offset])
        }
        return checksum
    }

    /// Accès en lecture directe aux octets mappés, sans copie. Utilisé par les chemins
    /// CPU de validation.
    public func withBytes<R>(_ body: (UnsafeRawBufferPointer) throws -> R) rethrows -> R {
        try body(UnsafeRawBufferPointer(start: base, count: byteCount))
    }

    deinit { munmap(base, mappedLength) }
}

/// Une installation `.hydra` ouverte et prête à alimenter les noyaux.
///
/// Deux fichiers seulement sont mappés en permanence :
///
/// - `resident.bin` — attention, routeurs, normes, sinks, tête LM. Lu à chaque token,
///   donc destiné à rester chaud.
/// - `embed.bin` — l'embedding. Mappé mais **volontairement hors du working set** : on
///   n'en lit qu'une ligne par token, il n'y a aucune raison d'en câbler 1,08 Gio.
///
/// Les experts ne sont pas mappés : ils passent par le cache de slots, avec des `pread`
/// bornés. TurboFieldfare a mesuré l'écart entre les deux approches — 0,50 tok/s en
/// `mmap` contre 3,97 en `pread` parallèle — parce que la pagination à la demande ne
/// laisse aucun contrôle sur le moment et la concurrence des lectures.
public final class ModelMapping: @unchecked Sendable {

    public let root: URL
    public let config: GptOssConfig
    public let manifest: HydraManifest
    public let layout: HydraLayout
    public let resident: MappedFile
    public let embedding: MappedFile

    public enum LoadError: Error, CustomStringConvertible {
        case tensorMissing(String)

        public var description: String {
            switch self {
            case .tensorMissing(let name):
                return "tenseur absent de la disposition résidente : \(name)"
            }
        }
    }

    public init(root: URL, config: GptOssConfig, device: MTLDevice) throws {
        self.root = root
        self.config = config
        self.manifest = try HydraManifest.read(from: root)
        try manifest.validate(against: config, root: root)
        self.layout = HydraLayout(config: config)
        self.resident = try MappedFile(url: root.appending(path: "resident.bin"), device: device)
        self.embedding = try MappedFile(url: root.appending(path: "embed.bin"), device: device)
    }

    /// Emplacement d'un tenseur résident dans le buffer unique `resident.bin`.
    ///
    /// Les noyaux lient une sous-plage par `setBuffer(offset:)` plutôt que de créer un
    /// buffer par tenseur. Le décalage est aligné sur 256 octets par construction, ce qui
    /// autorise les chargements vectoriels larges dans les shaders.
    public func residentTensor(_ name: String) throws -> (buffer: MTLBuffer, offset: Int, byteCount: Int) {
        guard let placement = layout.placement(of: name) else {
            throw LoadError.tensorMissing(name)
        }
        return (resident.buffer, placement.offset, placement.byteCount)
    }

    /// Lit une ligne d'embedding sans matérialiser la table. Le tampon de sortie est
    /// fourni par l'appelant et réutilisé d'un token à l'autre.
    public func readEmbedding(token: Int, into destination: UnsafeMutableBufferPointer<Float>) {
        precondition(destination.count == config.hiddenSize)
        let rowBytes = config.hiddenSize * 2
        let offset = token * rowBytes
        embedding.withBytes { raw in
            for i in 0..<config.hiddenSize {
                let bits = raw.loadUnaligned(fromByteOffset: offset + i * 2, as: UInt16.self)
                destination[i] = BF16.toFloat(UInt16(littleEndian: bits))
            }
        }
    }

    /// Fait entrer les poids résidents en mémoire par une lecture séquentielle.
    ///
    /// **L'embedding en est volontairement exclu.** On n'en lit qu'une ligne par token :
    /// le précharger reviendrait à câbler 1,08 Gio pour rien, à rebours de tout le projet.
    @discardableResult
    public func prefault() -> Int {
        resident.prefault()
    }

    /// Empreinte mémoire des mappages telle que le système la comptabilise.
    /// Seules les pages effectivement touchées comptent : un fichier mappé n'est pas un
    /// fichier chargé.
    public var mappedByteCount: Int { resident.byteCount + embedding.byteCount }
}
