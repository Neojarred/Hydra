import Darwin
import Foundation
import HydraCore
import Metal

/// Cache borné d'experts routés, un jeu de slots par couche.
///
/// C'est le cœur du projet. Le pool complet vit sur le SSD — 9,47 Gio pour le 20B,
/// 56,8 Gio pour le 120B — et seuls quelques experts par couche résident en mémoire.
///
/// Trois décisions, toutes reprises de mesures publiées par TurboFieldfare :
///
/// - **`pread` explicite plutôt que `mmap`.** La pagination à la demande ne donne aucun
///   contrôle sur le moment ni la concurrence des lectures : 0,50 tok/s contre 3,97.
/// - **Slots préalloués, alignés sur la page, enveloppés une fois pour toutes.** On
///   n'alloue jamais pendant le décodage ; un slot est rempli par `pread` puis relu tel
///   quel par le GPU.
/// - **Éviction LFU avec la récence en départage**, mesurée meilleure que LRU
///   (72,6 → 64,8 ms/token).
public final class ExpertSlotCache: @unchecked Sendable {

    public let config: GptOssConfig
    public let slotsPerLayer: Int
    public let slotBytes: Int

    private let root: URL
    private let device: MTLDevice
    private var layers: [LayerCache?]
    private let lock = NSLock()

    public struct Statistics: Sendable, Equatable {
        public var hits = 0
        public var misses = 0
        public var bytesRead = 0

        public var hitRate: Double {
            let total = hits + misses
            return total == 0 ? 0 : Double(hits) / Double(total)
        }
    }

    private var statistics = Statistics()

    public enum CacheError: Error, CustomStringConvertible {
        case layerFileMissing(Int, errno: Int32)
        case allocationFailed(bytes: Int)
        case bufferCreationFailed(bytes: Int)
        case readFailed(layer: Int, expert: Int, errno: Int32)
        case shortRead(layer: Int, expert: Int, expected: Int, got: Int)
        case expertOutOfRange(Int)

        public var description: String {
            switch self {
            case let .layerFileMissing(l, e):
                return "fichier de la couche \(l) inaccessible : \(String(cString: strerror(e)))"
            case .allocationFailed(let bytes):
                return "allocation alignée impossible de \(bytes) octets"
            case .bufferCreationFailed(let bytes):
                return "MTLBuffer sans copie impossible pour un slot de \(bytes) octets"
            case let .readFailed(l, x, e):
                return "lecture de l'expert \(x) (couche \(l)) : \(String(cString: strerror(e)))"
            case let .shortRead(l, x, expected, got):
                return "expert \(x) (couche \(l)) : \(got) octets lus, \(expected) attendus"
            case .expertOutOfRange(let x):
                return "identifiant d'expert hors bornes : \(x)"
            }
        }
    }

    /// Désactive le cache de pages du système sur les fichiers d'experts.
    ///
    /// Avec un cache applicatif de plusieurs gigaoctets, laisser macOS en conserver une
    /// seconde copie peut être un gaspillage — ou un cache de second niveau utile. La
    /// question est ouverte et se tranche par la mesure ; ce drapeau existe pour pouvoir
    /// la poser. Il sert aussi à mesurer honnêtement le coût d'un miss : sans lui, une
    /// relecture n'est pas froide.
    public let bypassPageCache: Bool

    public init(
        root: URL, config: GptOssConfig, slotsPerLayer: Int, device: MTLDevice,
        bypassPageCache: Bool = false
    ) {
        self.bypassPageCache = bypassPageCache
        self.root = root
        self.config = config
        self.slotsPerLayer = min(slotsPerLayer, config.expertCount)
        self.slotBytes = config.expertSlotBytes
        self.device = device
        self.layers = Array(repeating: nil, count: config.layerCount)
    }

    /// Mémoire réservée par le cache une fois toutes les couches ouvertes.
    public var reservedBytes: Int { config.layerCount * slotsPerLayer * slotBytes }

    public func statisticsSnapshot() -> Statistics {
        lock.lock()
        defer { lock.unlock() }
        return statistics
    }

    public func resetStatistics() {
        lock.lock()
        statistics = Statistics()
        lock.unlock()
    }

    /// Rend le buffer contenant l'expert demandé, en le lisant depuis le SSD s'il n'est
    /// pas déjà en cache. Le décalage rendu est celui du blob dans le buffer du slot.
    /// - Parameter pin: verrouille le slot contre l'éviction jusqu'à `release(layer:)`.
    ///   **Indispensable dès qu'une passe GPU référençant ce tampon est encodée** : une
    ///   éviction entre l'encodage et l'exécution ferait lire au GPU les poids d'un autre
    ///   expert, sans erreur ni signal. Ce bug s'est manifesté comme du non-déterminisme
    ///   en décodage glouton — trois exécutions identiques, trois sorties différentes.
    @discardableResult
    public func expert(
        layer: Int, expert: Int, pin: Bool = false
    ) throws -> (buffer: MTLBuffer, offset: Int) {
        guard expert >= 0, expert < config.expertCount else {
            throw CacheError.expertOutOfRange(expert)
        }

        lock.lock()
        let cache: LayerCache
        if let existing = layers[layer] {
            cache = existing
        } else {
            lock.unlock()
            let opened = try LayerCache(
                root: root, layer: layer, slotCount: slotsPerLayer,
                slotBytes: slotBytes, device: device, bypassPageCache: bypassPageCache)
            lock.lock()
            if let raced = layers[layer] {
                cache = raced
            } else {
                layers[layer] = opened
                cache = opened
            }
        }
        lock.unlock()

        let outcome = try cache.fetch(expert: expert, pin: pin)
        lock.lock()
        if outcome.wasHit {
            statistics.hits += 1
        } else {
            statistics.misses += 1
            statistics.bytesRead += slotBytes
        }
        lock.unlock()
        return (outcome.buffer, 0)
    }

    /// Vrai si l'expert est déjà en mémoire, sans rien déclencher ni attendre.
    ///
    /// Sert à calculer d'abord ce qui est prêt pendant que le reste se charge. Un expert
    /// en cours de lecture compte comme absent : le but est de savoir sur quoi on peut
    /// travailler tout de suite.
    public func isResident(layer: Int, expert: Int) -> Bool {
        lock.lock()
        let cache = layers[layer]
        lock.unlock()
        return cache?.contains(expert: expert) ?? false
    }

    /// Charge un jeu d'experts pour une couche, **lectures manquantes en parallèle**.
    ///
    /// C'est la forme qu'utilise le décodage : le routeur sort `top_k` identifiants d'un
    /// coup, et rien n'oblige à les lire l'un après l'autre. Le banc sur cette machine
    /// donne 3,0 Go/s à une lecture et 5,3-5,7 Go/s à partir de quatre — le parallélisme
    /// est le facteur dominant sur ce motif d'accès.
    /// Les slots chargés sont **verrouillés** jusqu'à `release(layer:)`. Sans cela, deux
    /// lectures parallèles peuvent choisir comme victime un slot que l'autre vient de
    /// remplir, et le GPU lirait ensuite les poids d'un expert qu'on n'a pas demandé.
    public func load(layer: Int, experts: [Int]) throws {
        guard experts.count > 1 else {
            if let single = experts.first {
                _ = try expert(layer: layer, expert: single, pin: true)
            }
            return
        }

        let failure = FailureBox()
        DispatchQueue.concurrentPerform(iterations: experts.count) { index in
            do {
                _ = try self.expert(layer: layer, expert: experts[index], pin: true)
            } catch {
                failure.record(error)
            }
        }
        if let error = failure.value { throw error }
    }

    /// Libère les slots d'une couche. À n'appeler **qu'après** que le tampon de commandes
    /// les référençant soit terminé.
    public func release(layer: Int) {
        lock.lock()
        let cache = layers[layer]
        lock.unlock()
        cache?.unpinAll()
    }

    /// Recueille la première erreur survenue dans un lot de lectures parallèles.
    private final class FailureBox: @unchecked Sendable {
        private var storage: Error?
        private let lock = NSLock()

        func record(_ error: Error) {
            lock.lock()
            if storage == nil { storage = error }
            lock.unlock()
        }

        var value: Error? {
            lock.lock()
            defer { lock.unlock() }
            return storage
        }
    }

    public func closeAll() {
        lock.lock()
        layers = Array(repeating: nil, count: config.layerCount)
        lock.unlock()
    }
}

/// Slots d'une couche : descripteur de fichier, tampons alignés, table d'occupation.
///
/// La synchronisation doit tenir sous lectures parallèles, qui sont le mode normal — on
/// remplit les quatre experts d'une couche simultanément. Deux règles s'imposent, et une
/// version naïve les enfreint toutes les deux :
///
/// - **un slot en cours de remplissage n'est jamais choisi comme victime**, sinon deux
///   `pread` écrivent dans le même tampon ;
/// - **un slot en cours de remplissage n'est jamais rendu à l'appelant**, sinon le GPU lit
///   des octets à moitié écrits.
///
/// Un drapeau `inFlight` et une `NSCondition` couvrent les deux cas : le demandeur d'un
/// expert déjà en vol attend sa fin plutôt que de relancer une lecture.
final class LayerCache: @unchecked Sendable {

    private let descriptor: Int32
    private let layer: Int
    private let slotBytes: Int
    private var slots: [Slot]
    private let condition = NSCondition()
    private var clock = 0

    struct Slot {
        var expert: Int = -1
        var frequency: Int = 0
        var lastUsed: Int = 0
        var inFlight: Bool = false
        /// Verrouillé tant qu'un tampon de commandes encodé le référence.
        var pinned: Bool = false
        let memory: UnsafeMutableRawPointer
        let buffer: MTLBuffer
    }

    init(
        root: URL, layer: Int, slotCount: Int, slotBytes: Int, device: MTLDevice,
        bypassPageCache: Bool
    ) throws {
        let path = root.appending(path: String(format: "experts/layer_%02d.bin", layer)).path
        let descriptor = open(path, O_RDONLY)
        guard descriptor >= 0 else {
            throw ExpertSlotCache.CacheError.layerFileMissing(layer, errno: errno)
        }
        if bypassPageCache { fcntl(descriptor, F_NOCACHE, 1) }
        self.descriptor = descriptor
        self.layer = layer
        self.slotBytes = slotBytes

        var built: [Slot] = []
        built.reserveCapacity(slotCount)
        let alignment = MappedFile.pageSize
        for _ in 0..<slotCount {
            var pointer: UnsafeMutableRawPointer?
            guard posix_memalign(&pointer, alignment, slotBytes) == 0, let memory = pointer else {
                throw ExpertSlotCache.CacheError.allocationFailed(bytes: slotBytes)
            }
            guard let buffer = device.makeBuffer(
                bytesNoCopy: memory, length: slotBytes, options: .storageModeShared,
                deallocator: nil)
            else {
                free(memory)
                throw ExpertSlotCache.CacheError.bufferCreationFailed(bytes: slotBytes)
            }
            built.append(Slot(memory: memory, buffer: buffer))
        }
        self.slots = built
    }

    /// Éviction LFU, récence en départage. Un slot libre est toujours préféré ; un slot
    /// en cours de remplissage n'est jamais éligible.
    private func victimIndex() -> Int? {
        var best: Int?
        for (index, slot) in slots.enumerated() where !slot.inFlight && !slot.pinned {
            if slot.expert < 0 { return index }
            guard let current = best else {
                best = index
                continue
            }
            let candidate = slots[current]
            if slot.frequency < candidate.frequency
                || (slot.frequency == candidate.frequency && slot.lastUsed < candidate.lastUsed)
            {
                best = index
            }
        }
        return best
    }

    func contains(expert: Int) -> Bool {
        condition.lock()
        defer { condition.unlock() }
        return slots.contains { $0.expert == expert && !$0.inFlight }
    }

    func fetch(expert: Int, pin: Bool) throws -> (buffer: MTLBuffer, wasHit: Bool) {
        condition.lock()

        // Si l'expert est déjà présent ou en cours de lecture, on attend qu'il soit prêt
        // plutôt que de lancer une seconde lecture sur le même blob.
        while let index = slots.firstIndex(where: { $0.expert == expert }) {
            if slots[index].inFlight {
                condition.wait()
                continue
            }
            clock += 1
            slots[index].frequency += 1
            slots[index].lastUsed = clock
            if pin { slots[index].pinned = true }
            let buffer = slots[index].buffer
            condition.unlock()
            return (buffer, true)
        }

        // Toutes les places peuvent être occupées par des lectures en vol : on attend
        // qu'une se libère.
        var index: Int
        while true {
            if let candidate = victimIndex() {
                index = candidate
                break
            }
            condition.wait()
        }

        clock += 1
        slots[index].expert = expert
        slots[index].frequency = 1
        slots[index].lastUsed = clock
        slots[index].inFlight = true
        if pin { slots[index].pinned = true }
        let memory = slots[index].memory
        let buffer = slots[index].buffer
        condition.unlock()

        do {
            try readBlob(into: memory, expert: expert)
        } catch {
            condition.lock()
            // La lecture a échoué : le slot ne contient rien d'exploitable.
            slots[index].expert = -1
            slots[index].inFlight = false
            slots[index].pinned = false
            condition.broadcast()
            condition.unlock()
            throw error
        }

        condition.lock()
        slots[index].inFlight = false
        condition.broadcast()
        condition.unlock()
        return (buffer, false)
    }

    private func readBlob(into memory: UnsafeMutableRawPointer, expert: Int) throws {
        let offset = off_t(expert * slotBytes)
        var read = 0
        while read < slotBytes {
            let n = pread(
                descriptor, memory.advanced(by: read), slotBytes - read, offset + off_t(read))
            if n < 0 {
                throw ExpertSlotCache.CacheError.readFailed(
                    layer: layer, expert: expert, errno: errno)
            }
            if n == 0 { break }
            read += n
        }
        guard read == slotBytes else {
            throw ExpertSlotCache.CacheError.shortRead(
                layer: layer, expert: expert, expected: slotBytes, got: read)
        }
    }

    /// Libère tous les verrous d'usage de la couche.
    func unpinAll() {
        condition.lock()
        for index in slots.indices { slots[index].pinned = false }
        condition.broadcast()
        condition.unlock()
    }

    deinit {
        for slot in slots { free(slot.memory) }
        close(descriptor)
    }
}
