import Darwin
import Foundation
import HydraCore
import Metal

/// A bounded cache of routed experts, one set of slots per layer.
///
/// This is the heart of the project. The full pool lives on SSD — 9.47 GiB for the 20B,
/// 56.8 GiB for the 120B — and only a few experts per layer are resident in memory.
///
/// Three decisions, all taken from measurements published by TurboFieldfare:
///
/// - **Explicit `pread` rather than `mmap`.** Demand paging gives no control over when
///   reads happen or how many run at once: 0.50 tok/s against 3.97.
/// - **Preallocated slots, page-aligned, wrapped once and for all.** We never allocate
///   during decoding; a slot is filled by `pread` and then read as-is by the GPU.
/// - **LFU eviction with recency as a tiebreaker**, measured better than LRU
///   (72.6 → 64.8 ms/token).
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
                return "layer \(l) file unreachable: \(String(cString: strerror(e)))"
            case .allocationFailed(let bytes):
                return "cannot make an aligned allocation of \(bytes) bytes"
            case .bufferCreationFailed(let bytes):
                return "no-copy MTLBuffer impossible for a \(bytes)-byte slot"
            case let .readFailed(l, x, e):
                return "reading expert \(x) (layer \(l)): \(String(cString: strerror(e)))"
            case let .shortRead(l, x, expected, got):
                return "expert \(x) (layer \(l)): \(got) bytes read, \(expected) expected"
            case .expertOutOfRange(let x):
                return "identifiant d'expert hors bornes : \(x)"
            }
        }
    }

    /// Disables the system page cache on the expert files.
    ///
    /// With a multi-gigabyte application cache, letting macOS keep a second copy may be
    /// waste — or a useful second-level cache. The question is open and settled by
    /// measurement; this flag exists so it can be asked. It also serves to measure the cost
    /// of a miss honestly: without it, a re-read is not cold.
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

    /// Memory reserved by the cache once every layer is open.
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

    /// Returns the buffer holding the requested expert, reading it from SSD if it is not
    /// already cached. The offset returned is that of the blob within the slot's buffer.
    /// - Parameter pin: locks the slot against eviction until `release(layer:)`.
    ///   **Indispensable as soon as a GPU pass referencing this buffer has been encoded**:
    ///   an eviction between encoding and execution would have the GPU read another
    ///   expert's weights, with no error and no signal. That bug showed up as
    ///   non-determinism under greedy decoding — three identical runs, three different
    ///   outputs.
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

    /// True if the expert is already in memory, triggering nothing and waiting for nothing.
    ///
    /// Used to compute what is ready first while the rest loads. An expert being read counts
    /// as absent: the point is to know what can be worked on right now.
    public func isResident(layer: Int, expert: Int) -> Bool {
        lock.lock()
        let cache = layers[layer]
        lock.unlock()
        return cache?.contains(expert: expert) ?? false
    }

    /// Loads a set of experts for a layer, **missing reads issued in parallel**.
    ///
    /// This is the form decoding uses: the router emits `top_k` identifiers at once, and
    /// nothing requires reading them one after another. The bench on this machine gives
    /// 3.0 GB/s for a single read and 5.3-5.7 GB/s from four upwards — parallelism is the
    /// dominant factor on this access pattern.
    ///
    /// Loaded slots are **pinned** until `release(layer:)`. Without that, two parallel reads
    /// could pick as a victim a slot the other has just filled, and the GPU would then read
    /// the weights of an expert nobody asked for.
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

    /// Releases a layer's slots. To be called **only after** the command buffer referencing
    /// them has completed.
    public func release(layer: Int) {
        lock.lock()
        let cache = layers[layer]
        lock.unlock()
        cache?.unpinAll()
    }

    /// Collects the first error raised in a batch of parallel reads.
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

/// A layer's slots: file descriptor, aligned buffers, occupancy table.
///
/// Synchronization has to hold under parallel reads, which are the normal mode — we fill a
/// layer's four experts simultaneously. Two rules follow, and a naive version breaks both:
///
/// - **a slot being filled is never chosen as a victim**, otherwise two `pread`s write into
///   the same buffer;
/// - **a slot being filled is never handed to the caller**, otherwise the GPU reads
///   half-written bytes.
///
/// An `inFlight` flag and an `NSCondition` cover both cases: whoever asks for an expert
/// already in flight waits for it rather than starting a second read.
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
        /// Pinned as long as an encoded command buffer references it.
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

    /// LFU eviction, recency as a tiebreaker. A free slot is always preferred; a slot being
    /// filled is never eligible.
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

        // If the expert is already present or being read, we wait for it to be ready rather
        // than starting a second read on the same blob.
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

        // Every place may be taken by an in-flight read: we wait for one to free up.
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
            // The read failed: the slot holds nothing usable.
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

    /// Releases all of the layer's usage pins.
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
