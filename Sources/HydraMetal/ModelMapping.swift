import Darwin
import Foundation
import HydraCore
import HydraFormat
import Metal

/// A `.hydra` file mapped read-only and exposed to Metal **with no copy**.
///
/// `mmap` gives a page-aligned pointer; `makeBuffer(bytesNoCopy:)` wraps it in an
/// `MTLBuffer` pointing at those same pages. No byte passes through the Swift heap, and the
/// kernel remains free to reclaim untouched pages.
///
/// This is why the format aligns its files on 16 KiB: `bytesNoCopy` requires an address
/// **and** a length that are multiples of the page size. A misaligned file would force a
/// copy.
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
                return "cannot open \(f): \(String(cString: strerror(e)))"
            case let .statFailed(f, e):
                return "cannot stat \(f): \(String(cString: strerror(e)))"
            case let .mapFailed(f, e):
                return "cannot mmap \(f): \(String(cString: strerror(e)))"
            case let .notPageAligned(f, bytes, page):
                return "\(f) is \(bytes) bytes, not a multiple of the page size (\(page)) — "
                    + "the no-copy Metal wrapper is impossible"
            case let .bufferCreationFailed(f, bytes):
                return "no-copy MTLBuffer impossible for \(f) (\(bytes) bytes)"
            case let .tooLargeForSingleBuffer(f, bytes, limit):
                return "\(f) is \(bytes) bytes, beyond maxBufferLength (\(limit))"
            }
        }
    }

    public static var pageSize: Int { Int(sysconf(_SC_PAGESIZE)) }

    /// - Parameter writable: maps the pages copy-on-write so the loaded weights can be
    ///   altered **in memory only** — the file on disk is never touched, `MAP_PRIVATE`
    ///   guarantees it. Used by the Q8 simulation of D-020.
    ///
    ///   It is not free: every page written stops being file-backed and becomes anonymous
    ///   memory, which the kernel can no longer reclaim. That breaks D-012's footprint
    ///   invariant, and is why this is confined to the measurement path and never enabled
    ///   for ordinary loading.
    public init(url: URL, device: MTLDevice, writable: Bool = false) throws {
        self.url = url
        let name = url.lastPathComponent

        let descriptor = open(url.path, O_RDONLY)
        guard descriptor >= 0 else { throw MappingError.openFailed(name, errno: errno) }
        defer { close(descriptor) }  // the mapping outlives the descriptor being closed

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

        let protection = writable ? PROT_READ | PROT_WRITE : PROT_READ
        guard let pointer = mmap(nil, size, protection, MAP_PRIVATE, descriptor, 0),
            pointer != MAP_FAILED
        else {
            throw MappingError.mapFailed(name, errno: errno)
        }

        // Weights are walked in file order at load time, then re-read by the GPU; the
        // sequential hint helps the kernel read ahead without ever forcing residency.
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

    /// Forces page residency, sequentially.
    ///
    /// Without this, the model's first pass brings them in one at a time, on demand:
    /// measured, that added **1.7 s** to the 20B's first prefill — far more than the compute
    /// itself. A sequential read lets the kernel batch the reads.
    ///
    /// This does not contradict the project's invariant: these pages are file-backed and
    /// remain reclaimable under memory pressure. We choose *when* they come in, not
    /// *whether* they do.
    @discardableResult
    public func prefault() -> Int {
        madvise(base, mappedLength, MADV_WILLNEED)
        // `madvise` is only a hint: we actually touch one value per page to guarantee
        // residency. The compiler cannot eliminate the accumulation.
        var checksum = 0
        let page = Self.pageSize
        let pointer = base.assumingMemoryBound(to: UInt8.self)
        for offset in stride(from: 0, to: byteCount, by: page) {
            checksum &+= Int(pointer[offset])
        }
        return checksum
    }

    /// Direct read access to the mapped bytes, with no copy. Used by the CPU validation
    /// paths.
    public func withBytes<R>(_ body: (UnsafeRawBufferPointer) throws -> R) rethrows -> R {
        try body(UnsafeRawBufferPointer(start: base, count: byteCount))
    }

    /// The BF16 words of one byte range, mutable. Only meaningful on a `writable` mapping;
    /// on a read-only one the write faults, which is the intended outcome — a silent
    /// failure here would produce a measurement of nothing at all.
    public func withMutableWords<R>(
        offset: Int, byteCount count: Int,
        _ body: (UnsafeMutableBufferPointer<UInt16>) throws -> R
    ) rethrows -> R {
        precondition(offset >= 0 && offset + count <= byteCount)
        precondition(offset % 2 == 0 && count % 2 == 0)
        let start = base.advanced(by: offset).assumingMemoryBound(to: UInt16.self)
        return try body(UnsafeMutableBufferPointer(start: start, count: count / 2))
    }

    deinit { munmap(base, mappedLength) }
}

/// A `.hydra` installation, opened and ready to feed the kernels.
///
/// Only two files are mapped permanently:
///
/// - `resident.bin` — attention, routers, norms, sinks, LM head. Read on every token, so
///   meant to stay warm.
/// - `embed.bin` — the embedding table. Mapped but **deliberately outside the working
///   set**: we read only one row per token, there is no reason to wire down 1.08 GiB.
///
/// Experts are not mapped: they go through the slot cache, with bounded `pread`s.
/// TurboFieldfare measured the gap between the two approaches — 0.50 tok/s with `mmap`
/// against 3.97 with parallel `pread` — because demand paging gives no control over when
/// reads happen or how many run at once.
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
                return "tensor missing from the resident layout: \(name)"
            }
        }
    }

    public init(
        root: URL, config: GptOssConfig, device: MTLDevice, mutableResident: Bool = false
    ) throws {
        self.root = root
        self.config = config
        self.manifest = try HydraManifest.read(from: root)
        try manifest.validate(against: config, root: root)
        // The layout follows what the installation was written with, not a default. Getting
        // this backwards would read quantized bytes as BF16 floats: no error, plausible
        // output, entirely wrong.
        self.layout = HydraLayout(config: config, precision: manifest.precisionPolicy)
        self.resident = try MappedFile(
            url: root.appending(path: "resident.bin"), device: device,
            writable: mutableResident)
        self.embedding = try MappedFile(url: root.appending(path: "embed.bin"), device: device)
    }

    /// Report of a Q8 simulation over the resident weights.
    public struct Q8Simulation: Sendable {
        /// Tensors actually quantized, and the bytes they cover.
        public let tensorsAffected: Int
        public let bytesAffected: Int
        /// Bytes those tensors would occupy in real Q8, at 8.5 bits per weight.
        public let bytesIfQuantized: Int
        /// Largest relative deviation introduced on a single weight.
        public let worstRelativeDeviation: Float

        public var savedFraction: Double {
            bytesAffected == 0 ? 0 : 1 - Double(bytesIfQuantized) / Double(bytesAffected)
        }
    }

    /// Quantizes then dequantizes every quantizable resident tensor, in memory.
    ///
    /// This is D-020's gate, and it is deliberately the cheapest possible form of it: the
    /// values the kernels read afterwards are those a real Q8 path would give them, so the
    /// effect on the logits is measurable **before** a kernel, a disk format or a repacker
    /// path exists. What it does not measure is throughput — the same number of bytes is
    /// still read.
    ///
    /// Which tensors move is decided by `TensorRole`, not by this function: routers, norms,
    /// biases and sinks are excluded at the source.
    ///
    /// Requires `mutableResident: true`, otherwise the write faults.
    @discardableResult
    public func simulateQ8Residents() -> Q8Simulation {
        var tensors = 0
        var bytes = 0
        var quantized = 0
        var worst: Float = 0

        for placement in layout.resident where placement.role.isQuantizable {
            // A tensor whose length is not a whole number of blocks would leave a tail in
            // BF16; none of GPT-OSS's dimensions produce one, and the simulation would
            // silently flatter itself if one appeared. Count only what really moved.
            let values = placement.byteCount / 2
            guard values >= Q8.blockSize else { continue }

            let deviation = resident.withMutableWords(
                offset: placement.offset, byteCount: placement.byteCount
            ) { Q8.simulateInPlace($0) }

            tensors += 1
            bytes += placement.byteCount
            quantized += Q8.encodedByteCount(values: values - values % Q8.blockSize)
            worst = max(worst, deviation)
        }

        return Q8Simulation(
            tensorsAffected: tensors, bytesAffected: bytes,
            bytesIfQuantized: quantized, worstRelativeDeviation: worst)
    }

    /// Where a resident tensor sits inside the single `resident.bin` buffer.
    ///
    /// Kernels bind a sub-range via `setBuffer(offset:)` rather than creating one buffer per
    /// tensor. The offset is 256-byte aligned by construction, which allows wide vector
    /// loads in the shaders.
    public func residentTensor(_ name: String) throws -> (buffer: MTLBuffer, offset: Int, byteCount: Int) {
        guard let placement = layout.placement(of: name) else {
            throw LoadError.tensorMissing(name)
        }
        return (resident.buffer, placement.offset, placement.byteCount)
    }

    /// Everything the encoders need to read a dense tensor, whatever its precision.
    ///
    /// Kept separate from `residentTensor` so the tensors that never change format — norms,
    /// sinks, biases — keep the simpler accessor and cannot accidentally acquire a scales
    /// offset that means nothing for them.
    public func denseTensor(
        _ name: String
    ) throws -> (buffer: MTLBuffer, offset: Int, scalesOffset: Int, precision: WeightPrecision) {
        guard let placement = layout.placement(of: name) else {
            throw LoadError.tensorMissing(name)
        }
        return (resident.buffer, placement.offset, placement.scaleOffset, placement.precision)
    }

    /// Reads one embedding row without materializing the table. The output buffer is
    /// supplied by the caller and reused from token to token.
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

    /// Brings the resident weights into memory with a sequential read.
    ///
    /// **The embedding is deliberately excluded.** We read only one row per token:
    /// prefaulting it would wire down 1.08 GiB for nothing, against the whole project.
    @discardableResult
    public func prefault() -> Int {
        resident.prefault()
    }

    /// Memory footprint of the mappings as the system accounts for it. Only pages actually
    /// touched count: a mapped file is not a loaded file.
    public var mappedByteCount: Int { resident.byteCount + embedding.byteCount }
}
