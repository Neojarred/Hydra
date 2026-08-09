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

    public init(url: URL, device: MTLDevice) throws {
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

        guard let pointer = mmap(nil, size, PROT_READ, MAP_PRIVATE, descriptor, 0),
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

    deinit { munmap(base, mappedLength) }
}

/// A `.hydra` installation, opened and ready to feed the kernels.
///
/// Only two files are mapped permanently:
///
/// - `resident.bin` — attention, routers, norms, sinks, LM head. Read on every token, so
///   meant to stay warm.
/// - `embed.bin` — the embedding table, **when the model has one**. Mapped but deliberately
///   outside the working set: only one row is read per token, so wiring down 1.08 GiB would
///   be waste. A model that ties its embedding to the output head has no such file — the
///   whole matrix is read every token, so it belongs with the resident weights.
///
/// Experts are not mapped: they go through the slot cache, with bounded `pread`s.
/// TurboFieldfare measured the gap between the two approaches — 0.50 tok/s with `mmap`
/// against 3.97 with parallel `pread` — because demand paging gives no control over when
/// reads happen or how many run at once.
public final class ModelMapping: @unchecked Sendable {

    public let root: URL
    public let model: any ModelDescriptor
    public let manifest: HydraManifest
    public let layout: HydraLayout
    public let resident: MappedFile
    /// The separate embedding file, when the model has one.
    ///
    /// `nil` for a model whose embedding is tied to the output head: it lives in
    /// `resident.bin` instead, and mapping a file that does not exist would fail at load.
    public let embedding: MappedFile?

    public enum LoadError: Error, CustomStringConvertible {
        case tensorMissing(String)

        public var description: String {
            switch self {
            case .tensorMissing(let name):
                return "tensor missing from the resident layout: \(name)"
            }
        }
    }

    public init(root: URL, model: any ModelDescriptor, device: MTLDevice) throws {
        self.root = root
        self.model = model
        self.manifest = try HydraManifest.read(from: root)
        try manifest.validate(against: model, root: root)
        self.layout = HydraLayout(model: model)
        self.resident = try MappedFile(url: root.appending(path: "resident.bin"), device: device)
        self.embedding = model.embeddingFileBytes > 0
            ? try MappedFile(url: root.appending(path: "embed.bin"), device: device)
            : nil
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

    /// Reads one embedding row without materializing the table. The output buffer is
    /// supplied by the caller and reused from token to token.
    public func readEmbedding(token: Int, into destination: UnsafeMutableBufferPointer<Float>) {
        precondition(destination.count == model.hiddenSize)
        let rowBytes = model.hiddenSize * 2

        // Two arrangements, and the runtime is told which rather than guessing: a dedicated
        // file mapped outside the working set, or a resident tensor that is also the output
        // head.
        let file: MappedFile
        let base: Int
        if let embedding {
            file = embedding
            base = 0
        } else if let name = model.residentEmbeddingTensor,
            let placement = layout.placement(of: name)
        {
            file = resident
            base = placement.offset
        } else {
            return
        }

        let offset = base + token * rowBytes
        file.withBytes { raw in
            for i in 0..<model.hiddenSize {
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
    public var mappedByteCount: Int { resident.byteCount + (embedding?.byteCount ?? 0) }
}
