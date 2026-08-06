import Foundation

/// Writes the files of a `.hydra` installation in progress.
///
/// The installation is built in a `<name>.hydra.partial` directory, never at the final
/// location. An interrupted install therefore stays visibly partial and cannot be mistaken
/// for a valid one. Promotion to the final name is atomic and happens only after the
/// manifest has been written.
public final class InstallationWriter: @unchecked Sendable {

    public let root: URL
    private var descriptors: [DestinationFile: Int32] = [:]
    private let lock = NSLock()

    public enum WriteError: Error, CustomStringConvertible {
        case openFailed(String, errno: Int32)
        case allocateFailed(String, errno: Int32)
        case writeFailed(String, offset: Int, errno: Int32)
        case shortWrite(String, expected: Int, got: Int)
        case promotionFailed(String)

        public var description: String {
            switch self {
            case let .openFailed(f, e):
                return "cannot create \(f): \(String(cString: strerror(e)))"
            case let .allocateFailed(f, e):
                return "cannot reserve space for \(f): \(String(cString: strerror(e)))"
            case let .writeFailed(f, o, e):
                return "cannot write to \(f) at \(o): \(String(cString: strerror(e)))"
            case let .shortWrite(f, expected, got):
                return "short write to \(f): \(got) bytes of \(expected)"
            case let .promotionFailed(m):
                return "cannot promote the installation: \(m)"
            }
        }
    }

    /// Creates the directory tree and preallocates each file at its final size.
    ///
    /// Preallocation serves two purposes: it fails **immediately** if the disk is insufficient,
    /// rather than after tens of gigabytes downloaded; and it keeps scattered writes from
    /// fragmenting the file.
    public init(root: URL, sizes: [DestinationFile: Int]) throws {
        self.root = root
        try FileManager.default.createDirectory(
            at: root.appending(path: "experts"), withIntermediateDirectories: true)

        for (file, size) in sizes.sorted(by: { $0.key.path < $1.key.path }) {
            let path = root.appending(path: file.path).path
            let fd = open(path, O_RDWR | O_CREAT, 0o644)
            guard fd >= 0 else { throw WriteError.openFailed(file.path, errno: errno) }
            if ftruncate(fd, off_t(size)) != 0 {
                let e = errno
                Foundation.close(fd)
                throw WriteError.allocateFailed(file.path, errno: e)
            }
            descriptors[file] = fd
        }
    }

    /// Writes a block at an absolute offset. Safe from several tasks: `pwrite` does not touch
    /// the descriptor's current offset.
    public func write(_ data: Data, to file: DestinationFile, at offset: Int) throws {
        lock.lock()
        let fd = descriptors[file]
        lock.unlock()
        guard let fd else { throw WriteError.openFailed(file.path, errno: EBADF) }

        try data.withUnsafeBytes { raw in
            var written = 0
            while written < raw.count {
                let n = pwrite(
                    fd, raw.baseAddress!.advanced(by: written),
                    raw.count - written, off_t(offset + written))
                if n < 0 {
                    throw WriteError.writeFailed(file.path, offset: offset + written, errno: errno)
                }
                if n == 0 { break }
                written += n
            }
            guard written == raw.count else {
                throw WriteError.shortWrite(file.path, expected: raw.count, got: written)
            }
        }
    }

    /// Forces every file physically to disk. Called before writing the manifest, so that a
    /// present manifest implies genuinely durable data.
    public func synchronize() throws {
        lock.lock()
        defer { lock.unlock() }
        for (file, fd) in descriptors {
            // F_FULLFSYNC goes further than fsync on macOS: it also flushes the drive's cache.
            if fcntl(fd, F_FULLFSYNC) == -1 && fsync(fd) != 0 {
                throw WriteError.writeFailed(file.path, offset: -1, errno: errno)
            }
        }
    }

    public func close() {
        lock.lock()
        defer { lock.unlock() }
        for (_, fd) in descriptors { Foundation.close(fd) }
        descriptors.removeAll()
    }

    /// Renames the partial directory to its final name.
    /// The rename is atomic: an installation is either absent or complete.
    public func promote(to destination: URL) throws {
        close()
        let fm = FileManager.default
        if fm.fileExists(atPath: destination.path) {
            throw WriteError.promotionFailed("\(destination.lastPathComponent) already exists")
        }
        do {
            try fm.moveItem(at: root, to: destination)
        } catch {
            throw WriteError.promotionFailed(error.localizedDescription)
        }
    }

    deinit { close() }
}
