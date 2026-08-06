import Foundation

/// Écrit les fichiers d'une installation `.hydra` en cours.
///
/// L'installation se construit dans un répertoire `<nom>.hydra.partial`, jamais à
/// l'emplacement final. Une installation interrompue reste donc visiblement partielle et
/// ne peut pas être prise pour une installation valide. La promotion vers le nom final
/// est atomique et n'intervient qu'après écriture du manifeste.
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
                return "création impossible de \(f) : \(String(cString: strerror(e)))"
            case let .allocateFailed(f, e):
                return "réservation d'espace impossible pour \(f) : \(String(cString: strerror(e)))"
            case let .writeFailed(f, o, e):
                return "écriture impossible dans \(f) à \(o) : \(String(cString: strerror(e)))"
            case let .shortWrite(f, expected, got):
                return "écriture courte dans \(f) : \(got) octets sur \(expected)"
            case let .promotionFailed(m):
                return "promotion de l'installation impossible : \(m)"
            }
        }
    }

    /// Crée l'arborescence et pré-alloue chaque fichier à sa taille finale.
    ///
    /// La pré-allocation sert deux buts : elle échoue **tout de suite** si le disque est
    /// insuffisant, plutôt qu'après des dizaines de gigaoctets téléchargés ; et elle évite
    /// que des écritures dispersées ne fragmentent le fichier.
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

    /// Écrit un bloc à un décalage absolu. Sûr depuis plusieurs tâches : `pwrite` ne
    /// touche pas au décalage courant du descripteur.
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

    /// Force l'écriture physique de tous les fichiers. Appelé avant d'écrire le manifeste,
    /// pour qu'un manifeste présent implique des données réellement durables.
    public func synchronize() throws {
        lock.lock()
        defer { lock.unlock() }
        for (file, fd) in descriptors {
            // F_FULLFSYNC va plus loin que fsync sur macOS : il vide aussi le cache du disque.
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

    /// Renomme le répertoire partiel vers son nom définitif.
    /// Le renommage est atomique : une installation est soit absente, soit complète.
    public func promote(to destination: URL) throws {
        close()
        let fm = FileManager.default
        if fm.fileExists(atPath: destination.path) {
            throw WriteError.promotionFailed("\(destination.lastPathComponent) existe déjà")
        }
        do {
            try fm.moveItem(at: root, to: destination)
        } catch {
            throw WriteError.promotionFailed(error.localizedDescription)
        }
    }

    deinit { close() }
}
