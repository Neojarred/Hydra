import Foundation

/// Lecture de l'**en-tête** d'un fichier safetensors, sans jamais toucher aux données.
///
/// C'est la brique de base du repacker : l'en-tête donne, pour chaque tenseur, son type,
/// sa forme et sa plage d'octets. Le repacker peut alors demander exactement ces plages —
/// par `pread` en local ou par requête HTTP `Range` en distant — sans matérialiser un
/// seul shard.
public struct SafetensorsHeader: Sendable {

    public struct Entry: Sendable, Equatable {
        public let dtype: DType
        public let shape: [Int]
        /// Décalages **relatifs au début de la section de données**.
        public let dataOffsets: (start: Int, end: Int)

        public var byteCount: Int { dataOffsets.end - dataOffsets.start }

        /// Nombre d'éléments logiques.
        public var elementCount: Int { shape.reduce(1, *) }

        public static func == (a: Entry, b: Entry) -> Bool {
            a.dtype == b.dtype && a.shape == b.shape
                && a.dataOffsets.start == b.dataOffsets.start
                && a.dataOffsets.end == b.dataOffsets.end
        }
    }

    public enum DType: String, Sendable, Codable {
        case bf16 = "BF16"
        case f16 = "F16"
        case f32 = "F32"
        case u8 = "U8"
        case i8 = "I8"
        case i32 = "I32"
        case i64 = "I64"
        case bool = "BOOL"

        public var byteWidth: Int {
            switch self {
            case .bf16, .f16: return 2
            case .f32, .i32: return 4
            case .i64: return 8
            case .u8, .i8, .bool: return 1
            }
        }
    }

    /// Taille de l'en-tête JSON, hors les 8 octets de préfixe.
    public let headerByteCount: Int
    public let tensors: [String: Entry]
    public let metadata: [String: String]

    /// Décalage absolu, dans le fichier, du début de la section de données.
    public var dataSectionOffset: Int { 8 + headerByteCount }

    /// Plage d'octets absolue d'un tenseur dans le fichier.
    public func fileRange(of name: String) -> Range<Int>? {
        guard let e = tensors[name] else { return nil }
        return (dataSectionOffset + e.dataOffsets.start)..<(dataSectionOffset + e.dataOffsets.end)
    }

    public enum ParseError: Error, CustomStringConvertible {
        case tooShort
        case declaredHeaderTooLarge(Int)
        case truncatedHeader(need: Int, got: Int)
        case malformedJSON
        case badEntry(String)
        case unknownDType(String, in: String)

        public var description: String {
            switch self {
            case .tooShort:
                return "safetensors : moins de 8 octets, préfixe de taille illisible"
            case let .declaredHeaderTooLarge(n):
                return "safetensors : en-tête déclaré à \(n) octets, refusé"
            case let .truncatedHeader(need, got):
                return "safetensors : en-tête tronqué, \(need) octets attendus, \(got) fournis"
            case .malformedJSON:
                return "safetensors : en-tête JSON illisible"
            case let .badEntry(k):
                return "safetensors : entrée « \(k) » malformée"
            case let .unknownDType(d, k):
                return "safetensors : dtype inconnu « \(d) » pour « \(k) »"
            }
        }
    }

    /// Borne de sûreté : un en-tête légitime pèse quelques dizaines de kio.
    /// Un préfixe corrompu ne doit pas déclencher une allocation géante.
    public static let maximumHeaderBytes = 128 * 1024 * 1024

    /// Lit la longueur d'en-tête depuis les 8 premiers octets du fichier.
    /// Permet de ne demander ensuite que le nombre exact d'octets nécessaires.
    public static func headerLength(fromPrefix prefix: Data) throws -> Int {
        guard prefix.count >= 8 else { throw ParseError.tooShort }
        let n = prefix.prefix(8).withUnsafeBytes { raw in
            UInt64(littleEndian: raw.loadUnaligned(as: UInt64.self))
        }
        guard n > 0, n <= UInt64(maximumHeaderBytes) else {
            throw ParseError.declaredHeaderTooLarge(Int(clamping: n))
        }
        return Int(n)
    }

    /// Parse un en-tête déjà extrait : `headerJSON` doit contenir exactement les
    /// `headerByteCount` octets qui suivent le préfixe de 8 octets.
    public init(headerJSON: Data, headerByteCount: Int) throws {
        guard headerJSON.count >= headerByteCount else {
            throw ParseError.truncatedHeader(need: headerByteCount, got: headerJSON.count)
        }
        let slice = headerJSON.prefix(headerByteCount)
        guard let root = try? JSONSerialization.jsonObject(with: slice) as? [String: Any] else {
            throw ParseError.malformedJSON
        }

        self.headerByteCount = headerByteCount
        self.metadata = (root["__metadata__"] as? [String: String]) ?? [:]

        var parsed: [String: Entry] = [:]
        parsed.reserveCapacity(root.count)
        for (key, value) in root where key != "__metadata__" {
            guard let obj = value as? [String: Any],
                let dtypeRaw = obj["dtype"] as? String,
                let shape = obj["shape"] as? [Int],
                let offsets = obj["data_offsets"] as? [Int],
                offsets.count == 2
            else {
                throw ParseError.badEntry(key)
            }
            guard let dtype = DType(rawValue: dtypeRaw) else {
                throw ParseError.unknownDType(dtypeRaw, in: key)
            }
            parsed[key] = Entry(
                dtype: dtype, shape: shape,
                dataOffsets: (start: offsets[0], end: offsets[1]))
        }
        self.tensors = parsed
    }

    /// Lit l'en-tête d'un fichier local en ne lisant que ce qui est nécessaire.
    /// Le corps du fichier n'est jamais chargé.
    public static func read(contentsOf url: URL) throws -> SafetensorsHeader {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        guard let prefix = try handle.read(upToCount: 8) else { throw ParseError.tooShort }
        let n = try headerLength(fromPrefix: prefix)
        guard let json = try handle.read(upToCount: n) else {
            throw ParseError.truncatedHeader(need: n, got: 0)
        }
        return try SafetensorsHeader(headerJSON: json, headerByteCount: n)
    }
}
