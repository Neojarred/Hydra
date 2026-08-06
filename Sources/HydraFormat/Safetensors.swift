import Foundation

/// Reads a safetensors file's **header**, without ever touching the data.
///
/// This is the repacker's building block: the header gives, for each tensor, its dtype,
/// shape and byte range. The repacker can then request exactly those ranges — by `pread`
/// locally or by HTTP `Range` request remotely — without materializing a single
/// shard.
public struct SafetensorsHeader: Sendable {

    public struct Entry: Sendable, Equatable {
        public let dtype: DType
        public let shape: [Int]
        /// Offsets **relative to the start of the data section**.
        public let dataOffsets: (start: Int, end: Int)

        public var byteCount: Int { dataOffsets.end - dataOffsets.start }

        /// The number of logical elements.
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

    /// The size of the JSON header, excluding the 8-byte prefix.
    public let headerByteCount: Int
    public let tensors: [String: Entry]
    public let metadata: [String: String]

    /// The absolute offset, within the file, of the start of the data section.
    public var dataSectionOffset: Int { 8 + headerByteCount }

    /// A tensor's absolute byte range within the file.
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
                return "safetensors: fewer than 8 bytes, size prefix unreadable"
            case let .declaredHeaderTooLarge(n):
                return "safetensors: header declared at \(n) bytes, refused"
            case let .truncatedHeader(need, got):
                return "safetensors: truncated header, \(need) bytes expected, \(got) supplied"
            case .malformedJSON:
                return "safetensors: unreadable JSON header"
            case let .badEntry(k):
                return "safetensors: malformed entry \"\(k)\""
            case let .unknownDType(d, k):
                return "safetensors : dtype inconnu « \(d) » pour « \(k) »"
            }
        }
    }

    /// A safety bound: a legitimate header weighs a few tens of KiB. A corrupted prefix must
    /// not trigger a giant allocation.
    public static let maximumHeaderBytes = 128 * 1024 * 1024

    /// Reads the header length from the file's first 8 bytes. This allows requesting only the
    /// exact number of bytes needed afterwards.
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

    /// Parses an already-extracted header: `headerJSON` must contain exactly the
    /// `headerByteCount` bytes that follow the 8-byte prefix.
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

    /// Reads a local file's header, reading only what is necessary. The file's body is never
    /// loaded.
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
