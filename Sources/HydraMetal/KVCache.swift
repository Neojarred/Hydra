import Foundation
import HydraCore
import Metal

/// FP16 key/value cache, with two layouts depending on the attention pattern.
///
/// GPT-OSS alternates **128-token sliding-window** layers (even indices) with
/// **full-attention** layers (odd indices). That asymmetry makes long context surprisingly
/// cheap: only half the layers grow with the context.
///
/// - Sliding layers use a **bounded ring** of `slidingWindow + chunk` rows. The margin of
///   one prefill chunk is necessary: in chunked prefill, all of a chunk's writes precede
///   that chunk's attentions, so a ring the size of the window alone would overwrite keys
///   that are still needed.
/// - Full layers use **linear** storage sized on the requested context.
///
/// For the 120B at 128k: 4.51 GiB in total, where all-full layers would cost twice that.
public final class KVCache: @unchecked Sendable {

    public let model: any ModelDescriptor
    public let contextLength: Int
    /// Tokens processed together during prefill.
    ///
    /// This is the parameter that governs the cost of prefill, and it is not incidental: one
    /// chunk touches nearly all of a layer's experts while the cache holds only a few, so
    /// **each chunk re-reads the whole pool** — 10.1 GB for the 20B. The per-token cost is
    /// inversely proportional to this size.
    ///
    /// Measured on a thousand-token prompt, total time to the end of the answer: 128 tokens
    /// per chunk gives 39–44 s, 512 gives 28–29 for 44 MiB more, 1024 gives 25–28 for
    /// 119 MiB. We stop at 512, the best ratio per megabyte
    /// (docs/02-MEASUREMENTS.md, M-019).
    ///
    /// The sliding-window ring is sized on `slidingWindow + prefillChunk`: the constraint is
    /// already parameterized, so raising the chunk stays correct by construction.
    public static let prefillChunk = 512

    /// `MTLBuffer` is not `Sendable` in the Metal headers, even though buffers are safe to
    /// share for reading across threads. We assert that explicitly rather than disabling the
    /// check across the whole module.
    public struct Layer: @unchecked Sendable {
        public let keys: MTLBuffer
        public let values: MTLBuffer
        /// Ring capacity, or 0 for linear storage.
        public let ringSize: Int
        public let capacity: Int
        /// Bytes one token occupies in this layer's key (or value) buffer.
        ///
        /// Per layer, not per model. Gemma 4's sliding layers hold 8 key/value heads of 256
        /// while its full layers hold 2 of 512 — a factor of two. Sizing the cache from one
        /// geometry would over-allocate half the layers and, worse, index the other half
        /// wrongly.
        public let entryBytes: Int
        /// True if this layer's attention is bounded to `slidingWindow` tokens.
        ///
        /// Distinct from `ringSize`: a layer can be sliding *for attention* while being
        /// stored linearly. Conflating the two would make linear storage turn attention into
        /// full attention — the model would change behaviour without signalling anything.
        public let windowed: Bool
    }

    public private(set) var layers: [Layer]
    /// Number of tokens already written.
    public private(set) var length = 0

    /// True if the cache can return to an earlier position.
    ///
    /// A ring cannot: past its capacity it has overwritten what would need to be recovered.
    /// Linear storage always can, the rows beyond simply being rewritten on the next pass.
    public var canRewind: Bool { layers.allSatisfy { $0.ringSize == 0 } }

    public enum CacheError: Error, CustomStringConvertible {
        case allocationFailed(layer: Int, bytes: Int)
        case overflow(position: Int, capacity: Int)

        public var description: String {
            switch self {
            case let .allocationFailed(layer, bytes):
                return "KV cache: cannot allocate \(bytes) B for layer \(layer)"
            case let .overflow(position, capacity):
                return "KV cache: position \(position) beyond capacity \(capacity)"
            }
        }
    }

    /// The context up to which sliding layers get linear storage.
    ///
    /// Linear storage is what makes the cache rewindable, hence reusable from one
    /// conversation turn to the next. For the 20B at 4k it costs 201 MiB instead of 31 —
    /// 170 MiB to remove nearly all of the time to first token on follow-up turns. Past this
    /// threshold the arithmetic inverts and the ring takes over again.
    public static let linearWindowLimit = 8192

    public init(model: any ModelDescriptor, contextLength: Int, device: MTLDevice) throws {
        self.model = model
        self.contextLength = contextLength

        var built: [Layer] = []
        built.reserveCapacity(model.layerCount)

        for index in 0..<model.layerCount {
            let geometry = model.attentionGeometry(atLayer: index)
            let entryBytes = geometry.keyValueDim * MemoryLayout<Float16>.size
            let sliding = model.attentionPattern(atLayer: index) == .sliding
            let linear = contextLength <= Self.linearWindowLimit
            let ringSize = (sliding && !linear) ? model.slidingWindow + Self.prefillChunk : 0
            let capacity = ringSize > 0 ? ringSize : contextLength
            let bytes = capacity * entryBytes

            // Shared rather than private storage: on Apple Silicon it costs nothing and
            // the CPU can inspect the cache, which validation needs.
            guard let keys = device.makeBuffer(length: bytes, options: .storageModeShared),
                let values = device.makeBuffer(length: bytes, options: .storageModeShared)
            else {
                throw CacheError.allocationFailed(layer: index, bytes: bytes)
            }
            built.append(Layer(
                keys: keys, values: values, ringSize: ringSize, capacity: capacity,
                entryBytes: entryBytes, windowed: sliding))
        }
        self.layers = built
    }

    /// Bytes actually allocated. Must match the budget estimate.
    public var byteCount: Int {
        layers.reduce(0) { $0 + 2 * $1.capacity * $1.entryBytes }
    }

    /// The keys visible to the query at `position`, for a given layer.
    ///
    /// For a full layer: the whole history. For a sliding layer:
    /// at most the last `slidingWindow` tokens, the start position following the window.
    public func visibleRange(layer index: Int, position: Int) -> (start: Int, count: Int) {
        guard layers[index].windowed else { return (0, position + 1) }
        let start = max(0, position - model.slidingWindow + 1)
        return (start, position - start + 1)
    }

    public func advance() throws {
        length += 1
        // Only full-attention layers can overflow: the rings of the sliding layers
        // recycle their rows by construction.
        for layer in layers where layer.ringSize == 0 {
            guard length <= layer.capacity else {
                throw CacheError.overflow(position: length - 1, capacity: layer.capacity)
            }
        }
    }

    public func reset() {
        length = 0
    }

    /// Rewinds the cache to `tokens` written tokens.
    ///
    /// Rows beyond stay in memory but become invisible: nothing reads them, and the next
    /// pass rewrites them. Only meaningful on linear storage.
    public func rewind(to tokens: Int) {
        precondition(canRewind, "a ring cannot be rewound")
        precondition(tokens >= 0 && tokens <= length, "position outside the written history")
        length = tokens
    }
}
