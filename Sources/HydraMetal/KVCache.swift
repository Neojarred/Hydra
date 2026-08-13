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
    /// **each chunk re-reads the whole pool**, 10.1 GB for the 20B. The per-token cost is
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
        /// while its full layers hold 2 of 512, a factor of two. Sizing the cache from one
        /// geometry would over-allocate half the layers and, worse, index the other half
        /// wrongly.
        public let entryBytes: Int
        /// True if this layer's attention is bounded to `slidingWindow` tokens.
        ///
        /// Distinct from `ringSize`: a layer can be sliding *for attention* while being
        /// stored linearly. Conflating the two would make linear storage turn attention into
        /// full attention, the model would change behaviour without signalling anything.
        public let windowed: Bool
        /// False for a recurrent layer, which keeps a fixed state elsewhere and nothing here.
        ///
        /// Such a layer still occupies an index, so `layers[i]` stays the layer's own entry and
        /// no caller has to translate indices. What it does not occupy is memory: the buffers
        /// are a single byte, and every read of them is guarded (D-027).
        public let keepsHistory: Bool
    }

    public private(set) var layers: [Layer]
    /// Number of tokens already written.
    public private(set) var length = 0

    /// True if the cache can return to an earlier position.
    ///
    /// A ring cannot: past its capacity it has overwritten what would need to be recovered.
    /// Linear storage always can, the rows beyond simply being rewritten on the next pass.
    /// True when *any* rewind is possible, linear storage only.
    ///
    /// Kept for callers that ask the blunt question, but it is the wrong one for a chat: a
    /// model with sliding layers answers `false` here and can still resume a conversation,
    /// which is what `canRewind(to:)` decides.
    public var canRewind: Bool { layers.allSatisfy { $0.ringSize == 0 } }

    /// The furthest back the cache can be rewound and still answer correctly.
    ///
    /// Linear layers keep everything, so they are unbounded. A ring holds
    /// `slidingWindow + prefillChunk` positions while attention at position `p` reads only
    /// `[p - slidingWindow, p]`, so the margin the chunked prefill needs is exactly the depth
    /// a rewind may spend: rewinding to `R` from `length` is safe while
    /// `R >= length - prefillChunk`.
    ///
    /// This matters more than it sounds. Gemma 4 has a ring on five layers in six, so it
    /// reports `canRewind == false`, and the app read that as "cannot resume" and re-prefilled
    /// the entire conversation on every turn, 38 s on a thousand-token chat where the new
    /// message was a dozen tokens. A turn needs a rewind of at most a few tokens.
    public var rewindFloor: Int {
        let margin = layers.compactMap { $0.ringSize == 0 ? nil : $0.ringSize }.min()
        guard let margin else { return 0 }
        return max(0, length - (margin - model.slidingWindow))
    }

    /// True if the cache can be rewound to `tokens` and still hold every key that follows it.
    public func canRewind(to tokens: Int) -> Bool {
        tokens >= 0 && tokens <= length && tokens >= rewindFloor
    }

    public enum CacheError: Error, CustomStringConvertible {
        case allocationFailed(layer: Int, bytes: Int)
        case overflow(position: Int, capacity: Int)
        case unsupportedPattern(layer: Int, pattern: AttentionPattern)

        public var description: String {
            switch self {
            case let .allocationFailed(layer, bytes):
                return "KV cache: cannot allocate \(bytes) B for layer \(layer)"
            case let .overflow(position, capacity):
                return "KV cache: position \(position) beyond capacity \(capacity)"
            case let .unsupportedPattern(layer, pattern):
                return "KV cache: layer \(layer) is \(pattern.rawValue) attention, which keeps "
                    + "a recurrent state rather than a key/value history"
            }
        }
    }

    /// The context up to which sliding layers get linear storage.
    ///
    /// Linear storage is what makes the cache rewindable, hence reusable from one
    /// conversation turn to the next. For the 20B at 4k it costs 201 MiB instead of 31,
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
            let pattern = model.attentionPattern(atLayer: index)

            // A recurrent layer keeps a fixed state elsewhere and nothing here, so it gets an
            // index and no memory. It used to be refused outright, which was right while
            // nothing could run such a model; now `RecurrentStateCache` holds what it needs and
            // a mixed model has to be allocatable.
            //
            // What has not changed is that its entry must never be read: the comparison this
            // replaced asked `== .sliding`, so a linear layer answered "no" and was handed a
            // context-sized buffer it would never touch (D-027).
            guard pattern.keepsKeyValueHistory else {
                guard let empty = device.makeBuffer(length: 1, options: .storageModeShared)
                else { throw CacheError.allocationFailed(layer: index, bytes: 1) }
                built.append(Layer(
                    keys: empty, values: empty, ringSize: 0, capacity: 0,
                    entryBytes: 0, windowed: false, keepsHistory: false))
                continue
            }

            let sliding = pattern == .sliding
            // "Linear" here means linear *storage*, the opposite of a ring, and has nothing to
            // do with linear attention. Named apart now that both exist.
            let wholeContext = contextLength <= Self.linearWindowLimit
            let ringSize = (sliding && !wholeContext)
                ? model.slidingWindow + Self.prefillChunk : 0
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
                entryBytes: entryBytes, windowed: sliding, keepsHistory: true))
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
        precondition(
            layers[index].keepsHistory,
            "layer \(index) is recurrent: it has no visible range, its state is elsewhere")
        guard layers[index].windowed else { return (0, position + 1) }
        let start = max(0, position - model.slidingWindow + 1)
        return (start, position - start + 1)
    }

    public func advance() throws {
        length += 1
        // Only full-attention layers can overflow: the rings of the sliding layers recycle
        // their rows by construction, and a recurrent layer has no rows at all. Without that
        // last exclusion a mixed model overflows at position zero, because a layer holding
        // nothing has a capacity of nothing.
        for layer in layers where layer.keepsHistory && layer.ringSize == 0 {
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
        precondition(
            canRewind(to: tokens),
            "rewind to \(tokens) is outside the history a ring still holds "
                + "(length \(length), floor \(rewindFloor))")
        length = tokens
    }
}
