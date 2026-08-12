import Foundation
import HydraCore
import Metal

/// The per-layer memory of a linear attention layer: a fixed state, and the convolution's
/// window.
///
/// `KVCache` holds a history that grows with the conversation and can be truncated to any
/// prefix. This holds neither property. A Gated DeltaNet layer keeps a `[keyDim][valueDim]`
/// state per value head that has already absorbed every token it has seen (D-027), so:
///
/// - **It does not grow.** Qwen carries 60 MiB across its thirty linear layers whether the
///   conversation is ten tokens or a hundred thousand. That is why only ten of its forty layers
///   make long context expensive.
/// - **It cannot be rewound.** There is no arithmetic that removes one token's contribution
///   from a decayed running sum. Truncating to a prefix is not slow here, it is impossible.
///
/// The second point has a user-visible consequence, which is what `checkpoint` exists for.
/// Appending to a conversation needs no rewind and is unaffected. Editing a message or
/// regenerating an answer rewinds to a point before the current one, and without a saved state
/// the only correct answer is to reset and reprocess the whole conversation. A snapshot taken
/// at each turn boundary turns that back into reprocessing one turn.
public final class RecurrentStateCache: @unchecked Sendable {

    /// One linear layer's memory.
    public struct Layer: @unchecked Sendable {
        /// `[valueHeads][keyDim][valueDim]`, float32 whatever the checkpoint stores (D-027).
        public let state: MTLBuffer
        /// The convolution's previous inputs, `[kernel - 1][convDim]`, oldest first.
        ///
        /// State in exactly the sense the recurrence is, and forgotten just as silently: a
        /// decode step that does not carry it computes the first tokens of every turn against
        /// zero padding, which is correct only at the very start of a sequence.
        public let window: MTLBuffer
        public let valueHeads: Int
        public let keyDim: Int
        public let valueDim: Int
        public let convDim: Int
        public let convKernel: Int

        public var stateBytes: Int { valueHeads * keyDim * valueDim * 4 }
        public var windowBytes: Int { (convKernel - 1) * convDim * 4 }
    }

    /// A saved state, and the token count it corresponds to.
    ///
    /// Deliberately opaque: a caller restores one or discards it, and cannot inspect or
    /// interpolate between them, because nothing sensible can be built from two states.
    public struct Checkpoint: @unchecked Sendable {
        public let tokens: Int
        fileprivate let state: [Data]
        fileprivate let window: [Data]
    }

    public let model: any ModelDescriptor
    /// The linear layers, in model order, with the layer index each belongs to.
    public private(set) var layers: [(index: Int, layer: Layer)] = []
    public private(set) var length = 0

    private let device: MTLDevice
    private var checkpoints: [Checkpoint] = []

    /// Snapshots kept before the oldest is dropped.
    ///
    /// Each is the whole state, 60 MiB for Qwen, so this is a memory budget rather than a
    /// history: four covers the recent turns a user is realistically going to edit, and the
    /// rest reprocess from the start.
    public static let maximumCheckpoints = 4

    public enum StateError: Error, CustomStringConvertible {
        case allocationFailed(layer: Int, bytes: Int)
        case notRewindable(to: Int, length: Int)

        public var description: String {
            switch self {
            case let .allocationFailed(layer, bytes):
                return "recurrent state: cannot allocate \(bytes) B for layer \(layer)"
            case let .notRewindable(to, length):
                return "recurrent state: cannot rewind from \(length) to \(to); a running "
                    + "state cannot be truncated and no checkpoint covers that position"
            }
        }
    }

    public init(model: any ModelDescriptor, geometry: RecurrentGeometry, device: MTLDevice) throws {
        self.model = model
        self.device = device

        for index in 0..<model.layerCount
        where model.attentionPattern(atLayer: index) == .linear {
            let stateBytes = geometry.valueHeads * geometry.keyDim * geometry.valueDim * 4
            let windowBytes = (geometry.convKernel - 1) * geometry.convDim * 4
            guard let state = device.makeBuffer(
                    length: stateBytes, options: .storageModeShared),
                let window = device.makeBuffer(length: windowBytes, options: .storageModeShared)
            else {
                throw StateError.allocationFailed(layer: index, bytes: stateBytes + windowBytes)
            }
            // A fresh sequence starts from zero, and so does a reset: the recurrence has no
            // other sensible initial condition, and the convolution's window is left padding.
            memset(state.contents(), 0, stateBytes)
            memset(window.contents(), 0, windowBytes)
            layers.append((index, Layer(
                state: state, window: window,
                valueHeads: geometry.valueHeads, keyDim: geometry.keyDim,
                valueDim: geometry.valueDim, convDim: geometry.convDim,
                convKernel: geometry.convKernel)))
        }
    }

    /// Total bytes held, independent of the context length.
    public var byteCount: Int {
        layers.reduce(0) { $0 + $1.layer.stateBytes + $1.layer.windowBytes }
    }

    public func advance(by tokens: Int = 1) { length += tokens }

    public func reset() {
        for (_, layer) in layers {
            memset(layer.state.contents(), 0, layer.stateBytes)
            memset(layer.window.contents(), 0, layer.windowBytes)
        }
        length = 0
        checkpoints.removeAll()
    }

    // MARK: - Checkpoints

    /// Saves the current state, so a later turn can be rewound to this point.
    ///
    /// Called at turn boundaries rather than per token: the cost is the whole state, and the
    /// positions a user can rewind to are the ones a conversation has as seams.
    public func checkpoint() {
        let saved = Checkpoint(
            tokens: length,
            state: layers.map { Data(bytes: $0.layer.state.contents(), count: $0.layer.stateBytes) },
            window: layers.map { Data(bytes: $0.layer.window.contents(), count: $0.layer.windowBytes) })
        checkpoints.removeAll { $0.tokens == length }
        checkpoints.append(saved)
        if checkpoints.count > Self.maximumCheckpoints { checkpoints.removeFirst() }
    }

    /// The furthest position at or below `tokens` that can be restored, or nil for none.
    ///
    /// Unlike a key/value cache, this is not a contiguous range: the answer is one of the saved
    /// positions, not any position below the current one.
    public func restorablePosition(atOrBelow tokens: Int) -> Int? {
        checkpoints.filter { $0.tokens <= tokens }.map(\.tokens).max()
    }

    /// True if the cache can resume from `tokens` exactly.
    ///
    /// Only the current position, which is an append and needs nothing restored, or a saved
    /// checkpoint. Every position in between is unreachable, which is the difference from
    /// `KVCache.canRewind(to:)` and the reason it is spelled out here.
    public func canRewind(to tokens: Int) -> Bool {
        tokens == length || checkpoints.contains { $0.tokens == tokens }
    }

    /// Restores the state saved at `tokens`.
    public func rewind(to tokens: Int) throws {
        if tokens == length { return }
        guard let saved = checkpoints.first(where: { $0.tokens == tokens }) else {
            throw StateError.notRewindable(to: tokens, length: length)
        }
        for (slot, entry) in layers.enumerated() {
            saved.state[slot].withUnsafeBytes {
                entry.layer.state.contents().copyMemory(
                    from: $0.baseAddress!, byteCount: entry.layer.stateBytes)
            }
            saved.window[slot].withUnsafeBytes {
                entry.layer.window.contents().copyMemory(
                    from: $0.baseAddress!, byteCount: entry.layer.windowBytes)
            }
        }
        length = tokens
        // Anything saved after the point we returned to describes a future that no longer
        // happens, and restoring one later would mix two histories.
        checkpoints.removeAll { $0.tokens > tokens }
    }
}

/// The shape of one linear attention layer, which the descriptor supplies.
public struct RecurrentGeometry: Sendable, Equatable {
    public let valueHeads: Int
    public let keyHeads: Int
    public let keyDim: Int
    public let valueDim: Int
    public let convDim: Int
    public let convKernel: Int

    public init(
        valueHeads: Int, keyHeads: Int, keyDim: Int, valueDim: Int,
        convDim: Int, convKernel: Int
    ) {
        self.valueHeads = valueHeads
        self.keyHeads = keyHeads
        self.keyDim = keyDim
        self.valueDim = valueDim
        self.convDim = convDim
        self.convKernel = convKernel
    }

    /// Query and key heads are shared across value heads, the same grouped arrangement the
    /// full-attention layers use, applied to a recurrence (D-027).
    public var groupedQueryFactor: Int { valueHeads / keyHeads }
}
