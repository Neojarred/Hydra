import Foundation

/// An expert blob's internal layout, and the stride from one expert to the next.
///
/// This computation lives in `HydraCore` because **two consumers depend on it and must
/// never diverge**: the on-disk format (`HydraFormat.HydraLayout`) and the sizing of memory
/// slots (`MemoryBudget`). An early version computed them separately; the slots came out
/// under-allocated by 128 bytes — exactly the alignment padding the format was adding.
/// A single source of truth removes this class of error.
///
public struct ExpertBlobLayout: Sendable, Equatable {

    /// A blob's alignment within its layer file, and hence that of every `pread`.
    /// A misaligned blob would force the kernel to read one extra page at each end, across
    /// 144 reads per token.
    public static let pageAlignment = 16384

    /// The alignment of each sub-tensor within the blob.
    ///
    /// The offsets passed to `setBuffer(offset:)` carry an alignment constraint, and the width
    /// of vector loads in the shaders depends on the address's actual alignment.
    /// TurboFieldfare documented a bug where a 32-bit path passed the tests at offset zero and
    /// then produced noise in decoding, because the live offsets were only 2-byte aligned. We
    /// align generously: the overhead is 128 bytes per blob, i.e. 0.001 %.
    ///
    public static let tensorAlignment = 256

    public struct Slot: Sendable, Equatable {
        public let offset: Int
        public let byteCount: Int
        public var end: Int { offset + byteCount }
    }

    public let gateUpBlocks: Slot
    public let gateUpScales: Slot
    public let gateUpBias: Slot
    public let downBlocks: Slot
    public let downScales: Slot
    public let downBias: Slot

    /// A blob's useful bytes once laid out, internal alignment padding included.
    /// Always greater than or equal to the raw sum of the source tensors.
    public let payloadBytes: Int

    /// The distance between two consecutive blobs, and **the size of a memory slot**.
    public let strideBytes: Int

    public var slots: [Slot] {
        [gateUpBlocks, gateUpScales, gateUpBias, downBlocks, downScales, downBias]
    }

    /// The raw sum of the source tensors, with no padding. This is the value that must match
    /// the Hugging Face checkpoint.
    public var sourceBytes: Int { slots.reduce(0) { $0 + $1.byteCount } }

    init(config: GptOssConfig) {
        let inBlocks = config.hiddenSize / MXFP4Layout.blockSize
        let downInBlocks = config.intermediateSize / MXFP4Layout.blockSize
        let gateUpRows = 2 * config.intermediateSize
        let downRows = config.hiddenSize

        var cursor = 0
        func place(_ size: Int) -> Slot {
            cursor = alignUp(cursor, to: Self.tensorAlignment)
            let slot = Slot(offset: cursor, byteCount: size)
            cursor += size
            return slot
        }

        // The order follows the MoE kernel's consumption order: gate_up first (projection then
        // SwiGLU), then down (reduction).
        self.gateUpBlocks = place(gateUpRows * inBlocks * MXFP4Layout.packedBytesPerBlock)
        self.gateUpScales = place(gateUpRows * inBlocks * MXFP4Layout.scaleBytesPerBlock)
        self.gateUpBias = place(gateUpRows * 2)
        self.downBlocks = place(downRows * downInBlocks * MXFP4Layout.packedBytesPerBlock)
        self.downScales = place(downRows * downInBlocks * MXFP4Layout.scaleBytesPerBlock)
        self.downBias = place(downRows * 2)

        self.payloadBytes = cursor
        self.strideBytes = alignUp(cursor, to: Self.pageAlignment)
    }
}

@inlinable
public func alignUp(_ value: Int, to alignment: Int) -> Int {
    let r = value % alignment
    return r == 0 ? value : value + (alignment - r)
}
