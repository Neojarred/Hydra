import Foundation

/// What the rest of the system needs to know about an expert blob, whatever is inside it.
///
/// The **sub-tensor offsets deliberately stay out of this**. `MemoryBudget`, `HydraLayout` and
/// `ExpertSlotCache` size slots and files; they never look inside a blob. Only the kernels and
/// the repack plan do, and those are architecture-specific by nature — MXFP4 carries packed
/// values, scales and biases, where a BF16 checkpoint carries two plain matrices. Putting the
/// offsets in the shared contract would force every architecture to pretend it has the others'
/// sub-tensors.
///
/// The two consumers below must never diverge: an early version computed the on-disk size and
/// the slot size separately, and the slots came out 128 bytes short — exactly the alignment
/// padding the format was adding.
public protocol ExpertBlob: Sendable {
    /// A blob's useful bytes once laid out, internal alignment padding included.
    var payloadBytes: Int { get }
    /// The distance between two consecutive blobs, and **the size of a memory slot**.
    var strideBytes: Int { get }
    /// The raw sum of the source tensors, with no padding. The value that must match the
    /// upstream checkpoint.
    var sourceBytes: Int { get }
}

/// An expert blob's internal layout, and the stride from one expert to the next.
///
/// This computation lives in `HydraCore` because **two consumers depend on it and must
/// never diverge**: the on-disk format (`HydraFormat.HydraLayout`) and the sizing of memory
/// slots (`MemoryBudget`). An early version computed them separately; the slots came out
/// under-allocated by 128 bytes — exactly the alignment padding the format was adding.
/// A single source of truth removes this class of error.
///
public struct ExpertBlobLayout: ExpertBlob, Equatable {

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


/// A Gemma 4 expert blob: two plain BF16 matrices, no scales and no biases.
///
/// The checkpoint stores `experts.gate_up_proj` as `(experts, 2 × moeIntermediate, hidden)`
/// and `experts.down_proj` as `(experts, hidden, moeIntermediate)` — fused per layer, which is
/// exactly the shape `ScatterCopy` already splits.
///
/// Being BF16 rather than MXFP4 is the whole reason a Gemma expert is 11.34 MiB against
/// GPT-OSS's 12.62 MiB despite being **far** smaller in parameters: 5.95 M values at two bytes
/// each, against 25.2 M values at 4.25 bits. That is the cost of running it as published
/// (D-021), and it is what makes the per-token read 2.86 GB.
public struct GemmaExpertBlobLayout: ExpertBlob, Equatable {

    public let gateUp: ExpertBlobLayout.Slot
    public let down: ExpertBlobLayout.Slot
    public let payloadBytes: Int
    public let strideBytes: Int

    public var slots: [ExpertBlobLayout.Slot] { [gateUp, down] }
    public var sourceBytes: Int { slots.reduce(0) { $0 + $1.byteCount } }

    public init(hiddenSize: Int, moeIntermediateSize: Int) {
        var cursor = 0
        func place(_ size: Int) -> ExpertBlobLayout.Slot {
            cursor = alignUp(cursor, to: ExpertBlobLayout.tensorAlignment)
            let slot = ExpertBlobLayout.Slot(offset: cursor, byteCount: size)
            cursor += size
            return slot
        }

        // Same order as MXFP4's, and for the same reason: the kernel consumes gate_up first
        // (projection then activation), then down (reduction).
        self.gateUp = place(2 * moeIntermediateSize * hiddenSize * 2)
        self.down = place(hiddenSize * moeIntermediateSize * 2)

        self.payloadBytes = cursor
        self.strideBytes = alignUp(cursor, to: ExpertBlobLayout.pageAlignment)
    }
}

@inlinable
public func alignUp(_ value: Int, to alignment: Int) -> Int {
    let r = value % alignment
    return r == 0 ? value : value + (alignment - r)
}

/// A Gemma 4 MLX expert blob: three affine-quantized matrices, each a triple.
///
/// Nine sub-tensors where the BF16 build has two and MXFP4 has six, because the checkpoint
/// keeps `gate_proj`, `up_proj` and `down_proj` **separate** — `experts.switch_glu.*` — and
/// every one of them carries its own scales *and* biases.
///
/// The unfused arrangement is the reason this is a layout of its own rather than a parameter.
/// The BF16 build stores `gate_up_proj` as one tensor of `2 × moeIntermediate` rows, which the
/// repacker splits; here there is nothing to split, and a layout that assumed the fused shape
/// would place `up` where the second half of `gate` belongs. Both produce a blob of plausible
/// size.
public struct MLXExpertBlobLayout: ExpertBlob, Equatable {

    public let gateWeights: ExpertBlobLayout.Slot
    public let gateScales: ExpertBlobLayout.Slot
    public let gateBiases: ExpertBlobLayout.Slot
    public let upWeights: ExpertBlobLayout.Slot
    public let upScales: ExpertBlobLayout.Slot
    public let upBiases: ExpertBlobLayout.Slot
    public let downWeights: ExpertBlobLayout.Slot
    public let downScales: ExpertBlobLayout.Slot
    public let downBiases: ExpertBlobLayout.Slot

    public let payloadBytes: Int
    public let strideBytes: Int

    /// The quantization of the `gate`/`up` matrices, and of `down`. They differ in shape, not
    /// in bit width.
    public let gateLayout: MLXAffineLayout
    public let downLayout: MLXAffineLayout

    public var slots: [ExpertBlobLayout.Slot] {
        [
            gateWeights, gateScales, gateBiases,
            upWeights, upScales, upBiases,
            downWeights, downScales, downBiases,
        ]
    }

    public var sourceBytes: Int { slots.reduce(0) { $0 + $1.byteCount } }

    public init(hiddenSize: Int, moeIntermediateSize: Int, bits: Int, groupSize: Int) {
        let gate = MLXAffineLayout(
            bits: bits, groupSize: groupSize, rows: moeIntermediateSize, cols: hiddenSize)
        let down = MLXAffineLayout(
            bits: bits, groupSize: groupSize, rows: hiddenSize, cols: moeIntermediateSize)
        self.gateLayout = gate
        self.downLayout = down

        var cursor = 0
        func place(_ size: Int) -> ExpertBlobLayout.Slot {
            cursor = alignUp(cursor, to: ExpertBlobLayout.tensorAlignment)
            let slot = ExpertBlobLayout.Slot(offset: cursor, byteCount: size)
            cursor += size
            return slot
        }

        // Consumption order: gate and up feed the activation, down reduces it. Keeping each
        // matrix's three parts adjacent is what lets one kernel bind them from a single blob
        // with three offsets.
        self.gateWeights = place(gate.weightBytes)
        self.gateScales = place(gate.scaleBytes)
        self.gateBiases = place(gate.biasBytes)
        self.upWeights = place(gate.weightBytes)
        self.upScales = place(gate.scaleBytes)
        self.upBiases = place(gate.biasBytes)
        self.downWeights = place(down.weightBytes)
        self.downScales = place(down.scaleBytes)
        self.downBiases = place(down.biasBytes)

        self.payloadBytes = cursor
        self.strideBytes = alignUp(cursor, to: ExpertBlobLayout.pageAlignment)
    }
}
