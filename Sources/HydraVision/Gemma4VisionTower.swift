import Foundation
import HydraCore
import HydraFormat
import HydraMetal
import Metal

/// Anything that can hand Gemma's tower a weight as a GPU buffer range.
public protocol Gemma4VisionWeightSource {
    func tensor(_ name: String) throws -> (buffer: MTLBuffer, offset: Int, byteCount: Int)
}

extension Gemma4VisionMapping: Gemma4VisionWeightSource {}

/// Gemma 4's vision tower, on the GPU.
///
/// Patches in, one vector a soft token out, ready to be spliced into the text model's embedding
/// stream. It follows `Gemma4VisionReference` exactly, which is what decides correct here.
///
/// Almost every kernel is one the text models already had. `rms_norm_batch` scales by the weight
/// directly, which is Gemma 4's convention rather than Gemma 3's `1 + w`; `rms_norm_heads`
/// normalizes q, k and v per head and already knew about the unscaled case; `gelu_mul` is the
/// GeGLU; `bf16_gemm` is every projection; and `vision_attention`, written for Qwen, takes its
/// scale as a parameter, so Gemma's 1.0 needs no new kernel. Only the rotary, the pool and one
/// unscaled norm are new.
public final class Gemma4VisionTower {

    public struct Timings: Sendable {
        public var positions: Double = 0
        public var blocks: Double = 0
        public var pool: Double = 0
        public var total: Double { positions + blocks + pool }
    }
    public private(set) var lastTimings = Timings()

    public let config: Gemma4VisionConfig
    public let context: MetalContext
    private let encoder: BatchEncoder
    private let forward_: ForwardEncoder
    private let weights: any Gemma4VisionWeightSource

    public init(
        config: Gemma4VisionConfig = .a4b, context: MetalContext,
        weights: any Gemma4VisionWeightSource
    ) {
        self.config = config
        self.context = context
        self.encoder = BatchEncoder(context: context)
        self.forward_ = ForwardEncoder(context: context)
        self.weights = weights
    }

    public enum TowerError: Error, CustomStringConvertible {
        case allocationFailed(String, bytes: Int)
        case wrongPatchCount(given: Int, expected: Int)

        public var description: String {
            switch self {
            case let .allocationFailed(what, bytes):
                return "cannot allocate \(bytes) bytes for the tower's \(what)"
            case let .wrongPatchCount(given, expected):
                return "given \(given) patch values, the grid implies \(expected)"
            }
        }
    }

    private func buffer(_ what: String, floats: Int) throws -> MTLBuffer {
        guard let buffer = context.device.makeBuffer(
            length: max(floats, 1) * 4, options: .storageModeShared)
        else { throw TowerError.allocationFailed(what, bytes: floats * 4) }
        return buffer
    }

    /// Runs the tower and returns `[tokenCount][outHiddenSize]`, flattened.
    public func forward(
        patches: [Float], gridHeight: Int, gridWidth: Int
    ) throws -> [Float] {
        let states = try encode(
            patches: patches, gridHeight: gridHeight, gridWidth: gridWidth)
        let mark = Date()
        let pooled = try pool(states, gridHeight: gridHeight, gridWidth: gridWidth)
        lastTimings.pool = Date().timeIntervalSince(mark)
        return pooled
    }

    /// The patch projection and the blocks, leaving the per-patch states on the GPU.
    private func encode(
        patches: [Float], gridHeight: Int, gridWidth: Int
    ) throws -> MTLBuffer {
        let count = gridHeight * gridWidth
        guard patches.count == count * config.patchElements else {
            throw TowerError.wrongPatchCount(
                given: patches.count, expected: count * config.patchElements)
        }
        let hidden = config.hiddenSize
        let heads = config.headCount
        let headDim = config.headDim
        let eps = config.rmsNormEps

        let input = try buffer("patches", floats: patches.count)
        patches.withUnsafeBytes { raw in
            input.contents().copyMemory(from: raw.baseAddress!, byteCount: raw.count)
        }
        let states = try buffer("states", floats: count * hidden)
        let normed = try buffer("normed", floats: count * hidden)
        let query = try buffer("query", floats: count * hidden)
        let key = try buffer("key", floats: count * hidden)
        let value = try buffer("value", floats: count * hidden)
        let qkv = try buffer("qkv", floats: count * 3 * hidden)
        let attended = try buffer("attention", floats: count * hidden)
        let gate = try buffer("gate", floats: count * config.intermediateSize)
        let up = try buffer("up", floats: count * config.intermediateSize)

        var timings = Timings()
        var mark = Date()
        let positions = try positionEmbeddings(gridHeight: gridHeight, gridWidth: gridWidth)
        let angles = try rotaryAngles(gridHeight: gridHeight, gridWidth: gridWidth)
        timings.positions = Date().timeIntervalSince(mark)
        mark = Date()

        // --- Patch projection, no bias, plus the two-table position ---
        let projection = try weights.tensor(Gemma4VisionMapping.Name.patchProjection)
        guard let setup = context.commandQueue.makeCommandBuffer() else {
            throw MetalContext.ContextError.noCommandQueue
        }
        try encoder.denseProjection(
            weights: projection.buffer, weightsOffset: projection.offset,
            bias: nil, biasOffset: 0, input: input, output: states,
            rows: hidden, cols: config.patchElements, tokens: count, in: setup)
        try encoder.addInPlace(
            target: states, addend: positions, size: count * hidden, in: setup)
        setup.commit()
        try context.wait(setup)

        for layer in 0..<config.depth {
            guard let command = context.commandQueue.makeCommandBuffer() else {
                throw MetalContext.ContextError.noCommandQueue
            }

            // --- Attention: norm, project, normalize each head, turn q and k, attend ---
            let inputNorm = try weights.tensor(Gemma4VisionMapping.Name.inputNorm(layer))
            try encoder.rmsNorm(
                input: states, scale: inputNorm.buffer, scaleOffset: inputNorm.offset,
                output: normed, size: hidden, tokens: count, eps: eps, in: command)

            for (name, destination) in [
                (Gemma4VisionMapping.Name.query(layer), query),
                (Gemma4VisionMapping.Name.key(layer), key),
                (Gemma4VisionMapping.Name.value(layer), value),
            ] {
                let w = try weights.tensor(name)
                try encoder.denseProjection(
                    weights: w.buffer, weightsOffset: w.offset, bias: nil, biasOffset: 0,
                    input: normed, output: destination,
                    rows: hidden, cols: hidden, tokens: count, in: command)
            }

            let qNorm = try weights.tensor(Gemma4VisionMapping.Name.queryNorm(layer))
            let kNorm = try weights.tensor(Gemma4VisionMapping.Name.keyNorm(layer))
            try encoder.rmsNormHeadsBatch(
                buffer: query, scale: qNorm.buffer, scaleOffset: qNorm.offset,
                headDim: headDim, heads: heads, tokens: count, eps: eps, in: command)
            try encoder.rmsNormHeadsBatch(
                buffer: key, scale: kNorm.buffer, scaleOffset: kNorm.offset,
                headDim: headDim, heads: heads, tokens: count, eps: eps, in: command)
            // The value is normalized too, by a norm with no learned weight. There is no tensor
            // for it in the checkpoint, which is why it is easy to miss entirely.
            try encoder.rmsNormHeadsBatch(
                buffer: value, scale: nil, scaleOffset: 0,
                headDim: headDim, heads: heads, tokens: count, eps: eps, in: command)

            try encoder.gemmaVisionRotary(
                buffer: query, angles: angles, tokens: count, heads: heads,
                headDim: headDim, perAxis: config.rotaryChannelsPerAxis, in: command)
            try encoder.gemmaVisionRotary(
                buffer: key, angles: angles, tokens: count, heads: heads,
                headDim: headDim, perAxis: config.rotaryChannelsPerAxis, in: command)

            // `vision_attention` reads one packed [q | k | v] buffer, as Qwen's tower produces.
            // Gemma projects the three separately, so they are gathered into that layout rather
            // than a second attention kernel being written for the same arithmetic.
            try pack(query: query, key: key, value: value, into: qkv, tokens: count,
                     hidden: hidden, in: command)
            try encoder.visionAttention(
                qkv: qkv, output: attended, patches: count, heads: heads, headDim: headDim,
                scale: config.attentionScale, in: command)

            let output = try weights.tensor(Gemma4VisionMapping.Name.output(layer))
            try encoder.denseProjection(
                weights: output.buffer, weightsOffset: output.offset, bias: nil, biasOffset: 0,
                input: attended, output: normed,
                rows: hidden, cols: hidden, tokens: count, in: command)
            // The second half of the sandwich: normed again, then the residual.
            let postAttention = try weights.tensor(
                Gemma4VisionMapping.Name.postAttentionNorm(layer))
            try encoder.rmsNorm(
                input: normed, scale: postAttention.buffer, scaleOffset: postAttention.offset,
                output: attended, size: hidden, tokens: count, eps: eps, in: command)
            try encoder.addInPlace(
                target: states, addend: attended, size: count * hidden, in: command)

            // --- MLP: norm, GeGLU, norm, add ---
            let preFeed = try weights.tensor(Gemma4VisionMapping.Name.preFeedforwardNorm(layer))
            try encoder.rmsNorm(
                input: states, scale: preFeed.buffer, scaleOffset: preFeed.offset,
                output: normed, size: hidden, tokens: count, eps: eps, in: command)

            let gateW = try weights.tensor(Gemma4VisionMapping.Name.gate(layer))
            let upW = try weights.tensor(Gemma4VisionMapping.Name.up(layer))
            try encoder.denseProjection(
                weights: gateW.buffer, weightsOffset: gateW.offset, bias: nil, biasOffset: 0,
                input: normed, output: gate,
                rows: config.intermediateSize, cols: hidden, tokens: count, in: command)
            try encoder.denseProjection(
                weights: upW.buffer, weightsOffset: upW.offset, bias: nil, biasOffset: 0,
                input: normed, output: up,
                rows: config.intermediateSize, cols: hidden, tokens: count, in: command)
            // `ForwardEncoder` holds one shared encoder open across its dispatches while
            // `BatchEncoder` opens one per dispatch, and Metal asserts if a second is opened
            // while the first is live. So the shared one is closed before handing the command
            // buffer back. Qwen's tower never hit this because it uses only `BatchEncoder`.
            try forward_.geluMultiply(
                gate: gate, up: up, output: gate,
                size: count * config.intermediateSize, in: command)
            context.closeEncoder()

            let downW = try weights.tensor(Gemma4VisionMapping.Name.down(layer))
            try encoder.denseProjection(
                weights: downW.buffer, weightsOffset: downW.offset, bias: nil, biasOffset: 0,
                input: gate, output: normed,
                rows: hidden, cols: config.intermediateSize, tokens: count, in: command)
            let postFeed = try weights.tensor(Gemma4VisionMapping.Name.postFeedforwardNorm(layer))
            try encoder.rmsNorm(
                input: normed, scale: postFeed.buffer, scaleOffset: postFeed.offset,
                output: attended, size: hidden, tokens: count, eps: eps, in: command)
            try encoder.addInPlace(
                target: states, addend: attended, size: count * hidden, in: command)

            command.commit()
            try context.wait(command)
        }
        timings.blocks = Date().timeIntervalSince(mark)
        lastTimings = timings
        return states
    }

    /// Everything except the projector, for the comparison against the reference.
    ///
    /// The projector is the tower's one quantized tensor and a synthetic fixture has no
    /// quantized weights, so the two sides meet at the standardized soft tokens. Nothing
    /// architectural lives past that point.
    public func pooledForTesting(
        patches: [Float], gridHeight: Int, gridWidth: Int
    ) throws -> [Float] {
        let states = try encode(
            patches: patches, gridHeight: gridHeight, gridWidth: gridWidth)
        let hidden = config.hiddenSize
        let k = config.poolingKernelSize
        let pooledWidth = gridWidth / k
        let tokens = (gridHeight / k) * pooledWidth
        let pooled = try buffer("pooled", floats: tokens * hidden)

        guard let command = context.commandQueue.makeCommandBuffer() else {
            throw MetalContext.ContextError.noCommandQueue
        }
        let bias = try weights.tensor(Gemma4VisionMapping.Name.standardizationBias)
        let scale = try weights.tensor(Gemma4VisionMapping.Name.standardizationScale)
        try encoder.gemmaVisionPool(
            patches: states, bias: bias.buffer, biasOffset: bias.offset,
            scale: scale.buffer, scaleOffset: scale.offset, output: pooled,
            hidden: hidden, gridWidth: gridWidth, kernelSize: k,
            pooledWidth: pooledWidth, pooledTokens: tokens, in: command)
        command.commit()
        try context.wait(command)
        return Array(UnsafeBufferPointer(
            start: pooled.contents().bindMemory(to: Float.self, capacity: tokens * hidden),
            count: tokens * hidden))
    }

    /// Gathers three separate projections into the packed `[q | k | v]` layout the attention
    /// kernel reads.
    private func pack(
        query: MTLBuffer, key: MTLBuffer, value: MTLBuffer, into qkv: MTLBuffer,
        tokens: Int, hidden: Int, in commandBuffer: MTLCommandBuffer
    ) throws {
        try encoder.gemmaVisionPackQKV(
            query: query, key: key, value: value, qkv: qkv,
            hidden: hidden, tokens: tokens, in: commandBuffer)
    }

    /// Pool, scale, standardize, then norm and project into the text model.
    private func pool(
        _ states: MTLBuffer, gridHeight: Int, gridWidth: Int
    ) throws -> [Float] {
        let hidden = config.hiddenSize
        let k = config.poolingKernelSize
        let pooledHeight = gridHeight / k, pooledWidth = gridWidth / k
        let tokens = pooledHeight * pooledWidth

        let pooled = try buffer("pooled", floats: tokens * hidden)
        let normed = try buffer("pooled normed", floats: tokens * hidden)
        let out = try buffer("tokens", floats: tokens * config.outHiddenSize)

        guard let command = context.commandQueue.makeCommandBuffer() else {
            throw MetalContext.ContextError.noCommandQueue
        }
        let bias = try weights.tensor(Gemma4VisionMapping.Name.standardizationBias)
        let scale = try weights.tensor(Gemma4VisionMapping.Name.standardizationScale)
        try encoder.gemmaVisionPool(
            patches: states, bias: bias.buffer, biasOffset: bias.offset,
            scale: scale.buffer, scaleOffset: scale.offset, output: pooled,
            hidden: hidden, gridWidth: gridWidth, kernelSize: k,
            pooledWidth: pooledWidth, pooledTokens: tokens, in: command)

        // A norm with no learned weight, then the one quantized tensor in the tower.
        try encoder.rmsNormUnscaledBatch(
            input: pooled, output: normed, size: hidden, tokens: tokens,
            eps: config.rmsNormEps, in: command)

        let weight = try weights.tensor(Gemma4VisionMapping.Name.projectionWeight)
        let scales = try weights.tensor(Gemma4VisionMapping.Name.projectionScales)
        let biases = try weights.tensor(Gemma4VisionMapping.Name.projectionBiases)
        // **The batched MLX projection wants staged activations, not raw ones.**
        //
        // Two preparations, both of which the kernel assumes and neither of which it checks.
        // The activations are read transposed, column-major over tokens; and the affine form
        // `q * scale + bias` needs the per-group sums of the activations, because the bias term
        // is `bias * sum(x)` and cannot be recovered from the quantized product alone.
        //
        // Passing row-major activations and a zeroed sums buffer produces a finite, plausible,
        // entirely wrong projection: the model described a flag as "overlapping text and
        // scrambled letters". Gemma's own layer runner stages every projection this way, which
        // is where the shape of this comes from.
        let padded = ForwardEncoder.paddedTokens(tokens)
        let transposed = try buffer("transposed", floats: padded * hidden)
        let sums = try buffer(
            "projection sums", floats: padded * (hidden / config.projectionGroupSize) + 64)
        try forward_.transposeActivations(
            input: normed, inputOffset: 0, output: transposed, outputOffset: 0,
            tokens: tokens, cols: hidden, in: command)
        try forward_.chunkSums(
            input: transposed, inputOffset: 0, output: sums, outputOffset: 0,
            tokens: tokens, cols: hidden, bits: config.projectionBits, in: command)
        try forward_.mlxAffineBatchedProjection(
            words: weight.buffer, wordsOffset: weight.offset,
            scales: scales.buffer, scalesOffset: scales.offset,
            biases: biases.buffer, biasesOffset: biases.offset,
            input: transposed, inputOffset: 0,
            sums: sums, sumsOffset: 0,
            output: out, outputOffset: 0,
            rows: config.outHiddenSize, cols: hidden, tokens: tokens,
            bits: config.projectionBits, groupSize: config.projectionGroupSize, in: command)
        context.closeEncoder()
        command.commit()
        try context.wait(command)

        return Array(UnsafeBufferPointer(
            start: out.contents().bindMemory(
                to: Float.self, capacity: tokens * config.outHiddenSize),
            count: tokens * config.outHiddenSize))
    }

    // MARK: - Positions

    /// The two position tables, looked up and summed, one vector a patch.
    private func positionEmbeddings(gridHeight: Int, gridWidth: Int) throws -> MTLBuffer {
        let table = try weights.tensor(Gemma4VisionMapping.Name.positionTable)
        let hidden = config.hiddenSize
        let count = gridHeight * gridWidth
        let out = try buffer("positions", floats: count * hidden)
        let destination = out.contents().bindMemory(to: Float.self, capacity: count * hidden)
        let raw = table.buffer.contents().advanced(by: table.offset)
        // The y table begins after the whole x table: the leading 2 of [2, 10240, 1152].
        let yTable = config.positionEmbeddingSize * hidden

        for index in 0..<count {
            let position = config.patchPosition(atIndex: index, gridWidth: gridWidth)
            let base = index * hidden
            for i in 0..<hidden {
                let x = raw.loadUnaligned(
                    fromByteOffset: (position.x * hidden + i) * 2, as: UInt16.self)
                let y = raw.loadUnaligned(
                    fromByteOffset: (yTable + position.y * hidden + i) * 2, as: UInt16.self)
                destination[base + i] = BF16.toFloat(UInt16(littleEndian: x))
                    + BF16.toFloat(UInt16(littleEndian: y))
            }
        }
        return out
    }

    /// The rotary angles of every patch, x angles then y angles.
    private func rotaryAngles(gridHeight: Int, gridWidth: Int) throws -> MTLBuffer {
        let reference = Gemma4VisionReference(config: config)
        let perAxis = config.rotaryChannelsPerAxis
        let count = gridHeight * gridWidth
        let out = try buffer("angles", floats: count * perAxis)
        let destination = out.contents().bindMemory(to: Float.self, capacity: count * perAxis)
        for index in 0..<count {
            let position = config.patchPosition(atIndex: index, gridWidth: gridWidth)
            let angles = reference.rotaryAngles(x: position.x, y: position.y)
            for i in 0..<perAxis { destination[index * perAxis + i] = Float(angles[i]) }
        }
        return out
    }
}
