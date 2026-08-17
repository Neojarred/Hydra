import Foundation
import HydraCore
import HydraFormat
import HydraMetal
import Metal

/// Anything that can hand the tower a weight as a GPU buffer range.
///
/// A protocol rather than `VisionMapping` directly so the GPU tower can be run against a
/// synthetic tiny tower in tests. Checking 27 blocks of 1152 against a double-precision CPU
/// reference is not practical; checking two blocks of 16 is, and it is the same code path.
public protocol VisionWeightSource {
    func tensor(_ name: String) throws -> (buffer: MTLBuffer, offset: Int, byteCount: Int)
}

extension VisionMapping: VisionWeightSource {}

/// Qwen 3.6's vision tower, on the GPU.
///
/// Patches in, one vector a merged token out, ready to be spliced into the text model's
/// embedding stream. The structure follows `VisionReference` exactly, which is the point: the
/// reference decides what correct means and this is checked against it.
///
/// Every linear here is `bf16_gemm` through `BatchEncoder.denseProjection`, which already
/// handles a BF16 matrix against a batch of float rows with an optional BF16 bias. Only the
/// four operations the text models have no use for are new.
public final class VisionTower {

    public let config: Qwen35VisionConfig
    public let context: MetalContext
    private let encoder: BatchEncoder
    private let weights: any VisionWeightSource

    public init(
        config: Qwen35VisionConfig = .a3b, context: MetalContext,
        weights: any VisionWeightSource
    ) {
        self.config = config
        self.context = context
        self.encoder = BatchEncoder(context: context)
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
        patches: [Float], grid: Qwen35VisionConfig.Grid
    ) throws -> [Float] {
        let count = grid.patchCount
        guard patches.count == count * config.patchElements else {
            throw TowerError.wrongPatchCount(
                given: patches.count, expected: count * config.patchElements)
        }
        let hidden = config.hiddenSize
        let heads = config.headCount
        let headDim = config.headDim

        // --- Inputs and scratch ---
        let input = try buffer("patches", floats: patches.count)
        patches.withUnsafeBytes { raw in
            input.contents().copyMemory(from: raw.baseAddress!, byteCount: raw.count)
        }
        let states = try buffer("states", floats: count * hidden)
        let normed = try buffer("normed", floats: count * hidden)
        let qkv = try buffer("qkv", floats: count * 3 * hidden)
        let attended = try buffer("attention", floats: count * hidden)
        let wide = try buffer("mlp", floats: count * config.intermediateSize)

        // --- Positions, computed once on the CPU ---
        //
        // Both of them are pure functions of the patch's place in the image, so they cost a few
        // hundred kilobytes and a few milliseconds once, against being recomputed inside 27
        // blocks. The learned grid is a four-tap gather the GPU would do no faster.
        let positions = try positionEmbeddings(grid: grid)
        let angles = try rotaryAngles(grid: grid)

        // --- Patch embedding, plus the resampled learned position ---
        let patchWeight = try weights.tensor(VisionMapping.Name.patchWeight)
        let patchBias = try weights.tensor(VisionMapping.Name.patchBias)
        guard let setup = context.commandQueue.makeCommandBuffer() else {
            throw MetalContext.ContextError.noCommandQueue
        }
        try encoder.denseProjection(
            weights: patchWeight.buffer, weightsOffset: patchWeight.offset,
            bias: patchBias.buffer, biasOffset: patchBias.offset,
            input: input, output: states,
            rows: hidden, cols: config.patchElements, tokens: count, in: setup)
        try encoder.addInPlace(
            target: states, addend: positions, size: count * hidden, in: setup)
        setup.commit()
        try context.wait(setup)

        // --- The blocks ---
        for layer in 0..<config.depth {
            guard let command = context.commandQueue.makeCommandBuffer() else {
                throw MetalContext.ContextError.noCommandQueue
            }

            // Attention: norm, project, turn, attend, project back, add.
            let norm1W = try weights.tensor(VisionMapping.Name.norm1Weight(layer))
            let norm1B = try weights.tensor(VisionMapping.Name.norm1Bias(layer))
            try encoder.visionLayerNorm(
                input: states, weight: norm1W.buffer, weightOffset: norm1W.offset,
                bias: norm1B.buffer, biasOffset: norm1B.offset, output: normed,
                width: hidden, tokens: count, in: command)

            let qkvW = try weights.tensor(VisionMapping.Name.qkvWeight(layer))
            let qkvB = try weights.tensor(VisionMapping.Name.qkvBias(layer))
            try encoder.denseProjection(
                weights: qkvW.buffer, weightsOffset: qkvW.offset,
                bias: qkvB.buffer, biasOffset: qkvB.offset,
                input: normed, output: qkv,
                rows: 3 * hidden, cols: hidden, tokens: count, in: command)
            try encoder.visionRotary(
                qkv: qkv, angles: angles,
                patches: count, heads: heads, headDim: headDim, in: command)
            try encoder.visionAttention(
                qkv: qkv, output: attended,
                patches: count, heads: heads, headDim: headDim, in: command)

            let projW = try weights.tensor(VisionMapping.Name.projWeight(layer))
            let projB = try weights.tensor(VisionMapping.Name.projBias(layer))
            try encoder.denseProjection(
                weights: projW.buffer, weightsOffset: projW.offset,
                bias: projB.buffer, biasOffset: projB.offset,
                input: attended, output: normed,
                rows: hidden, cols: hidden, tokens: count, in: command)
            try encoder.addInPlace(
                target: states, addend: normed, size: count * hidden, in: command)

            // MLP: norm, up, GELU, down, add.
            let norm2W = try weights.tensor(VisionMapping.Name.norm2Weight(layer))
            let norm2B = try weights.tensor(VisionMapping.Name.norm2Bias(layer))
            try encoder.visionLayerNorm(
                input: states, weight: norm2W.buffer, weightOffset: norm2W.offset,
                bias: norm2B.buffer, biasOffset: norm2B.offset, output: normed,
                width: hidden, tokens: count, in: command)

            let fc1W = try weights.tensor(VisionMapping.Name.fc1Weight(layer))
            let fc1B = try weights.tensor(VisionMapping.Name.fc1Bias(layer))
            try encoder.denseProjection(
                weights: fc1W.buffer, weightsOffset: fc1W.offset,
                bias: fc1B.buffer, biasOffset: fc1B.offset,
                input: normed, output: wide,
                rows: config.intermediateSize, cols: hidden, tokens: count, in: command)
            try encoder.visionGELU(
                wide, count: count * config.intermediateSize, in: command)

            let fc2W = try weights.tensor(VisionMapping.Name.fc2Weight(layer))
            let fc2B = try weights.tensor(VisionMapping.Name.fc2Bias(layer))
            try encoder.denseProjection(
                weights: fc2W.buffer, weightsOffset: fc2W.offset,
                bias: fc2B.buffer, biasOffset: fc2B.offset,
                input: wide, output: normed,
                rows: hidden, cols: config.intermediateSize, tokens: count, in: command)
            try encoder.addInPlace(
                target: states, addend: normed, size: count * hidden, in: command)

            command.commit()
            try context.wait(command)
        }

        return try merge(states, patchCount: count)
    }

    /// Four patches into one text token.
    ///
    /// **No gather is needed.** The norm runs at 1152 over every patch, and the result buffer,
    /// read as `[patchCount / 4][4608]`, already *is* the concatenation the merger wants,
    /// because the patches were laid out in 2x2 merge order before they ever reached the tower.
    /// That is what `Qwen35VisionConfig.patchPosition` buys: the expensive permutation happens
    /// once, on the CPU, on pixels.
    private func merge(_ states: MTLBuffer, patchCount: Int) throws -> [Float] {
        let unit = config.spatialMergeSize * config.spatialMergeSize
        let tokens = patchCount / unit
        let normed = try buffer("merger", floats: patchCount * config.hiddenSize)
        let wide = try buffer("merger wide", floats: tokens * config.mergedWidth)
        let out = try buffer("tokens", floats: tokens * config.outHiddenSize)

        guard let command = context.commandQueue.makeCommandBuffer() else {
            throw MetalContext.ContextError.noCommandQueue
        }
        let normW = try weights.tensor(VisionMapping.Name.mergerNormWeight)
        let normB = try weights.tensor(VisionMapping.Name.mergerNormBias)
        try encoder.visionLayerNorm(
            input: states, weight: normW.buffer, weightOffset: normW.offset,
            bias: normB.buffer, biasOffset: normB.offset, output: normed,
            width: config.hiddenSize, tokens: patchCount, in: command)

        let fc1W = try weights.tensor(VisionMapping.Name.mergerFC1Weight)
        let fc1B = try weights.tensor(VisionMapping.Name.mergerFC1Bias)
        try encoder.denseProjection(
            weights: fc1W.buffer, weightsOffset: fc1W.offset,
            bias: fc1B.buffer, biasOffset: fc1B.offset,
            input: normed, output: wide,
            rows: config.mergedWidth, cols: config.mergedWidth, tokens: tokens, in: command)
        try encoder.visionGELU(wide, count: tokens * config.mergedWidth, in: command)

        let fc2W = try weights.tensor(VisionMapping.Name.mergerFC2Weight)
        let fc2B = try weights.tensor(VisionMapping.Name.mergerFC2Bias)
        try encoder.denseProjection(
            weights: fc2W.buffer, weightsOffset: fc2W.offset,
            bias: fc2B.buffer, biasOffset: fc2B.offset,
            input: wide, output: out,
            rows: config.outHiddenSize, cols: config.mergedWidth, tokens: tokens, in: command)
        command.commit()
        try context.wait(command)

        return Array(UnsafeBufferPointer(
            start: out.contents().bindMemory(
                to: Float.self, capacity: tokens * config.outHiddenSize),
            count: tokens * config.outHiddenSize))
    }

    // MARK: - Positions

    /// The learned grid, resampled onto this image, one vector a patch.
    private func positionEmbeddings(grid: Qwen35VisionConfig.Grid) throws -> MTLBuffer {
        let reference = VisionReference(config: config)
        let table = try weights.tensor(VisionMapping.Name.positionEmbedding)
        let side = config.positionGridSide
        let hidden = config.hiddenSize
        let out = try buffer("positions", floats: grid.patchCount * hidden)
        let destination = out.contents().bindMemory(
            to: Float.self, capacity: grid.patchCount * hidden)

        let raw = table.buffer.contents().advanced(by: table.offset)
        for index in 0..<grid.patchCount {
            let position = config.patchPosition(atSequenceIndex: index, grid: grid)
            let rows = reference.interpolationTaps(index: position.y, size: grid.height)
            let columns = reference.interpolationTaps(index: position.x, size: grid.width)
            let base = index * hidden
            for i in 0..<hidden { destination[base + i] = 0 }
            for row in rows where row.weight > 0 {
                for column in columns where column.weight > 0 {
                    let weight = Float(row.weight * column.weight)
                    let entry = (row.tap * side + column.tap) * hidden
                    for i in 0..<hidden {
                        let bits = raw.loadUnaligned(
                            fromByteOffset: (entry + i) * 2, as: UInt16.self)
                        destination[base + i] += weight * BF16.toFloat(UInt16(littleEndian: bits))
                    }
                }
            }
        }
        return out
    }

    /// The rotary angle of every patch: 18 frequencies against its row, then 18 against its
    /// column, in the layout `vision_rotary` indexes.
    private func rotaryAngles(grid: Qwen35VisionConfig.Grid) throws -> MTLBuffer {
        let reference = VisionReference(config: config)
        let half = config.headDim / 2
        let out = try buffer("angles", floats: grid.patchCount * half)
        let destination = out.contents().bindMemory(
            to: Float.self, capacity: grid.patchCount * half)
        for index in 0..<grid.patchCount {
            let position = config.patchPosition(atSequenceIndex: index, grid: grid)
            let angles = reference.rotaryAngles(y: position.y, x: position.x)
            for i in 0..<half { destination[index * half + i] = Float(angles[i]) }
        }
        return out
    }
}
