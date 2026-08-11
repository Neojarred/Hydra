import Foundation
import HydraCore
import HydraFormat
import HydraReference
import Metal
import Testing

@testable import HydraMetal

/// The GPU decoder for MLX affine quantization, against the CPU oracle.
///
/// Checked at both bit widths, because the checkpoint uses both: 4 bits for the experts,
/// attention and embeddings, 8 for every layer's dense MLP and router. A kernel that handled
/// only the default would decode 120 tensors at the wrong width and produce matrices of half
/// the intended shape from bytes that are all present.
@Suite("MLX affine kernel")
struct MLXAffineKernelTests {

    private func makeContext() throws -> MetalContext { try MetalContext() }

    private func buffer(_ context: MetalContext, _ bytes: [UInt8]) -> MTLBuffer? {
        context.device.makeBuffer(bytes: bytes, length: max(bytes.count, 4), options: .storageModeShared)
    }

    private func floatBuffer(_ context: MetalContext, _ values: [Float]) -> MTLBuffer? {
        values.withUnsafeBytes { raw in
            context.device.makeBuffer(
                bytes: raw.baseAddress!, length: raw.count, options: .storageModeShared)
        }
    }

    private func run(_ context: MetalContext, _ body: (MTLCommandBuffer) throws -> Void) throws {
        guard let command = context.commandQueue.makeCommandBuffer() else { return }
        try body(command)
        context.commit(command)
        command.waitUntilCompleted()
    }

    /// The batched projection must equal the per-token one **bit for bit**.
    ///
    /// Prefill's whole cost is that it runs the GEMV once per token, re-reading the weights
    /// every time. The GEMM shares the read across a tile of tokens, and the only thing that
    /// makes it a safe substitution is that it leaves each token's arithmetic alone — same
    /// chunk order, same `fma` structure, same `simd_sum`. So the check is equality, not a
    /// tolerance: any drift means the accumulation order moved, and a kernel that is merely
    /// close would hide it.
    ///
    /// Token counts are chosen around the tile of eight: below it, exactly it, one past it,
    /// and a realistic chunk.
    @Test("The batched projection is bit-identical to the per-token one",
        arguments: [(4, 128, 256), (8, 96, 192)], [1, 5, 8, 9, 17, 128])
    func batchedMatchesPerToken(shape: (Int, Int, Int), tokens: Int) throws {
        let context = try makeContext()
        let encoder = ForwardEncoder(context: context)
        let (bits, rows, cols) = shape
        let groupSize = 64

        let f = fixture(rows: rows, cols: cols, bits: bits, groupSize: groupSize, seed: 909)
        var state: UInt64 = 4242
        func next() -> UInt64 {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return state >> 33
        }
        // A distinct vector per token, so a kernel reading the wrong row is caught.
        let xs = (0..<tokens).flatMap { _ in
            (0..<cols).map { _ in Float(Int(next() % 100) - 50) / 25 }
        }

        let wordBytes = f.words.withUnsafeBytes { Array($0) }
        let scaleBytes = f.scales.withUnsafeBytes { Array($0) }
        let biasBytes = f.biases.withUnsafeBytes { Array($0) }
        let padded = ForwardEncoder.paddedTokens(tokens)
        guard let words = buffer(context, wordBytes),
            let scales = buffer(context, scaleBytes),
            let biases = buffer(context, biasBytes),
            let x = floatBuffer(context, xs),
            // The transpose is part of what is under test: the batched path is only a
            // substitution for the loop if the rearrangement it needs is also correct.
            let xt = context.device.makeBuffer(
                length: cols * padded * 4, options: .storageModeShared),
            let sums = context.device.makeBuffer(
                length: max(ForwardEncoder.chunkCount(cols: cols, bits: bits), 1) * padded * 4,
                options: .storageModeShared),
            let batched = context.device.makeBuffer(
                length: tokens * rows * 4, options: .storageModeShared),
            let single = context.device.makeBuffer(
                length: tokens * rows * 4, options: .storageModeShared)
        else { return }

        try run(context) { command in
            try encoder.transposeActivations(
                input: x, inputOffset: 0, output: xt, outputOffset: 0,
                tokens: tokens, cols: cols, in: command)
            try encoder.chunkSums(
                input: xt, inputOffset: 0, output: sums, outputOffset: 0,
                tokens: tokens, cols: cols, bits: bits, in: command)
            try encoder.mlxAffineBatchedProjection(
                words: words, wordsOffset: 0, scales: scales, scalesOffset: 0,
                biases: biases, biasesOffset: 0, input: xt, inputOffset: 0,
                sums: sums, sumsOffset: 0,
                output: batched, outputOffset: 0,
                rows: rows, cols: cols, tokens: tokens, bits: bits, groupSize: groupSize,
                in: command)
            for token in 0..<tokens {
                try encoder.mlxAffineProjection(
                    words: words, wordsOffset: 0, scales: scales, scalesOffset: 0,
                    biases: biases, biasesOffset: 0,
                    input: x, inputOffset: token * cols * 4,
                    output: single, outputOffset: token * rows * 4,
                    rows: rows, cols: cols, bits: bits, groupSize: groupSize, in: command)
            }
        }

        let a = batched.contents().bindMemory(to: Float.self, capacity: tokens * rows)
        let b = single.contents().bindMemory(to: Float.self, capacity: tokens * rows)
        var mismatches = 0
        for i in 0..<(tokens * rows) where a[i] != b[i] { mismatches += 1 }
        #expect(mismatches == 0,
            "\(mismatches) of \(tokens * rows) differ at \(bits) bits, \(tokens) tokens")
    }

    /// Deterministic quantized data, plus the exact double-precision answer.
    private func fixture(
        rows: Int, cols: Int, bits: Int, groupSize: Int, seed: UInt64
    ) -> (words: [UInt32], scales: [UInt16], biases: [UInt16], x: [Float],
          expected: [Double]) {
        var state = seed
        func next() -> UInt64 {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return state >> 33
        }

        let perWord = 32 / bits
        let wordsPerRow = cols / perWord
        let groupsPerRow = cols / groupSize

        var words: [UInt32] = []
        for _ in 0..<(rows * wordsPerRow) { words.append(UInt32(truncatingIfNeeded: next())) }

        // Scales and biases are BF16 in the checkpoint, so the fixture stores the bit patterns
        // and the oracle reads them back through the same conversion the kernel uses.
        var scaleBits: [UInt16] = []
        var biasBits: [UInt16] = []
        for _ in 0..<(rows * groupsPerRow) {
            let scale = Float(Int(next() % 200) - 100) / 8000
            let bias = Float(Int(next() % 200) - 100) / 400
            scaleBits.append(BF16.fromFloat(scale))
            biasBits.append(BF16.fromFloat(bias))
        }

        let x = (0..<cols).map { _ in Float(Int(next() % 100) - 50) / 25 }

        let expected = MLXAffine.matvec(
            words: words,
            scales: scaleBits.map { Double(BF16.toFloat($0)) },
            biases: biasBits.map { Double(BF16.toFloat($0)) },
            rows: rows, cols: cols, bits: bits, groupSize: groupSize,
            x: x.map(Double.init))

        return (words, scaleBits, biasBits, x, expected)
    }

    /// The oracle is double precision and the kernel is `float`, so the bound is set by
    /// float32 summation rather than by the decoder.
    ///
    /// It is looser at 8 bits on purpose: quantized values reach 255 there against 15 at 4
    /// bits, so the partial sums over 2816 terms are an order of magnitude larger and carry
    /// proportionally more rounding. Tightening it would be asserting that `float` is more
    /// accurate than it is; loosening it further would stop catching a wrong decode, which
    /// misses by whole percent, not by parts per hundred thousand.
    private func check(
        bits: Int, rows: Int, cols: Int, groupSize: Int, seed: UInt64,
        tolerance: Double = 1e-5
    ) throws {
        let context = try makeContext()
        let encoder = ForwardEncoder(context: context)
        let f = fixture(rows: rows, cols: cols, bits: bits, groupSize: groupSize, seed: seed)

        let wordBytes = f.words.withUnsafeBytes { Array($0) }
        let scaleBytes = f.scales.withUnsafeBytes { Array($0) }
        let biasBytes = f.biases.withUnsafeBytes { Array($0) }

        guard let w = buffer(context, wordBytes), let s = buffer(context, scaleBytes),
            let b = buffer(context, biasBytes), let x = floatBuffer(context, f.x),
            let y = floatBuffer(context, [Float](repeating: 0, count: rows))
        else { return }

        try run(context) {
            try encoder.mlxAffineProjection(
                words: w, wordsOffset: 0, scales: s, scalesOffset: 0,
                biases: b, biasesOffset: 0,
                input: x, inputOffset: 0, output: y, outputOffset: 0,
                rows: rows, cols: cols, bits: bits, groupSize: groupSize, in: $0)
        }

        let got = UnsafeBufferPointer(
            start: y.contents().bindMemory(to: Float.self, capacity: rows), count: rows)
        var worst = 0.0
        for row in 0..<rows {
            let scale = max(abs(f.expected[row]), 1.0)
            worst = max(worst, abs(Double(got[row]) - f.expected[row]) / scale)
        }
        #expect(got.allSatisfy { $0.isFinite })
        #expect(worst < tolerance, "\(bits)-bit \(rows)x\(cols): worst relative deviation \(worst)")
    }

    /// The experts, attention and embeddings.
    @Test("4-bit matches the oracle at the checkpoint's shapes")
    func fourBitMatches() throws {
        // An expert's gate_proj and down_proj, at the real widths.
        try check(bits: 4, rows: 32, cols: 2816, groupSize: 64, seed: 0x51D)
        try check(bits: 4, rows: 24, cols: 704, groupSize: 64, seed: 0xA11)
    }

    /// Every layer's dense MLP and router — 120 tensors the default width would decode wrongly.
    @Test("8-bit matches the oracle at the checkpoint's shapes")
    func eightBitMatches() throws {
        try check(bits: 8, rows: 32, cols: 2816, groupSize: 64, seed: 0x8B1, tolerance: 1e-4)
        try check(bits: 8, rows: 16, cols: 704, groupSize: 64, seed: 0xC0F, tolerance: 1e-4)
    }

    /// The bias is not decoration. Zeroing it must change the answer, or the kernel is
    /// reconstructing `q · scale` and quietly discarding a per-group offset.
    @Test("The bias reaches the result")
    func biasIsApplied() throws {
        let context = try makeContext()
        let encoder = ForwardEncoder(context: context)
        let rows = 8, cols = 256, bits = 4, group = 64
        let f = fixture(rows: rows, cols: cols, bits: bits, groupSize: group, seed: 0xB1A5)

        func project(biases: [UInt16]) throws -> [Float] {
            guard let w = buffer(context, f.words.withUnsafeBytes { Array($0) }),
                let s = buffer(context, f.scales.withUnsafeBytes { Array($0) }),
                let b = buffer(context, biases.withUnsafeBytes { Array($0) }),
                let x = floatBuffer(context, f.x),
                let y = floatBuffer(context, [Float](repeating: 0, count: rows))
            else { return [] }
            try run(context) {
                try encoder.mlxAffineProjection(
                    words: w, wordsOffset: 0, scales: s, scalesOffset: 0,
                    biases: b, biasesOffset: 0,
                    input: x, inputOffset: 0, output: y, outputOffset: 0,
                    rows: rows, cols: cols, bits: bits, groupSize: group, in: $0)
            }
            return Array(UnsafeBufferPointer(
                start: y.contents().bindMemory(to: Float.self, capacity: rows), count: rows))
        }

        let withBias = try project(biases: f.biases)
        let withoutBias = try project(biases: [UInt16](repeating: 0, count: f.biases.count))
        #expect(withBias != withoutBias, "the bias made no difference to the result")
    }
}
