import Foundation
import HydraCore
import HydraMetal
import Metal

/// An isolated bench for the quantized GEMV, at the shapes Gemma actually decodes with.
///
/// Two hypotheses about this kernel have now been wrong, both aimed at a gap inferred from an
/// aggregate: `cb1` moves its bytes at about 20 GB/s against a 95 GB/s ceiling, so the kernel
/// filling `cb1` looked four times slower than the machine allows. But `cb1` is not the kernel
/// — it is also the norms, the rotary, the attention and the router, none of which move enough
/// bytes to be measured that way, and all of which are charged into the same average.
///
/// So measure the kernel alone. If it is already near the ceiling at these shapes, the time is
/// somewhere else and no amount of work on it will show up.
enum BenchGEMV {

    /// (name, rows, cols, bits) — Gemma 4 26B-A4B, MLX 4-bit.
    private static func shapes(_ c: Gemma4MLXConfig) -> [(String, Int, Int, Int)] {
        // The sliding geometry, which is five layers in six.
        let heads = c.base.attentionHeadCount * c.base.slidingHeadDim
        let kv = c.base.slidingKeyValueHeadCount * c.base.slidingHeadDim
        return [
            ("q_proj", heads, c.base.hiddenSize, c.quantBits),
            ("k_proj", kv, c.base.hiddenSize, c.quantBits),
            ("o_proj", c.base.hiddenSize, heads, c.quantBits),
            ("mlp.gate (dense)", c.base.intermediateSize, c.base.hiddenSize, c.denseBits),
            ("mlp.down (dense)", c.base.hiddenSize, c.base.intermediateSize, c.denseBits),
            ("expert.gate", c.base.moeIntermediateSize, c.base.hiddenSize, c.quantBits),
            ("expert.down", c.base.hiddenSize, c.base.moeIntermediateSize, c.quantBits),
            ("head (tied embed)", c.base.vocabSize, c.base.hiddenSize, c.quantBits),
        ]
    }

    /// What a command buffer and a dispatch cost before any kernel runs.
    ///
    /// The isolated kernels sum to about 22 ms of GPU work a token, but GPU busy measures
    /// 48-59. The difference has to be fixed overhead, and a token pays it about 31 times for
    /// command buffers and 660 times for dispatches. Which of the two dominates decides
    /// whether there is anything left to win by restructuring rather than by writing faster
    /// kernels, so it is worth knowing rather than assuming.
    ///
    /// Measured by encoding N copies of a kernel with nothing to do and reading the GPU's own
    /// clock: the slope is a dispatch, the intercept is the buffer.
    static func runOverhead() throws {
        let context = try MetalContext()
        let device = context.device
        let forward = ForwardEncoder(context: context)
        guard let tiny = device.makeBuffer(length: 64, options: .storageModeShared) else { return }

        print("\n  dispatches per command buffer → GPU time (a 16-element copy, so ~no work)")
        var points: [(Int, Double)] = []
        for count in [1, 2, 4, 8, 16, 32, 64, 128] {
            var best = Double.greatestFiniteMagnitude
            for _ in 0..<20 {
                guard let cb = context.commandQueue.makeCommandBuffer() else { return }
                for _ in 0..<count {
                    try forward.copy(
                        into: tiny, destinationOffset: 0, from: tiny, sourceOffset: 0,
                        size: 16, in: cb)
                }
                context.commit(cb)
                cb.waitUntilCompleted()
                best = min(best, cb.gpuEndTime - cb.gpuStartTime)
            }
            points.append((count, best))
            print("  " + pad("\(count)", 8) + String(format: "%.1f µs", best * 1e6))
        }

        // Two points far apart give the split without needing a fit.
        if let low = points.first, let high = points.last, high.0 > low.0 {
            let perDispatch = (high.1 - low.1) / Double(high.0 - low.0)
            let perBuffer = low.1 - perDispatch * Double(low.0)
            print(String(format: "\n  a dispatch costs %.1f µs · a command buffer costs %.1f µs",
                perDispatch * 1e6, perBuffer * 1e6))
            print(String(format:
                "  a token pays ~660 dispatches and ~31 buffers: %.1f ms + %.1f ms",
                perDispatch * 660 * 1000, perBuffer * 31 * 1000))
        }
    }

    /// The rest of a decode layer: everything that is not a projection.
    ///
    /// The projections account for about 20 ms of a token's 60 ms on the GPU. The other 40 is
    /// here, in kernels nobody has timed — the norms, the rotary, the attention, the router,
    /// the copies. M-040 is the argument for measuring them rather than reasoning about them.
    static func runLayer(config c: Gemma4MLXConfig) throws {
        let context = try MetalContext()
        let device = context.device
        let forward = ForwardEncoder(context: context)
        let base = c.base

        let heads = base.attentionHeadCount
        let kvHeads = base.slidingKeyValueHeadCount
        let headDim = base.slidingHeadDim
        let hidden = base.hiddenSize
        let queryDim = heads * headDim

        func buffer(_ floats: Int) -> MTLBuffer? {
            let b = device.makeBuffer(length: max(floats, 1) * 4, options: .storageModeShared)
            if let b { memset(b.contents(), 0x3C, b.length) }
            return b
        }
        // The sliding window is what a decode step actually attends over.
        let window = base.slidingWindow
        guard let query = buffer(queryDim), let out = buffer(queryDim),
            let keys = buffer(kvHeads * headDim * window),
            let values = buffer(kvHeads * headDim * window),
            let sinks = buffer(heads),
            let vec = buffer(hidden), let vec2 = buffer(hidden),
            let scale = buffer(hidden), let big = buffer(base.intermediateSize),
            let big2 = buffer(base.intermediateSize),
            let cosTable = buffer(headDim), let sinTable = buffer(headDim)
        else { return }

        print("\n  " + pad("kernel", 26) + pad("shape", 22) + pad("µs", 9) + "× 30 layers, ms")

        func time(_ name: String, _ shape: String, _ passes: Int = 64,
                  _ body: (MTLCommandBuffer) throws -> Void) rethrows {
            var best = Double.greatestFiniteMagnitude
            for _ in 0..<5 {
                guard let cb = context.commandQueue.makeCommandBuffer() else { return }
                for _ in 0..<passes { try body(cb) }
                context.commit(cb)
                cb.waitUntilCompleted()
                best = min(best, (cb.gpuEndTime - cb.gpuStartTime) / Double(passes))
            }
            print("  " + pad(name, 26) + pad(shape, 22)
                + pad(String(format: "%.0f", best * 1e6), 9)
                + String(format: "%.2f", best * 30 * 1000))
        }

        try time("attention_decode", "\(heads)h × \(window) keys") { cb in
            try forward.attention(
                query: query, queryOffset: 0, keyCache: keys, valueCache: values,
                sinks: sinks, sinksOffset: 0, output: out, outputOffset: 0,
                qHeads: heads, kvHeads: kvHeads, headDim: headDim, keyCount: window,
                ringSize: window, startPosition: 0, smScale: 1.0, in: cb)
        }
        try time("rms_norm_heads (q)", "\(heads) × \(headDim)") { cb in
            try forward.rmsNormHeads(
                vector: query, scale: scale, scaleOffset: 0, heads: heads, headDim: headDim,
                eps: base.rmsNormEps, in: cb)
        }
        try time("rms_norm", "\(hidden)") { cb in
            try forward.rmsNorm(
                input: vec, inputOffset: 0, scale: scale, scaleOffset: 0,
                output: vec2, outputOffset: 0, size: hidden, eps: base.rmsNormEps, in: cb)
        }
        try time("fused_norm_add_copy", "\(hidden)") { cb in
            try forward.fusedNormAddCopy(
                input: vec, scale: scale, scaleOffset: 0, hidden: vec2, residual: out,
                size: hidden, eps: base.rmsNormEps, in: cb)
        }
        try time("apply_rope", "\(heads) × \(headDim)") { cb in
            try forward.applyRoPE(
                vector: query, vectorOffset: 0, cos: cosTable, sin: sinTable, tableOffset: 0,
                heads: heads, headDim: headDim, in: cb)
        }
        try time("gelu_mul", "\(base.intermediateSize)") { cb in
            try forward.geluMultiply(
                gate: big, up: big2, output: big, size: base.intermediateSize, in: cb)
        }
        try time("add_in_place", "\(hidden)") { cb in
            try forward.addInPlace(
                target: vec, targetOffset: 0, addend: vec2, addendOffset: 0, size: hidden,
                in: cb)
        }
        try time("copy", "\(hidden)") { cb in
            try forward.copy(
                into: vec2, destinationOffset: 0, from: vec, sourceOffset: 0, size: hidden,
                in: cb)
        }
    }

    static func run(config: Gemma4MLXConfig) throws {
        let context = try MetalContext()
        let device = context.device
        let forward = ForwardEncoder(context: context)

        let ceiling = context.measureMemoryBandwidth()
        print(String(format: "the machine's memory bandwidth: %.0f GB/s\n", ceiling / 1e9))
        print("  " + pad("shape", 20) + pad("rows", 9) + pad("cols", 7) + pad("bits", 6)
            + pad("MiB", 8) + pad("µs", 9) + pad("GB/s", 8) + "% of BW")

        for (name, rows, cols, bits) in shapes(config) {
            let layout = MLXAffineLayout(
                bits: bits, groupSize: config.groupSize, rows: rows, cols: cols)
            let wordBytes = rows * layout.wordsPerRow * 4
            let groupBytes = rows * layout.groupsPerRow * 2
            let bytes = wordBytes + 2 * groupBytes

            guard let words = device.makeBuffer(length: wordBytes, options: .storageModeShared),
                let scales = device.makeBuffer(length: groupBytes, options: .storageModeShared),
                let biases = device.makeBuffer(length: groupBytes, options: .storageModeShared),
                let x = device.makeBuffer(length: cols * 4, options: .storageModeShared),
                let y = device.makeBuffer(length: rows * 4, options: .storageModeShared)
            else { continue }
            // Arbitrary content: this measures throughput, not correctness.
            memset(words.contents(), 0x11, wordBytes)
            memset(scales.contents(), 0x3C, groupBytes)
            memset(biases.contents(), 0, groupBytes)
            memset(x.contents(), 0x3C, cols * 4)

            // Several passes to one command buffer, so submission latency is not the number.
            // The GPU's own clock, not the wall: the wall includes the CPU's encoding.
            let passes = max(1, min(64, 64 * 1_048_576 / max(bytes, 1)))
            var best = Double.greatestFiniteMagnitude
            for _ in 0..<5 {
                guard let cb = context.commandQueue.makeCommandBuffer() else { continue }
                for _ in 0..<passes {
                    try forward.mlxAffineProjection(
                        words: words, wordsOffset: 0, scales: scales, scalesOffset: 0,
                        biases: biases, biasesOffset: 0, input: x, inputOffset: 0,
                        output: y, outputOffset: 0, rows: rows, cols: cols,
                        bits: bits, groupSize: config.groupSize, in: cb)
                }
                context.commit(cb)
                cb.waitUntilCompleted()
                best = min(best, (cb.gpuEndTime - cb.gpuStartTime) / Double(passes))
            }

            let rate = Double(bytes) / best
            print("  " + pad(name, 20) + pad("\(rows)", 9) + pad("\(cols)", 7)
                + pad("\(bits)", 6)
                + pad(String(format: "%.1f", Double(bytes) / 1_048_576), 8)
                + pad(String(format: "%.0f", best * 1e6), 9)
                + pad(String(format: "%.0f", rate / 1e9), 8)
                + String(format: "%.0f %%", rate / ceiling * 100))
        }
    }
}
