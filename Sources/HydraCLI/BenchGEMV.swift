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
                cb.commit()
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
