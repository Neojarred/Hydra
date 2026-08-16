import Foundation
import HydraCore
import HydraMetal
import Metal

/// Decode attention alone, one threadgroup a head against keys split across threadgroups.
///
/// M-067 found this kernel about twelve times off the rate the weight projections achieve, and
/// put a cost on it: at 21k tokens Qwen's ten unbounded attention layers add 128 ms a token.
/// The diagnosis was the launch shape. `attention_decode` dispatches one threadgroup a query
/// head, sixteen for Qwen, on a GPU with ten cores that wants a few hundred.
///
/// Measured here in isolation rather than end to end, for the reason M-064 gives: a decode step
/// mixes attention with expert I/O, the mixture and the head, and the kernel is a tenth of it,
/// so an end-to-end number would be mostly noise about the SSD. The end-to-end effect is worth
/// having too, but only once the isolated one says there is something to find.
///
/// The two arms **alternate inside one process**, which is the protocol M-063 exists to enforce.
/// Two runs minutes apart on a machine that is warming up differ by more than this change does.
enum BenchAttention {

    /// GPU-busy time, from the command buffer's own timestamps rather than from the wall clock.
    private static func gpuSeconds(_ buffer: MTLCommandBuffer) -> Double {
        max(buffer.gpuEndTime - buffer.gpuStartTime, 0)
    }

    private static func median(_ values: [Double]) -> Double {
        let sorted = values.sorted()
        guard !sorted.isEmpty else { return 0 }
        return sorted.count % 2 == 1
            ? sorted[sorted.count / 2]
            : (sorted[sorted.count / 2 - 1] + sorted[sorted.count / 2]) / 2
    }

    /// One layer's worth of decode attention, timed on the GPU.
    ///
    /// Repeated `perSample` times inside one command buffer: a single dispatch at these shapes
    /// is tens of microseconds, close enough to the timestamp resolution that a lone one would
    /// be measuring the clock.
    private static func time(
        encoder: ForwardEncoder, context: MetalContext,
        query: MTLBuffer, keys: MTLBuffer, values: MTLBuffer, sinks: MTLBuffer,
        output: MTLBuffer, qHeads: Int, kvHeads: Int, headDim: Int, keyCount: Int,
        perSample: Int
    ) throws -> Double {
        guard let commandBuffer = context.commandQueue.makeCommandBuffer() else { return 0 }
        for _ in 0..<perSample {
            try encoder.attention(
                query: query, queryOffset: 0, keyCache: keys, valueCache: values,
                sinks: sinks, sinksOffset: 0, output: output, outputOffset: 0,
                qHeads: qHeads, kvHeads: kvHeads, headDim: headDim, keyCount: keyCount,
                ringSize: 0, startPosition: 0, smScale: 0.125, in: commandBuffer)
        }
        context.commit(commandBuffer)
        try context.wait(commandBuffer)
        return gpuSeconds(commandBuffer) / Double(perSample)
    }

    static func run(
        qHeads: Int = 16, kvHeads: Int = 2, headDim: Int = 128,
        keyCounts: [Int] = [512, 1024, 2048, 4096, 8192, 16384, 21504],
        pairs: Int = 5, perSample: Int = 32
    ) throws {
        let context = try MetalContext()
        let encoder = ForwardEncoder(context: context)
        let capacity = keyCounts.max() ?? 1024

        // Key and value bytes a *layer* at each length, which is the quantity the kernel is
        // bound by and the one that makes the GB/s column meaningful.
        let bytesPerKey = kvHeads * headDim * 2 * 2

        guard let query = context.device.makeBuffer(
                length: qHeads * headDim * 4, options: .storageModeShared),
            let keys = context.device.makeBuffer(
                length: capacity * kvHeads * headDim * 2, options: .storageModeShared),
            let values = context.device.makeBuffer(
                length: capacity * kvHeads * headDim * 2, options: .storageModeShared),
            let sinks = context.device.makeBuffer(length: qHeads * 2, options: .storageModeShared),
            let output = context.device.makeBuffer(
                length: qHeads * headDim * 4, options: .storageModeShared)
        else { return }

        // Anything but zeros: a cache of zeros makes every logit identical, and a softmax over
        // identical logits is a different amount of work for the exponential than a real one.
        let queryValues = query.contents().bindMemory(to: Float.self, capacity: qHeads * headDim)
        for i in 0..<(qHeads * headDim) { queryValues[i] = Float(i % 17) * 0.01 - 0.08 }
        let keyValues = keys.contents().bindMemory(
            to: Float16.self, capacity: capacity * kvHeads * headDim)
        let valueValues = values.contents().bindMemory(
            to: Float16.self, capacity: capacity * kvHeads * headDim)
        for i in 0..<(capacity * kvHeads * headDim) {
            keyValues[i] = Float16(Float(i % 23) * 0.01 - 0.11)
            valueValues[i] = Float16(Float(i % 19) * 0.01 - 0.09)
        }
        memset(sinks.contents(), 0, sinks.length)

        print("""
            decode attention, \(qHeads) query heads over \(kvHeads) key/value heads, \
            \(headDim) wide
            \(pairs) alternating pairs, \(perSample) dispatches a sample, GPU-busy time
            """)
        print("\n   keys  chunks   one head/group      split        change      GB/s single → split")

        for keyCount in keyCounts {
            var singleTimes: [Double] = []
            var splitTimes: [Double] = []

            // One untimed pass each, so neither arm pays for the other's first pipeline build.
            for enabled in [false, true] {
                ForwardEncoder.splitAttentionEnabled = enabled
                _ = try time(
                    encoder: encoder, context: context, query: query, keys: keys, values: values,
                    sinks: sinks, output: output, qHeads: qHeads, kvHeads: kvHeads,
                    headDim: headDim, keyCount: keyCount, perSample: perSample)
            }

            for pair in 0..<pairs {
                // The order flips every pair, so a machine that is drifting in one direction
                // does not hand the advantage to whichever arm is measured second.
                let order = pair % 2 == 0 ? [false, true] : [true, false]
                for enabled in order {
                    ForwardEncoder.splitAttentionEnabled = enabled
                    let seconds = try time(
                        encoder: encoder, context: context, query: query, keys: keys,
                        values: values, sinks: sinks, output: output, qHeads: qHeads,
                        kvHeads: kvHeads, headDim: headDim, keyCount: keyCount,
                        perSample: perSample)
                    if enabled { splitTimes.append(seconds) } else { singleTimes.append(seconds) }
                }
            }
            ForwardEncoder.splitAttentionEnabled = true

            let single = median(singleTimes)
            let split = median(splitTimes)
            let bytes = Double(keyCount * bytesPerKey)
            let chunks = ForwardEncoder.attentionChunks(keyCount: keyCount)
            print(String(
                format: "  %6d  %6@  %9.1f µs  %9.1f µs   %+6.1f %%   %6.1f → %6.1f",
                keyCount, chunks.map(String.init) ?? "none",
                single * 1e6, split * 1e6,
                (single - split) / max(single, 1e-12) * 100,
                bytes / max(single, 1e-12) / 1e9, bytes / max(split, 1e-12) / 1e9))
        }

        // --- Is the automatic chunk count the right one? ---
        //
        // The rule aims at 256 threadgroups because that is a plausible number for a ten-core
        // GPU, which is a guess, and a guess inside a shipped heuristic is the kind of thing
        // M-063 punishes. Swept here at the long lengths that matter, interleaved the same way.
        print("\n  chunk sweep, µs a dispatch, same interleaving")
        let sweep = [2, 4, 8, 16, 32]
        print("   keys " + sweep.map { String(format: "%9d", $0) }.joined())
        for keyCount in keyCounts {
            var times: [Int: [Double]] = [:]
            for forced in sweep {
                ForwardEncoder.attentionChunkOverride = forced
                _ = try time(
                    encoder: encoder, context: context, query: query, keys: keys, values: values,
                    sinks: sinks, output: output, qHeads: qHeads, kvHeads: kvHeads,
                    headDim: headDim, keyCount: keyCount, perSample: perSample)
            }
            for round in 0..<pairs {
                for forced in (round % 2 == 0 ? sweep : sweep.reversed()) {
                    ForwardEncoder.attentionChunkOverride = forced
                    let seconds = try time(
                        encoder: encoder, context: context, query: query, keys: keys,
                        values: values, sinks: sinks, output: output, qHeads: qHeads,
                        kvHeads: kvHeads, headDim: headDim, keyCount: keyCount,
                        perSample: perSample)
                    times[forced, default: []].append(seconds)
                }
            }
            ForwardEncoder.attentionChunkOverride = nil
            print(String(format: "  %5d ", keyCount)
                + sweep.map { String(format: "%9.1f", median(times[$0] ?? [0]) * 1e6) }.joined())
        }

        let peak = context.measureMemoryBandwidth()
        print(String(format: "\n  this machine reads at %.0f GB/s, measured now", peak / 1e9))
    }
}
