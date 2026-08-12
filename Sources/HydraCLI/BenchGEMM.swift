import Foundation
import HydraCore
import HydraFormat
import HydraMetal
import Metal

/// An isolated bench for the dense projection kernels, at the model's real dimensions.
///
/// Measuring end to end dilutes the signal: prefill mixes attention, router, I/O and
/// experts, and a factor of two on one kernel is lost in it. Here we time the kernel alone,
/// with the same method as for the GEMVs, several passes per command buffer, so that
/// synchronization latency does not dominate.
enum BenchGEMM {

    static func run(config: GptOssConfig) throws {
        let context = try MetalContext()
        let device = context.device
        let encoder = BatchEncoder(context: context)
        let forward = ForwardEncoder(context: context)

        let rows = config.attentionHeadCount * config.headDim   // q_proj: 4096
        let cols = config.hiddenSize                            // 2880
        let weightBytes = rows * cols * 2

        print("q_proj [\(rows) × \(cols)] BF16, \(weightBytes / 1_048_576) MiB of weights")
        let bandwidth = context.measureMemoryBandwidth()
        print(String(format: "the machine's memory bandwidth: %.0f GB/s\n", bandwidth / 1e9))

        guard let weights = device.makeBuffer(length: weightBytes, options: .storageModeShared),
            let bias = device.makeBuffer(length: rows * 2, options: .storageModeShared)
        else { return }
        // Arbitrary content: we measure throughput, not correctness (checked elsewhere).
        memset(weights.contents(), 0x3C, weightBytes)
        memset(bias.contents(), 0, rows * 2)

        print("  " + pad("kernel", 26) + pad("tokens", 8) + pad("ms", 9)
            + pad("useful GB/s", 13) + "% of BW")

        for tokens in [1, 16, 68, 128] {
            guard let x = device.makeBuffer(length: tokens * cols * 4, options: .storageModeShared),
                let y = device.makeBuffer(length: tokens * rows * 4, options: .storageModeShared)
            else { return }
            memset(x.contents(), 0, tokens * cols * 4)

            // Irreducible traffic: the weights once, plus the activations.
            let useful = Double(weightBytes + tokens * cols * 4 + tokens * rows * 4)

            func time(_ body: (MTLCommandBuffer) throws -> Void, iterations: Int = 20) throws -> Double {
                guard let warm = context.commandQueue.makeCommandBuffer() else { return 0 }
                try body(warm)
                warm.commit()
                warm.waitUntilCompleted()

                var samples: [Double] = []
                for _ in 0..<5 {
                    guard let buffer = context.commandQueue.makeCommandBuffer() else { return 0 }
                    for _ in 0..<iterations { try body(buffer) }
                    let start = Date()
                    buffer.commit()
                    buffer.waitUntilCompleted()
                    samples.append(Date().timeIntervalSince(start) / Double(iterations))
                }
                return Bench.median(samples)
            }

            func report(_ name: String, _ elapsed: Double) {
                print("  " + pad(name, 26) + pad("\(tokens)", 8)
                    + pad(String(format: "%.2f", elapsed * 1000), 9)
                    + pad(String(format: "%.1f", useful / elapsed / 1e9), 13)
                    + String(format: "%.0f %%", useful / elapsed / bandwidth * 100))
            }

            // Repeated GEMV: what prefill did before batched processing.
            let gemv = try time({ buffer in
                for token in 0..<tokens {
                    try forward.denseProjection(
                        weights: weights, weightsOffset: 0, bias: bias, biasOffset: 0,
                        input: x, inputOffset: token * cols * 4,
                        output: y, outputOffset: token * rows * 4,
                        rows: rows, cols: cols, in: buffer)
                }
            }, iterations: max(1, 20 / max(1, tokens / 8)))
            report("bf16_gemv × tokens", gemv)

            let tiled = try time { buffer in
                try encoder.denseProjection(
                    weights: weights, weightsOffset: 0, bias: bias, biasOffset: 0,
                    input: x, output: y, rows: rows, cols: cols, tokens: tokens, in: buffer)
            }
            report("bf16_gemm_tiled", tiled)
            print(String(format: "  %@×%.2f against the repeated GEMV\n",
                         String(repeating: " ", count: 26), gemv / tiled))
        }

        try layerPasses(config: config, context: context, tokens: 68)
    }

    /// Times each pass of a prefill layer separately.
    ///
    /// Timing a block's `cb1` does not say where the time goes: it holds seven passes of very
    /// different natures. Without this breakdown one optimizes at random, I lost several
    /// iterations fixing bottlenecks that were not bottlenecks.
    static func layerPasses(config: GptOssConfig, context: MetalContext, tokens: Int) throws {
        let device = context.device
        let encoder = BatchEncoder(context: context)
        let qDim = config.attentionHeadCount * config.headDim
        let kvDim = config.keyValueHeadCount * config.headDim
        let hidden = config.hiddenSize

        func buffer(_ floats: Int) -> MTLBuffer? {
            let b = device.makeBuffer(length: max(floats * 4, 256), options: .storageModeShared)
            if let b { memset(b.contents(), 0, b.length) }
            return b
        }
        func weights(_ bytes: Int) -> MTLBuffer? {
            let b = device.makeBuffer(length: max(bytes, 256), options: .storageModeShared)
            if let b { memset(b.contents(), 0x3C, b.length) }
            return b
        }

        guard let x = buffer(tokens * hidden), let normed = buffer(tokens * hidden),
            let query = buffer(tokens * qDim), let key = buffer(tokens * kvDim),
            let value = buffer(tokens * kvDim), let attn = buffer(tokens * qDim),
            let projected = buffer(tokens * hidden),
            let routerLogits = buffer(tokens * config.expertCount),
            let routerIndices = buffer(tokens * config.expertsPerToken),
            let routerWeights = buffer(tokens * config.expertsPerToken),
            let cosTable = buffer(tokens * config.headDim / 2),
            let sinTable = buffer(tokens * config.headDim / 2),
            let scale = weights(hidden * 2), let sinks = weights(qDim * 2),
            let qWeight = weights(qDim * hidden * 2), let oWeight = weights(hidden * qDim * 2),
            let kvWeight = weights(kvDim * hidden * 2),
            let routerWeight = weights(config.expertCount * hidden * 2),
            let kCache = device.makeBuffer(length: 4096 * kvDim * 2, options: .storageModeShared),
            let vCache = device.makeBuffer(length: 4096 * kvDim * 2, options: .storageModeShared)
        else { return }

        func time(_ label: String, _ body: (MTLCommandBuffer) throws -> Void) throws {
            guard let warm = context.commandQueue.makeCommandBuffer() else { return }
            try body(warm)
            warm.commit()
            warm.waitUntilCompleted()

            var samples: [Double] = []
            for _ in 0..<5 {
                guard let b = context.commandQueue.makeCommandBuffer() else { return }
                for _ in 0..<20 { try body(b) }
                let start = Date()
                b.commit()
                b.waitUntilCompleted()
                samples.append(Date().timeIntervalSince(start) / 20)
            }
            let ms = Bench.median(samples) * 1000
            print("  " + pad(label, 34)
                + pad(String(format: "%.3f ms", ms), 12)
                + String(format: "× %d layers = %.2f s", config.layerCount,
                         ms * Double(config.layerCount) / 1000))
        }

        print("PASSES OF ONE PREFILL LAYER, \(tokens) tokens\n")

        try time("rms_norm_batch") { b in
            try encoder.rmsNorm(
                input: x, scale: scale, scaleOffset: 0, output: normed,
                size: hidden, tokens: tokens, eps: 1e-5, in: b)
        }
        try time("q_proj  [4096 × 2880]") { b in
            try encoder.denseProjection(
                weights: qWeight, weightsOffset: 0, bias: nil, biasOffset: 0,
                input: normed, output: query, rows: qDim, cols: hidden, tokens: tokens, in: b)
        }
        try time("k_proj + v_proj [512 × 2880]") { b in
            for out in [key, value] {
                try encoder.denseProjection(
                    weights: kvWeight, weightsOffset: 0, bias: nil, biasOffset: 0,
                    input: normed, output: out, rows: kvDim, cols: hidden,
                    tokens: tokens, in: b)
            }
        }
        try time("o_proj  [2880 × 4096]") { b in
            try encoder.denseProjection(
                weights: oWeight, weightsOffset: 0, bias: nil, biasOffset: 0,
                input: attn, output: projected, rows: hidden, cols: qDim, tokens: tokens, in: b)
        }
        try time("rope_apply_batch ×2") { b in
            try encoder.applyRoPE(
                vector: query, cos: cosTable, sin: sinTable,
                heads: config.attentionHeadCount, headDim: config.headDim,
                tokens: tokens, in: b)
            try encoder.applyRoPE(
                vector: key, cos: cosTable, sin: sinTable,
                heads: config.keyValueHeadCount, headDim: config.headDim, tokens: tokens, in: b)
        }
        try time("kv_cache_write_batch") { b in
            try encoder.writeKeyValue(
                key: key, value: value, keyCache: kCache, valueCache: vCache,
                kvHeads: config.keyValueHeadCount, headDim: config.headDim,
                tokens: tokens, firstPosition: 0, ringSize: 0, in: b)
        }
        try time("attention_prefill") { b in
            try encoder.attention(
                query: query, keyCache: kCache, valueCache: vCache,
                sinks: sinks, sinksOffset: 0, output: attn,
                qHeads: config.attentionHeadCount, kvHeads: config.keyValueHeadCount,
                headDim: config.headDim, tokens: tokens,
                ringSize: 0, firstPosition: 0, slidingWindow: 0, smScale: 0.125, in: b)
        }
        try time("router [32 × 2880] + top-k") { b in
            try encoder.denseProjection(
                weights: routerWeight, weightsOffset: 0, bias: nil, biasOffset: 0,
                input: normed, output: routerLogits,
                rows: config.expertCount, cols: hidden, tokens: tokens, in: b)
            try encoder.routerTopK(
                logits: routerLogits, indices: routerIndices, weights: routerWeights,
                expertCount: config.expertCount, topK: config.expertsPerToken,
                tokens: tokens, in: b)
        }
    }
}
