import Foundation
import HydraCore
import HydraMetal
import Metal

/// The recurrence, alone, at the shape prefill actually runs it.
///
/// M-060 found the token mixer is 36.3 s of a 60.6 s prefill and the experts only 10.1. Three
/// layers in four of this model are the recurrence, and `qwen_delta_rule_chunk` carries a
/// sequential loop over the chunk inside 32 threadgroups of 128 threads. On a 10-core GPU that
/// is 4096 threads holding a 512-iteration loop.
///
/// This times it against the other two halves of a linear layer, the convolution and the
/// projections, so the share is attributed rather than assumed.
enum BenchDelta {

    static func run(config: Qwen35MoeConfig, tokens: Int = 512, repeats: Int = 5) throws {
        let context = try MetalContext()
        let encoder = ForwardEncoder(context: context)

        let valueHeads = config.linearValueHeads
        let keyHeads = config.linearKeyHeads
        let keyDim = config.linearKeyHeadDim
        let valueDim = config.linearValueHeadDim
        let convDim = config.linearConvDim
        let valueSpan = valueHeads * valueDim

        func make(_ floats: Int) -> MTLBuffer? {
            context.device.makeBuffer(length: max(floats, 1) * 4, options: .storageModePrivate)
        }
        func bf16(_ count: Int) -> MTLBuffer? {
            context.device.makeBuffer(length: max(count, 1) * 2, options: .storageModeShared)
        }

        guard let state = make(valueHeads * keyDim * valueDim),
            let qkv = make(tokens * convDim),
            let qkvRaw = make(tokens * convDim),
            let window = make((config.linearConvKernel - 1) * convDim),
            let a = make(tokens * valueHeads), let b = make(tokens * valueHeads),
            let out = make(tokens * valueSpan),
            let logA = bf16(valueHeads), let dtBias = bf16(valueHeads),
            let convWeight = bf16(convDim * config.linearConvKernel)
        else { return }

        print("""
            \(config.name), one linear layer at \(tokens) tokens
            \(valueHeads) value heads, \(keyHeads) key heads, key \(keyDim), value \(valueDim)
            conv dim \(convDim), \(config.layerCount - config.fullAttentionLayerCount) \
            recurrent layers of \(config.layerCount)
            """)

        func time(_ label: String, _ body: (MTLCommandBuffer) throws -> Void) throws -> Double {
            // One untimed pass so the pipeline is built and the buffers are touched.
            if let warm = context.commandQueue.makeCommandBuffer() {
                try body(warm); context.commit(warm); try context.wait(warm)
            }
            var best = Double.infinity
            var gpu = 0.0
            for _ in 0..<repeats {
                guard let command = context.commandQueue.makeCommandBuffer() else { break }
                let start = Date()
                try body(command)
                context.commit(command)
                try context.wait(command)
                let wall = Date().timeIntervalSince(start)
                if wall < best { best = wall; gpu = command.gpuEndTime - command.gpuStartTime }
            }
            print(String(
                format: "  %-22s %7.1f ms wall, %7.1f ms GPU",
                (label as NSString).utf8String!, best * 1000, gpu * 1000))
            return best
        }

        let delta = try time("delta rule chunk") { command in
            try encoder.qwenDeltaRuleChunk(
                state: state, stateOffset: 0, qkv: qkv, a: a, b: b,
                logA: logA, dtBias: dtBias, output: out, tokens: tokens,
                valueHeads: valueHeads, keyHeads: keyHeads,
                keyDim: keyDim, valueDim: valueDim, eps: config.rmsNormEps, in: command)
        }
        let conv = try time("causal conv chunk") { command in
            try encoder.qwenCausalConvChunk(
                window: window, windowOffset: 0, input: qkvRaw,
                weight: convWeight, bias: nil, output: qkv,
                tokens: tokens, convDim: convDim,
                kernel: config.linearConvKernel, in: command)
        }
        let norm = try time("gated norm, per head") { command in
            try encoder.qwenGatedRMSNormHeads(
                input: out, weight: logA, weightOffset: 0, gate: out, output: out,
                heads: tokens * valueHeads, dim: valueDim,
                eps: config.rmsNormEps, in: command)
        }

        let recurrentLayers = config.layerCount - config.fullAttentionLayerCount
        let perChunk = (delta + conv + norm) * Double(recurrentLayers)
        print(String(
            format: """

                  one recurrent layer   %.1f ms, of which the delta rule is %.0f %%
                  x %d layers a chunk   %.1f s
                """,
            (delta + conv + norm) * 1000, delta / (delta + conv + norm) * 100,
            recurrentLayers, perChunk))
        print(String(
            format: "  a 1560-token prompt is 4 chunks at 512, so about %.1f s of recurrence",
            perChunk * 4))
    }
}
