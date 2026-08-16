import Foundation
import HydraCore
import HydraFormat
import HydraMetal
import Metal

/// End-to-end decode at a long context, with and without the attention key split.
///
/// `bench-attention` measures the kernel alone and says it is 39 % faster at 21k. That number
/// is real and it is not the answer: a decoded token is also expert I/O, the mixture and the LM
/// head, and M-067 put attention at roughly a third of Qwen's token at that length. What the
/// user gets is the fraction of a fraction, and the only honest way to know it is to measure it.
///
/// **One prefill, two arms.** Prefilling 21k tokens costs about fifteen minutes; doing it twice
/// would double the run and, worse, would put the two arms minutes apart on a machine that
/// drifts, which is the confound M-063 exists to prevent. So the context is built once and the
/// arms alternate over it, a few tokens at a time, flipping order every round.
///
/// The tokens generated do lengthen the context as the run proceeds, by about 0.3 % of it, which
/// is far below the difference being measured and is shared by both arms anyway.
enum BenchLongDecode {

    static func run(
        model: any ModelDescriptor, root: URL, contextTokens: Int,
        rounds: Int = 6, tokensPerSample: Int = 4, slotsPerLayer: Int? = nil,
        contextLength: Int = 32768
    ) throws {
        let context = try MetalContext()
        let profile = context.hardwareProfile(memoryBandwidth: 94e9, diskBandwidth: 5.5e9)
        let policy: ExpertCachePolicy = slotsPerLayer.map { .slotsPerLayer($0) } ?? .balanced
        let budget = MemoryBudget(
            config: model, hardware: profile, contextLength: contextLength, policy: policy)

        let mapping = try ModelMapping(root: root, model: model, device: context.device)
        let expertCache = ExpertSlotCache(
            root: root, model: model, slotsPerLayer: budget.expertSlotsPerLayer,
            device: context.device)
        let runner = try ModelRuntime.makeRunner(
            model: model, context: context, mapping: mapping,
            expertCache: expertCache, contextLength: contextLength)
        mapping.prefault()

        print("""
            \(model.name), \(budget.expertSlotsPerLayer)/\(model.expertCount) experts cached
            building a \(contextTokens)-token context, then \(rounds) alternating pairs of \
            \(tokensPerSample) tokens
            """)

        // Identifiers rather than text: this measures the decode path, and what the model
        // believes about a nonsense prompt costs exactly what a sensible one costs.
        let filler = (0..<contextTokens).map { 1000 + ($0 * 7919) % 20000 }
        let prefillStart = Date()
        _ = try runner.prefill(tokens: filler)
        print(String(
            format: "  prefilled %d tokens in %.0f s (%.0f tokens/s)\n",
            contextTokens, Date().timeIntervalSince(prefillStart),
            Double(contextTokens) / Date().timeIntervalSince(prefillStart)))

        func decode(_ count: Int) throws -> Double {
            let start = Date()
            for _ in 0..<count {
                _ = try runner.forward(token: 4242, needsLogits: false)
            }
            return Double(count) / Date().timeIntervalSince(start)
        }

        // Untimed, so neither arm pays for the first token after the prefill.
        for enabled in [false, true] {
            ForwardEncoder.splitAttentionEnabled = enabled
            _ = try decode(tokensPerSample)
        }

        var singleRates: [Double] = []
        var splitRates: [Double] = []
        print("  round   position   one head/group      split      change")
        for round in 0..<rounds {
            var line = ""
            for enabled in (round % 2 == 0 ? [false, true] : [true, false]) {
                ForwardEncoder.splitAttentionEnabled = enabled
                let rate = try decode(tokensPerSample)
                if enabled { splitRates.append(rate) } else { singleRates.append(rate) }
            }
            ForwardEncoder.splitAttentionEnabled = true
            let single = singleRates.last ?? 0
            let split = splitRates.last ?? 0
            line = String(
                format: "  %5d   %8d   %10.2f tok/s %8.2f    %+6.1f %%",
                round + 1, runner.position, single, split,
                (split - single) / max(single, 1e-9) * 100)
            print(line)
        }

        func median(_ v: [Double]) -> Double {
            let s = v.sorted()
            guard !s.isEmpty else { return 0 }
            return s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2
        }
        let single = median(singleRates)
        let split = median(splitRates)
        print(String(
            format: """

                  median  one threadgroup a head  %.2f tok/s
                          keys split across eight %.2f tok/s
                          change                  %+.1f %%
                """,
            single, split, (split - single) / max(single, 1e-9) * 100))
    }
}
