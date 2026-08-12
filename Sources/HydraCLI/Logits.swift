import Foundation
import HydraCore
import HydraFormat
import HydraInstall
import HydraMetal
import HydraTokenize

/// What the model actually believes, one position at a time.
///
/// Written the first time real Gemma weights produced a plausible-looking run, 6 tok/s, 99 %
/// cache hits, that emitted the same unused token sixty times. Throughput told us nothing;
/// the distribution tells us everything. A flat distribution means the head read garbage, a
/// spike on one token means the state collapsed, and a sensible ranking means the bug is
/// downstream in the prompt or the parser.
enum Logits {

    static func run(
        model: any ModelDescriptor, root: URL, prompt: String,
        contextLength: Int, topK: Int, raw: Bool, trace: Bool,
        reasoning: ReasoningLevel = .off
    ) throws {
        let tokenizer = try TokenizerInstaller.load(
            from: root, architecture: model.architecture)
        let context = try MetalContext()
        let profile = context.hardwareProfile(memoryBandwidth: 94e9, diskBandwidth: 5.5e9)
        let budget = MemoryBudget(
            config: model, hardware: profile, contextLength: contextLength, policy: .minimal)
        let mapping = try ModelMapping(root: root, model: model, device: context.device)
        let cache = ExpertSlotCache(
            root: root, model: model,
            slotsPerLayer: budget.expertSlotsPerLayer, device: context.device)
        let runner = try ModelRuntime.makeRunner(
            model: model, context: context, mapping: mapping,
            expertCache: cache, contextLength: contextLength)

        // `--raw` skips the chat template, to separate "the model is broken" from "the prompt
        // is written wrong". They look identical from the output.
        let format = ConversationFormats.format(for: runner.architecture)
        let rendered = raw
            ? prompt
            : format.render(turns: [.user(prompt)], settings: PromptSettings(reasoning: reasoning))
        let tokens = tokenizer.encode(rendered, allowSpecial: true)

        print("format: \(raw ? "raw (no template)" : format.name)")
        print("prompt: \(tokens.count) tokens, reasoning \(reasoning.rawValue)")
        for id in tokens {
            print("  \(pad(String(id), 8)) \(escape(tokenizer.decode([id])))")
        }

        // Per-stage statistics, so a divergence names the operation that produced it.
        if trace, let gemma = runner as? Gemma4ModelRunner {
            print("\nper stage")
            var reported = false
            gemma.stageObserver = { layer, stage, values in
                // Only the first stage that goes wrong is worth printing: everything after it
                // is downstream of a value that is already NaN.
                guard !reported else { return }
                let bad = values.filter { !$0.isFinite }.count
                let finite = values.filter { $0.isFinite }
                if stage == "expertIndices" {
                    let out = values.filter { $0 < 0 || $0 >= Float(gemma.config.expertCount) }
                    if !out.isEmpty {
                        print("  layer \(layer) \(stage): OUT OF RANGE \(values)")
                        reported = true
                    }
                    return
                }
                if bad > 0 {
                    print("  ✘ layer \(layer), first non-finite at stage «\(stage)» "
                        + "(\(bad) of \(values.count))")
                    reported = true
                    return
                }
                if layer >= 25 {
                    let magnitude = finite.map { abs($0) }.max() ?? 0
                    print(String(
                        format: "  layer %2d %-15@ min %14.4f  max %14.4f  |max| %14.4f",
                        layer, stage as NSString,
                        finite.min() ?? 0, finite.max() ?? 0, magnitude))
                    return
                }
                if stage == "hidden", layer % 6 == 0 || layer >= 24 {
                    print(String(
                        format: "  layer %2d %-13@ min %10.3f  max %10.3f",
                        layer, stage as NSString,
                        finite.min() ?? 0, finite.max() ?? 0))
                }
            }
        }

        let distribution = try runner.prefill(tokens: tokens)
        if let gemma = runner as? Gemma4ModelRunner { gemma.stageObserver = nil }

        // The shape of the distribution, before its contents. A head that read the wrong bytes
        // produces values that are flat, enormous, or not finite, each visible here and in
        // none of the per-operator tests, which use a 256-entry vocabulary.
        var minimum = Float.greatestFiniteMagnitude
        var maximum = -Float.greatestFiniteMagnitude
        var sum = 0.0
        var nonFinite = 0
        for value in distribution {
            guard value.isFinite else { nonFinite += 1; continue }
            minimum = min(minimum, value)
            maximum = max(maximum, value)
            sum += Double(value)
        }
        let mean = sum / Double(max(distribution.count - nonFinite, 1))
        print("\nlogits over \(distribution.count) entries")
        print(String(format: "  min %.4f  max %.4f  mean %.4f", minimum, maximum, mean))
        if nonFinite > 0 { print("  ✘ \(nonFinite) non-finite values") }
        if maximum - minimum < 1e-3 {
            print("  ✘ the distribution is flat, the head produced no signal")
        }

        let best = TokenSampler.largestIndices(distribution, count: topK)
        print("\ntop \(topK)")
        for id in best {
            print(String(
                format: "  %@ %8.4f  %@",
                pad(String(id), 8), distribution[id], escape(tokenizer.decode([id]))))
        }
    }

    private static func pad(_ s: String, _ width: Int) -> String {
        s.count >= width ? s : String(repeating: " ", count: width - s.count) + s
    }

    /// Control characters and the metaspace made visible: a leading U+2581 is the difference
    /// between a token that starts a word and one that continues it.
    private static func escape(_ s: String) -> String {
        var out = ""
        for scalar in s.unicodeScalars {
            switch scalar {
            case "\n": out += "\\n"
            case "\t": out += "\\t"
            case "\u{2581}": out += "▁"
            default: out.unicodeScalars.append(scalar)
            }
        }
        return "«\(out)»"
    }
}
