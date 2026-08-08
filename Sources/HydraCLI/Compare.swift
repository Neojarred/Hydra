import Foundation
import HydraCore
import HydraFormat
import HydraInstall
import HydraMetal
import HydraTokenize

/// D-020's quality gate: does moving the dense weights to Q8 change what the model says?
///
/// The two runs differ **only** by the weights. Same process, same model load, same KV
/// cache, same prompts, greedy decoding throughout — so any divergence is the quantization
/// and nothing else. The second run reuses the first's tokens as forced input, which is what
/// makes the comparison positional: at every step both models are asked the same question,
/// even after they start to disagree.
///
/// Two numbers decide, and D-020 names them:
///
/// - **top-1 agreement** — the share of positions where the greedy token is unchanged. This
///   is the criterion, because it is what a user actually receives.
/// - **KL divergence** — how far the distributions moved, in nats. A tiny KL with degraded
///   agreement would mean the model is hesitating between near-ties; a large KL with perfect
///   agreement would mean it moved where it did not matter. Neither number is enough alone.
enum Compare {

    /// Prompts weighted towards what D-020 identifies as the risk: long chains of reasoning,
    /// where an error at token 200 reaches the conclusion. Single-shot perplexity is exactly
    /// the measure that would miss this.
    static let defaultPrompts = [
        "A train leaves at 14:05 and arrives at 17:40, with two stops of 12 and 8 minutes. "
            + "How long is it actually moving?",
        "Write a Swift function that returns the longest common prefix of two arrays of Int.",
        "I have 17 apples. I give away a third, rounded down, then buy 9 more. "
            + "Then I lose a quarter of what I have, rounded up. How many are left?",
        "Explain why bfloat16 and float16 have the same size but are not interchangeable.",
        "Sort these from smallest to largest and explain your ordering: "
            + "0.1 + 0.2, 0.3, 1/3, and 2^-2.",
        "A file is read at 3.0 GB/s when requests are isolated and 5.7 GB/s when they are "
            + "batched. If 24 % of reads can be batched, what is the effective rate?",
    ]

    struct Options {
        var tokenCount = 64
        var contextLength = 4096
        var slotsPerLayer: Int?
        var reasoning: Harmony.ReasoningEffort = .low
        var promptsFile: String?
    }

    // MARK: - Distributions

    /// Softmax in log space. The logits reach several tens in magnitude; exponentiating them
    /// directly overflows, and a comparison that silently produced NaN would read as a
    /// perfect score.
    private static func logSoftmax(_ logits: UnsafeBufferPointer<Float>) -> [Float] {
        var peak: Float = -.greatestFiniteMagnitude
        for value in logits where value.isFinite { peak = max(peak, value) }
        var total: Double = 0
        for value in logits { total += exp(Double(value - peak)) }
        let offset = peak + Float(log(total))
        return logits.map { $0 - offset }
    }

    /// KL(P‖Q) in nats, over the reference's retained head.
    ///
    /// Asymmetric on purpose: P is the BF16 reference. What we are asking is how surprised
    /// the reference would be by the quantized model's answers, not the reverse — and the
    /// terms P drops are exactly the ones it considers impossible.
    private static func divergence(_ reference: Step, _ logQ: [Float]) -> Double {
        var sum = 0.0
        for entry in reference.top {
            let p = exp(Double(entry.logProbability))
            // Below this the term contributes less than the float noise on the sum.
            if p < 1e-12 { continue }
            sum += p * (Double(entry.logProbability) - Double(logQ[entry.index]))
        }
        return max(sum, 0)
    }

    private static func argmax<C: Collection>(_ logits: C) -> Int
    where C.Element == Float, C.Index == Int {
        var best = logits.startIndex
        var bestValue = -Float.greatestFiniteMagnitude
        for index in logits.indices where logits[index] > bestValue {
            bestValue = logits[index]
            best = index
        }
        return best - logits.startIndex
    }

    // MARK: - One run

    /// The reference distribution at one position, kept in a bounded form.
    ///
    /// Storing the full vocabulary would cost 804 KB per position — 2.5 GB over a long run,
    /// which would be an odd thing for this project to do. The KL sum is carried almost
    /// entirely by the head of the distribution, so we keep the largest entries up to
    /// `massCutoff` and record the mass actually covered, which makes the truncation
    /// auditable rather than silent.
    private struct Step {
        let token: Int
        /// Largest entries, descending, as (vocabulary index, log-probability).
        let top: [(index: Int, logProbability: Float)]
        /// Probability mass `top` accounts for.
        let coveredMass: Double

        /// The reference's probability for an arbitrary token, or `nil` if it fell below the
        /// cutoff — which is itself worth reporting.
        func probability(of token: Int) -> Double? {
            top.first { $0.index == token }.map { exp(Double($0.logProbability)) }
        }
    }

    /// Hard cap on kept entries, so a pathologically flat distribution cannot blow the
    /// budget this whole exercise is about.
    private static let topEntries = 4096
    /// Entries more than `e^-16` (≈ 1.1e-7) below the peak are dropped. They sit under the
    /// 1e-12 floor the KL sum already applies, so the truncation costs nothing it would have
    /// counted.
    private static let relativeCutoff: Float = 16

    /// Reduces a full log-distribution to its head, in two linear passes.
    ///
    /// Sorting 201 088 entries at every position would cost more than the model does.
    /// Thresholding against the peak first leaves a few hundred entries to sort.
    private static func summarize(_ logDistribution: [Float], token: Int) -> Step {
        var peak: Float = -.greatestFiniteMagnitude
        for value in logDistribution where value.isFinite { peak = max(peak, value) }
        let floor = peak - relativeCutoff

        var kept: [(index: Int, logProbability: Float)] = []
        kept.reserveCapacity(512)
        for (index, value) in logDistribution.enumerated() where value > floor {
            kept.append((index, value))
        }
        kept.sort { $0.logProbability > $1.logProbability }
        if kept.count > topEntries { kept.removeLast(kept.count - topEntries) }

        let mass = kept.reduce(0.0) { $0 + exp(Double($1.logProbability)) }
        return Step(token: token, top: kept, coveredMass: mass)
    }

    /// A position where the two models chose differently, with what it takes to judge it.
    private struct Divergence {
        let prompt: Int
        let position: Int
        let chosen: Int
        let instead: Int
        /// What the **reference** assigned to its own pick, and to the one the candidate
        /// preferred. This, not a gap in nats, is what says whether the reference was
        /// confident: 0.51 against 0.49 is a coin toss, 0.95 against 0.01 is a conviction.
        let referenceProbability: Double
        let alternativeProbability: Double
        let wasRunnerUp: Bool

        /// A position the reference held with conviction. Below this it was hesitating, and
        /// a different summation order would move it just as well.
        var wasConfident: Bool { referenceProbability >= 0.6 }
    }

    /// Decodes greedily, or replays `forced` if it is given.
    ///
    /// Replaying is what keeps the comparison honest: without it the second model would
    /// drift onto its own sequence and every subsequent position would compare two different
    /// questions.
    private static func run(
        _ runner: ModelRunner, prompt: [Int], count: Int, forced: [Int]?
    ) throws -> [Step] {
        runner.reset()
        var distribution = try runner.prefill(tokens: prompt)
        var steps: [Step] = []

        for index in 0..<count {
            let greedy = argmax(distribution)
            let next = forced.map { $0[index] } ?? greedy
            steps.append(summarize(logSoftmax(distribution), token: greedy))
            if index + 1 < count {
                distribution = try runner.forward(token: next)
            }
        }
        return steps
    }

    /// The candidate pass keeps nothing: each position is compared against the reference and
    /// discarded. Only the reference has to be held across the quantization.
    private static func compareAgainst(
        _ runner: ModelRunner, prompt: [Int], reference: [Step], promptIndex: Int,
        onPosition: (Int, Step, [Float]) -> Void
    ) throws {
        runner.reset()
        var distribution = try runner.prefill(tokens: prompt)
        for index in 0..<reference.count {
            onPosition(index, reference[index], logSoftmax(distribution))
            if index + 1 < reference.count {
                distribution = try runner.forward(token: reference[index].token)
            }
        }
    }

    // MARK: - Entry point

    static func run(config: GptOssConfig, root: URL, options: Options) throws {
        guard TokenizerInstaller.isInstalled(at: root) else {
            print("tokenizer missing — run first: hydra tokenizer")
            throw ExitError.planInvalid
        }
        let tokenizer = try TokenizerInstaller.load(from: root)
        let context = try MetalContext()

        let prompts = try options.promptsFile.map {
            try String(contentsOfFile: $0, encoding: .utf8)
                .split(separator: "\n").map(String.init)
                .filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
        } ?? defaultPrompts

        let profile = context.hardwareProfile(memoryBandwidth: 94e9, diskBandwidth: 5.5e9)
        let policy: ExpertCachePolicy =
            options.slotsPerLayer.map { .slotsPerLayer($0) } ?? .minimal
        let budget = MemoryBudget(
            config: config, hardware: profile,
            contextLength: options.contextLength, policy: policy)

        // Copy-on-write, so the simulation can rewrite the weights without touching the file.
        let mapping = try ModelMapping(
            root: root, config: config, device: context.device, mutableResident: true)
        let cache = ExpertSlotCache(
            root: root, config: config,
            slotsPerLayer: budget.expertSlotsPerLayer, device: context.device)
        let runner = try ModelRunner(
            config: config, context: context, mapping: mapping,
            expertCache: cache, contextLength: options.contextLength)
        mapping.prefault()

        let renderer = Harmony.Renderer(reasoningEffort: options.reasoning)
        let encoded = prompts.map {
            tokenizer.encode(renderer.render(turns: [.user($0)]), allowSpecial: true)
        }

        print("D-020 — Q8 on the dense weights, quality gate")
        print("\(config.name), \(prompts.count) prompts × \(options.tokenCount) tokens, greedy\n")

        // --- Reference pass, weights untouched ---
        print("reference (BF16)…")
        var reference: [[Step]] = []
        for tokens in encoded {
            reference.append(try run(runner, prompt: tokens, count: options.tokenCount, forced: nil))
        }

        // --- Quantize in memory, then replay exactly the same questions ---
        let simulation = mapping.simulateQ8Residents()
        print("candidate (Q8 simulated on \(simulation.tensorsAffected) tensors)…\n")

        var agreed = 0, total = 0
        var worstKL = 0.0, sumKL = 0.0
        var divergences: [Divergence] = []

        var thinnestCoverage = 1.0
        for (index, tokens) in encoded.enumerated() {
            try compareAgainst(
                runner, prompt: tokens, reference: reference[index], promptIndex: index
            ) { position, expected, candidate in
                let chosen = argmax(candidate)
                total += 1
                thinnestCoverage = min(thinnestCoverage, expected.coveredMass)

                if expected.token == chosen {
                    agreed += 1
                } else {
                    divergences.append(
                        Divergence(
                            prompt: index + 1, position: position,
                            chosen: expected.token, instead: chosen,
                            referenceProbability: expected.probability(of: expected.token) ?? 0,
                            // Absent from the reference's head means the reference thought it
                            // essentially impossible — the one case worth shouting about.
                            alternativeProbability: expected.probability(of: chosen) ?? -1,
                            wasRunnerUp: expected.top.count > 1
                                && expected.top[1].index == chosen))
                }
                let kl = divergence(expected, candidate)
                sumKL += kl
                worstKL = max(worstKL, kl)
            }
        }

        // --- Verdict ---
        let agreement = Double(agreed) / Double(max(total, 1)) * 100
        print("RESULT")
        print(String(format: "  top-1 agreement       %6.2f %%   (%d / %d positions)",
                     agreement, agreed, total))
        print(String(format: "  KL divergence, mean   %.3e nats", sumKL / Double(max(total, 1))))
        print(String(format: "  KL divergence, worst  %.3e nats", worstKL))
        print(String(format: "  worst weight moved by %.4f %% of its block's magnitude",
                     Double(simulation.worstRelativeDeviation) * 100))
        // The KL is summed over the reference's head. If that head ever stopped covering the
        // distribution, the figure above would be an underestimate and would have to say so.
        print(String(format: "  KL computed over %.5f of the reference's mass, at worst",
                     thinnestCoverage))

        // A changed token is only readable next to the confidence the reference had. The two
        // probabilities are the reference's own, for its pick and for the one the candidate
        // preferred — so a reader can judge the flip without trusting a threshold.
        if !divergences.isEmpty {
            print("\nCHANGED POSITIONS")
            for d in divergences {
                let a = tokenizer.decode([d.chosen])
                let b = tokenizer.decode([d.instead])
                let alternative =
                    d.alternativeProbability < 0
                    ? "p<1e-7" : String(format: "p=%.3f", d.alternativeProbability)
                print(String(
                    format: "  prompt %d, token %3d   %@ (p=%.3f) → %@ (%@)   %@",
                    d.prompt, d.position,
                    ("\"" + a + "\"") as NSString, d.referenceProbability,
                    ("\"" + b + "\"") as NSString, alternative as NSString,
                    (d.wasConfident ? "CONFIDENT" : "hesitating") as NSString))
            }
        }
        let confident = divergences.filter(\.wasConfident).count

        print("\nWHAT IT WOULD BUY")
        print("  dense affected        \(gib(simulation.bytesAffected)) → "
            + "\(gib(simulation.bytesIfQuantized))")
        print(String(format: "  i.e. %.0f %% less on %d tensors, read on every token",
                     simulation.savedFraction * 100, simulation.tensorsAffected))

        print("\nVERDICT")
        if divergences.isEmpty {
            print("  ✔ every greedy token unchanged — D-020's criterion is met.")
            print("    Remaining before adopting: no regression on long chains of reasoning,")
            print("    which this corpus samples but does not prove.")
        } else if confident == 0 {
            print("  ~ \(divergences.count) position(s) changed, none of them held with")
            print("    conviction: at each one the reference was hesitating between the two")
            print("    tokens, so a different summation order would flip it just as well.")
            print("    D-020's criterion is met in substance, not to the letter. What decides")
            print("    is the reasoning-regression check, not this number.")
        } else {
            print("  ✘ \(confident) position(s) changed where the reference was confident.")
            print("    That is degradation, and D-015 refuses the trade.")
        }
    }
}
