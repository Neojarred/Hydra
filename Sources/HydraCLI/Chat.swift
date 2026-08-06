import Foundation
import HydraCore
import HydraFormat
import HydraInstall
import HydraMetal
import HydraTokenize

/// Conversation avec un modèle installé, au format Harmony.
enum Chat {

    struct Options {
        var tokenCount = 512
        var contextLength = 4096
        var slotsPerLayer: Int?
        var temperature: Float = 1.0
        var topP: Float = 1.0
        var reasoning: Harmony.ReasoningEffort = .medium
        var showAnalysis = false
        var instructions: String?
    }

    static func run(config: GptOssConfig, root: URL, prompt: String, options: Options) throws {
        guard TokenizerInstaller.isInstalled(at: root) else {
            print("tokeniseur absent — lancer d'abord : hydra tokenizer")
            throw ExitError.planInvalid
        }

        var start = Date()
        let tokenizer = try TokenizerInstaller.load(from: root)
        let tokenizerTime = Date().timeIntervalSince(start)

        let context = try MetalContext()
        let profile = context.hardwareProfile(memoryBandwidth: 94e9, diskBandwidth: 5.5e9)
        let policy: ExpertCachePolicy = options.slotsPerLayer.map { .slotsPerLayer($0) } ?? .minimal
        let budget = MemoryBudget(
            config: config, hardware: profile,
            contextLength: options.contextLength, policy: policy)

        start = Date()
        let mapping = try ModelMapping(root: root, config: config, device: context.device)
        let expertCache = ExpertSlotCache(
            root: root, config: config,
            slotsPerLayer: budget.expertSlotsPerLayer, device: context.device)
        let runner = try ModelRunner(
            config: config, context: context, mapping: mapping,
            expertCache: expertCache, contextLength: options.contextLength)
        // Faire entrer les pages en une lecture séquentielle plutôt que par défauts de
        // page dispersés pendant la première passe.
        let warmStart = Date()
        mapping.prefault()
        let warmTime = Date().timeIntervalSince(warmStart)
        let loadTime = Date().timeIntervalSince(start)

        FileHandle.standardError.write(Data(
            ("\(config.name) — \(budget.expertSlotsPerLayer)/\(config.expertCount) experts en cache, "
             + "empreinte prévue \(gib(budget.totalFootprintBytes))\n"
             + String(format: "tokeniseur %.1f s, modèle %.1f s (dont %.1f s de préchargement)\n\n",
                       tokenizerTime, loadTime, warmTime)).utf8))

        // --- Invite Harmony ---
        let renderer = Harmony.Renderer(
            reasoningEffort: options.reasoning, instructions: options.instructions)
        let rendered = renderer.render(turns: [.user(prompt)])
        let promptTokens = tokenizer.encode(rendered, allowSpecial: true)

        FileHandle.standardError.write(Data(
            "invite : \(promptTokens.count) jetons — prefill…\n".utf8))

        // --- Prefill par blocs ---
        //
        // Les jetons de l'invite sont connus d'avance : rien n'oblige à les traiter un
        // par un. Par blocs, les poids denses sont lus une fois pour tout le bloc au lieu
        // d'une fois par jeton — même calcul, ordre différent.
        start = Date()
        var distribution = try runner.prefill(tokens: promptTokens)
        let prefillTime = Date().timeIntervalSince(start)
        let t = runner.lastTimings
        FileHandle.standardError.write(Data(
            String(format: "  %d jetons en %.1f s (%.0f jetons/s)\n"
                   + "  cb1 %.2f s · I/O experts %.2f s · experts %.2f s · tête %.2f s\n"
                   + "  lu sur SSD : %.2f Gio\n\n",
                   promptTokens.count, prefillTime,
                   Double(promptTokens.count) / prefillTime,
                   t.attentionAndRouter, t.expertIO, t.mixture, t.head,
                   Double(expertCache.statisticsSnapshot().bytesRead) / 1_073_741_824).utf8))

        // --- Génération ---
        let parser = Harmony.Parser(tokenizer: tokenizer)
        var session = Harmony.Parser.Session()
        let sampling = ModelRunner.Sampling(
            temperature: options.temperature, topP: options.topP)

        var generated = 0
        start = Date()
        var analysisShown = false

        while generated < options.tokenCount && !session.isFinished {
            let token = runner.sample(from: distribution, using: sampling)
            generated += 1

            for event in parser.consume(token, session: &session) {
                switch event {
                case .text(let channel, let text):
                    switch channel {
                    case .final:
                        print(text, terminator: "")
                        fflush(stdout)
                    case .analysis where options.showAnalysis:
                        if !analysisShown {
                            FileHandle.standardError.write(Data("\u{1B}[2m[raisonnement] ".utf8))
                            analysisShown = true
                        }
                        FileHandle.standardError.write(Data(text.utf8))
                    default:
                        break
                    }
                case .channelEnded(let channel):
                    if channel == .analysis, analysisShown {
                        FileHandle.standardError.write(Data("\u{1B}[0m\n\n".utf8))
                    }
                case .stopped:
                    break
                }
            }
            if session.isFinished { break }
            distribution = try runner.forward(token: token)
        }
        let generationTime = Date().timeIntervalSince(start)

        let stats = expertCache.statisticsSnapshot()
        FileHandle.standardError.write(Data(
            String(format:
                "\n\n— %d jetons en %.1f s (%.2f tok/s) — hit cache %.0f %% — empreinte %@\n",
                generated, generationTime, Double(generated) / generationTime,
                stats.hitRate * 100, mib(MemoryFootprint.current())).utf8))

        if !options.showAnalysis && !session.analysisText.isEmpty {
            let hidden = "  (\(session.analysisText.count) caractères de raisonnement "
                + "masqués, --analysis pour les voir)\n"
            FileHandle.standardError.write(Data(hidden.utf8))
        }
    }
}
