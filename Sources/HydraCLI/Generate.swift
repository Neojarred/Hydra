import Foundation
import HydraCore
import HydraFormat
import HydraMetal
import Metal

/// Passe avant complète sur un modèle installé.
///
/// Sans tokenizer, on ne produit encore que des identifiants — mais cela suffit à valider
/// que la chaîne tient debout de bout en bout et à chiffrer le débit réel. Le texte
/// viendra avec `o200k_harmony`.
enum Generate {

    static func run(
        config: GptOssConfig, root: URL, contextLength: Int,
        tokenCount: Int, slotsPerLayer: Int?
    ) throws {
        let context = try MetalContext()
        let device = context.device

        print("GPU        \(device.name), famille \(context.gpuFamily)")
        print("plafond    \(gib(Int(device.recommendedMaxWorkingSetSize)))")

        let profile = context.hardwareProfile(memoryBandwidth: 94e9, diskBandwidth: 5.5e9)
        let policy: ExpertCachePolicy =
            slotsPerLayer.map { .slotsPerLayer($0) } ?? .minimal
        let budget = MemoryBudget(
            config: config, hardware: profile, contextLength: contextLength, policy: policy)

        print("\nBUDGET")
        for line in budget.breakdown {
            print("  " + pad(line.label, 40) + gib(line.bytes))
        }
        print("  " + pad("EMPREINTE PRÉVUE", 40) + gib(budget.totalFootprintBytes))
        print(String(format: "  soit %.0f %% du modèle installé (%@)",
                     budget.residentFractionOfCheckpoint * 100, gib(config.installedBytes)))

        let baseline = MemoryFootprint.current()
        let mapping = try ModelMapping(root: root, config: config, device: device)
        let expertCache = ExpertSlotCache(
            root: root, config: config,
            slotsPerLayer: budget.expertSlotsPerLayer, device: device)
        let runner = try ModelRunner(
            config: config, context: context, mapping: mapping,
            expertCache: expertCache, contextLength: contextLength)

        print("\nCHARGEMENT")
        print("  \(budget.expertSlotsPerLayer) slots/couche sur \(config.expertCount)")
        print("  réservé par le runtime : \(gib(runner.reservedBytes))")
        print("  empreinte après chargement : \(mib(MemoryFootprint.current())) "
            + "(avant : \(mib(baseline)))")

        // Amorce arbitraire : sans tokenizer, seuls comptent la forme et le débit.
        let prompt = [1, 2, 3, 4]
        print("\nGÉNÉRATION — \(tokenCount) tokens, amorce de \(prompt.count) identifiants")

        let peak = MemoryFootprint.Peak()
        var perToken: [Double] = []
        var attentionTotal = 0.0, ioTotal = 0.0, mixtureTotal = 0.0, headTotal = 0.0
        let started = Date()

        let produced = try runner.generate(prompt: prompt, count: tokenCount) { token, timings in
            peak.sample()
            perToken.append(timings.total)
            attentionTotal += timings.attentionAndRouter
            ioTotal += timings.expertIO
            mixtureTotal += timings.mixture
            headTotal += timings.head
            FileHandle.standardError.write(Data(
                String(format: "\r  token %3d → %6d   %5.0f ms   %@   ",
                       perToken.count, token, timings.total * 1000,
                       mib(MemoryFootprint.current())).utf8))
        }

        let elapsed = Date().timeIntervalSince(started)
        let stats = expertCache.statisticsSnapshot()
        print("\n")
        print("RÉSULTAT")
        print("  \(produced.count) tokens en \(String(format: "%.1f", elapsed)) s")
        print(String(format: "  débit : %.2f tok/s  (médiane %.0f ms/token)",
                     Double(produced.count) / elapsed, Bench.median(perToken) * 1000))
        print("  premiers identifiants : \(produced.prefix(12).map(String.init).joined(separator: " "))")

        let steps = max(perToken.count, 1)
        let decodeTime = attentionTotal + ioTotal + mixtureTotal + headTotal
        print("\nRÉPARTITION DU TEMPS PAR TOKEN")
        print(String(format: "  cb1  attention + routeur   %6.1f ms   %4.0f %%",
                     attentionTotal / Double(steps) * 1000,
                     attentionTotal / max(decodeTime, 1e-9) * 100))
        print(String(format: "  I/O  lecture des experts   %6.1f ms   %4.0f %%",
                     ioTotal / Double(steps) * 1000, ioTotal / max(decodeTime, 1e-9) * 100))
        print(String(format: "  cb2  mélange d'experts     %6.1f ms   %4.0f %%",
                     mixtureTotal / Double(steps) * 1000, mixtureTotal / max(decodeTime, 1e-9) * 100))
        print(String(format: "  tête LM                    %6.1f ms   %4.0f %%",
                     headTotal / Double(steps) * 1000, headTotal / max(decodeTime, 1e-9) * 100))

        print("\nCACHE D'EXPERTS")
        print(String(format: "  %d hits, %d miss — taux de hit %.1f %%",
                     stats.hits, stats.misses, stats.hitRate * 100))
        print("  lu depuis le SSD : \(gib(stats.bytesRead))")

        print("\nMÉMOIRE")
        print("  engagée par le processus     : \(mib(peak.value))  (phys_footprint)")
        print("  résidente                    : \(mib(peak.residentValue))  (resident_size)")
        print("     Ces deux compteurs couvrent la mémoire anonyme : slots d'experts,")
        print("     scratch, KV cache, logits. Ils correspondent bien à la réservation")
        print("     annoncée ci-dessus.")
        print("  poids mappés, en plus        : \(gib(mapping.mappedByteCount))")
        print("     resident.bin et embed.bin sont adossés à des fichiers. Le noyau ne")
        print("     les impute à aucun des deux compteurs et peut les reprendre sous")
        print("     pression — mais ils occupent bien de la RAM tant qu'il y en a.")
        print("     C'est une élasticité, pas une gratuité : l'annoncer comme nulle")
        print("     serait malhonnête.")
        print("  modèle installé              : \(gib(config.installedBytes))")
        print(String(format: "  plafond haut (engagée + mappé) : %@ soit %.0f %% de l'installé",
                     gib(peak.value + mapping.mappedByteCount),
                     Double(peak.value + mapping.mappedByteCount)
                        / Double(config.installedBytes) * 100))
    }
}
