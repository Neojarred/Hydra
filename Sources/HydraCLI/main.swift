import Foundation
import HydraCore
import HydraFormat
import HydraInstall
import HydraMetal
import HydraTokenize

// Outil en ligne de commande minimal. Sans dépendance externe : la surface est trop
// petite pour justifier ArgumentParser à ce stade.

func gib(_ bytes: Int) -> String { String(format: "%.2f Gio", Double(bytes) / 1_073_741_824) }
func pad(_ s: String, _ n: Int) -> String {
    s.count >= n ? s : s + String(repeating: " ", count: n - s.count)
}

/// Le profil matériel de production est lu sur la machine hôte par la couche Metal.
/// Ici, hors ligne, on prend un profil de référence — explicitement, pas par défaut caché.
let referenceHardware = HardwareProfile.appleM4_24GB

func printBudget(_ config: GptOssConfig, contextLength: Int) {
    print(String(repeating: "=", count: 82))
    print("  \(config.name) — \(config.layerCount) couches, \(config.expertCount) experts/couche, "
        + "top-\(config.expertsPerToken), contexte \(contextLength / 1024)k")
    print(String(repeating: "=", count: 82))

    let ref = MemoryBudget(
        config: config, hardware: referenceHardware,
        contextLength: contextLength, policy: .minimal)

    print("\n  SUR DISQUE")
    print("    Installation complète       \(gib(config.installedBytes))")
    print("    dont pool d'experts         \(gib(config.expertPoolBytes))")
    print("    Blob d'un expert            \(config.expertBlobBytes.formatted()) o")

    print("\n  PLANCHER INCOMPRESSIBLE EN MÉMOIRE")
    print("    Poids résidents             \(gib(config.residentBytes))")
    print("      dont tête LM (BF16)       \(gib(config.lmHeadBytes))")
    print("      dont attention + routeurs \(gib(config.layerCount * config.residentPerLayerBytes))")
    print("    KV cache FP16               \(gib(ref.kvCacheBytes))")
    print("    Scratch (provisoire)        \(gib(ref.scratchBytes))")
    print("    Embedding : mappé, hors empreinte (\(gib(config.embeddingBytes)))")

    print("\n  POLITIQUES DE CACHE D'EXPERTS")
    print("    " + pad("politique", 22) + pad("slots/couche", 14) + pad("cache", 11)
        + pad("EMPREINTE", 12) + pad("% du modèle", 13) + "tok/s estimés")

    let policies: [(String, ExpertCachePolicy)] = [
        ("minimal", .minimal),
        ("8 slots", .slotsPerLayer(8)),
        ("16 slots", .slotsPerLayer(16)),
        ("cible 4 Gio", .memoryTarget(bytes: 4 * 1_073_741_824)),
        ("cible 8 Gio", .memoryTarget(bytes: 8 * 1_073_741_824)),
        ("maximum (référence)", .maximize),
    ]
    for (label, policy) in policies {
        let b = MemoryBudget(
            config: config, hardware: referenceHardware,
            contextLength: contextLength, policy: policy)
        guard b.fits else {
            print("    " + pad(label, 22) + "NE TIENT PAS")
            continue
        }
        let rate = b.isFullyResident
            ? b.maximumTokensPerSecond
            : b.estimatedTokensPerSecond(cacheHitRate: b.uniformRoutingHitRate)
        let note = b.isFullyResident ? "  (aucune I/O)" : ""
        print("    " + pad(label, 22)
            + pad("\(b.expertSlotsPerLayer)", 14)
            + pad(gib(b.expertCacheBytes), 11)
            + pad(gib(b.totalFootprintBytes), 12)
            + pad(String(format: "%.0f %%", b.residentFractionOfCheckpoint * 100), 13)
            + String(format: "%.1f", rate) + note)
    }

    print(String(format: "\n    Plancher de calcul %.1f ms/token → plafond absolu %.1f tok/s",
                 ref.computeFloorSeconds * 1000, ref.maximumTokensPerSecond))
    print("    (les tok/s supposent un routage uniforme : borne basse, le routage réel est biaisé)")
    print()
}

func inspect(path: String) throws {
    let header = try SafetensorsHeader.read(contentsOf: URL(fileURLWithPath: path))
    print("en-tête : \(header.headerByteCount) o, données à l'offset \(header.dataSectionOffset)")
    print("tenseurs : \(header.tensors.count)")
    for key in header.tensors.keys.sorted() {
        let e = header.tensors[key]!
        print("  " + pad(key, 56) + pad(e.dtype.rawValue, 5)
            + " \(e.shape) \(e.byteCount.formatted()) o")
    }
}

/// Calcule et vérifie le plan de repack sans télécharger le moindre poids.
func plan(repo: String, config: GptOssConfig) async throws {
    let client = HuggingFaceClient()
    print("lecture de l'index de \(repo)…")
    let index = try await client.fetchIndex(repo: repo)
    print("  \(index.weightMap.count) tenseurs, \(index.shards.count) shards, "
        + "total déclaré \(index.totalSize?.formatted() ?? "n/a") o")

    print("lecture des en-têtes (\(index.shards.count) requêtes bornées)…")
    var headers: [String: SafetensorsHeader] = [:]
    var headerBytes = 0
    for shard in index.shards.sorted() {
        let h = try await client.fetchHeader(repo: repo, file: shard)
        headers[shard] = h
        headerBytes += h.headerByteCount
        print("  \(pad(shard, 40)) \(h.tensors.count) tenseurs, en-tête \(h.headerByteCount) o")
    }
    print("  \(headerBytes.formatted()) octets d'en-tête lus au total — aucun poids téléchargé")

    let plan = try RepackPlan(config: config, weightMap: index.weightMap, headers: headers)
    print("\nplan de repack")
    print("  opérations                \(plan.operations.count)")
    print("  octets sources couverts   \(plan.totalSourceBytes.formatted())")
    print("  octets destination        \(plan.totalDestinationBytes.formatted())")
    print("  fichiers destination      \(plan.destinationSizes.count)")
    print("  stride d'un blob          \(plan.layout.expertBlob.strideBytes.formatted()) o "
        + "(charge utile \(plan.layout.expertBlob.payloadBytes.formatted()) o)")

    let problems = plan.validate(declaredSourceTotal: index.totalSize)
    if problems.isEmpty {
        print("\n  ✔ plan vérifié : couverture exacte du checkpoint, aucun chevauchement")
    } else {
        print("\n  ✘ \(problems.count) problème(s) :")
        for p in problems { print("      \(p.description)") }
        exit(1)
    }
}

func mib(_ bytes: Int) -> String { String(format: "%.1f Mio", Double(bytes) / 1_048_576) }

/// Emplacement par défaut des modèles, conforme aux conventions macOS.
func defaultModelDirectory() throws -> URL {
    let base = try FileManager.default.url(
        for: .applicationSupportDirectory, in: .userDomainMask,
        appropriateFor: nil, create: true)
    let dir = base.appending(path: "Hydra/Models")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir
}

func install(repo: String, config: GptOssConfig, slug: String, into directory: URL) async throws {
    let client = HuggingFaceClient()
    print("source : \(repo)")
    let index = try await client.fetchIndex(repo: repo)
    var headers: [String: SafetensorsHeader] = [:]
    for shard in index.shards.sorted() {
        headers[shard] = try await client.fetchHeader(repo: repo, file: shard)
    }
    let plan = try RepackPlan(config: config, weightMap: index.weightMap, headers: headers)

    let problems = plan.validate(declaredSourceTotal: index.totalSize)
    guard problems.isEmpty else {
        for p in problems { print("  ✘ \(p.description)") }
        throw ExitError.planInvalid
    }

    // Garde-fou disque (D-004) : on avertit, on ne bloque pas.
    let needed = plan.totalDestinationBytes
    if let values = try? directory.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]),
        let free = values.volumeAvailableCapacityForImportantUsage {
        let after = Int(free) - needed
        print("disque : \(gib(Int(free))) libres, \(gib(needed)) requis, "
            + "\(gib(max(after, 0))) restants après installation")
        if after < 10_000_000_000 {
            print("  ⚠︎ moins de 10 Go resteraient libres")
        }
        if after < 0 {
            print("  ✘ espace insuffisant")
            throw ExitError.notEnoughSpace
        }
    }

    let destination = directory.appending(path: "\(slug).hydra")
    if FileManager.default.fileExists(atPath: destination.path) {
        print("déjà installé : \(destination.path)")
        return
    }

    let installer = TokenizerInstaller(client: client, repo: repo)
    let repacker = StreamingRepacker(
        plan: plan, source: HuggingFaceSource(client: client, repo: repo),
        auxiliary: { partial in _ = try await installer.install(into: partial) })
    print("installation vers \(destination.path)")
    print("\(plan.spans.count) régions contiguës — la réponse est consommée au fil de l'eau,")
    print("le checkpoint n'existe à aucun moment en mémoire\n")

    let peak = MemoryFootprint.Peak()
    let peakBlock = Observed(0)
    let started = Date()
    let throttle = Throttle(every: 0.5)

    let manifest = try await repacker.run(destination: destination) { progress in
        peak.sample()
        peakBlock.update { $0 = max($0, progress.peakPayloadBytes) }
        let now = Date()
        guard throttle.ready(now) else { return }
        let elapsed = now.timeIntervalSince(started)
        let rate = elapsed > 0 ? Double(progress.bytesDone) / elapsed : 0
        let remaining = rate > 0 ? Double(progress.bytesTotal - progress.bytesDone) / rate : 0
        let percent = Double(progress.bytesDone) / Double(progress.bytesTotal) * 100
        FileHandle.standardError.write(Data(
            String(format: "\r  %5.1f %%  %@ / %@  %.0f Mo/s  reste %.0f s  tampon max %@  empreinte %@   ",
                   percent, gib(progress.bytesDone), gib(progress.bytesTotal),
                   rate / 1e6, remaining,
                   mib(progress.peakPayloadBytes), mib(peak.value)).utf8))
    }

    let elapsed = Date().timeIntervalSince(started)
    print("\n\n  ✔ installation terminée en \(String(format: "%.0f", elapsed)) s")
    print("  \(manifest.tensors.count) tenseurs, \(gib(manifest.sourceTotalBytes)) transférés")
    print("  empreinte mémoire maximale du processus : \(mib(peak.value))")
    print("  bloc réseau maximal reçu               : \(mib(peakBlock.value))")
}

enum ExitError: Error { case planInvalid, notEnoughSpace, verificationFailed }

/// Vérifie une installation : manifeste, puis comparaison d'échantillons avec la source.
func verify(repo: String, config: GptOssConfig, root: URL, samples: Int = 24) async throws {
    let manifest = try HydraManifest.read(from: root)
    try manifest.validate(against: config, root: root)
    print("manifeste  ✔ format \(manifest.format), \(manifest.model.name)")
    print("           \(manifest.files.count) fichiers, tailles conformes")
    print("           source : \(manifest.sourceDescription)")

    let client = HuggingFaceClient()
    let index = try await client.fetchIndex(repo: repo)
    var headers: [String: SafetensorsHeader] = [:]
    for shard in index.shards.sorted() {
        headers[shard] = try await client.fetchHeader(repo: repo, file: shard)
    }
    let plan = try RepackPlan(config: config, weightMap: index.weightMap, headers: headers)

    print("\ncomparaison d'échantillons avec le checkpoint amont…")
    let verifier = InstallationVerifier(
        plan: plan, source: HuggingFaceSource(client: client, repo: repo), root: root)
    let report = try await verifier.spotCheck(sampleCount: samples)

    print("  \(report.samplesChecked) fenêtres, \(report.bytesCompared.formatted()) octets comparés")
    if report.passed {
        print("  ✔ tous les échantillons concordent")
    } else {
        for m in report.mismatches { print("  ✘ \(m.description)") }
        throw ExitError.verificationFailed
    }

    try inspectExpertBlob(config: config, root: root)
}

/// Relit un blob d'expert installé et décode ses échelles MXFP4.
///
/// La comparaison d'octets ci-dessus prouve que les données sont fidèles à la source.
/// Ce contrôle-ci prouve autre chose : que le **découpage en sous-tenseurs** est correct.
/// Si `scales` et `blocks` étaient intervertis, ou décalés, les octets d'échelle
/// ressembleraient à du bruit uniforme sur 0-255. Sur un vrai checkpoint, les exposants
/// E8M0 se concentrent dans une plage étroite autour de 127.
func inspectExpertBlob(config: GptOssConfig, root: URL) throws {
    let blob = config.expertBlobLayout
    let layerFile = root.appending(path: "experts/layer_00.bin")
    guard let handle = FileHandle(forReadingAtPath: layerFile.path) else { return }
    defer { try? handle.close() }

    // Échelles du premier expert de la couche 0.
    try handle.seek(toOffset: UInt64(blob.gateUpScales.offset))
    guard let scales = try handle.read(upToCount: min(65536, blob.gateUpScales.byteCount)),
        !scales.isEmpty
    else { return }

    let exponents = scales.map { Int($0) - 127 }
    let minimum = exponents.min() ?? 0
    let maximum = exponents.max() ?? 0
    let mean = Double(exponents.reduce(0, +)) / Double(exponents.count)
    let nans = scales.filter { $0 == 0xFF }.count

    print("\nstructure du blob d'expert (couche 0, expert 0, gate_up)")
    print("  \(scales.count.formatted()) échelles E8M0 lues")
    print(String(format: "  exposants : min %d, max %d, moyenne %.1f", minimum, maximum, mean))
    print("  échelles NaN (0xFF) : \(nans)")
    if maximum - minimum <= 40 && nans == 0 {
        print("  ✔ distribution étroite et sans NaN — le découpage blocks/scales est correct")
    } else {
        print("  ⚠︎ distribution inattendue — vérifier le découpage des sous-tenseurs")
    }

    // Décodage d'un bloc, pour boucler avec le décodeur validé au jalon 1.2.
    try handle.seek(toOffset: UInt64(blob.gateUpBlocks.offset))
    guard let packed = try handle.read(upToCount: MXFP4.packedBytesPerBlock) else { return }
    let values = try MXFP4.decode(packed: packed, scales: scales.prefix(1))
    let magnitudes = values.map { abs($0) }.filter { $0 > 0 }
    print(String(format: "  premier bloc décodé : %d valeurs, |v| max %.3e, %d nulles",
                 values.count, magnitudes.max() ?? 0, values.count - magnitudes.count))
}

/// Boîte thread-safe, le rappel de progression étant `@Sendable`.
final class Observed<Value>: @unchecked Sendable {
    private var storage: Value
    private let lock = NSLock()
    init(_ initial: Value) { storage = initial }
    var value: Value {
        lock.lock(); defer { lock.unlock() }
        return storage
    }
    func update(_ transform: (inout Value) -> Void) {
        lock.lock(); defer { lock.unlock() }
        transform(&storage)
    }
}

/// Limite la fréquence d'affichage. Le rappel de progression est appelé des milliers de
/// fois par seconde ; réécrire la ligne à chaque fois coûterait plus cher que le repack.
final class Throttle: @unchecked Sendable {
    private var last = Date.distantPast
    private let interval: TimeInterval
    private let lock = NSLock()

    init(every interval: TimeInterval) { self.interval = interval }

    /// Vrai au plus une fois par `interval`.
    func ready(_ now: Date = Date()) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard now.timeIntervalSince(last) >= interval else { return false }
        last = now
        return true
    }
}

// MARK: - Analyse des arguments

let args = Array(CommandLine.arguments.dropFirst())

func configNamed(_ s: String?) -> (GptOssConfig, String) {
    switch s {
    case "120b", "gpt-oss-120b": return (.b120, "openai/gpt-oss-120b")
    default: return (.b20, "openai/gpt-oss-20b")
    }
}

do {
    switch args.first {
    case "budget", nil:
        let ctx = args.count > 1 ? (Int(args[1]) ?? 8192) : 8192
        printBudget(.b20, contextLength: ctx)
        printBudget(.b120, contextLength: ctx)

    case "inspect":
        guard args.count >= 2 else {
            print("usage : hydra inspect <fichier.safetensors>")
            exit(2)
        }
        try inspect(path: args[1])

    case "plan":
        let (config, repo) = configNamed(args.count > 1 ? args[1] : nil)
        try await plan(repo: repo, config: config)

    case "chat":
        var options = Chat.Options()
        var promptParts: [String] = []
        var which = "20b"
        var index = 1
        while index < args.count {
            switch args[index] {
            case "20b", "120b": which = args[index]
            case "--tokens": index += 1; options.tokenCount = Int(args[index]) ?? 512
            case "--slots": index += 1; options.slotsPerLayer = Int(args[index])
            case "--context": index += 1; options.contextLength = Int(args[index]) ?? 4096
            case "--temperature": index += 1; options.temperature = Float(args[index]) ?? 1.0
            case "--top-p": index += 1; options.topP = Float(args[index]) ?? 1.0
            case "--reasoning":
                index += 1
                options.reasoning = Harmony.ReasoningEffort(rawValue: args[index]) ?? .medium
            case "--analysis": options.showAnalysis = true
            case "--instructions": index += 1; options.instructions = args[index]
            default: promptParts.append(args[index])
            }
            index += 1
        }
        let (chatConfig, chatRepo) = configNamed(which)
        let chatSlug = chatRepo.split(separator: "/").last.map(String.init) ?? which
        let promptText = promptParts.isEmpty
            ? "Explique en trois phrases pourquoi le ciel est bleu."
            : promptParts.joined(separator: " ")
        try Chat.run(
            config: chatConfig,
            root: try defaultModelDirectory().appending(path: "\(chatSlug).hydra"),
            prompt: promptText, options: options)

    case "generate":
        let which = args.count > 1 ? args[1] : "20b"
        let (config, repo) = configNamed(which)
        let slug = repo.split(separator: "/").last.map(String.init) ?? which
        let tokens = args.count > 2 ? (Int(args[2]) ?? 16) : 16
        let slots = args.count > 3 ? Int(args[3]) : nil
        try Generate.run(
            config: config,
            root: try defaultModelDirectory().appending(path: "\(slug).hydra"),
            contextLength: 4096, tokenCount: tokens, slotsPerLayer: slots)

    case "bench-gemm":
        let (gemmConfig, _) = configNamed(args.count > 1 ? args[1] : nil)
        try BenchGEMM.run(config: gemmConfig)

    case "bench":
        let which = args.count > 1 ? args[1] : "20b"
        let (config, repo) = configNamed(which)
        let slug = repo.split(separator: "/").last.map(String.init) ?? which
        try Bench.run(
            config: config, root: try defaultModelDirectory().appending(path: "\(slug).hydra"))

    case "probe":
        let which = args.count > 1 ? args[1] : "20b"
        let (config, repo) = configNamed(which)
        let slug = repo.split(separator: "/").last.map(String.init) ?? which
        let ctx = args.count > 2 ? (Int(args[2]) ?? 8192) : 8192
        try Probe.run(
            config: config,
            root: try defaultModelDirectory().appending(path: "\(slug).hydra"),
            contextLength: ctx)

    case "tokenizer":
        let which = args.count > 1 ? args[1] : "20b"
        let (_, repo) = configNamed(which)
        let slug = repo.split(separator: "/").last.map(String.init) ?? which
        let root = try defaultModelDirectory().appending(path: "\(slug).hydra")
        print("téléchargement du tokeniseur de \(repo)…")
        let written = try await TokenizerInstaller(repo: repo).install(into: root)
        print("  \(written) fichiers dans \(root.path)/tokenizer")
        let tokenizer = try TokenizerInstaller.load(from: root)
        print("  vocabulaire : \(tokenizer.count) entrées, "
            + "\(tokenizer.specialTokens.count) jetons spéciaux")

    case "verify":
        let which = args.count > 1 ? args[1] : "20b"
        let (config, repo) = configNamed(which)
        let directory = args.count > 2
            ? URL(fileURLWithPath: args[2]) : try defaultModelDirectory()
        let slug = repo.split(separator: "/").last.map(String.init) ?? which
        let samples = args.count > 3 ? (Int(args[3]) ?? 24) : 24
        try await verify(
            repo: repo, config: config,
            root: directory.appending(path: "\(slug).hydra"), samples: samples)

    case "install":
        let which = args.count > 1 ? args[1] : "20b"
        let (config, repo) = configNamed(which)
        let directory = args.count > 2
            ? URL(fileURLWithPath: args[2]) : try defaultModelDirectory()
        try await install(
            repo: repo, config: config,
            slug: repo.split(separator: "/").last.map(String.init) ?? which,
            into: directory)

    default:
        print("""
            usage : hydra <commande>

              budget [contexte]         empreinte mémoire et débit projeté, par politique de cache
              plan [20b|120b]           calcule et vérifie le plan de repack, sans télécharger
              install [20b|120b] [rép.]  installe le modèle au format .hydra en streaming
              tokenizer [20b|120b]      installe le tokeniseur dans une installation existante
              verify [20b|120b] [rép.]   compare des fenêtres installées aux octets amont
              probe [20b|120b] [ctx]     exerce mappage, cache d'experts et noyaux GPU
              bench [20b|120b]           comparaisons appariées I/O et noyaux
              bench-gemm [20b|120b]      banc isolé des projections denses
              generate [20b|120b] [n] [slots]  passe avant complète, débit et empreinte
              chat [20b|120b] <texte> [options]
                  --tokens N --slots N --context N --temperature F --top-p F
                  --reasoning low|medium|high --analysis --instructions "…"
              inspect <fichier>         en-tête d'un safetensors, sans lire les données
            """)
        exit(2)
    }
} catch {
    FileHandle.standardError.write(Data("erreur : \(error)\n".utf8))
    exit(1)
}
