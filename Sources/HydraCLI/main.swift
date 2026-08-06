import Foundation
import HydraCore
import HydraFormat
import HydraInstall
import HydraMetal
import HydraTokenize

// A minimal command-line tool. With no external dependency: the surface is too small to
// justify ArgumentParser at this stage.

func gib(_ bytes: Int) -> String { String(format: "%.2f GiB", Double(bytes) / 1_073_741_824) }
func pad(_ s: String, _ n: Int) -> String {
    s.count >= n ? s : s + String(repeating: " ", count: n - s.count)
}

/// The production hardware profile is read from the host machine by the Metal layer.
/// Here, offline, we take a reference profile — explicitly, not as a hidden default.
let referenceHardware = HardwareProfile.appleM4_24GB

func printBudget(_ config: GptOssConfig, contextLength: Int) {
    print(String(repeating: "=", count: 82))
    print("  \(config.name) — \(config.layerCount) layers, \(config.expertCount) experts/layer, "
        + "top-\(config.expertsPerToken), context \(contextLength / 1024)k")
    print(String(repeating: "=", count: 82))

    let ref = MemoryBudget(
        config: config, hardware: referenceHardware,
        contextLength: contextLength, policy: .minimal)

    print("\n  ON DISK")
    print("    Complete installation       \(gib(config.installedBytes))")
    print("    of which the expert pool    \(gib(config.expertPoolBytes))")
    print("    One expert's blob           \(config.expertBlobBytes.formatted()) B")

    print("\n  IRREDUCIBLE MEMORY FLOOR")
    print("    Resident weights            \(gib(config.residentBytes))")
    print("      of which LM head (BF16)   \(gib(config.lmHeadBytes))")
    print("      of which attn + routers   \(gib(config.layerCount * config.residentPerLayerBytes))")
    print("    KV cache FP16               \(gib(ref.kvCacheBytes))")
    print("    Scratch (provisional)       \(gib(ref.scratchBytes))")
    print("    Embedding: mapped, outside the footprint (\(gib(config.embeddingBytes)))")

    print("\n  EXPERT CACHE POLICIES")
    print("    " + pad("policy", 22) + pad("slots/layer", 14) + pad("cache", 11)
        + pad("FOOTPRINT", 12) + pad("% of model", 13) + "estimated tok/s")

    let policies: [(String, ExpertCachePolicy)] = [
        ("minimal", .minimal),
        ("8 slots", .slotsPerLayer(8)),
        ("16 slots", .slotsPerLayer(16)),
        ("4 GiB target", .memoryTarget(bytes: 4 * 1_073_741_824)),
        ("8 GiB target", .memoryTarget(bytes: 8 * 1_073_741_824)),
        ("maximum (reference)", .maximize),
    ]
    for (label, policy) in policies {
        let b = MemoryBudget(
            config: config, hardware: referenceHardware,
            contextLength: contextLength, policy: policy)
        guard b.fits else {
            print("    " + pad(label, 22) + "DOES NOT FIT")
            continue
        }
        let rate = b.isFullyResident
            ? b.maximumTokensPerSecond
            : b.estimatedTokensPerSecond(cacheHitRate: b.uniformRoutingHitRate)
        let note = b.isFullyResident ? "  (no I/O)" : ""
        print("    " + pad(label, 22)
            + pad("\(b.expertSlotsPerLayer)", 14)
            + pad(gib(b.expertCacheBytes), 11)
            + pad(gib(b.totalFootprintBytes), 12)
            + pad(String(format: "%.0f %%", b.residentFractionOfCheckpoint * 100), 13)
            + String(format: "%.1f", rate) + note)
    }

    print(String(format: "\n    Compute floor %.1f ms/token → absolute ceiling %.1f tok/s",
                 ref.computeFloorSeconds * 1000, ref.maximumTokensPerSecond))
    print("    (the tok/s assume uniform routing: a lower bound, real routing is skewed)")
    print()
}

func inspect(path: String) throws {
    let header = try SafetensorsHeader.read(contentsOf: URL(fileURLWithPath: path))
    print("header: \(header.headerByteCount) B, data at offset \(header.dataSectionOffset)")
    print("tensors: \(header.tensors.count)")
    for key in header.tensors.keys.sorted() {
        let e = header.tensors[key]!
        print("  " + pad(key, 56) + pad(e.dtype.rawValue, 5)
            + " \(e.shape) \(e.byteCount.formatted()) B")
    }
}

/// Computes and checks the repack plan without downloading a single weight.
func plan(repo: String, config: GptOssConfig) async throws {
    let client = HuggingFaceClient()
    print("reading \(repo)'s index…")
    let index = try await client.fetchIndex(repo: repo)
    print("  \(index.weightMap.count) tensors, \(index.shards.count) shards, "
        + "declared total \(index.totalSize?.formatted() ?? "n/a") B")

    print("reading the headers (\(index.shards.count) bounded requests)…")
    var headers: [String: SafetensorsHeader] = [:]
    var headerBytes = 0
    for shard in index.shards.sorted() {
        let h = try await client.fetchHeader(repo: repo, file: shard)
        headers[shard] = h
        headerBytes += h.headerByteCount
        print("  \(pad(shard, 40)) \(h.tensors.count) tensors, header \(h.headerByteCount) B")
    }
    print("  \(headerBytes.formatted()) header bytes read in total — no weight downloaded")

    let plan = try RepackPlan(config: config, weightMap: index.weightMap, headers: headers)
    print("\nrepack plan")
    print("  operations                \(plan.operations.count)")
    print("  source bytes covered      \(plan.totalSourceBytes.formatted())")
    print("  destination bytes         \(plan.totalDestinationBytes.formatted())")
    print("  destination files         \(plan.destinationSizes.count)")
    print("  a blob's stride           \(plan.layout.expertBlob.strideBytes.formatted()) B "
        + "(payload \(plan.layout.expertBlob.payloadBytes.formatted()) B)")

    let problems = plan.validate(declaredSourceTotal: index.totalSize)
    if problems.isEmpty {
        print("\n  ✔ plan verified: exact coverage of the checkpoint, no overlap")
    } else {
        print("\n  ✘ \(problems.count) problem(s):")
        for p in problems { print("      \(p.description)") }
        exit(1)
    }
}

func mib(_ bytes: Int) -> String { String(format: "%.1f MiB", Double(bytes) / 1_048_576) }

/// The default location for models, following macOS conventions.
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
    print("source: \(repo)")
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

    // Disk guardrail (D-004): we warn, we do not block.
    let needed = plan.totalDestinationBytes
    if let values = try? directory.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]),
        let free = values.volumeAvailableCapacityForImportantUsage {
        let after = Int(free) - needed
        print("disk: \(gib(Int(free))) free, \(gib(needed)) required, "
            + "\(gib(max(after, 0))) left after installation")
        if after < 10_000_000_000 {
            print("  ⚠︎ under 10 GB would remain free")
        }
        if after < 0 {
            print("  ✘ not enough space")
            throw ExitError.notEnoughSpace
        }
    }

    let destination = directory.appending(path: "\(slug).hydra")
    if FileManager.default.fileExists(atPath: destination.path) {
        print("already installed: \(destination.path)")
        return
    }

    let installer = TokenizerInstaller(client: client, repo: repo)
    let repacker = StreamingRepacker(
        plan: plan, source: HuggingFaceSource(client: client, repo: repo),
        auxiliary: { partial in _ = try await installer.install(into: partial) })
    print("installing to \(destination.path)")
    print("\(plan.spans.count) contiguous regions — the response is consumed as it streams,")
    print("the checkpoint never exists in memory\n")

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
            String(format: "\r  %5.1f %%  %@ / %@  %.0f MB/s  %.0f s left  peak buffer %@  footprint %@   ",
                   percent, gib(progress.bytesDone), gib(progress.bytesTotal),
                   rate / 1e6, remaining,
                   mib(progress.peakPayloadBytes), mib(peak.value)).utf8))
    }

    let elapsed = Date().timeIntervalSince(started)
    print("\n\n  ✔ installation finished in \(String(format: "%.0f", elapsed)) s")
    print("  \(manifest.tensors.count) tensors, \(gib(manifest.sourceTotalBytes)) transferred")
    print("  peak process memory footprint: \(mib(peak.value))")
    print("  largest network block received: \(mib(peakBlock.value))")
}

enum ExitError: Error { case planInvalid, notEnoughSpace, verificationFailed }

/// Verifies an installation: the manifest, then a sample comparison with the source.
func verify(repo: String, config: GptOssConfig, root: URL, samples: Int = 24) async throws {
    let manifest = try HydraManifest.read(from: root)
    try manifest.validate(against: config, root: root)
    print("manifest   ✔ format \(manifest.format), \(manifest.model.name)")
    print("           \(manifest.files.count) files, sizes as expected")
    print("           source: \(manifest.sourceDescription)")

    let client = HuggingFaceClient()
    let index = try await client.fetchIndex(repo: repo)
    var headers: [String: SafetensorsHeader] = [:]
    for shard in index.shards.sorted() {
        headers[shard] = try await client.fetchHeader(repo: repo, file: shard)
    }
    let plan = try RepackPlan(config: config, weightMap: index.weightMap, headers: headers)

    print("\ncomparing samples against the upstream checkpoint…")
    let verifier = InstallationVerifier(
        plan: plan, source: HuggingFaceSource(client: client, repo: repo), root: root)
    let report = try await verifier.spotCheck(sampleCount: samples)

    print("  \(report.samplesChecked) windows, \(report.bytesCompared.formatted()) bytes compared")
    if report.passed {
        print("  ✔ every sample agrees")
    } else {
        for m in report.mismatches { print("  ✘ \(m.description)") }
        throw ExitError.verificationFailed
    }

    try inspectExpertBlob(config: config, root: root)
}

/// Re-reads an installed expert blob and decodes its MXFP4 scales.
///
/// The byte comparison above proves the data is faithful to the source. This check proves
/// something else: that the **split into sub-tensors** is correct. If `scales` and `blocks`
/// were swapped, or offset, the scale bytes would look like uniform noise over 0-255. On a
/// real checkpoint the E8M0 exponents cluster in a narrow range around 127.
///
func inspectExpertBlob(config: GptOssConfig, root: URL) throws {
    let blob = config.expertBlobLayout
    let layerFile = root.appending(path: "experts/layer_00.bin")
    guard let handle = FileHandle(forReadingAtPath: layerFile.path) else { return }
    defer { try? handle.close() }

    // The scales of layer 0's first expert.
    try handle.seek(toOffset: UInt64(blob.gateUpScales.offset))
    guard let scales = try handle.read(upToCount: min(65536, blob.gateUpScales.byteCount)),
        !scales.isEmpty
    else { return }

    let exponents = scales.map { Int($0) - 127 }
    let minimum = exponents.min() ?? 0
    let maximum = exponents.max() ?? 0
    let mean = Double(exponents.reduce(0, +)) / Double(exponents.count)
    let nans = scales.filter { $0 == 0xFF }.count

    print("\nexpert blob structure (layer 0, expert 0, gate_up)")
    print("  \(scales.count.formatted()) E8M0 scales read")
    print(String(format: "  exponents: min %d, max %d, mean %.1f", minimum, maximum, mean))
    print("  NaN scales (0xFF): \(nans)")
    if maximum - minimum <= 40 && nans == 0 {
        print("  ✔ narrow distribution with no NaN — the blocks/scales split is correct")
    } else {
        print("  ⚠︎ unexpected distribution — check the sub-tensor split")
    }

    // Decoding one block, to close the loop with the decoder validated at milestone 1.2.
    try handle.seek(toOffset: UInt64(blob.gateUpBlocks.offset))
    guard let packed = try handle.read(upToCount: MXFP4.packedBytesPerBlock) else { return }
    let values = try MXFP4.decode(packed: packed, scales: scales.prefix(1))
    let magnitudes = values.map { abs($0) }.filter { $0 > 0 }
    print(String(format: "  first block decoded: %d values, max |v| %.3e, %d zero",
                 values.count, magnitudes.max() ?? 0, values.count - magnitudes.count))
}

/// A thread-safe box, the progress callback being `@Sendable`.
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

/// Limits the display rate. The progress callback is invoked thousands of times a second;
/// rewriting the line every time would cost more than the repack itself.
final class Throttle: @unchecked Sendable {
    private var last = Date.distantPast
    private let interval: TimeInterval
    private let lock = NSLock()

    init(every interval: TimeInterval) { self.interval = interval }

    /// True at most once per `interval`.
    func ready(_ now: Date = Date()) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard now.timeIntervalSince(last) >= interval else { return false }
        last = now
        return true
    }
}

// MARK: - Argument parsing

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
            print("usage: hydra inspect <file.safetensors>")
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
            ? "Explain in three sentences why the sky is blue."
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
        print("downloading \(repo)'s tokenizer…")
        let written = try await TokenizerInstaller(repo: repo).install(into: root)
        print("  \(written) files in \(root.path)/tokenizer")
        let tokenizer = try TokenizerInstaller.load(from: root)
        print("  vocabulary: \(tokenizer.count) entries, "
            + "\(tokenizer.specialTokens.count) special tokens")

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
            usage: hydra <command>

              budget [context]          memory footprint and projected throughput, per cache policy
              plan [20b|120b]           computes and checks the repack plan, without downloading
              install [20b|120b] [dir]   installs the model in the .hydra format, streaming
              tokenizer [20b|120b]      installs the tokenizer into an existing installation
              verify [20b|120b] [dir]    compares installed windows against the upstream bytes
              probe [20b|120b] [ctx]     exercises mapping, the expert cache and the GPU kernels
              bench [20b|120b]           paired comparisons of I/O and kernels
              bench-gemm [20b|120b]      isolated bench of the dense projections
              generate [20b|120b] [n] [slots]  complete forward pass, throughput and footprint
              chat [20b|120b] <text> [options]
                  --tokens N --slots N --context N --temperature F --top-p F
                  --reasoning low|medium|high --analysis --instructions "…"
              inspect <file>            a safetensors header, without reading the data
            """)
        exit(2)
    }
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n".utf8))
    exit(1)
}
