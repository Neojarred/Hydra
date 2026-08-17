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
/// Here, offline, we take a reference profile, explicitly, not as a hidden default.
let referenceHardware = HardwareProfile.appleM4_24GB

func printBudget(_ config: GptOssConfig, contextLength: Int) {
    print(String(repeating: "=", count: 82))
    print("  \(config.name), \(config.layerCount) layers, \(config.expertCount) experts/layer, "
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
func plan(repo: String, model: any ModelDescriptor) async throws {
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
    print("  \(headerBytes.formatted()) header bytes read in total, no weight downloaded")

    let plan = try RepackPlanFactory.plan(
        for: model, weightMap: index.weightMap, headers: headers)
    print("\nrepack plan, \(model.architecture.label)")
    print("  operations                \(plan.operations.count)")
    print("  source bytes covered      \(plan.totalSourceBytes.formatted())")
    print("  destination bytes         \(plan.totalDestinationBytes.formatted())")
    print("  destination files         \(plan.destinationSizes.count)")
    print("  a blob's stride           \(plan.layout.expertBlob.strideBytes.formatted()) B "
        + "(payload \(plan.layout.expertBlob.payloadBytes.formatted()) B)")

    let problems = plan.validate(
        weightMap: index.weightMap, declaredSourceTotal: index.totalSize)
    if problems.isEmpty {
        print("\n  ✔ plan verified: every tensor accounted for, no overlap")
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

func install(
    repo: String, model: any ModelDescriptor, slug: String, into directory: URL
) async throws {
    let client = HuggingFaceClient()
    print("source: \(repo)")
    let index = try await client.fetchIndex(repo: repo)
    var headers: [String: SafetensorsHeader] = [:]
    for shard in index.shards.sorted() {
        headers[shard] = try await client.fetchHeader(repo: repo, file: shard)
    }
    let plan = try RepackPlanFactory.plan(
        for: model, weightMap: index.weightMap, headers: headers)

    let problems = plan.validate(
        weightMap: index.weightMap, declaredSourceTotal: index.totalSize)
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
    print("\(plan.spans.count) contiguous regions, the response is consumed as it streams,")
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
        print("  ✔ narrow distribution with no NaN, the blocks/scales split is correct")
    } else {
        print("  ⚠︎ unexpected distribution, check the sub-tensor split")
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

/// GPT-OSS only. Several subcommands below are measurement tools written against this
/// architecture's tensors, `bench-gemm`, `probe`, `verify`, and they say so by taking a
/// `GptOssConfig` rather than pretending to be general.
func configNamed(_ s: String?) -> (GptOssConfig, String) {
    switch s {
    case "120b", "gpt-oss-120b": return (.b120, "openai/gpt-oss-120b")
    default: return (.b20, "openai/gpt-oss-20b")
    }
}

/// The names a subcommand will accept as a model rather than as prompt text.
///
/// Centralized after a silent failure: `chat gemma-q4 "..."` matched no case in the argument
/// loop, so the name fell through to the prompt, the command loaded the default model, and it
/// answered, correctly, and quickly, and from the wrong model. A throughput number was quoted
/// from it before the tokenizer dump showed «gem»«ma»«-q»«4» at the head of the prompt.
///
/// Anything added to `modelNamed` has to be added here, which is why they now sit together.

/// Every model the runtime can install and run, for the subcommands that are architecture
/// neutral: `install` and `chat`.
/// The identifier is **declared, not derived from the repository name.**
///
/// It names the installation directory, and the app's catalogue uses its own identifier for
/// the same purpose. Deriving one from `openai/gpt-oss-20b` happened to give `gpt-oss-20b`,
/// which is what the catalogue calls it, so the two agreed by luck for a year. Deriving one
/// from `google/gemma-4-26B-A4B-it` gives a directory the app never looks in: installing from
/// the command line then produced a model the application could not see, with nothing to
/// indicate why.
/// The models, their aliases, and the identifier each installs under. **One table**, so the
/// set of recognized names and the lookup cannot disagree.
///
/// They did. `modelNamed` learned about Qwen and the `modelNames` beside it did not, so
/// `hydra chat qwen-q4 "..."` did not fail: the unrecognized word fell through to the prompt
/// and GPT-OSS answered a question that began with "qwen-q4". Finite, plausible, and the wrong
/// model, which is the failure this project keeps meeting and had just built into its own
/// command line.
let modelTable: [(aliases: [String], model: any ModelDescriptor, repo: String, id: String)] = [
    (["20b", "gpt-oss-20b"], GptOssConfig.b20, "openai/gpt-oss-20b", "gpt-oss-20b"),
    (["120b", "gpt-oss-120b"], GptOssConfig.b120, "openai/gpt-oss-120b", "gpt-oss-120b"),
    (["gemma", "gemma-4", "gemma-4-26b", "gemma-4-26b-a4b"],
     Gemma4Config.a4b, "google/gemma-4-26B-A4B-it", "gemma-4-26b-a4b"),
    (["gemma-q4", "gemma-4-q4", "gemma-mlx", "gemma-4-26b-a4b-q4"],
     Gemma4MLXConfig.a4b, "lmstudio-community/gemma-4-26B-A4B-it-QAT-MLX-4bit",
     "gemma-4-26b-a4b-q4"),
    (["qwen", "qwen-q4", "qwen-3-6-35b-a3b-q4"],
     Qwen35MoeConfig.a3bQ4, "lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit",
     "qwen-3-6-35b-a3b-q4"),
    (["qwen-q8", "qwen-3-6-35b-a3b-q8"],
     Qwen35MoeConfig.a3bQ8, "lmstudio-community/Qwen3.6-35B-A3B-MLX-8bit",
     "qwen-3-6-35b-a3b-q8"),
]

let modelNames: Set<String> = Set(modelTable.flatMap(\.aliases))

/// The identifier is **declared, not derived from the repository name.**
///
/// It names the installation directory, and the app's catalogue uses its own identifier for
/// the same purpose. Deriving one from `openai/gpt-oss-20b` happened to give `gpt-oss-20b`,
/// which is what the catalogue calls it, so the two agreed by luck for a year. Deriving one
/// from `google/gemma-4-26B-A4B-it` gives a directory the app never looks in: installing from
/// the command line then produced a model the application could not see, with nothing to
/// indicate why.
///
/// An unrecognized name **stops**. Falling back to the 20B means a typo installs or runs a
/// model the caller did not ask for and does not notice.
func modelNamed(_ s: String?) -> (model: any ModelDescriptor, repo: String, id: String) {
    guard let s else {
        let first = modelTable[0]
        return (first.model, first.repo, first.id)
    }
    guard let entry = modelTable.first(where: { $0.aliases.contains(s) }) else {
        FileHandle.standardError.write(
            Data("error: unknown model '\(s)'. Known: \(modelNames.sorted().joined(separator: ", "))\n".utf8))
        exit(2)
    }
    return (entry.model, entry.repo, entry.id)
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
        let (model, repo, _) = modelNamed(args.count > 1 ? args[1] : nil)
        try await plan(repo: repo, model: model)

    case "chat":
        var options = Chat.Options()
        var promptParts: [String] = []
        var which = "20b"
        var index = 1
        while index < args.count {
            switch args[index] {
            case let name where modelNames.contains(name):
                which = name
            case "--tokens": index += 1; options.tokenCount = Int(args[index]) ?? 512
            case "--slots": index += 1; options.slotsPerLayer = Int(args[index])
            case "--prefill-chunk": index += 1; options.prefillChunk = Int(args[index])
            case "--context": index += 1; options.contextLength = Int(args[index]) ?? 4096
            case "--temperature": index += 1; options.temperature = Float(args[index])
            case "--top-p": index += 1; options.topP = Float(args[index])
            case "--top-k": index += 1; options.topKOverride = Int(args[index])
            case "--presence-penalty": index += 1
                options.presencePenaltyOverride = Float(args[index])
            case "--seed": index += 1; options.seed = UInt64(args[index]) ?? 0x5EED_1234
            case "--reasoning":
                index += 1
                options.reasoning = ReasoningLevel(rawValue: args[index]) ?? .medium
            case "--analysis": options.showAnalysis = true
            case "--dump-tokens": options.dumpTokens = true
            case "--instructions": index += 1; options.instructions = args[index]
            default: promptParts.append(args[index])
            }
            index += 1
        }
        let (chatModel, _, chatSlug) = modelNamed(which)
        let promptText = promptParts.isEmpty
            ? "Explain in three sentences why the sky is blue."
            : promptParts.joined(separator: " ")
        try Chat.run(
            model: chatModel,
            root: try defaultModelDirectory().appending(path: "\(chatSlug).hydra"),
            prompt: promptText, options: options)

    case "weights":
        let (wModel, _, wSlug) = modelNamed(args.count > 1 ? args[1] : nil)
        try Weights.run(
            model: wModel,
            root: try defaultModelDirectory().appending(path: "\(wSlug).hydra"))

    case "logits":
        var which = "20b"
        var promptParts: [String] = []
        var topK = 20
        var ctx = 1024
        var raw = false
        var trace = false
        var logitsReasoning = ReasoningLevel.off
        var index = 1
        while index < args.count {
            switch args[index] {
            case let name where modelNames.contains(name):
                which = name
            case "--top": index += 1; topK = Int(args[index]) ?? 20
            case "--context": index += 1; ctx = Int(args[index]) ?? 1024
            case "--raw": raw = true
            case "--trace": trace = true
            case "--reasoning":
                index += 1
                logitsReasoning = ReasoningLevel(rawValue: args[index]) ?? .off
            default: promptParts.append(args[index])
            }
            index += 1
        }
        let (logitsModel, _, logitsSlug) = modelNamed(which)
        try Logits.run(
            model: logitsModel,
            root: try defaultModelDirectory().appending(path: "\(logitsSlug).hydra"),
            prompt: promptParts.isEmpty ? "The capital of France is" : promptParts.joined(separator: " "),
            contextLength: ctx, topK: topK, raw: raw, trace: trace,
            reasoning: logitsReasoning)

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

    case "bench-gemv":
        try BenchGEMV.run(config: Gemma4MLXConfig.a4b)
        try BenchGEMV.runLayer(config: Gemma4MLXConfig.a4b)
        try BenchGEMV.runOverhead()
        try BenchGEMV.runPrefill(config: Gemma4MLXConfig.a4b)

    case "bench-gemm":
        let (gemmConfig, _) = configNamed(args.count > 1 ? args[1] : nil)
        try BenchGEMM.run(config: gemmConfig)

    case "vision":
        let which = args.count > 1 ? args[1] : "qwen-q4"
        let (_, _, visionSlug) = modelNamed(which)
        try VisionInspect.run(
            root: try defaultModelDirectory().appending(path: "\(visionSlug).hydra"),
            images: args.dropFirst(2).map { URL(fileURLWithPath: $0) })

    case "bench-attention":
        // Qwen's shape by default: the model M-067 found losing 48 % of its decode speed to
        // long context, and the one with no sliding window to bound it. The shape is
        // overridable because the grouped-query ratio is the variable under test.
        try BenchAttention.run(
            qHeads: args.count > 1 ? (Int(args[1]) ?? 16) : 16,
            kvHeads: args.count > 2 ? (Int(args[2]) ?? 2) : 2,
            headDim: args.count > 3 ? (Int(args[3]) ?? 128) : 128)

    case "bench-long-decode":
        let which = args.count > 1 ? args[1] : "qwen-q4"
        let (longModel, _, longSlug) = modelNamed(which)
        try BenchLongDecode.run(
            model: longModel,
            root: try defaultModelDirectory().appending(path: "\(longSlug).hydra"),
            contextTokens: args.count > 2 ? (Int(args[2]) ?? 21000) : 21000,
            rounds: args.count > 3 ? (Int(args[3]) ?? 6) : 6,
            tokensPerSample: args.count > 4 ? (Int(args[4]) ?? 4) : 4)

    case "bench-delta":
        let which = args.count > 1 ? args[1] : "qwen-q4"
        let (model, _, _) = modelNamed(which)
        guard let qwen = model as? Qwen35MoeConfig else {
            print("bench-delta is Qwen's recurrence; pass qwen-q4 or qwen-q8")
            exit(2)
        }
        try BenchDelta.run(config: qwen)

    case "bench-cold":
        let which = args.count > 1 ? args[1] : "qwen-q8"
        let (model, _, slug) = modelNamed(which)
        guard let qwen = model as? Qwen35MoeConfig else {
            print("bench-cold is written against Qwen's expert layout; pass qwen-q4 or qwen-q8")
            exit(2)
        }
        try BenchCold.run(
            config: qwen, root: try defaultModelDirectory().appending(path: "\(slug).hydra"))

    case "bench-map":
        let which = args.count > 1 ? args[1] : "qwen-q4"
        let (model, _, slug) = modelNamed(which)
        guard let qwen = model as? Qwen35MoeConfig else {
            print("bench-map is written against Qwen's expert layout; pass qwen-q4 or qwen-q8")
            exit(2)
        }
        try BenchMap.run(
            config: qwen, root: try defaultModelDirectory().appending(path: "\(slug).hydra"))

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
        let (tokModel, repo, slug) = modelNamed(which)
        let root = try defaultModelDirectory().appending(path: "\(slug).hydra")
        print("downloading \(repo)'s tokenizer…")
        let written = try await TokenizerInstaller(repo: repo).install(into: root)
        print("  \(written) files in \(root.path)/tokenizer")
        let tokenizer = try TokenizerInstaller.load(
            from: root, architecture: tokModel.architecture)
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
        let (model, repo, slug) = modelNamed(which)
        let directory = args.count > 2
            ? URL(fileURLWithPath: args[2]) : try defaultModelDirectory()
        try await install(repo: repo, model: model, slug: slug, into: directory)

    default:
        print("""
            usage: hydra <command>

              budget [context]          memory footprint and projected throughput, per cache policy
              plan [model]              computes and checks the repack plan, without downloading
              install [model] [dir]     installs the model in the .hydra format, streaming
              tokenizer [model]         installs the tokenizer into an existing installation
              verify [20b|120b] [dir]    compares installed windows against the upstream bytes
              probe [20b|120b] [ctx]     exercises mapping, the expert cache and the GPU kernels
              bench [20b|120b]           paired comparisons of I/O and kernels
              bench-gemm [20b|120b]      isolated bench of the dense projections
              bench-gemv                 isolated bench of the quantized GEMV
              bench-map [qwen-q4|qwen-q8]  pread into a slot against a mapped read
              bench-cold [qwen-q4|qwen-q8] the same four ways, on verified cold files
              bench-delta [qwen-q4|qwen-q8] the recurrence alone, at prefill's shape
              bench-attention [q kv dim] decode attention, split against unsplit, interleaved
              bench-long-decode [model] [ctx]  the same, end to end, over one long prefill
              generate [20b|120b] [n] [slots]  complete forward pass, throughput and footprint
              chat [model] <text> [options]
                  --tokens N --slots N --context N --prefill-chunk N
                  --temperature F --top-p F --top-k N --presence-penalty F --seed N
                  (all default to what the model itself publishes)
                  --reasoning off|low|medium|high --analysis --instructions "…"
              vision [model] [images…]  the installed tower, and what an image costs
              weights [model]           resident tensors as the runtime resolves them
              logits [model] <text> [--top N] [--raw] [--trace]
                                        what the model believes comes next, ranked
              inspect <file>            a safetensors header, without reading the data

            [model] is 20b, 120b, gemma, gemma-q4, qwen-q4 or qwen-q8. `install` and
            `chat` accept all of them; the
            measurement commands above them are written against GPT-OSS's tensors and
            take 20b or 120b only.
            """)
        exit(2)
    }
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n".utf8))
    exit(1)
}
