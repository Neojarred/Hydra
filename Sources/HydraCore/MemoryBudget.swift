import Foundation

/// The hardware characteristics needed for sizing.
///
/// `HydraCore` does not import Metal (D-002): this profile is **injected** by the platform
/// layer, which queries `MTLDevice` at runtime. No production path may depend on the
/// defaults — they exist only for offline tooling and tests, and describe one machine among
/// others, not *the* machine.
public struct HardwareProfile: Sendable, Equatable {

    /// `MTLDevice.recommendedMaxWorkingSetSize`, or its equivalent on another platform.
    public var metalWorkingSetCeiling: Int

    /// GPU memory bandwidth on streaming reads, in bytes/s.
    public var memoryBandwidth: Double

    /// Storage bandwidth on reads the size of an expert blob, issued in parallel, with the page
    /// cache disabled. In bytes/s.
    public var diskBandwidth: Double

    public init(metalWorkingSetCeiling: Int, memoryBandwidth: Double, diskBandwidth: Double) {
        self.metalWorkingSetCeiling = metalWorkingSetCeiling
        self.memoryBandwidth = memoryBandwidth
        self.diskBandwidth = diskBandwidth
    }

    /// The reference development machine: MacBook Apple M4, 10 GPU cores, 24 GiB.
    /// Every value is measured — see docs/00-FEASIBILITY.md, §1.
    /// **To be used for offline tooling only.**
    public static let appleM4_24GB = HardwareProfile(
        metalWorkingSetCeiling: 19_069_665_280,  // 17,76 Gio
        memoryBandwidth: 94e9,
        diskBandwidth: 5.5e9
    )
}

/// Comment dimensionner le cache d'experts.
///
/// The project's point is to **reduce** the memory footprint, not to fill it. The slot count
/// is therefore never "whatever the hardware ceiling allows" by default: it is an explicit
/// policy, chosen knowingly.
public enum ExpertCachePolicy: Sendable, Equatable {

    /// The strict minimum to decode: one slot per selected expert at each layer.
    /// Floor footprint, minimal throughput. This is the configuration that demonstrates the
    /// project's thesis.
    case minimal

    /// A slot count per layer, imposed.
    case slotsPerLayer(Int)

    /// A total memory footprint targeted for the process, resident weights and KV included.
    case memoryTarget(bytes: Int)

    /// Whatever the hardware ceiling allows. Useful as a **correctness reference**: a fully
    /// resident run must produce exactly the same tokens as a run at the minimum. Not to be
    /// confused with a default mode.
    case maximize
}

/// A model's memory budget, for a given context length and cache policy.
///
/// Computed **at load time**, not at compile time: the context length is chosen by the user
/// when loading the model (D-005), and the hardware profile is read from the host
/// machine.
public struct MemoryBudget: Sendable {

    public let config: GptOssConfig
    public let hardware: HardwareProfile
    public let contextLength: Int
    public let scratchBytes: Int
    public let policy: ExpertCachePolicy

    /// The reusable scratch reserve: prefill arena, activation buffers, logits.
    ///
    /// A **provisional** value. The real need is modest — a 128-token prefill chunk is about
    /// 1.5 MiB of expert activations, 0.7 MiB of hidden states and 0.8 MiB of logits — but it
    /// will be measured rather than assumed, like everything else.
    public static let defaultScratchBytes = 128 * 1024 * 1024

    public init(
        config: GptOssConfig,
        hardware: HardwareProfile = .appleM4_24GB,
        contextLength: Int,
        policy: ExpertCachePolicy = .minimal,
        scratchBytes: Int = MemoryBudget.defaultScratchBytes
    ) {
        self.config = config
        self.hardware = hardware
        self.contextLength = contextLength
        self.policy = policy
        self.scratchBytes = scratchBytes
    }

    // MARK: - Postes incompressibles

    /// The weights that must occupy memory permanently: attention, routers, norms, LM head.
    /// The embedding is excluded — we read one row per token, so it stays mapped and paged on
    /// demand.
    public var residentBytes: Int { config.residentBytes }

    public var kvCacheBytes: Int { config.kvCacheBytes(contextLength: contextLength) }

    /// Everything that is not the expert cache. This is the model's absolute floor: no cache
    /// policy can go below it.
    public var fixedBytes: Int { residentBytes + kvCacheBytes + scratchBytes }

    // MARK: - Cache d'experts

    /// The minimum slots per layer: the experts selected for the current token must fit
    /// simultaneously.
    public var minimumSlotsPerLayer: Int { config.expertsPerToken }

    /// The most slots the hardware ceiling allows.
    public var ceilingSlotsPerLayer: Int {
        let free = hardware.metalWorkingSetCeiling - fixedBytes
        guard free > 0 else { return 0 }
        return min(config.expertCount, free / config.expertSlotBytes / config.layerCount)
    }

    /// The slots actually retained: the policy applied, then bounded by the hardware.
    public var expertSlotsPerLayer: Int {
        let requested: Int
        switch policy {
        case .minimal:
            requested = minimumSlotsPerLayer
        case .slotsPerLayer(let n):
            requested = n
        case .memoryTarget(let target):
            let free = target - fixedBytes
            requested = free <= 0 ? 0 : free / config.expertSlotBytes / config.layerCount
        case .maximize:
            requested = config.expertCount
        }
        return min(max(requested, minimumSlotsPerLayer), ceilingSlotsPerLayer)
    }

    public var expertCacheBytes: Int {
        expertSlotsPerLayer * config.layerCount * config.expertSlotBytes
    }

    /// The process's total memory footprint, excluding the system file cache.
    public var totalFootprintBytes: Int { fixedBytes + expertCacheBytes }

    /// The floor footprint reachable for this model and this context.
    public var minimumFootprintBytes: Int {
        fixedBytes + minimumSlotsPerLayer * config.layerCount * config.expertSlotBytes
    }

    /// True if the model does not fit, even at the strict minimum.
    public var fits: Bool { ceilingSlotsPerLayer >= minimumSlotsPerLayer }

    /// True if the whole expert pool is resident: no I/O at all after loading.
    /// This is the **correctness reference**, not a target.
    public var isFullyResident: Bool { expertSlotsPerLayer >= config.expertCount }

    /// The share of a layer's experts that fits in cache.
    public var expertCoverage: Double {
        Double(expertSlotsPerLayer) / Double(config.expertCount)
    }

    /// The share of the installed model that is resident. This is the number that expresses the
    /// project's thesis: running a 12.8 GiB model in a fraction of that.
    public var residentFractionOfCheckpoint: Double {
        Double(totalFootprintBytes - scratchBytes) / Double(config.installedBytes)
    }

    // MARK: - Throughput

    /// The compute floor: the time it takes just to move the weights through memory bandwidth.
    /// No I/O optimization gets below this threshold.
    public var computeFloorSeconds: Double {
        Double(config.gpuBytesPerDecodedToken) / hardware.memoryBandwidth
    }

    public var maximumTokensPerSecond: Double { 1.0 / computeFloorSeconds }

    /// A pessimistic estimate: compute and I/O strictly added.
    ///
    /// Since GPT-OSS has no shared expert, there is no dense branch to run during the reads —
    /// overlap is structurally weak (§2.2a). The serial model is therefore the honest
    /// assumption, not a precaution.
    public func estimatedTokensPerSecond(cacheHitRate: Double) -> Double {
        let io = Double(config.diskBytesPerDecodedToken(cacheHitRate: cacheHitRate))
            / hardware.diskBandwidth
        return 1.0 / (computeFloorSeconds + io)
    }

    /// A lower bound on the hit rate: the one we would get if the router chose uniformly. Real
    /// MoE routing is skewed, so the observed rate should be higher — milestone 1.7 measures by
    /// how much.
    public var uniformRoutingHitRate: Double { expertCoverage }

    // MARK: - Restitution

    public struct Line: Sendable {
        public let label: String
        public let bytes: Int
    }

    public var breakdown: [Line] {
        [
            Line(label: "Resident weights", bytes: residentBytes),
            Line(label: "KV cache FP16 (\(contextLength / 1024)k)", bytes: kvCacheBytes),
            Line(label: "Reusable scratch", bytes: scratchBytes),
            Line(
                label: "Expert cache (\(expertSlotsPerLayer)/\(config.expertCount) per layer)",
                bytes: expertCacheBytes),
        ]
    }
}
