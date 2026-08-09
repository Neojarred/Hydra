import Testing

@testable import HydraCore

/// These tests lock our model of the checkpoint against the values **actually published**
/// by Hugging Face. The most important one is `installedSizeMatchesIndex`: if our
/// reconstruction of every tensor lands byte for byte on the declared `total_size`, the
/// model's structure is understood exactly — not approximated.
struct GptOssConfigTests {

    @Test("An expert blob is the same size for both models")
    func expertBlobSize() {
        #expect(GptOssConfig.b20.expertBlobBytes == 13_236_480)
        #expect(GptOssConfig.b120.expertBlobBytes == 13_236_480)
    }

    /// `metadata.total_size` of the official `model.safetensors.index.json`.
    @Test(
        "The reconstructed install matches the index total_size byte for byte",
        arguments: [
            (GptOssConfig.b20, 13_761_264_768),
            (GptOssConfig.b120, 65_248_815_744),
        ])
    func installedSizeMatchesIndex(config: GptOssConfig, declaredTotal: Int) {
        #expect(config.installedBytes == declaredTotal, "\(config.name)")
    }

    @Test("The 120B expert pool")
    func expertPool() {
        #expect(GptOssConfig.b120.expertPoolBytes == 60_993_699_840)
        #expect(GptOssConfig.b20.expertPoolBytes == 10_165_616_640)
    }

    @Test("The embedding is excluded from the resident weights")
    func embeddingIsNotResident() {
        let c = GptOssConfig.b120
        // We read one row per token: it stays mapped, outside the Metal working set.
        #expect(c.embeddingBytes == 1_158_266_880)
        #expect(c.residentBytes < c.installedBytes - c.expertPoolBytes)
        #expect(c.installedBytes - c.expertPoolBytes - c.residentBytes == c.embeddingBytes)
    }

    @Test("Attention alternates sliding and full starting at layer 0")
    func attentionAlternates() {
        let c = GptOssConfig.b120
        #expect(c.attentionPattern(atLayer: 0) == .sliding)
        #expect(c.attentionPattern(atLayer: 1) == .full)
        #expect(c.fullAttentionLayerCount == 18)
        #expect(c.slidingAttentionLayerCount == 18)
        #expect(c.groupedQueryFactor == 8)
    }

    @Test("The KV cache stays modest thanks to the 128 window")
    func kvCacheIsCheap() {
        let c = GptOssConfig.b120
        // 18 full layers x 2048 B/token, plus 18 bounded rings of 256 rows.
        #expect(c.kvBytesPerTokenPerFullLayer == 2048)
        #expect(c.kvCacheBytes(contextLength: 32768) == 1_217_396_736)
        // At 128k we stay under 5 GiB where all-full layers would cost twice that.
        #expect(c.kvCacheBytes(contextLength: 131_072) < 5 * 1_073_741_824)
    }

    @Test("The MXFP4 layout constants are consistent across modules")
    func layoutConstants() {
        #expect(MXFP4Layout.blockSize == 32)
        #expect(MXFP4Layout.packedBytesPerBlock == 16)
        // 32 values in 16 packed bytes + 1 scale byte = 4.25 bits per weight.
        let bitsPerWeight =
            Double((MXFP4Layout.packedBytesPerBlock + MXFP4Layout.scaleBytesPerBlock) * 8)
            / Double(MXFP4Layout.blockSize)
        #expect(bitsPerWeight == 4.25)
    }
}

/// The project's goal is to **reduce** the memory footprint. These tests therefore lock in
/// that the default policy is the minimum, and not "fill the ceiling".
struct MemoryBudgetTests {

    @Test("By default we keep only the strictly necessary experts",
          arguments: [GptOssConfig.b20, .b120])
    func defaultPolicyIsMinimal(config: GptOssConfig) {
        let b = MemoryBudget(config: config, contextLength: 8192)
        #expect(b.policy == .minimal)
        #expect(b.expertSlotsPerLayer == config.expertsPerToken)
        #expect(!b.isFullyResident, "the minimum must never load everything")
        #expect(b.totalFootprintBytes == b.minimumFootprintBytes)
    }

    /// This is the project's thesis, expressed as a single number.
    @Test("At the minimum, the footprint is a small fraction of the installed model")
    func minimalFootprintIsSmallFraction() {
        let b20 = MemoryBudget(config: GptOssConfig.b20, contextLength: 8192)
        #expect(b20.residentFractionOfCheckpoint < 0.35, "20B : \(b20.residentFractionOfCheckpoint)")

        let b120 = MemoryBudget(config: GptOssConfig.b120, contextLength: 8192)
        // The 120B is 60.8 GiB installed; we target under 10 % resident.
        #expect(b120.residentFractionOfCheckpoint < 0.10, "120B : \(b120.residentFractionOfCheckpoint)")
    }

    /// The floor is no longer the expert cache but the unquantized BF16 weights.
    /// Unlike Gemma 4, GPT-OSS keeps attention, routers and LM head in BF16.
    @Test("The floor is dominated by the resident weights, not by the experts")
    func residentWeightsDominateTheFloor() {
        for config in [GptOssConfig.b20, .b120] {
            let b = MemoryBudget(config: config, contextLength: 8192)
            #expect(b.residentBytes > b.expertCacheBytes, Comment(rawValue: config.name))
        }
    }

    @Test("An explicit memory target is respected")
    func memoryTargetIsHonoured() {
        let target = 6 * 1_073_741_824
        let b = MemoryBudget(
            config: GptOssConfig.b120, contextLength: 8192, policy: .memoryTarget(bytes: target))
        #expect(b.totalFootprintBytes <= target)
        // And it must buy more than the minimum.
        #expect(b.expertSlotsPerLayer > b.minimumSlotsPerLayer)
    }

    @Test("An unreachable target falls back to the minimum rather than failing")
    func impossibleTargetFallsBackToMinimum() {
        let b = MemoryBudget(
            config: GptOssConfig.b120, contextLength: 8192, policy: .memoryTarget(bytes: 1_000_000))
        #expect(b.expertSlotsPerLayer == b.minimumSlotsPerLayer)
    }

    @Test("The maximal policy serves as a correctness reference on the 20B")
    func maximizeIsTheCorrectnessReference() {
        let b = MemoryBudget(config: GptOssConfig.b20, contextLength: 8192, policy: .maximize)
        #expect(b.isFullyResident, "the 20B must be able to run with no I/O at all, for comparison")
        // The 120B never can be: that is the whole point of the project.
        let big = MemoryBudget(config: GptOssConfig.b120, contextLength: 8192, policy: .maximize)
        #expect(!big.isFullyResident)
    }

    @Test("At maximal policy, a longer context costs slots")
    func longerContextCostsSlotsWhenMaximizing() {
        let short = MemoryBudget(config: GptOssConfig.b120, contextLength: 8192, policy: .maximize)
        let long = MemoryBudget(config: GptOssConfig.b120, contextLength: 131_072, policy: .maximize)
        #expect(long.expertSlotsPerLayer < short.expertSlotsPerLayer)
        #expect(long.kvCacheBytes > short.kvCacheBytes)
    }

    @Test("The minimum, in contrast, does not depend on the context")
    func minimumIsContextIndependent() {
        let short = MemoryBudget(config: GptOssConfig.b120, contextLength: 8192)
        let long = MemoryBudget(config: GptOssConfig.b120, contextLength: 131_072)
        #expect(long.expertSlotsPerLayer == short.expertSlotsPerLayer)
        // Seul le KV cache grossit.
        #expect(long.totalFootprintBytes > short.totalFootprintBytes)
    }

    @Test("Slots are sized on the aligned blob, not on the raw sum")
    func slotsUseAlignedSize() {
        let config = GptOssConfig.b120
        #expect(config.expertSlotBytes >= config.expertBlobBytes)
        let b = MemoryBudget(config: config, contextLength: 8192)
        #expect(b.expertCacheBytes
            == b.expertSlotsPerLayer * config.layerCount * config.expertSlotBytes)
    }

    @Test("The compute ceiling does not depend on storage")
    func computeFloorIgnoresDisk() {
        var fastDisk = HardwareProfile.appleM4_24GB
        fastDisk.diskBandwidth = 1e15
        let b = MemoryBudget(config: GptOssConfig.b120, hardware: fastDisk, contextLength: 8192)
        #expect(b.estimatedTokensPerSecond(cacheHitRate: 0) <= b.maximumTokensPerSecond)
        #expect(b.maximumTokensPerSecond < 20)
        #expect(b.maximumTokensPerSecond > 17)
    }

    @Test("A perfect hit rate reaches the compute ceiling")
    func perfectCacheReachesFloor() {
        let b = MemoryBudget(config: GptOssConfig.b120, contextLength: 8192)
        #expect(b.config.diskBytesPerDecodedToken(cacheHitRate: 1.0) == 0)
        #expect(abs(b.estimatedTokensPerSecond(cacheHitRate: 1.0) - b.maximumTokensPerSecond) < 1e-9)
    }

    /// A much smaller machine must still be served: nothing in the sizing is specific to the
    /// development machine.
    @Test("The 20B fits at minimum on an 8 GiB machine")
    func smallMachineStillFits() {
        let small = HardwareProfile(
            metalWorkingSetCeiling: 5 * 1_073_741_824,  // ~ ce qu'expose une machine 8 Gio
            memoryBandwidth: 68e9, diskBandwidth: 2.5e9)
        let b = MemoryBudget(config: GptOssConfig.b20, hardware: small, contextLength: 4096)
        #expect(b.fits)
        #expect(b.expertSlotsPerLayer == b.minimumSlotsPerLayer)
        #expect(b.totalFootprintBytes <= small.metalWorkingSetCeiling)
    }
}
