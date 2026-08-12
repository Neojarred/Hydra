import Foundation
import HydraCore
import HydraFormat
import HydraMetal
import Metal

/// Exercises the complete chain on a model actually installed: zero-copy mapping, the SSD
/// expert cache, the MXFP4 kernel on the GPU, then checks the output against the validated
/// CPU decoder.
///
/// This is the first point where the repacker, the format, the cache and the kernels meet on
/// real data. A layout error that survived the synthetic tests shows up here.
///
enum Probe {

    static func run(config: GptOssConfig, root: URL, contextLength: Int) throws {
        let context = try MetalContext()
        let device = context.device

        print("GPU        \(device.name), family \(context.gpuFamily)")
        print("ceiling    \(gib(Int(device.recommendedMaxWorkingSetSize))) "
            + "(recommendedMaxWorkingSetSize)")
        print("maxBuffer  \(gib(device.maxBufferLength))")

        let bandwidth = context.measureMemoryBandwidth()
        print(String(format: "measured memory bandwidth: %.0f GB/s", bandwidth / 1e9))

        let profile = context.hardwareProfile(memoryBandwidth: bandwidth, diskBandwidth: 5.5e9)
        let budget = MemoryBudget(
            config: config, hardware: profile, contextLength: contextLength, policy: .minimal)

        // --- Mapping ---
        let baseline = MemoryFootprint.current()
        let start = Date()
        let mapping = try ModelMapping(root: root, model: config, device: device)
        let mapped = Date().timeIntervalSince(start)

        print("\nMAPPING")
        print(String(format: "  opened in %.0f ms", mapped * 1000))
        print("  resident.bin  \(gib(mapping.resident.byteCount)) wrapped without copying")
        if let embedding = mapping.embedding {
            print("  embed.bin     \(gib(embedding.byteCount)) mapped, not resident")
        } else {
            print("  embed.bin     absent, the embedding is tied to the output head")
        }
        print("  footprint after mapping: \(mib(MemoryFootprint.current()))"
            + "  (before: \(mib(baseline)))")

        // --- Expert cache ---
        let cache = ExpertSlotCache(
root: root, model: config,
            slotsPerLayer: budget.expertSlotsPerLayer, device: device)
        print("\nEXPERT CACHE")
        print("  minimal policy: \(budget.expertSlotsPerLayer) slots/layer "
            + "of \(config.expertCount)")
        print("  total reservation: \(gib(cache.reservedBytes))")

        // A cold read of a few experts, to measure what a miss really costs.
        let coldStart = Date()
        var reads = 0
        for layer in 0..<min(8, config.layerCount) {
            for expert in 0..<config.expertsPerToken {
                _ = try cache.expert(layer: layer, expert: expert)
                reads += 1
            }
        }
        let coldElapsed = Date().timeIntervalSince(coldStart)
        let coldBytes = reads * config.expertSlotBytes
        print(String(format: "  %d cold reads: %.0f ms, %.1f GB/s, %.2f ms per expert",
                     reads, coldElapsed * 1000,
                     Double(coldBytes) / coldElapsed / 1e9, coldElapsed / Double(reads) * 1000))

        // The re-read: must be served entirely from the cache.
        cache.resetStatistics()
        let warmStart = Date()
        for layer in 0..<min(8, config.layerCount) {
            for expert in 0..<config.expertsPerToken {
                _ = try cache.expert(layer: layer, expert: expert)
            }
        }
        let warmElapsed = Date().timeIntervalSince(warmStart)
        let stats = cache.statisticsSnapshot()
        print(String(format: "  re-read: %.2f ms, hit rate %.0f %%",
                     warmElapsed * 1000, stats.hitRate * 100))

        // --- The MXFP4 kernel on a real expert ---
        print("\nMXFP4 KERNEL ON REAL WEIGHTS")
        let blob = config.expertBlobLayout
        let (slot, _) = try cache.expert(layer: 0, expert: 0)

        let rows = 2 * config.intermediateSize  // gate_up: [5760, 2880]
        let cols = config.hiddenSize
        let input = (0..<cols).map { Float(sin(Double($0) * 0.01)) }

        guard let xBuffer = device.makeBuffer(
            length: cols * 4, options: .storageModeShared),
            let yBuffer = device.makeBuffer(length: rows * 4, options: .storageModeShared)
        else {
            print("  allocation failed")
            return
        }
        input.withUnsafeBytes { xBuffer.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }

        let kernels = MXFP4Kernels(context: context)
        guard let commandBuffer = context.commandQueue.makeCommandBuffer() else { return }
        let kernelStart = Date()
        try kernels.gemv(
            blocks: slot, blocksOffset: blob.gateUpBlocks.offset,
            scales: slot, scalesOffset: blob.gateUpScales.offset,
            bias: slot, biasOffset: blob.gateUpBias.offset,
            x: xBuffer, xOffset: 0,
            output: yBuffer, outputOffset: 0,
            rows: rows, cols: cols,
            in: commandBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        let kernelElapsed = Date().timeIntervalSince(kernelStart)

        let y = UnsafeBufferPointer(
            start: yBuffer.contents().bindMemory(to: Float.self, capacity: rows), count: rows)
        print(String(format: "  gate_up [%d x %d] in %.2f ms", rows, cols, kernelElapsed * 1000))
        print(String(format: "  output: min %.4f, max %.4f, %d non-finite",
                     y.min() ?? 0, y.max() ?? 0, y.filter { !$0.isFinite }.count))

        // --- Verification against the CPU decoder ---
        let checked = try verifyAgainstCPU(
            slot: slot, blob: blob, input: input, gpu: Array(y), rows: rows, cols: cols)
        print(String(format: "  %d rows recomputed on CPU, worst relative deviation %.2e",
                     checked.rows, checked.worstRelative))
        if checked.worstRelative < 1e-4 {
            print("  ✔ the GPU kernel agrees with the CPU decoder on real weights")
        } else {
            print("  ✘ divergence, the blob layout or the kernel is at fault")
        }

        print("\nFINAL FOOTPRINT: \(mib(MemoryFootprint.current()))")
        print("installed model  : \(gib(config.installedBytes))")
    }

    /// Recomputes a few rows on the CPU from the slot's bytes, with the bit-exact MXFP4
    /// decoder, summing in double precision.
    private static func verifyAgainstCPU(
        slot: MTLBuffer, blob: ExpertBlobLayout, input: [Float], gpu: [Float],
        rows: Int, cols: Int, sampleRows: Int = 16
    ) throws -> (rows: Int, worstRelative: Double) {
        let blocksPerRow = cols / MXFP4Layout.blockSize
        let bytesPerRow = blocksPerRow * MXFP4Layout.packedBytesPerBlock
        let base = slot.contents()

        var worst = 0.0
        let step = max(1, rows / sampleRows)
        var examined = 0

        for row in stride(from: 0, to: rows, by: step) {
            let packed = Data(
                bytes: base.advanced(by: blob.gateUpBlocks.offset + row * bytesPerRow),
                count: bytesPerRow)
            let scales = Data(
                bytes: base.advanced(by: blob.gateUpScales.offset + row * blocksPerRow),
                count: blocksPerRow)
            let weights = try MXFP4.decode(packed: packed, scales: scales)

            var expected = 0.0
            for c in 0..<cols { expected += Double(weights[c]) * Double(input[c]) }
            let biasBits = base.advanced(by: blob.gateUpBias.offset + row * 2)
                .loadUnaligned(as: UInt16.self)
            expected += Double(BF16.toFloat(UInt16(littleEndian: biasBits)))

            worst = max(worst, abs(Double(gpu[row]) - expected) / max(abs(expected), 1e-4))
            examined += 1
        }
        return (examined, worst)
    }
}
