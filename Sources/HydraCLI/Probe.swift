import Foundation
import HydraCore
import HydraFormat
import HydraMetal
import Metal

/// Exerce la chaîne complète sur un modèle réellement installé : mappage sans copie,
/// cache d'experts sur SSD, noyau MXFP4 sur GPU — puis vérifie la sortie contre le
/// décodeur CPU validé.
///
/// C'est le premier point où le repacker, le format, le cache et les noyaux se
/// rencontrent sur de vraies données. Une erreur de disposition qui aurait survécu aux
/// tests synthétiques se voit ici.
enum Probe {

    static func run(config: GptOssConfig, root: URL, contextLength: Int) throws {
        let context = try MetalContext()
        let device = context.device

        print("GPU        \(device.name), famille \(context.gpuFamily)")
        print("plafond    \(gib(Int(device.recommendedMaxWorkingSetSize))) "
            + "(recommendedMaxWorkingSetSize)")
        print("maxBuffer  \(gib(device.maxBufferLength))")

        let bandwidth = context.measureMemoryBandwidth()
        print(String(format: "bande passante mémoire mesurée : %.0f Go/s", bandwidth / 1e9))

        let profile = context.hardwareProfile(memoryBandwidth: bandwidth, diskBandwidth: 5.5e9)
        let budget = MemoryBudget(
            config: config, hardware: profile, contextLength: contextLength, policy: .minimal)

        // --- Mappage ---
        let baseline = MemoryFootprint.current()
        let start = Date()
        let mapping = try ModelMapping(root: root, config: config, device: device)
        let mapped = Date().timeIntervalSince(start)

        print("\nMAPPAGE")
        print(String(format: "  ouvert en %.0f ms", mapped * 1000))
        print("  resident.bin  \(gib(mapping.resident.byteCount)) enveloppé sans copie")
        print("  embed.bin     \(gib(mapping.embedding.byteCount)) mappé, non résident")
        print("  empreinte après mappage : \(mib(MemoryFootprint.current()))"
            + "  (avant : \(mib(baseline)))")

        // --- Cache d'experts ---
        let cache = ExpertSlotCache(
            root: root, config: config,
            slotsPerLayer: budget.expertSlotsPerLayer, device: device)
        print("\nCACHE D'EXPERTS")
        print("  politique minimale : \(budget.expertSlotsPerLayer) slots/couche "
            + "sur \(config.expertCount)")
        print("  réservation totale : \(gib(cache.reservedBytes))")

        // Lecture froide de quelques experts, pour mesurer le coût réel d'un miss.
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
        print(String(format: "  %d lectures froides : %.0f ms, %.1f Go/s, %.2f ms par expert",
                     reads, coldElapsed * 1000,
                     Double(coldBytes) / coldElapsed / 1e9, coldElapsed / Double(reads) * 1000))

        // Relecture : doit être entièrement servie par le cache.
        cache.resetStatistics()
        let warmStart = Date()
        for layer in 0..<min(8, config.layerCount) {
            for expert in 0..<config.expertsPerToken {
                _ = try cache.expert(layer: layer, expert: expert)
            }
        }
        let warmElapsed = Date().timeIntervalSince(warmStart)
        let stats = cache.statisticsSnapshot()
        print(String(format: "  relecture : %.2f ms, taux de hit %.0f %%",
                     warmElapsed * 1000, stats.hitRate * 100))

        // --- Noyau MXFP4 sur un vrai expert ---
        print("\nNOYAU MXFP4 SUR POIDS RÉELS")
        let blob = config.expertBlobLayout
        let (slot, _) = try cache.expert(layer: 0, expert: 0)

        let rows = 2 * config.intermediateSize  // gate_up : [5760, 2880]
        let cols = config.hiddenSize
        let input = (0..<cols).map { Float(sin(Double($0) * 0.01)) }

        guard let xBuffer = device.makeBuffer(
            length: cols * 4, options: .storageModeShared),
            let yBuffer = device.makeBuffer(length: rows * 4, options: .storageModeShared)
        else {
            print("  allocation impossible")
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
        print(String(format: "  gate_up [%d x %d] en %.2f ms", rows, cols, kernelElapsed * 1000))
        print(String(format: "  sortie : min %.4f, max %.4f, %d non finis",
                     y.min() ?? 0, y.max() ?? 0, y.filter { !$0.isFinite }.count))

        // --- Vérification contre le décodeur CPU ---
        let checked = try verifyAgainstCPU(
            slot: slot, blob: blob, input: input, gpu: Array(y), rows: rows, cols: cols)
        print(String(format: "  %d lignes recalculées sur CPU, pire écart relatif %.2e",
                     checked.rows, checked.worstRelative))
        if checked.worstRelative < 1e-4 {
            print("  ✔ le noyau GPU concorde avec le décodeur CPU sur des poids réels")
        } else {
            print("  ✘ divergence — la disposition du blob ou le noyau est en cause")
        }

        print("\nEMPREINTE FINALE : \(mib(MemoryFootprint.current()))")
        print("modèle installé  : \(gib(config.installedBytes))")
    }

    /// Recalcule quelques lignes sur CPU à partir des octets du slot, avec le décodeur
    /// MXFP4 validé bit à bit, et somme en double précision.
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
