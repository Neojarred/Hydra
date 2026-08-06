import Foundation
import HydraCore
import HydraFormat
import HydraMetal
import Metal

/// Comparaisons appariées sur le modèle réellement installé.
///
/// Discipline reprise de TurboFieldfare : contrôle et candidat **alternent** au lieu de
/// tourner l'un après l'autre, parce que l'état thermique, le cache de pages et la
/// fréquence GPU dérivent pendant une mesure. On rapporte la médiane, et un candidat ne
/// devient le défaut que s'il gagne de façon reproductible **sans changer les sorties**.
enum Bench {

    static func median(_ values: [Double]) -> Double {
        let sorted = values.sorted()
        guard !sorted.isEmpty else { return 0 }
        return sorted.count % 2 == 1
            ? sorted[sorted.count / 2]
            : (sorted[sorted.count / 2 - 1] + sorted[sorted.count / 2]) / 2
    }

    static func run(config: GptOssConfig, root: URL) throws {
        let context = try MetalContext()
        let device = context.device
        let kernels = MXFP4Kernels(context: context)
        let blob = config.expertBlobLayout

        try benchmarkIO(config: config, root: root, device: device)
        try benchmarkKernels(
            config: config, root: root, device: device, kernels: kernels, blob: blob)
    }

    // MARK: - Entrées/sorties

    private static func benchmarkIO(
        config: GptOssConfig, root: URL, device: MTLDevice
    ) throws {
        print("LECTURE D'EXPERTS")
        print("  Chaque tour ouvre un cache neuf sur une couche encore jamais lue, et")
        print("  contourne le cache de pages : sans quoi on mesure la RAM, pas le SSD.")
        print("  C'est le piège que TurboFieldfare a documenté — des pages chaudes font")
        print("  paraître n'importe quelle stratégie de lecture excellente.\n")

        let experts = Array(0..<config.expertsPerToken)
        let bytes = Double(config.expertsPerToken * config.expertSlotBytes)

        for bypass in [true, false] {
            var serial: [Double] = []
            var parallel: [Double] = []
            // Couches distinctes à chaque mesure : une couche déjà lue fausserait la suite.
            var layer = 0
            for _ in 0..<5 {
                let a = ExpertSlotCache(
                    root: root, config: config, slotsPerLayer: config.expertsPerToken,
                    device: device, bypassPageCache: bypass)
                var start = Date()
                for id in experts { _ = try a.expert(layer: layer % config.layerCount, expert: id) }
                serial.append(Date().timeIntervalSince(start))
                layer += 1

                let b = ExpertSlotCache(
                    root: root, config: config, slotsPerLayer: config.expertsPerToken,
                    device: device, bypassPageCache: bypass)
                start = Date()
                try b.load(layer: layer % config.layerCount, experts: experts)
                parallel.append(Date().timeIntervalSince(start))
                layer += 1
            }

            let s = median(serial), p = median(parallel)
            print("  \(bypass ? "F_NOCACHE (froid, ce que coûte un vrai miss)" : "cache de pages autorisé")")
            print(String(format: "    série     %6.1f ms   %.2f Go/s", s * 1000, bytes / s / 1e9))
            print(String(format: "    parallèle %6.1f ms   %.2f Go/s   ×%.2f",
                         p * 1000, bytes / p / 1e9, s / p))
        }
    }

    // MARK: - Noyaux

    private static func benchmarkKernels(
        config: GptOssConfig, root: URL, device: MTLDevice,
        kernels: MXFP4Kernels, blob: ExpertBlobLayout
    ) throws {
        print("\nGEMV MXFP4 — trois variantes, sur un expert réel")

        let cache = ExpertSlotCache(
            root: root, config: config, slotsPerLayer: config.expertsPerToken, device: device)
        let (slot, _) = try cache.expert(layer: 0, expert: 0)

        let rows = 2 * config.intermediateSize
        let cols = config.hiddenSize
        let input = (0..<cols).map { Float(sin(Double($0) * 0.01)) }

        guard let xBuffer = device.makeBuffer(length: cols * 4, options: .storageModeShared)
        else { return }
        input.withUnsafeBytes {
            xBuffer.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count)
        }

        let variants = ["mxfp4_gemv", "mxfp4_gemv_vectorized", "mxfp4_gemv_simd", "mxfp4_gemv_tiled"]
        var outputs: [String: MTLBuffer] = [:]
        for name in variants {
            guard let buffer = device.makeBuffer(length: rows * 4, options: .storageModeShared)
            else { return }
            outputs[name] = buffer
        }

        // Un aller-retour CPU-GPU coûte quelques centaines de microsecondes — du même
        // ordre que le noyau lui-même. Mesurer une passe par tampon de commandes revient
        // donc à chronométrer la synchronisation, ce qui fait paraître toutes les
        // variantes identiques. On encode `iterations` passes dans un seul tampon et on
        // divise : la latence est alors amortie et l'écart entre variantes redevient
        // visible.
        let iterations = 50
        func time(_ function: String) throws -> Double {
            guard let commandBuffer = context(kernels).commandQueue.makeCommandBuffer()
            else { return 0 }
            for _ in 0..<iterations {
                try kernels.gemv(
                    function: function,
                    blocks: slot, blocksOffset: blob.gateUpBlocks.offset,
                    scales: slot, scalesOffset: blob.gateUpScales.offset,
                    bias: slot, biasOffset: blob.gateUpBias.offset,
                    x: xBuffer, xOffset: 0, output: outputs[function]!, outputOffset: 0,
                    rows: rows, cols: cols, in: commandBuffer)
            }
            let start = Date()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            return Date().timeIntervalSince(start) / Double(iterations)
        }

        // Coût d'un aller-retour à vide, pour situer ce que la mesure précédente incluait.
        let emptyStart = Date()
        for _ in 0..<20 {
            guard let empty = context(kernels).commandQueue.makeCommandBuffer() else { break }
            empty.commit()
            empty.waitUntilCompleted()
        }
        print(String(format: "  latence d'un aller-retour CPU-GPU à vide : %.0f µs",
                     Date().timeIntervalSince(emptyStart) / 20 * 1e6))

        // Chauffe : la première exécution paie la construction du pipeline.
        for name in variants { _ = try time(name) }

        var samples: [String: [Double]] = [:]
        for _ in 0..<7 {
            for name in variants { samples[name, default: []].append(try time(name)) }
        }

        // Référence de correction : le décodeur CPU validé bit à bit, sommé en double.
        let reference = try cpuReference(
            slot: slot, blob: blob, input: input, rows: rows, cols: cols)

        let weightBytes = Double(rows * (cols / 32) * 17)  // 16 octets packés + 1 d'échelle
        print("  " + pad("variante", 26) + pad("ms", 9) + pad("Go/s", 10) + "écart / CPU")
        var best = (name: "", time: Double.greatestFiniteMagnitude)
        for name in variants {
            let elapsed = median(samples[name] ?? [])
            let error = deviation(outputs[name]!, reference: reference, rows: rows)
            print("  " + pad(name, 26)
                + pad(String(format: "%.2f", elapsed * 1000), 9)
                + pad(String(format: "%.1f", weightBytes / elapsed / 1e9), 10)
                + String(format: "%.2e", error))
            if error < 1e-4, elapsed < best.time { best = (name, elapsed) }
        }

        let referenceTime = median(samples["mxfp4_gemv"] ?? [])
        if !best.name.isEmpty {
            print(String(format: "\n  meilleur candidat correct : %@ (×%.2f)",
                         best.name as NSString, referenceTime / best.time))
        }

        // Le noyau reste très loin de la bande passante mémoire : c'est le prochain
        // chantier, pas une conclusion.
        let bandwidth = context(kernels).measureMemoryBandwidth()
        print(String(format: "  bande passante mémoire de la machine : %.0f Go/s", bandwidth / 1e9))
        print(String(format: "  le meilleur noyau en exploite %.0f %%",
                     weightBytes / best.time / bandwidth * 100))

        // ------------------------------------------------------- Extrapolation
        let perExpert = best.time * 1.5  // gate_up puis down, moitié moins large
        let moePerToken = perExpert * Double(config.expertsPerToken * config.layerCount)
        print(String(format: "\n  MoE seul, extrapolé : %.0f ms/token → %.1f tok/s",
                     moePerToken * 1000, 1 / moePerToken))
        print("  (attention, tête LM et I/O non comprises — borne haute optimiste)")
    }

    private static func context(_ kernels: MXFP4Kernels) -> MetalContext { kernels.context }

    /// Produit de référence, calculé en double précision depuis les octets du slot.
    private static func cpuReference(
        slot: MTLBuffer, blob: ExpertBlobLayout, input: [Float], rows: Int, cols: Int
    ) throws -> [Double] {
        let blocksPerRow = cols / MXFP4Layout.blockSize
        let bytesPerRow = blocksPerRow * MXFP4Layout.packedBytesPerBlock
        let base = slot.contents()
        var out = [Double](repeating: 0, count: rows)

        for row in 0..<rows {
            let packed = Data(
                bytesNoCopy: base.advanced(by: blob.gateUpBlocks.offset + row * bytesPerRow),
                count: bytesPerRow, deallocator: .none)
            let scales = Data(
                bytesNoCopy: base.advanced(by: blob.gateUpScales.offset + row * blocksPerRow),
                count: blocksPerRow, deallocator: .none)
            let weights = try MXFP4.decode(packed: packed, scales: scales)

            var sum = 0.0
            for c in 0..<cols { sum += Double(weights[c]) * Double(input[c]) }
            let biasBits = base.advanced(by: blob.gateUpBias.offset + row * 2)
                .loadUnaligned(as: UInt16.self)
            out[row] = sum + Double(BF16.toFloat(UInt16(littleEndian: biasBits)))
        }
        return out
    }

    /// Écart relatif rapporté à l'amplitude du **vecteur**, pas de chaque composante.
    ///
    /// Rapporter à la composante donne des chiffres alarmants sur les sorties proches de
    /// zéro, où la moindre annulation catastrophique domine — un artefact de la mesure,
    /// pas un défaut du noyau. L'amplitude du vecteur est la grandeur qui compte pour la
    /// suite du calcul.
    private static func deviation(_ buffer: MTLBuffer, reference: [Double], rows: Int) -> Double {
        let values = UnsafeBufferPointer(
            start: buffer.contents().bindMemory(to: Float.self, capacity: rows), count: rows)
        var worst = 0.0
        var scale = 0.0
        for i in 0..<rows { scale = max(scale, abs(reference[i])) }
        for i in 0..<rows {
            worst = max(worst, abs(Double(values[i]) - reference[i]) / max(scale, 1e-6))
        }
        return worst
    }
}
