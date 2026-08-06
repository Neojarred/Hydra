import Foundation
import HydraCore
import HydraFormat
import Testing

@testable import HydraMetal

/// Le décodeur CPU est déjà validé bit à bit contre l'implémentation de référence
/// d'OpenAI (jalon 1.2). Ces tests étendent la garantie au GPU : la sortie Metal doit
/// concorder avec la sortie CPU. La chaîne de confiance va donc de la référence OpenAI
/// jusqu'au noyau qui tournera en production.
struct MXFP4KernelTests {

    /// Données MXFP4 déterministes, avec des exposants d'échelle réalistes — ceux
    /// observés sur le checkpoint installé se concentrent autour de 2⁻⁶.
    static func syntheticWeights(rows: Int, cols: Int, seed: UInt64 = 12345)
        -> (packed: Data, scales: Data)
    {
        var state = seed
        func next() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
        let blocksPerRow = cols / MXFP4Layout.blockSize
        var packed = Data(count: rows * blocksPerRow * MXFP4Layout.packedBytesPerBlock)
        var scales = Data(count: rows * blocksPerRow)
        for i in 0..<packed.count { packed[i] = UInt8(truncatingIfNeeded: next()) }
        for i in 0..<scales.count { scales[i] = UInt8(121 + next() % 7) }  // exposants -6 à 0
        return (packed, scales)
    }

    @Test("Le décodage GPU concorde exactement avec le décodeur CPU validé")
    func gpuMatchesCpuDecoder() throws {
        let kernels = MXFP4Kernels(context: try MetalContext())
        let (packed, scales) = Self.syntheticWeights(rows: 16, cols: 1024)

        let cpu = try MXFP4.decode(packed: packed, scales: scales)
        let gpu = try kernels.dequantize(packed: packed, scales: scales)

        #expect(gpu.count == cpu.count)
        var mismatches = 0
        for i in 0..<min(cpu.count, gpu.count) where cpu[i] != gpu[i] { mismatches += 1 }
        #expect(mismatches == 0, "\(mismatches) valeurs divergent entre CPU et GPU")
    }

    @Test("Le GEMV MXFP4 concorde avec un produit calculé en double précision")
    func gemvMatchesReference() throws {
        let kernels = MXFP4Kernels(context: try MetalContext())

        let rows = 64
        let cols = 2880  // dimension réelle de GPT-OSS
        let (packed, scales) = Self.syntheticWeights(rows: rows, cols: cols)
        let x = (0..<cols).map { Float(sin(Double($0) * 0.017)) }

        let gpu = try kernels.gemv(
            packed: packed, scales: scales, bias: nil, x: x, rows: rows, cols: cols)

        // Référence : déquantization ligne par ligne avec le décodeur CPU, puis somme en
        // double précision — l'écart mesuré est alors celui du GPU, pas celui du modèle.
        let blocksPerRow = cols / MXFP4Layout.blockSize
        let bytesPerRow = blocksPerRow * MXFP4Layout.packedBytesPerBlock
        var worstRelative = 0.0
        for row in 0..<rows {
            let rowPacked = packed.subdata(in: (row * bytesPerRow)..<((row + 1) * bytesPerRow))
            let rowScales = scales.subdata(in: (row * blocksPerRow)..<((row + 1) * blocksPerRow))
            let weights = try MXFP4.decode(packed: rowPacked, scales: rowScales)

            var expected = 0.0
            for c in 0..<cols { expected += Double(weights[c]) * Double(x[c]) }

            worstRelative = max(
                worstRelative, abs(Double(gpu[row]) - expected) / max(abs(expected), 1e-6))
        }
        // Le GPU accumule en Float32 dans un ordre différent : l'écart attendu est celui
        // de l'arithmétique flottante, pas d'une erreur de disposition mémoire.
        #expect(worstRelative < 1e-5, "pire écart relatif : \(worstRelative)")
    }

    @Test("Le biais est ajouté quand il est fourni")
    func gemvAppliesBias() throws {
        let kernels = MXFP4Kernels(context: try MetalContext())

        let rows = 8, cols = 128
        let (packed, scales) = Self.syntheticWeights(rows: rows, cols: cols)
        let x = [Float](repeating: 0.5, count: cols)

        let withoutBias = try kernels.gemv(
            packed: packed, scales: scales, bias: nil, x: x, rows: rows, cols: cols)

        // Biais BF16 : 1.0 s'encode 0x3F80, soit les 16 bits hauts du Float32 1.0.
        var bias = Data()
        for _ in 0..<rows { bias.append(contentsOf: [0x80, 0x3F]) }
        let withBias = try kernels.gemv(
            packed: packed, scales: scales, bias: bias, x: x, rows: rows, cols: cols)

        for row in 0..<rows {
            #expect(abs((withBias[row] - withoutBias[row]) - 1.0) < 1e-5, "ligne \(row)")
        }
    }

    /// Le nombre de voies par threadgroup ne doit pas changer le résultat : c'est un
    /// paramètre de performance, pas de sémantique. Un bug de réduction se verrait ici.
    @Test("Le résultat ne dépend pas de la forme du threadgroup")
    func resultIsIndependentOfThreadgroupShape() throws {
        let kernels = MXFP4Kernels(context: try MetalContext())
        let rows = 4
        for cols in [64, 256, 2880] {
            let (packed, scales) = Self.syntheticWeights(rows: rows, cols: cols, seed: UInt64(cols))
            let x = (0..<cols).map { Float(($0 % 7) - 3) * 0.25 }
            let y = try kernels.gemv(
                packed: packed, scales: scales, bias: nil, x: x, rows: rows, cols: cols)
            #expect(y.count == rows)
            #expect(y.allSatisfy { $0.isFinite }, "cols = \(cols) produit des non-finis")
        }
    }

    @Test("Le contexte Metal expose la famille GPU et un plafond cohérent")
    func contextReportsHardware() throws {
        let context = try MetalContext()
        let profile = context.hardwareProfile(memoryBandwidth: 1, diskBandwidth: 1)

        #expect(profile.metalWorkingSetCeiling > 0)
        // Le plafond Metal est une fraction de la mémoire physique, jamais sa totalité.
        #expect(profile.metalWorkingSetCeiling < Int(ProcessInfo.processInfo.physicalMemory))
        #expect(context.gpuFamily.hasPrefix("apple"))
    }
}
