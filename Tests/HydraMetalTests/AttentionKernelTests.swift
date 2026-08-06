import Foundation
import HydraCore
import HydraFormat
import HydraReference
import Testing

@testable import HydraMetal

/// La chaîne de confiance va d'OpenAI jusqu'au GPU : une transcription indépendante de
/// `gpt_oss/torch/model.py` produit des vecteurs de référence ; `HydraReference` les
/// reproduit à 1e-12 ; ces tests vérifient que les noyaux Metal reproduisent
/// `HydraReference`. Chaque maillon est vérifié séparément.
struct AttentionKernelTests {

    static func deterministic(_ count: Int, seed: UInt64) -> [Float] {
        var state = seed
        return (0..<count).map { _ in
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            z = z ^ (z >> 31)
            return Float(Double(z % 20000) / 10000.0 - 1.0)
        }
    }

    /// Écart rapporté à l'amplitude du vecteur, pas à chaque composante : près de zéro,
    /// l'écart relatif par composante mesure l'annulation catastrophique, pas le noyau.
    static func deviation(_ actual: [Float], _ expected: [Double]) -> Double {
        var scale = 0.0
        for value in expected { scale = max(scale, abs(value)) }
        var worst = 0.0
        for (a, e) in zip(actual, expected) {
            worst = max(worst, abs(Double(a) - e) / max(scale, 1e-9))
        }
        return worst
    }

    @Test("RMSNorm : le GPU concorde avec la référence CPU")
    func rmsNorm() throws {
        let kernels = AttentionKernels(context: try MetalContext())
        let size = 2880  // dimension réelle de GPT-OSS
        let x = Self.deterministic(size, seed: 11)
        let scale = Self.deterministic(size, seed: 12).map { $0 * 0.5 + 1.0 }

        let gpu = try kernels.rmsNorm(x, scale: scale)
        // L'échelle traverse un aller-retour BF16 côté GPU : la référence doit voir les
        // mêmes valeurs, sinon on mesurerait la quantization et non le noyau.
        let quantized = BF16.decode(BF16.encode(scale)).map(Double.init)
        let cpu = ReferenceOps.rmsNorm(x.map(Double.init), scale: quantized, eps: 1e-5)

        #expect(Self.deviation(gpu, cpu) < 1e-6)
    }

    @Test("RoPE : le GPU concorde, découpage en deux moitiés compris")
    func rope() throws {
        let kernels = AttentionKernels(context: try MetalContext())
        let heads = 8, headDim = 64
        let x = Self.deterministic(heads * headDim, seed: 21)

        let (cosTable, sinTable) = ReferenceOps.cosSin(positions: [137])
        let cos = cosTable[0].map(Float.init)
        let sin = sinTable[0].map(Float.init)

        let gpu = try kernels.applyRoPE(x, heads: heads, headDim: headDim, cos: cos, sin: sin)

        for head in 0..<heads {
            let slice = Array(x[(head * headDim)..<((head + 1) * headDim)]).map(Double.init)
            let cpu = ReferenceOps.applyRoPE(
                slice, cos: cos.map(Double.init), sin: sin.map(Double.init))
            let actual = Array(gpu[(head * headDim)..<((head + 1) * headDim)])
            #expect(Self.deviation(actual, cpu) < 1e-6, "tête \(head)")
        }
    }

    @Test("SwiGLU : le GPU reproduit l'écrêtage asymétrique et le +1")
    func swiglu() throws {
        let kernels = AttentionKernels(context: try MetalContext())
        var x = Self.deterministic(2 * 2880, seed: 31).map { $0 * 12.0 }
        // Valeurs qui franchissent les seuils dans les deux sens.
        x[0] = 20; x[1] = -20; x[2] = -20; x[3] = 20

        let gpu = try kernels.swiglu(x)
        let cpu = ReferenceOps.swiglu(x.map(Double.init))

        #expect(gpu.count == x.count / 2)
        #expect(Self.deviation(gpu, cpu) < 1e-6)
    }

    @Test("Attention avec puits : le GPU concorde, en attention pleine")
    func attentionFull() throws {
        let kernels = AttentionKernels(context: try MetalContext())
        let qHeads = 64, kvHeads = 8, headDim = 64, keyCount = 40  // GQA groupe 8, réel
        let qMult = qHeads / kvHeads

        let query = Self.deterministic(qHeads * headDim, seed: 41)
        let k = Self.deterministic(keyCount * kvHeads * headDim, seed: 42)
        let v = Self.deterministic(keyCount * kvHeads * headDim, seed: 43)
        let sinks = Self.deterministic(qHeads, seed: 44)
        let smScale = 1.0 / Float(headDim).squareRoot()

        let gpu = try kernels.attentionDecode(
            query: query, kCache: k.map(Float16.init), vCache: v.map(Float16.init),
            sinks: sinks, qHeads: qHeads, kvHeads: kvHeads, headDim: headDim,
            keyCount: keyCount, smScale: smScale)

        // Référence : une seule requête, à la dernière position, contre toutes les clés.
        // Le cache est en FP16 et les puits en BF16 côté GPU : la référence doit voir les
        // mêmes valeurs pour que l'écart mesuré soit celui du noyau.
        let kQuantized = k.map { Double(Float16($0)) }
        let vQuantized = v.map { Double(Float16($0)) }
        let sinkQuantized = BF16.decode(BF16.encode(sinks)).map(Double.init)

        for head in 0..<qHeads {
            let kvHead = head / qMult
            var accumulator = [Double](repeating: 0, count: headDim)
            var logits = [Double](repeating: 0, count: keyCount)
            var peak = sinkQuantized[head]
            for key in 0..<keyCount {
                var dot = 0.0
                for i in 0..<headDim {
                    dot += Double(query[head * headDim + i])
                        * kQuantized[(key * kvHeads + kvHead) * headDim + i]
                }
                logits[key] = dot * Double(smScale)
                peak = max(peak, logits[key])
            }
            var denominator = exp(sinkQuantized[head] - peak)
            for value in logits { denominator += exp(value - peak) }
            for key in 0..<keyCount {
                let weight = exp(logits[key] - peak) / denominator
                for i in 0..<headDim {
                    accumulator[i] += weight * vQuantized[(key * kvHeads + kvHead) * headDim + i]
                }
            }
            let actual = Array(gpu[(head * headDim)..<((head + 1) * headDim)])
            #expect(Self.deviation(actual, accumulator) < 1e-4, "tête \(head)")
        }
    }

    /// L'anneau des couches à fenêtre glissante doit donner exactement le même résultat
    /// qu'un stockage linéaire, tant que la fenêtre n'a pas débordé.
    @Test("Le cache circulaire équivaut au stockage linéaire avant débordement")
    func ringMatchesLinear() throws {
        let kernels = AttentionKernels(context: try MetalContext())
        let qHeads = 8, kvHeads = 2, headDim = 64, keyCount = 20
        let ringSize = 256

        let query = Self.deterministic(qHeads * headDim, seed: 51)
        let k = Self.deterministic(ringSize * kvHeads * headDim, seed: 52).map(Float16.init)
        let v = Self.deterministic(ringSize * kvHeads * headDim, seed: 53).map(Float16.init)
        let sinks = Self.deterministic(qHeads, seed: 54)
        let smScale = 1.0 / Float(headDim).squareRoot()

        let linear = try kernels.attentionDecode(
            query: query, kCache: k, vCache: v, sinks: sinks,
            qHeads: qHeads, kvHeads: kvHeads, headDim: headDim, keyCount: keyCount,
            ringSize: 0, startPosition: 0, smScale: smScale)
        let ring = try kernels.attentionDecode(
            query: query, kCache: k, vCache: v, sinks: sinks,
            qHeads: qHeads, kvHeads: kvHeads, headDim: headDim, keyCount: keyCount,
            ringSize: ringSize, startPosition: 0, smScale: smScale)

        #expect(linear == ring)
    }

    /// Un puits très négatif ne doit plus peser : l'attention redevient classique et
    /// les poids somment à 1. Avec une seule clé, la sortie est alors exactement V.
    @Test("Un puits négligeable rend l'attention classique")
    func sinkVanishes() throws {
        let kernels = AttentionKernels(context: try MetalContext())
        let qHeads = 4, kvHeads = 1, headDim = 64
        let query = Self.deterministic(qHeads * headDim, seed: 61)
        let k = Self.deterministic(kvHeads * headDim, seed: 62).map(Float16.init)
        let v = Self.deterministic(kvHeads * headDim, seed: 63).map(Float16.init)

        let negligible = try kernels.attentionDecode(
            query: query, kCache: k, vCache: v,
            sinks: [Float](repeating: -1e9, count: qHeads),
            qHeads: qHeads, kvHeads: kvHeads, headDim: headDim, keyCount: 1, smScale: 0.125)

        for head in 0..<qHeads {
            for i in 0..<headDim {
                #expect(abs(negligible[head * headDim + i] - Float(v[i])) < 1e-3)
            }
        }

        // Avec un puits nul, la masse se partage : la sortie doit s'éloigner de V.
        let active = try kernels.attentionDecode(
            query: query, kCache: k, vCache: v,
            sinks: [Float](repeating: 0, count: qHeads),
            qHeads: qHeads, kvHeads: kvHeads, headDim: headDim, keyCount: 1, smScale: 0.125)
        var differs = false
        for i in 0..<headDim where abs(active[i] - negligible[i]) > 1e-4 { differs = true }
        #expect(differs, "le puits n'a aucun effet : il n'est pas pris en compte")
    }

    @Test("Routeur : le GPU sélectionne et pondère comme la référence")
    func routerTopK() throws {
        let kernels = AttentionKernels(context: try MetalContext())
        for expertCount in [32, 128] {
            let logits = Self.deterministic(expertCount, seed: UInt64(70 + expertCount))
            let (indices, weights) = try kernels.routerTopK(logits, topK: 4)
            let cpu = ReferenceOps.router(logits.map(Double.init), topK: 4)

            #expect(indices == cpu.indices, "\(expertCount) experts")
            #expect(Self.deviation(weights, cpu.weights) < 1e-6)
            #expect(abs(weights.reduce(0, +) - 1.0) < 1e-5)
        }
    }
}
