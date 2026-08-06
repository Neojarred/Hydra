import Foundation
import HydraCore

/// Implémentations CPU de référence des opérateurs de GPT-OSS, en double précision.
///
/// Elles ne servent pas à l'inférence — beaucoup trop lentes — mais de **vérité de
/// terrain** pour valider les noyaux Metal. C'est le même dispositif que pour MXFP4 :
/// une implémentation lente et évidemment correcte, contre laquelle on mesure l'écart
/// d'une implémentation rapide et subtile.
///
/// Chaque opérateur est vérifié contre un vecteur produit par une transcription
/// indépendante de `gpt_oss/torch/model.py` (voir `tools/gen_reference_fixtures.py`).
/// Les détails ci-dessous ne se devinent pas, et se tromper sur l'un d'eux donne un
/// modèle qui génère du texte plausible mais dégradé, sans jamais lever d'erreur.
public enum ReferenceOps {

    // MARK: - RMSNorm

    /// `x / sqrt(mean(x²) + eps) * scale`, moyenne calculée en double.
    public static func rmsNorm(_ x: [Double], scale: [Double], eps: Double = 1e-5) -> [Double] {
        precondition(x.count == scale.count)
        let meanSquare = x.reduce(0) { $0 + $1 * $1 } / Double(x.count)
        let inverse = 1.0 / (meanSquare + eps).squareRoot()
        return zip(x, scale).map { $0 * inverse * $1 }
    }

    // MARK: - RoPE avec extension YaRN

    /// Alias vers `HydraCore.RoPETables.Parameters`.
    ///
    /// Le calcul YaRN n'est **pas** réimplémenté ici : le runtime et la référence
    /// partagent la même source, dans `HydraCore`. L'indépendance qui valide ce calcul
    /// vient d'ailleurs — une transcription Python du code d'OpenAI, dans les fixtures.
    /// Dupliquer l'implémentation en Swift n'apporterait rien et créerait un risque de
    /// divergence.
    public typealias YarnParameters = RoPETables.Parameters

    /// Concentration YaRN et fréquences inverses.
    public static func yarn(_ p: YarnParameters) -> (concentration: Double, invFreq: [Double]) {
        let tables = RoPETables(p)
        return (tables.concentration, tables.inverseFrequencies)
    }

    /// Tables cos/sin pour un jeu de positions, concentration déjà appliquée.
    public static func cosSin(
        positions: [Int], parameters: YarnParameters = .gptOss
    ) -> (cos: [[Double]], sin: [[Double]]) {
        let tables = RoPETables(parameters)
        var cosTable: [[Double]] = []
        var sinTable: [[Double]] = []
        for position in positions {
            let (c, s) = tables.tables(at: position)
            cosTable.append(c)
            sinTable.append(s)
        }
        return (cosTable, sinTable)
    }

    /// Applique RoPE à un vecteur de tête.
    ///
    /// **Découpage en deux moitiés**, pas en paires entrelacées : `x1` est la première
    /// moitié des composantes, `x2` la seconde. C'est l'inverse du SwiGLU du même modèle,
    /// qui lui découpe en indices pairs et impairs — deux conventions opposées dans la
    /// même architecture, et aucune erreur ne signalera une inversion.
    public static func applyRoPE(_ head: [Double], cos: [Double], sin: [Double]) -> [Double] {
        let half = head.count / 2
        precondition(cos.count == half && sin.count == half)
        var out = [Double](repeating: 0, count: head.count)
        for i in 0..<half {
            let x1 = head[i], x2 = head[half + i]
            out[i] = x1 * cos[i] - x2 * sin[i]
            out[half + i] = x2 * cos[i] + x1 * sin[i]
        }
        return out
    }

    // MARK: - SwiGLU

    /// SwiGLU de GPT-OSS. Trois écarts par rapport à la formulation habituelle.
    ///
    /// 1. **Découpage en indices pairs et impairs**, pas en deux moitiés : les lignes de
    ///    `gate_up_proj` sont entrelacées `[gate₀, up₀, gate₁, up₁, …]`.
    /// 2. **Écrêtage asymétrique** : la branche gate n'est bornée que par le haut, la
    ///    branche linéaire des deux côtés.
    /// 3. **La branche linéaire reçoit +1**, et le swish utilise `sigmoid(1,702·x)` et non
    ///    `sigmoid(x)`.
    public static func swiglu(_ row: [Double], alpha: Double = 1.702, limit: Double = 7.0)
        -> [Double]
    {
        precondition(row.count % 2 == 0)
        var out = [Double](repeating: 0, count: row.count / 2)
        for i in 0..<out.count {
            let gate = min(row[2 * i], limit)
            let linear = min(max(row[2 * i + 1], -limit), limit)
            let activated = gate * (1.0 / (1.0 + exp(-alpha * gate)))
            out[i] = activated * (linear + 1.0)
        }
        return out
    }

    // MARK: - Attention

    /// Attention avec **puits** (attention sinks) et fenêtre glissante facultative.
    ///
    /// Le puits est un logit appris par tête, ajouté comme **colonne supplémentaire** dans
    /// le softmax puis retirée. Il n'apporte donc aucune valeur au résultat : il grossit
    /// le dénominateur, ce qui permet à une tête de « ne rien regarder » en répartissant
    /// sa masse sur le puits. Mécanisme absent de Gemma 4, donc sans précédent dans
    /// TurboFieldfare.
    ///
    /// - Parameters:
    ///   - q: `[tokens][kvHeads][qMult][headDim]`
    ///   - k, v: `[tokens][kvHeads][headDim]`
    ///   - sinks: `[kvHeads * qMult]`
    ///   - slidingWindow: 0 pour une attention pleine.
    public static func sdpa(
        q: [[[[Double]]]], k: [[[Double]]], v: [[[Double]]],
        sinks: [Double], smScale: Double, slidingWindow: Int = 0
    ) -> [[Double]] {
        let tokens = q.count
        let kvHeads = q[0].count
        let qMult = q[0][0].count
        let headDim = q[0][0][0].count

        var out = [[Double]](
            repeating: [Double](repeating: 0, count: kvHeads * qMult * headDim), count: tokens)

        for h in 0..<kvHeads {
            for m in 0..<qMult {
                let sink = sinks[h * qMult + m]
                for query in 0..<tokens {
                    // Positions visibles : causales, et dans la fenêtre si elle est active.
                    var keys: [Int] = []
                    for key in 0...query {
                        if slidingWindow > 0 && key <= query - slidingWindow { continue }
                        keys.append(key)
                    }

                    var logits = [Double](repeating: 0, count: keys.count)
                    var peak = sink
                    for (index, key) in keys.enumerated() {
                        var dot = 0.0
                        for i in 0..<headDim { dot += q[query][h][m][i] * k[key][h][i] }
                        logits[index] = dot * smScale
                        peak = max(peak, logits[index])
                    }

                    // Le puits participe au dénominateur, jamais au numérateur.
                    var denominator = exp(sink - peak)
                    for value in logits { denominator += exp(value - peak) }

                    let base = (h * qMult + m) * headDim
                    for (index, key) in keys.enumerated() {
                        let weight = exp(logits[index] - peak) / denominator
                        for i in 0..<headDim {
                            out[query][base + i] += weight * v[key][h][i]
                        }
                    }
                }
            }
        }
        return out
    }

    // MARK: - Routeur

    /// Sélectionne les `topK` meilleurs experts puis applique le softmax **sur ces seuls
    /// logits**, pas sur la distribution complète. Les poids obtenus somment donc à 1
    /// entre les experts retenus.
    ///
    /// En cas d'égalité, l'indice le plus petit gagne — convention nécessaire pour que le
    /// décodage reste reproductible.
    public static func router(_ logits: [Double], topK: Int) -> (indices: [Int], weights: [Double]) {
        let order = logits.indices.sorted {
            logits[$0] == logits[$1] ? $0 < $1 : logits[$0] > logits[$1]
        }
        let selected = Array(order.prefix(topK))
        let values = selected.map { logits[$0] }
        let peak = values.max() ?? 0
        let exponentials = values.map { exp($0 - peak) }
        let total = exponentials.reduce(0, +)
        return (selected, exponentials.map { $0 / total })
    }
}
