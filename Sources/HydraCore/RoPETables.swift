import Foundation

/// Tables cos/sin de RoPE, avec l'extension de contexte YaRN.
///
/// Ce calcul vit dans `HydraCore` parce que **deux consommateurs en dépendent** : le
/// runtime, qui en a besoin à chaque token, et l'implémentation de référence qui valide
/// les noyaux. Les dupliquer exposerait à ce qu'ils divergent — précisément le genre
/// d'écart qui ne lève aucune erreur et dégrade silencieusement les sorties.
///
/// L'indépendance qui compte pour la validation est ailleurs : ces valeurs sont
/// comparées à une transcription Python du code de référence d'OpenAI
/// (`tools/gen_reference_fixtures.py`).
public struct RoPETables: Sendable {

    public struct Parameters: Sendable, Equatable {
        public let headDim: Int
        public let base: Double
        public let initialContextLength: Int
        public let scalingFactor: Double
        /// `beta_slow` dans la configuration Hugging Face.
        public let ntkAlpha: Double
        /// `beta_fast` dans la configuration Hugging Face.
        public let ntkBeta: Double

        public init(
            headDim: Int, base: Double, initialContextLength: Int,
            scalingFactor: Double, ntkAlpha: Double, ntkBeta: Double
        ) {
            self.headDim = headDim
            self.base = base
            self.initialContextLength = initialContextLength
            self.scalingFactor = scalingFactor
            self.ntkAlpha = ntkAlpha
            self.ntkBeta = ntkBeta
        }

        public init(config: GptOssConfig) {
            self.init(
                headDim: config.headDim, base: Double(config.ropeTheta),
                initialContextLength: config.yarnOriginalContext,
                scalingFactor: Double(config.yarnFactor),
                ntkAlpha: Double(config.yarnBetaSlow), ntkBeta: Double(config.yarnBetaFast))
        }
    }

    /// Multiplie cos et sin. **Le point le plus facile à manquer de YaRN** : l'algorithme
    /// ne se contente pas de réétaler les fréquences, il resserre aussi l'amplitude.
    /// Pour GPT-OSS cela vaut 1,3466 ; l'omettre ne casse rien de visible mais décale
    /// toute l'attention.
    public let concentration: Double
    public let inverseFrequencies: [Double]
    public let parameters: Parameters

    public init(_ parameters: Parameters) {
        self.parameters = parameters
        let p = parameters
        let frequencies = stride(from: 0, to: p.headDim, by: 2).map {
            pow(p.base, Double($0) / Double(p.headDim))
        }

        guard p.scalingFactor > 1.0 else {
            self.concentration = 1.0
            self.inverseFrequencies = frequencies.map { 1.0 / $0 }
            return
        }

        self.concentration = 0.1 * log(p.scalingFactor) + 1.0
        let halfDim = Double(p.headDim) / 2
        let context = Double(p.initialContextLength)
        // Bornes de la rampe NTK par parties : en deçà on interpole, au-delà on extrapole,
        // entre les deux on mélange linéairement.
        let low = halfDim * log(context / (p.ntkBeta * 2 * .pi)) / log(p.base)
        let high = halfDim * log(context / (p.ntkAlpha * 2 * .pi)) / log(p.base)
        precondition(low > 0 && low < high && high < halfDim - 1, "bornes YaRN incohérentes")

        var inverse: [Double] = []
        inverse.reserveCapacity(frequencies.count)
        for (index, frequency) in frequencies.enumerated() {
            let interpolation = 1.0 / (p.scalingFactor * frequency)
            let extrapolation = 1.0 / frequency
            let ramp = (Double(index) - low) / (high - low)
            let mask = 1.0 - min(max(ramp, 0.0), 1.0)
            inverse.append(interpolation * (1.0 - mask) + extrapolation * mask)
        }
        self.inverseFrequencies = inverse
    }

    /// Tables d'une position, concentration déjà appliquée.
    public func tables(at position: Int) -> (cos: [Double], sin: [Double]) {
        let t = Double(position)
        return (
            inverseFrequencies.map { Foundation.cos(t * $0) * concentration },
            inverseFrequencies.map { Foundation.sin(t * $0) * concentration }
        )
    }

    /// Variante sans allocation, pour le chemin de décodage.
    public func write(
        position: Int, cos cosOut: UnsafeMutableBufferPointer<Float>,
        sin sinOut: UnsafeMutableBufferPointer<Float>
    ) {
        let t = Double(position)
        for (index, frequency) in inverseFrequencies.enumerated() {
            cosOut[index] = Float(Foundation.cos(t * frequency) * concentration)
            sinOut[index] = Float(Foundation.sin(t * frequency) * concentration)
        }
    }
}

extension RoPETables.Parameters {
    /// Valeurs réelles de GPT-OSS, 20B comme 120B.
    public static let gptOss = RoPETables.Parameters(
        headDim: 64, base: 150_000, initialContextLength: 4096,
        scalingFactor: 32, ntkAlpha: 1, ntkBeta: 32)
}
