import Foundation

/// RoPE cos/sin tables, with the YaRN context extension.
///
/// This computation lives in `HydraCore` because **two consumers depend on it**: the
/// runtime, which needs it on every token, and the reference implementation that validates
/// the kernels. Duplicating them would risk divergence — precisely the kind of gap that
/// raises no error and silently degrades the outputs.
///
/// The independence that matters for validation is elsewhere: these values are compared
/// against a Python transcription of OpenAI's reference code
/// (`tools/gen_reference_fixtures.py`).
public struct RoPETables: Sendable {

    public struct Parameters: Sendable, Equatable {
        public let headDim: Int
        public let base: Double
        public let initialContextLength: Int
        public let scalingFactor: Double
        /// `beta_slow` in the Hugging Face configuration.
        public let ntkAlpha: Double
        /// `beta_fast` in the Hugging Face configuration.
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

    /// Multiplies cos and sin. **The easiest part of YaRN to miss**: the algorithm does not
    /// merely rescale the frequencies, it also tightens the amplitude. For GPT-OSS this is
    /// 1.3466; omitting it breaks nothing visible but shifts all of attention.
    ///
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
        // Bounds of the piecewise NTK ramp: below we interpolate, above we extrapolate, and
        // in between we blend linearly.
        let low = halfDim * log(context / (p.ntkBeta * 2 * .pi)) / log(p.base)
        let high = halfDim * log(context / (p.ntkAlpha * 2 * .pi)) / log(p.base)
        precondition(low > 0 && low < high && high < halfDim - 1, "inconsistent YaRN bounds")

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

    /// One position's tables, concentration already applied.
    public func tables(at position: Int) -> (cos: [Double], sin: [Double]) {
        let t = Double(position)
        return (
            inverseFrequencies.map { Foundation.cos(t * $0) * concentration },
            inverseFrequencies.map { Foundation.sin(t * $0) * concentration }
        )
    }

    /// An allocation-free variant, for the decoding path.
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
    /// GPT-OSS's real values, both 20B and 120B.
    public static let gptOss = RoPETables.Parameters(
        headDim: 64, base: 150_000, initialContextLength: 4096,
        scalingFactor: 32, ntkAlpha: 1, ntkBeta: 32)
}
