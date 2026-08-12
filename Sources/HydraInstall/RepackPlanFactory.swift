import Foundation
import HydraCore
import HydraFormat

/// Builds the right plan for a model. **One of the two places dispatch happens** (D-023).
///
/// The rule that governs this file: choosing an architecture is a decision taken once, where
/// the model is picked, and never again. The install path downstream of here, downloading
/// spans, scattering bytes, writing a manifest, sees only `any InstallablePlan` and cannot
/// tell the two apart.
public enum RepackPlanFactory {

    public enum FactoryError: Error, CustomStringConvertible {
        case unsupported(ModelArchitecture, actual: String)

        public var description: String {
            switch self {
            case let .unsupported(architecture, actual):
                return "the descriptor claims \(architecture.rawValue) but is a \(actual)"
            }
        }
    }

    /// - Throws: when a descriptor's declared architecture does not match its concrete type.
    ///   That mismatch is only reachable by writing a new descriptor and forgetting this
    ///   switch, which is exactly the case worth an error rather than a silent wrong plan.
    public static func plan(
        for model: any ModelDescriptor,
        weightMap: [String: String],
        headers: [String: SafetensorsHeader]
    ) throws -> any InstallablePlan {
        switch model.architecture {
        case .gptOss:
            guard let config = model as? GptOssConfig else {
                throw FactoryError.unsupported(.gptOss, actual: "\(type(of: model))")
            }
            return try RepackPlan(config: config, weightMap: weightMap, headers: headers)
        case .gemma4:
            // Two encodings of one architecture, so the family is not the discriminator: the
            // BF16 build and the MLX 4-bit build are both `.gemma4` and must stay that way,
            // because the tokenizer, the prompt format and the topology are identical. What
            // differs is how the weights are written down, which is the descriptor's concrete
            // type.
            if let mlx = model as? Gemma4MLXConfig {
                return try GemmaMLXRepackPlan(
                    config: mlx, weightMap: weightMap, headers: headers)
            }
            guard let config = model as? Gemma4Config else {
                throw FactoryError.unsupported(.gemma4, actual: "\(type(of: model))")
            }
            return try GemmaRepackPlan(config: config, weightMap: weightMap, headers: headers)
        }
    }
}
