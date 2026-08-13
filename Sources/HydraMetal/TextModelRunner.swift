import Foundation
import HydraCore
import Metal

/// What a caller needs from a model, whichever one it is.
///
/// The dispatch seam D-023 asks for: chosen once, where the model is picked, so that
/// `InferenceEngine`, the CLI and the app never ask which architecture they are holding. A
/// conditional in a decoding loop is how the 20B silently acquires Gemma's attention scaling.
///
/// Extracted from `ModelRunner`'s existing surface rather than designed: every member here is
/// one that already existed and that a caller already used. Nothing was added speculatively,
/// and `ModelRunner` conforms without a line of change to its body.
public protocol TextModelRunner: AnyObject, Sendable {

    /// What the runtime is executing. The one place downstream code may branch, and only to
    /// choose a prompt format or a stop-token set, never inside a decoding step.
    var architecture: ModelArchitecture { get }

    var kvCache: KVCache { get }
    var expertCache: ExpertSlotCache { get }
    /// Memory the runtime reserved: expert slots, KV, scratch, logits.
    var reservedBytes: Int { get }

    /// Tokens currently represented in the cache.
    var position: Int { get }
    /// Whether the cache can be rewound, which is what makes a turn's work reusable.
    var canRewind: Bool { get }

    /// True if the runner can resume from `tokens` already-processed tokens.
    ///
    /// Distinct from `canRewind`, which asks whether *any* rewind is possible and answers
    /// `false` for every model with a sliding window, Gemma 4 among them. Resuming a
    /// conversation needs to go back a handful of tokens, not to the beginning, and a bounded
    /// ring can do that.
    func canRewind(to tokens: Int) -> Bool
    func rewind(to tokens: Int)
    func reset()
    func resetSampling()

    /// Processes a prompt and returns the distribution for the token that follows it.
    @discardableResult
    func prefill(tokens: [Int]) throws -> UnsafeBufferPointer<Float>

    /// One decoding step.
    @discardableResult
    func forward(token: Int, needsLogits: Bool) throws -> UnsafeBufferPointer<Float>

    /// Draws a token from a distribution.
    func sample(
        from distribution: UnsafeBufferPointer<Float>, using sampling: ModelRunner.Sampling
    ) -> Int

    func greedyToken(from distribution: UnsafeBufferPointer<Float>) -> Int

    /// Verifies a batch of candidate tokens in one pass, returning per-position logits.
    /// The basis of speculative decoding.
    func verify(tokens: [Int]) throws -> [UnsafeBufferPointer<Float>]

    /// Advances by a batch, accepting the longest correct prefix of `draft`.
    func step(
        from distribution: UnsafeBufferPointer<Float>, draft: [Int],
        sampling: ModelRunner.Sampling
    ) throws -> (tokens: [Int], next: UnsafeBufferPointer<Float>)

    var lastTimings: ModelRunner.Timings { get }
}

extension ModelRunner: TextModelRunner {
    public var architecture: ModelArchitecture { config.architecture }
}

extension Gemma4ModelRunner: TextModelRunner {
    public var architecture: ModelArchitecture { config.architecture }
}

/// Builds the right runner for a model. **The other of the two places dispatch happens**
/// (D-023), the first being the install plan.
///
/// Everything past this call holds `any TextModelRunner`. That is the whole point: a
/// conditional inside a decoding loop is how the 20B silently acquires Gemma's attention
/// scaling, and the only defence that actually works is for the loop to have nothing to
/// condition on.
public enum ModelRuntime {

    public enum RuntimeError: Error, CustomStringConvertible {
        case unsupported(ModelArchitecture, actual: String)

        public var description: String {
            switch self {
            case let .unsupported(architecture, actual):
                return "the descriptor claims \(architecture.rawValue) but is a \(actual)"
            }
        }
    }

    public static func makeRunner(
        model: any ModelDescriptor, context: MetalContext, mapping: ModelMapping,
        expertCache: ExpertSlotCache, contextLength: Int
    ) throws -> any TextModelRunner {
        switch model.architecture {
        case .gptOss:
            guard let config = model as? GptOssConfig else {
                throw RuntimeError.unsupported(.gptOss, actual: "\(type(of: model))")
            }
            return try ModelRunner(
                config: config, context: context, mapping: mapping,
                expertCache: expertCache, contextLength: contextLength)
        case .gemma4:
            // Two encodings of one architecture. The geometry is shared, which is why the MLX
            // descriptor wraps the BF16 one, and only the weight source differs.
            if let mlx = model as? Gemma4MLXConfig {
                return try Gemma4ModelRunner(
                    config: mlx.base, context: context, mapping: mapping,
                    expertCache: expertCache, contextLength: contextLength,
                    weights: Gemma4MLXWeights(config: mlx, mapping: mapping))
            }
            guard let config = model as? Gemma4Config else {
                throw RuntimeError.unsupported(.gemma4, actual: "\(type(of: model))")
            }
            return try Gemma4ModelRunner(
                config: config, context: context, mapping: mapping,
                expertCache: expertCache, contextLength: contextLength)
        case .qwen35Moe:
            // The descriptor, the recurrent state cache and the CPU reference exist; the Metal
            // kernels for Gated DeltaNet do not. Refusing here is what keeps a half-built
            // architecture from being loaded and answering, which is the only failure mode
            // worse than not loading at all.
            throw RuntimeError.unsupported(.qwen35Moe, actual: "\(type(of: model))")
        }
    }
}
