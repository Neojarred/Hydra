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
    /// choose a prompt format or a stop-token set — never inside a decoding step.
    var architecture: ModelArchitecture { get }

    var kvCache: KVCache { get }
    var expertCache: ExpertSlotCache { get }
    /// Memory the runtime reserved: expert slots, KV, scratch, logits.
    var reservedBytes: Int { get }

    /// Tokens currently represented in the cache.
    var position: Int { get }
    /// Whether the cache can be rewound, which is what makes a turn's work reusable.
    var canRewind: Bool { get }
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
