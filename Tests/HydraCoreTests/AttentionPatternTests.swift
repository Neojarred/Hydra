import Foundation
import Testing

@testable import HydraCore

/// The third kind of attention layer, and what it must not be mistaken for.
///
/// Qwen's linear layers keep a fixed recurrent state rather than a key/value history (D-027).
/// Nothing in the codebase switches exhaustively on `AttentionPattern`, so adding the case
/// produced no compiler errors at all: every site compares against `.sliding` or `.full` with
/// an `if`, and a third kind falls silently into whichever branch is the `else`. That is the
/// failure this suite exists to prevent, and these assertions stand in for the exhaustiveness
/// the type does not give us.
@Suite("Attention pattern")
struct AttentionPatternTests {

    @Test("Only the recurrent kind lacks a key/value history")
    func historyIsKeptByAttentionOnly() {
        #expect(AttentionPattern.full.keepsKeyValueHistory)
        #expect(AttentionPattern.sliding.keepsKeyValueHistory)
        #expect(!AttentionPattern.linear.keepsKeyValueHistory)
    }

    /// A linear layer is not a full layer with a different window.
    ///
    /// The comparison this replaces asked `== .sliding`, so `.linear` answered "no" and was
    /// treated as full attention: a context-sized key/value buffer allocated per layer and
    /// never read, and a per-token budget that grows with the conversation on the thirty layers
    /// whose whole point is that it does not.
    @Test("A recurrent layer contributes nothing per token")
    func recurrentLayersCostNothingPerToken() {
        struct Recurrent: ModelDescriptor {
            var architecture: ModelArchitecture { .gptOss }
            var name: String { "recurrent fixture" }
            var layerCount: Int { 4 }
            var hiddenSize: Int { 64 }
            var vocabSize: Int { 128 }
            var rmsNormEps: Float { 1e-6 }
            var maxPositionEmbeddings: Int { 1024 }
            var expertCount: Int { 8 }
            var expertsPerToken: Int { 2 }
            var slidingWindow: Int { 32 }
            var layerTypes: [AttentionPattern] { [.linear, .linear, .linear, .full] }
            func attentionGeometry(atLayer index: Int) -> AttentionGeometry {
                AttentionGeometry(headDim: 16, attentionHeadCount: 4, keyValueHeadCount: 2)
            }
            var expertBlob: any ExpertBlob { GptOssConfig.tiny.expertBlobLayout }
            var expertFormat: String { "fixture" }
            var residentTensors: [(name: String, byteCount: Int)] { [] }
            var embeddingFileBytes: Int { 0 }
            var residentEmbeddingTensor: String? { nil }
        }

        let model = Recurrent()
        for layer in 0..<3 {
            #expect(
                model.kvBytesPerToken(atLayer: layer) == 0,
                "layer \(layer) is recurrent and keeps no per-token history")
        }
        #expect(model.kvBytesPerToken(atLayer: 3) > 0, "the full layer still keeps one")

        // Only the full layer grows, so doubling the context does not double the cache.
        let short = model.kvCacheBytes(contextLength: 512)
        let long = model.kvCacheBytes(contextLength: 1024)
        #expect(long == short * 2 - 0, "with one full layer of four, growth is that layer's")
        #expect(model.recurrentStateBytes == 0, "the default is zero until a model overrides it")
    }
}
