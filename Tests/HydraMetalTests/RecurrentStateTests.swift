import Foundation
import HydraCore
import Metal
import Testing

@testable import HydraMetal

/// The memory a linear attention layer keeps, and the two ways it differs from a KV cache.
///
/// It does not grow with the context, and it cannot be truncated. The second is the one with a
/// user-visible consequence, and these tests pin the behaviour rather than the implementation:
/// what a caller may resume from, and what it must be refused.
@Suite("Recurrent state cache")
struct RecurrentStateTests {

    /// Three linear layers to one full, the ratio Qwen uses.
    struct Mixed: ModelDescriptor {
        var architecture: ModelArchitecture { .gptOss }
        var name: String { "recurrent fixture" }
        var layerCount: Int { 8 }
        var hiddenSize: Int { 64 }
        var vocabSize: Int { 128 }
        var rmsNormEps: Float { 1e-6 }
        var maxPositionEmbeddings: Int { 4096 }
        var expertCount: Int { 8 }
        var expertsPerToken: Int { 2 }
        var slidingWindow: Int { 32 }
        var layerTypes: [AttentionPattern] {
            [.linear, .linear, .linear, .full, .linear, .linear, .linear, .full]
        }
        func attentionGeometry(atLayer index: Int) -> AttentionGeometry {
            AttentionGeometry(headDim: 16, attentionHeadCount: 4, keyValueHeadCount: 2)
        }
        var expertBlob: any ExpertBlob { GptOssConfig.tiny.expertBlobLayout }
        var expertFormat: String { "fixture" }
        var residentTensors: [(name: String, byteCount: Int)] { [] }
        var embeddingFileBytes: Int { 0 }
        var residentEmbeddingTensor: String? { nil }
    }

    private let geometry = RecurrentGeometry(
        valueHeads: 4, keyHeads: 2, keyDim: 8, valueDim: 8, convDim: 32, convKernel: 4)

    private func makeCache() throws -> RecurrentStateCache {
        try RecurrentStateCache(
            model: Mixed(), geometry: geometry, device: MetalContext().device)
    }

    /// Only the linear layers get a state, and its size does not depend on the context.
    ///
    /// This is the property that makes long conversations cheap on this architecture, so it is
    /// asserted rather than assumed: a cache that quietly scaled with position would look
    /// correct in every functional test and only show up as memory.
    @Test("The state covers the linear layers and does not grow")
    func fixedSize() throws {
        let cache = try makeCache()
        #expect(cache.layers.count == 6, "six linear layers of eight")
        #expect(cache.layers.map(\.index) == [0, 1, 2, 4, 5, 6])

        let before = cache.byteCount
        cache.advance(by: 4096)
        #expect(cache.byteCount == before, "the state is fixed, whatever the position")

        let perLayer = geometry.valueHeads * geometry.keyDim * geometry.valueDim * 4
            + (geometry.convKernel - 1) * geometry.convDim * 4
        #expect(before == 6 * perLayer)
    }

    /// A mixed model allocates both caches, and neither pays for the other's layers.
    ///
    /// `KVCache` used to refuse a recurrent layer outright, which was right while nothing could
    /// run such a model and is exactly wrong now: a Qwen cache has to hold histories for ten
    /// layers and nothing for thirty. It gives those thirty an index and no memory, so callers
    /// still write `layers[i]` without translating, and every read of one is guarded.
    @Test("A mixed model allocates history only for the attending layers")
    func mixedModelAllocatesBoth() throws {
        let context = try MetalContext()
        let model = Mixed()
        let kv = try KVCache(model: model, contextLength: 256, device: context.device)

        for (index, pattern) in model.layerTypes.enumerated() {
            #expect(
                kv.layers[index].keepsHistory == (pattern != .linear),
                "layer \(index) is \(pattern)")
            if pattern == .linear {
                #expect(kv.layers[index].capacity == 0, "and holds nothing")
            } else {
                #expect(kv.layers[index].capacity > 0)
            }
        }

        // The two caches together, and neither counts the other's layers.
        let state = try RecurrentStateCache(
            model: model, geometry: geometry, device: context.device)
        #expect(state.layers.count == model.layerTypes.count { $0 == .linear })
        #expect(kv.layers.count == model.layerCount, "indices stay the layer's own")
    }

    /// A fresh state is zero, and a reset returns it there.
    @Test("A reset clears the state and the convolution window")
    func resetClears() throws {
        let cache = try makeCache()
        let layer = cache.layers[0].layer
        let state = layer.state.contents().bindMemory(
            to: Float.self, capacity: layer.stateBytes / 4)
        state[0] = 1.5
        state[17] = -2.5
        cache.advance(by: 10)

        cache.reset()
        #expect(cache.length == 0)
        #expect(state[0] == 0 && state[17] == 0)
    }

    /// Appending needs no rewind, and that is the common case.
    ///
    /// A conversation continuing from where it left off asks to resume at the current position,
    /// which restores nothing. Qwen is no worse than Gemma here, and saying so is the point:
    /// the limitation below applies to editing, not to talking.
    @Test("Resuming at the current position is always possible")
    func appendNeedsNothing() throws {
        let cache = try makeCache()
        cache.advance(by: 120)
        #expect(cache.canRewind(to: 120))
        try cache.rewind(to: 120)
        #expect(cache.length == 120)
    }

    /// Without a checkpoint, going back is refused rather than approximated.
    ///
    /// There is no arithmetic that removes a token's contribution from a decayed running sum.
    /// A cache that silently accepted this and carried on would answer from a state describing
    /// tokens the caller believes it has discarded, which is finite, plausible and wrong.
    @Test("Rewinding below the current position is refused without a checkpoint")
    func rewindRefusedWithoutCheckpoint() throws {
        let cache = try makeCache()
        cache.advance(by: 120)
        #expect(!cache.canRewind(to: 60))
        #expect(throws: RecurrentStateCache.StateError.self) { try cache.rewind(to: 60) }
        #expect(cache.length == 120, "a refused rewind leaves the cache alone")
    }

    /// A checkpoint makes exactly its own position reachable, and restores the values.
    @Test("A checkpoint restores the state it saved")
    func checkpointRestores() throws {
        let cache = try makeCache()
        let layer = cache.layers[0].layer
        let state = layer.state.contents().bindMemory(
            to: Float.self, capacity: layer.stateBytes / 4)

        state[0] = 3.25
        cache.advance(by: 40)
        cache.checkpoint()

        // The conversation continues, and the state moves on.
        state[0] = 9.75
        cache.advance(by: 35)
        #expect(cache.length == 75)

        #expect(cache.canRewind(to: 40))
        #expect(!cache.canRewind(to: 50), "only saved positions, not any position below")
        try cache.rewind(to: 40)
        #expect(cache.length == 40)
        #expect(state[0] == 3.25, "the saved values came back, not just the position")
    }

    /// Restoring drops the checkpoints that describe a future which no longer happens.
    @Test("Rewinding forgets the checkpoints taken after it")
    func rewindDropsLaterCheckpoints() throws {
        let cache = try makeCache()
        cache.advance(by: 20); cache.checkpoint()
        cache.advance(by: 20); cache.checkpoint()   // 40
        cache.advance(by: 20); cache.checkpoint()   // 60

        try cache.rewind(to: 20)
        #expect(cache.canRewind(to: 20))
        #expect(!cache.canRewind(to: 40), "that turn was abandoned")
        #expect(!cache.canRewind(to: 60))
    }

    /// The nearest saved position at or below a target, which is what a caller needs to decide
    /// how much to reprocess.
    @Test("The cache reports the nearest position it can restore")
    func nearestRestorable() throws {
        let cache = try makeCache()
        cache.advance(by: 30); cache.checkpoint()
        cache.advance(by: 30); cache.checkpoint()   // 60

        #expect(cache.restorablePosition(atOrBelow: 75) == 60)
        #expect(cache.restorablePosition(atOrBelow: 45) == 30)
        #expect(cache.restorablePosition(atOrBelow: 10) == nil, "reprocess from the start")
    }

    /// Old checkpoints are dropped, because each is the whole state.
    @Test("Only the most recent checkpoints are kept")
    func checkpointsAreBounded() throws {
        let cache = try makeCache()
        for turn in 1...(RecurrentStateCache.maximumCheckpoints + 3) {
            cache.advance(by: 10)
            cache.checkpoint()
            _ = turn
        }
        #expect(cache.restorablePosition(atOrBelow: 10) == nil, "the earliest fell off")
        #expect(cache.canRewind(to: cache.length))
        let kept = (1...(RecurrentStateCache.maximumCheckpoints + 3))
            .map { $0 * 10 }
            .filter { cache.canRewind(to: $0) }
        #expect(kept.count == RecurrentStateCache.maximumCheckpoints)
    }
}
