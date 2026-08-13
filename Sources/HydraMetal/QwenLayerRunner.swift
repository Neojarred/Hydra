import Foundation
import HydraCore
import Metal

/// Qwen's forty layers, in order.
///
/// Three recurrent to one attending, each followed by the mixture, all pre-norm with the
/// residual carried through (D-027):
///
/// ```
/// for each layer:
///     hidden += token_mixer(input_layernorm(hidden))     linear or attention
///     hidden += mixture(post_attention_layernorm(hidden))
/// ```
///
/// Both blocks add their own residual, so this adds none: the sequencing is the whole job, and
/// the one thing it can get wrong that no block can catch is which block a layer gets.
///
/// Experts arrive through a closure rather than a cache reference. In production that closure
/// reads the slot cache; in a test it reads an array. The seam exists because the router has to
/// run before anything knows which experts to fetch, and because a layer runner that owned an
/// installer could not be tested before one existed.
public struct QwenLayerRunner {

    /// What one layer needs, whichever kind it is.
    public enum TokenMixer {
        case linear(QwenLinearBlock.Weights)
        case attention(QwenAttentionBlock.Weights)
    }

    public struct LayerWeights {
        public let mixer: TokenMixer
        public let mixture: QwenMixtureBlock.Weights
        public init(mixer: TokenMixer, mixture: QwenMixtureBlock.Weights) {
            self.mixer = mixer
            self.mixture = mixture
        }
    }

    public final class Scratch {
        public let linear: QwenLinearBlock.Scratch
        public let attention: QwenAttentionBlock.Scratch
        public let mixture: QwenMixtureBlock.Scratch

        public init(config: Qwen35MoeConfig, device: MTLDevice) throws {
            linear = try QwenLinearBlock.Scratch(config: config, device: device)
            attention = try QwenAttentionBlock.Scratch(config: config, device: device)
            mixture = try QwenMixtureBlock.Scratch(config: config, device: device)
        }
    }

    private let config: Qwen35MoeConfig
    private let linearBlock: QwenLinearBlock
    private let attentionBlock: QwenAttentionBlock
    private let mixtureBlock: QwenMixtureBlock
    private let context: MetalContext

    public init(config: Qwen35MoeConfig, encoder: ForwardEncoder, context: MetalContext) {
        self.config = config
        self.context = context
        linearBlock = QwenLinearBlock(config: config, encoder: encoder)
        attentionBlock = QwenAttentionBlock(config: config, encoder: encoder)
        mixtureBlock = QwenMixtureBlock(config: config, encoder: encoder)
    }

    /// One token through one layer.
    ///
    /// Two command buffers, because the router's choice has to reach the CPU before the experts
    /// it selected can be read. That is the only unavoidable synchronization in a layer, and it
    /// is the same one Gemma has.
    ///
    /// - Parameters:
    ///   - fetchExperts: called with the layer and the selected expert indices, in rank order,
    ///     after the router has run and before the mixture is completed.
    public func encodeLayer(
        _ layer: Int, hidden: MTLBuffer, weights: LayerWeights, scratch: Scratch,
        kvCache: KVCache, state: RecurrentStateCache, position: Int,
        cos: MTLBuffer, sin: MTLBuffer, sinks: MTLBuffer,
        commandBuffer: () throws -> MTLCommandBuffer,
        fetchExperts: (Int, [Int]) throws -> [QwenMixtureBlock.Expert]
    ) throws {
        let first = try commandBuffer()

        switch weights.mixer {
        case .linear(let linearWeights):
            // The recurrent layers are stored in their own order; find this layer's slot.
            guard let entry = state.layers.first(where: { $0.index == layer }) else {
                throw ModelRunner.RunnerError.allocationFailed(
                    "no recurrent state for layer \(layer)")
            }
            try linearBlock.encode(
                hidden: hidden, weights: linearWeights, scratch: scratch.linear,
                state: entry.layer.state, stateOffset: 0,
                window: entry.layer.window, windowOffset: 0, in: first)

        case .attention(let attentionWeights):
            let visible = kvCache.visibleRange(layer: layer, position: position)
            try attentionBlock.encode(
                hidden: hidden, weights: attentionWeights, scratch: scratch.attention,
                keyCache: kvCache.layers[layer].keys,
                valueCache: kvCache.layers[layer].values, sinks: sinks,
                position: position, visibleStart: visible.start, visibleCount: visible.count,
                ringSize: kvCache.layers[layer].ringSize,
                cos: cos, sin: sin, in: first)
        }

        // The router and the shared branch go in the same buffer: neither needs the SSD, and
        // the shared branch is work the GPU can do while the wait below happens.
        try mixtureBlock.encodeRouterAndShared(
            hidden: hidden, weights: weights.mixture, scratch: scratch.mixture, in: first)
        context.commit(first)
        try context.wait(first)

        let selected = mixtureBlock.selectedExperts(scratch.mixture)
        let experts = try fetchExperts(layer, selected)

        let second = try commandBuffer()
        for (rank, expert) in experts.enumerated() {
            try mixtureBlock.encodeExpert(
                expert, rank: rank, scratch: scratch.mixture, in: second)
        }
        try mixtureBlock.encodeCombine(
            hidden: hidden, scratch: scratch.mixture, in: second)
        context.commit(second)
        try context.wait(second)
    }

    /// Which block a layer gets, from the descriptor rather than from an index calculation
    /// written twice.
    public func mixerKind(atLayer layer: Int) -> AttentionPattern {
        config.attentionPattern(atLayer: layer)
    }
}
