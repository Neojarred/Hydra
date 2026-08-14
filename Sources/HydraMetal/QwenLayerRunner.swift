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

    /// A layer's token mixer, its router and its shared branch, into a caller's buffer.
    ///
    /// **The caller owns the command buffers**, which is the whole point of the split. There is
    /// exactly one unavoidable synchronization in a layer: the router picks the experts and the
    /// CPU cannot read them from SSD until that choice has come back. Everything else can share
    /// a buffer with its neighbour, and a runner that commits twice a layer pays eighty round
    /// trips a token for one that is needed forty times (M-042).
    public func encodeMixerAndRouter(
        _ layer: Int, hidden: MTLBuffer, weights: LayerWeights, scratch: Scratch,
        kvCache: KVCache, state: RecurrentStateCache, position: Int,
        cos: MTLBuffer, sin: MTLBuffer, sinks: MTLBuffer,
        in commandBuffer: MTLCommandBuffer
    ) throws {
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
                window: entry.layer.window, windowOffset: 0, in: commandBuffer)

        case .attention(let attentionWeights):
            let visible = kvCache.visibleRange(layer: layer, position: position)
            try attentionBlock.encode(
                hidden: hidden, weights: attentionWeights, scratch: scratch.attention,
                keyCache: kvCache.layers[layer].keys,
                valueCache: kvCache.layers[layer].values, sinks: sinks,
                position: position, visibleStart: visible.start, visibleCount: visible.count,
                ringSize: kvCache.layers[layer].ringSize,
                cos: cos, sin: sin, in: commandBuffer)
        }

        // The router and the shared branch go in the same buffer: neither needs the SSD, and
        // the shared branch is work the GPU can do while the CPU reads the chosen experts.
        try mixtureBlock.encodeRouterAndShared(
            hidden: hidden, weights: weights.mixture, scratch: scratch.mixture, in: commandBuffer)
    }

    /// The experts the router chose, read back from the GPU.
    public func selectedExperts(_ scratch: Scratch) -> [Int] {
        mixtureBlock.selectedExperts(scratch.mixture)
    }

    /// The routed experts and the sum that folds them back into the residual.
    public func encodeExperts(
        _ experts: [QwenMixtureBlock.Expert], hidden: MTLBuffer, scratch: Scratch,
        in commandBuffer: MTLCommandBuffer
    ) throws {
        try mixtureBlock.encodeExperts(experts, scratch: scratch.mixture, in: commandBuffer)
        try mixtureBlock.encodeCombine(
            hidden: hidden, scratch: scratch.mixture, in: commandBuffer)
    }

    /// Which block a layer gets, from the descriptor rather than from an index calculation
    /// written twice.
    public func mixerKind(atLayer layer: Int) -> AttentionPattern {
        config.attentionPattern(atLayer: layer)
    }
}
