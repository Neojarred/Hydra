import Foundation
import HydraCore
import Metal

/// Qwen's mixture of experts, encoded onto the GPU.
///
/// A shared expert that always runs and is scaled by a sigmoid of its own single-row
/// projection, plus the routed ones, summed:
///
/// ```
/// routed = Σ  weight_e · down_e(silu(gate_e(h)) · up_e(h))
/// shared = sigmoid(shared_expert_gate(h)) · down(silu(gate(h)) · up(h))
/// out    = routed + shared
/// ```
///
/// Each routed contribution is written into **its own slot** and the slots are summed at the
/// end, so the order of the sum is fixed by the expert's rank in the router's output and not by
/// which expert's weights happened to arrive from the SSD first. Without that, the same prompt
/// gives different logits depending on the state of the slot cache, which is a bug this project
/// has already met once.
public struct QwenMixtureBlock {

    public final class Scratch {
        public let normed: MTLBuffer
        public let routerLogits: MTLBuffer
        public let routerIndices: MTLBuffer
        public let routerWeights: MTLBuffer
        /// One slot per selected expert, summed at the end in slot order.
        public let slices: MTLBuffer
        public let expertGate: MTLBuffer
        public let expertUp: MTLBuffer
        public let expertActivated: MTLBuffer
        public let expertOut: MTLBuffer
        public let sharedGate: MTLBuffer
        public let sharedUp: MTLBuffer
        public let sharedActivated: MTLBuffer
        public let sharedOut: MTLBuffer
        public let sharedGateLogit: MTLBuffer
        public let combined: MTLBuffer

        public init(config: Qwen35MoeConfig, device: MTLDevice) throws {
            func make(_ floats: Int, _ name: String) throws -> MTLBuffer {
                guard let buffer = device.makeBuffer(
                    length: max(floats, 1) * 4, options: .storageModeShared)
                else { throw ModelRunner.RunnerError.allocationFailed(name) }
                return buffer
            }
            normed = try make(config.hiddenSize, "qwen.moe.normed")
            routerLogits = try make(config.expertCount, "qwen.moe.logits")
            guard let indices = device.makeBuffer(
                length: config.expertsPerToken * 4, options: .storageModeShared)
            else { throw ModelRunner.RunnerError.allocationFailed("qwen.moe.indices") }
            routerIndices = indices
            routerWeights = try make(config.expertsPerToken, "qwen.moe.weights")
            slices = try make(config.expertsPerToken * config.hiddenSize, "qwen.moe.slices")
            // One slot an expert, not one shared by all of them, so the activation and the
            // scaling can be a single dispatch over the whole set rather than one an expert
            // (M-065). The cost is `expertsPerToken` times the intermediate width, which is
            // 8 x 512 floats here: 16 KiB.
            let experts = config.expertsPerToken
            expertGate = try make(experts * config.moeIntermediateSize, "qwen.moe.expertGate")
            expertUp = try make(experts * config.moeIntermediateSize, "qwen.moe.expertUp")
            expertActivated = try make(
                experts * config.moeIntermediateSize, "qwen.moe.expertAct")
            expertOut = try make(config.hiddenSize, "qwen.moe.expertOut")
            sharedGate = try make(config.sharedExpertIntermediateSize, "qwen.moe.sharedGate")
            sharedUp = try make(config.sharedExpertIntermediateSize, "qwen.moe.sharedUp")
            sharedActivated = try make(
                config.sharedExpertIntermediateSize, "qwen.moe.sharedAct")
            sharedOut = try make(config.hiddenSize, "qwen.moe.sharedOut")
            sharedGateLogit = try make(1, "qwen.moe.sharedGateLogit")
            combined = try make(config.hiddenSize, "qwen.moe.combined")
        }
    }

    /// One expert's three matrices, however they are encoded.
    public struct Expert {
        public let gate: ForwardEncoder.ProjectionSource
        public let up: ForwardEncoder.ProjectionSource
        public let down: ForwardEncoder.ProjectionSource
        public init(
            gate: ForwardEncoder.ProjectionSource, up: ForwardEncoder.ProjectionSource,
            down: ForwardEncoder.ProjectionSource
        ) {
            self.gate = gate
            self.up = up
            self.down = down
        }
    }

    public struct Weights {
        public let postAttentionNorm: (buffer: MTLBuffer, offset: Int)
        public let router: ForwardEncoder.ProjectionSource
        public let sharedGate: ForwardEncoder.ProjectionSource
        public let shared: Expert
        public init(
            postAttentionNorm: (buffer: MTLBuffer, offset: Int),
            router: ForwardEncoder.ProjectionSource,
            sharedGate: ForwardEncoder.ProjectionSource, shared: Expert
        ) {
            self.postAttentionNorm = postAttentionNorm
            self.router = router
            self.sharedGate = sharedGate
            self.shared = shared
        }
    }

    private let config: Qwen35MoeConfig
    private let encoder: ForwardEncoder

    public init(config: Qwen35MoeConfig, encoder: ForwardEncoder) {
        self.config = config
        self.encoder = encoder
    }

    /// Encodes the router and the shared branch, up to the point where the CPU must read which
    /// experts were selected.
    ///
    /// Split here for the same reason `Gemma4ModelRunner` splits: the router picks the experts
    /// and nothing can `pread` them until that choice has come back. Everything that does not
    /// depend on the choice runs on this side of the wait.
    public func encodeRouterAndShared(
        hidden: MTLBuffer, weights: Weights, scratch: Scratch,
        in commandBuffer: MTLCommandBuffer
    ) throws {
        let hiddenSize = config.hiddenSize

        try encoder.rmsNorm(
            input: hidden, inputOffset: 0,
            scale: weights.postAttentionNorm.buffer,
            scaleOffset: weights.postAttentionNorm.offset,
            output: scratch.normed, outputOffset: 0,
            size: hiddenSize, eps: config.rmsNormEps, in: commandBuffer)

        try encoder.encodeProjection(
            weights.router, input: scratch.normed, inputOffset: 0,
            output: scratch.routerLogits, outputOffset: 0,
            rows: config.expertCount, cols: hiddenSize, in: commandBuffer)
        // GPT-OSS's kernel, not Gemma's, for two reasons.
        //
        // The weights are the same either way: a softmax over the top-k alone equals a softmax
        // over everything followed by a renormalization, because the full denominator cancels
        // (D-027). So Qwen's `norm_topk_prob: true` is satisfied by this kernel.
        //
        // What is *not* the same is that Gemma's applies a per-expert scale from a tensor Qwen
        // does not have. That is the reason to pick this one, and it is a different reason from
        // the normalization.
        try encoder.routerTopK(
            logits: scratch.routerLogits, logitsOffset: 0,
            indices: scratch.routerIndices, weights: scratch.routerWeights,
            expertCount: config.expertCount, topK: config.expertsPerToken, in: commandBuffer)

        // The shared branch, which needs nothing from the SSD.
        try encoder.encodeProjection(
            weights.shared.gate, input: scratch.normed, inputOffset: 0,
            output: scratch.sharedGate, outputOffset: 0,
            rows: config.sharedExpertIntermediateSize, cols: hiddenSize, in: commandBuffer)
        try encoder.encodeProjection(
            weights.shared.up, input: scratch.normed, inputOffset: 0,
            output: scratch.sharedUp, outputOffset: 0,
            rows: config.sharedExpertIntermediateSize, cols: hiddenSize, in: commandBuffer)
        try encoder.qwenSiluMultiply(
            gate: scratch.sharedGate, up: scratch.sharedUp, output: scratch.sharedActivated,
            size: config.sharedExpertIntermediateSize, in: commandBuffer)
        try encoder.encodeProjection(
            weights.shared.down, input: scratch.sharedActivated, inputOffset: 0,
            output: scratch.sharedOut, outputOffset: 0,
            rows: hiddenSize, cols: config.sharedExpertIntermediateSize, in: commandBuffer)

        // Its gate is one row, so one logit, and the sigmoid scales the branch's **output**.
        try encoder.encodeProjection(
            weights.sharedGate, input: scratch.normed, inputOffset: 0,
            output: scratch.sharedGateLogit, outputOffset: 0,
            rows: 1, cols: hiddenSize, in: commandBuffer)
        try encoder.qwenScaleBySigmoid(
            target: scratch.sharedOut, logit: scratch.sharedGateLogit,
            size: hiddenSize, in: commandBuffer)

        // The slots start at zero: a rank whose expert somehow never runs contributes nothing
        // rather than whatever the last token left there.
        try encoder.fillZero(
            scratch.slices, offset: 0,
            size: config.expertsPerToken * hiddenSize, in: commandBuffer)
    }

    /// The experts the router selected, read back from the GPU.
    public func selectedExperts(_ scratch: Scratch) -> [Int] {
        let pointer = scratch.routerIndices.contents().bindMemory(
            to: UInt32.self, capacity: config.expertsPerToken)
        return (0..<config.expertsPerToken).map { Int(pointer[$0]) }
    }

    /// Every routed expert, phase by phase rather than expert by expert.
    ///
    /// The old shape ran one expert to completion before starting the next: gate, up, silu,
    /// down, write, times eight, times forty layers. The two elementwise steps in that list are
    /// **one threadgroup each**, and there were 350 silu launches and 311 writes a token at a
    /// width that cannot fill a 10-core GPU (M-065). Apple's own guidance is to batch small work
    /// into one kernel, because a launch stalls for as much as 10 microseconds whatever it does.
    ///
    /// The experts are independent, so the order is free. Running all eight gates, then all
    /// eight ups, then **one** activation over the whole set, then all eight downs, then **one**
    /// scaling, turns 40 dispatches a layer into 26 and moves the two narrow ones to a width of
    /// eight times what they had.
    ///
    /// The projections stay per expert: each reads a different weight matrix from a different
    /// slot buffer, and at 187 threadgroups they are already wide enough to fill the machine.
    public func encodeExperts(
        _ experts: [Expert], scratch: Scratch, in commandBuffer: MTLCommandBuffer
    ) throws {
        guard !experts.isEmpty else { return }
        let hiddenSize = config.hiddenSize
        let inner = config.moeIntermediateSize
        let float = MemoryLayout<Float>.size

        for (rank, expert) in experts.enumerated() {
            try encoder.encodeProjection(
                expert.gate, input: scratch.normed, inputOffset: 0,
                output: scratch.expertGate, outputOffset: rank * inner * float,
                rows: inner, cols: hiddenSize, in: commandBuffer)
        }
        for (rank, expert) in experts.enumerated() {
            try encoder.encodeProjection(
                expert.up, input: scratch.normed, inputOffset: 0,
                output: scratch.expertUp, outputOffset: rank * inner * float,
                rows: inner, cols: hiddenSize, in: commandBuffer)
        }

        // One dispatch for all of them.
        try encoder.qwenSiluMultiply(
            gate: scratch.expertGate, up: scratch.expertUp, output: scratch.expertActivated,
            size: experts.count * inner, in: commandBuffer)

        // Straight into the slot the rank names, unscaled; the scaling is the next dispatch.
        // Writing by slot rather than by arrival is what fixes the order of the final sum.
        for (rank, expert) in experts.enumerated() {
            try encoder.encodeProjection(
                expert.down, input: scratch.expertActivated,
                inputOffset: rank * inner * float,
                output: scratch.slices, outputOffset: rank * hiddenSize * float,
                rows: hiddenSize, cols: inner, in: commandBuffer)
        }
        try encoder.qwenScaleSlices(
            slices: scratch.slices, weights: scratch.routerWeights,
            size: hiddenSize, count: experts.count, in: commandBuffer)
    }

    /// Sums the slots, adds the shared branch, and adds the result to the residual.
    public func encodeCombine(
        hidden: MTLBuffer, scratch: Scratch, in commandBuffer: MTLCommandBuffer
    ) throws {
        let hiddenSize = config.hiddenSize
        // In slot order, which is rank order, which is fixed.
        try encoder.sumExpertSlices(
            into: scratch.combined, slices: scratch.slices,
            size: hiddenSize, count: config.expertsPerToken, in: commandBuffer)
        try encoder.addInPlace(
            target: scratch.combined, targetOffset: 0,
            addend: scratch.sharedOut, addendOffset: 0, size: hiddenSize, in: commandBuffer)
        try encoder.addInPlace(
            target: hidden, targetOffset: 0,
            addend: scratch.combined, addendOffset: 0, size: hiddenSize, in: commandBuffer)
    }
}
