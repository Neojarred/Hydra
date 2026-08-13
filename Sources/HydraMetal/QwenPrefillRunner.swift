import Foundation
import HydraCore
import HydraFormat
import Metal

/// Qwen's prompt processing, reordered so a layer's experts are read once a chunk.
///
/// Decoding reads 540 MiB of experts for **every token** at 4 bits. Processing a prompt one
/// token at a time pays that per token, so a thousand-token prompt moves half a terabyte from
/// SSD. Grouping the work by layer instead reads each expert a chunk selected at all, once: the
/// union saturates at the 256 experts of a layer, so a chunk of 256 tokens reads at most the
/// whole pool, 16.9 GiB, against 540 GiB token by token.
///
/// The same reordering Gemma has (M-047), with one thing this architecture adds. Three layers in
/// four are a recurrence, and a recurrence cannot be spread across tokens: token `t`'s state
/// update reads what `t-1` wrote. What it can avoid is a dispatch a token, which
/// `qwen_delta_rule_chunk` and `qwen_causal_conv_chunk` do by moving the loop inside the kernel.
/// Everything on either side of them, the projections, the norms, the gates, is per token and
/// batched.
public final class QwenPrefillRunner {

    /// Tokens a chunk.
    ///
    /// 256 for Gemma's measured reason (M-046): the expert union saturates, so a larger chunk
    /// reads no more, and a smaller one reads the same experts again. Not measured for this
    /// model yet, and that is worth saying rather than presenting an inherited constant as a
    /// result.
    public static let chunk = 256

    /// The chunk this instance was built for. Configurable so a test can cross a boundary
    /// without a prompt of hundreds of tokens: carrying the recurrent state and the convolution
    /// window from one chunk to the next is the part that has no equivalent in Gemma, and a
    /// suite that never crossed a boundary would not be testing it.
    public let chunkTokens: Int

    private let config: Qwen35MoeConfig
    private let encoder: ForwardEncoder
    private let weights: Qwen35MoeWeights
    private let layerWeights: [QwenLayerRunner.LayerWeights]

    // The residual stream and the staging for one chunk.
    let hidden: MTLBuffer            // [chunk][hidden]
    private let normed: MTLBuffer            // [chunk][hidden]
    private let transposed: MTLBuffer        // [cols][paddedTokens], the batched GEMM's input
    private let sums: MTLBuffer              // per-chunk activation sums
    private let projected: MTLBuffer         // [chunk][max(hidden, queryDim)]

    // The linear layers.
    private let qkvRaw: MTLBuffer            // [chunk][convDim]
    private let qkv: MTLBuffer               // [chunk][convDim], convolved
    private let z: MTLBuffer                 // [chunk][valueSpan]
    private let aValues: MTLBuffer           // [chunk][valueHeads]
    private let bValues: MTLBuffer           // [chunk][valueHeads]
    private let mixed: MTLBuffer             // [chunk][valueSpan]
    private let gated: MTLBuffer             // [chunk][valueSpan]

    // The attention layers.
    private let combined: MTLBuffer          // [chunk][queryProjectionRows]
    private let query: MTLBuffer             // [chunk][queryDim]
    private let gate: MTLBuffer              // [chunk][queryDim]
    private let key: MTLBuffer               // [chunk][keyValueDim]
    private let value: MTLBuffer             // [chunk][keyValueDim]
    private let attended: MTLBuffer          // [chunk][queryDim]
    private let cosTables: MTLBuffer         // [chunk][headDim / 2]
    private let sinTables: MTLBuffer
    private let sinks: MTLBuffer

    // The mixture.
    private let routerLogits: MTLBuffer      // [chunk][expertCount]
    private let routerIndices: MTLBuffer     // [chunk][topK], UInt32
    private let routerWeights: MTLBuffer     // [chunk][topK]
    private let expertSlices: MTLBuffer      // [chunk][topK][hidden]
    private let sharedGate: MTLBuffer
    private let sharedUp: MTLBuffer
    private let sharedActivated: MTLBuffer
    private let sharedOut: MTLBuffer         // [chunk][hidden]
    private let sharedGateLogit: MTLBuffer   // [chunk]
    private let combinedExperts: MTLBuffer   // [chunk][hidden]

    // One expert's group: the member tokens gathered, projected, and scattered back.
    private let groupInput: MTLBuffer        // [chunk][hidden]
    private let groupGate: MTLBuffer
    private let groupUp: MTLBuffer
    private let groupActivated: MTLBuffer
    private let groupOutput: MTLBuffer       // [chunk][hidden]

    public private(set) var byteCount = 0

    public init(
        config: Qwen35MoeConfig, encoder: ForwardEncoder, weights: Qwen35MoeWeights,
        layerWeights: [QwenLayerRunner.LayerWeights], device: MTLDevice,
        chunkTokens: Int = QwenPrefillRunner.chunk
    ) throws {
        precondition(chunkTokens > 0)
        self.chunkTokens = chunkTokens
        self.config = config
        self.encoder = encoder
        self.weights = weights
        self.layerWeights = layerWeights

        let chunk = chunkTokens
        let hiddenSize = config.hiddenSize
        let valueSpan = config.linearValueHeads * config.linearValueHeadDim
        var total = 0

        func make(_ floats: Int, _ name: String) throws -> MTLBuffer {
            guard let buffer = device.makeBuffer(
                length: max(floats, 1) * 4, options: .storageModePrivate)
            else { throw ModelRunner.RunnerError.allocationFailed(name) }
            total += buffer.length
            return buffer
        }
        func shared(_ floats: Int, _ name: String) throws -> MTLBuffer {
            guard let buffer = device.makeBuffer(
                length: max(floats, 1) * 4, options: .storageModeShared)
            else { throw ModelRunner.RunnerError.allocationFailed(name) }
            total += buffer.length
            return buffer
        }

        // The widest activation any staged projection reads, which is what the transpose and
        // the sums have to fit: the convolved q/k/v vector, wider than the hidden size.
        let widest = max(
            hiddenSize, max(config.linearConvDim, max(config.queryDim, valueSpan)))
        let padded = ForwardEncoder.paddedTokens(chunk)

        hidden = try shared(chunk * hiddenSize, "qwen.prefill.hidden")
        normed = try make(chunk * hiddenSize, "qwen.prefill.normed")
        transposed = try make(widest * padded, "qwen.prefill.transposed")
        sums = try make(
            max(ForwardEncoder.chunkCount(cols: widest, bits: 4), 1) * padded,
            "qwen.prefill.sums")
        projected = try make(chunk * max(hiddenSize, config.queryDim), "qwen.prefill.projected")

        qkvRaw = try make(chunk * config.linearConvDim, "qwen.prefill.qkvRaw")
        qkv = try make(chunk * config.linearConvDim, "qwen.prefill.qkv")
        z = try make(chunk * valueSpan, "qwen.prefill.z")
        aValues = try make(chunk * config.linearValueHeads, "qwen.prefill.a")
        bValues = try make(chunk * config.linearValueHeads, "qwen.prefill.b")
        mixed = try make(chunk * valueSpan, "qwen.prefill.mixed")
        gated = try make(chunk * valueSpan, "qwen.prefill.gated")

        combined = try make(chunk * config.queryProjectionRows, "qwen.prefill.combined")
        query = try make(chunk * config.queryDim, "qwen.prefill.query")
        gate = try make(chunk * config.queryDim, "qwen.prefill.gate")
        key = try make(chunk * config.keyValueDim, "qwen.prefill.key")
        value = try make(chunk * config.keyValueDim, "qwen.prefill.value")
        attended = try make(chunk * config.queryDim, "qwen.prefill.attended")
        cosTables = try shared(chunk * config.headDim / 2, "qwen.prefill.cos")
        sinTables = try shared(chunk * config.headDim / 2, "qwen.prefill.sin")

        let sinkBits = [UInt16](
            repeating: BF16.fromFloat(-1e30),
            count: max(config.attentionHeadCount, 1))
        guard let sinkBuffer = sinkBits.withUnsafeBytes({
            device.makeBuffer(
                bytes: $0.baseAddress!, length: max($0.count, 4), options: .storageModeShared)
        }) else { throw ModelRunner.RunnerError.allocationFailed("qwen.prefill.sinks") }
        sinks = sinkBuffer
        total += sinkBuffer.length

        routerLogits = try make(chunk * config.expertCount, "qwen.prefill.routerLogits")
        routerIndices = try shared(chunk * config.expertsPerToken, "qwen.prefill.routerIndices")
        routerWeights = try make(chunk * config.expertsPerToken, "qwen.prefill.routerWeights")
        expertSlices = try make(
            chunk * config.expertsPerToken * hiddenSize, "qwen.prefill.expertSlices")
        sharedGate = try make(
            chunk * config.sharedExpertIntermediateSize, "qwen.prefill.sharedGate")
        sharedUp = try make(chunk * config.sharedExpertIntermediateSize, "qwen.prefill.sharedUp")
        sharedActivated = try make(
            chunk * config.sharedExpertIntermediateSize, "qwen.prefill.sharedActivated")
        sharedOut = try make(chunk * hiddenSize, "qwen.prefill.sharedOut")
        sharedGateLogit = try make(chunk, "qwen.prefill.sharedGateLogit")
        combinedExperts = try make(chunk * hiddenSize, "qwen.prefill.combined")

        groupInput = try make(chunk * hiddenSize, "qwen.prefill.groupInput")
        groupGate = try make(chunk * config.moeIntermediateSize, "qwen.prefill.groupGate")
        groupUp = try make(chunk * config.moeIntermediateSize, "qwen.prefill.groupUp")
        groupActivated = try make(
            chunk * config.moeIntermediateSize, "qwen.prefill.groupActivated")
        groupOutput = try make(chunk * hiddenSize, "qwen.prefill.groupOutput")

        byteCount = total
    }

    private var float: Int { MemoryLayout<Float>.size }

    /// Stages an activation for the batched projection: transposed, plus its per-chunk sums.
    private func stage(
        _ input: MTLBuffer, cols: Int, tokens: Int, bits: Int,
        in commandBuffer: MTLCommandBuffer
    ) throws {
        try encoder.transposeActivations(
            input: input, inputOffset: 0, output: transposed, outputOffset: 0,
            tokens: tokens, cols: cols, in: commandBuffer)
        try encoder.chunkSums(
            input: transposed, inputOffset: 0, output: sums, outputOffset: 0,
            tokens: tokens, cols: cols, bits: bits, in: commandBuffer)
    }

    private func project(
        _ source: ForwardEncoder.ProjectionSource, into output: MTLBuffer,
        rows: Int, cols: Int, tokens: Int, in commandBuffer: MTLCommandBuffer
    ) throws {
        try encoder.encodeBatchedProjection(
            source, input: transposed, sums: sums, output: output,
            rows: rows, cols: cols, tokens: tokens, in: commandBuffer)
    }

    // MARK: - The two kinds of token mixer

    private func encodeLinearChunk(
        _ layer: Int, weights w: QwenLinearBlock.Weights, tokens: Int,
        state: RecurrentStateCache, in commandBuffer: MTLCommandBuffer
    ) throws {
        let hiddenSize = config.hiddenSize
        let valueSpan = config.linearValueHeads * config.linearValueHeadDim
        let bits = ForwardEncoder.bits(of: w.qkv) ?? config.quantBits

        try encoder.rmsNormBatched(
            input: hidden, inputOffset: 0,
            scale: w.inputNorm.buffer, scaleOffset: w.inputNorm.offset,
            output: normed, outputOffset: 0,
            tokens: tokens, size: hiddenSize, eps: config.rmsNormEps, in: commandBuffer)

        try stage(normed, cols: hiddenSize, tokens: tokens, bits: bits, in: commandBuffer)
        try project(
            w.qkv, into: qkvRaw, rows: config.linearConvDim, cols: hiddenSize,
            tokens: tokens, in: commandBuffer)
        try project(
            w.z, into: z, rows: valueSpan, cols: hiddenSize, tokens: tokens, in: commandBuffer)
        try project(
            w.a, into: aValues, rows: config.linearValueHeads, cols: hiddenSize,
            tokens: tokens, in: commandBuffer)
        try project(
            w.b, into: bValues, rows: config.linearValueHeads, cols: hiddenSize,
            tokens: tokens, in: commandBuffer)

        guard let entry = state.layers.first(where: { $0.index == layer }) else {
            throw ModelRunner.RunnerError.allocationFailed(
                "no recurrent state for layer \(layer)")
        }

        // The two sequential parts, one dispatch each rather than one a token.
        try encoder.qwenCausalConvChunk(
            window: entry.layer.window, windowOffset: 0, input: qkvRaw,
            weight: w.convWeight.buffer, weightOffset: w.convWeight.offset,
            bias: w.convBias?.buffer, biasOffset: w.convBias?.offset ?? 0,
            output: qkv, tokens: tokens, convDim: config.linearConvDim,
            kernel: config.linearConvKernel, in: commandBuffer)
        try encoder.qwenDeltaRuleChunk(
            state: entry.layer.state, stateOffset: 0, qkv: qkv, a: aValues, b: bValues,
            logA: w.logA.buffer, logAOffset: w.logA.offset,
            dtBias: w.dtBias.buffer, dtBiasOffset: w.dtBias.offset,
            output: mixed, tokens: tokens,
            valueHeads: config.linearValueHeads, keyHeads: config.linearKeyHeads,
            keyDim: config.linearKeyHeadDim, valueDim: config.linearValueHeadDim,
            eps: config.rmsNormEps, in: commandBuffer)

        // Per head, and the heads of every token are independent, so the token rides the grid.
        try encoder.qwenGatedRMSNormHeads(
            input: mixed, weight: w.normWeight.buffer, weightOffset: w.normWeight.offset,
            gate: z, output: gated,
            heads: tokens * config.linearValueHeads, dim: config.linearValueHeadDim,
            eps: config.rmsNormEps, in: commandBuffer)

        try stage(gated, cols: valueSpan, tokens: tokens, bits: bits, in: commandBuffer)
        try project(
            w.outProj, into: projected, rows: hiddenSize, cols: valueSpan,
            tokens: tokens, in: commandBuffer)
        try encoder.addInPlace(
            target: hidden, targetOffset: 0, addend: projected, addendOffset: 0,
            size: tokens * hiddenSize, in: commandBuffer)
    }

    private func encodeAttentionChunk(
        _ layer: Int, weights w: QwenAttentionBlock.Weights, tokens: Int, firstPosition: Int,
        kvCache: KVCache, in commandBuffer: MTLCommandBuffer
    ) throws {
        let hiddenSize = config.hiddenSize
        let heads = config.attentionHeadCount
        let kvHeads = config.keyValueHeadCount
        let headDim = config.headDim
        let bits = ForwardEncoder.bits(of: w.qProj) ?? config.quantBits
        let ring = kvCache.layers[layer].ringSize

        try encoder.rmsNormBatched(
            input: hidden, inputOffset: 0,
            scale: w.inputNorm.buffer, scaleOffset: w.inputNorm.offset,
            output: normed, outputOffset: 0,
            tokens: tokens, size: hiddenSize, eps: config.rmsNormEps, in: commandBuffer)

        try stage(normed, cols: hiddenSize, tokens: tokens, bits: bits, in: commandBuffer)
        try project(
            w.qProj, into: combined, rows: config.queryProjectionRows, cols: hiddenSize,
            tokens: tokens, in: commandBuffer)
        try project(
            w.kProj, into: key, rows: config.keyValueDim, cols: hiddenSize,
            tokens: tokens, in: commandBuffer)
        try project(
            w.vProj, into: value, rows: config.keyValueDim, cols: hiddenSize,
            tokens: tokens, in: commandBuffer)

        // Every one of these is the decode kernel with a larger head count: the query and its
        // gate are laid out per head, so `tokens · heads` addresses the chunk with no change to
        // the kernel and no separate batched variant to keep in step with it.
        try encoder.qwenSplitQueryGate(
            combined: combined, query: query, gate: gate,
            heads: tokens * heads, headDim: headDim, in: commandBuffer)
        try encoder.rmsNormHeads(
            vector: query, scale: w.qNorm.buffer, scaleOffset: w.qNorm.offset,
            heads: tokens * heads, headDim: headDim, eps: config.rmsNormEps, in: commandBuffer)
        try encoder.rmsNormHeads(
            vector: key, scale: w.kNorm.buffer, scaleOffset: w.kNorm.offset,
            heads: tokens * kvHeads, headDim: headDim, eps: config.rmsNormEps,
            in: commandBuffer)

        try encoder.applyRoPEBatched(
            vector: query, vectorOffset: 0, cos: cosTables, sin: sinTables,
            tableStride: headDim / 2, tokens: tokens, heads: heads, headDim: headDim,
            in: commandBuffer)
        try encoder.applyRoPEBatched(
            vector: key, vectorOffset: 0, cos: cosTables, sin: sinTables,
            tableStride: headDim / 2, tokens: tokens, heads: kvHeads, headDim: headDim,
            in: commandBuffer)

        for token in 0..<tokens {
            try encoder.writeKeyValue(
                key: key, keyOffset: token * config.keyValueDim * float,
                value: value, valueOffset: token * config.keyValueDim * float,
                keyCache: kvCache.layers[layer].keys,
                valueCache: kvCache.layers[layer].values,
                kvHeads: kvHeads, headDim: headDim, position: firstPosition + token,
                ringSize: ring, in: commandBuffer)
        }
        // Every Qwen attention layer is full attention, so the window is zero: the recurrent
        // layers are what bounds this model's memory, not a sliding window.
        try encoder.attentionPrefill(
            query: query, keyCache: kvCache.layers[layer].keys,
            valueCache: kvCache.layers[layer].values, sinks: sinks, sinksOffset: 0,
            output: attended, qHeads: heads, kvHeads: kvHeads, headDim: headDim,
            tokens: tokens, firstPosition: firstPosition, window: 0, ringSize: ring,
            smScale: 1.0 / Float(headDim).squareRoot(), in: commandBuffer)

        try encoder.qwenApplyOutputGate(
            output: attended, gate: gate, count: tokens * config.queryDim, in: commandBuffer)

        try stage(attended, cols: config.queryDim, tokens: tokens, bits: bits, in: commandBuffer)
        try project(
            w.oProj, into: projected, rows: hiddenSize, cols: config.queryDim,
            tokens: tokens, in: commandBuffer)
        try encoder.addInPlace(
            target: hidden, targetOffset: 0, addend: projected, addendOffset: 0,
            size: tokens * hiddenSize, in: commandBuffer)
    }

    // MARK: - The mixture

    private func encodeRouterAndShared(
        weights w: QwenMixtureBlock.Weights, tokens: Int, in commandBuffer: MTLCommandBuffer
    ) throws {
        let hiddenSize = config.hiddenSize
        let inner = config.sharedExpertIntermediateSize
        let bits = ForwardEncoder.bits(of: w.shared.gate) ?? config.quantBits

        try encoder.rmsNormBatched(
            input: hidden, inputOffset: 0,
            scale: w.postAttentionNorm.buffer, scaleOffset: w.postAttentionNorm.offset,
            output: normed, outputOffset: 0,
            tokens: tokens, size: hiddenSize, eps: config.rmsNormEps, in: commandBuffer)

        try stage(
            normed, cols: hiddenSize, tokens: tokens,
            bits: ForwardEncoder.bits(of: w.router) ?? config.gateBits, in: commandBuffer)
        try project(
            w.router, into: routerLogits, rows: config.expertCount, cols: hiddenSize,
            tokens: tokens, in: commandBuffer)
        try project(
            w.sharedGate, into: sharedGateLogit, rows: 1, cols: hiddenSize,
            tokens: tokens, in: commandBuffer)

        // Per token: the top-k is a sort, and there is nothing in it to spread over a grid.
        for token in 0..<tokens {
            try encoder.routerTopK(
                logits: routerLogits, logitsOffset: token * config.expertCount * float,
                indices: routerIndices, indicesOffset: token * config.expertsPerToken * 4,
                weights: routerWeights, weightsOffset: token * config.expertsPerToken * float,
                expertCount: config.expertCount, topK: config.expertsPerToken,
                in: commandBuffer)
        }

        // The shared branch, which waits on nothing.
        try stage(normed, cols: hiddenSize, tokens: tokens, bits: bits, in: commandBuffer)
        try project(
            w.shared.gate, into: sharedGate, rows: inner, cols: hiddenSize,
            tokens: tokens, in: commandBuffer)
        try project(
            w.shared.up, into: sharedUp, rows: inner, cols: hiddenSize,
            tokens: tokens, in: commandBuffer)
        try encoder.qwenSiluMultiply(
            gate: sharedGate, up: sharedUp, output: sharedActivated,
            size: tokens * inner, in: commandBuffer)
        try stage(sharedActivated, cols: inner, tokens: tokens, bits: bits, in: commandBuffer)
        try project(
            w.shared.down, into: sharedOut, rows: hiddenSize, cols: inner,
            tokens: tokens, in: commandBuffer)
        try encoder.qwenScaleBySigmoidBatched(
            target: sharedOut, logit: sharedGateLogit, size: hiddenSize, tokens: tokens,
            in: commandBuffer)

        // The expert input is the normalized hidden state, and the group path reads it after
        // this buffer has been reused by the shared branch's staging. Kept as its own copy.
        try encoder.copy(
            into: groupInput, destinationOffset: 0, from: normed, sourceOffset: 0,
            size: tokens * hiddenSize, in: commandBuffer)
    }

    /// One expert against the tokens that chose it, gathered so its weights are read once.
    private func encodeExpertGroup(
        _ expert: QwenMixtureBlock.Expert, members: [(token: Int, rank: Int)],
        in commandBuffer: MTLCommandBuffer
    ) throws {
        let hiddenSize = config.hiddenSize
        let inner = config.moeIntermediateSize
        let bits = ForwardEncoder.bits(of: expert.gate) ?? config.quantBits

        // Gathered into rows, so the batched projection sees a dense chunk.
        for (row, member) in members.enumerated() {
            try encoder.copy(
                into: normed, destinationOffset: row * hiddenSize * float,
                from: groupInput, sourceOffset: member.token * hiddenSize * float,
                size: hiddenSize, in: commandBuffer)
        }

        try stage(normed, cols: hiddenSize, tokens: members.count, bits: bits, in: commandBuffer)
        try project(
            expert.gate, into: groupGate, rows: inner, cols: hiddenSize,
            tokens: members.count, in: commandBuffer)
        try project(
            expert.up, into: groupUp, rows: inner, cols: hiddenSize,
            tokens: members.count, in: commandBuffer)
        try encoder.qwenSiluMultiply(
            gate: groupGate, up: groupUp, output: groupActivated,
            size: members.count * inner, in: commandBuffer)
        try stage(
            groupActivated, cols: inner, tokens: members.count, bits: bits, in: commandBuffer)
        try project(
            expert.down, into: groupOutput, rows: hiddenSize, cols: inner,
            tokens: members.count, in: commandBuffer)

        // Scattered back **by slot**, which is what fixes the order of the final sum. Reading
        // the experts in a different order must not change a bit of the result.
        for (row, member) in members.enumerated() {
            let slot = member.token * config.expertsPerToken + member.rank
            try encoder.writeExpertScaled(
                into: expertSlices, outputOffset: slot * hiddenSize * float,
                contribution: groupOutput, contributionOffset: row * hiddenSize * float,
                weights: routerWeights, weightIndex: slot, size: hiddenSize, in: commandBuffer)
        }
    }

    private func encodeCombine(tokens: Int, in commandBuffer: MTLCommandBuffer) throws {
        let hiddenSize = config.hiddenSize
        try encoder.sumExpertSlicesBatched(
            into: combinedExperts, slices: expertSlices, size: hiddenSize,
            count: config.expertsPerToken, tokens: tokens, in: commandBuffer)
        try encoder.addInPlace(
            target: combinedExperts, targetOffset: 0, addend: sharedOut, addendOffset: 0,
            size: tokens * hiddenSize, in: commandBuffer)
        try encoder.addInPlace(
            target: hidden, targetOffset: 0, addend: combinedExperts, addendOffset: 0,
            size: tokens * hiddenSize, in: commandBuffer)
    }

    // MARK: - The chunk

    /// Processes one chunk, leaving the last token's hidden state in `hidden`.
    func run(
        tokenCount: Int, firstPosition: Int,
        embeddings: (Int, UnsafeMutableBufferPointer<Float>) -> Void,
        kvCache: KVCache, state: RecurrentStateCache, expertCache: ExpertSlotCache,
        inverseFrequencies: [Double],
        commandBuffer: () throws -> MTLCommandBuffer,
        timings: inout ModelRunner.Timings
    ) throws {
        precondition(tokenCount > 0 && tokenCount <= chunkTokens)
        let hiddenSize = config.hiddenSize
        let pairs = config.headDim / 2

        let base = hidden.contents().bindMemory(
            to: Float.self, capacity: chunkTokens * hiddenSize)
        for token in 0..<tokenCount {
            embeddings(
                token,
                UnsafeMutableBufferPointer(
                    start: base.advanced(by: token * hiddenSize), count: hiddenSize))
        }

        // One rotary table a token. A single pair reused across the chunk gives every token the
        // table the CPU wrote last, which is finite and wrong by a little (M-047).
        let cos = cosTables.contents().bindMemory(
            to: Float.self, capacity: chunkTokens * pairs)
        let sin = sinTables.contents().bindMemory(
            to: Float.self, capacity: chunkTokens * pairs)
        for token in 0..<tokenCount {
            for i in 0..<pairs {
                let angle = Double(firstPosition + token) * inverseFrequencies[i]
                cos[token * pairs + i] = Float(Foundation.cos(angle))
                sin[token * pairs + i] = Float(Foundation.sin(angle))
            }
        }

        for layer in 0..<config.layerCount {
            // --- Phase A: the token mixer, the router and the shared branch ---
            var start = Date()
            let first = try commandBuffer()
            switch layerWeights[layer].mixer {
            case .linear(let w):
                try encodeLinearChunk(
                    layer, weights: w, tokens: tokenCount, state: state, in: first)
            case .attention(let w):
                try encodeAttentionChunk(
                    layer, weights: w, tokens: tokenCount, firstPosition: firstPosition,
                    kvCache: kvCache, in: first)
            }
            try encodeRouterAndShared(
                weights: layerWeights[layer].mixture, tokens: tokenCount, in: first)
            encoder.commit(first)
            try encoder.context.wait(first)
            timings.attentionAndRouter += Date().timeIntervalSince(start)

            // --- Phase B: the experts, grouped so each is read from SSD once ---
            let selection = readSelection(tokenCount: tokenCount)
            var groups: [Int: [(token: Int, rank: Int)]] = [:]
            for token in 0..<tokenCount {
                for rank in 0..<config.expertsPerToken {
                    groups[selection[token][rank], default: []].append((token, rank))
                }
            }
            // Every slot is written exactly once, so the slices are not zeroed first.
            //
            // The single-token path does zero them, and copying that here cost 16 MiB of
            // pointless writes a layer, 670 MiB a chunk, which no test could see because the
            // writes it guards against cannot happen: the groups partition the selections, so
            // each (token, rank) belongs to exactly one group and every group runs. The
            // partition is what the zeroing was standing in for, so it is checked directly.
            //
            // Both halves are checked, not the total. A count can be satisfied by one slot
            // claimed twice beside one claimed never, which is exactly the state the fill used
            // to paper over (M-052).
            var written = [Bool](repeating: false, count: tokenCount * config.expertsPerToken)
            for (expert, members) in groups {
                for member in members {
                    let slot = member.token * config.expertsPerToken + member.rank
                    precondition(
                        !written[slot],
                        "expert \(expert) claims slot \(slot), which another group also writes")
                    written[slot] = true
                }
            }
            precondition(
                written.allSatisfy { $0 },
                "an expert slot would be summed without anything writing it")

            start = Date()
            let ordered = groups.keys.sorted()
            var index = 0
            while index < ordered.count {
                let batch = Array(
                    ordered[index..<min(index + expertCache.slotsPerLayer, ordered.count)])
                index += batch.count

                var phase = Date()
                try expertCache.load(layer: layer, experts: batch)
                timings.expertIO += Date().timeIntervalSince(phase)

                phase = Date()
                let mixture = try commandBuffer()
                for expert in batch {
                    let (blob, offset) = try expertCache.expert(
                        layer: layer, expert: expert, pin: true)
                    try encodeExpertGroup(
                        weights.expert(blob: blob, offset: offset),
                        members: groups[expert]!, in: mixture)
                }
                encoder.commit(mixture)
                try encoder.context.wait(mixture)
                expertCache.release(layer: layer)
                timings.mixture += Date().timeIntervalSince(phase)
            }

            let combine = try commandBuffer()
            try encodeCombine(tokens: tokenCount, in: combine)
            encoder.commit(combine)
            try encoder.context.wait(combine)
        }
    }

    /// The last token's hidden state, which is what the head reads.
    func copyLastRow(
        tokenCount: Int, into destination: MTLBuffer, in commandBuffer: MTLCommandBuffer
    ) throws {
        try encoder.copy(
            into: destination, destinationOffset: 0,
            from: hidden, sourceOffset: (tokenCount - 1) * config.hiddenSize * float,
            size: config.hiddenSize, in: commandBuffer)
    }

    private func readSelection(tokenCount: Int) -> [[Int]] {
        let pointer = routerIndices.contents().bindMemory(
            to: UInt32.self, capacity: tokenCount * config.expertsPerToken)
        return (0..<tokenCount).map { token in
            (0..<config.expertsPerToken).map {
                Int(pointer[token * config.expertsPerToken + $0])
            }
        }
    }
}
