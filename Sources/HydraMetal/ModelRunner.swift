import Foundation
import HydraCore
import HydraFormat
import Metal

/// The complete forward pass: embedding → layers → final norm → LM head → logits.
///
/// The decoding step follows exactly the structure the router imposes, layer after layer:
/// `cb1` up to the expert identifiers, the SSD read, `cb2` for the mixture. The LM head runs
/// only once, at the end.
///
/// No allocation happens during decoding. All scratch is reserved at load time, which makes
/// the footprint constant from one token to the next — the property the project has to
/// demonstrate.
public final class ModelRunner: @unchecked Sendable {

    public let config: GptOssConfig
    public let context: MetalContext
    public let mapping: ModelMapping
    public let expertCache: ExpertSlotCache
    public let kvCache: KVCache

    private let encoder: ForwardEncoder
    private let layerRunner: LayerRunner
    private let scratch: DecodeScratch
    private let prefillRunner: PrefillRunner
    private let prefillScratch: PrefillScratch
    private let ropeTables: RoPETables
    private let logits: MTLBuffer
    /// Logits for several positions at once, for speculative verification.
    /// Allocated on first use: ordinary decoding does not need it.
    private var batchLogits: MTLBuffer?

    /// Position of the next token to process.
    public private(set) var position = 0

    public struct Timings: Sendable {
        public var attentionAndRouter = 0.0
        public var expertIO = 0.0
        public var mixture = 0.0
        public var head = 0.0
        public var sampling = 0.0
        public var total: Double { attentionAndRouter + expertIO + mixture + head + sampling }
    }

    public private(set) var lastTimings = Timings()

    public enum RunnerError: Error, CustomStringConvertible {
        case allocationFailed(String)
        case commandBufferUnavailable

        public var description: String {
            switch self {
            case .allocationFailed(let name): return "allocation impossible : \(name)"
            case .commandBufferUnavailable: return "tampon de commandes indisponible"
            }
        }
    }

    public init(
        config: GptOssConfig, context: MetalContext, mapping: ModelMapping,
        expertCache: ExpertSlotCache, contextLength: Int,
        prefillChunk: Int = KVCache.prefillChunk
    ) throws {
        self.config = config
        self.context = context
        self.mapping = mapping
        self.expertCache = expertCache
        self.encoder = ForwardEncoder(context: context)
        self.scratch = try DecodeScratch(config: config, device: context.device)
        self.kvCache = try KVCache(
            model: config, contextLength: contextLength, device: context.device)
        self.layerRunner = LayerRunner(
            config: config, encoder: encoder, mapping: mapping, cache: expertCache)
        self.ropeTables = RoPETables(RoPETables.Parameters(config: config))

        // The chunk cannot exceed the margin of the sliding layers' ring: all of a chunk's
        // KV writes precede its attentions.
        let chunk = min(prefillChunk, KVCache.prefillChunk)
        self.prefillScratch = try PrefillScratch(
            config: config, maximumTokens: chunk, device: context.device)
        self.prefillRunner = PrefillRunner(
            config: config, encoder: BatchEncoder(context: context),
            mapping: mapping, cache: expertCache)

        guard let logits = context.device.makeBuffer(
            length: config.vocabSize * MemoryLayout<Float>.size, options: .storageModeShared)
        else { throw RunnerError.allocationFailed("logits") }
        self.logits = logits
    }

    /// Memory reserved by the runtime, excluding mappings.
    public var reservedBytes: Int {
        scratch.byteCount + prefillScratch.byteCount + kvCache.byteCount
            + expertCache.reservedBytes + config.vocabSize * MemoryLayout<Float>.size
    }

    // MARK: - Prefill par blocs

    /// Processes a prompt in chunks and returns the last token's distribution.
    ///
    /// The computation is **identical** to processing token by token — only the order of the
    /// reads changes. On a 78-token prompt of the 20B, it takes dense-weight re-reads from
    /// 92.9 GiB down to 1.2 GiB. A test verifies that both paths produce the same state.
    ///
    /// The chunk size is bounded by the margin of the sliding-window layers' ring: all of a
    /// chunk's KV writes precede its attentions, so a chunk larger than the margin would
    /// overwrite keys that are still needed.
    @discardableResult
    public func prefill(tokens: [Int]) throws -> UnsafeBufferPointer<Float> {
        precondition(!tokens.isEmpty, "the prompt cannot be empty")
        var timings = Timings()
        let chunkSize = prefillScratch.maximumTokens

        var offset = 0
        var lastChunkCount = 0
        while offset < tokens.count {
            let count = min(chunkSize, tokens.count - offset)
            try prefillChunk(
                Array(tokens[offset..<(offset + count)]),
                firstPosition: position, timings: &timings)
            lastChunkCount = count
            offset += count
        }

        if ProcessInfo.processInfo.environment["HYDRA_PROFILE"] != nil {
            FileHandle.standardError.write(Data(String(
                format: "prefill %d tokens: I/O %.1f s · compute %.1f s · attention %.1f s\n",
                tokens.count, timings.expertIO, timings.mixture,
                timings.attentionAndRouter).utf8))
        }

        // The state to read is the last row of the **last** chunk processed.
        return try finishWithLanguageModelHead(
            from: prefillScratch.hidden,
            rowOffset: (lastChunkCount - 1) * config.hiddenSize,
            timings: &timings)
    }

    private func prefillChunk(
        _ chunk: [Int], firstPosition: Int, timings: inout Timings
    ) throws {
        let count = chunk.count

        // The chunk's embeddings, one row per token.
        let hidden = prefillScratch.hidden.contents().bindMemory(
            to: Float.self, capacity: count * config.hiddenSize)
        for (index, token) in chunk.enumerated() {
            mapping.readEmbedding(
                token: token,
                into: UnsafeMutableBufferPointer(
                    start: hidden + index * config.hiddenSize, count: config.hiddenSize))
        }

        // RoPE tables: each token has its own position.
        let halfDim = config.headDim / 2
        let cosTable = prefillScratch.cosTable.contents().bindMemory(
            to: Float.self, capacity: count * halfDim)
        let sinTable = prefillScratch.sinTable.contents().bindMemory(
            to: Float.self, capacity: count * halfDim)
        for index in 0..<count {
            ropeTables.write(
                position: firstPosition + index,
                cos: UnsafeMutableBufferPointer(start: cosTable + index * halfDim, count: halfDim),
                sin: UnsafeMutableBufferPointer(start: sinTable + index * halfDim, count: halfDim))
        }

        for layer in 0..<config.layerCount {
            var start = Date()
            guard let first = context.commandQueue.makeCommandBuffer() else {
                throw RunnerError.commandBufferUnavailable
            }
            try prefillRunner.encodeAttentionAndRouter(
                layer: layer, tokens: count, firstPosition: firstPosition,
                scratch: prefillScratch, kvCache: kvCache, in: first)
            try prefillRunner.encodeMixtureStart(
                tokens: count, scratch: prefillScratch, in: first)
            first.commit()
            first.waitUntilCompleted()
            timings.attentionAndRouter += Date().timeIntervalSince(start)

            // Experts are processed in **tiles the size of the cache**.
            //
            // A 128-token chunk often touches all of a layer's experts, but the cache holds
            // only a few. Loading them all at once would pin more than its capacity and
            // deadlock eviction. So we advance by tiles: load, compute, release. The number
            // of occupied slots never exceeds that of decoding — which is what makes chunked
            // prefill neutral for memory.
            let assignments = prefillRunner.assignments(prefillScratch, tokens: count)
            let tileSize = max(1, expertCache.slotsPerLayer)

            var index = 0
            while index < assignments.count {
                let tile = Array(assignments[index..<min(index + tileSize, assignments.count)])

                start = Date()
                try expertCache.load(layer: layer, experts: tile.map(\.expert))
                timings.expertIO += Date().timeIntervalSince(start)

                start = Date()
                guard let buffer = context.commandQueue.makeCommandBuffer() else {
                    throw RunnerError.commandBufferUnavailable
                }
                for (slot, assignment) in tile.enumerated() {
                    try prefillRunner.encodeExpert(
                        layer: layer, assignment: assignment, slot: slot,
                        scratch: prefillScratch, in: buffer)
                }
                buffer.commit()
                buffer.waitUntilCompleted()
                expertCache.release(layer: layer)
                timings.mixture += Date().timeIntervalSince(start)

                index += tileSize
            }

            start = Date()
            guard let last = context.commandQueue.makeCommandBuffer() else {
                throw RunnerError.commandBufferUnavailable
            }
            try prefillRunner.encodeMixtureEnd(
                tokens: count, scratch: prefillScratch, in: last)
            last.commit()
            last.waitUntilCompleted()
            timings.mixture += Date().timeIntervalSince(start)
        }

        position += count
        for _ in 0..<count { try kvCache.advance() }
    }

    /// Final norm and LM head over a single row of hidden state.
    private func finishWithLanguageModelHead(
        from buffer: MTLBuffer, rowOffset: Int, timings: inout Timings
    ) throws -> UnsafeBufferPointer<Float> {
        let start = Date()
        guard let head = context.commandQueue.makeCommandBuffer() else {
            throw RunnerError.commandBufferUnavailable
        }
        let finalNorm = try mapping.residentTensor("model.norm.weight")
        try encoder.rmsNorm(
            input: buffer, inputOffset: rowOffset * MemoryLayout<Float>.size,
            scale: finalNorm.buffer, scaleOffset: finalNorm.offset,
            output: scratch.normed, size: config.hiddenSize, eps: config.rmsNormEps, in: head)

        let lmHead = try mapping.residentTensor("lm_head.weight")
        try encoder.denseProjection(
            weights: lmHead.buffer, weightsOffset: lmHead.offset,
            bias: nil, biasOffset: 0,
            input: scratch.normed, inputOffset: 0,
            output: logits, outputOffset: 0,
            rows: config.vocabSize, cols: config.hiddenSize, in: head)
        head.commit()
        head.waitUntilCompleted()
        timings.head += Date().timeIntervalSince(start)
        lastTimings = timings

        return UnsafeBufferPointer(
            start: logits.contents().bindMemory(to: Float.self, capacity: config.vocabSize),
            count: config.vocabSize)
    }

    public func reset() {
        position = 0
        kvCache.reset()
    }

    /// Resets the sampler's pseudo-random sequence.
    ///
    /// Deliberately distinct from `reset()`: two successive generations on the same prompt
    /// must be able to differ, otherwise "regenerate" would always return the same text.
    /// Used to make a measurement reproducible.
    public func resetSampling() {
        samplerState = 0
    }

    // MARK: - Speculative decoding

    /// Produces one or more tokens, verifying a draft when it is worth doing.
    ///
    /// **The output is identical to ordinary decoding**, token for token, at equal seed. This
    /// is not an approximation: the draft only serves to avoid computation. Two properties
    /// guarantee it.
    ///
    /// First, every emitted token consumes exactly one draw, as without speculation: the
    /// pseudo-random sequence is therefore the same. Second, the logits at position `P+i`
    /// were computed only under the assumption of tokens `P..P+i-1`, and are used only if
    /// those tokens were accepted — that is, only if the assumption held.
    ///
    /// The first token is drawn **before** the batched pass. If the draft is wrong from the
    /// start, we fall back on an ordinary step having spent nothing.
    public func step(
        from distribution: UnsafeBufferPointer<Float>,
        draft: [Int], sampling: Sampling
    ) throws -> (tokens: [Int], next: UnsafeBufferPointer<Float>) {
        let first = sample(from: distribution, using: sampling)

        // Rewinding is indispensable: rejecting a draft means removing from the KV cache
        // what was just written into it.
        guard canRewind, draft.count > 1, first == draft[0],
            draft.count <= prefillScratch.maximumTokens
        else {
            return ([first], try forward(token: first))
        }

        let origin = position
        let logits = try verify(tokens: draft)

        var accepted = [first]
        for index in 1..<draft.count {
            let token = sample(from: logits[index - 1], using: sampling)
            accepted.append(token)
            if token != draft[index] {
                // What follows in the cache was computed on a false assumption.
                rewind(to: origin + index)
                return (accepted, try forward(token: token))
            }
        }
        return (accepted, logits[draft.count - 1])
    }

    // MARK: - Speculative verification

    /// Processes `tokens` in one pass and returns the logits of **every** position.
    ///
    /// This is the heart of speculative decoding. An ordinary pass re-reads every weight to
    /// produce a single token; this one re-reads them once to verify `n`. The dense weights —
    /// attention, routers, LM head, 2.88 GiB on the 120B — are read once instead of `n`
    /// times, and experts touched by several tokens of the batch are read once as well.
    ///
    /// Row `i` of the result predicts the token at the position **following** `tokens[i]`.
    /// The caller is responsible for rewinding whatever it does not accept.
    public func verify(tokens: [Int]) throws -> [UnsafeBufferPointer<Float>] {
        precondition(!tokens.isEmpty, "nothing to verify")
        precondition(tokens.count <= prefillScratch.maximumTokens, "batch too large")

        var timings = Timings()
        let count = tokens.count
        let firstPosition = position
        try prefillChunk(tokens, firstPosition: firstPosition, timings: &timings)

        let bytes = count * config.vocabSize * MemoryLayout<Float>.size
        if batchLogits == nil || batchLogits!.length < bytes {
            guard let buffer = context.device.makeBuffer(
                length: bytes, options: .storageModeShared)
            else { throw RunnerError.allocationFailed("batchLogits") }
            batchLogits = buffer
        }
        guard let output = batchLogits else {
            throw RunnerError.allocationFailed("batchLogits")
        }

        let start = Date()
        let head = try commandBuffer()
        let batch = BatchEncoder(context: context)
        let finalNorm = try mapping.residentTensor("model.norm.weight")
        try batch.rmsNorm(
            input: prefillScratch.hidden, scale: finalNorm.buffer,
            scaleOffset: finalNorm.offset, output: prefillScratch.normed,
            size: config.hiddenSize, tokens: count, eps: config.rmsNormEps, in: head)

        let lmHead = try mapping.residentTensor("lm_head.weight")
        try batch.denseProjection(
            weights: lmHead.buffer, weightsOffset: lmHead.offset,
            bias: nil, biasOffset: 0,
            input: prefillScratch.normed, output: output,
            rows: config.vocabSize, cols: config.hiddenSize, tokens: count, in: head)
        head.commit()
        head.waitUntilCompleted()
        timings.head += Date().timeIntervalSince(start)
        lastTimings = timings

        let base = output.contents().bindMemory(
            to: Float.self, capacity: count * config.vocabSize)
        return (0..<count).map {
            UnsafeBufferPointer(
                start: base + $0 * config.vocabSize, count: config.vocabSize)
        }
    }

    /// True if the state can return to an earlier position without being rebuilt.
    public var canRewind: Bool { kvCache.canRewind }

    /// Rewinds the state to `tokens` processed tokens.
    ///
    /// Used to reuse work already done from one conversation turn to the next: the prefix
    /// common to both prompts stays valid, and only what diverges is recomputed.
    public func rewind(to tokens: Int) {
        precondition(tokens <= position, "we do not rewind forwards")
        kvCache.rewind(to: tokens)
        position = tokens
    }

    private func commandBuffer() throws -> MTLCommandBuffer {
        guard let buffer = context.commandQueue.makeCommandBuffer() else {
            throw RunnerError.commandBufferUnavailable
        }
        return buffer
    }

    /// Processes one token and returns the output distribution.
    ///
    /// The vector returned points into a reused buffer: it is valid until the next call.
    /// That is deliberate — copying 201,088 floats on every token for nothing would be the
    /// kind of waste that eventually adds up.
    /// - Parameter needsLogits: pass `false` during prefill, except for the prompt's last
    ///   token. The LM head costs 1.08 GiB of reading — computing it for a token whose
    ///   distribution will never be read is pure waste.
    @discardableResult
    public func forward(
        token: Int, needsLogits: Bool = true
    ) throws -> UnsafeBufferPointer<Float> {
        var timings = Timings()

        // --- Embedding: a single row read, the table stays mapped ---
        let hiddenPointer = UnsafeMutableBufferPointer(
            start: scratch.hidden.contents().bindMemory(
                to: Float.self, capacity: config.hiddenSize),
            count: config.hiddenSize)
        mapping.readEmbedding(token: token, into: hiddenPointer)

        // --- RoPE tables for the current position ---
        ropeTables.write(
            position: position,
            cos: UnsafeMutableBufferPointer(
                start: scratch.cosTable.contents().bindMemory(
                    to: Float.self, capacity: config.headDim / 2),
                count: config.headDim / 2),
            sin: UnsafeMutableBufferPointer(
                start: scratch.sinTable.contents().bindMemory(
                    to: Float.self, capacity: config.headDim / 2),
                count: config.headDim / 2))

        // The only incompressible synchronization point is reading the router: the CPU must
        // know which experts were chosen before it can read their weights. Everything else
        // fits in the same command buffer.
        //
        // Decoding used to make seven round trips per layer — attention, mixture start, one
        // per expert, end. At 90 µs for an empty round trip, that was 168 waits per token
        // over 24 layers. Fusing one layer's mixture with the next layer's attention leaves
        // just one per layer.
        //
        // Overlapping reads with compute was tried and removed, on both models:
        // `ExpertSlotCache.load` already reads the `top_k` experts in parallel, so they
        // arrive together. There is no staggered availability to exploit, and staggering
        // them to create one would cost the read parallelism — 3.0 GB/s instead of 5.7
        // (docs/02-MEASUREMENTS.md, M-022).
        var start = Date()
        let opening = try commandBuffer()
        try layerRunner.encodeAttentionAndRouter(
            layer: 0, position: position, scratch: scratch, kvCache: kvCache, in: opening)
        opening.commit()
        opening.waitUntilCompleted()
        timings.attentionAndRouter += Date().timeIntervalSince(start)

        var selected = layerRunner.selectedExperts(scratch)

        for layer in 0..<config.layerCount {
            let next = layer + 1

            // We compute the experts **already in memory** first, while the missing ones
            // are read.
            //
            // At a 76 % hit rate, on average only one expert in four is missing: three are
            // ready and waiting only for the GPU. Waiting for all of them before starting
            // left the GPU idle for the whole read.
            //
            // The compute order does not affect the result: each expert writes into its own
            // slot, and the sum is then taken in the fixed order of the slots.
            let ready = selected.enumerated().filter {
                expertCache.isResident(layer: layer, expert: $0.element)
            }
            let awaited = selected.enumerated().filter {
                !expertCache.isResident(layer: layer, expert: $0.element)
            }

            start = Date()

            // Encoding is what pins the experts it references. It must therefore precede
            // launching the reads: otherwise those pick the still-free slots as victims —
            // precisely the ones about to be used. Measured: the hit rate fell from 76 to
            // 64 %.
            var warm: MTLCommandBuffer?
            if !ready.isEmpty {
                let buffer = try commandBuffer()
                for (slot, expert) in ready {
                    try layerRunner.encodeSingleExpert(
                        layer: layer, expert: expert, weightIndex: slot,
                        scratch: scratch, in: buffer)
                }
                warm = buffer
            }

            if !awaited.isEmpty {
                let cache = expertCache
                let layerIndex = layer
                let missing = awaited.map(\.element)
                DispatchQueue.global(qos: .userInitiated).async {
                    try? cache.load(layer: layerIndex, experts: missing)
                }
            }

            if let warm {
                warm.commit()
                warm.waitUntilCompleted()
            }
            timings.mixture += Date().timeIntervalSince(start)

            start = Date()
            let buffer = try commandBuffer()
            for (slot, expert) in awaited {
                // Blocks on this expert: the read launched above ran during the compute of
                // the already-warm experts.
                try layerRunner.encodeSingleExpert(
                    layer: layer, expert: expert, weightIndex: slot,
                    scratch: scratch, in: buffer)
            }
            timings.expertIO += Date().timeIntervalSince(start)

            start = Date()
            try layerRunner.encodeCombineSlices(
                count: selected.count, scratch: scratch, in: buffer)

            // Encoders within one buffer run in the order they were created: the next
            // layer's attention will read the residual the mixture has just written.
            if next < config.layerCount {
                try layerRunner.encodeAttentionAndRouter(
                    layer: next, position: position, scratch: scratch, kvCache: kvCache,
                    in: buffer)
            }
            buffer.commit()
            buffer.waitUntilCompleted()
            expertCache.release(layer: layer)
            timings.mixture += Date().timeIntervalSince(start)

            if next < config.layerCount {
                selected = layerRunner.selectedExperts(scratch)
            }
        }

        // --- Final norm and LM head ---
        start = Date()
        guard needsLogits else {
            position += 1
            try kvCache.advance()
            lastTimings = timings
            return UnsafeBufferPointer(
                start: logits.contents().bindMemory(to: Float.self, capacity: config.vocabSize),
                count: config.vocabSize)
        }
        guard let head = context.commandQueue.makeCommandBuffer() else {
            throw RunnerError.commandBufferUnavailable
        }
        let finalNorm = try mapping.residentTensor("model.norm.weight")
        try encoder.rmsNorm(
            input: scratch.hidden, scale: finalNorm.buffer, scaleOffset: finalNorm.offset,
            output: scratch.normed, size: config.hiddenSize, eps: config.rmsNormEps, in: head)

        let lmHead = try mapping.residentTensor("lm_head.weight")
        try encoder.denseProjection(
            weights: lmHead.buffer, weightsOffset: lmHead.offset,
            bias: nil, biasOffset: 0,
            input: scratch.normed, inputOffset: 0,
            output: logits, outputOffset: 0,
            rows: config.vocabSize, cols: config.hiddenSize, in: head)
        head.commit()
        head.waitUntilCompleted()
        timings.head = Date().timeIntervalSince(start)

        position += 1
        try kvCache.advance()
        lastTimings = timings

        return UnsafeBufferPointer(
            start: logits.contents().bindMemory(to: Float.self, capacity: config.vocabSize),
            count: config.vocabSize)
    }

    /// Sampling parameters.
    ///
    /// OpenAI recommends `temperature = 1.0` and `top_p = 1.0` for GPT-OSS — that is, the raw
    /// distribution with no truncation. That is unusual: most models call for something
    /// tighter. We therefore keep those as the defaults here rather than imposing habits
    /// formed elsewhere.
    public struct Sampling: Sendable {
        public var temperature: Float
        public var topP: Float
        public var seed: UInt64

        public init(temperature: Float = 1.0, topP: Float = 1.0, seed: UInt64 = 0x5EED_1234) {
            self.temperature = temperature
            self.topP = topP
            self.seed = seed
        }

        /// Deterministic decoding, for comparing two runs.
        public static let greedy = Sampling(temperature: 0)
    }

    private var samplerState: UInt64 = 0

    /// Draws a token from the distribution.
    ///
    /// `temperature = 0` switches to greedy decoding, which makes the run reproducible —
    /// indispensable for checking that a change in cache size does not alter outputs.
    public func sample(
        from distribution: UnsafeBufferPointer<Float>, using sampling: Sampling
    ) -> Int {
        guard sampling.temperature > 0 else { return greedyToken(from: distribution) }

        // Nothing is allocated at the size of the vocabulary.
        //
        // The previous version built two arrays of 201,088 entries on every token — the
        // probabilities and the ordering — 2.4 MiB to allocate, fill and discard, before
        // sorting the lot to keep about thirty. Here the sum is computed in one pass with no
        // storage, and only the retained candidates are materialized.
        // The maximum falls out of the selection itself: the largest logit is the first
        // candidate. One full pass over the vocabulary saved.
        let candidates = sampling.topP < 1.0
            ? Self.largestIndices(distribution, count: 64) : []
        var peak = -Float.greatestFiniteMagnitude
        if let best = candidates.first {
            peak = distribution[best]
        } else {
            for value in distribution { peak = max(peak, value) }
        }

        let inverseTemperature = 1 / sampling.temperature
        var total: Float = 0
        for value in distribution { total += exp((value - peak) * inverseTemperature) }

        if samplerState == 0 { samplerState = sampling.seed | 1 }
        samplerState &+= 0x9E37_79B9_7F4A_7C15
        var z = samplerState
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        z = z ^ (z >> 31)
        let uniform = Float(Double(z % 1_000_000) / 1_000_000.0)

        guard sampling.topP < 1.0 else {
            // Without truncation a simple walk suffices: the enumeration order does not
            // change the distribution being sampled.
            let target = uniform * total
            var cumulative: Float = 0
            for (index, value) in distribution.enumerated() {
                cumulative += exp((value - peak) * inverseTemperature)
                if cumulative >= target { return index }
            }
            return distribution.count - 1
        }

        // The top-p nucleus holds only a handful of tokens. We extract a small batch,
        // widened only if the target mass is not reached.
        var limit = 64
        var nucleus: [Int] = []
        var mass: Float = 0
        var pool = candidates
        while true {
            nucleus = []
            mass = 0
            var reached = false
            for index in pool {
                nucleus.append(index)
                mass += exp((distribution[index] - peak) * inverseTemperature)
                if mass / total >= sampling.topP { reached = true; break }
            }
            if reached || limit >= distribution.count { break }
            limit = min(limit * 4, distribution.count)
            pool = Self.largestIndices(distribution, count: limit)
        }

        let target = uniform * mass
        var cumulative: Float = 0
        for index in nucleus {
            cumulative += exp((distribution[index] - peak) * inverseTemperature)
            if cumulative >= target { return index }
        }
        return nucleus.last ?? 0
    }

    /// The `count` largest values, in descending order.
    ///
    /// A min-heap of size `count`: a single pass over the vocabulary, and the only
    /// allocation is the heap itself — a few dozen entries, not two hundred thousand.
    static func largestIndices(
        _ values: UnsafeBufferPointer<Float>, count: Int
    ) -> [Int] {
        let size = min(count, values.count)
        guard size > 0 else { return [] }

        var heapValues = [Float](repeating: 0, count: size)
        var heapIndices = [Int](repeating: 0, count: size)
        var filled = 0

        func siftDown(_ start: Int) {
            var parent = start
            while true {
                let left = 2 * parent + 1
                guard left < filled else { return }
                var smallest = left
                let right = left + 1
                if right < filled, heapValues[right] < heapValues[left] { smallest = right }
                if heapValues[parent] <= heapValues[smallest] { return }
                heapValues.swapAt(parent, smallest)
                heapIndices.swapAt(parent, smallest)
                parent = smallest
            }
        }

        for index in 0..<values.count {
            let value = values[index]
            if filled < size {
                heapValues[filled] = value
                heapIndices[filled] = index
                filled += 1
                // Sift up from the inserted leaf.
                var child = filled - 1
                while child > 0 {
                    let parent = (child - 1) / 2
                    if heapValues[parent] <= heapValues[child] { break }
                    heapValues.swapAt(parent, child)
                    heapIndices.swapAt(parent, child)
                    child = parent
                }
            } else if value > heapValues[0] {
                heapValues[0] = value
                heapIndices[0] = index
                siftDown(0)
            }
        }

        return heapIndices[0..<filled].sorted { values[$0] > values[$1] }
    }

    public func greedyToken(from distribution: UnsafeBufferPointer<Float>) -> Int {
        var bestIndex = 0
        var bestValue = -Float.greatestFiniteMagnitude
        for (index, value) in distribution.enumerated() where value > bestValue {
            bestValue = value
            bestIndex = index
        }
        return bestIndex
    }

    /// Generates `count` tokens from a prompt, with greedy decoding.
    ///
    /// The prompt is processed token by token — chunked prefill comes later, it brings
    /// nothing until throughput on long prompts is what we are after.
    public func generate(
        prompt: [Int], count: Int,
        onToken: ((Int, Timings) -> Void)? = nil
    ) throws -> [Int] {
        precondition(!prompt.isEmpty, "the prompt cannot be empty")

        var distribution = try forward(token: prompt[0], needsLogits: prompt.count == 1)
        for (index, token) in prompt.dropFirst().enumerated() {
            distribution = try forward(token: token, needsLogits: index == prompt.count - 2)
        }

        var produced: [Int] = []
        produced.reserveCapacity(count)
        for step in 0..<count {
            let next = greedyToken(from: distribution)
            produced.append(next)
            onToken?(next, lastTimings)
            // The last token produced does not need to be fed back in.
            if step < count - 1 { distribution = try forward(token: next) }
        }
        return produced
    }
}
