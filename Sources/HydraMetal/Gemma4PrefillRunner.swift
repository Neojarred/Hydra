import Foundation
import HydraCore
import Metal

/// Processes a prompt layer by layer instead of token by token.
///
/// **The reordering is the whole optimization, and it is about I/O rather than arithmetic.**
///
/// Token-major prefill, the obvious loop, and what this replaces, reads a layer's experts,
/// moves to the next layer, and by the time it returns to that layer for the following token
/// the eight slots hold someone else's experts. Every token pays for its own eight reads at
/// every layer. Measured on the real model that is 1.7 GiB from SSD **per token**, and prefill
/// ran at roughly one token a second: a twenty-five token prompt cost the user twenty seconds
/// before a single word appeared.
///
/// Layer-major with the experts grouped changes the count. Within one layer, the tokens of a
/// chunk select from the same 128 experts and overlap heavily, so each distinct expert is read
/// **once for the whole chunk** rather than once per token that wanted it.
///
/// Nothing about the arithmetic changes, and that is deliberate: every kernel here is the one
/// the decode path already uses, called with different offsets. A second implementation of the
/// expert math would run, return finite numbers, and disagree with decoding by a little, the
/// failure this project keeps meeting. The test that matters asserts the batched result is
/// bit-identical to feeding the same tokens one at a time.
public final class Gemma4PrefillRunner {

    /// Tokens processed together.
    ///
    /// Measured, at 800 prompt tokens with the app's cache settings, the two costs pull
    /// against each other, so the curve has an interior optimum:
    ///
    ///     chunk        128    256    384    512
    ///     total (s)   27.7   25.4   25.7   27.8
    ///     expert I/O   4.5    3.4    3.0    3.9
    ///     cb1         15.3   15.2   16.2   17.5
    ///
    /// A larger chunk shares each expert's read across more tokens, which is what the expert
    /// I/O column shows. But the batched projections re-read the activations far more than the
    /// weights, and past 256 the transposed activation buffer stops fitting in cache, 16.8 MiB
    /// at 512 against 4.2 at 128, which is what takes cb1 back up.
    ///
    /// Bounded by memory, not by principle: the per-token buffers below are `chunk × hidden`,
    /// and the expert slices are `chunk × topK × hidden`, 11.5 MiB at 128 tokens for the real
    /// model. Larger chunks share expert reads better, so this is the knob that decides how
    /// much of the win is collected.
    public static let chunk = 256

    private let config: Gemma4Config
    private let encoder: ForwardEncoder
    private let layerRunner: Gemma4LayerRunner
    private let mapping: ModelMapping

    /// One row per token, for everything that has to outlive the token's turn in the scratch.
    private let hidden: MTLBuffer        // [chunk × hidden]
    private let residual: MTLBuffer      // [chunk × hidden]
    private let denseOutput: MTLBuffer   // [chunk × hidden]
    private let expertInput: MTLBuffer   // [chunk × hidden]
    private let routerWeights: MTLBuffer // [chunk × topK]
    private let routerIndices: MTLBuffer // [chunk × topK] UInt32
    /// [chunk × topK × hidden], every contribution keeps its own slot, so the sum's order is
    /// the slot order and never the order the experts happened to be read in.
    private let expertSlices: MTLBuffer

    /// One rotary table per token, [chunk × maxHeadDim / 2].
    ///
    /// The scratch has a single pair, rewritten by the CPU before each token. That is correct
    /// only when each token is committed before the next is encoded. Holding a row per token is
    /// what allows a layer's whole chunk to go into one command buffer.
    private let cosTables: MTLBuffer
    private let sinTables: MTLBuffer
    private let tablePairs: Int

    /// The chunk-wide buffers the staged path works in, or nil when the model has no batched
    /// projection, the BF16 build, which keeps the per-token path.
    private let chunkScratch: Gemma4ChunkScratch?

    public enum PrefillError: Error, CustomStringConvertible {
        case allocationFailed(String)

        public var description: String {
            switch self {
            case .allocationFailed(let what): return "prefill: cannot allocate \(what)"
            }
        }
    }

    init(
        config: Gemma4Config, encoder: ForwardEncoder,
        layerRunner: Gemma4LayerRunner, mapping: ModelMapping, device: MTLDevice,
        staged: Bool = true
    ) throws {
        self.config = config
        self.encoder = encoder
        self.layerRunner = layerRunner
        self.mapping = mapping

        let float = MemoryLayout<Float>.size
        let rows = Self.chunk * config.hiddenSize * float
        func make(_ bytes: Int, _ name: String) throws -> MTLBuffer {
            guard let buffer = device.makeBuffer(length: bytes, options: .storageModeShared)
            else { throw PrefillError.allocationFailed(name) }
            return buffer
        }
        self.hidden = try make(rows, "hidden")
        self.residual = try make(rows, "residual")
        self.denseOutput = try make(rows, "denseOutput")
        self.expertInput = try make(rows, "expertInput")
        self.routerWeights = try make(Self.chunk * config.expertsPerToken * float, "routerWeights")
        self.routerIndices = try make(Self.chunk * config.expertsPerToken * 4, "routerIndices")
        self.expertSlices = try make(Self.chunk * config.expertsPerToken * rows / Self.chunk,
                                     "expertSlices")
        let maxHeadDim = (0..<config.layerCount)
            .map { config.attentionGeometry(atLayer: $0).headDim }.max() ?? 0
        self.tablePairs = maxHeadDim / 2
        self.cosTables = try make(Self.chunk * tablePairs * float, "cosTables")
        self.sinTables = try make(Self.chunk * tablePairs * float, "sinTables")

        self.chunkScratch = staged && layerRunner.supportsChunkedPath
            ? try Gemma4ChunkScratch(config: config, tokens: Self.chunk, device: device)
            : nil
    }

    public var byteCount: Int {
        hidden.length + residual.length + denseOutput.length + expertInput.length
            + routerWeights.length + routerIndices.length + expertSlices.length
            + cosTables.length + sinTables.length
    }

    private func rowOffset(_ token: Int) -> Int {
        token * config.hiddenSize * MemoryLayout<Float>.size
    }

    /// Runs one chunk. `hiddenStates` holds the scaled embeddings of every token in it, and is
    /// overwritten with the states leaving the last layer.
    ///
    /// - Parameter firstPosition: the position of `tokens[0]` in the conversation, which is what
    ///   the rotary tables and the attention window are computed from.
    func run(
        tokenCount: Int, firstPosition: Int,
        embeddings: (Int, UnsafeMutableBufferPointer<Float>) -> Void,
        scratch: Gemma4DecodeScratch, kvCache: KVCache, expertCache: ExpertSlotCache,
        ropeTables: [Gemma4RoPETables], commandBuffer: () throws -> MTLCommandBuffer,
        timings: inout ModelRunner.Timings
    ) throws -> Int {
        precondition(tokenCount <= Self.chunk)
        let size = config.hiddenSize
        let float = MemoryLayout<Float>.size

        // The embeddings, written straight into the batch rows.
        let base = hidden.contents().bindMemory(to: Float.self, capacity: Self.chunk * size)
        for token in 0..<tokenCount {
            let row = UnsafeMutableBufferPointer(
                start: base.advanced(by: token * size), count: size)
            embeddings(token, row)
        }

        for layer in 0..<config.layerCount {
            // --- Phase A: attention and the router, the whole chunk in one pass ---
            //
            // Dispatches inside a command buffer run in order, so token t's attention already
            // sees the keys and values token t-1 wrote: the dependency that makes prefill look
            // sequential does not need a synchronization.
            //
            // What it does need is a rotary table per token. With the scratch's single pair,
            // encoding the chunk before committing gave every token the table the CPU wrote
            // last, finite, plausible, and wrong by a little. The batched-versus-sequential
            // test caught that, which is the only reason it is not in the shipped path.
            var start = Date()
            for token in 0..<tokenCount {
                writeRopeTables(
                    ropeTables[layer], at: firstPosition + token,
                    slot: token)
            }
            let attention = try commandBuffer()
            if let chunk = chunkScratch {
                // Staged: one dispatch a stage for the whole chunk, rather than twenty a token.
                //
                // The per-token loop issued about 334,000 dispatches for a 557-token prompt and
                // 13.7 s of the 21.8 s this phase measured, most of it moving a few kilobytes
                // at a time. Every kernel in the staged path is the one the per-token path
                // uses, so the result is unchanged; only the token index moves from a Swift
                // loop onto the grid.
                try layerRunner.encodeChunk(
                    layer: layer, firstPosition: firstPosition, tokens: tokenCount,
                    hidden: hidden, residual: residual, denseOutput: denseOutput,
                    expertInput: expertInput, routerLogits: chunk.routerLogits,
                    scratch: chunk, kvCache: kvCache,
                    ropeCos: cosTables, ropeSin: sinTables, tableStride: tablePairs,
                    in: attention)
                let scale = try layerRunner.weights.plain(
                    "router.per_expert_scale", layer: layer)
                // One dispatch, one threadgroup a token. Per token this was a dispatch of a
                // single threadgroup in the prompt's critical path; the same pattern on Qwen
                // was 19.4 s of a 60 s prompt (M-061).
                try encoder.gemmaRouterTopKBatched(
                    logits: chunk.routerLogits,
                    perExpertScale: scale.buffer, perExpertScaleOffset: scale.offset,
                    indices: routerIndices, weights: routerWeights,
                    expertCount: config.expertCount, topK: config.expertsPerToken,
                    tokens: tokenCount, in: attention)
            } else {
                for token in 0..<tokenCount {
                    try encoder.copy(
                        into: scratch.hidden, destinationOffset: 0,
                        from: hidden, sourceOffset: rowOffset(token), size: size, in: attention)
                    try layerRunner.encodeAttentionAndRouter(
                        layer: layer, position: firstPosition + token,
                        scratch: scratch, kvCache: kvCache,
                        ropeCos: cosTables, ropeSin: sinTables,
                        ropeOffset: token * tablePairs * float,
                        in: attention)

                    let scale = try layerRunner.weights.plain(
                        "router.per_expert_scale", layer: layer)
                    try encoder.gemmaRouterTopK(
                        logits: scratch.routerLogits,
                        perExpertScale: scale.buffer, perExpertScaleOffset: scale.offset,
                        indices: routerIndices,
                        indicesOffset: token * config.expertsPerToken * 4,
                        weights: routerWeights,
                        weightsOffset: token * config.expertsPerToken * float,
                        expertCount: config.expertCount, topK: config.expertsPerToken,
                        in: attention)

                    for (destination, source) in [
                        (residual, scratch.residual),
                        (denseOutput, scratch.denseOutput),
                        (expertInput, scratch.expertInput),
                    ] {
                        try encoder.copy(
                            into: destination, destinationOffset: rowOffset(token),
                            from: source, sourceOffset: 0, size: size, in: attention)
                    }
                }
            }
            encoder.commit(attention)
            try encoder.context.wait(attention)
            timings.attentionAndRouter += Date().timeIntervalSince(start)

            // --- Phase B: the experts, grouped so each is read once ---
            //
            // The rank within a token's selection is carried through, because the slot decides
            // the order of the final sum. Reading the experts in a different order must not
            // change a single bit of the result.
            let selection = readSelection(tokenCount: tokenCount)
            var groups: [Int: [(token: Int, rank: Int)]] = [:]
            for token in 0..<tokenCount {
                for rank in 0..<config.expertsPerToken {
                    groups[selection[token][rank], default: []].append((token, rank))
                }
            }

            // The slots are not cleared, because the groups above partition them.
            //
            // Every (token, rank) pair is appended to exactly one group, every group is
            // processed below, and `writeExpertScaled` *assigns* the whole slot rather than
            // accumulating into it. So each of the `tokenCount * expertsPerToken` slots the
            // sum reads is written first, and a zero fill only wrote bytes that were
            // overwritten before anything read them: for a 256-token chunk of the 26B, 22 MiB
            // a layer, 650 MiB a chunk, plus a commit and a wait per layer to do it.
            //
            // What replaces it is the invariant itself, checked. A slot left out would be read
            // holding whatever the previous chunk or layer put there: a plausible number in the
            // sum and no error anywhere, which is the failure this asserts away.
            var written = [Bool](repeating: false, count: tokenCount * config.expertsPerToken)
            for (expert, members) in groups {
                for member in members {
                    let slot = member.token * config.expertsPerToken + member.rank
                    precondition(
                        !written[slot],
                        "expert \(expert) claims slot \(slot), which another group already writes")
                    written[slot] = true
                }
            }
            precondition(
                written.allSatisfy { $0 },
                "an expert slot would be summed without anything writing it")

            // Loaded in slot-sized batches so the reads run in parallel.
            //
            // One expert at a time was the first version, and it quietly threw away what the
            // slot cache is for: `load` issues its `pread`s concurrently, and asking for a
            // single expert makes that a serial read. The batch is the slot count, because
            // that is how many can be pinned at once.
            // Half the slots a batch, so two batches are resident at once and the reads for
            // the next one run while the GPU works on this one. On Qwen the same change took
            // `expert I/O` from 14.7 s to 7.2 and the prompt from 41.7 s to 34.0 (M-062).
            let ordered = groups.keys.sorted()
            let batchSize = max(1, expertCache.slotsPerLayer / 2)
            var batches: [[Int]] = []
            var cursor = 0
            while cursor < ordered.count {
                let end = min(cursor + batchSize, ordered.count)
                batches.append(Array(ordered[cursor..<end]))
                cursor = end
            }

            final class Prefetch: @unchecked Sendable {
                let done = DispatchSemaphore(value: 0)
                var failure: Error?
            }
            var inFlight: Prefetch? = nil

            for (position, batch) in batches.enumerated() {
                var phase = Date()
                if let prefetch = inFlight {
                    prefetch.done.wait()
                    if let failure = prefetch.failure { throw failure }
                    inFlight = nil
                } else {
                    try expertCache.load(layer: layer, experts: batch)
                }
                timings.expertIO += Date().timeIntervalSince(phase)

                phase = Date()
                let mixture = try commandBuffer()
                for expert in batch {
                    let (blob, _) = try expertCache.expert(
                        layer: layer, expert: expert, pin: true)
                    let members = groups[expert]!

                    if let chunk = chunkScratch {
                        // The group's tokens gathered, so the expert's weights are read once
                        // for all of them rather than once each.
                        for (row, member) in members.enumerated() {
                            try encoder.copy(
                                into: chunk.expertInput, destinationOffset: row * size * float,
                                from: expertInput, sourceOffset: rowOffset(member.token),
                                size: size, in: mixture)
                        }
                        let output = try layerRunner.encodeExpertGroup(
                            blob: blob, members: members.count, scratch: chunk, in: mixture)
                        // Scattered back by slot, which is what fixes the sum's order.
                        for (row, member) in members.enumerated() {
                            let slot = member.token * config.expertsPerToken + member.rank
                            try encoder.writeExpertScaled(
                                into: expertSlices, outputOffset: slot * size * float,
                                contribution: output, contributionOffset: row * size * float,
                                weights: routerWeights, weightIndex: slot, size: size,
                                in: mixture)
                        }
                        continue
                    }

                    for member in members {
                        try encoder.copy(
                            into: scratch.expertInput, destinationOffset: 0,
                            from: expertInput, sourceOffset: rowOffset(member.token),
                            size: size, in: mixture)
                        try layerRunner.encodeSingleExpert(
                            buffer: blob, weightIndex: member.rank, scratch: scratch,
                            destination: expertSlices,
                            destinationSlot: member.token * config.expertsPerToken + member.rank,
                            routerWeights: routerWeights,
                            weightSlot: member.token * config.expertsPerToken + member.rank,
                            in: mixture)
                    }
                }
                encoder.commit(mixture)

                // After the commit and before the wait: this is the overlap.
                if position + 1 < batches.count {
                    let next = batches[position + 1]
                    let prefetch = Prefetch()
                    inFlight = prefetch
                    let cache = expertCache
                    DispatchQueue.global(qos: .userInitiated).async {
                        do { try cache.load(layer: layer, experts: next) }
                        catch { prefetch.failure = error }
                        prefetch.done.signal()
                    }
                }

                try encoder.context.wait(mixture)
                // This batch only: releasing the layer would unpin the batch the prefetch has
                // just filled, which the batch after it could then evict.
                expertCache.release(layer: layer, experts: batch)
                timings.mixture += Date().timeIntervalSince(phase)
            }

            // --- Phase C: close the layer, token by token, in one pass ---
            start = Date()
            let combine = try commandBuffer()
            for token in 0..<tokenCount {
                try encoder.copy(
                    into: scratch.residual, destinationOffset: 0,
                    from: residual, sourceOffset: rowOffset(token), size: size, in: combine)
                try encoder.copy(
                    into: scratch.denseOutput, destinationOffset: 0,
                    from: denseOutput, sourceOffset: rowOffset(token), size: size, in: combine)
                try encoder.copy(
                    into: scratch.expertSlices, destinationOffset: 0,
                    from: expertSlices,
                    sourceOffset: token * config.expertsPerToken * size * float,
                    size: config.expertsPerToken * size, in: combine)
                try layerRunner.encodeCombineBranches(
                    layer: layer, count: config.expertsPerToken,
                    scratch: scratch, in: combine)
                try encoder.copy(
                    into: hidden, destinationOffset: rowOffset(token),
                    from: scratch.hidden, sourceOffset: 0, size: size, in: combine)
            }
            encoder.commit(combine)
            try encoder.context.wait(combine)
            timings.mixture += Date().timeIntervalSince(start)
        }

        // The caller wants the final token's state, which is where the head reads from.
        let final = try commandBuffer()
        try encoder.copy(
            into: scratch.hidden, destinationOffset: 0,
            from: hidden, sourceOffset: rowOffset(tokenCount - 1),
            size: size, in: final)
        encoder.commit(final)
        try encoder.context.wait(final)
        return tokenCount
    }

    private func writeRopeTables(
        _ tables: Gemma4RoPETables, at position: Int, slot: Int
    ) {
        let computed = tables.tables(at: position)
        let pairs = computed.cos.count
        let cos = cosTables.contents().bindMemory(
            to: Float.self, capacity: Self.chunk * tablePairs)
        let sin = sinTables.contents().bindMemory(
            to: Float.self, capacity: Self.chunk * tablePairs)
        for i in 0..<pairs {
            cos[slot * tablePairs + i] = Float(computed.cos[i])
            sin[slot * tablePairs + i] = Float(computed.sin[i])
        }
    }

    private func readSelection(tokenCount: Int) -> [[Int]] {
        let pointer = routerIndices.contents().bindMemory(
            to: UInt32.self, capacity: Self.chunk * config.expertsPerToken)
        return (0..<tokenCount).map { token in
            (0..<config.expertsPerToken).map {
                Int(pointer[token * config.expertsPerToken + $0])
            }
        }
    }
}
