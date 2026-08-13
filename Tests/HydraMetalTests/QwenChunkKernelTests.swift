import Foundation
import HydraCore
import HydraFormat
import HydraReference
import Metal
import Testing

@testable import HydraMetal

/// The chunked forms of the two recurrent kernels, against the CPU reference.
///
/// **Against the reference, not against the step kernels.** Comparing a chunk to a loop of steps
/// would be two implementations built from the same reasoning agreeing with each other, which is
/// exactly what let a ten percent error live in `mlx_affine_gemv` through three releases
/// (M-051). The step forms are compared here too, but only as a second, weaker check.
@Suite("Qwen chunked kernels")
struct QwenChunkKernelTests {

    private let keyHeads = 2
    private let valueHeads = 4
    private let keyDim = 16
    private let valueDim = 16
    private let convKernel = 4
    private let eps: Float = 1e-6

    private var keySpan: Int { keyHeads * keyDim }
    private var valueSpan: Int { valueHeads * valueDim }
    private var convDim: Int { 2 * keySpan + valueSpan }

    private func values(_ count: Int, seed: UInt64) -> [Float] {
        var state = seed | 1
        return (0..<count).map { _ in
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float(Int(state >> 33) % 2000 - 1000) / 1000
        }
    }

    private func buffer(_ context: MetalContext, _ v: [Float]) -> MTLBuffer? {
        v.withUnsafeBytes {
            context.device.makeBuffer(
                bytes: $0.baseAddress!, length: max($0.count, 4), options: .storageModeShared)
        }
    }

    private func bf16Buffer(
        _ context: MetalContext, _ v: [Float], pad: Int = 0
    ) -> (buffer: MTLBuffer, offset: Int, rounded: [Float])? {
        let bits = [UInt16](repeating: 0x7F7F, count: pad) + v.map { BF16.fromFloat($0) }
        let rounded = bits.dropFirst(pad).map { BF16.toFloat($0) }
        return bits.withUnsafeBytes {
            context.device.makeBuffer(
                bytes: $0.baseAddress!, length: max($0.count, 4), options: .storageModeShared)
        }.map { ($0, pad * 2, Array(rounded)) }
    }

    // MARK: - The delta rule

    @Test("The chunked delta rule matches the reference over a sequence")
    func deltaRuleChunkMatchesReference() throws {
        let context = try MetalContext()
        let encoder = ForwardEncoder(context: context)
        let tokens = 9

        // One convolved vector a token, q then k then v, which is how the block lays it out.
        let qkvFlat = values(tokens * convDim, seed: 0x51)
        let aFlat = values(tokens * valueHeads, seed: 0x52)
        let bFlat = values(tokens * valueHeads, seed: 0x53)

        guard let qkv = buffer(context, qkvFlat),
            let a = buffer(context, aFlat), let b = buffer(context, bFlat),
            let (logA, logAAt, logARounded) = bf16Buffer(
                context, values(valueHeads, seed: 0x54), pad: 5),
            let (dt, dtAt, dtRounded) = bf16Buffer(
                context, values(valueHeads, seed: 0x55), pad: 3),
            let state = context.device.makeBuffer(
                length: valueHeads * keyDim * valueDim * 4, options: .storageModeShared),
            let output = context.device.makeBuffer(
                length: tokens * valueSpan * 4, options: .storageModeShared),
            let command = context.commandQueue.makeCommandBuffer()
        else { return }
        memset(state.contents(), 0, state.length)

        try encoder.qwenDeltaRuleChunk(
            state: state, stateOffset: 0, qkv: qkv, a: a, b: b,
            logA: logA, logAOffset: logAAt, dtBias: dt, dtBiasOffset: dtAt,
            output: output, tokens: tokens,
            valueHeads: valueHeads, keyHeads: keyHeads, keyDim: keyDim, valueDim: valueDim,
            eps: eps, in: command)
        context.commit(command)
        try context.wait(command)

        // The same sequence on the CPU, one token at a time.
        var reference = (0..<valueHeads).map { _ in
            [[Double]](
                repeating: [Double](repeating: 0, count: valueDim), count: keyDim)
        }
        var expected = [Double](repeating: 0, count: tokens * valueSpan)
        for t in 0..<tokens {
            let row = Array(qkvFlat[(t * convDim)..<((t + 1) * convDim)]).map(Double.init)
            for head in 0..<valueHeads {
                let keyHead = head / (valueHeads / keyHeads)
                let out = QwenReferenceOps.deltaRuleStep(
                    query: Array(row[(keyHead * keyDim)..<((keyHead + 1) * keyDim)]),
                    key: Array(row[(keySpan + keyHead * keyDim)..<(keySpan + (keyHead + 1) * keyDim)]),
                    value: Array(
                        row[(2 * keySpan + head * valueDim)..<(2 * keySpan + (head + 1) * valueDim)]),
                    decay: QwenReferenceOps.decay(
                        a: Double(aFlat[t * valueHeads + head]),
                        logA: Double(logARounded[head]), dtBias: Double(dtRounded[head])),
                    beta: QwenReferenceOps.sigmoid(Double(bFlat[t * valueHeads + head])),
                    state: &reference[head], eps: Double(eps))
                for i in 0..<valueDim { expected[t * valueSpan + head * valueDim + i] = out[i] }
            }
        }

        let got = output.contents().bindMemory(to: Float.self, capacity: tokens * valueSpan)
        var worst = 0.0
        for i in 0..<(tokens * valueSpan) {
            worst = max(worst, abs(Double(got[i]) - expected[i]))
        }
        #expect(worst < 2e-5, "the chunk diverges from the reference by \(worst)")

        // And the state it leaves is the state the reference leaves, which is what the next
        // chunk continues from. A kernel that produced the right outputs and a stale state
        // would pass everything above and be wrong from the next chunk onwards.
        let left = state.contents().bindMemory(
            to: Float.self, capacity: valueHeads * keyDim * valueDim)
        var stateWorst = 0.0
        for head in 0..<valueHeads {
            for i in 0..<keyDim {
                for j in 0..<valueDim {
                    let at = (head * keyDim + i) * valueDim + j
                    stateWorst = max(stateWorst, abs(Double(left[at]) - reference[head][i][j]))
                }
            }
        }
        #expect(stateWorst < 2e-5, "the carried state diverges by \(stateWorst)")
    }

    /// The chunk continues from whatever state it is given, rather than assuming a fresh one.
    @Test("Two chunks in succession equal one chunk of both")
    func chunksCompose() throws {
        let context = try MetalContext()
        let encoder = ForwardEncoder(context: context)
        let tokens = 8
        let qkvFlat = values(tokens * convDim, seed: 0x61)
        let aFlat = values(tokens * valueHeads, seed: 0x62)
        let bFlat = values(tokens * valueHeads, seed: 0x63)

        func run(splitAt: Int?) throws -> [Float] {
            guard let qkv = buffer(context, qkvFlat),
                let a = buffer(context, aFlat), let b = buffer(context, bFlat),
                let (logA, _, _) = bf16Buffer(context, values(valueHeads, seed: 0x64)),
                let (dt, _, _) = bf16Buffer(context, values(valueHeads, seed: 0x65)),
                let state = context.device.makeBuffer(
                    length: valueHeads * keyDim * valueDim * 4, options: .storageModeShared),
                let output = context.device.makeBuffer(
                    length: tokens * valueSpan * 4, options: .storageModeShared)
            else { return [] }
            memset(state.contents(), 0, state.length)

            let spans: [(offset: Int, count: Int)] = splitAt.map {
                [(0, $0), ($0, tokens - $0)]
            } ?? [(0, tokens)]

            for span in spans {
                // Slices of the same buffers, so the second chunk sees the first's state.
                guard let slice = buffer(
                        context,
                        Array(qkvFlat[(span.offset * convDim)..<((span.offset + span.count) * convDim)])),
                    let aSlice = buffer(
                        context,
                        Array(aFlat[(span.offset * valueHeads)..<((span.offset + span.count) * valueHeads)])),
                    let bSlice = buffer(
                        context,
                        Array(bFlat[(span.offset * valueHeads)..<((span.offset + span.count) * valueHeads)])),
                    let partial = context.device.makeBuffer(
                        length: span.count * valueSpan * 4, options: .storageModeShared),
                    let command = context.commandQueue.makeCommandBuffer()
                else { return [] }
                _ = qkv; _ = a; _ = b

                try encoder.qwenDeltaRuleChunk(
                    state: state, stateOffset: 0, qkv: slice, a: aSlice, b: bSlice,
                    logA: logA, dtBias: dt, output: partial, tokens: span.count,
                    valueHeads: valueHeads, keyHeads: keyHeads,
                    keyDim: keyDim, valueDim: valueDim, eps: eps, in: command)
                context.commit(command)
                try context.wait(command)

                let source = partial.contents().bindMemory(
                    to: Float.self, capacity: span.count * valueSpan)
                let destination = output.contents().bindMemory(
                    to: Float.self, capacity: tokens * valueSpan)
                destination.advanced(by: span.offset * valueSpan)
                    .update(from: source, count: span.count * valueSpan)
            }

            let base = output.contents().bindMemory(
                to: Float.self, capacity: tokens * valueSpan)
            return Array(UnsafeBufferPointer(start: base, count: tokens * valueSpan))
        }

        let whole = try run(splitAt: nil)
        let split = try run(splitAt: 3)
        #expect(whole.count == tokens * valueSpan)
        var worst: Float = 0
        for i in 0..<whole.count { worst = max(worst, abs(whole[i] - split[i])) }
        #expect(worst < 1e-5, "splitting the chunk changes the answer by \(worst)")
    }

    // MARK: - The convolution

    @Test("The chunked convolution matches the reference, window carried")
    func convChunkMatchesReference() throws {
        let context = try MetalContext()
        let encoder = ForwardEncoder(context: context)
        let tokens = 7
        let inputFlat = values(tokens * convDim, seed: 0x71)

        guard let input = buffer(context, inputFlat),
            let (weight, weightAt, weightRounded) = bf16Buffer(
                context, values(convDim * convKernel, seed: 0x72), pad: 6),
            let (bias, biasAt, biasRounded) = bf16Buffer(
                context, values(convDim, seed: 0x73), pad: 2),
            let window = context.device.makeBuffer(
                length: (convKernel - 1) * convDim * 4, options: .storageModeShared),
            let output = context.device.makeBuffer(
                length: tokens * convDim * 4, options: .storageModeShared),
            let command = context.commandQueue.makeCommandBuffer()
        else { return }
        memset(window.contents(), 0, window.length)

        try encoder.qwenCausalConvChunk(
            window: window, windowOffset: 0, input: input,
            weight: weight, weightOffset: weightAt,
            bias: bias, biasOffset: biasAt,
            output: output, tokens: tokens, convDim: convDim, kernel: convKernel,
            in: command)
        context.commit(command)
        try context.wait(command)

        let weight64 = (0..<convDim).map { c in
            (0..<convKernel).map { Double(weightRounded[c * convKernel + $0]) }
        }
        let bias64 = biasRounded.map(Double.init)
        var history: [[Double]] = []
        let got = output.contents().bindMemory(to: Float.self, capacity: tokens * convDim)
        for t in 0..<tokens {
            let row = Array(inputFlat[(t * convDim)..<((t + 1) * convDim)]).map(Double.init)
            let expected = QwenReferenceOps.causalDepthwiseConv(
                input: row, history: history, weight: weight64, bias: bias64)
            for c in 0..<convDim {
                #expect(
                    abs(Double(got[t * convDim + c]) - expected[c]) < 1e-5,
                    "token \(t), channel \(c)")
            }
            history.append(row)
            if history.count > convKernel - 1 { history.removeFirst() }
        }

        // The window the chunk leaves is the last `kernel - 1` inputs, oldest first: the next
        // chunk's first tokens are computed against it, and a stale one is wrong only there.
        let held = window.contents().bindMemory(
            to: Float.self, capacity: (convKernel - 1) * convDim)
        for (rowIndex, source) in [tokens - 3, tokens - 2, tokens - 1].enumerated() {
            for c in 0..<convDim {
                #expect(
                    abs(held[rowIndex * convDim + c] - inputFlat[source * convDim + c]) < 1e-6,
                    "window row \(rowIndex)")
            }
        }
    }

    /// The convolution continues from the window it is given.
    ///
    /// Every other test here starts from a zeroed window, where seeding from the carried one
    /// and not seeding at all are the same thing. A falsification pass found that by deleting
    /// the seeding and passing, which is what this test exists to stop.
    @Test("Two convolution chunks in succession equal one chunk of both")
    func convChunksCompose() throws {
        let context = try MetalContext()
        let encoder = ForwardEncoder(context: context)
        let tokens = 9
        let inputFlat = values(tokens * convDim, seed: 0x91)

        func run(splitAt: Int?) throws -> [Float] {
            guard let (weight, _, _) = bf16Buffer(
                    context, values(convDim * convKernel, seed: 0x92)),
                let window = context.device.makeBuffer(
                    length: (convKernel - 1) * convDim * 4, options: .storageModeShared),
                let output = context.device.makeBuffer(
                    length: tokens * convDim * 4, options: .storageModeShared)
            else { return [] }
            memset(window.contents(), 0, window.length)

            let spans: [(offset: Int, count: Int)] =
                splitAt.map { [(0, $0), ($0, tokens - $0)] } ?? [(0, tokens)]
            for span in spans {
                guard let slice = buffer(
                        context,
                        Array(inputFlat[(span.offset * convDim)..<((span.offset + span.count) * convDim)])),
                    let partial = context.device.makeBuffer(
                        length: span.count * convDim * 4, options: .storageModeShared),
                    let command = context.commandQueue.makeCommandBuffer()
                else { return [] }
                try encoder.qwenCausalConvChunk(
                    window: window, windowOffset: 0, input: slice,
                    weight: weight, bias: nil, output: partial,
                    tokens: span.count, convDim: convDim, kernel: convKernel, in: command)
                context.commit(command)
                try context.wait(command)
                output.contents().advanced(by: span.offset * convDim * 4)
                    .copyMemory(from: partial.contents(), byteCount: span.count * convDim * 4)
            }
            let base = output.contents().bindMemory(
                to: Float.self, capacity: tokens * convDim)
            return Array(UnsafeBufferPointer(start: base, count: tokens * convDim))
        }

        let whole = try run(splitAt: nil)
        let split = try run(splitAt: 4)
        #expect(whole.count == tokens * convDim)
        var worst: Float = 0
        for i in 0..<whole.count { worst = max(worst, abs(whole[i] - split[i])) }
        #expect(worst < 1e-6, "splitting the chunk changes the answer by \(worst)")
    }

    /// The chunk and the step agree, which is the weaker check of the two.
    @Test("The chunked convolution agrees with the step form")
    func convChunkAgreesWithStep() throws {
        let context = try MetalContext()
        let encoder = ForwardEncoder(context: context)
        let tokens = 5
        let inputFlat = values(tokens * convDim, seed: 0x81)

        guard let input = buffer(context, inputFlat),
            let (weight, _, _) = bf16Buffer(context, values(convDim * convKernel, seed: 0x82)),
            let windowA = context.device.makeBuffer(
                length: (convKernel - 1) * convDim * 4, options: .storageModeShared),
            let windowB = context.device.makeBuffer(
                length: (convKernel - 1) * convDim * 4, options: .storageModeShared),
            let chunked = context.device.makeBuffer(
                length: tokens * convDim * 4, options: .storageModeShared),
            let stepped = context.device.makeBuffer(
                length: tokens * convDim * 4, options: .storageModeShared)
        else { return }
        memset(windowA.contents(), 0, windowA.length)
        memset(windowB.contents(), 0, windowB.length)

        guard let command = context.commandQueue.makeCommandBuffer() else { return }
        try encoder.qwenCausalConvChunk(
            window: windowA, windowOffset: 0, input: input, weight: weight, bias: nil,
            output: chunked, tokens: tokens, convDim: convDim, kernel: convKernel, in: command)
        context.commit(command)
        try context.wait(command)

        for t in 0..<tokens {
            guard let row = buffer(
                    context, Array(inputFlat[(t * convDim)..<((t + 1) * convDim)])),
                let single = context.device.makeBuffer(
                    length: convDim * 4, options: .storageModeShared),
                let step = context.commandQueue.makeCommandBuffer()
            else { return }
            try encoder.qwenCausalConvStep(
                window: windowB, windowOffset: 0, input: row, weight: weight, bias: nil,
                output: single, convDim: convDim, kernel: convKernel, in: step)
            context.commit(step)
            try context.wait(step)
            let source = single.contents().bindMemory(to: Float.self, capacity: convDim)
            stepped.contents().advanced(by: t * convDim * 4)
                .copyMemory(from: source, byteCount: convDim * 4)
        }

        let a = chunked.contents().bindMemory(to: Float.self, capacity: tokens * convDim)
        let b = stepped.contents().bindMemory(to: Float.self, capacity: tokens * convDim)
        for i in 0..<(tokens * convDim) {
            #expect(abs(a[i] - b[i]) < 1e-6, "element \(i)")
        }
    }
}
