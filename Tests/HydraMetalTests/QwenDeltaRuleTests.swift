import Foundation
import HydraCore
import HydraFormat
import HydraReference
import Metal
import Testing

@testable import HydraMetal

/// The delta rule on the GPU, against the CPU reference that was itself checked against an
/// independent Python transcription.
///
/// Run over a sequence with the state carried, never a single step. A kernel that decays after
/// reading, or that loses the state between dispatches, agrees on the first token and diverges
/// from the second, so a one-token comparison would pass on both.
@Suite("Qwen delta rule on GPU")
struct QwenDeltaRuleTests {

    private let valueHeads = 4
    private let keyHeads = 2
    // Deliberately different, though Qwen's are both 128.
    //
    // With them equal, a kernel scaling the query by `1/sqrt(valueDim)` instead of
    // `1/sqrt(keyDim)` is indistinguishable, and that is exactly the mistake D-027 warns about.
    // The reference handles unequal dimensions, so the test can too, and pins which one the
    // scale reads.
    private let keyDim = 16
    private let valueDim = 12
    private let eps: Float = 1e-6

    /// Deterministic, modest in magnitude, and different for every (kind, index) pair.
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

    /// `A_log` and `dt_bias` in the precision the checkpoint stores them in, BF16.
    ///
    /// Both feed an exponential, so rounding them is not a small perturbation of the output:
    /// building them as float32 here tested a kernel that could not read the real model.
    ///
    /// - Parameter pad: junk in front, so the tensor starts at a non-zero offset, as every
    ///   tensor in `resident.bin` does.
    private func bf16Buffer(
        _ context: MetalContext, _ v: [Float], pad: Int = 0
    ) -> (buffer: MTLBuffer, offset: Int, rounded: [Float])? {
        let bits = [UInt16](repeating: 0x7F7F, count: pad) + v.map { BF16.fromFloat($0) }
        return bits.withUnsafeBytes {
            context.device.makeBuffer(
                bytes: $0.baseAddress!, length: max($0.count, 4), options: .storageModeShared)
        }.map { ($0, pad * 2, Array(bits.dropFirst(pad)).map { BF16.toFloat($0) }) }
    }

    @Test("The kernel matches the CPU reference across a carried sequence")
    func matchesReference() throws {
        let context = try MetalContext()
        let encoder = ForwardEncoder(context: context)
        let tokens = 6

        let logA = values(valueHeads, seed: 0x10)
        let dtBias = values(valueHeads, seed: 0x20)
        let queries = (0..<tokens).map { values(keyHeads * keyDim, seed: 0x100 + UInt64($0)) }
        let keys = (0..<tokens).map { values(keyHeads * keyDim, seed: 0x200 + UInt64($0)) }
        let vals = (0..<tokens).map { values(valueHeads * valueDim, seed: 0x300 + UInt64($0)) }
        let aSeq = (0..<tokens).map { values(valueHeads, seed: 0x400 + UInt64($0)) }
        let bSeq = (0..<tokens).map { values(valueHeads, seed: 0x500 + UInt64($0)) }

        guard let state = context.device.makeBuffer(
                length: valueHeads * keyDim * valueDim * 4, options: .storageModeShared),
            let output = context.device.makeBuffer(
                length: valueHeads * valueDim * 4, options: .storageModeShared),
            let (logABuffer, logAAt, logARounded) = bf16Buffer(context, logA, pad: 9),
            let (dtBuffer, dtAt, dtRounded) = bf16Buffer(context, dtBias, pad: 2)
        else { return }
        memset(state.contents(), 0, state.length)

        // The reference's state, carried alongside, one per value head.
        var reference = (0..<valueHeads).map { _ in
            [[Double]](repeating: [Double](repeating: 0, count: valueDim), count: keyDim)
        }

        for t in 0..<tokens {
            guard let q = buffer(context, queries[t]), let k = buffer(context, keys[t]),
                let v = buffer(context, vals[t]), let a = buffer(context, aSeq[t]),
                let b = buffer(context, bSeq[t]),
                let command = context.commandQueue.makeCommandBuffer()
            else { return }

            try encoder.qwenDeltaRuleStep(
                state: state, stateOffset: 0, query: q, key: k, value: v,
                a: a, b: b,
                logA: logABuffer, logAOffset: logAAt,
                dtBias: dtBuffer, dtBiasOffset: dtAt, output: output,
                valueHeads: valueHeads, keyHeads: keyHeads, keyDim: keyDim,
                valueDim: valueDim, eps: eps, in: command)
            context.commit(command)
            try context.wait(command)

            let got = output.contents().bindMemory(
                to: Float.self, capacity: valueHeads * valueDim)

            for head in 0..<valueHeads {
                let keyHead = head / (valueHeads / keyHeads)
                let q64 = (0..<keyDim).map { Double(queries[t][keyHead * keyDim + $0]) }
                let k64 = (0..<keyDim).map { Double(keys[t][keyHead * keyDim + $0]) }
                let v64 = (0..<valueDim).map { Double(vals[t][head * valueDim + $0]) }

                let g = QwenReferenceOps.decay(
                    a: Double(aSeq[t][head]), logA: Double(logARounded[head]),
                    dtBias: Double(dtRounded[head]))
                let beta = QwenReferenceOps.sigmoid(Double(bSeq[t][head]))
                let expected = QwenReferenceOps.deltaRuleStep(
                    query: q64, key: k64, value: v64, decay: g, beta: beta,
                    state: &reference[head], eps: Double(eps))

                for j in 0..<valueDim {
                    let deviation = abs(Double(got[head * valueDim + j]) - expected[j])
                    #expect(
                        deviation < 2e-5,
                        "token \(t), head \(head), component \(j): the carried state diverges")
                }
            }
        }
    }

    /// The state the kernel leaves behind is the state the reference leaves behind.
    ///
    /// Checked separately from the outputs because a kernel can produce the right answer for a
    /// token and still corrupt what it hands to the next one, and that only shows up later.
    @Test("The kernel leaves the same state as the reference")
    func stateMatchesAfterSequence() throws {
        let context = try MetalContext()
        let encoder = ForwardEncoder(context: context)
        let tokens = 4

        let logA = values(valueHeads, seed: 0x11)
        let dtBias = values(valueHeads, seed: 0x21)
        guard let state = context.device.makeBuffer(
                length: valueHeads * keyDim * valueDim * 4, options: .storageModeShared),
            let output = context.device.makeBuffer(
                length: valueHeads * valueDim * 4, options: .storageModeShared),
            let (logABuffer, logAAt, logARounded) = bf16Buffer(context, logA, pad: 9),
            let (dtBuffer, dtAt, dtRounded) = bf16Buffer(context, dtBias, pad: 2)
        else { return }
        memset(state.contents(), 0, state.length)

        var reference = (0..<valueHeads).map { _ in
            [[Double]](repeating: [Double](repeating: 0, count: valueDim), count: keyDim)
        }

        for t in 0..<tokens {
            let qs = values(keyHeads * keyDim, seed: 0x600 + UInt64(t))
            let ks = values(keyHeads * keyDim, seed: 0x700 + UInt64(t))
            let vs = values(valueHeads * valueDim, seed: 0x800 + UInt64(t))
            let asx = values(valueHeads, seed: 0x900 + UInt64(t))
            let bs = values(valueHeads, seed: 0xA00 + UInt64(t))

            guard let q = buffer(context, qs), let k = buffer(context, ks),
                let v = buffer(context, vs), let a = buffer(context, asx),
                let b = buffer(context, bs),
                let command = context.commandQueue.makeCommandBuffer()
            else { return }
            try encoder.qwenDeltaRuleStep(
                state: state, stateOffset: 0, query: q, key: k, value: v,
                a: a, b: b,
                logA: logABuffer, logAOffset: logAAt,
                dtBias: dtBuffer, dtBiasOffset: dtAt, output: output,
                valueHeads: valueHeads, keyHeads: keyHeads, keyDim: keyDim,
                valueDim: valueDim, eps: eps, in: command)
            context.commit(command)
            try context.wait(command)

            for head in 0..<valueHeads {
                let keyHead = head / (valueHeads / keyHeads)
                _ = QwenReferenceOps.deltaRuleStep(
                    query: (0..<keyDim).map { Double(qs[keyHead * keyDim + $0]) },
                    key: (0..<keyDim).map { Double(ks[keyHead * keyDim + $0]) },
                    value: (0..<valueDim).map { Double(vs[head * valueDim + $0]) },
                    decay: QwenReferenceOps.decay(
                        a: Double(asx[head]), logA: Double(logARounded[head]),
                        dtBias: Double(dtRounded[head])),
                    beta: QwenReferenceOps.sigmoid(Double(bs[head])),
                    state: &reference[head], eps: Double(eps))
            }
        }

        let got = state.contents().bindMemory(
            to: Float.self, capacity: valueHeads * keyDim * valueDim)
        var worst = 0.0
        for head in 0..<valueHeads {
            for i in 0..<keyDim {
                for j in 0..<valueDim {
                    let at = (head * keyDim + i) * valueDim + j
                    worst = max(worst, abs(Double(got[at]) - reference[head][i][j]))
                }
            }
        }
        #expect(worst < 2e-5, "the carried state diverges by \(worst)")
    }
}
