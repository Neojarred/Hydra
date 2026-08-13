import Foundation
import HydraCore
import HydraReference
import Metal
import Testing

@testable import HydraMetal

/// The output gate on Qwen's full-attention layers.
///
/// Two details that cannot be guessed, both from D-027. The gate is packed **per head** inside
/// `q_proj`'s output, not appended after every query; and it multiplies the attention output
/// through a sigmoid, not the query going in.
@Suite("Qwen attention gate")
struct QwenAttentionGateTests {

    private let heads = 4
    private let headDim = 6

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

    /// The split is per head, and the test data makes the wrong split detectable.
    ///
    /// Every value carries its head and position, so a kernel that halves the tensor instead of
    /// each head's slice produces values belonging to another head rather than merely different
    /// numbers.
    @Test("The query and its gate are split within each head")
    func splitIsPerHead() throws {
        let context = try MetalContext()
        let encoder = ForwardEncoder(context: context)

        // combined[head][i]      = head + i/100      (the query)
        // combined[head][dim+i]  = -(head + i/100)   (its gate)
        var combined = [Float](repeating: 0, count: heads * headDim * 2)
        for head in 0..<heads {
            for i in 0..<headDim {
                let tag = Float(head) + Float(i) / 100
                combined[head * headDim * 2 + i] = tag
                combined[head * headDim * 2 + headDim + i] = -tag
            }
        }

        guard let source = buffer(context, combined),
            let query = context.device.makeBuffer(
                length: heads * headDim * 4, options: .storageModeShared),
            let gate = context.device.makeBuffer(
                length: heads * headDim * 4, options: .storageModeShared),
            let command = context.commandQueue.makeCommandBuffer()
        else { return }

        try encoder.qwenSplitQueryGate(
            combined: source, query: query, gate: gate,
            heads: heads, headDim: headDim, in: command)
        context.commit(command)
        try context.wait(command)

        let expected = QwenReferenceOps.splitQueryAndGate(
            combined.map(Double.init), heads: heads, headDim: headDim)
        let gotQuery = query.contents().bindMemory(to: Float.self, capacity: heads * headDim)
        let gotGate = gate.contents().bindMemory(to: Float.self, capacity: heads * headDim)

        for index in 0..<(heads * headDim) {
            #expect(
                abs(Double(gotQuery[index]) - expected.query[index]) < 1e-6,
                "query \(index): head \(index / headDim) got another head's slice")
            #expect(
                abs(Double(gotGate[index]) - expected.gate[index]) < 1e-6,
                "gate \(index)")
            // The tagging makes the relationship explicit: a gate is the negation of its query.
            #expect(abs(Double(gotQuery[index]) + Double(gotGate[index])) < 1e-6)
        }
    }

    @Test("The gate multiplies the attention output through a sigmoid")
    func gateMultipliesOutput() throws {
        let context = try MetalContext()
        let encoder = ForwardEncoder(context: context)
        let count = heads * headDim
        let outputs = values(count, seed: 0x31)
        let gates = values(count, seed: 0x41)

        guard let output = buffer(context, outputs), let gate = buffer(context, gates),
            let command = context.commandQueue.makeCommandBuffer()
        else { return }
        try encoder.qwenApplyOutputGate(
            output: output, gate: gate, count: count, in: command)
        context.commit(command)
        try context.wait(command)

        let expected = QwenReferenceOps.applyOutputGate(
            outputs.map(Double.init), gate: gates.map(Double.init))
        let got = output.contents().bindMemory(to: Float.self, capacity: count)
        for index in 0..<count {
            #expect(abs(Double(got[index]) - expected[index]) < 1e-6, "component \(index)")
        }
    }

    /// For text, the interleaved mRoPE is the ordinary rotary.
    ///
    /// `apply_interleaved_mrope` overwrites the temporal frequencies with the height and width
    /// ones at strides of three. Every one of those components is the same number for a text
    /// token, so the rewrite copies a value onto itself and the result is the rotary this
    /// project already implements. That is worth asserting rather than believing, because it is
    /// the reason no new rotary kernel is needed until images arrive.
    @Test("Interleaved mRoPE degenerates to ordinary RoPE for text positions")
    func mropeIsOrdinaryForText() {
        let section = [11, 11, 10]
        let pairs = section.reduce(0, +) * 2      // 64 rotary components, a quarter of 256
        #expect(pairs == 64, "partial_rotary_factor 0.25 of head_dim 256")

        // The three position components a text token carries are identical.
        let position = 137.0
        let components = [position, position, position]

        // The reference's rewrite: start from temporal, then take height at indices 1, 4, 7 …
        // and width at 2, 5, 8 …, each bounded by its section times three.
        var frequencies = [Double](repeating: components[0], count: pairs)
        for (dim, offset) in [(1, 1), (2, 2)] {
            var index = offset
            while index < section[dim] * 3 {
                frequencies[index] = components[dim]
                index += 3
            }
        }
        #expect(
            frequencies.allSatisfy { $0 == position },
            "with equal components the interleave is a copy onto itself")
    }
}
