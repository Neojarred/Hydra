import Foundation
import HydraCore
import HydraFormat
import HydraReference
import Metal
import Testing

@testable import HydraMetal

/// The gated RMS norm, per head, against the CPU reference.
@Suite("Qwen gated norm on GPU")
struct QwenGatedNormTests {

    private let heads = 5
    private let dim = 12

    private func values(_ count: Int, seed: UInt64) -> [Float] {
        var state = seed | 1
        return (0..<count).map { _ in
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float(Int(state >> 33) % 2000 - 1000) / 1000
        }
    }

    private func floats(_ context: MetalContext, _ v: [Float]) -> MTLBuffer? {
        v.withUnsafeBytes {
            context.device.makeBuffer(
                bytes: $0.baseAddress!, length: max($0.count, 4), options: .storageModeShared)
        }
    }

    /// Every head is normalized on its own and gated by its own slice.
    ///
    /// The gate values are made distinctive per head, so a kernel that gates every head with
    /// the first slice disagrees on heads one and above rather than merely being inaccurate.
    @Test("Each head is normalized alone and gated by its own slice")
    func matchesReference() throws {
        let context = try MetalContext()
        let encoder = ForwardEncoder(context: context)

        let inputs = values(heads * dim, seed: 0x51)
        let weightFloats = values(dim, seed: 0x61)
        var gates = values(heads * dim, seed: 0x71)
        // Separate the heads: head h's gate sits in a band around h, so borrowing another
        // head's slice changes the sign of the result rather than a digit.
        for head in 0..<heads {
            for i in 0..<dim {
                gates[head * dim + i] += Float(head) * 2 - Float(heads)
            }
        }

        let weightBits = weightFloats.map { BF16.fromFloat($0) }
        guard let input = floats(context, inputs), let gate = floats(context, gates),
            let weight = weightBits.withUnsafeBytes({
                context.device.makeBuffer(
                    bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
            }),
            let output = context.device.makeBuffer(
                length: heads * dim * 4, options: .storageModeShared),
            let command = context.commandQueue.makeCommandBuffer()
        else { return }

        try encoder.qwenGatedRMSNormHeads(
            input: input, weight: weight, weightOffset: 0, gate: gate, output: output,
            heads: heads, dim: dim, eps: 1e-6, in: command)
        context.commit(command)
        try context.wait(command)

        let got = output.contents().bindMemory(to: Float.self, capacity: heads * dim)
        // The reference reads the same BF16 the kernel does, so the deviation measured is the
        // kernel's and not the weight's rounding.
        let weight64 = weightBits.map { Double(BF16.toFloat($0)) }
        for head in 0..<heads {
            let slice = (0..<dim).map { Double(inputs[head * dim + $0]) }
            let gateSlice = (0..<dim).map { Double(gates[head * dim + $0]) }
            let expected = QwenReferenceOps.gatedRMSNorm(
                slice, weight: weight64, gate: gateSlice, eps: 1e-6)
            for i in 0..<dim {
                #expect(
                    abs(Double(got[head * dim + i]) - expected[i]) < 1e-5,
                    "head \(head), component \(i)")
            }
        }
    }
}
