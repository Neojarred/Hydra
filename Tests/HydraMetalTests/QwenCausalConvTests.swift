import Foundation
import HydraCore
import HydraReference
import Metal
import Testing

@testable import HydraMetal

/// The depthwise causal convolution on the GPU, against the CPU reference.
///
/// Fed one token at a time with the window carried, which is how decoding runs it. The window
/// is the part that goes wrong quietly: a kernel that computes the right sum and forgets to
/// advance it agrees on token zero and drifts from token one, and only the first `kernel - 1`
/// tokens of a turn are affected, which is small enough to read as the model being odd.
@Suite("Qwen causal convolution on GPU")
struct QwenCausalConvTests {

    private let convDim = 24
    private let kernel = 4

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

    @Test("The convolution matches the reference over a sequence, window carried")
    func matchesReference() throws {
        let context = try MetalContext()
        let encoder = ForwardEncoder(context: context)
        let tokens = 7

        let weightFlat = values(convDim * kernel, seed: 0xC1)
        let biasFlat = values(convDim, seed: 0xB1)
        let inputs = (0..<tokens).map { values(convDim, seed: 0xD1 + UInt64($0)) }

        guard let window = context.device.makeBuffer(
                length: (kernel - 1) * convDim * 4, options: .storageModeShared),
            let output = context.device.makeBuffer(
                length: convDim * 4, options: .storageModeShared),
            let weight = buffer(context, weightFlat), let bias = buffer(context, biasFlat)
        else { return }
        // Zeroed, which is the left padding at the start of a sequence.
        memset(window.contents(), 0, window.length)

        let weight64 = (0..<convDim).map { c in
            (0..<kernel).map { Double(weightFlat[c * kernel + $0]) }
        }
        let bias64 = biasFlat.map(Double.init)
        var history: [[Double]] = []

        for t in 0..<tokens {
            guard let input = buffer(context, inputs[t]),
                let command = context.commandQueue.makeCommandBuffer()
            else { return }
            try encoder.qwenCausalConvStep(
                window: window, windowOffset: 0, input: input, weight: weight, bias: bias,
                output: output, convDim: convDim, kernel: kernel, in: command)
            context.commit(command)
            try context.wait(command)

            let expected = QwenReferenceOps.causalDepthwiseConv(
                input: inputs[t].map(Double.init), history: history,
                weight: weight64, bias: bias64)
            let got = output.contents().bindMemory(to: Float.self, capacity: convDim)
            for c in 0..<convDim {
                #expect(
                    abs(Double(got[c]) - expected[c]) < 1e-5,
                    "token \(t), channel \(c): the window diverges")
            }

            history.append(inputs[t].map(Double.init))
            if history.count > kernel - 1 { history.removeFirst() }
        }
    }

    /// Without a bias tensor the sum starts at zero, and the buffer bound in its place is
    /// never read.
    @Test("The convolution is correct with no bias")
    func noBias() throws {
        let context = try MetalContext()
        let encoder = ForwardEncoder(context: context)
        let weightFlat = values(convDim * kernel, seed: 0xC2)
        let inputFlat = values(convDim, seed: 0xD2)

        guard let window = context.device.makeBuffer(
                length: (kernel - 1) * convDim * 4, options: .storageModeShared),
            let output = context.device.makeBuffer(
                length: convDim * 4, options: .storageModeShared),
            let weight = buffer(context, weightFlat), let input = buffer(context, inputFlat),
            let command = context.commandQueue.makeCommandBuffer()
        else { return }
        memset(window.contents(), 0, window.length)

        try encoder.qwenCausalConvStep(
            window: window, windowOffset: 0, input: input, weight: weight, bias: nil,
            output: output, convDim: convDim, kernel: kernel, in: command)
        context.commit(command)
        try context.wait(command)

        let expected = QwenReferenceOps.causalDepthwiseConv(
            input: inputFlat.map(Double.init), history: [],
            weight: (0..<convDim).map { c in
                (0..<kernel).map { Double(weightFlat[c * kernel + $0]) }
            },
            bias: nil)
        let got = output.contents().bindMemory(to: Float.self, capacity: convDim)
        for c in 0..<convDim {
            #expect(abs(Double(got[c]) - expected[c]) < 1e-5, "channel \(c)")
        }
    }

    /// The window holds the last `kernel - 1` inputs, oldest first.
    ///
    /// Asserted on the buffer rather than inferred from the outputs, because a shift in the
    /// wrong direction still produces plausible values and would show up only as the model
    /// being slightly wrong at the start of every turn.
    @Test("The window holds the previous inputs, oldest first")
    func windowLayout() throws {
        let context = try MetalContext()
        let encoder = ForwardEncoder(context: context)
        let weightFlat = [Float](repeating: 0, count: convDim * kernel)
        let inputs = (0..<5).map { values(convDim, seed: 0xE1 + UInt64($0)) }

        guard let window = context.device.makeBuffer(
                length: (kernel - 1) * convDim * 4, options: .storageModeShared),
            let output = context.device.makeBuffer(
                length: convDim * 4, options: .storageModeShared),
            let weight = buffer(context, weightFlat)
        else { return }
        memset(window.contents(), 0, window.length)

        for token in inputs {
            guard let input = buffer(context, token),
                let command = context.commandQueue.makeCommandBuffer()
            else { return }
            try encoder.qwenCausalConvStep(
                window: window, windowOffset: 0, input: input, weight: weight, bias: nil,
                output: output, convDim: convDim, kernel: kernel, in: command)
            context.commit(command)
            try context.wait(command)
        }

        let held = window.contents().bindMemory(
            to: Float.self, capacity: (kernel - 1) * convDim)
        // After five tokens the window is inputs 2, 3 and 4, oldest first.
        for (row, source) in [2, 3, 4].enumerated() {
            for c in 0..<convDim {
                #expect(
                    held[row * convDim + c] == inputs[source][c],
                    "row \(row) should hold token \(source)")
            }
        }
    }
}
