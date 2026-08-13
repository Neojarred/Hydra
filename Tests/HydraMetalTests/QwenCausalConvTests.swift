import Foundation
import HydraCore
import HydraFormat
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

    /// The learned tensors, in the precision the checkpoint stores them in.
    ///
    /// The kernel reads the convolution's weight and bias as BF16, because that is what they
    /// are in `resident.bin`. Building them as float32 here made this test agree with a kernel
    /// that could not read the real model, which is the shape of blindness worth naming: the
    /// fixture was more capable than the thing it stood for.
    /// - Parameter pad: junk values placed in front, so the tensor starts at a non-zero
    ///   offset. In `resident.bin` nothing starts at zero, and an encoder that ignores an
    ///   offset reads a neighbouring tensor: finite, wrong, and invisible to a fixture where
    ///   every buffer holds exactly one tensor at its start.
    private func bf16Buffer(
        _ context: MetalContext, _ v: [Float], pad: Int = 0
    ) -> (buffer: MTLBuffer, offset: Int, rounded: [Float])? {
        let bits = [UInt16](repeating: 0x7F7F, count: pad) + v.map { BF16.fromFloat($0) }
        let rounded = bits.dropFirst(pad).map { BF16.toFloat($0) }
        return bits.withUnsafeBytes {
            context.device.makeBuffer(
                bytes: $0.baseAddress!, length: max($0.count, 4), options: .storageModeShared)
        }.map { ($0, pad * 2, rounded) }
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
            let (weight, weightAt, weightRounded) = bf16Buffer(context, weightFlat, pad: 7),
            let (bias, biasAt, biasRounded) = bf16Buffer(context, biasFlat, pad: 3)
        else { return }
        // Zeroed, which is the left padding at the start of a sequence.
        memset(window.contents(), 0, window.length)

        let weight64 = (0..<convDim).map { c in
            (0..<kernel).map { Double(weightRounded[c * kernel + $0]) }
        }
        let bias64 = biasRounded.map(Double.init)
        var history: [[Double]] = []

        for t in 0..<tokens {
            guard let input = buffer(context, inputs[t]),
                let command = context.commandQueue.makeCommandBuffer()
            else { return }
            try encoder.qwenCausalConvStep(
                window: window, windowOffset: 0, input: input,
                weight: weight, weightOffset: weightAt,
                bias: bias, biasOffset: biasAt,
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
            let (weight, weightAt, weightRounded) = bf16Buffer(context, weightFlat, pad: 5),
            let input = buffer(context, inputFlat),
            let command = context.commandQueue.makeCommandBuffer()
        else { return }
        memset(window.contents(), 0, window.length)

        try encoder.qwenCausalConvStep(
            window: window, windowOffset: 0, input: input,
            weight: weight, weightOffset: weightAt, bias: nil,
            output: output, convDim: convDim, kernel: kernel, in: command)
        context.commit(command)
        try context.wait(command)

        let expected = QwenReferenceOps.causalDepthwiseConv(
            input: inputFlat.map(Double.init), history: [],
            weight: (0..<convDim).map { c in
                (0..<kernel).map { Double(weightRounded[c * kernel + $0]) }
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
            let (weight, _, _) = bf16Buffer(context, weightFlat)
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
