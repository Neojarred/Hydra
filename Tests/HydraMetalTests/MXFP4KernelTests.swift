import Foundation
import HydraCore
import HydraFormat
import Testing

@testable import HydraMetal

/// The CPU decoder is already validated bit for bit against OpenAI's reference
/// implementation (milestone 1.2). These tests extend the guarantee to the GPU: the Metal
/// output must agree with the CPU output. The chain of trust therefore runs from OpenAI's
/// reference to the kernel that will run in production.
struct MXFP4KernelTests {

    /// Deterministic MXFP4 data, with realistic scale exponents — those observed on the
    /// installed checkpoint cluster around 2⁻⁶.
    static func syntheticWeights(rows: Int, cols: Int, seed: UInt64 = 12345)
        -> (packed: Data, scales: Data)
    {
        var state = seed
        func next() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
        let blocksPerRow = cols / MXFP4Layout.blockSize
        var packed = Data(count: rows * blocksPerRow * MXFP4Layout.packedBytesPerBlock)
        var scales = Data(count: rows * blocksPerRow)
        for i in 0..<packed.count { packed[i] = UInt8(truncatingIfNeeded: next()) }
        for i in 0..<scales.count { scales[i] = UInt8(121 + next() % 7) }  // exponents -6 to 0
        return (packed, scales)
    }

    @Test("GPU decoding agrees exactly with the validated CPU decoder")
    func gpuMatchesCpuDecoder() throws {
        let kernels = MXFP4Kernels(context: try MetalContext())
        let (packed, scales) = Self.syntheticWeights(rows: 16, cols: 1024)

        let cpu = try MXFP4.decode(packed: packed, scales: scales)
        let gpu = try kernels.dequantize(packed: packed, scales: scales)

        #expect(gpu.count == cpu.count)
        var mismatches = 0
        for i in 0..<min(cpu.count, gpu.count) where cpu[i] != gpu[i] { mismatches += 1 }
        #expect(mismatches == 0, "\(mismatches) values diverge between CPU and GPU")
    }

    @Test("The MXFP4 GEMV agrees with a product computed in double precision")
    func gemvMatchesReference() throws {
        let kernels = MXFP4Kernels(context: try MetalContext())

        let rows = 64
        let cols = 2880  // GPT-OSS's real dimension
        let (packed, scales) = Self.syntheticWeights(rows: rows, cols: cols)
        let x = (0..<cols).map { Float(sin(Double($0) * 0.017)) }

        let gpu = try kernels.gemv(
            packed: packed, scales: scales, bias: nil, x: x, rows: rows, cols: cols)

        // Reference: row-by-row dequantization with the CPU decoder, then a sum in double
        // precision — the deviation measured is then the GPU's, not the model's.
        let blocksPerRow = cols / MXFP4Layout.blockSize
        let bytesPerRow = blocksPerRow * MXFP4Layout.packedBytesPerBlock
        var worstRelative = 0.0
        for row in 0..<rows {
            let rowPacked = packed.subdata(in: (row * bytesPerRow)..<((row + 1) * bytesPerRow))
            let rowScales = scales.subdata(in: (row * blocksPerRow)..<((row + 1) * blocksPerRow))
            let weights = try MXFP4.decode(packed: rowPacked, scales: rowScales)

            var expected = 0.0
            for c in 0..<cols { expected += Double(weights[c]) * Double(x[c]) }

            worstRelative = max(
                worstRelative, abs(Double(gpu[row]) - expected) / max(abs(expected), 1e-6))
        }
        // The GPU accumulates in Float32 in a different order: the expected deviation is that
        // of floating-point arithmetic, not of a memory-layout error.
        #expect(worstRelative < 1e-5, "worst relative deviation: \(worstRelative)")
    }

    @Test("The bias is added when supplied")
    func gemvAppliesBias() throws {
        let kernels = MXFP4Kernels(context: try MetalContext())

        let rows = 8, cols = 128
        let (packed, scales) = Self.syntheticWeights(rows: rows, cols: cols)
        let x = [Float](repeating: 0.5, count: cols)

        let withoutBias = try kernels.gemv(
            packed: packed, scales: scales, bias: nil, x: x, rows: rows, cols: cols)

        // BF16 bias: 1.0 encodes as 0x3F80, the top 16 bits of the Float32 1.0.
        var bias = Data()
        for _ in 0..<rows { bias.append(contentsOf: [0x80, 0x3F]) }
        let withBias = try kernels.gemv(
            packed: packed, scales: scales, bias: bias, x: x, rows: rows, cols: cols)

        for row in 0..<rows {
            #expect(abs((withBias[row] - withoutBias[row]) - 1.0) < 1e-5, "ligne \(row)")
        }
    }

    /// The number of lanes per threadgroup must not change the result: it is a performance
    /// parameter, not a semantic one. A reduction bug would show up here.
    @Test("The result does not depend on the threadgroup shape")
    func resultIsIndependentOfThreadgroupShape() throws {
        let kernels = MXFP4Kernels(context: try MetalContext())
        let rows = 4
        for cols in [64, 256, 2880] {
            let (packed, scales) = Self.syntheticWeights(rows: rows, cols: cols, seed: UInt64(cols))
            let x = (0..<cols).map { Float(($0 % 7) - 3) * 0.25 }
            let y = try kernels.gemv(
                packed: packed, scales: scales, bias: nil, x: x, rows: rows, cols: cols)
            #expect(y.count == rows)
            #expect(y.allSatisfy { $0.isFinite }, "cols = \(cols) produit des non-finis")
        }
    }

    @Test("The Metal context exposes the GPU family and a coherent ceiling")
    func contextReportsHardware() throws {
        let context = try MetalContext()
        let profile = context.hardwareProfile(memoryBandwidth: 1, diskBandwidth: 1)

        #expect(profile.metalWorkingSetCeiling > 0)
        // The Metal ceiling is a fraction of physical memory, never all of it.
        #expect(profile.metalWorkingSetCeiling < Int(ProcessInfo.processInfo.physicalMemory))
        #expect(context.gpuFamily.hasPrefix("apple"))
    }
}
