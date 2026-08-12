import Foundation
import HydraCore
import HydraFormat
import HydraMetal
import Metal

/// Paired comparisons on the model actually installed.
///
/// A discipline taken from TurboFieldfare: control and candidate **alternate** instead of
/// running one after the other, because thermal state, the page cache and GPU frequency all
/// drift during a measurement. We report the median, and a candidate becomes the default
/// only if it wins reproducibly **without changing the outputs**.
enum Bench {

    static func median(_ values: [Double]) -> Double {
        let sorted = values.sorted()
        guard !sorted.isEmpty else { return 0 }
        return sorted.count % 2 == 1
            ? sorted[sorted.count / 2]
            : (sorted[sorted.count / 2 - 1] + sorted[sorted.count / 2]) / 2
    }

    static func run(config: GptOssConfig, root: URL) throws {
        let context = try MetalContext()
        let device = context.device
        let kernels = MXFP4Kernels(context: context)
        let blob = config.expertBlobLayout

        try benchmarkIO(config: config, root: root, device: device)
        try benchmarkKernels(
            config: config, root: root, device: device, kernels: kernels, blob: blob)
    }

    // MARK: - I/O

    private static func benchmarkIO(
        config: GptOssConfig, root: URL, device: MTLDevice
    ) throws {
        print("EXPERT READS")
        print("  Each round opens a fresh cache on a layer never yet read, and bypasses")
        print("  the page cache: without that we measure RAM, not the SSD.")
        print("  This is the trap TurboFieldfare documented, warm pages make any read")
        print("  strategy look excellent.\n")

        let experts = Array(0..<config.expertsPerToken)
        let bytes = Double(config.expertsPerToken * config.expertSlotBytes)

        for bypass in [true, false] {
            var serial: [Double] = []
            var parallel: [Double] = []
            // A distinct layer per measurement: an already-read layer would skew the rest.
            var layer = 0
            for _ in 0..<5 {
                let a = ExpertSlotCache(
root: root, model: config, slotsPerLayer: config.expertsPerToken,
                    device: device, bypassPageCache: bypass)
                var start = Date()
                for id in experts { _ = try a.expert(layer: layer % config.layerCount, expert: id) }
                serial.append(Date().timeIntervalSince(start))
                layer += 1

                let b = ExpertSlotCache(
root: root, model: config, slotsPerLayer: config.expertsPerToken,
                    device: device, bypassPageCache: bypass)
                start = Date()
                try b.load(layer: layer % config.layerCount, experts: experts)
                parallel.append(Date().timeIntervalSince(start))
                layer += 1
            }

            let s = median(serial), p = median(parallel)
            print("  \(bypass ? "F_NOCACHE (cold, what a real miss costs)" : "page cache allowed")")
            print(String(format: "    serial    %6.1f ms   %.2f GB/s", s * 1000, bytes / s / 1e9))
            print(String(format: "    parallel  %6.1f ms   %.2f GB/s   ×%.2f",
                         p * 1000, bytes / p / 1e9, s / p))
        }
    }

    // MARK: - Kernels

    private static func benchmarkKernels(
        config: GptOssConfig, root: URL, device: MTLDevice,
        kernels: MXFP4Kernels, blob: ExpertBlobLayout
    ) throws {
        print("\nMXFP4 GEMV, three variants, on a real expert")

        let cache = ExpertSlotCache(
root: root, model: config, slotsPerLayer: config.expertsPerToken, device: device)
        let (slot, _) = try cache.expert(layer: 0, expert: 0)

        let rows = 2 * config.intermediateSize
        let cols = config.hiddenSize
        let input = (0..<cols).map { Float(sin(Double($0) * 0.01)) }

        guard let xBuffer = device.makeBuffer(length: cols * 4, options: .storageModeShared)
        else { return }
        input.withUnsafeBytes {
            xBuffer.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count)
        }

        let variants = ["mxfp4_gemv", "mxfp4_gemv_vectorized", "mxfp4_gemv_simd", "mxfp4_gemv_tiled"]
        var outputs: [String: MTLBuffer] = [:]
        for name in variants {
            guard let buffer = device.makeBuffer(length: rows * 4, options: .storageModeShared)
            else { return }
            outputs[name] = buffer
        }

        // A CPU-GPU round trip costs a few hundred microseconds, the same order as the
        // kernel itself. Timing one pass per command buffer therefore amounts to timing the
        // synchronization, which makes every variant look identical. We encode `iterations`
        // passes into a single buffer and divide: the latency is then amortized and the gap
        // between variants becomes visible again.
        //
        let iterations = 50
        func time(_ function: String) throws -> Double {
            guard let commandBuffer = context(kernels).commandQueue.makeCommandBuffer()
            else { return 0 }
            for _ in 0..<iterations {
                try kernels.gemv(
                    function: function,
                    blocks: slot, blocksOffset: blob.gateUpBlocks.offset,
                    scales: slot, scalesOffset: blob.gateUpScales.offset,
                    bias: slot, biasOffset: blob.gateUpBias.offset,
                    x: xBuffer, xOffset: 0, output: outputs[function]!, outputOffset: 0,
                    rows: rows, cols: cols, in: commandBuffer)
            }
            let start = Date()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            return Date().timeIntervalSince(start) / Double(iterations)
        }

        // The cost of an empty round trip, to place what the previous measurement included.
        let emptyStart = Date()
        for _ in 0..<20 {
            guard let empty = context(kernels).commandQueue.makeCommandBuffer() else { break }
            empty.commit()
            empty.waitUntilCompleted()
        }
        print(String(format: "  empty CPU-GPU round-trip latency: %.0f µs",
                     Date().timeIntervalSince(emptyStart) / 20 * 1e6))

        // Warm-up: the first run pays for building the pipeline.
        for name in variants { _ = try time(name) }

        var samples: [String: [Double]] = [:]
        for _ in 0..<7 {
            for name in variants { samples[name, default: []].append(try time(name)) }
        }

        // The correctness reference: the bit-exact CPU decoder, summed in double.
        let reference = try cpuReference(
            slot: slot, blob: blob, input: input, rows: rows, cols: cols)

        let weightBytes = Double(rows * (cols / 32) * 17)  // 16 packed bytes + 1 scale
        print("  " + pad("variant", 26) + pad("ms", 9) + pad("GB/s", 10) + "deviation / CPU")
        var best = (name: "", time: Double.greatestFiniteMagnitude)
        for name in variants {
            let elapsed = median(samples[name] ?? [])
            let error = deviation(outputs[name]!, reference: reference, rows: rows)
            print("  " + pad(name, 26)
                + pad(String(format: "%.2f", elapsed * 1000), 9)
                + pad(String(format: "%.1f", weightBytes / elapsed / 1e9), 10)
                + String(format: "%.2e", error))
            if error < 1e-4, elapsed < best.time { best = (name, elapsed) }
        }

        let referenceTime = median(samples["mxfp4_gemv"] ?? [])
        if !best.name.isEmpty {
            print(String(format: "\n  best correct candidate: %@ (×%.2f)",
                         best.name as NSString, referenceTime / best.time))
        }

        // The kernel is still far from memory bandwidth: that is the next piece of work,
        // not a conclusion.
        let bandwidth = context(kernels).measureMemoryBandwidth()
        print(String(format: "  the machine's memory bandwidth: %.0f GB/s", bandwidth / 1e9))
        print(String(format: "  the best kernel uses %.0f %% of it",
                     weightBytes / best.time / bandwidth * 100))

        // ------------------------------------------------------- Extrapolation
        let perExpert = best.time * 1.5  // gate_up then down, half as wide
        let moePerToken = perExpert * Double(config.expertsPerToken * config.layerCount)
        print(String(format: "\n  MoE alone, extrapolated: %.0f ms/token → %.1f tok/s",
                     moePerToken * 1000, 1 / moePerToken))
        print("  (attention, LM head and I/O excluded, an optimistic upper bound)")
    }

    private static func context(_ kernels: MXFP4Kernels) -> MetalContext { kernels.context }

    /// The reference product, computed in double precision from the slot's bytes.
    private static func cpuReference(
        slot: MTLBuffer, blob: ExpertBlobLayout, input: [Float], rows: Int, cols: Int
    ) throws -> [Double] {
        let blocksPerRow = cols / MXFP4Layout.blockSize
        let bytesPerRow = blocksPerRow * MXFP4Layout.packedBytesPerBlock
        let base = slot.contents()
        var out = [Double](repeating: 0, count: rows)

        for row in 0..<rows {
            let packed = Data(
                bytesNoCopy: base.advanced(by: blob.gateUpBlocks.offset + row * bytesPerRow),
                count: bytesPerRow, deallocator: .none)
            let scales = Data(
                bytesNoCopy: base.advanced(by: blob.gateUpScales.offset + row * blocksPerRow),
                count: blocksPerRow, deallocator: .none)
            let weights = try MXFP4.decode(packed: packed, scales: scales)

            var sum = 0.0
            for c in 0..<cols { sum += Double(weights[c]) * Double(input[c]) }
            let biasBits = base.advanced(by: blob.gateUpBias.offset + row * 2)
                .loadUnaligned(as: UInt16.self)
            out[row] = sum + Double(BF16.toFloat(UInt16(littleEndian: biasBits)))
        }
        return out
    }

    /// Relative deviation against the **vector's** magnitude, not each component's.
    ///
    /// Relating it to the component gives alarming figures on outputs near zero, where the
    /// slightest catastrophic cancellation dominates, an artifact of the measurement, not a
    /// defect in the kernel. The vector's magnitude is the quantity that matters for the rest
    /// of the computation.
    private static func deviation(_ buffer: MTLBuffer, reference: [Double], rows: Int) -> Double {
        let values = UnsafeBufferPointer(
            start: buffer.contents().bindMemory(to: Float.self, capacity: rows), count: rows)
        var worst = 0.0
        var scale = 0.0
        for i in 0..<rows { scale = max(scale, abs(reference[i])) }
        for i in 0..<rows {
            worst = max(worst, abs(Double(values[i]) - reference[i]) / max(scale, 1e-6))
        }
        return worst
    }
}
