import Foundation
import HydraCore
import HydraFormat
import HydraMetal
import Metal

/// `pread` into a preallocated slot against a mapped buffer the GPU reads in place.
///
/// `ExpertSlotCache` chose `pread` on an inherited measurement, 0.50 tok/s against 3.97, whose
/// stated reason is that demand paging gives no control over when reads happen or how many run
/// at once. That is a claim about **cold** reads: a fault is serial and one page wide, where
/// `load(layer:experts:)` issues eight `pread`s at once and gets 5.5 GB/s instead of 3.0.
///
/// M-055 found the machine no longer reads cold. 8.2 GiB of the model sits in the OS file cache,
/// so a `pread` copies RAM to RAM and a fault would find the page already resident. In that
/// regime the copy is pure loss, and the mapped form should win by exactly the memcpy.
///
/// This measures both, interleaved, in whatever state the cache happens to be in, and reports
/// the two separately rather than averaging a cold run into a warm one.
enum BenchMap {

    /// Forces the GPU to actually read a blob, rather than timing an unread mapping.
    ///
    /// A projection over the expert's gate matrix touches its words, its scales and its biases,
    /// which is what a decoding step does with it.
    private static func encodeRead(
        encoder: ForwardEncoder, blob: MTLBuffer, offset: Int,
        layout: MLXExpertBlobLayout, config: Qwen35MoeConfig,
        input: MTLBuffer, output: MTLBuffer, in command: MTLCommandBuffer
    ) throws {
        try encoder.mlxAffineProjection(
            words: blob, wordsOffset: offset + layout.gateWeights.offset,
            scales: blob, scalesOffset: offset + layout.gateScales.offset,
            biases: blob, biasesOffset: offset + layout.gateBiases.offset,
            input: input, inputOffset: 0, output: output, outputOffset: 0,
            rows: config.moeIntermediateSize, cols: config.hiddenSize,
            bits: config.quantBits, groupSize: config.groupSize, in: command)
    }

    static func run(config: Qwen35MoeConfig, root: URL, layer: Int = 0, experts: Int = 32) throws {
        let context = try MetalContext()
        let encoder = ForwardEncoder(context: context)
        let layout = config.expertBlobLayout
        let stride = layout.strideBytes
        let file = root.appending(path: String(format: "experts/layer_%02d.bin", layer))

        guard FileManager.default.fileExists(atPath: file.path) else {
            print("no expert file at \(file.path)")
            return
        }
        let size = (try FileManager.default.attributesOfItem(atPath: file.path)[.size] as? Int) ?? 0
        print("""
            expert blob \(stride) B = \(stride / 16384) pages of 16 KiB
            layer file  \(size) B, \(size / max(stride, 1)) experts
            reading \(experts) of them, both ways, interleaved
            """)

        guard let input = context.device.makeBuffer(
                length: config.hiddenSize * 4, options: .storageModeShared),
            let output = context.device.makeBuffer(
                length: config.moeIntermediateSize * 4, options: .storageModeShared)
        else { return }
        memset(input.contents(), 0, input.length)

        // --- The mapped form: one buffer over the whole file, experts at offsets ---
        //
        // `bytesNoCopy` needs a page-aligned base and length, which `mmap` gives and
        // `ExpertBlobLayout.pageAlignment` guarantees for every blob boundary inside it.
        let descriptor = open(file.path, O_RDONLY)
        guard descriptor >= 0 else { print("cannot open the layer file"); return }
        defer { close(descriptor) }
        let mappedLength = (size + 16383) / 16384 * 16384
        guard let mapped = mmap(nil, mappedLength, PROT_READ, MAP_PRIVATE, descriptor, 0),
            mapped != MAP_FAILED
        else { print("mmap failed"); return }
        defer { munmap(mapped, mappedLength) }

        guard let mappedBuffer = context.device.makeBuffer(
            bytesNoCopy: mapped, length: mappedLength, options: .storageModeShared,
            deallocator: nil)
        else {
            print("makeBuffer(bytesNoCopy:) refused the mapping")
            return
        }

        // --- The copying form: a preallocated slot, filled by pread, as the cache does ---
        guard let slot = context.device.makeBuffer(
            length: stride, options: .storageModeShared) else { return }

        func timedCopy() throws -> (read: Double, gpu: Double) {
            var readSeconds = 0.0
            var gpuSeconds = 0.0
            for index in 0..<experts {
                let start = Date()
                let got = pread(descriptor, slot.contents(), stride, off_t(index * stride))
                precondition(got == stride, "short read")
                readSeconds += Date().timeIntervalSince(start)

                let gpuStart = Date()
                guard let command = context.commandQueue.makeCommandBuffer() else { return (0, 0) }
                try encodeRead(
                    encoder: encoder, blob: slot, offset: 0, layout: layout, config: config,
                    input: input, output: output, in: command)
                context.commit(command)
                try context.wait(command)
                gpuSeconds += Date().timeIntervalSince(gpuStart)
            }
            return (readSeconds, gpuSeconds)
        }

        func timedMapped() throws -> (read: Double, gpu: Double) {
            var gpuSeconds = 0.0
            for index in 0..<experts {
                let gpuStart = Date()
                guard let command = context.commandQueue.makeCommandBuffer() else { return (0, 0) }
                try encodeRead(
                    encoder: encoder, blob: mappedBuffer, offset: index * stride,
                    layout: layout, config: config,
                    input: input, output: output, in: command)
                context.commit(command)
                try context.wait(command)
                gpuSeconds += Date().timeIntervalSince(gpuStart)
            }
            return (0, gpuSeconds)
        }

        // One untimed pass each, so neither scheme pays for the other's first touch.
        _ = try timedCopy()
        _ = try timedMapped()

        var copyTotals: [Double] = []
        var mappedTotals: [Double] = []
        print("\n  pair   pread+GPU        mapped GPU     (ms for \(experts) experts)")
        for pair in 1...6 {
            let c = try timedCopy()
            let m = try timedMapped()
            copyTotals.append((c.read + c.gpu) * 1000)
            mappedTotals.append(m.gpu * 1000)
            print(String(
                format: "  %4d   %6.1f (%4.1f read)   %6.1f",
                pair, (c.read + c.gpu) * 1000, c.read * 1000, m.gpu * 1000))
        }

        func mean(_ v: [Double]) -> Double { v.reduce(0, +) / Double(v.count) }
        let copyMean = mean(copyTotals)
        let mappedMean = mean(mappedTotals)
        print(String(
            format: """

                  copying   %.1f ms   (%.2f ms an expert)
                  mapped    %.1f ms   (%.2f ms an expert)
                  mapped is %+.1f %%
                """,
            copyMean, copyMean / Double(experts),
            mappedMean, mappedMean / Double(experts),
            (copyMean - mappedMean) / copyMean * 100))

        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / 4)
        _ = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        print(String(format: "\n  process footprint %.0f MiB",
                     Double(info.resident_size) / 1048576))
    }
}
