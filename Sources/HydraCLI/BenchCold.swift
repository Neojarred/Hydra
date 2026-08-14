import Foundation
import HydraCore
import HydraFormat
import HydraMetal
import Metal

/// The four ways to get an expert blob in front of the GPU, measured **cold**.
///
/// M-056 measured warm and found mapping 31 % faster with no copy. Cold is the other half, and
/// it is where the inherited decision lives: `pread` at 3.97 tok/s against `mmap` at 0.50,
/// because a fault is serial and one page wide while `load(layer:experts:)` issues eight reads
/// at once and gets 5.5 GB/s instead of 3.0.
///
/// **Coldness is verified, not assumed.** `mincore` reports which pages of a mapping are
/// resident, so a layer is only used as a cold sample if almost none of it is. A benchmark that
/// hoped for a cold cache would silently become a second warm benchmark, which is exactly the
/// confound that has cost this project four retracted results.
///
/// Each measurement consumes its layer: reading it makes it warm. So every sample gets its own
/// file, and the schemes are assigned round-robin so none of them always gets the first one.
enum BenchCold {

    enum Scheme: Int, CaseIterable {
        case preadSerial, preadParallel, mapped, mappedWillNeed

        var name: String {
            switch self {
            case .preadSerial: return "pread, serial"
            case .preadParallel: return "pread, 8 at once"
            case .mapped: return "mapped, faulting"
            case .mappedWillNeed: return "mapped + WILLNEED"
            }
        }
    }

    /// The fraction of a mapping already in RAM, from the kernel rather than from a guess.
    private static func residentFraction(_ pointer: UnsafeMutableRawPointer, _ length: Int)
        -> Double
    {
        let pageSize = 16384
        let pages = (length + pageSize - 1) / pageSize
        var vector = [CChar](repeating: 0, count: pages)
        guard mincore(pointer, length, &vector) == 0 else { return -1 }
        let resident = vector.reduce(0) { $0 + (($1 & 1) != 0 ? 1 : 0) }
        return Double(resident) / Double(pages)
    }

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

    static func run(
        config: Qwen35MoeConfig, root: URL, blobs: Int = 32, samples: Int = 12
    ) throws {
        let context = try MetalContext()
        let encoder = ForwardEncoder(context: context)
        let layout = config.expertBlobLayout
        let stride = layout.strideBytes

        guard let input = context.device.makeBuffer(
                length: config.hiddenSize * 4, options: .storageModeShared),
            let output = context.device.makeBuffer(
                length: config.moeIntermediateSize * 4, options: .storageModeShared)
        else { return }
        memset(input.contents(), 0, input.length)

        // One slot per parallel reader, so the concurrent form has somewhere to put its bytes.
        let readers = 8
        var slots: [MTLBuffer] = []
        for _ in 0..<readers {
            guard let slot = context.device.makeBuffer(
                length: stride, options: .storageModeShared) else { return }
            slots.append(slot)
        }

        print("""
            \(config.name)
            blob \(stride / 1024) KiB, reading \(blobs) of them a sample = \
            \(blobs * stride / 1_048_576) MiB
            each sample takes a fresh layer file, and is used only if mincore says it is cold
            """)

        var results: [Scheme: [Double]] = [:]
        var skipped = 0
        var layer = 0

        for sample in 0..<samples {
            let scheme = Scheme.allCases[sample % Scheme.allCases.count]
            var opened = false

            while layer < config.layerCount && !opened {
                let path = root.appending(
                    path: String(format: "experts/layer_%02d.bin", layer)).path
                layer += 1
                guard FileManager.default.fileExists(atPath: path) else { continue }
                let size = (try FileManager.default.attributesOfItem(atPath: path)[.size]
                    as? Int) ?? 0
                let span = blobs * stride
                guard size >= span else { continue }

                let descriptor = open(path, O_RDONLY)
                guard descriptor >= 0 else { continue }
                let mappedLength = (size + 16383) / 16384 * 16384
                guard let mapped = mmap(nil, mappedLength, PROT_READ, MAP_PRIVATE, descriptor, 0),
                    mapped != MAP_FAILED
                else { close(descriptor); continue }

                // Only the span this sample will actually touch has to be cold.
                let fraction = residentFraction(mapped, span)
                if fraction > 0.10 {
                    skipped += 1
                    munmap(mapped, mappedLength)
                    close(descriptor)
                    continue
                }
                opened = true

                let start = Date()
                switch scheme {
                case .preadSerial:
                    for index in 0..<blobs {
                        let got = pread(
                            descriptor, slots[0].contents(), stride, off_t(index * stride))
                        precondition(got == stride, "short read")
                    }
                case .preadParallel:
                    // What `ExpertSlotCache.load` does: the reads run at once, which is the
                    // whole reason the inherited measurement favoured `pread`.
                    var batch = 0
                    while batch < blobs {
                        let count = min(readers, blobs - batch)
                        DispatchQueue.concurrentPerform(iterations: count) { i in
                            _ = pread(
                                descriptor, slots[i].contents(), stride,
                                off_t((batch + i) * stride))
                        }
                        batch += count
                    }
                case .mapped, .mappedWillNeed:
                    if scheme == .mappedWillNeed {
                        // Ask for the whole span at once, which is the batching the fault path
                        // otherwise lacks.
                        madvise(mapped, span, MADV_WILLNEED)
                    }
                    guard let buffer = context.device.makeBuffer(
                        bytesNoCopy: mapped, length: mappedLength,
                        options: .storageModeShared, deallocator: nil)
                    else { break }
                    for index in 0..<blobs {
                        guard let command = context.commandQueue.makeCommandBuffer() else { break }
                        try encodeRead(
                            encoder: encoder, blob: buffer, offset: index * stride,
                            layout: layout, config: config,
                            input: input, output: output, in: command)
                        context.commit(command)
                        try context.wait(command)
                    }
                }
                let seconds = Date().timeIntervalSince(start)
                results[scheme, default: []].append(seconds)

                print(String(
                    format: "  layer %2d  %-18s %6.0f ms  %5.2f GB/s   (was %.1f %% resident)",
                    layer - 1, (scheme.name as NSString).utf8String!, seconds * 1000,
                    Double(span) / seconds / 1e9, fraction * 100))

                munmap(mapped, mappedLength)
                close(descriptor)
            }

            if !opened {
                print("  ran out of cold layers after \(sample) samples")
                break
            }
        }

        print("\n  scheme               mean       GB/s   samples")
        let span = Double(blobs * stride)
        for scheme in Scheme.allCases {
            guard let times = results[scheme], !times.isEmpty else { continue }
            let mean = times.reduce(0, +) / Double(times.count)
            print(String(
                format: "  %-18s %7.0f ms  %5.2f     %d",
                (scheme.name as NSString).utf8String!, mean * 1000, span / mean / 1e9,
                times.count))
        }
        if skipped > 0 { print("\n  \(skipped) layer(s) skipped as already warm") }
    }
}
