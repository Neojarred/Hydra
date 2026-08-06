import Foundation
import HydraCore
import HydraFormat
import HydraMetal
import Metal

/// A complete forward pass on an installed model.
///
/// Without a tokenizer we still produce only identifiers — but that is enough to confirm
/// the chain holds up end to end and to put a number on real throughput. Text comes with
/// `o200k_harmony`.
enum Generate {

    static func run(
        config: GptOssConfig, root: URL, contextLength: Int,
        tokenCount: Int, slotsPerLayer: Int?
    ) throws {
        let context = try MetalContext()
        let device = context.device

        print("GPU        \(device.name), family \(context.gpuFamily)")
        print("ceiling    \(gib(Int(device.recommendedMaxWorkingSetSize)))")

        let profile = context.hardwareProfile(memoryBandwidth: 94e9, diskBandwidth: 5.5e9)
        let policy: ExpertCachePolicy =
            slotsPerLayer.map { .slotsPerLayer($0) } ?? .minimal
        let budget = MemoryBudget(
            config: config, hardware: profile, contextLength: contextLength, policy: policy)

        print("\nBUDGET")
        for line in budget.breakdown {
            print("  " + pad(line.label, 40) + gib(line.bytes))
        }
        print("  " + pad("EXPECTED FOOTPRINT", 40) + gib(budget.totalFootprintBytes))
        print(String(format: "  i.e. %.0f %% of the installed model (%@)",
                     budget.residentFractionOfCheckpoint * 100, gib(config.installedBytes)))

        let baseline = MemoryFootprint.current()
        let mapping = try ModelMapping(root: root, config: config, device: device)
        let expertCache = ExpertSlotCache(
            root: root, config: config,
            slotsPerLayer: budget.expertSlotsPerLayer, device: device)
        let runner = try ModelRunner(
            config: config, context: context, mapping: mapping,
            expertCache: expertCache, contextLength: contextLength)

        print("\nLOADING")
        print("  \(budget.expertSlotsPerLayer) slots/layer of \(config.expertCount)")
        print("  reserved by the runtime: \(gib(runner.reservedBytes))")
        print("  footprint after loading: \(mib(MemoryFootprint.current())) "
            + "(before: \(mib(baseline)))")

        // An arbitrary seed: without a tokenizer only shape and throughput matter.
        let prompt = [1, 2, 3, 4]
        print("\nGENERATION — \(tokenCount) tokens, seed of \(prompt.count) identifiers")

        let peak = MemoryFootprint.Peak()
        var perToken: [Double] = []
        var attentionTotal = 0.0, ioTotal = 0.0, mixtureTotal = 0.0, headTotal = 0.0
        let started = Date()

        let produced = try runner.generate(prompt: prompt, count: tokenCount) { token, timings in
            peak.sample()
            perToken.append(timings.total)
            attentionTotal += timings.attentionAndRouter
            ioTotal += timings.expertIO
            mixtureTotal += timings.mixture
            headTotal += timings.head
            FileHandle.standardError.write(Data(
                String(format: "\r  token %3d → %6d   %5.0f ms   %@   ",
                       perToken.count, token, timings.total * 1000,
                       mib(MemoryFootprint.current())).utf8))
        }

        let elapsed = Date().timeIntervalSince(started)
        let stats = expertCache.statisticsSnapshot()
        print("\n")
        print("RESULT")
        print("  \(produced.count) tokens in \(String(format: "%.1f", elapsed)) s")
        print(String(format: "  throughput: %.2f tok/s  (median %.0f ms/token)",
                     Double(produced.count) / elapsed, Bench.median(perToken) * 1000))
        print("  first identifiers: \(produced.prefix(12).map(String.init).joined(separator: " "))")

        let steps = max(perToken.count, 1)
        let decodeTime = attentionTotal + ioTotal + mixtureTotal + headTotal
        print("\nTIME BREAKDOWN PER TOKEN")
        print(String(format: "  cb1  attention + router    %6.1f ms   %4.0f %%",
                     attentionTotal / Double(steps) * 1000,
                     attentionTotal / max(decodeTime, 1e-9) * 100))
        print(String(format: "  I/O  reading the experts   %6.1f ms   %4.0f %%",
                     ioTotal / Double(steps) * 1000, ioTotal / max(decodeTime, 1e-9) * 100))
        print(String(format: "  cb2  mixture of experts    %6.1f ms   %4.0f %%",
                     mixtureTotal / Double(steps) * 1000, mixtureTotal / max(decodeTime, 1e-9) * 100))
        print(String(format: "  LM head                    %6.1f ms   %4.0f %%",
                     headTotal / Double(steps) * 1000, headTotal / max(decodeTime, 1e-9) * 100))

        print("\nEXPERT CACHE")
        print(String(format: "  %d hits, %d misses — hit rate %.1f %%",
                     stats.hits, stats.misses, stats.hitRate * 100))
        print("  read from the SSD: \(gib(stats.bytesRead))")

        print("\nMEMORY")
        print("  committed by the process     : \(mib(peak.value))  (phys_footprint)")
        print("  resident                     : \(mib(peak.residentValue))  (resident_size)")
        print("     Both counters cover anonymous memory: expert slots, scratch, KV")
        print("     cache, logits. They do match the reservation announced above.")
        print("")
        print("  mapped weights, on top       : \(gib(mapping.mappedByteCount))")
        print("     resident.bin and embed.bin are file-backed. The kernel charges them")
        print("     to neither counter and can reclaim them under pressure — but they do")
        print("     occupy RAM as long as there is any. That is elasticity, not free")
        print("     memory: announcing it as zero would be dishonest.")
        print("")
        print("  installed model              : \(gib(config.installedBytes))")
        print(String(format: "  upper bound (committed + mapped): %@ i.e. %.0f %% of the installed",
                     gib(peak.value + mapping.mappedByteCount),
                     Double(peak.value + mapping.mappedByteCount)
                        / Double(config.installedBytes) * 100))
    }
}
