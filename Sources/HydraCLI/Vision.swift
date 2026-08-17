import Foundation
import HydraCore
import HydraFormat
import HydraMetal
import HydraVision
import Metal

/// What the installed vision tower actually contains, and what an image would cost.
///
/// The tower has been installed and unexecuted since D-021. Before any of it runs, this answers
/// the two questions that decide whether the rest is worth writing: are all 333 tensors present
/// with the shapes the config implies, and how many text tokens does a real photograph become.
enum VisionInspect {

    static func run(root: URL, images: [URL], run: Bool) throws {
        let context = try MetalContext()
        let mapping = try VisionMapping(root: root, device: context.device)
        let config = mapping.config

        print("""
            vision tower at \(root.lastPathComponent)
              \(config.depth) blocks, width \(config.hiddenSize), \(config.headCount) heads \
            of \(config.headDim), MLP \(config.intermediateSize)
              patch \(config.patchSize), \(config.spatialMergeSize)x\(config.spatialMergeSize) \
            merge, projecting to \(config.outHiddenSize)
              \(mapping.byteCount / 1_048_576) MiB, every tensor present and the shape the \
            config implies
            """)

        // A couple of representative weights, to show the bytes are real rather than zeros.
        for name in [VisionMapping.Name.patchBias, VisionMapping.Name.mergerFC2Bias] {
            let values = try mapping.floats(name)
            let magnitude = values.reduce(0) { $0 + abs($1) } / Float(values.count)
            print(String(
                format: "  %-40s %6d values, mean |x| %.4f",
                (name as NSString).utf8String!, values.count, magnitude))
        }

        guard !images.isEmpty else {
            print("\n  pass image paths to see what each would cost in tokens")
            return
        }
        let patcher = ImagePatcher(config: config)
        print("\n  image                              resized      grid     tokens")
        for url in images {
            do {
                let planned = try patcher.plan(for: url)
                print(String(
                    format: "  %-32s %5dx%-5d %4dx%-4d %7d",
                    (url.lastPathComponent as NSString).utf8String!,
                    planned.grid.width * config.patchSize,
                    planned.grid.height * config.patchSize,
                    planned.grid.width, planned.grid.height, planned.tokens))
            } catch {
                print("  \(url.lastPathComponent): \(error)")
            }
        }

        guard run else {
            print("\n  add --run to execute the tower on each of them")
            return
        }

        // What the tower actually costs, which is the number the design of the attention
        // kernel turns on. Predicted slow; measured is better than predicted.
        let tower = VisionTower(config: config, context: context, weights: mapping)
        print("\n  image                          patches    tower  positions   blocks    merge")
        for url in images {
            do {
                let patched = try patcher.patch(contentsOf: url)
                let start = Date()
                let embeddings = try tower.forward(
                    patches: patched.values, grid: patched.grid)
                let seconds = Date().timeIntervalSince(start)
                let tokens = config.tokenCount(for: patched.grid)

                // Finiteness and spread: a tower that returned zeros would be fast and useless.
                let finite = embeddings.allSatisfy { $0.isFinite }
                let spread = (embeddings.max() ?? 0) - (embeddings.min() ?? 0)
                let t = tower.lastTimings
                _ = tokens
                print(String(
                    format: "  %-28s %7d %7.1fs %8.2fs %7.1fs %7.2fs  %@",
                    (url.lastPathComponent as NSString).utf8String!,
                    patched.grid.patchCount, seconds, t.positions, t.blocks, t.merge,
                    finite && spread > 1e-4 ? "" : "SUSPECT: flat or non-finite"))
            } catch {
                print("  \(url.lastPathComponent): \(error)")
            }
        }
    }
}
