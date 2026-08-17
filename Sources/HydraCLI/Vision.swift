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

    static func run(root: URL, images: [URL]) throws {
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
    }
}
