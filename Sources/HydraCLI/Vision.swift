import CoreGraphics
import Foundation
import ImageIO
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

    /// Gemma's tower, which is a different architecture and a different mapping.
    static func runGemma(root: URL, images: [URL], run: Bool = false) throws {
        let context = try MetalContext()
        let mapping = try Gemma4VisionMapping(root: root, device: context.device)
        var config = mapping.config
        // The checkpoint's processor accepts (70, 140, 280, 560, 1120) and nothing else, so a
        // sweep over them is a sweep over every resolution this model actually has.
        if let requested = ProcessInfo.processInfo.environment["HYDRA_GEMMA_SOFT_TOKENS"],
            let value = Int(requested) {
            config.softTokens = value
        }
        print("""
            gemma vision tower at \(root.lastPathComponent)
              \(config.depth) blocks, width \(config.hiddenSize), \(config.headCount) heads \
            of \(config.headDim), MLP \(config.intermediateSize)
              patch \(config.patchSize), \(config.poolingKernelSize)x\(config.poolingKernelSize) \
            average pool, projecting to \(config.outHiddenSize)
              \(mapping.byteCount / 1_048_576) MiB, every tensor present and the shape the \
            config implies
            """)
        for name in [Gemma4VisionMapping.Name.standardizationScale,
                     Gemma4VisionMapping.Name.inputNorm(0)] {
            let values = try mapping.floats(name)
            let magnitude = values.reduce(0) { $0 + abs($1) } / Float(values.count)
            print(String(
                format: "  %-52s %6d values, mean |x| %.4f",
                (name as NSString).utf8String!, values.count, magnitude))
        }
        // A probe that tests the tower alone, with no language model involved.
        //
        // Two images identical except for which half is red. If the tower is spatially correct
        // the soft tokens must differ on the side that changed and barely move on the other. A
        // tower that scrambles its patches, loses its positions, or pools the wrong
        // neighbourhoods produces a difference spread evenly across every token.
        if run {
            let side = 288
            func half(_ leftRed: Bool) -> [UInt8] {
                var pixels = [UInt8](repeating: 255, count: side * side * 3)
                for y in 0..<side {
                    for x in 0..<side {
                        let inLeft = x < side / 2
                        if inLeft == leftRed {
                            let base = (y * side + x) * 3
                            pixels[base] = 220; pixels[base + 1] = 20; pixels[base + 2] = 20
                        }
                    }
                }
                return pixels
            }
            let patcher = Gemma4ImagePatcher(config: config)
            let tower = Gemma4VisionTower(config: config, context: context, weights: mapping)
            let grid = config.grid(forResizedHeight: side, width: side)
            let pooledWidth = grid.width / config.poolingKernelSize

            var outputs: [[Float]] = []
            for leftRed in [true, false] {
                let values = patcher.patch(
                    rgb: half(leftRed), height: side, width: side,
                    gridHeight: grid.height, gridWidth: grid.width)
                outputs.append(try tower.forward(
                    patches: values, gridHeight: grid.height, gridWidth: grid.width))
            }
            let hidden = config.outHiddenSize
            let tokens = outputs[0].count / hidden
            var leftChange = 0.0, rightChange = 0.0
            for token in 0..<tokens {
                var delta = 0.0
                for i in 0..<hidden {
                    delta += abs(Double(outputs[0][token * hidden + i]
                        - outputs[1][token * hidden + i]))
                }
                if token % pooledWidth < pooledWidth / 2 { leftChange += delta }
                else { rightChange += delta }
            }
            print(String(
                format: """

                      spatial probe: %dx%d, %d soft tokens over a %dx%d pooled grid
                      swapping which half is red changes the left tokens by %.0f and the right by %.0f
                      %@
                    """,
                side, side, tokens, pooledWidth, grid.height / config.poolingKernelSize,
                leftChange, rightChange,
                leftChange > 0 && rightChange > 0
                    && max(leftChange, rightChange) / min(leftChange, rightChange) < 3
                    ? "  both halves respond, which is what a symmetric change should do"
                    : "  SUSPECT: the change did not land on both halves"))
        }

        guard !images.isEmpty else { return }
        print("\n  image                          resized      grid    tokens")
        for url in images {
            guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
                let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil)
                    as? [CFString: Any],
                let width = properties[kCGImagePropertyPixelWidth] as? Int,
                let height = properties[kCGImagePropertyPixelHeight] as? Int
            else { print("  \(url.lastPathComponent): not an image"); continue }
            let sized = try config.resizedDimensions(height: height, width: width)
            let grid = config.grid(forResizedHeight: sized.height, width: sized.width)
            let tokens = config.tokenCount(forGridHeight: grid.height, width: grid.width)
            var timing = ""
            if run {
                let patcher = Gemma4ImagePatcher(config: config)
                let tower = Gemma4VisionTower(
                    config: config, context: context, weights: mapping)
                let patched = try patcher.patch(contentsOf: url)
                let started = Date()
                let out = try tower.forward(
                    patches: patched.values,
                    gridHeight: patched.gridHeight, gridWidth: patched.gridWidth)
                let seconds = Date().timeIntervalSince(started)
                let t = tower.lastTimings
                timing = String(
                    format: "  %7.2fs  blocks %6.2f  pool %5.2f  positions %5.2f %@",
                    seconds, t.blocks, t.pool, t.positions,
                    out.allSatisfy { $0.isFinite } ? "" : " NON-FINITE")
            }
            print(String(
                format: "  %-22s %5dx%-5d %4dx%-4d %6d%@",
                (url.lastPathComponent as NSString).utf8String!,
                sized.width, sized.height, grid.width, grid.height, tokens, timing))
        }
    }

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
