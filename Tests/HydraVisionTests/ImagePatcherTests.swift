import CoreGraphics
import Foundation
import HydraCore
import ImageIO
import Testing
import UniformTypeIdentifiers

@testable import HydraVision

/// Pixels in, patches out, checked against images whose every value is known in advance.
///
/// The patch layout cannot be validated by looking at it. A patcher that emits the right number
/// of floats in the wrong arrangement produces an image the tower processes perfectly happily
/// and describes wrongly, with nothing anywhere to raise. So the tests here build images where
/// each pixel encodes its own coordinates, and then assert where those coordinates land.
@Suite("Image patching")
struct ImagePatcherTests {

    private let patcher = ImagePatcher()
    private let config = Qwen35VisionConfig.a3b

    /// An image whose pixel `(y, x)` is `(y, x, 0)`, so a misplaced patch is self-identifying.
    private func coordinateImage(height: Int, width: Int) -> [UInt8] {
        var pixels = [UInt8](repeating: 0, count: height * width * 3)
        for y in 0..<height {
            for x in 0..<width {
                let base = (y * width + x) * 3
                pixels[base] = UInt8(y % 256)
                pixels[base + 1] = UInt8(x % 256)
                pixels[base + 2] = 0
            }
        }
        return pixels
    }

    @Test("Each patch holds the pixels of its own square, in merge order")
    func patchesCarryTheRightPixels() {
        let grid = Qwen35VisionConfig.Grid(temporal: 1, height: 4, width: 6)
        let height = grid.height * config.patchSize     // 64
        let width = grid.width * config.patchSize       // 96
        let values = patcher.patch(
            rgb: coordinateImage(height: height, width: width),
            height: height, width: width, grid: grid)

        #expect(values.count == grid.patchCount * config.patchElements)

        // Undo the normalization to recover the byte that was written.
        func byte(_ value: Float) -> Int { Int((value * config.imageStd + config.imageMean) * 255 + 0.5) }

        for index in 0..<grid.patchCount {
            let position = config.patchPosition(atSequenceIndex: index, grid: grid)
            let base = index * config.patchElements
            // The first pixel of the patch is its top-left corner.
            #expect(byte(values[base]) == position.y * config.patchSize, "patch \(index) row")
            #expect(byte(values[base + 1]) == position.x * config.patchSize, "patch \(index) column")

            // And the last pixel of the first frame is the bottom-right corner.
            let lastOfFrame = base + (config.patchSize * config.patchSize - 1) * config.inChannels
            #expect(byte(values[lastOfFrame]) == position.y * config.patchSize + 15)
            #expect(byte(values[lastOfFrame + 1]) == position.x * config.patchSize + 15)
        }
    }

    /// The temporal axis is the same frame twice, not one frame and a hole.
    ///
    /// Zero-filling the second half is the natural mistake for a still image, and it is not
    /// absence to the model: it is a patch whose second frame is uniformly mid-grey, which the
    /// tower will happily encode as content.
    @Test("A still image fills both frames of the patch")
    func stillImagesRepeatTheFrame() {
        let grid = Qwen35VisionConfig.Grid(temporal: 1, height: 2, width: 2)
        let height = 32, width = 32
        let values = patcher.patch(
            rgb: coordinateImage(height: height, width: width),
            height: height, width: width, grid: grid)

        let frameElements = config.patchSize * config.patchSize * config.inChannels
        for index in 0..<grid.patchCount {
            let base = index * config.patchElements
            let first = Array(values[base..<(base + frameElements)])
            let second = Array(values[(base + frameElements)..<(base + 2 * frameElements)])
            #expect(first == second, "patch \(index)'s two frames differ")
            #expect(first.contains { $0 != 0 }, "the frame is empty, so the test proves nothing")
        }
    }

    /// Normalization is `(x / 255 - 0.5) / 0.5`, which maps black to -1 and white to +1.
    @Test("Normalization spans minus one to one")
    func normalization() {
        let grid = Qwen35VisionConfig.Grid(temporal: 1, height: 2, width: 2)
        var pixels = [UInt8](repeating: 0, count: 32 * 32 * 3)
        for i in 0..<(16 * 32 * 3) { pixels[i] = 255 }      // top half white

        let values = patcher.patch(rgb: pixels, height: 32, width: 32, grid: grid)
        #expect(values.contains { abs($0 - 1.0) < 1e-6 }, "white did not reach +1")
        #expect(values.contains { abs($0 + 1.0) < 1e-6 }, "black did not reach -1")
        #expect(values.allSatisfy { $0 >= -1.0001 && $0 <= 1.0001 })
    }

    // MARK: - Through a real file

    /// Writes a PNG so the decoding path is exercised, not only the arithmetic.
    private func writePNG(height: Int, width: Int) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appending(path: "hydra-vision-\(UUID().uuidString).png")
        let space = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8,
            bytesPerRow: 0, space: space,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        // Grey, deliberately: the flatness check below measures the spread over every value in
        // the buffer, and a coloured fill has three different channel values, so it would
        // "spread" by a full unit while being perfectly uniform. That is what the first version
        // of this test measured.
        context.setFillColor(CGColor(red: 0.5, green: 0.5, blue: 0.5, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        let image = context.makeImage()!
        let destination = CGImageDestinationCreateWithURL(
            url as CFURL, UTType.png.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(destination, image, nil)
        #expect(CGImageDestinationFinalize(destination))
        return url
    }

    @Test("A file on disk becomes a grid the tower can consume")
    func endToEnd() throws {
        let url = try writePNG(height: 480, width: 640)
        defer { try? FileManager.default.removeItem(at: url) }

        // The plan is taken from the header alone, and must agree with what patching produces.
        let planned = try patcher.plan(for: url)
        let patched = try patcher.patch(contentsOf: url)

        #expect(patched.grid == planned.grid, "the header and the pixels disagree on the grid")
        #expect(patched.pixelHeight == patched.grid.height * config.patchSize)
        #expect(patched.pixelWidth == patched.grid.width * config.patchSize)
        #expect(patched.values.count == patched.grid.patchCount * config.patchElements)
        #expect(patched.values.allSatisfy { $0.isFinite && $0 >= -1.0001 && $0 <= 1.0001 })

        // 640x480 is 307,200 pixels, inside the budget, so it only rounds to the grid.
        let expected = try config.resizedDimensions(height: 480, width: 640)
        #expect((patched.pixelHeight, patched.pixelWidth) == (expected.height, expected.width))

        // A flat grey must stay flat. A spread here would mean the rasterizer padded its rows
        // and the pack read across the padding, which shears the image by a pixel a row.
        let spread = (patched.values.max() ?? 0) - (patched.values.min() ?? 0)
        #expect(spread < 0.05, "a uniform image came out with a spread of \(spread)")
    }

    @Test("A file that is not an image is refused")
    func refusesRubbish() throws {
        let url = FileManager.default.temporaryDirectory
            .appending(path: "hydra-vision-\(UUID().uuidString).png")
        try Data("not a png".utf8).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(throws: ImagePatcher.PatchError.self) { _ = try patcher.plan(for: url) }
    }
}
