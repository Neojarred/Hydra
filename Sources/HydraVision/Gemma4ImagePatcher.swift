import CoreGraphics
import Foundation
import HydraCore
import ImageIO
import UniformTypeIdentifiers

/// Turns an image on disk into the float buffer Gemma's patch embedder expects.
///
/// Same four steps as Qwen's patcher and three of them differ:
///
/// * the resize **preserves the aspect ratio and fills a patch budget**, scaling small images up
///   rather than leaving them alone, so the token count varies per image;
/// * patches come out in **reading order**, not grouped into merge blocks, because the pool
///   averages 3x3 neighbourhoods rather than folding 2x2 ones;
/// * a patch is **one frame**, not two, so it is 768 values rather than 1536.
///
/// The element order inside a patch is the same: row, then column, then channel. The reference
/// reaches it by `reshape(C, nh, ps, nw, ps).permute(1, 3, 2, 4, 0)`, which is that order.
public struct Gemma4ImagePatcher: Sendable {

    public let config: Gemma4VisionConfig

    public init(config: Gemma4VisionConfig = .a4b) {
        self.config = config
    }

    public enum PatchError: Error, CustomStringConvertible {
        case cannotRead(URL)
        case cannotDecode(URL)
        case cannotDraw

        public var description: String {
            switch self {
            case .cannotRead(let url): return "cannot read an image at \(url.path)"
            case .cannotDecode(let url): return "\(url.lastPathComponent) is not an image"
            case .cannotDraw: return "cannot rasterize the image into RGB"
            }
        }
    }

    public struct Patched: Sendable {
        /// `[patchCount][patchElements]`, patch-major, in reading order.
        public let values: [Float]
        public let gridHeight: Int
        public let gridWidth: Int
        public let pixelHeight: Int
        public let pixelWidth: Int
        /// Soft tokens this image occupies, which varies with its shape.
        public let tokens: Int
    }

    /// What an image will cost, from its header alone.
    public func plan(for url: URL) throws
        -> (gridHeight: Int, gridWidth: Int, pixelHeight: Int, pixelWidth: Int, tokens: Int)
    {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
            let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil)
                as? [CFString: Any],
            let width = properties[kCGImagePropertyPixelWidth] as? Int,
            let height = properties[kCGImagePropertyPixelHeight] as? Int
        else {
            throw PatchError.cannotDecode(url)
        }
        // A quarter turn in the EXIF tag means the displayed image is the transpose.
        let orientation = (properties[kCGImagePropertyOrientation] as? Int) ?? 1
        let (w, h) = orientation >= 5 ? (height, width) : (width, height)

        let sized = try config.resizedDimensions(height: h, width: w)
        let grid = config.grid(forResizedHeight: sized.height, width: sized.width)
        return (
            grid.height, grid.width, sized.height, sized.width,
            config.tokenCount(forGridHeight: grid.height, width: grid.width))
    }

    /// Decodes to tightly packed RGB at the size the tower wants.
    ///
    /// Drawn as RGBA and packed down, because CoreGraphics has no 24-bit RGB context here, with
    /// the row stride set explicitly so a padded row cannot shear the image.
    private func rasterize(_ url: URL, height: Int, width: Int) throws -> [UInt8] {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else {
            throw PatchError.cannotRead(url)
        }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: max(height, width),
        ]
        let decoded = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary)
            ?? CGImageSourceCreateImageAtIndex(source, 0, nil)
        guard let image = decoded else { throw PatchError.cannotDecode(url) }

        var rgba = [UInt8](repeating: 0, count: height * width * 4)
        let space = CGColorSpaceCreateDeviceRGB()
        let drawn: Bool = rgba.withUnsafeMutableBytes { raw in
            guard let context = CGContext(
                data: raw.baseAddress, width: width, height: height,
                bitsPerComponent: 8, bytesPerRow: width * 4, space: space,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
            else { return false }
            context.interpolationQuality = .high
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard drawn else { throw PatchError.cannotDraw }

        var pixels = [UInt8](repeating: 0, count: height * width * 3)
        for pixel in 0..<(height * width) {
            pixels[pixel * 3] = rgba[pixel * 4]
            pixels[pixel * 3 + 1] = rgba[pixel * 4 + 1]
            pixels[pixel * 3 + 2] = rgba[pixel * 4 + 2]
        }
        return pixels
    }

    public func patch(contentsOf url: URL) throws -> Patched {
        let planned = try plan(for: url)
        let pixels = try rasterize(
            url, height: planned.pixelHeight, width: planned.pixelWidth)
        return Patched(
            values: patch(
                rgb: pixels, height: planned.pixelHeight, width: planned.pixelWidth,
                gridHeight: planned.gridHeight, gridWidth: planned.gridWidth),
            gridHeight: planned.gridHeight, gridWidth: planned.gridWidth,
            pixelHeight: planned.pixelHeight, pixelWidth: planned.pixelWidth,
            tokens: planned.tokens)
    }

    /// Lays out tightly packed RGB as patches, in reading order, normalized to `[-1, 1]`.
    ///
    /// The reference splits the normalization between the processor, which rescales to `[0, 1]`,
    /// and the patch embedder, which does `2 * (x - 0.5)`. Composed, that is what happens here in
    /// one step; splitting it would only give two places to get the same constant wrong.
    public func patch(
        rgb pixels: [UInt8], height: Int, width: Int, gridHeight: Int, gridWidth: Int
    ) -> [Float] {
        let patchSize = config.patchSize
        let channels = config.inChannels
        let elements = config.patchElements
        var out = [Float](repeating: 0, count: gridHeight * gridWidth * elements)

        out.withUnsafeMutableBufferPointer { destination in
            pixels.withUnsafeBufferPointer { source in
                for index in 0..<(gridHeight * gridWidth) {
                    let position = config.patchPosition(atIndex: index, gridWidth: gridWidth)
                    let originY = position.y * patchSize
                    let originX = position.x * patchSize
                    var cursor = index * elements
                    for row in 0..<patchSize {
                        let rowBase = (originY + row) * width * channels
                        for column in 0..<patchSize {
                            let pixel = rowBase + (originX + column) * channels
                            for channel in 0..<channels {
                                destination[cursor] = config.normalize(source[pixel + channel])
                                cursor += 1
                            }
                        }
                    }
                }
            }
        }
        return out
    }
}
