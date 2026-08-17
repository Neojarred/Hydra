import CoreGraphics
import Foundation
import HydraCore
import ImageIO
import UniformTypeIdentifiers

/// Turns an image on disk into the float buffer Qwen's patch embedding expects.
///
/// Four steps, and each of them is a place where a wrong answer looks like a right one:
/// decode to RGB, resize to a grid-aligned size, normalize to [-1, 1], and lay the pixels out
/// as patches in the **merge order** the tower reads them in. Nothing downstream can tell that
/// the last step went wrong, which is why `Qwen35VisionConfig.patchPosition` is tested against
/// the reference permutation rather than trusted.
///
/// The output is one contiguous `[patchCount][patchElements]` array, patch-major. That is the
/// shape the reference's `Conv3d` sees after its own `view(-1, C, T, P, P)`, so the projection
/// is an ordinary matrix multiply against a `[1152][1536]` weight and no convolution kernel is
/// needed anywhere.
public struct ImagePatcher: Sendable {

    public let config: Qwen35VisionConfig

    public init(config: Qwen35VisionConfig = .a3b) {
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

    /// One image, ready for the tower.
    public struct Patched: Sendable {
        /// `[patchCount][patchElements]`, patch-major, in merge order.
        public let values: [Float]
        public let grid: Qwen35VisionConfig.Grid
        /// The size the image was resized to, kept for diagnostics: a surprising token count is
        /// almost always a surprising resize.
        public let pixelHeight: Int
        public let pixelWidth: Int

        public init(
            values: [Float], grid: Qwen35VisionConfig.Grid, pixelHeight: Int, pixelWidth: Int
        ) {
            self.values = values
            self.grid = grid
            self.pixelHeight = pixelHeight
            self.pixelWidth = pixelWidth
        }
    }

    // MARK: - Decoding

    /// Decodes an image to tightly packed 8-bit RGB at the size the tower wants.
    ///
    /// The resize happens in the draw, so CoreGraphics does the resampling. `preprocessor_config`
    /// names bicubic; `.high` is CoreGraphics' bicubic-class filter and is the closest this can
    /// get without reimplementing the resampler. A few tenths of a pixel of difference from the
    /// reference is expected and harmless, unlike a wrong *size*, which is not.
    ///
    /// Orientation is applied rather than ignored. A photograph from a phone is usually stored
    /// rotated with an EXIF tag saying so, and a tower fed the unrotated pixels would describe a
    /// sideways picture perfectly correctly.
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

        // Drawn as RGBA and packed down to RGB afterwards.
        //
        // CoreGraphics has no 24-bit RGB context on this platform: `CGImageAlphaInfo.none` at 8
        // bits a component is rejected outright, so a three-byte buffer cannot be drawn into
        // however tidy it would be. The pack costs one pass over the pixels, against a decode
        // and a resize, and it keeps `patch(rgb:)` taking a tightly packed buffer, which is what
        // makes it testable on pixels this file did not produce.
        //
        // The row stride is set explicitly to `width * 4`. Letting CoreGraphics choose it lets
        // it pad rows for alignment, and the pack below would then shear the image by a pixel a
        // row: an image that is subtly diagonal, and perfectly finite.
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

    /// The size an image on disk will be resized to, without decoding its pixels.
    ///
    /// Reading the header alone is enough, and it is what lets the app show a token cost beside
    /// an attachment before anything expensive happens.
    public func plan(for url: URL) throws -> (grid: Qwen35VisionConfig.Grid, tokens: Int) {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
            let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil)
                as? [CFString: Any],
            let width = properties[kCGImagePropertyPixelWidth] as? Int,
            let height = properties[kCGImagePropertyPixelHeight] as? Int
        else {
            throw PatchError.cannotDecode(url)
        }
        // An orientation of 5 to 8 means the stored pixels are rotated a quarter turn, so the
        // displayed image is the transpose and the grid must be computed on that.
        let orientation = (properties[kCGImagePropertyOrientation] as? Int) ?? 1
        let (w, h) = orientation >= 5 ? (height, width) : (width, height)

        let sized = try config.resizedDimensions(height: h, width: w)
        let grid = config.grid(forResizedHeight: sized.height, width: sized.width)
        return (grid, config.tokenCount(for: grid))
    }

    // MARK: - Patching

    public func patch(contentsOf url: URL) throws -> Patched {
        let planned = try plan(for: url)
        let height = planned.grid.height * config.patchSize
        let width = planned.grid.width * config.patchSize
        let pixels = try rasterize(url, height: height, width: width)
        return Patched(
            values: patch(rgb: pixels, height: height, width: width, grid: planned.grid),
            grid: planned.grid, pixelHeight: height, pixelWidth: width)
    }

    /// Lays out tightly packed RGB as patches, normalized.
    ///
    /// Split out from the decoding so it can be tested on pixels this file did not produce,
    /// which is the only way to check the ordering against a hand-built image.
    ///
    /// Inside one patch the reference's `Conv3d` weight is indexed `[out][T][P][P][C]`, so the
    /// element order is frame, then row, then column, then channel. The temporal axis is the
    /// **same frame twice** for a still image: the tower has no separate image path, and
    /// halving it would leave the second half of every patch zero, which the model would read
    /// as a real signal rather than as absence.
    public func patch(
        rgb pixels: [UInt8], height: Int, width: Int, grid: Qwen35VisionConfig.Grid
    ) -> [Float] {
        let patchSize = config.patchSize
        let channels = config.inChannels
        let frames = config.temporalPatchSize
        let elements = config.patchElements
        let mean = config.imageMean, std = config.imageStd

        var out = [Float](repeating: 0, count: grid.patchCount * elements)
        out.withUnsafeMutableBufferPointer { destination in
            pixels.withUnsafeBufferPointer { source in
                for index in 0..<grid.patchCount {
                    let position = config.patchPosition(atSequenceIndex: index, grid: grid)
                    let originY = position.y * patchSize
                    let originX = position.x * patchSize
                    var cursor = index * elements

                    for _ in 0..<frames {
                        for row in 0..<patchSize {
                            let rowBase = (originY + row) * width * channels
                            for column in 0..<patchSize {
                                let pixel = rowBase + (originX + column) * channels
                                for channel in 0..<channels {
                                    let value = Float(source[pixel + channel]) / 255
                                    destination[cursor] = (value - mean) / std
                                    cursor += 1
                                }
                            }
                        }
                    }
                }
            }
        }
        return out
    }
}
