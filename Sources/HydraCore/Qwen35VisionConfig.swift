import Foundation

/// The geometry of Qwen 3.6's vision tower, and the image layout it expects.
///
/// **Every number here is transcribed from the checkpoint's own `config.json` and
/// `preprocessor_config.json`**, in the same spirit as `expertQuantization`: recorded, not
/// chosen. The tower has been installed since D-021 and never executed, so nothing until now
/// has had an opinion about its shape, and the manifest describes each of its 333 tensors with
/// a dtype and a shape precisely so that this file could be written later against the file
/// rather than against a memory of the architecture.
///
/// The tower is a 27-block ViT at width 1152. What makes it not an ordinary ViT, and what the
/// rest of this type exists to express, is that the image is **not resized to a fixed square**:
/// Qwen keeps the aspect ratio and varies the token count with the area, so almost everything
/// downstream is a function of the grid rather than a constant.
public struct Qwen35VisionConfig: Sendable, Equatable {

    // MARK: - The tower

    public var depth: Int = 27
    public var hiddenSize: Int = 1152
    public var headCount: Int = 16
    public var intermediateSize: Int = 4304
    /// What the merger projects into: the text model's hidden size.
    public var outHiddenSize: Int = 2048
    public var headDim: Int { hiddenSize / headCount }          // 72

    // MARK: - How an image becomes patches

    public var patchSize: Int = 16
    public var inChannels: Int = 3
    /// Frames a patch spans. A still image is fed as **two identical frames**, because the
    /// patch embedding is a 3-D convolution and this is how the reference presents one image
    /// to it; there is no separate image path.
    public var temporalPatchSize: Int = 2
    /// The 2x2 block of patches the merger folds into one text token.
    public var spatialMergeSize: Int = 2

    /// The grid dimensions must both be multiples of this, since the merger consumes 2x2
    /// blocks of patches and a half block has no meaning.
    public var sizeFactor: Int { patchSize * spatialMergeSize }  // 32

    /// Values one patch carries into the first projection: `2 * 16 * 16 * 3`.
    public var patchElements: Int {
        temporalPatchSize * patchSize * patchSize * inChannels   // 1536
    }

    /// Width of the merger's input, four patches side by side.
    public var mergedWidth: Int { hiddenSize * spatialMergeSize * spatialMergeSize }  // 4608

    // MARK: - Learned position grid

    /// Entries in `vision_tower.pos_embed`, a square grid resampled to each image's own.
    public var positionEmbeddingCount: Int = 2304
    /// 48. Bilinear interpolation from this grid to the image's is how a variable resolution
    /// gets a learned position at all.
    public var positionGridSide: Int { Int(Double(positionEmbeddingCount).squareRoot()) }

    // MARK: - Pixel budget

    /// Area bounds in pixels, from `preprocessor_config.json`'s `size`. They read as odd
    /// numbers because they are areas, not edges: 65536 is 256x256 and 16777216 is 4096x4096.
    public var minimumPixels: Int = 65536

    /// The published ceiling, 4096x4096. Kept as the bound no setting may exceed: past it the
    /// learned 48x48 position grid is being stretched further than the checkpoint ever saw.
    public static let publishedMaximumPixels = 16_777_216

    /// **How many text tokens one image may become.** This is the knob, and it is expressed in
    /// tokens rather than pixels because tokens are what the user pays: context, and prefill
    /// time.
    ///
    /// Nothing is rejected at any setting. `smart_resize` already scales every image to fit the
    /// budget, so a lower number is a smaller image, never a refusal.
    ///
    /// The published budget allows 16.7 million pixels, which turns a 4032x3024 phone photo
    /// into **11,844 tokens**: around eight minutes of text prefill on this machine, and a third
    /// of a 32k context spent on one picture. That ceiling is written for server deployments.
    ///
    /// **1024 is a placeholder, and it stands on the tower rather than on prefill.** The first
    /// figure here was 4096, chosen against text prefill speed before the tower had ever been
    /// run. M-070 then ran it: the tower is 99 % attention, quadratic in the patch count, and at
    /// 4096 tokens it projects to ten minutes on its own. At 1024 it is 37 seconds.
    ///
    /// That kernel needs rewriting rather than tuning, and when it is this number should be
    /// decided again on a fresh measurement instead of inherited from this one.
    public var maximumTokens: Int = 1024

    /// The pixel budget the token budget implies, never above what the checkpoint published.
    ///
    /// One token is `patchSize^2 * spatialMergeSize^2` pixels, 1024 of them at this geometry,
    /// so the two are the same number in different units.
    public var maximumPixels: Int {
        let perToken = patchSize * patchSize * spatialMergeSize * spatialMergeSize
        return min(maximumTokens * perToken, Self.publishedMaximumPixels)
    }
    /// Rescale and normalize: `(x / 255 - mean) / std`, with mean and std 0.5 on every channel,
    /// which puts a pixel in [-1, 1].
    public var imageMean: Float = 0.5
    public var imageStd: Float = 0.5

    // MARK: - The tokens that carry an image in the text stream

    public var imageTokenID: Int = 248_056
    public var videoTokenID: Int = 248_057
    public var visionStartTokenID: Int = 248_053
    public var visionEndTokenID: Int = 248_054

    public init() {}

    public static let a3b = Qwen35VisionConfig()

    // MARK: - Resizing

    public enum ImageError: Error, CustomStringConvertible {
        case extremeAspectRatio(height: Int, width: Int)
        case degenerate(height: Int, width: Int)

        public var description: String {
            switch self {
            case .extremeAspectRatio(let h, let w):
                return "aspect ratio \(Double(max(h, w)) / Double(min(h, w))) exceeds 200, "
                    + "for a \(w)x\(h) image"
            case .degenerate(let h, let w):
                return "an image of \(w)x\(h) has no pixels"
            }
        }
    }

    /// The dimensions an image is resized to before patching.
    ///
    /// A direct transcription of the reference `smart_resize`. Three conditions, in this order:
    /// both sides divisible by `sizeFactor`, total area inside the pixel budget, aspect ratio
    /// preserved as closely as those allow.
    ///
    /// The order of the branches is load-bearing and easy to get subtly wrong. Rounding happens
    /// **first**, and the budget is then checked against the *rounded* area; when it is
    /// exceeded, the scale is computed from the **original** area rather than the rounded one,
    /// and the sides are floored rather than rounded so the result cannot creep back over the
    /// bound. A version that rounded after scaling would return a grid a few patches too large
    /// on some images and nothing would catch it: the tower would run, the merger would run, and
    /// the model would receive a plausible number of slightly wrong tokens.
    public func resizedDimensions(height: Int, width: Int) throws -> (height: Int, width: Int) {
        guard height > 0, width > 0 else {
            throw ImageError.degenerate(height: height, width: width)
        }
        let factor = sizeFactor
        let tall = Double(max(height, width)), short = Double(min(height, width))
        guard tall / short <= 200 else {
            throw ImageError.extremeAspectRatio(height: height, width: width)
        }

        func round(_ value: Double) -> Int {
            Int((value / Double(factor)).rounded()) * factor
        }
        var h = round(Double(height))
        var w = round(Double(width))

        if h * w > maximumPixels {
            let beta = (Double(height) * Double(width) / Double(maximumPixels)).squareRoot()
            h = max(factor, Int((Double(height) / beta / Double(factor)).rounded(.down)) * factor)
            w = max(factor, Int((Double(width) / beta / Double(factor)).rounded(.down)) * factor)
        } else if h * w < minimumPixels {
            let beta = (Double(minimumPixels) / (Double(height) * Double(width))).squareRoot()
            h = Int((Double(height) * beta / Double(factor)).rounded(.up)) * factor
            w = Int((Double(width) * beta / Double(factor)).rounded(.up)) * factor
        }
        // Rounding a side below one factor is possible for a very thin image, and a grid of
        // zero patches would divide by zero downstream rather than fail here.
        return (max(factor, h), max(factor, w))
    }

    /// The patch grid, and the number of text tokens the image will occupy.
    ///
    /// `temporal` is 1 for a still image. The merger folds 2x2 patches into one token, so the
    /// token count is the patch count over four, and that is the number of `imageTokenID`
    /// placeholders the prompt has to carry.
    public struct Grid: Sendable, Equatable {
        public let temporal: Int
        public let height: Int
        public let width: Int

        public init(temporal: Int, height: Int, width: Int) {
            self.temporal = temporal
            self.height = height
            self.width = width
        }

        public var patchCount: Int { temporal * height * width }
    }

    public func grid(forResizedHeight height: Int, width: Int, frames: Int = 1) -> Grid {
        Grid(temporal: frames, height: height / patchSize, width: width / patchSize)
    }

    // MARK: - Patch order

    /// Where the patch at sequence position `index` sits in the image.
    ///
    /// **Patches are not in reading order.** They are grouped into the 2x2 blocks the merger
    /// folds, so four consecutive patches are a square, and the blocks themselves run left to
    /// right then top to bottom. The reference expresses this as a reshape to
    /// `(h/m, m, w/m, m)` followed by `transpose(1, 2)`; this is the same permutation in closed
    /// form, checked against it.
    ///
    /// A 4x6 grid begins `(0,0) (0,1) (1,0) (1,1) (0,2) (0,3) (1,2) (1,3)`: the first four
    /// patches are the top-left square, not the first row.
    ///
    /// Getting this wrong is the quietest failure in the whole pipeline. Reading order is the
    /// obvious implementation, it produces the right number of patches and the right number of
    /// tokens, every kernel runs, and the model simply sees an image whose pixels have been
    /// shuffled into the wrong 2x2 groups. There is nothing to catch it downstream.
    public func patchPosition(atSequenceIndex index: Int, grid: Grid)
        -> (frame: Int, y: Int, x: Int)
    {
        let merge = spatialMergeSize
        let perFrame = grid.height * grid.width
        let frame = index / perFrame
        let within = index % perFrame

        let blocksPerRow = grid.width / merge
        let blockArea = merge * merge
        let blockRow = within / (blockArea * blocksPerRow)
        let remainder = within % (blockArea * blocksPerRow)
        let blockColumn = remainder / blockArea
        let inside = remainder % blockArea

        return (frame, blockRow * merge + inside / merge, blockColumn * merge + inside % merge)
    }

    /// Text tokens one image occupies, which is what the prompt must reserve.
    public func tokenCount(for grid: Grid) -> Int {
        grid.patchCount / (spatialMergeSize * spatialMergeSize)
    }
}
