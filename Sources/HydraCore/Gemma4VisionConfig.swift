import Foundation

/// The geometry of Gemma 4's vision tower, transcribed from the checkpoint's own `config.json`
/// and from `modeling_gemma4.py`.
///
/// It is a 27-block encoder at width 1152, the same two numbers as Qwen's tower, and **almost
/// nothing else is the same**. Enough is different that reusing Qwen's implementation with the
/// dimensions swapped would produce a tower that runs and answers wrongly, so the differences
/// are listed here rather than discovered one at a time:
///
/// | | Qwen 3.6 | Gemma 4 |
/// | --- | --- | --- |
/// | norms | LayerNorm, mean removed, with bias | RMSNorm, four a layer, sandwiched |
/// | patch embedding | conv over 2 frames, with bias | linear over one frame, **no bias** |
/// | learned positions | one 48x48 grid, bilinearly resampled | **two tables, looked up and summed** |
/// | rotary | one 2-D turn, theta 10000 | one 2-D turn, **theta 100** |
/// | attention scale | `1/sqrt(headDim)` | **1.0** |
/// | q, k, v | rotary on q and k | **all three normalized first** |
/// | MLP | fc1, GELU, fc2 | gate and up, GELU, down |
/// | token count | patches / 4, from a 2x2 merge | patches / 9, from **average pooling** |
/// | projection | two linears with bias | one linear, no bias, **quantized** |
///
/// Nothing in that table announces itself at runtime. Every row is a finite, plausible tower.
public struct Gemma4VisionConfig: Sendable, Equatable {

    // MARK: - The tower

    public var depth: Int = 27
    public var hiddenSize: Int = 1152
    public var headCount: Int = 16
    public var headDim: Int = 72
    public var intermediateSize: Int = 4304
    public var rmsNormEps: Float = 1e-6
    /// What the projector lands in: the text model's hidden size.
    public var outHiddenSize: Int = 2816

    /// **One, not `1/sqrt(headDim)`.** The reference sets `self.scaling = 1.0` outright, because
    /// the query and key are RMS-normalized before the product and the usual scale would be
    /// applied twice. Using the customary value here shrinks every logit by a factor of 8.5,
    /// which flattens the attention rather than breaking it: an image described vaguely.
    public var attentionScale: Float = 1.0

    // MARK: - How an image becomes patches

    public var patchSize: Int = 16
    public var inChannels: Int = 3
    /// `16 * 16 * 3`. One frame, unlike Qwen, which feeds a still image twice.
    public var patchElements: Int { patchSize * patchSize * inChannels }   // 768

    /// The square of patches averaged into one soft token.
    public var poolingKernelSize: Int = 3
    /// Patches a soft token comes from: nine, not Qwen's four.
    public var patchesPerToken: Int { poolingKernelSize * poolingKernelSize }

    /// Soft tokens an image occupies, from `vision_soft_tokens_per_image`.
    ///
    /// **A count, not a consequence.** Qwen's token count falls out of the image's grid; here it
    /// is requested, and the pooling kernel is derived from the ratio between the patch count and
    /// this. So the grid must carry `patchesPerToken * softTokens` patches exactly, and an image
    /// that does not is padded to it.
    public var softTokens: Int = 280

    // MARK: - Positions

    /// Entries in each of the two position tables.
    ///
    /// **Two tables, looked up and summed**, `table[0][x] + table[1][y]`, not one grid resampled.
    /// The tensor is `[2, 10240, 1152]`: the leading 2 is the axis, not a batch. Reading it as a
    /// flat table of 20480 entries would index the y table with x positions past 10240, which
    /// stays in bounds and returns the wrong vector.
    public var positionEmbeddingSize: Int = 10240

    /// **100, not 10000.** Three orders of magnitude, and the smaller value is deliberate: the
    /// positions being turned are patch coordinates, tens of them across an image, not thousands
    /// of tokens along a conversation.
    public var ropeTheta: Double = 100

    /// The rotary turns each half of a head with its own axis: `2 * (headDim / 4)` channels for
    /// x and the same for y.
    public var rotaryChannelsPerAxis: Int { 2 * (headDim / 4) }             // 36

    // MARK: - The tokens that carry an image

    public var imageTokenID: Int = 258_880

    public init() {}

    public static let a4b = Gemma4VisionConfig()

    // MARK: - Pixels

    /// **Normalization happens in the model, not in preprocessing.** The patch embedder does
    /// `2 * (x - 0.5)` on values already rescaled to `[0, 1]`, which lands in `[-1, 1]`: the same
    /// range Qwen reaches, by a different route and at a different stage.
    public func normalize(_ byte: UInt8) -> Float { 2 * (Float(byte) / 255 - 0.5) }

    /// Patches the budget allows: `softTokens * patchesPerToken`.
    public var maximumPatches: Int { softTokens * patchesPerToken }        // 2520
    /// Both sides must divide by this, so the 3x3 pool lands on whole blocks.
    public var sideMultiple: Int { poolingKernelSize * patchSize }         // 48

    public enum ImageError: Error, CustomStringConvertible {
        case degenerate(height: Int, width: Int)

        public var description: String {
            switch self {
            case let .degenerate(h, w):
                return "a \(w)x\(h) image resizes to nothing at this patch budget"
            }
        }
    }

    /// The size an image is resized to, transcribed from `get_aspect_ratio_preserving_size`.
    ///
    /// **The aspect ratio is preserved and the area is filled**, which is a different rule from
    /// Qwen's in two ways worth stating because both are easy to assume away.
    ///
    /// A small image is scaled **up**. The factor is `sqrt(targetArea / area)` with no upper
    /// clamp, so a 320x240 thumbnail is enlarged to 912x672 rather than left alone: the budget is
    /// a target, not a ceiling. And the token count **varies per image**, because it falls out of
    /// the resized grid; `softTokens` is the maximum an image may reach, not the number it takes.
    /// Qwen's is the same for every picture and Gemma's is not.
    ///
    /// Sides are floored to a multiple of 48 so the pool divides them, which is why the result is
    /// usually a little under the budget rather than exactly on it.
    public func resizedDimensions(height: Int, width: Int) throws -> (height: Int, width: Int) {
        guard height > 0, width > 0 else {
            throw ImageError.degenerate(height: height, width: width)
        }
        let targetArea = Double(maximumPatches * patchSize * patchSize)
        let factor = (targetArea / (Double(height) * Double(width))).squareRoot()
        let side = sideMultiple

        var h = Int((factor * Double(height) / Double(side)).rounded(.down)) * side
        var w = Int((factor * Double(width) / Double(side)).rounded(.down)) * side

        // A very thin image rounds one side to nothing. The reference gives that side the
        // minimum and derives the other from the raw ratio, bounded by the longest strip the
        // budget can hold.
        let maximumSide = (maximumPatches / patchesPerToken) * side
        if h == 0 && w == 0 {
            throw ImageError.degenerate(height: height, width: width)
        } else if h == 0 {
            h = side
            w = min(Int((Double(width) / Double(height)).rounded(.down)) * side, maximumSide)
        } else if w == 0 {
            w = side
            h = min(Int((Double(height) / Double(width)).rounded(.down)) * side, maximumSide)
        }
        return (h, w)
    }

    /// The patch grid of a resized image. Patches are in **reading order**, unlike Qwen's, which
    /// are grouped into the blocks its merger folds.
    public func grid(forResizedHeight height: Int, width: Int) -> (height: Int, width: Int) {
        (height / patchSize, width / patchSize)
    }

    /// Soft tokens an image of this grid occupies. **Varies per image**, at most `softTokens`.
    public func tokenCount(forGridHeight height: Int, width: Int) -> Int {
        (height / poolingKernelSize) * (width / poolingKernelSize)
    }

    /// Where patch `index` sits, as the position tables are indexed: `(x, y)`, column first.
    ///
    /// The reference builds these with `meshgrid(arange(width), arange(height), indexing: "xy")`,
    /// which puts the **column** in the first slot. The tables are looked up as `table[0][x]` and
    /// `table[1][y]`, so swapping them indexes the x table with a row number: in range, wrong
    /// vector, and only visibly wrong on a non-square image.
    public func patchPosition(atIndex index: Int, gridWidth: Int) -> (x: Int, y: Int) {
        (x: index % gridWidth, y: index / gridWidth)
    }
}
