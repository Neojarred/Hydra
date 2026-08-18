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

    /// The patch grid an image is resized to for a given soft-token budget.
    ///
    /// The pooler needs `poolingKernelSize` to divide both sides, and the product to be exactly
    /// `patchesPerToken * softTokens`, so the grid is chosen rather than derived from the image.
    ///
    /// **Not yet checked against the reference.** Gemma's processor hands the model
    /// `pixel_position_ids` already computed, and how it arrives at them has not been read. The
    /// constraints below are forced by the pooler and are certainly necessary; which of the
    /// admissible grids the reference picks is a guess, and a wrong guess here is a picture
    /// squeezed into the wrong shape, which is finite, plausible and wrong in the way this whole
    /// file exists to warn about. It must be verified before the tower is trusted on real images.
    ///
    /// The error is measured on the logarithm of the ratio so that a 2:1 image and a 1:2 image
    /// are treated alike; on the ratio itself they are not, and 2:1 picks a visibly worse grid.
    public func gridSides(forAspectRatio ratio: Double) -> (height: Int, width: Int) {
        let total = patchesPerToken * softTokens
        let k = poolingKernelSize
        // Blocks of k by k, so the pooled grid is whole. Choose the block counts closest to the
        // image's shape whose product is the budget.
        let blocks = total / (k * k)
        // Ties are real rather than hypothetical. 280's divisors offer 2.8 and 1.43, the same
        // distance from 2 in log space, so a 2:1 image has two equally good grids and iteration
        // order alone decides between them: a portrait grid for a landscape picture, on a whim.
        //
        // The tie goes to the squarer grid, which distorts the image less and, because the tie
        // sets of a ratio and its reciprocal are transposes of each other, makes a 2:1 image and
        // a 1:2 image land on transposed grids rather than unrelated ones.
        var best = (height: 1, width: blocks)
        var bestKey = (error: Double.infinity, squareness: Double.infinity)
        for h in 1...blocks where blocks % h == 0 {
            let w = blocks / h
            let shape = Foundation.log(Double(w) / Double(h))
            let key = (error: abs(shape - Foundation.log(ratio)), squareness: abs(shape))
            if key.error < bestKey.error - 1e-12
                || (abs(key.error - bestKey.error) <= 1e-12 && key.squareness < bestKey.squareness)
            {
                bestKey = key
                best = (h, w)
            }
        }
        return (best.height * k, best.width * k)
    }
}
