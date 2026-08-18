import Foundation
import Testing

@testable import HydraCore

/// Gemma's tower geometry, and the places it differs from Qwen's.
///
/// The two towers share a depth and a width and almost nothing else. These tests exist mostly to
/// state the differences in a form that fails if someone later "unifies" the two, because every
/// one of them produces a tower that runs and answers wrongly.
@Suite("Gemma 4 vision geometry")
struct Gemma4VisionConfigTests {

    private let config = Gemma4VisionConfig.a4b

    @Test("The transcribed constants match the published config")
    func constants() {
        #expect(config.depth == 27)
        #expect(config.hiddenSize == 1152)
        #expect(config.headDim == 72)
        #expect(config.intermediateSize == 4304)
        #expect(config.outHiddenSize == 2816, "the text model's hidden size")
        #expect(config.patchElements == 768, "16 x 16 x 3, one frame")
        #expect(config.softTokens == 280)
        #expect(config.patchesPerToken == 9, "a 3x3 average pool, not a 2x2 merge")
        #expect(config.positionEmbeddingSize == 10240)
        #expect(config.rotaryChannelsPerAxis == 36, "half a head to each axis")
    }

    /// The two values most likely to be copied from Qwen's tower, where they are different.
    @Test("The attention scale is one and the rotary base is a hundred")
    func theTwoThatLookLikeTypos() {
        #expect(config.attentionScale == 1.0, "not 1/sqrt(72); the reference sets it outright")
        #expect(config.ropeTheta == 100, "not 10000; these are patch coordinates, not tokens")

        // Stated as the comparison, so a later merge of the two towers fails here.
        let qwenLike = 1 / Float(config.headDim).squareRoot()
        #expect(config.attentionScale != qwenLike)
    }

    /// Pixels reach `[-1, 1]`, by the model's own arithmetic rather than the preprocessor's.
    @Test("Normalization spans minus one to one")
    func normalization() {
        #expect(config.normalize(0) == -1)
        #expect(config.normalize(255) == 1)
        #expect(abs(config.normalize(128) - 0.00392) < 1e-4)
    }

    /// The grid is chosen to fit the soft-token budget, not derived from the image.
    @Test("Every grid holds exactly the budget and pools evenly", arguments: [
        4.0 / 3.0, 3.0 / 4.0, 16.0 / 9.0, 1.0, 2.0, 0.5,
    ])
    func gridFitsTheBudget(ratio: Double) {
        let grid = config.gridSides(forAspectRatio: ratio)
        let patches = grid.height * grid.width

        #expect(
            patches == config.patchesPerToken * config.softTokens,
            "\(grid.width)x\(grid.height) is \(patches) patches, budget is 2520")
        #expect(grid.height % config.poolingKernelSize == 0, "the pool must divide the height")
        #expect(grid.width % config.poolingKernelSize == 0, "and the width")

        let pooled = (grid.height / config.poolingKernelSize)
            * (grid.width / config.poolingKernelSize)
        #expect(pooled == config.softTokens, "pooled to \(pooled), expected 280")
    }

    /// A wide image gets a wide grid and a tall one a tall grid, or the pooling would be
    /// squeezing the picture into a shape it never had.
    ///
    /// The symmetry is the point: a 2:1 image and a 1:2 image must get transposed grids, which
    /// only holds if the match is scored on the logarithm of the ratio. Scored on the ratio
    /// itself, 2:1 lands on a visibly worse grid than 1:2 does.
    ///
    /// **Which admissible grid the reference picks is not yet known**, so this pins a property
    /// rather than a value. See the note on `gridSides`.
    @Test("The grid follows the image's shape")
    func gridFollowsTheShape() {
        let wide = config.gridSides(forAspectRatio: 2.0)
        let tall = config.gridSides(forAspectRatio: 0.5)
        #expect(wide.width > wide.height)
        #expect(tall.height > tall.width)
        #expect(wide.width == tall.height && wide.height == tall.width, "and symmetrically")
    }
}
