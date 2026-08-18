import Foundation
import Testing

@testable import HydraCore

/// mRoPE, against the reference's strided assignment rather than against the config's prose.
///
/// Two readings of `mrope_section: [11, 11, 10]` produce the right counts and different models.
/// The blocked one, "the first 11 frequencies are time", is the natural reading and is wrong;
/// the interleaved one, every third frequency, is what the reference implements and what
/// `mrope_interleaved: true` announces. Nothing about shape, finiteness or magnitude separates
/// them, so they are separated here.
@Suite("Qwen 3.6 mRoPE")
struct Qwen35MRoPETests {

    /// The reference's own construction: start with every frequency on time, then overwrite a
    /// stride with height and another with width. Deliberately written the reference's way and
    /// not the implementation's, so the two can disagree.
    private func referenceAxes(frequencies: Int, sections: [Int]) -> [Int] {
        var axes = [Int](repeating: 0, count: frequencies)
        for (dim, offset) in [(1, 1), (2, 2)] {
            let length = sections[dim] * 3
            var index = offset
            while index < length {
                if index < frequencies { axes[index] = dim }
                index += 3
            }
        }
        return axes
    }

    @Test("The axis of each frequency matches the reference's strided assignment")
    func axesMatchReference() {
        let frequencies = 32          // head dim 256 at a partial rotary factor of 0.25
        let expected = referenceAxes(frequencies: frequencies, sections: [11, 11, 10])
        for j in 0..<frequencies {
            #expect(
                Qwen35MRoPE.axis(forFrequency: j).rawValue == expected[j],
                "frequency \(j): got \(Qwen35MRoPE.axis(forFrequency: j)), reference \(expected[j])")
        }
    }

    @Test("The published section sizes fall out of the interleaving")
    func sectionSizes() {
        #expect(Qwen35MRoPE.sectionSizes(frequencies: 32) == [11, 11, 10])
    }

    /// The blocked reading is a different assignment, stated so it cannot be silently adopted.
    @Test("A blocked layout would be a different model")
    func blockedLayoutDiffers() {
        var blocked: [Int] = []
        for (axis, count) in [11, 11, 10].enumerated() {
            blocked += [Int](repeating: axis, count: count)
        }
        let interleaved = (0..<32).map { Qwen35MRoPE.axis(forFrequency: $0).rawValue }
        #expect(blocked != interleaved, "the two readings agree, so this suite proves nothing")
        #expect(blocked.count == interleaved.count)
    }

    /// The property that makes today's text-only path correct: with one axis, mRoPE is RoPE.
    @Test("Text alone gives all three axes the same position")
    func textIsOrdinaryRoPE() {
        let ids = Qwen35MRoPE.positionIDs(for: [.text(count: 12)])
        #expect(ids.t == Array(0..<12))
        #expect(ids.t == ids.h && ids.h == ids.w)
    }

    /// An image occupies many tokens and few positions.
    @Test("An image advances the position by its longer side, not by its token count")
    func imageAdvancesByItsLongerSide() {
        let ids = Qwen35MRoPE.positionIDs(for: [
            .text(count: 3),
            .image(frames: 1, height: 4, width: 6),
            .text(count: 2),
        ])
        #expect(ids.t.count == 3 + 24 + 2)

        // The image's 24 tokens all sit at time 3, rows 3...6 and columns 3...8.
        let imageT = Array(ids.t[3..<27]), imageH = Array(ids.h[3..<27]), imageW = Array(ids.w[3..<27])
        #expect(Set(imageT) == [3], "a single frame is one time position")
        #expect(Set(imageH) == Set(3..<7), "four rows")
        #expect(Set(imageW) == Set(3..<9), "six columns")

        // Row-major: the first six tokens are row 3, columns 3 to 8.
        #expect(Array(imageH.prefix(6)) == [Int](repeating: 3, count: 6))
        #expect(Array(imageW.prefix(6)) == Array(3..<9))

        // The text after it resumes at 3 + max(4, 6) = 9, not at 3 + 24.
        #expect(Array(ids.t.suffix(2)) == [9, 10])
        #expect(Array(ids.h.suffix(2)) == [9, 10])
    }

    /// Advancing by the token count is the natural mistake and stays perfectly finite.
    @Test("Advancing by the token count would move everything after an image")
    func tokenCountAdvanceWouldDiffer() {
        let ids = Qwen35MRoPE.positionIDs(for: [
            .image(frames: 1, height: 64, width: 64), .text(count: 1),
        ])
        #expect(ids.t.last == 64, "a 4096-token image occupies 64 positions, not 4096")
    }

    /// Several images each advance by their own longer side.
    @Test("Two images stack")
    func twoImages() {
        let ids = Qwen35MRoPE.positionIDs(for: [
            .image(frames: 1, height: 2, width: 3),
            .image(frames: 1, height: 5, width: 4),
            .text(count: 1),
        ])
        #expect(ids.t.count == 6 + 20 + 1)
        // The second image starts at max(2, 3) = 3, and the text at 3 + max(5, 4) = 8.
        #expect(ids.h[6] == 3)
        #expect(ids.t.last == 8)
    }

    /// The angle reads the axis its frequency belongs to, and no other.
    @Test("An angle uses only its own axis")
    func angleSelectsItsAxis() {
        // Frequencies 0, 1, 2 are time, height, width.
        #expect(Qwen35MRoPE.angle(frequency: 0, inverseFrequency: 1, t: 7, h: 3, w: 5) == 7)
        #expect(Qwen35MRoPE.angle(frequency: 1, inverseFrequency: 1, t: 7, h: 3, w: 5) == 3)
        #expect(Qwen35MRoPE.angle(frequency: 2, inverseFrequency: 1, t: 7, h: 3, w: 5) == 5)
        #expect(Qwen35MRoPE.angle(frequency: 3, inverseFrequency: 1, t: 7, h: 3, w: 5) == 7)
    }
}
