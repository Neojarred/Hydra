import Foundation
import Testing

@testable import HydraTokenize

/// Images in Gemma's prompt, and how the token stream is cut on them.
@Suite("Gemma 4 vision prompt")
struct Gemma4VisionPromptTests {

    private let placeholder = 258_880

    private func render(_ turn: Gemma4Prompt.Turn) -> String {
        Gemma4Prompt.Renderer(thinking: false, instructions: nil).render(turns: [turn])
    }

    @Test("A picture is rendered ahead of the words that ask about it")
    func placeholderLeadsTheTurn() {
        let out = render(.user("What is this?", images: 1))
        #expect(out.contains("<|image|>What is this?"))
    }

    @Test("Two pictures give two placeholders, and none give none")
    func placeholderCount() {
        #expect(render(.user("Compare", images: 2))
            .components(separatedBy: "<|image|>").count - 1 == 2)
        #expect(!render(.user("Hello")).contains("<|image|>"))
    }

    /// Only the user's turns carry pictures, whatever is set on a model turn.
    @Test("A model turn renders no placeholder")
    func modelTurnsCarryNone() {
        let out = Gemma4Prompt.Renderer(thinking: false, instructions: nil)
            .render(turns: [Gemma4Prompt.Turn(role: .model, content: "I see a cat.", images: 3)])
        #expect(!out.contains("<|image|>"))
    }

    @Test("The stream is cut at each placeholder, which is dropped")
    func splitsAtThePlaceholder() {
        #expect(
            Gemma4Prompt.split(tokens: [1, 2, placeholder, 3], atPlaceholder: placeholder, images: 1)
                == [.text([1, 2]), .image(index: 0), .text([3])])
    }

    @Test("Two images stay in order and keep their indices")
    func twoImagesInOrder() {
        #expect(
            Gemma4Prompt.split(
                tokens: [placeholder, placeholder, 9], atPlaceholder: placeholder, images: 2)
                == [.image(index: 0), .image(index: 1), .text([9])])
    }

    /// A mismatch is refused rather than splicing one picture twice.
    @Test("A mismatch between placeholders and images is refused")
    func mismatchIsRefused() {
        #expect(Gemma4Prompt.split(
            tokens: [placeholder, placeholder], atPlaceholder: placeholder, images: 1) == nil)
        #expect(Gemma4Prompt.split(tokens: [1], atPlaceholder: placeholder, images: 1) == nil)
    }

    @Test("A text-only stream comes back as one run")
    func textOnly() {
        #expect(Gemma4Prompt.split(tokens: [1, 2], atPlaceholder: placeholder, images: 0)
            == [.text([1, 2])])
    }
}
