import Foundation
import Testing

@testable import HydraTokenize

/// Images in Qwen's prompt: where the placeholders go, and how the token stream is cut on them.
@Suite("Qwen vision prompt")
struct QwenVisionPromptTests {

    private let format = QwenFormat()

    @Test("A picture is rendered ahead of the words that ask about it")
    func placeholderLeadsTheTurn() {
        let rendered = format.render(
            turns: [ChatTurn(role: .user, content: "What is this?", images: 1)],
            settings: PromptSettings())
        #expect(rendered.contains("<|vision_start|><|image_pad|><|vision_end|>What is this?"))
    }

    @Test("Two pictures give two placeholders, and none give none")
    func placeholderCount() {
        let two = format.render(
            turns: [ChatTurn(role: .user, content: "Compare", images: 2)],
            settings: PromptSettings())
        #expect(two.components(separatedBy: "<|image_pad|>").count - 1 == 2)

        let none = format.render(
            turns: [ChatTurn(role: .user, content: "Hello")], settings: PromptSettings())
        #expect(!none.contains("<|vision_start|>"))
        #expect(!none.contains("<|image_pad|>"))
    }

    /// The assistant's own turns never carry pictures, whatever is set on them.
    @Test("Only the user's turns render placeholders")
    func assistantTurnsCarryNoImages() {
        let rendered = format.render(
            turns: [ChatTurn(role: .assistant, content: "I see a cat.", images: 3)],
            settings: PromptSettings())
        #expect(!rendered.contains("<|image_pad|>"))
    }

    // MARK: - Splitting

    private let pad = 248_056

    @Test("The stream is cut at each pad, and the pad itself is dropped")
    func splitsAtThePad() {
        let pieces = QwenFormat.split(
            tokens: [1, 2, pad, 3, 4, 5], atImagePad: pad, images: 1)
        #expect(pieces == [.text([1, 2]), .image(index: 0), .text([3, 4, 5])])
    }

    /// Dropping the pad is deliberate: it stands in for embeddings arriving from elsewhere, and
    /// keeping it would feed the model a spare token meaning "an image goes here" in the middle
    /// of the image it introduces.
    @Test("The pad does not survive into any text run")
    func padIsNotKept() {
        let pieces = QwenFormat.split(tokens: [pad, 7], atImagePad: pad, images: 1) ?? []
        for piece in pieces {
            if case .text(let tokens) = piece { #expect(!tokens.contains(pad)) }
        }
    }

    @Test("Two images, back to back, stay in order and keep their indices")
    func twoImagesInOrder() {
        let pieces = QwenFormat.split(
            tokens: [pad, pad, 9], atImagePad: pad, images: 2)
        #expect(pieces == [.image(index: 0), .image(index: 1), .text([9])])
    }

    @Test("A prompt that is only an image has no text runs")
    func imageOnly() {
        #expect(QwenFormat.split(tokens: [pad], atImagePad: pad, images: 1) == [.image(index: 0)])
    }

    /// A count mismatch is refused rather than patched up.
    ///
    /// Two pads and one image would otherwise splice the same picture twice, which is finite,
    /// plausible, and describes an image the user never sent.
    @Test("A mismatch between pads and images is refused")
    func mismatchIsRefused() {
        #expect(QwenFormat.split(tokens: [pad, pad], atImagePad: pad, images: 1) == nil)
        #expect(QwenFormat.split(tokens: [pad], atImagePad: pad, images: 2) == nil)
        #expect(QwenFormat.split(tokens: [1, 2], atImagePad: pad, images: 1) == nil)
    }

    @Test("A text-only stream comes back as one run")
    func textOnly() {
        #expect(QwenFormat.split(tokens: [1, 2, 3], atImagePad: pad, images: 0) == [.text([1, 2, 3])])
    }
}
