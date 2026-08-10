import Foundation
import HydraCore
import Testing

@testable import HydraTokenize

/// The seam between a conversation and a model's own prompt dialect.
///
/// Everything here guards one failure mode, and it is the one D-023 singles out as worth a
/// test: sending a model a prompt in another model's format raises nothing. Gemma fed Harmony
/// markers does not crash — it reads `<|start|>assistant<|channel|>` as ordinary text and
/// answers something plausible and worse. No assertion about tensors or shapes would catch it.
@Suite("Conversation formats")
struct ConversationFormatTests {

    private let turns: [ChatTurn] = [
        .user("what is the sky"), .assistant("blue"), .user("why"),
    ]

    /// A tokenizer carrying Gemma's real marker identifiers, as `Gemma4PromptTests` builds it.
    private func makeGemmaTokenizer() throws -> BPETokenizer {
        var vocabulary: [String: Int] = [:]
        var next = 300
        func add(_ piece: String) {
            vocabulary[piece] = next
            next += 1
        }
        for scalar in "abcdefghijklmnopqrstuvwxyz .,!?" { add(String(scalar)) }
        add("\u{2581}")
        add("thought")
        add("\n")
        for byte in 0...255 { vocabulary[String(format: "<0x%02X>", byte)] = 1000 + byte }

        var special: [String: Int] = [:]
        for marker in Gemma4Prompt.Marker.allCases { special[marker.rawValue] = marker.publishedID }

        return try BPETokenizer(
            vocabulary: vocabulary, merges: [], specialTokens: special,
            conventions: .gemma4)
    }

    @Test("Each architecture gets its own format")
    func dispatchIsCorrect() {
        #expect(ConversationFormats.format(for: .gptOss).name == "harmony")
        #expect(ConversationFormats.format(for: .gemma4).name == "gemma-4")

        // Every architecture must resolve. A missing case would be a compile error today, and
        // this keeps that true if the switch ever grows a `default`.
        for architecture in ModelArchitecture.allCases {
            #expect(!ConversationFormats.format(for: architecture).name.isEmpty)
        }
    }

    /// The whole point of the seam, stated as an assertion.
    @Test("The two formats produce genuinely different prompts")
    func formatsDoNotCollide() {
        let settings = PromptSettings(reasoning: .medium)
        let harmony = HarmonyFormat().render(turns: turns, settings: settings)
        let gemma = Gemma4Format().render(turns: turns, settings: settings)

        #expect(harmony != gemma)
        // Neither may contain the other's markers, which is what a wrong dispatch produces.
        #expect(!gemma.contains("<|start|>"))
        #expect(!gemma.contains("<|channel|>"))
        #expect(!harmony.contains(Gemma4Prompt.Marker.turnOpen.rawValue))

        // The same content reaches both, so a difference is the dialect and not lost text.
        #expect(harmony.contains("what is the sky") && gemma.contains("what is the sky"))
        #expect(harmony.contains("blue") && gemma.contains("blue"))
    }

    /// The adapters must not paraphrase: what goes through the seam is what the format itself
    /// produces, or the tests each format already has stop meaning anything.
    @Test("Rendering through the seam matches rendering directly")
    func renderingIsUnchanged() {
        let settings = PromptSettings(reasoning: .high, instructions: "be terse")

        let direct = Harmony.Renderer(reasoningEffort: .high, instructions: "be terse")
            .render(turns: [
                Harmony.Turn(role: .user, content: "what is the sky"),
                Harmony.Turn(role: .assistant, content: "blue"),
                Harmony.Turn(role: .user, content: "why"),
            ])
        #expect(HarmonyFormat().render(turns: turns, settings: settings) == direct)

        let gemmaDirect = Gemma4Prompt.Renderer(thinking: true, instructions: "be terse")
            .render(turns: [
                Gemma4Prompt.Turn(role: .user, content: "what is the sky"),
                Gemma4Prompt.Turn(role: .model, content: "blue"),
                Gemma4Prompt.Turn(role: .user, content: "why"),
            ])
        #expect(Gemma4Format().render(turns: turns, settings: settings) == gemmaDirect)
    }

    /// `assistant` and `model` name the same speaker and neither format may be handed the
    /// other's word. The neutral turn is what makes that impossible to get wrong.
    @Test("The assistant is named by each format's own convention")
    func rolesAreTranslated() {
        let rendered = Gemma4Format().render(
            turns: [.assistant("hi")], settings: PromptSettings(reasoning: .off))
        #expect(rendered.contains("model"))
        #expect(!rendered.contains("assistant"))
    }

    /// Reasoning `off` is the case the two formats genuinely disagree on, so it is asserted
    /// rather than assumed: Gemma can be told not to think, GPT-OSS cannot.
    @Test("Reasoning off closes Gemma's channel and only lowers Harmony's effort")
    func reasoningOffIsHonest() {
        let off = PromptSettings(reasoning: .off)

        let thinking = Gemma4Format().render(
            turns: turns, settings: PromptSettings(reasoning: .medium))
        #expect(Gemma4Format().render(turns: turns, settings: off) != thinking)

        #expect(HarmonyFormat.effort(.off) == .low)
        #expect(HarmonyFormat.effort(.high) == .high)
        // GPT-OSS still reasons, so the prompt must still be a valid Harmony prompt.
        #expect(HarmonyFormat().render(turns: turns, settings: off).contains("<|start|>"))
    }

    /// The conventions a vocabulary is keyed by are per-model, and nothing in the file says
    /// which. This is the assertion that would have caught the real bug.
    ///
    /// A convenience initializer used to default to `.gptOss`, so an installed Gemma tokenizer
    /// loaded with GPT-2 byte-level encoding: every space became `Ġ` rather than `▁`,
    /// `▁capital` never formed, and the model was handed a sequence no Gemma has ever seen. It
    /// answered fluently and wrongly. The default is gone — the type no longer compiles without
    /// a choice — and this pins the choice itself.
    @Test("Each architecture keys its vocabulary its own way")
    func conventionsArePerArchitecture() throws {
        #expect(BPETokenizer.Conventions.for(.gemma4) == .gemma4)
        #expect(BPETokenizer.Conventions.for(.gptOss) == .gptOss)
        #expect(BPETokenizer.Conventions.for(.gemma4) != .gptOss)

        // The difference is not a label: it changes what a space becomes.
        #expect(BPETokenizer.Conventions.for(.gemma4).encoding == .metaSpaceWithByteFallback)
        #expect(BPETokenizer.Conventions.for(.gptOss).encoding == .byteLevel)

        // End to end, on the vocabulary shape that actually diverges. `▁capital` exists as one
        // piece; under byte-level conventions the space becomes `Ġ` and it can never form.
        var vocabulary: [String: Int] = ["\u{2581}": 1, "capital": 2, "\u{2581}capital": 3]
        // Single characters, as a real vocabulary has them: without these the merges never
        // start and byte fallback answers instead, which is a different code path.
        var next = 10
        for character in "capital" where vocabulary[String(character)] == nil {
            vocabulary[String(character)] = next
            next += 1
        }
        for byte in 0...255 { vocabulary[String(format: "<0x%02X>", byte)] = 1000 + byte }

        let merges: [(String, String)] = [
            ("c", "a"), ("ca", "p"), ("cap", "i"), ("capi", "t"), ("capit", "a"),
            ("capita", "l"), ("\u{2581}", "capital"),
        ]
        let gemma = try BPETokenizer(
            vocabulary: vocabulary, merges: merges, specialTokens: [:],
            conventions: .for(.gemma4))
        #expect(gemma.encode(" capital") == [3], "the metaspace merge did not form")

        // The same vocabulary under GPT-OSS's conventions cannot reach it: the space becomes
        // `Ġ`, which is not a piece here at all. That asymmetry is the bug, reproduced.
        let wrong = try BPETokenizer(
            vocabulary: vocabulary, merges: merges, specialTokens: [:],
            conventions: .for(.gptOss))
        #expect(wrong.encode(" capital") != [3])
    }

    /// Thinking is a switch for Gemma, and the prompt is what throws it.
    ///
    /// The published template stops at `<|turn>model` and expects the model to write
    /// `<|channel>thought` itself. Measured on real weights it does not: asked what follows a
    /// bare `<|channel>`, its best token is « eyes» at a logit of 5.1, against «The» at 26.7
    /// once the header is seeded. So the renderer writes the header, and the parser has to be
    /// told the prompt already opened the channel — otherwise the whole of the reasoning is
    /// filed as the answer, which is what it did.
    @Test("Thinking opens the channel in the prompt, and the parser is told")
    func thinkingSeedsTheChannel() throws {
        let open = Gemma4Format().render(
            turns: [.user("hi")], settings: PromptSettings(reasoning: .low))
        let closed = Gemma4Format().render(
            turns: [.user("hi")], settings: PromptSettings(reasoning: .off))

        let header = Gemma4Prompt.Marker.channelOpen.rawValue + "thought\n"
        #expect(open.hasSuffix(header), "thinking must leave the thought channel open")
        #expect(closed.hasSuffix(header + Gemma4Prompt.Marker.channelClose.rawValue),
                "not thinking must close it immediately")

        // And the parser starts inside the channel, so the first thing generated is reasoning.
        let tokenizer = try makeGemmaTokenizer()
        let thinking = Gemma4Format().makeParser(
            tokenizer: tokenizer, settings: PromptSettings(reasoning: .low))
        var events: [PromptEvent] = []
        for token in tokenizer.encode("let me see") {
            events += thinking.consume(token)
        }
        #expect(events.allSatisfy {
            if case .reasoning = $0 { return true }
            return false
        }, "reasoning was filed as the answer")

        // With thinking off the same tokens are the answer.
        let direct = Gemma4Format().makeParser(
            tokenizer: tokenizer, settings: PromptSettings(reasoning: .off))
        var answers: [PromptEvent] = []
        for token in tokenizer.encode("the answer") { answers += direct.consume(token) }
        #expect(answers.contains {
            if case .answer = $0 { return true }
            return false
        })
    }

    /// The adapter owns a session where the format's parser takes one `inout`. A session that
    /// failed to advance would report nothing and finish never, so both are checked.
    @Test("The Gemma parser adapter separates answer from reasoning")
    func gemmaParserAdapterWorks() throws {
        let tokenizer = try makeGemmaTokenizer()
        let parser = Gemma4Format().makeParser(
            tokenizer: tokenizer, settings: PromptSettings(reasoning: .off))

        var tokens = [Gemma4Prompt.Marker.channelOpen.publishedID]
        tokens += tokenizer.encode("thought\n")
        tokens += tokenizer.encode("let me see")
        tokens.append(Gemma4Prompt.Marker.channelClose.publishedID)
        tokens += tokenizer.encode("the answer")
        tokens.append(Gemma4Prompt.Marker.turnClose.publishedID)

        var answer = ""
        var reasoning = ""
        for token in tokens {
            for event in parser.consume(token) {
                switch event {
                case .answer(let text): answer += text
                case .reasoning(let text): reasoning += text
                case .stopped: break
                }
            }
        }

        #expect(reasoning.contains("let me see"))
        #expect(answer.contains("the answer"))
        // The reasoning must not have leaked into the answer, which is the bug the channel
        // name accumulation was written to fix.
        #expect(!answer.contains("let me see"))
        #expect(parser.isFinished)

        // A finished parser consumes nothing more.
        #expect(parser.consume(tokenizer.encode("more")[0]).isEmpty)
    }
}
