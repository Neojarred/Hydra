import Foundation
import Testing

@testable import HydraMarkdown

/// The model writes Markdown and LaTeX as it goes. Parsing has to hold on **incomplete**
/// text — a missing closing marker mid-generation is the rule, not the exception — and
/// must never lose content.
struct MarkdownTests {

    // MARK: - Blocks

    @Test("Headings are recognized with their level")
    func headings() {
        let blocks = MarkdownBlock.parse("## Why is the sky blue?\n\n### 1. Rayleigh")
        #expect(blocks.count == 2)
        if case .heading(let level, let text) = blocks[0] {
            #expect(level == 2)
            #expect(text == "Why is the sky blue?")
        } else {
            Issue.record("first block: \(blocks[0])")
        }
        if case .heading(let level, _) = blocks[1] { #expect(level == 3) }
    }

    @Test("Bullet lists are grouped into a single block")
    func bulletList() {
        let blocks = MarkdownBlock.parse("- **Density**: small molecules\n- **Effect**: scattering")
        #expect(blocks.count == 1)
        if case .list(let items, let ordered) = blocks[0] {
            #expect(items.count == 2)
            #expect(!ordered)
            #expect(items[0] == "**Density**: small molecules")
        } else {
            Issue.record("block: \(blocks[0])")
        }
    }

    @Test("Numbered lists are distinguished from bullets")
    func numberedList() {
        let blocks = MarkdownBlock.parse("1. first\n2. second\n3) third")
        #expect(blocks.count == 1)
        if case .list(let items, let ordered) = blocks[0] {
            #expect(ordered)
            #expect(items == ["first", "second", "third"])
        }
    }

    @Test("Switching list type opens a new block")
    func listTypeSwitch() {
        let blocks = MarkdownBlock.parse("- bullet\n1. number")
        #expect(blocks.count == 2)
    }

    @Test("A code block is not interpreted")
    func codeBlock() {
        let source = "```swift\nlet x = **not bold**\n# not a heading\n```"
        let blocks = MarkdownBlock.parse(source)
        #expect(blocks.count == 1)
        if case .code(let language, let content) = blocks[0] {
            #expect(language == "swift")
            #expect(content == "let x = **not bold**\n# not a heading")
        } else {
            Issue.record("bloc : \(blocks[0])")
        }
    }

    /// While generating, a code block's closing fence has not arrived yet.
    @Test("An unterminated code block stays displayable")
    func unterminatedCodeBlock() {
        let blocks = MarkdownBlock.parse("```\nstill being written")
        #expect(blocks.count == 1)
        if case .code(_, let content) = blocks[0] {
            #expect(content == "still being written")
        } else {
            Issue.record("bloc : \(blocks[0])")
        }
    }

    @Test("Three dashes give a rule, not a list")
    func horizontalRule() {
        let blocks = MarkdownBlock.parse("text\n\n---\n\nmore")
        #expect(blocks.count == 3)
        if case .rule = blocks[1] {} else { Issue.record("block: \(blocks[1])") }
    }

    @Test("Display maths are extracted from their delimiters")
    func displayMath() {
        let blocks = MarkdownBlock.parse("\\[ I = I_0 e^{-\\sigma N} \\]")
        #expect(blocks.count == 1)
        if case .math(let content) = blocks[0] {
            #expect(content.contains("I_0"))
            #expect(!content.contains("\\["))
        } else {
            Issue.record("bloc : \(blocks[0])")
        }
    }

    @Test("The lines of a paragraph are joined")
    func paragraphJoining() {
        let blocks = MarkdownBlock.parse("first line\nsecond line\n\nanother paragraph")
        #expect(blocks.count == 2)
        if case .paragraph(let text) = blocks[0] {
            #expect(text == "first line second line")
        }
    }

    /// An end-to-end check: no content may disappear during the split.
    @Test("A complete model answer splits without loss")
    func realisticDocument() {
        let source = """
            ## Why is the sky blue?

            ### 1. Rayleigh scattering
            - **Air molecule density**: The atmosphere is made of molecules.
            - **Scattering effect**: it is proportional to \\(1/\\lambda^4\\).

            ---

            ### 2. Modelling
            \\[ I(\\lambda) = I_0 e^{-\\sigma N} \\]
            """
        let blocks = MarkdownBlock.parse(source)
        #expect(blocks.count == 6, "got \(blocks.count) blocks")

        // A word present in the source must be found in some block.
        var flattened = ""
        for block in blocks {
            switch block {
            case .heading(_, let t), .paragraph(let t), .math(let t): flattened += t
            case .list(let items, _): flattened += items.joined()
            case .code(_, let c): flattened += c
            case .rule: break
            }
        }
        #expect(flattened.contains("atmosphere") || flattened.contains("Atmosphere")
                || flattened.contains("The atmosphere"))
        #expect(flattened.contains("Modelling"))
    }

    // MARK: - LaTeX

    @Test("Greek letters become their symbol")
    func greekLetters() {
        #expect(LaTeX.render("\\lambda") == "λ")
        #expect(LaTeX.render("\\sigma \\theta \\Omega") == "σ θ Ω")
    }

    /// `\le` is a prefix of `\leq`: handling the symbols out of order would truncate the
    /// latter.
    @Test("A symbol that prefixes another is not truncated")
    func longestSymbolFirst() {
        #expect(LaTeX.render("\\leq") == "≤")
        #expect(LaTeX.render("\\le") == "≤")
        #expect(LaTeX.render("\\rightarrow") == "→")
    }

    @Test("Superscripts and subscripts use Unicode characters")
    func scripts() {
        #expect(LaTeX.render("\\lambda^4") == "λ⁴")
        #expect(LaTeX.render("I_0") == "I₀")
        #expect(LaTeX.render("x^{12}") == "x¹²")
        #expect(LaTeX.render("x^{-1}") == "x⁻¹")
    }

    @Test("Fractions become a readable division")
    func fractions() {
        #expect(LaTeX.render("\\frac{1}{\\lambda}") == "(1)/(λ)")
    }

    @Test("The case that motivated this renderer")
    func realisticFormula() {
        // Exactly what the model produced, and what was being shown raw.
        #expect(LaTeX.render("1/\\lambda^4") == "1/λ⁴")
        let rendered = LaTeX.render("I(\\lambda, \\theta) = I_0(\\lambda) \\, e^{- \\sigma(\\lambda) \\, N}")
        #expect(rendered.contains("λ"))
        #expect(rendered.contains("θ"))
        #expect(rendered.contains("σ"))
        #expect(!rendered.contains("\\"), "backslashes remain: \(rendered)")
    }

    @Test("Unknown LaTeX does not make the text disappear")
    func unknownCommandsSurvive() {
        let rendered = LaTeX.render("\\unknowncmd{value}")
        #expect(rendered.contains("value"))
    }
}
