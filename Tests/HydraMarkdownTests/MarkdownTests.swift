import Foundation
import Testing

@testable import HydraMarkdown

/// Le modèle écrit du Markdown et du LaTeX en continu. L'analyse doit tenir sur du texte
/// **incomplet** — un marqueur de fin manquant au milieu d'une génération est la règle,
/// pas l'exception — et ne jamais perdre de contenu.
struct MarkdownTests {

    // MARK: - Blocs

    @Test("Les titres sont reconnus avec leur niveau")
    func headings() {
        let blocks = MarkdownBlock.parse("## Pourquoi le ciel est bleu ?\n\n### 1. Rayleigh")
        #expect(blocks.count == 2)
        if case .heading(let level, let text) = blocks[0] {
            #expect(level == 2)
            #expect(text == "Pourquoi le ciel est bleu ?")
        } else {
            Issue.record("premier bloc : \(blocks[0])")
        }
        if case .heading(let level, _) = blocks[1] { #expect(level == 3) }
    }

    @Test("Les listes à puces sont regroupées en un seul bloc")
    func bulletList() {
        let blocks = MarkdownBlock.parse("- **Densité** : petites molécules\n- **Effet** : diffusion")
        #expect(blocks.count == 1)
        if case .list(let items, let ordered) = blocks[0] {
            #expect(items.count == 2)
            #expect(!ordered)
            #expect(items[0] == "**Densité** : petites molécules")
        } else {
            Issue.record("bloc : \(blocks[0])")
        }
    }

    @Test("Les listes numérotées sont distinguées des puces")
    func numberedList() {
        let blocks = MarkdownBlock.parse("1. premier\n2. deuxième\n3) troisième")
        #expect(blocks.count == 1)
        if case .list(let items, let ordered) = blocks[0] {
            #expect(ordered)
            #expect(items == ["premier", "deuxième", "troisième"])
        }
    }

    @Test("Changer de type de liste ouvre un nouveau bloc")
    func listTypeSwitch() {
        let blocks = MarkdownBlock.parse("- puce\n1. numéro")
        #expect(blocks.count == 2)
    }

    @Test("Un bloc de code n'est pas interprété")
    func codeBlock() {
        let source = "```swift\nlet x = **pas du gras**\n# pas un titre\n```"
        let blocks = MarkdownBlock.parse(source)
        #expect(blocks.count == 1)
        if case .code(let language, let content) = blocks[0] {
            #expect(language == "swift")
            #expect(content == "let x = **pas du gras**\n# pas un titre")
        } else {
            Issue.record("bloc : \(blocks[0])")
        }
    }

    /// Pendant la génération, la clôture d'un bloc de code n'est pas encore arrivée.
    @Test("Un bloc de code non clos reste affichable")
    func unterminatedCodeBlock() {
        let blocks = MarkdownBlock.parse("```\nen cours d'écriture")
        #expect(blocks.count == 1)
        if case .code(_, let content) = blocks[0] {
            #expect(content == "en cours d'écriture")
        } else {
            Issue.record("bloc : \(blocks[0])")
        }
    }

    @Test("Trois tirets donnent un filet, pas une liste")
    func horizontalRule() {
        let blocks = MarkdownBlock.parse("texte\n\n---\n\nsuite")
        #expect(blocks.count == 3)
        if case .rule = blocks[1] {} else { Issue.record("bloc : \(blocks[1])") }
    }

    @Test("Les maths en bloc sont extraites de leurs délimiteurs")
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

    @Test("Les lignes d'un paragraphe sont réunies")
    func paragraphJoining() {
        let blocks = MarkdownBlock.parse("première ligne\ndeuxième ligne\n\nautre paragraphe")
        #expect(blocks.count == 2)
        if case .paragraph(let text) = blocks[0] {
            #expect(text == "première ligne deuxième ligne")
        }
    }

    /// Contrôle d'ensemble : aucun contenu ne doit disparaître au découpage.
    @Test("Une réponse complète du modèle se découpe sans perte")
    func realisticDocument() {
        let source = """
            ## Pourquoi le ciel est bleu ?

            ### 1. La diffusion de Rayleigh
            - **Densité des molécules d'air** : L'atmosphère est composée de molécules.
            - **Effet de diffusion** : elle est proportionnelle à \\(1/\\lambda^4\\).

            ---

            ### 2. Modélisation
            \\[ I(\\lambda) = I_0 e^{-\\sigma N} \\]
            """
        let blocks = MarkdownBlock.parse(source)
        #expect(blocks.count == 6, "obtenu \(blocks.count) blocs")

        // Un mot présent dans la source doit se retrouver dans un bloc.
        var flattened = ""
        for block in blocks {
            switch block {
            case .heading(_, let t), .paragraph(let t), .math(let t): flattened += t
            case .list(let items, _): flattened += items.joined()
            case .code(_, let c): flattened += c
            case .rule: break
            }
        }
        #expect(flattened.contains("atmosphère".capitalized) || flattened.contains("atmosphère")
                || flattened.contains("L'atmosphère"))
        #expect(flattened.contains("Modélisation"))
    }

    // MARK: - LaTeX

    @Test("Les lettres grecques deviennent leur symbole")
    func greekLetters() {
        #expect(LaTeX.render("\\lambda") == "λ")
        #expect(LaTeX.render("\\sigma \\theta \\Omega") == "σ θ Ω")
    }

    /// `\le` est un préfixe de `\leq` : traiter les symboles dans le désordre amputerait
    /// le second.
    @Test("Un symbole préfixe d'un autre n'est pas amputé")
    func longestSymbolFirst() {
        #expect(LaTeX.render("\\leq") == "≤")
        #expect(LaTeX.render("\\le") == "≤")
        #expect(LaTeX.render("\\rightarrow") == "→")
    }

    @Test("Les exposants et indices utilisent les caractères Unicode")
    func scripts() {
        #expect(LaTeX.render("\\lambda^4") == "λ⁴")
        #expect(LaTeX.render("I_0") == "I₀")
        #expect(LaTeX.render("x^{12}") == "x¹²")
        #expect(LaTeX.render("x^{-1}") == "x⁻¹")
    }

    @Test("Les fractions deviennent une division lisible")
    func fractions() {
        #expect(LaTeX.render("\\frac{1}{\\lambda}") == "(1)/(λ)")
    }

    @Test("Le cas qui a motivé ce rendu")
    func realisticFormula() {
        // Exactement ce que le modèle a produit, et qui s'affichait brut.
        #expect(LaTeX.render("1/\\lambda^4") == "1/λ⁴")
        let rendered = LaTeX.render("I(\\lambda, \\theta) = I_0(\\lambda) \\, e^{- \\sigma(\\lambda) \\, N}")
        #expect(rendered.contains("λ"))
        #expect(rendered.contains("θ"))
        #expect(rendered.contains("σ"))
        #expect(!rendered.contains("\\"), "des antislashs subsistent : \(rendered)")
    }

    @Test("Un LaTeX inconnu ne fait pas disparaître le texte")
    func unknownCommandsSurvive() {
        let rendered = LaTeX.render("\\unknowncmd{valeur}")
        #expect(rendered.contains("valeur"))
    }
}
