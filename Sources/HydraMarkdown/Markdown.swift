import Foundation

// Analyse du Markdown produit par le modèle, séparée de son rendu.
//
// GPT-OSS écrit spontanément en Markdown — titres, listes, gras — et glisse du LaTeX
// dans les passages techniques. Affiché brut, cela donne une bouillie de dièses,
// d'astérisques et d'antislashs.
//
// `AttributedString(markdown:)` d'Apple ne traite que le niveau *inline* : il ignore les
// titres, les listes et les blocs de code, et bute sur les antislashs du LaTeX. On
// découpe donc soi-même.
//
// Cette logique vit dans son propre module parce qu'elle est purement textuelle, donc
// testable sans interface — et elle a exactement le genre de cas limites qui se cassent
// en silence : marqueur de fin manquant sur un texte en cours de génération, accolades
// imbriquées, symbole dont le nom est le préfixe d'un autre.

// MARK: - Découpage en blocs

public enum MarkdownBlock {
    case heading(level: Int, text: String)
    case paragraph(String)
    case list(items: [String], ordered: Bool)
    case code(language: String?, content: String)
    case math(String)
    case rule

    public static func parse(_ source: String) -> [MarkdownBlock] {
        var blocks: [MarkdownBlock] = []
        var paragraph: [String] = []
        var listItems: [String] = []
        var listOrdered = false

        func flushParagraph() {
            let text = paragraph.joined(separator: " ").trimmingCharacters(in: .whitespaces)
            paragraph.removeAll()
            if !text.isEmpty { blocks.append(.paragraph(text)) }
        }
        func flushList() {
            if !listItems.isEmpty {
                blocks.append(.list(items: listItems, ordered: listOrdered))
                listItems.removeAll()
            }
        }
        func flushAll() {
            flushParagraph()
            flushList()
        }

        var lines = source.components(separatedBy: .newlines)[...]
        while let raw = lines.first {
            lines = lines.dropFirst()
            let line = raw.trimmingCharacters(in: .whitespaces)

            // Bloc de code : on avale jusqu'à la clôture, sans rien interpréter.
            if line.hasPrefix("```") {
                flushAll()
                let language = String(line.dropFirst(3)).trimmingCharacters(in: .whitespaces)
                var body: [String] = []
                while let next = lines.first,
                    !next.trimmingCharacters(in: .whitespaces).hasPrefix("```")
                {
                    body.append(next)
                    lines = lines.dropFirst()
                }
                if !lines.isEmpty { lines = lines.dropFirst() }  // ligne de clôture
                blocks.append(.code(
                    language: language.isEmpty ? nil : language,
                    content: body.joined(separator: "\n")))
                continue
            }

            if line.isEmpty {
                flushAll()
                continue
            }

            // Filet horizontal : au moins trois tirets, rien d'autre.
            if line.allSatisfy({ $0 == "-" }) && line.count >= 3 {
                flushAll()
                blocks.append(.rule)
                continue
            }

            // Maths en bloc : \[ … \] ou $$ … $$
            if line.hasPrefix("\\[") || line.hasPrefix("$$") {
                flushAll()
                let opener = line.hasPrefix("$$") ? "$$" : "\\["
                let closer = opener == "$$" ? "$$" : "\\]"
                var body = String(line.dropFirst(opener.count))
                while !body.contains(closer), let next = lines.first {
                    body += "\n" + next
                    lines = lines.dropFirst()
                }
                if let range = body.range(of: closer) { body = String(body[..<range.lowerBound]) }
                blocks.append(.math(body.trimmingCharacters(in: .whitespacesAndNewlines)))
                continue
            }

            if line.hasPrefix("#") {
                flushAll()
                let level = line.prefix(while: { $0 == "#" }).count
                let text = String(line.dropFirst(level)).trimmingCharacters(in: .whitespaces)
                blocks.append(.heading(level: min(level, 4), text: text))
                continue
            }

            if let item = bulletItem(line) {
                flushParagraph()
                if !listItems.isEmpty && listOrdered { flushList() }
                listOrdered = false
                listItems.append(item)
                continue
            }
            if let item = numberedItem(line) {
                flushParagraph()
                if !listItems.isEmpty && !listOrdered { flushList() }
                listOrdered = true
                listItems.append(item)
                continue
            }

            flushList()
            paragraph.append(line)
        }
        flushAll()
        return blocks
    }

    private static func bulletItem(_ line: String) -> String? {
        for marker in ["- ", "* ", "+ "] where line.hasPrefix(marker) {
            return String(line.dropFirst(marker.count))
        }
        return nil
    }

    private static func numberedItem(_ line: String) -> String? {
        let digits = line.prefix(while: \.isNumber)
        guard !digits.isEmpty else { return nil }
        let rest = line.dropFirst(digits.count)
        guard rest.hasPrefix(". ") || rest.hasPrefix(") ") else { return nil }
        return String(rest.dropFirst(2))
    }
}


// MARK: - LaTeX

/// Substitution des symboles LaTeX courants par leur équivalent Unicode.
///
/// Composer du LaTeX serait un projet en soi. Le remplacement de symboles couvre
/// l'essentiel de ce qu'écrit un modèle de langue — lettres grecques, opérateurs,
/// exposants — et transforme `\(1/\lambda^4\)` en `1/λ⁴`.
public enum LaTeX {

    private static let symbols: [String: String] = [
        "\\alpha": "α", "\\beta": "β", "\\gamma": "γ", "\\delta": "δ", "\\epsilon": "ε",
        "\\zeta": "ζ", "\\eta": "η", "\\theta": "θ", "\\iota": "ι", "\\kappa": "κ",
        "\\lambda": "λ", "\\mu": "μ", "\\nu": "ν", "\\xi": "ξ", "\\pi": "π",
        "\\rho": "ρ", "\\sigma": "σ", "\\tau": "τ", "\\upsilon": "υ", "\\phi": "φ",
        "\\chi": "χ", "\\psi": "ψ", "\\omega": "ω",
        "\\Gamma": "Γ", "\\Delta": "Δ", "\\Theta": "Θ", "\\Lambda": "Λ", "\\Xi": "Ξ",
        "\\Pi": "Π", "\\Sigma": "Σ", "\\Phi": "Φ", "\\Psi": "Ψ", "\\Omega": "Ω",
        "\\times": "×", "\\cdot": "·", "\\div": "÷", "\\pm": "±", "\\mp": "∓",
        "\\leq": "≤", "\\le": "≤", "\\geq": "≥", "\\ge": "≥", "\\neq": "≠", "\\ne": "≠",
        "\\approx": "≈", "\\equiv": "≡", "\\propto": "∝", "\\sim": "∼",
        "\\infty": "∞", "\\partial": "∂", "\\nabla": "∇", "\\sum": "∑", "\\prod": "∏",
        "\\int": "∫", "\\sqrt": "√", "\\in": "∈", "\\notin": "∉", "\\subset": "⊂",
        "\\cup": "∪", "\\cap": "∩", "\\forall": "∀", "\\exists": "∃",
        "\\rightarrow": "→", "\\to": "→", "\\leftarrow": "←", "\\Rightarrow": "⇒",
        "\\leftrightarrow": "↔", "\\ldots": "…", "\\dots": "…", "\\cdots": "⋯",
        "\\quad": "  ", "\\qquad": "    ", "\\,": " ", "\\;": " ", "\\!": "",
        "\\left": "", "\\right": "", "\\displaystyle": "", "\\text": "",
    ]

    private static let superscripts: [Character: Character] = [
        "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
        "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
        "+": "⁺", "-": "⁻", "n": "ⁿ", "i": "ⁱ",
    ]
    private static let subscripts: [Character: Character] = [
        "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄",
        "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉",
        "+": "₊", "-": "₋",
    ]

    public static func render(_ source: String) -> String {
        var text = source

        // \frac{a}{b} devient a/b — la barre horizontale n'existe pas en texte simple.
        while let range = text.range(of: "\\frac{") {
            guard let (numerator, afterNumerator) = braced(text, from: range.upperBound),
                text[afterNumerator...].hasPrefix("{"),
                let (denominator, afterDenominator) = braced(
                    text, from: text.index(after: afterNumerator))
            else { break }
            text.replaceSubrange(
                range.lowerBound..<afterDenominator,
                with: "(\(render(numerator)))/(\(render(denominator)))")
        }

        // Symboles, du plus long au plus court : `\le` ne doit pas amputer `\leq`.
        for key in symbols.keys.sorted(by: { $0.count > $1.count }) {
            text = text.replacingOccurrences(of: key, with: symbols[key]!)
        }

        text = script(text, marker: "^", table: superscripts)
        text = script(text, marker: "_", table: subscripts)
        text = text.replacingOccurrences(of: "{", with: "")
        text = text.replacingOccurrences(of: "}", with: "")
        return text.trimmingCharacters(in: .whitespaces)
    }

    /// Exposants et indices, y compris groupés entre accolades.
    private static func script(
        _ source: String, marker: Character, table: [Character: Character]
    ) -> String {
        var out = ""
        var index = source.startIndex
        while index < source.endIndex {
            guard source[index] == marker else {
                out.append(source[index])
                index = source.index(after: index)
                continue
            }
            let next = source.index(after: index)
            guard next < source.endIndex else { break }

            if source[next] == "{" {
                if let (group, after) = braced(source, from: source.index(after: next)),
                    group.allSatisfy({ table[$0] != nil })
                {
                    out += String(group.map { table[$0]! })
                    index = after
                    continue
                }
            } else if let replacement = table[source[next]] {
                out.append(replacement)
                index = source.index(after: next)
                continue
            }
            out.append(source[index])
            index = next
        }
        return out
    }

    /// Contenu jusqu'à l'accolade fermante correspondante, et l'indice qui la suit.
    private static func braced(
        _ source: String, from start: String.Index
    ) -> (String, String.Index)? {
        var depth = 1
        var index = start
        var content = ""
        while index < source.endIndex {
            let character = source[index]
            if character == "{" { depth += 1 }
            if character == "}" {
                depth -= 1
                if depth == 0 { return (content, source.index(after: index)) }
            }
            content.append(character)
            index = source.index(after: index)
        }
        return nil
    }
}
