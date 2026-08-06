import HydraMarkdown
import SwiftUI

/// Rendu du Markdown produit par le modèle, avec un traitement raisonnable des maths.
///
/// GPT-OSS écrit spontanément en Markdown — titres, listes, gras — et glisse du LaTeX
/// dans les passages techniques. Affiché brut, cela donne une bouillie de dièses,
/// d'astérisques et d'antislashs.
///
/// `AttributedString(markdown:)` d'Apple ne traite que le niveau *inline* : il ignore les
/// titres, les listes et les blocs de code, et bute sur les antislashs du LaTeX. On
/// découpe donc soi-même en blocs, et on rend chacun avec la vue qui lui convient.
///
/// **Le LaTeX n'est pas composé**, ce serait un projet en soi. Les symboles courants sont
/// remplacés par leur équivalent Unicode et le reste est présenté dans un style distinct :
/// `\(1/\lambda^4\)` devient *1/λ⁴*, ce qui est lisible, honnête, et sans commune mesure
/// avec l'affichage brut.
struct MarkdownView: View {
    let source: String

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(Array(MarkdownBlock.parse(source).enumerated()), id: \.offset) { _, block in
                view(for: block)
            }
        }
    }

    @ViewBuilder private func view(for block: MarkdownBlock) -> some View {
        switch block {
        case .heading(let level, let text):
            Text(InlineMarkdown.attributed(text))
                .font(headingFont(level))
                .padding(.top, level <= 2 ? 6 : 3)

        case .paragraph(let text):
            Text(InlineMarkdown.attributed(text))
                .fixedSize(horizontal: false, vertical: true)

        case .list(let items, let ordered):
            VStack(alignment: .leading, spacing: 4) {
                ForEach(Array(items.enumerated()), id: \.offset) { index, item in
                    HStack(alignment: .firstTextBaseline, spacing: 7) {
                        Text(ordered ? "\(index + 1)." : "•")
                            .font(.body.monospacedDigit())
                            .foregroundStyle(.secondary)
                            .frame(minWidth: ordered ? 18 : 10, alignment: .trailing)
                        Text(InlineMarkdown.attributed(item))
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
            .padding(.leading, 2)

        case .code(let language, let content):
            VStack(alignment: .leading, spacing: 3) {
                if let language, !language.isEmpty {
                    Text(language).font(.caption2).foregroundStyle(.secondary)
                }
                Text(content)
                    .font(.system(.callout, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(9)
                    .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 7))
            }

        case .math(let content):
            Text(LaTeX.render(content))
                .font(.system(.callout, design: .serif).italic())
                .frame(maxWidth: .infinity, alignment: .center)
                .padding(.vertical, 5)

        case .rule:
            Divider().padding(.vertical, 3)
        }
    }

    private func headingFont(_ level: Int) -> Font {
        switch level {
        case 1: return .title2.weight(.semibold)
        case 2: return .title3.weight(.semibold)
        case 3: return .headline
        default: return .subheadline.weight(.semibold)
        }
    }
}

// MARK: - Rendu inline

enum InlineMarkdown {

    /// Gras, italique, code et maths inline, en une passe.
    ///
    /// Écrit à la main plutôt que confié à `AttributedString(markdown:)` : celui-ci
    /// échoue sur les antislashs du LaTeX et sur les astérisques déséquilibrés, fréquents
    /// dans un texte en cours de génération.
    static func attributed(_ source: String) -> AttributedString {
        var out = AttributedString()
        var plain = ""
        var index = source.startIndex

        func flushPlain() {
            if !plain.isEmpty {
                out.append(AttributedString(plain))
                plain = ""
            }
        }

        while index < source.endIndex {
            let rest = source[index...]

            if let (content, next) = span(rest, opener: "**", closer: "**") {
                flushPlain()
                var piece = attributed(content)
                piece.inlinePresentationIntent = .stronglyEmphasized
                out.append(piece)
                index = next
                continue
            }
            if let (content, next) = span(rest, opener: "`", closer: "`") {
                flushPlain()
                var piece = AttributedString(content)
                piece.font = .system(.callout, design: .monospaced)
                out.append(piece)
                index = next
                continue
            }
            if let (content, next) = span(rest, opener: "\\(", closer: "\\)") {
                flushPlain()
                var piece = AttributedString(LaTeX.render(content))
                piece.font = .system(.body, design: .serif).italic()
                out.append(piece)
                index = next
                continue
            }
            if let (content, next) = span(rest, opener: "$", closer: "$") {
                flushPlain()
                var piece = AttributedString(LaTeX.render(content))
                piece.font = .system(.body, design: .serif).italic()
                out.append(piece)
                index = next
                continue
            }
            // Italique : une astérisque isolée, jamais une paire (déjà traitée).
            if rest.hasPrefix("*"), !rest.hasPrefix("**"),
                let (content, next) = span(rest, opener: "*", closer: "*")
            {
                flushPlain()
                var piece = attributed(content)
                piece.inlinePresentationIntent = .emphasized
                out.append(piece)
                index = next
                continue
            }

            plain.append(source[index])
            index = source.index(after: index)
        }
        flushPlain()
        return out
    }

    /// Contenu entre deux marqueurs, et l'indice qui suit le marqueur de fin.
    /// Rend `nil` si le marqueur de fin manque — un texte en cours de génération en
    /// contient forcément, et il doit rester affichable.
    private static func span(
        _ text: Substring, opener: String, closer: String
    ) -> (String, String.Index)? {
        guard text.hasPrefix(opener) else { return nil }
        let afterOpener = text.index(text.startIndex, offsetBy: opener.count)
        guard afterOpener < text.endIndex,
            let range = text[afterOpener...].range(of: closer)
        else { return nil }
        let content = String(text[afterOpener..<range.lowerBound])
        return content.isEmpty ? nil : (content, range.upperBound)
    }
}

