import HydraMarkdown
import SwiftUI

/// Renders the Markdown the model produces, with reasonable handling of maths.
///
/// GPT-OSS writes in Markdown of its own accord, headings, lists, bold, and slips LaTeX
/// into technical passages. Shown raw, that is a mess of hashes, asterisks and backslashes.
///
/// Apple's `AttributedString(markdown:)` handles the *inline* level only: it ignores
/// headings, lists and code blocks, and trips over LaTeX backslashes. So we split into
/// blocks ourselves, and render each with the view that suits it.
///
/// **LaTeX is not typeset**, that would be a project of its own. Common symbols are
/// replaced by their Unicode equivalent and the rest is presented in a distinct style:
/// `\(1/\lambda^4\)` becomes *1/λ⁴*, which is readable, honest, and nothing like the raw
/// display.
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

    /// Bold, italic, code and inline maths, in one pass.
    ///
    /// Written by hand rather than handed to `AttributedString(markdown:)`: that one fails
    /// on LaTeX backslashes and on unbalanced asterisks, both common in text that is still
    /// being generated.
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
            // Italic: a lone asterisk, never a pair (already handled above).
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

    /// The content between two markers, and the index just past the closing marker.
    /// Returns `nil` if the closing marker is missing, text still being generated is bound
    /// to contain such cases, and it must stay displayable.
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

