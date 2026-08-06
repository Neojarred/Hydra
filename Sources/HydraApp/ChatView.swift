import AppKit
import HydraTokenize
import SwiftUI
import UniformTypeIdentifiers

// MARK: - Jauge de contexte

/// Petit anneau qui se remplit à mesure que le contexte se consomme, du bleu au rouge.
///
/// Le contexte est une ressource silencieuse : rien ne prévient qu'on en approche la
/// limite, et le dépassement se traduit par une erreur ou une conversation tronquée. Un
/// indicateur permanent évite la surprise.
struct ContextRing: View {
    let used: Int
    let capacity: Int
    /// Faux tant qu'aucun modèle n'est chargé : l'anneau montre alors la capacité prévue,
    /// en retrait, plutôt que de laisser croire à une mesure.
    var isLive: Bool = true

    private var fraction: Double {
        capacity > 0 ? min(1, Double(used) / Double(capacity)) : 0
    }
    private var color: Color {
        // Bleu jusqu'à la moitié, puis dérive vers l'orange et le rouge.
        Color(hue: 0.58 * (1 - min(1, fraction * 1.15)), saturation: 0.85, brightness: 0.85)
    }

    var body: some View {
        HStack(spacing: 5) {
            ZStack {
                Circle().stroke(Color.secondary.opacity(0.2), lineWidth: 2.5)
                Circle()
                    .trim(from: 0, to: fraction)
                    .stroke(color, style: StrokeStyle(lineWidth: 2.5, lineCap: .round))
                    .rotationEffect(.degrees(-90))
            }
            .frame(width: 14, height: 14)
            .animation(.easeOut(duration: 0.3), value: fraction)

            Text("\(used) / \(capacity / 1024)k")
                .font(.caption2.monospacedDigit())
                .foregroundStyle(fraction > 0.85 ? color : .secondary)
        }
        .opacity(isLive ? 1 : 0.45)
        .help(isLive
              ? "Contexte : \(used) jetons occupés sur \(capacity)"
              : "Contexte prévu : \(capacity) jetons — chargez un modèle pour le mesurer")
    }
}

// MARK: - Message

struct MessageRow: View {
    @Bindable var model: AppModel
    let message: Message
    let isGenerating: Bool

    @State private var showReasoning = false
    @State private var isEditing = false
    @State private var draft = ""
    @State private var hovering = false
    @State private var copied = false

    private var isUser: Bool { message.role == .user }

    var body: some View {
        VStack(alignment: isUser ? .trailing : .leading, spacing: 4) {
            if !message.reasoning.isEmpty {
                reasoningPanel
            }

            if isEditing {
                editor
            } else {
                bubble
            }

            if !message.attachments.isEmpty {
                attachmentChips
            }

            metadata
        }
        .frame(maxWidth: .infinity, alignment: isUser ? .trailing : .leading)
        .onHover { hovering = $0 }
    }

    private var reasoningPanel: some View {
        DisclosureGroup(isExpanded: $showReasoning) {
            Text(message.reasoning)
                .font(.caption.italic())
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.top, 4)
        } label: {
            Label("Raisonnement", systemImage: "brain")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var bubble: some View {
        Group {
            if isUser {
                Text(message.text)
                    .textSelection(.enabled)
            } else if message.text.isEmpty && isGenerating {
                HStack(spacing: 6) {
                    ProgressView().controlSize(.small)
                    Text("réflexion…").font(.callout).foregroundStyle(.secondary)
                }
            } else if isGenerating {
                // Pendant la génération, texte brut.
                //
                // `MarkdownView` reparse la totalité du message à chaque rendu. Appelé à
                // chaque fragment sur un message qui grandit, le coût est quadratique en
                // sa longueur — et il s'ajoute au décodage au lieu de s'y superposer.
                // La mise en forme apparaît à la fin, une fois pour toutes.
                Text(message.text)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                // Le modèle écrit en Markdown, avec du LaTeX dans les passages techniques.
                MarkdownView(source: message.text)
                    .textSelection(.enabled)
            }
        }
        .padding(.horizontal, 13).padding(.vertical, 10)
        .background(
            isUser
                ? AnyShapeStyle(Color.accentColor.opacity(0.16))
                : AnyShapeStyle(.quaternary.opacity(0.45)),
            in: RoundedRectangle(cornerRadius: 13))
        // L'alignement doit suivre le côté de la bulle.
        //
        // Avec `.leading`, le message de l'utilisateur se collait à gauche d'une boîte de
        // 520 points, elle-même alignée à droite : la bulle flottait donc 520 points avant
        // le bord, sans jamais l'atteindre.
        .frame(maxWidth: isUser ? 520 : .infinity, alignment: isUser ? .trailing : .leading)
    }

    private var editor: some View {
        VStack(alignment: .trailing, spacing: 6) {
            TextEditor(text: $draft)
                .font(.body)
                .frame(minHeight: 70)
                .padding(6)
                .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 9))
            HStack(spacing: 8) {
                Button("Annuler") { isEditing = false }
                    .buttonStyle(.bordered).controlSize(.small)
                Button("Valider et régénérer") {
                    isEditing = false
                    model.editUserMessage(message.id, to: draft)
                }
                .buttonStyle(.borderedProminent).controlSize(.small)
                .disabled(draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .frame(maxWidth: 520)
    }

    private var attachmentChips: some View {
        HStack(spacing: 6) {
            ForEach(message.attachments) { attachment in
                Label(attachment.name, systemImage: "doc.text")
                    .font(.caption2)
                    .padding(.horizontal, 7).padding(.vertical, 3)
                    .background(.quaternary.opacity(0.5), in: Capsule())
            }
        }
    }

    /// Métriques et actions. Les métriques restent visibles ; les actions n'apparaissent
    /// qu'au survol, pour ne pas encombrer la lecture.
    private var metadata: some View {
        HStack(spacing: 10) {
            if isUser { Spacer(minLength: 0) }

            if message.hasSeveralVariants { variantSelector }

            if let tokens = message.current.outputTokens {
                metric("\(tokens)", icon: "number")
                    .help("\(tokens) jetons produits")
            }
            if let ttft = message.current.timeToFirstToken {
                metric(String(format: "%.1f s", ttft), icon: "timer")
                    .help("Temps jusqu'au premier jeton, prefill compris")
            }
            if let rate = message.current.tokensPerSecond {
                metric(String(format: "%.1f/s", rate), icon: "speedometer")
                    .help("Jetons produits par seconde")
            }

            // Les actions restent visibles et cliquables en permanence ; le survol ne fait
            // que les rendre franches.
            //
            // Les lier au survol les a fait échouer deux fois : d'abord parce que les
            // insérer décalait la ligne sous le pointeur, ensuite parce qu'une cible
            // conditionnelle reste une cible qu'on doit atteindre avant qu'elle ne change
            // d'avis. Un bouton toujours présent n'a pas ce problème.
            if !isGenerating {
                actions.opacity(hovering ? 1 : 0.4)
            }

            if !isUser { Spacer(minLength: 0) }
        }
        .frame(minHeight: 22)
        .animation(.easeOut(duration: 0.12), value: hovering)
    }

    /// Une icône plutôt qu'un intitulé : « 128 jetons · 1,4 s avant réponse · 7,5 jetons/s »
    /// répétait le mot « jetons » deux fois par message et occupait toute la largeur. Le
    /// symbole porte le sens, l'infobulle donne la phrase complète.
    private func metric(_ text: String, icon: String) -> some View {
        HStack(spacing: 3) {
            Image(systemName: icon).imageScale(.small)
            Text(text)
        }
        .font(.caption.monospacedDigit())
        .foregroundStyle(.secondary)
    }

    private var variantSelector: some View {
        HStack(spacing: 3) {
            Button {
                model.selectVariant(message.id, index: message.activeVariant - 1)
            } label: {
                Image(systemName: "chevron.left")
            }
            .disabled(message.activeVariant == 0)
            .help("Réponse précédente")

            Text("\(message.activeVariant + 1)/\(message.variants.count)")
                .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                .help("\(message.variants.count) réponses générées pour ce message")

            Button {
                model.selectVariant(message.id, index: message.activeVariant + 1)
            } label: {
                Image(systemName: "chevron.right")
            }
            .disabled(message.activeVariant == message.variants.count - 1)
            .help("Réponse suivante")
        }
        .buttonStyle(.borderless)
        .font(.caption)
    }

    private var actions: some View {
        HStack(spacing: 2) {
            actionButton(
                icon: copied ? "checkmark" : "doc.on.doc",
                help: copied ? "Copié" : "Copier le message"
            ) {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(message.text, forType: .string)
                copied = true
                DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) { copied = false }
            }

            if isUser {
                actionButton(
                    icon: "pencil", help: "Modifier la question et régénérer la réponse",
                    disabled: model.loaded == nil
                ) {
                    draft = message.text
                    isEditing = true
                }
            } else {
                actionButton(
                    icon: "arrow.clockwise",
                    help: "Régénérer — la réponse actuelle est conservée",
                    disabled: model.loaded == nil
                ) {
                    model.regenerate(message.id)
                }
            }

            actionButton(
                icon: "trash",
                help: isUser ? "Supprimer la question et sa réponse" : "Supprimer la réponse"
            ) {
                model.deleteMessage(message.id)
            }
        }
        .foregroundStyle(.secondary)
    }

    /// Cible de clic explicite.
    ///
    /// Un `Button` dont l'étiquette est une simple icône n'offre que les quelques points
    /// du glyphe, et le style sans bordure ne l'élargit pas. Le cadre et `contentShape`
    /// donnent une zone franche, indépendante de la forme du symbole.
    private func actionButton(
        icon: String, help: String, disabled: Bool = false, action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            Image(systemName: icon)
                .font(.callout)
                .frame(width: 24, height: 22)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(disabled)
        .help(help)
    }
}

// MARK: - Vue de conversation

struct ChatView: View {
    /// Largeur et marges communes à la transcription et au composeur.
    ///
    /// Les deux avaient leurs propres valeurs — 760 et 22 d'un côté, 800 et 16 de
    /// l'autre — et les marges s'appliquant avant le plafond, les colonnes utiles
    /// tombaient à 716 et 768. Les bulles ne s'alignaient donc sur rien, et le décalage
    /// laissait un blanc à droite.
    static let columnWidth: CGFloat = 860
    static let columnPadding: CGFloat = 20

    @Bindable var model: AppModel
    @State private var draft = ""
    @State private var attachments: [Message.Attachment] = []
    @State private var showingImporter = false

    var body: some View {
        VStack(spacing: 0) {
            if let conversation = model.current {
                transcript(conversation)
                Divider()
                composer(conversation)
            } else {
                ContentUnavailableView(
                    "Aucune conversation", systemImage: "bubble.left.and.bubble.right")
            }
        }
        // Minimum porté par la colonne de détail plutôt que par la racine du split view.
        // La taille idéale doit être donnée elle aussi : sans elle, la fenêtre s'ouvre à
        // sa largeur minimale et à la hauteur de l'écran.
        .frame(minWidth: 520, idealWidth: 900, minHeight: 360, idealHeight: 720)
        .navigationTitle(model.current?.title ?? "Hydra")
        .fileImporter(
            isPresented: $showingImporter,
            allowedContentTypes: [.plainText, .sourceCode, .json, .yaml, .commaSeparatedText],
            allowsMultipleSelection: true
        ) { result in
            if case .success(let urls) = result { attach(urls) }
        }
    }

    private func transcript(_ conversation: Conversation) -> some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 18) {
                    if conversation.messages.isEmpty { emptyState }
                    ForEach(conversation.messages) { message in
                        MessageRow(
                            model: model, message: message,
                            isGenerating: model.generatingMessage == message.id)
                            .id(message.id)
                    }
                }
                .padding(.horizontal, Self.columnPadding).padding(.vertical, 20)
                // Colonne centrée : sur une fenêtre large, du texte pleine largeur
                // devient pénible à lire.
                .frame(maxWidth: Self.columnWidth)
                .frame(maxWidth: .infinity)
            }
            // Une ScrollView annonce comme hauteur idéale celle de son contenu. Une longue
            // conversation demandait ainsi une fenêtre de plusieurs milliers de points de
            // haut — bien au-delà de l'écran. La hauteur idéale est donc fixée ici ; la
            // vue reste libre de s'étendre, elle ne dicte plus la taille de la fenêtre.
            .frame(idealHeight: 420)
            .onChange(of: conversation.messages.last?.text) { _, _ in
                // Sans animation : elle se relance à chaque rafraîchissement et une
                // animation interrompue vingt fois par seconde coûte plus qu'elle n'apporte.
                if let last = conversation.messages.last {
                    proxy.scrollTo(last.id, anchor: .bottom)
                }
            }
        }
    }

    private var emptyState: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(model.loaded == nil
                 ? "Chargez un modèle pour commencer."
                 : "Posez votre question.")
                .foregroundStyle(.secondary)
            if let loaded = model.loaded {
                Text("\(loaded.entry.displayName) · contexte \(loaded.contextLength / 1024)k "
                     + "· \(loaded.slotsPerLayer) experts en cache par couche")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func composer(_ conversation: Conversation) -> some View {
        VStack(spacing: 8) {
            settingsRow(conversation)

            if !attachments.isEmpty {
                HStack(spacing: 6) {
                    ForEach(attachments) { attachment in
                        HStack(spacing: 4) {
                            Image(systemName: "doc.text")
                            Text(attachment.name).lineLimit(1)
                            Button {
                                attachments.removeAll { $0.id == attachment.id }
                            } label: {
                                Image(systemName: "xmark")
                            }
                            .buttonStyle(.borderless)
                        }
                        .font(.caption2)
                        .padding(.horizontal, 7).padding(.vertical, 3)
                        .background(.quaternary.opacity(0.5), in: Capsule())
                    }
                    Spacer()
                }
            }

            HStack(alignment: .bottom, spacing: 8) {
                Button {
                    showingImporter = true
                } label: {
                    Image(systemName: "paperclip").imageScale(.large)
                }
                .buttonStyle(.borderless)
                // Le champ fait 33 points de haut sur une ligne : 8 de marge, le texte,
                // 8 de marge. Sans hauteur explicite, le trombone se calait sur le bas de
                // la pile et flottait au-dessus de la ligne de base du champ.
                .frame(height: 33)
                .help("Joindre un fichier texte à la conversation")
                .disabled(model.isGenerating)

                TextField("Message…", text: $draft, axis: .vertical)
                    .textFieldStyle(.plain)
                    .lineLimit(1...6)
                    .padding(.horizontal, 11).padding(.vertical, 8)
                    .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 9))
                    .onSubmit(send)
                    .disabled(model.loaded == nil || model.isGenerating)

                if model.isGenerating {
                    Button("Arrêter", systemImage: "stop.fill") { model.stop() }
                        .buttonStyle(.bordered)
                } else {
                    Button("Envoyer", systemImage: "arrow.up") { send() }
                        .buttonStyle(.borderedProminent)
                        .disabled(model.loaded == nil || (draft.trimmingCharacters(
                            in: .whitespacesAndNewlines).isEmpty && attachments.isEmpty))
                }
            }
        }
        .padding(.horizontal, Self.columnPadding).padding(.vertical, 12)
        .frame(maxWidth: Self.columnWidth)
        .frame(maxWidth: .infinity)
    }

    private func settingsRow(_ conversation: Conversation) -> some View {
        HStack(spacing: 16) {
            Picker("", selection: binding(\.reasoningEffort)) {
                ForEach(Harmony.ReasoningEffort.allCases, id: \.rawValue) { effort in
                    Text(label(for: effort)).tag(effort.rawValue)
                }
            }
            .labelsHidden()
            .frame(width: 190)

            Spacer()

            // Température et remplissage du contexte côte à côte : ce sont les deux
            // réglages qu'on ajuste en cours de conversation, et les deux seules mesures
            // qui changent d'un message à l'autre.
            HStack(spacing: 6) {
                Image(systemName: "thermometer.medium")
                    .imageScale(.small).foregroundStyle(.secondary)
                    .help("Température : plus elle est haute, plus les réponses varient")
                Slider(value: binding(\.temperature), in: 0...1.5, step: 0.1).frame(width: 84)
                Text(String(format: "%.1f", conversation.settings.temperature))
                    .font(.caption.monospacedDigit()).frame(width: 24)
            }

            // Affiché même sans modèle chargé : sinon l'indicateur n'apparaît qu'une fois
            // le modèle en place, c'est-à-dire jamais au moment où on le cherche. La
            // capacité est alors celle choisie dans les réglages de chargement.
            ContextRing(
                used: conversation.contextUsed,
                capacity: model.loaded?.contextLength ?? model.contextLength,
                isLive: model.loaded != nil)
        }
    }

    private func label(for effort: Harmony.ReasoningEffort) -> String {
        switch effort {
        case .low: return "Raisonnement bref"
        case .medium: return "Raisonnement moyen"
        case .high: return "Raisonnement poussé"
        }
    }

    private func binding<T>(_ path: WritableKeyPath<GenerationSettings, T>) -> Binding<T> {
        Binding(
            get: { model.current?.settings[keyPath: path] ?? GenerationSettings()[keyPath: path] },
            set: { newValue in
                guard var conversation = model.current else { return }
                conversation.settings[keyPath: path] = newValue
                model.current = conversation
            })
    }

    private func attach(_ urls: [URL]) {
        for url in urls {
            let accessed = url.startAccessingSecurityScopedResource()
            defer { if accessed { url.stopAccessingSecurityScopedResource() } }
            guard let data = try? Data(contentsOf: url) else { continue }
            // Les fichiers joints entrent dans l'invite : au-delà de quelques dizaines de
            // milliers de caractères, ils rempliraient le contexte à eux seuls.
            let limit = 60_000
            var content = String(decoding: data, as: UTF8.self)
            if content.count > limit {
                content = String(content.prefix(limit)) + "\n[…tronqué…]"
            }
            attachments.append(Message.Attachment(
                name: url.lastPathComponent, content: content, byteCount: data.count))
        }
    }

    private func send() {
        let text = draft
        let files = attachments
        draft = ""
        attachments = []
        model.send(text, attachments: files)
    }
}
