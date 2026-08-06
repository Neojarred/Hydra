import AppKit
import HydraTokenize
import SwiftUI
import UniformTypeIdentifiers

// MARK: - Jauge de contexte

/// A small ring that fills as the context is consumed, from blue to red.
///
/// Context is a silent resource: nothing warns you that you are approaching the limit,
/// and overrunning it shows up as an error or a truncated conversation. A permanent
/// indicator avoids the surprise.
struct ContextRing: View {
    let used: Int
    let capacity: Int
    /// False while no model is loaded: the ring then shows the planned capacity, dimmed,
    /// rather than implying a measurement.
    var isLive: Bool = true

    private var fraction: Double {
        capacity > 0 ? min(1, Double(used) / Double(capacity)) : 0
    }
    private var color: Color {
        // Blue up to halfway, then drifting towards orange and red.
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
              ? "Context: \(used) tokens used of \(capacity)"
              : "Planned context: \(capacity) tokens — load a model to measure it")
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
            Label("Reasoning", systemImage: "brain")
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
                    Text("thinking…").font(.callout).foregroundStyle(.secondary)
                }
            } else if isGenerating {
                // Plain text while generating.
                //
                // `MarkdownView` re-parses the whole message on every render. Called on every
                // fragment of a growing message, the cost is quadratic in its length — and it
                // adds to the decoding instead of overlapping with it.
                // Formatting appears at the end, once and for all.
                Text(message.text)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                // The model writes Markdown, with LaTeX in the technical passages.
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
        // The alignment must follow the side the bubble is on.
        //
        // With `.leading`, the user's message hugged the left of a 520-point box that was
        // itself right-aligned: the bubble therefore floated 520 points short of the edge,
        // never reaching it.
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
                Button("Cancel") { isEditing = false }
                    .buttonStyle(.bordered).controlSize(.small)
                Button("Save and regenerate") {
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

    /// Metrics and actions. The metrics stay visible; the actions only sharpen on hover,
    /// so as not to clutter reading.
    private var metadata: some View {
        HStack(spacing: 10) {
            if isUser { Spacer(minLength: 0) }

            if message.hasSeveralVariants { variantSelector }

            if let tokens = message.current.outputTokens {
                metric("\(tokens)", icon: "number")
                    .help("\(tokens) tokens produced")
            }
            if let ttft = message.current.timeToFirstToken {
                metric(String(format: "%.1f s", ttft), icon: "timer")
                    .help("Time to first token, prefill included")
            }
            if let rate = message.current.tokensPerSecond {
                metric(String(format: "%.1f/s", rate), icon: "speedometer")
                    .help("Tokens produced per second")
            }

            // The actions stay visible and clickable at all times; hovering only makes them
            // crisp.
            //
            // Tying them to hover made them fail twice: first because inserting them shifted
            // the row under the pointer, then because a conditional target is still a target
            // you have to reach before it changes its mind. A button that is always there
            // does not have that problem.
            if !isGenerating {
                actions.opacity(hovering ? 1 : 0.4)
            }

            if !isUser { Spacer(minLength: 0) }
        }
        .frame(minHeight: 22)
        .animation(.easeOut(duration: 0.12), value: hovering)
    }

    /// An icon rather than a label: "128 tokens · 1.4 s to first token · 7.5 tok/s"
    /// repeated the word "tokens" twice per message and took the full width. The symbol
    /// carries the meaning, the tooltip gives the full sentence.
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
            .help("Previous answer")

            Text("\(message.activeVariant + 1)/\(message.variants.count)")
                .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                .help("\(message.variants.count) answers generated for this message")

            Button {
                model.selectVariant(message.id, index: message.activeVariant + 1)
            } label: {
                Image(systemName: "chevron.right")
            }
            .disabled(message.activeVariant == message.variants.count - 1)
            .help("Next answer")
        }
        .buttonStyle(.borderless)
        .font(.caption)
    }

    private var actions: some View {
        HStack(spacing: 2) {
            actionButton(
                icon: copied ? "checkmark" : "doc.on.doc",
                help: copied ? "Copied" : "Copy message"
            ) {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(message.text, forType: .string)
                copied = true
                DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) { copied = false }
            }

            if isUser {
                actionButton(
                    icon: "pencil", help: "Edit the question and regenerate the answer",
                    disabled: model.loaded == nil
                ) {
                    draft = message.text
                    isEditing = true
                }
            } else {
                actionButton(
                    icon: "arrow.clockwise",
                    help: "Regenerate — the current answer is kept",
                    disabled: model.loaded == nil
                ) {
                    model.regenerate(message.id)
                }
            }

            actionButton(
                icon: "trash",
                help: isUser ? "Delete the question and its answer" : "Delete the answer"
            ) {
                model.deleteMessage(message.id)
            }
        }
        .foregroundStyle(.secondary)
    }

    /// Cible de clic explicite.
    ///
    /// A `Button` whose label is a bare icon offers only the few points of the glyph, and
    /// the borderless style does not widen it. The frame and `contentShape` give a crisp
    /// area, independent of the symbol's shape.
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
    /// The width and padding shared by the transcript and the composer.
    ///
    /// The two had their own values — 760 and 22 on one side, 800 and 16 on the other —
    /// and since padding applies before the cap, the usable columns came out at 716 and
    /// 768. The bubbles therefore lined up with nothing, and the offset left a blank
    /// strip on the right.
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
                    "No conversation", systemImage: "bubble.left.and.bubble.right")
            }
        }
        // The minimum is carried by the detail column rather than the split view's root.
        // The ideal size must be given too: without it the window opens at its minimum
        // width and the height of the screen.
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
                // A centred column: on a wide window, full-width text becomes tiring to
                // read.
                .frame(maxWidth: Self.columnWidth)
                .frame(maxWidth: .infinity)
            }
            // A ScrollView announces its content's height as its ideal height. A long
            // conversation thus asked for a window several thousand points tall — far
            // beyond the screen. The ideal height is therefore pinned here; the view stays
            // free to expand, it just no longer dictates the window's size.
            .frame(idealHeight: 420)
            .onChange(of: conversation.messages.last?.text) { _, _ in
                // No animation: it restarts on every refresh, and an animation interrupted
                // twenty times a second costs more than it gives.
                if let last = conversation.messages.last {
                    proxy.scrollTo(last.id, anchor: .bottom)
                }
            }
        }
    }

    private var emptyState: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(model.loaded == nil
                 ? "Load a model to begin."
                 : "Ask your question.")
                .foregroundStyle(.secondary)
            if let loaded = model.loaded {
                Text("\(loaded.entry.displayName) · contexte \(loaded.contextLength / 1024)k "
                     + "· \(loaded.slotsPerLayer) cached experts per layer")
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
                // The field is 33 points tall on one line: 8 of padding, the text, 8 of
                // padding. Without an explicit height, the paperclip settled on the bottom of
                // the stack and floated above the field's baseline.
                .frame(height: 33)
                .help("Attach a text file to the conversation")
                .disabled(model.isGenerating)

                TextField("Message…", text: $draft, axis: .vertical)
                    .textFieldStyle(.plain)
                    .lineLimit(1...6)
                    .padding(.horizontal, 11).padding(.vertical, 8)
                    .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 9))
                    .onSubmit(send)
                    .disabled(model.loaded == nil || model.isGenerating)

                if model.isGenerating {
                    Button("Stop", systemImage: "stop.fill") { model.stop() }
                        .buttonStyle(.bordered)
                } else {
                    Button("Send", systemImage: "arrow.up") { send() }
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

            // Temperature and context fill side by side: these are the two settings one
            // adjusts mid-conversation, and the only two measurements that change from one
            // message to the next.
            HStack(spacing: 6) {
                Image(systemName: "thermometer.medium")
                    .imageScale(.small).foregroundStyle(.secondary)
                    .help("Temperature: the higher it is, the more answers vary")
                Slider(value: binding(\.temperature), in: 0...1.5, step: 0.1).frame(width: 84)
                Text(String(format: "%.1f", conversation.settings.temperature))
                    .font(.caption.monospacedDigit()).frame(width: 24)
            }

            // Shown even with no model loaded: otherwise the indicator only appears once the
            // model is in place — that is, never when one goes looking for it. The capacity
            // is then the one chosen in the load settings.
            ContextRing(
                used: conversation.contextUsed,
                capacity: model.loaded?.contextLength ?? model.contextLength,
                isLive: model.loaded != nil)
        }
    }

    private func label(for effort: Harmony.ReasoningEffort) -> String {
        switch effort {
        case .low: return "Low reasoning"
        case .medium: return "Medium reasoning"
        case .high: return "High reasoning"
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
            // Attached files go into the prompt: beyond a few tens of thousands of
            // characters, they would fill the context on their own.
            let limit = 60_000
            var content = String(decoding: data, as: UTF8.self)
            if content.count > limit {
                content = String(content.prefix(limit)) + "\n[…truncated…]"
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
