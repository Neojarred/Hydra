import AppKit
import HydraCore
import HydraTokenize
import HydraVision
import SwiftUI
import UniformTypeIdentifiers

// MARK: - Context gauge

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
              : "Planned context: \(capacity) tokens, load a model to measure it")
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
                // fragment of a growing message, the cost is quadratic in its length, and it
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
                Label(
                    attachment.isImage
                        ? "\(attachment.name)  ·  \(attachment.image?.tokens ?? 0) tokens"
                        : attachment.name,
                    systemImage: attachment.isImage ? "photo" : "doc.text")
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
                    .help("Time to first token, prompt processing included")
            }
            // The prompt tokens that were not already in the cache.
            //
            // Without it the wait looks arbitrary, a long turn continuing a conversation
            // beats a short fresh paste, because only the new part is processed. This is the
            // number that time is proportional to.
            if let new = message.current.newPromptTokens, new > 0 {
                metric("\(new)", icon: "text.append")
                    .help("\(new) prompt tokens processed; the rest was reused from the "
                        + "previous turn")
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
                    help: "Regenerate, the current answer is kept",
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
    /// The two had their own values, 760 and 22 on one side, 800 and 16 on the other,
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
            // `.image` covers PNG, JPEG, HEIC and the rest through the type hierarchy, so the
            // picker offers them all without naming any. A picture chosen while a text-only
            // model is loaded is read as a document, which is what `attach` decides.
            allowedContentTypes: [
                .plainText, .sourceCode, .json, .yaml, .commaSeparatedText, .image,
            ],
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
            // conversation thus asked for a window several thousand points tall, far
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
                Text("\(loaded.entry.displayName) · context \(loaded.contextLength / 1024)k "
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
                ForEach(reasoningChoices, id: \.rawValue) { level in
                    Text(label(for: level)).tag(level.rawValue)
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
                    .help(
                        conversation.settings.followsModel
                            ? "Temperature, as this model recommends it. Move the slider to "
                                + "choose your own."
                            : "Temperature: the higher it is, the more answers vary")
                Slider(
                    value: samplingBinding(\.temperature, shown: effectiveSampling.temperature),
                    in: 0...1.5, step: 0.1
                ).frame(width: 84)
                Text(String(format: "%.1f", effectiveSampling.temperature))
                    .font(.caption.monospacedDigit()).frame(width: 24)
            }

            // Shown even with no model loaded: otherwise the indicator only appears once the
            // model is in place, that is, never when one goes looking for it. The capacity
            // is then the one chosen in the load settings.
            ContextRing(
                used: conversation.contextUsed,
                capacity: model.loaded?.contextLength ?? model.contextLength,
                isLive: model.loaded != nil)
        }
    }

    /// Each model is offered the choices it actually has, and no others.
    ///
    /// The architectures differ in kind, not in degree. GPT-OSS always reasons and its levels
    /// say how much, so `off` is not one of its options. Gemma's is a switch: the prompt either
    /// leaves the thought channel open or closes it, and there is no third state, so `low`,
    /// `medium` and `high` would be three controls doing the same thing, which is worse than one
    /// control that says what it does.
    ///
    /// Qwen is the same switch. Its template expresses "off" by pre-filling an empty
    /// `<think></think>` block, and there is nothing in the checkpoint that distinguishes a
    /// low from a high, so offering three would be inventing behaviour it does not have.
    private var reasoningChoices: [ReasoningLevel] {
        isThinkingSwitch
            ? [.off, .medium]
            : ReasoningLevel.allCases.filter { $0 != .off }
    }

    private var isThinkingSwitch: Bool {
        switch model.loaded?.entry.architecture {
        case .gemma4, .qwen35Moe: return true
        case .gptOss, nil: return false
        }
    }

    private func label(for level: ReasoningLevel) -> String {
        if isThinkingSwitch {
            return level == .off ? "No thinking" : "Thinking"
        }
        switch level {
        case .off: return "No reasoning"
        case .low: return "Low reasoning"
        case .medium: return "Medium reasoning"
        case .high: return "High reasoning"
        }
    }

    /// The sampling recipe in force: the loaded model's, unless the user has moved a slider.
    private var effectiveSampling: SamplingDefaults {
        let settings = model.current?.settings ?? GenerationSettings()
        guard settings.followsModel, let published = model.loaded?.entry.model.samplingDefaults
        else {
            return SamplingDefaults(
                temperature: Float(settings.temperature), topP: Float(settings.topP))
        }
        return published
    }

    /// A slider over a sampling value. Moving one takes the wheel from the model.
    ///
    /// Separate from `binding` because the *display* has to show what generation will actually
    /// use. A slider reading 0.7 while the model is being sampled at 1.0 is worse than no
    /// slider: it is a number that looks like a fact and is not one.
    private func samplingBinding(
        _ path: WritableKeyPath<GenerationSettings, Double>, shown: Float
    ) -> Binding<Double> {
        Binding(
            get: { Double(shown) },
            set: { newValue in
                guard var conversation = model.current else { return }
                if conversation.settings.followsModel {
                    // Adopt what was on screen, so the other slider does not jump when this one
                    // is nudged.
                    conversation.settings.temperature = Double(effectiveSampling.temperature)
                    conversation.settings.topP = Double(effectiveSampling.topP)
                    conversation.settings.usesModelDefaults = false
                }
                conversation.settings[keyPath: path] = newValue
                model.current = conversation
            })
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

    /// Whether the loaded model can read pictures at all.
    ///
    /// Only Qwen, today. Gemma's tower is installed and not yet implemented, and GPT-OSS has
    /// none at all.
    private var modelReadsImages: Bool {
        model.loaded?.entry.model.architecture == .qwen35Moe
    }

    /// Whether the file is a picture, whatever the loaded model can do with one.
    ///
    /// Asked separately from `modelReadsImages` on purpose. The first version of this checked
    /// only whether the *model* could read images and let everything else fall through to the
    /// document path, where a file is decoded as UTF-8 and truncated at 60,000 characters. A PNG
    /// down that path is not a document: it is sixty thousand characters of binary mistaken for
    /// text, which is what a user got when they attached a photograph to Gemma. Fifty thousand
    /// tokens of it.
    private func isImage(_ url: URL) -> Bool {
        (try? url.resourceValues(forKeys: [.contentTypeKey]).contentType)?
            .conforms(to: .image) ?? false
    }

    private func attach(_ urls: [URL]) {
        for url in urls {
            let accessed = url.startAccessingSecurityScopedResource()
            defer { if accessed { url.stopAccessingSecurityScopedResource() } }

            // A picture the loaded model cannot read is refused, out loud.
            //
            // Never silently, and never as a document: both of those produce a plausible-looking
            // turn built from something the model never saw.
            if isImage(url), !modelReadsImages {
                let name = model.loaded?.entry.displayName ?? "This model"
                model.errorMessage =
                    "\(name) cannot read images. Load Qwen 3.6 to ask about \(url.lastPathComponent)."
                continue
            }

            // `plan` reads the header alone, so the token cost is on the chip before anything
            // has been decoded.
            if modelReadsImages, isImage(url),
                let planned = try? ImagePatcher(config: Qwen35VisionConfig.a3b).plan(for: url)
            {
                let bytes = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
                let config = Qwen35VisionConfig.a3b
                attachments.append(Message.Attachment(
                    name: url.lastPathComponent, content: "", byteCount: bytes,
                    image: Message.Attachment.Image(
                        path: url.path, tokens: planned.tokens,
                        pixelWidth: planned.grid.width * config.patchSize,
                        pixelHeight: planned.grid.height * config.patchSize)))
                continue
            }

            if isImage(url) {
                model.errorMessage = "\(url.lastPathComponent) could not be read as an image."
                continue
            }
            guard let data = try? Data(contentsOf: url) else { continue }

            // Binary is refused rather than mangled.
            //
            // `String(decoding:as:UTF8.self)` never fails: it substitutes a replacement
            // character for every byte it cannot read, so any binary file becomes tens of
            // thousands of characters of noise that tokenize into tens of thousands of tokens
            // and describe nothing. `String(data:encoding:)` returns nil instead, which is the
            // answer wanted here.
            guard let decoded = String(data: data, encoding: .utf8) else {
                model.errorMessage =
                    "\(url.lastPathComponent) is not a text file, so there is nothing to attach."
                continue
            }
            // Attached files go into the prompt: beyond a few tens of thousands of
            // characters, they would fill the context on their own.
            let limit = 60_000
            var content = decoded
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
