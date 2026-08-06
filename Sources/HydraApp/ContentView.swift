import SwiftUI

struct ContentView: View {
    @Bindable var model: AppModel
    /// Collapsed once a model is loaded: they are no longer needed, and they crowded out the
    /// conversation list, which is what one looks at most often.
    @State private var showingLibrary = true

    var body: some View {
        NavigationSplitView {
            sidebar
        } detail: {
            ChatView(model: model)
        }
        .alert(
            "Error", isPresented: Binding(
                get: { model.errorMessage != nil },
                set: { if !$0 { model.errorMessage = nil } })
        ) {
            Button("OK", role: .cancel) { model.errorMessage = nil }
        } message: {
            Text(model.errorMessage ?? "")
        }
        .overlay(alignment: .top) { loadingBanner }
        .onChange(of: model.loaded?.entry.id) { _, loaded in
            withAnimation(.easeInOut(duration: 0.2)) { showingLibrary = loaded == nil }
        }
    }

    @ViewBuilder private var loadingBanner: some View {
        if let message = model.loadingMessage {
            HStack(spacing: 8) {
                ProgressView().controlSize(.small)
                Text(message).font(.callout)
            }
            .padding(.horizontal, 16).padding(.vertical, 10)
            .background(.regularMaterial, in: Capsule())
            .shadow(radius: 8, y: 3)
            .padding(.top, 12)
        }
    }

    /// The top section sits in a ScrollView, and that is not cosmetic.
    ///
    /// Stacked freely, the gauge, the library and the settings announced a minimum height of
    /// more than three thousand points. That height propagated up to the window's
    /// `contentMinSize`, which then opened far taller than the screen: the input field ended
    /// up below the bottom edge, out of reach.
    ///
    /// A ScrollView breaks that propagation — it scrolls instead of demanding — and makes the
    /// sidebar usable on a short screen along the way.
    private var sidebar: some View {
        VStack(spacing: 0) {
            ScrollView {
                VStack(spacing: 0) {
                    MemoryGauge(
                        memory: model.memory,
                        modelName: model.loaded?.entry.displayName)
                        .padding(.horizontal, 12).padding(.top, 12).padding(.bottom, 10)

                    DisclosureGroup(isExpanded: $showingLibrary) {
                        VStack(alignment: .leading, spacing: 14) {
                            ModelLibraryView(model: model)
                            Divider()
                            LoadSettingsView(model: model)
                        }
                        .padding(.top, 10)
                    } label: {
                        Label("Models", systemImage: "shippingbox").font(.headline)
                    }
                    .padding(.horizontal, 14)
                    .padding(.bottom, 10)
                }
            }
            .scrollBounceBehavior(.basedOnSize)
            // Without an ideal height, the ScrollView would claim its content's and simply
            // move the problem one level up.
            .frame(idealHeight: 300)
            .layoutPriority(1)

            Divider()
            conversationList
        }
        .navigationSplitViewColumnWidth(min: 320, ideal: 350, max: 440)
    }

    private var conversationList: some View {
        VStack(spacing: 0) {
            HStack {
                Label("Conversations", systemImage: "bubble.left.and.bubble.right")
                    .font(.headline)
                Spacer()
                Button { model.newConversation() } label: {
                    Image(systemName: "square.and.pencil")
                }
                .buttonStyle(.borderless)
                .help("New conversation")
            }
            .padding(.horizontal, 14).padding(.vertical, 8)

            List(selection: $model.selection) {
                ForEach(model.conversations) { conversation in
                    VStack(alignment: .leading, spacing: 1) {
                        Text(conversation.title).lineLimit(1)
                        HStack(spacing: 5) {
                            Text(formatDate(conversation.updatedAt))
                            if conversation.messages.count > 1 {
                                Text("· \(conversation.messages.count) messages")
                            }
                        }
                        .font(.caption2).foregroundStyle(.secondary)
                    }
                    .tag(conversation.id)
                    .contextMenu {
                        Button("Delete", role: .destructive) {
                            model.delete(conversation.id)
                        }
                    }
                }
            }
            .listStyle(.sidebar)
        }
        // Same reason as the top section: a List announces the height of its rows as its
        // ideal height, and would grow the window with every conversation.
        .frame(minHeight: 160, idealHeight: 260)
    }
}
