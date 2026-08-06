import SwiftUI

struct ContentView: View {
    @Bindable var model: AppModel
    /// Repliés une fois un modèle chargé : ils ne servent plus, et ils écrasaient la
    /// liste des conversations, qui est ce qu'on consulte le plus souvent.
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

    /// La partie haute est dans une ScrollView, et ce n'est pas cosmétique.
    ///
    /// Empilées librement, la jauge, la bibliothèque et les réglages annonçaient une
    /// hauteur minimale de plus de trois mille points. Cette hauteur remontait jusqu'à
    /// `contentMinSize` de la fenêtre, qui s'ouvrait alors bien plus haute que l'écran :
    /// la zone de saisie se retrouvait sous le bord inférieur, hors d'atteinte.
    ///
    /// Une ScrollView rompt cette propagation — elle défile au lieu d'exiger — et rend au
    /// passage la barre latérale utilisable sur un écran court.
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
            // Sans hauteur idéale, la ScrollView réclamerait celle de son contenu et
            // reporterait le problème d'un cran.
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
        // Même raison que la partie haute : une List annonce la hauteur de ses lignes
        // comme hauteur idéale, et ferait grandir la fenêtre à chaque conversation.
        .frame(minHeight: 160, idealHeight: 260)
    }
}
