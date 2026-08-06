import HydraCore
import SwiftUI

// MARK: - Formatage

func formatBytes(_ bytes: Int) -> String {
    let gib = Double(bytes) / 1_073_741_824
    if gib >= 1 { return String(format: "%.2f Gio", gib) }
    return String(format: "%.0f Mio", Double(bytes) / 1_048_576)
}

/// Date absolue et stable.
///
/// Une date relative (« il y a 24 min et 23 s ») se réévalue en continu et attire l'œil
/// à chaque seconde, pour une information dont personne n'a besoin en temps réel.
func formatDate(_ date: Date) -> String {
    let formatter = DateFormatter()
    formatter.locale = Locale(identifier: "fr_FR")
    if Calendar.current.isDateInToday(date) {
        formatter.dateFormat = "HH:mm"
    } else if Calendar.current.isDateInYesterday(date) {
        return "hier"
    } else {
        formatter.dateFormat = "d MMM"
    }
    return formatter.string(from: date)
}

// MARK: - Jauge mémoire

/// L'élément central de l'application.
///
/// Elle répond à la question que le projet pose : combien de mémoire faut-il réellement
/// pour faire tourner ce modèle ? Les trois grandeurs restent séparées — mémoire engagée,
/// poids mappés, taille installée — parce que les confondre donnerait un chiffre flatteur
/// et faux.
struct MemoryGauge: View {
    let memory: MemorySnapshot
    let modelName: String?

    private var engagedFraction: Double {
        memory.installed > 0 ? Double(memory.engaged) / Double(memory.installed) : 0
    }
    private var mappedFraction: Double {
        memory.installed > 0 ? Double(memory.mapped) / Double(memory.installed) : 0
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            HStack(alignment: .firstTextBaseline) {
                Label("Mémoire", systemImage: "memorychip").font(.headline)
                Spacer()
                if let modelName {
                    Text(modelName).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                }
            }

            if memory.installed == 0 {
                Text("Chargez un modèle pour voir son empreinte réelle.")
                    .font(.caption).foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.vertical, 6)
            } else {
                bar
                legend
                Divider()
                headline
            }
        }
        .padding(13)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 10))
    }

    private var bar: some View {
        GeometryReader { geometry in
            let width = geometry.size.width
            ZStack(alignment: .leading) {
                RoundedRectangle(cornerRadius: 5).fill(Color.secondary.opacity(0.15))
                HStack(spacing: 0) {
                    Rectangle().fill(Color.accentColor)
                        .frame(width: max(2, width * engagedFraction))
                    Rectangle().fill(Color.accentColor.opacity(0.35))
                        .frame(width: width * mappedFraction)
                }
                .clipShape(RoundedRectangle(cornerRadius: 5))
            }
        }
        .frame(height: 13)
    }

    /// Légende en lignes plutôt qu'en colonnes : les intitulés sont longs, et sur trois
    /// colonnes ils se coupaient au milieu d'un mot.
    private var legend: some View {
        VStack(spacing: 5) {
            row(color: Color.accentColor, icon: "memorychip.fill",
                label: "En mémoire", value: memory.engaged,
                detail: "slots d'experts, cache KV, scratch")
            row(color: Color.accentColor.opacity(0.35), icon: "arrow.up.arrow.down.circle",
                label: "Poids mappés", value: memory.mapped,
                detail: "repris par le système sous pression")
            row(color: .secondary.opacity(0.22), icon: "internaldrive",
                label: "Modèle sur disque", value: memory.installed,
                detail: "ce qu'il faudrait charger en entier")
        }
    }

    /// La pastille reste : c'est elle qui relie la ligne au segment de la barre. L'icône
    /// s'ajoute devant l'intitulé, où elle aide à repérer la ligne cherchée sans lire.
    private func row(
        color: Color, icon: String, label: String, value: Int, detail: String
    ) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 7) {
            RoundedRectangle(cornerRadius: 2).fill(color)
                .frame(width: 9, height: 9)
                .offset(y: 1)
            VStack(alignment: .leading, spacing: 0) {
                Label(label, systemImage: icon).font(.caption)
                Text(detail).font(.caption2).foregroundStyle(.secondary).opacity(0.8)
            }
            Spacer(minLength: 6)
            Text(formatBytes(value))
                .font(.callout.monospacedDigit().weight(.medium))
        }
    }

    private var headline: some View {
        HStack(alignment: .center, spacing: 9) {
            Text(String(format: "%.0f %%", memory.fraction * 100))
                .font(.system(size: 27, weight: .semibold, design: .rounded))
                .foregroundStyle(Color.accentColor)
            VStack(alignment: .leading, spacing: 1) {
                Text("du modèle réellement en mémoire")
                Text("\(formatBytes(memory.saved)) économisés")
                    .foregroundStyle(.secondary)
            }
            .font(.caption)
            Spacer(minLength: 0)
        }
    }
}

// MARK: - Bibliothèque de modèles

struct ModelLibraryView: View {
    @Bindable var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(CatalogEntry.all) { entry in
                ModelRow(model: model, entry: entry)
            }
            if let free = ModelLocations.availableBytes() {
                Label("\(formatBytes(free)) libres sur le disque", systemImage: "externaldrive")
                    .font(.caption2).foregroundStyle(.secondary)
            }
        }
    }
}

struct ModelRow: View {
    @Bindable var model: AppModel
    let entry: CatalogEntry
    @State private var confirmingUninstall = false

    private var state: InstallationState { model.installations[entry.id] ?? .absent }
    private var isLoaded: Bool { model.loaded?.entry.id == entry.id }

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 1) {
                    HStack(spacing: 6) {
                        Text(entry.displayName).font(.callout.weight(.medium))
                        if isLoaded {
                            Label("chargé", systemImage: "checkmark.circle.fill")
                                .font(.caption2.weight(.semibold))
                                .padding(.horizontal, 5).padding(.vertical, 1)
                                .background(Color.accentColor.opacity(0.2), in: Capsule())
                        }
                    }
                    VStack(alignment: .leading, spacing: 2) {
                        Label(formatBytes(entry.installedBytes), systemImage: "internaldrive")
                        Label(entry.summary, systemImage: "square.stack.3d.up")
                    }
                    .font(.caption2).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: 8)
                actions
            }

            if case .installing(let fraction, let throughput) = state {
                VStack(alignment: .leading, spacing: 3) {
                    // Une fois les octets transférés, il reste le tokeniseur, le manifeste
                    // et le renommage — plusieurs minutes sur un modèle de soixante
                    // gigaoctets, pendant lesquelles aucun octet n'est compté. Un « 100 % »
                    // immobile donne l'impression d'un blocage ; on nomme donc l'étape.
                    if fraction >= 0.999 {
                        ProgressView().progressViewStyle(.linear)
                        Text("Finalisation : tokeniseur, manifeste, vérification…")
                            .font(.caption2).foregroundStyle(.secondary)
                    } else {
                        ProgressView(value: fraction)
                        Text(String(format: "%.0f %% · %.0f Mo/s",
                                    fraction * 100, throughput / 1e6))
                            .font(.caption2).foregroundStyle(.secondary)
                    }
                }
            }
        }
        .confirmationDialog(
            "Désinstaller \(entry.displayName) ?",
            isPresented: $confirmingUninstall, titleVisibility: .visible
        ) {
            Button("Désinstaller", role: .destructive) { model.uninstall(entry) }
            Button("Annuler", role: .cancel) {}
        } message: {
            Text("\(formatBytes(entry.installedBytes)) seront libérés. "
                 + "Le modèle devra être retéléchargé pour être réutilisé.")
        }
    }

    @ViewBuilder private var actions: some View {
        switch state {
        case .absent, .partial:
            Button(state == .partial ? "Reprendre" : "Installer") { model.install(entry) }
                .buttonStyle(.borderedProminent).controlSize(.small)
        case .installing:
            Button("Arrêter") { model.cancelInstall(entry) }
                .buttonStyle(.bordered).controlSize(.small)
        case .installed:
            HStack(spacing: 5) {
                if isLoaded {
                    Button("Décharger") { model.unload() }
                        .buttonStyle(.bordered).controlSize(.small)
                } else {
                    Button("Charger") { model.load(entry) }
                        .buttonStyle(.borderedProminent).controlSize(.small)
                        .disabled(model.loadingMessage != nil || model.isGenerating)
                }
                Button { confirmingUninstall = true } label: { Image(systemName: "trash") }
                    .buttonStyle(.borderless).controlSize(.small)
                    .disabled(model.isGenerating)
            }
        }
    }
}

// MARK: - Réglages de chargement

struct LoadSettingsView: View {
    @Bindable var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            HStack {
                Label("Contexte", systemImage: "text.alignleft").font(.caption)
                Spacer()
                Picker("", selection: $model.contextLength) {
                    ForEach(AppModel.contextChoices, id: \.self) { value in
                        Text("\(value / 1024)k").tag(value)
                    }
                }
                .labelsHidden().frame(width: 90)
            }
            .disabled(model.loaded != nil)

            Toggle(isOn: $model.useMinimalSlots) {
                Label("Experts en cache : minimum", systemImage: "square.stack.3d.down.right")
            }
            .font(.caption)
            .disabled(model.loaded != nil)

            if !model.useMinimalSlots {
                Stepper(
                    "\(model.slotsPerLayer) slots par couche",
                    value: $model.slotsPerLayer, in: 4...128, step: 4)
                    .font(.caption)
                    .disabled(model.loaded != nil)
            }

            Text(model.loaded == nil
                 ? "Plus de slots accélèrent la génération et augmentent l'empreinte."
                 : "Déchargez le modèle pour modifier ces réglages.")
                .font(.caption2).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}
