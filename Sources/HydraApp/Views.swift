import AppKit
import HydraCore
import SwiftUI

// MARK: - Formatage

func formatBytes(_ bytes: Int) -> String {
    let gib = Double(bytes) / 1_073_741_824
    if gib >= 1 { return String(format: "%.2f Gio", gib) }
    return String(format: "%.0f Mio", Double(bytes) / 1_048_576)
}

/// An absolute, stable date.
///
/// A relative date ("24 min 23 s ago") re-evaluates continuously and pulls the eye every
/// second, for something nobody needs in real time.
///
/// The formatter follows the system locale rather than a hard-coded one: the time of day
/// and the month name should read the way the reader expects them to.
func formatDate(_ date: Date) -> String {
    if Calendar.current.isDateInYesterday(date) { return "yesterday" }
    let formatter = DateFormatter()
    formatter.locale = Locale.autoupdatingCurrent
    formatter.setLocalizedDateFormatFromTemplate(
        Calendar.current.isDateInToday(date) ? "j:mm" : "d MMM")
    return formatter.string(from: date)
}

// MARK: - Memory gauge

/// The centrepiece of the application.
///
/// It answers the question the project poses: how much memory does running this model
/// actually take? The three quantities stay separate, engaged memory, mapped weights,
/// installed size, because conflating them would give a flattering, false number.
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
                Label("Memory", systemImage: "memorychip").font(.headline)
                Spacer()
                if let modelName {
                    Text(modelName).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                }
            }

            if memory.installed == 0 {
                Text("Load a model to see its real footprint.")
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

    /// A legend in rows rather than columns: the labels are long, and across three columns
    /// they broke mid-word.
    private var legend: some View {
        VStack(spacing: 5) {
            row(color: Color.accentColor, icon: "memorychip.fill",
                label: "Resident", value: memory.engaged,
                detail: "expert slots, KV cache, scratch")
            row(color: Color.accentColor.opacity(0.35), icon: "arrow.up.arrow.down.circle",
                label: "Mapped weights", value: memory.mapped,
                detail: "reclaimable by the system under pressure")
            row(color: .secondary.opacity(0.22), icon: "internaldrive",
                label: "Model on disk", value: memory.installed,
                detail: "what loading it whole would cost")
        }
    }

    /// The swatch stays: it is what ties the row to its segment of the bar. The icon is
    /// added before the label, where it helps find the right row without reading.
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
                Text("of the model actually resident")
                Text("\(formatBytes(memory.saved)) saved")
                    .foregroundStyle(.secondary)
            }
            .font(.caption)
            Spacer(minLength: 0)
        }
    }
}

// MARK: - Model library

struct ModelLibraryView: View {
    @Bindable var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(CatalogEntry.all) { entry in
                ModelRow(model: model, entry: entry)
            }
            if let free = ModelLocations.availableBytes() {
                Label("\(formatBytes(free)) free on disk", systemImage: "externaldrive")
                    .font(.caption2).foregroundStyle(.secondary)
            }

            // The models folder can be revealed from here.
            //
            // It lives under the user's account, not inside the app bundle: updating Hydra
            // must not cost a 73 GiB re-download. The counterpart is that trashing the app
            // leaves the models behind, macOS runs no code on deletion, so no application
            // can clean up after itself. The only honest remedy is to make the place
            // visible.
            if let directory = try? ModelLocations.directory() {
                Button {
                    NSWorkspace.shared.activateFileViewerSelecting([directory])
                } label: {
                    Label("Show the models folder", systemImage: "folder")
                        .font(.caption2)
                }
                .buttonStyle(.link)
                .help("Models stay on disk if you delete Hydra: "
                      + "\(directory.path)")
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
                            Label("loaded", systemImage: "checkmark.circle.fill")
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
                    // Once the bytes are transferred, the tokenizer, the manifest and the
                    // rename remain, several minutes on a sixty-gigabyte model, during
                    // which no byte is counted. A motionless "100 %" reads as a hang, so we
                    // name the step instead.
                    if fraction >= 0.999 {
                        ProgressView().progressViewStyle(.linear)
                        Text("Finishing: tokenizer, manifest, verification…")
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
            "Uninstall \(entry.displayName)?",
            isPresented: $confirmingUninstall, titleVisibility: .visible
        ) {
            Button("Uninstall", role: .destructive) { model.uninstall(entry) }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("\(formatBytes(entry.installedBytes)) will be freed. "
                 + "The model will have to be downloaded again to be used.")
        }
    }

    @ViewBuilder private var actions: some View {
        switch state {
        case .absent, .partial:
            Button(state == .partial ? "Resume" : "Install") { model.install(entry) }
                .buttonStyle(.borderedProminent).controlSize(.small)
        case .installing:
            Button("Stop") { model.cancelInstall(entry) }
                .buttonStyle(.bordered).controlSize(.small)
        case .installed:
            HStack(spacing: 5) {
                if isLoaded {
                    Button("Unload") { model.unload() }
                        .buttonStyle(.bordered).controlSize(.small)
                } else {
                    Button("Load") { model.load(entry) }
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

// MARK: - Load settings

struct LoadSettingsView: View {
    @Bindable var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            HStack {
                Label("Context", systemImage: "text.alignleft").font(.caption)
                Spacer()
                Picker("", selection: $model.contextLength) {
                    ForEach(AppModel.contextChoices, id: \.self) { value in
                        Text("\(value / 1024)k").tag(value)
                    }
                }
                .labelsHidden().frame(width: 90)
            }
            .disabled(model.loaded != nil)

            // The bounds come from the selected model and this machine, not from a constant.
            // `nil` means no Metal device, or a model this machine cannot hold at all.
            let entry = model.settingsEntry
            let bounds = entry.flatMap { model.slotBounds(for: $0, at: model.slotsPerLayer) }

            Toggle(isOn: $model.useRecommendedSlots) {
                Label(
                    bounds.map { "Cached experts: recommended (\($0.recommended))" }
                        ?? "Cached experts: recommended",
                    systemImage: "square.stack.3d.down.right")
            }
            .font(.caption)
            .disabled(model.loaded != nil)

            if !model.useRecommendedSlots, let bounds {
                Stepper(
                    "\(model.slotsPerLayer) slots per layer",
                    value: $model.slotsPerLayer,
                    in: bounds.recommended...max(bounds.recommended, bounds.maximum),
                    step: 4)
                    .font(.caption)
                    .disabled(model.loaded != nil)
                    .onAppear { model.clampSlots(to: bounds) }
                    .onChange(of: bounds.maximum) { _, _ in model.clampSlots(to: bounds) }
            }

            Text(caption(entry: entry, bounds: bounds))
                .font(.caption2).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    /// What the current choice costs, in the terms the setting is actually about.
    ///
    /// Deliberately does **not** promise that more slots decode faster. It used to, and
    /// interleaved measurement found no effect on this machine at any count between 8 and 24
    /// (M-054, retracted): the reads more slots save are already served from page cache. What
    /// they reliably do is raise the footprint, which is the number shown.
    private func caption(
        entry: CatalogEntry?, bounds: (recommended: Int, maximum: Int, footprint: Int)?
    ) -> String {
        guard model.loaded == nil else { return "Unload the model to change these settings." }
        guard let entry, let bounds else {
            return "Install a model to see what this machine can hold."
        }
        let name = entry.displayName
        let gib = Double(bounds.footprint) / 1_073_741_824
        let footprint = String(format: "%.2f GiB", gib)
        if model.useRecommendedSlots {
            return "\(name): \(bounds.recommended) slots a layer, about \(footprint) in "
                + "memory. This machine can hold up to \(bounds.maximum)."
        }
        return "\(name): about \(footprint) in memory. This machine can hold up to "
            + "\(bounds.maximum) slots a layer; more of them raise the footprint and are not "
            + "measurably faster."
    }
}
