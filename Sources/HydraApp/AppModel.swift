import Foundation
import HydraCore
import HydraFormat
import HydraInstall
import HydraMetal
import Observation

/// The memory snapshot the gauge displays.
///
/// **Three distinct quantities, never conflated.** Mixing them would give a flattering,
/// false number, against what the project sets out to demonstrate:
///
/// - `engaged` — memory owned by the process: expert slots, KV, scratch, logits;
/// - `mapped` — file-backed weights, in RAM while there is room, but reclaimable by the
///   kernel under pressure;
/// - `installed` — the model's size on disk, the point of comparison.
public struct MemorySnapshot: Sendable, Equatable {
    public var engaged = 0
    public var mapped = 0
    public var installed = 0

    /// What the model would occupy if it had to be loaded whole, against what Hydra
    /// actually occupies.
    public var total: Int { engaged + mapped }
    public var fraction: Double {
        installed > 0 ? min(1, Double(total) / Double(installed)) : 0
    }
    public var saved: Int { max(0, installed - total) }
}

@MainActor
@Observable
public final class AppModel {

    // MARK: - State

    public var conversations: [Conversation] = []
    public var selection: Conversation.ID?
    public var installations: [String: InstallationState] = [:]
    public var memory = MemorySnapshot()

    public var loaded: InferenceEngine.Loaded?
    public var loadingMessage: String?
    public var isGenerating = false
    public var errorMessage: String?

    /// Settings chosen at load time (D-005). The slot count is exposed deliberately:
    /// it is what makes the memory/speed trade-off tangible.
    public var contextLength = 4096
    /// The minimum by default: one slot per selected expert.
    ///
    /// Eight slots were the default for a while, on a measurement that gave them a 42 %
    /// lead. That lead was in fact the lead of the fixed overheads they masked — GPU
    /// synchronization and sampling. Once those were fixed, the gap falls to 10 % in a
    /// paired comparison (7.7 against 8.5 tok/s), for twice the cache footprint:
    /// 1.18 GiB against 2.37 (docs/02-MEASUREMENTS.md, M-018).
    ///
    /// Ten per cent is not worth 1.19 GiB in an application whose point is to show what
    /// it is enough to hold in memory. The setting stays exposed.
    public var slotsPerLayer = 8
    public var useMinimalSlots = true

    public static let contextChoices = [2048, 4096, 8192, 16384, 32768, 65536, 131_072]

    private let engine = InferenceEngine()
    private let store: ConversationStore
    private var memoryTimer: Timer?
    private var installTasks: [String: Task<Void, Never>] = [:]

    public init() {
        store = (try? ConversationStore()) ?? ConversationStore.unavailable
        conversations = store.load()
        if conversations.isEmpty { newConversation() }
        selection = conversations.first?.id
        refreshInstallations()
        startMemorySampling()
    }

    // MARK: - Conversations

    public var current: Conversation? {
        get { conversations.first { $0.id == selection } }
        set {
            guard let newValue, let index = conversations.firstIndex(where: { $0.id == newValue.id })
            else { return }
            conversations[index] = newValue
            store.save(conversations)
        }
    }

    public func newConversation() {
        let entry = loaded?.entry.id ?? CatalogEntry.all[0].id
        var conversation = Conversation(modelID: entry)
        conversation.title = "New conversation"
        conversations.insert(conversation, at: 0)
        selection = conversation.id
        store.save(conversations)
    }

    public func delete(_ id: Conversation.ID) {
        conversations.removeAll { $0.id == id }
        if selection == id { selection = conversations.first?.id }
        if conversations.isEmpty { newConversation() }
        store.save(conversations)
    }

    public func flush() { store.flush(conversations) }

    // MARK: - Library

    public func refreshInstallations() {
        for entry in CatalogEntry.all where !(installations[entry.id]?.isInstalling ?? false) {
            installations[entry.id] = ModelLocations.state(of: entry)
        }
    }

    public func install(_ entry: CatalogEntry) {
        guard installTasks[entry.id] == nil else { return }
        installations[entry.id] = .installing(fraction: 0, throughput: 0)

        // The progress callback is built here, outside the task: nesting a weak capture
        // inside an already-capturing `@Sendable` closure does not compile.
        let report: @Sendable (Double, Double) -> Void = { [weak self] fraction, throughput in
            Task { @MainActor in
                self?.installations[entry.id] = .installing(
                    fraction: fraction, throughput: throughput)
            }
        }
        let finish: @Sendable (String?) -> Void = { [weak self] failure in
            Task { @MainActor in
                guard let self else { return }
                self.installTasks[entry.id] = nil
                if let failure {
                    self.errorMessage = "Installing \(entry.displayName): \(failure)"
                }
                self.refreshInstallations()
            }
        }

        installTasks[entry.id] = Task {
            do {
                let client = HuggingFaceClient()
                let index = try await client.fetchIndex(repo: entry.repository)
                var headers: [String: SafetensorsHeader] = [:]
                for shard in index.shards.sorted() {
                    headers[shard] = try await client.fetchHeader(
                        repo: entry.repository, file: shard)
                }
                let plan = try RepackPlan(
                    config: entry.config, weightMap: index.weightMap, headers: headers)
                let problems = plan.validate(declaredSourceTotal: index.totalSize)
                guard problems.isEmpty else {
                    throw InstallFailure.plan(problems.map(\.description).joined(separator: " ; "))
                }

                let installer = TokenizerInstaller(client: client, repo: entry.repository)
                let repacker = StreamingRepacker(
                    plan: plan,
                    source: HuggingFaceSource(client: client, repo: entry.repository),
                    precision: entry.precision,
                    auxiliary: { partial in _ = try await installer.install(into: partial) })

                let started = Date()
                let destination = try ModelLocations.root(for: entry)
                _ = try await repacker.run(destination: destination) { progress in
                    let elapsed = Date().timeIntervalSince(started)
                    report(
                        Double(progress.bytesDone) / Double(max(progress.bytesTotal, 1)),
                        elapsed > 0 ? Double(progress.bytesDone) / elapsed : 0)
                }
                finish(nil)
            } catch {
                finish("\(error)")
            }
        }
    }

    public func cancelInstall(_ entry: CatalogEntry) {
        installTasks[entry.id]?.cancel()
        installTasks[entry.id] = nil
        refreshInstallations()
    }

    public func uninstall(_ entry: CatalogEntry) {
        if loaded?.entry.id == entry.id { unload() }
        if let root = try? ModelLocations.root(for: entry) {
            try? FileManager.default.removeItem(at: root)
            try? FileManager.default.removeItem(at: root.appendingPathExtension("partial"))
        }
        refreshInstallations()
    }

    enum InstallFailure: Error, CustomStringConvertible {
        case plan(String)
        var description: String {
            switch self { case .plan(let detail): return detail }
        }
    }

    // MARK: - Model loading

    public func load(_ entry: CatalogEntry) {
        guard installations[entry.id]?.isInstalled == true else { return }
        loadingMessage = "Preparing…"
        let slots = useMinimalSlots ? nil : slotsPerLayer
        engine.load(
            entry: entry, contextLength: contextLength, slotsPerLayer: slots,
            progress: { message in
                Task { @MainActor [weak self] in self?.loadingMessage = message }
            },
            completion: { result in
                Task { @MainActor [weak self] in
                    guard let self else { return }
                    self.loadingMessage = nil
                    switch result {
                    case .success(let loaded):
                        self.loaded = loaded
                        self.memory.installed = loaded.entry.installedBytes
                        self.memory.mapped = loaded.mappedBytes
                    case .failure(let failure):
                        self.errorMessage = failure.description
                    }
                }
            })
    }

    public func unload() {
        engine.unload()
        loaded = nil
        memory = MemorySnapshot()
    }

    // MARK: - Generation

    /// The message currently being written, so events can be routed to it.
    public private(set) var generatingMessage: Message.ID?

    public func send(_ text: String, attachments: [Message.Attachment] = []) {
        guard var conversation = current, loaded != nil, !isGenerating else { return }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty || !attachments.isEmpty else { return }

        conversation.messages.append(
            Message(role: .user, text: trimmed, attachments: attachments))
        conversation.messages.append(Message(role: .assistant, text: ""))
        conversation.retitleFromFirstMessage()
        conversation.updatedAt = Date()
        current = conversation

        startGeneration(in: conversation.id, answering: conversation.messages.count - 1)
    }

    /// Regenerates an assistant message's answer: the new generation becomes an extra
    /// **variant**, and the old one stays available.
    public func regenerate(_ messageID: Message.ID) {
        guard var conversation = current, loaded != nil, !isGenerating,
            let index = conversation.messages.firstIndex(where: { $0.id == messageID }),
            conversation.messages[index].role == .assistant
        else { return }

        conversation.messages[index].variants.append(Variant())
        conversation.messages[index].activeVariant =
            conversation.messages[index].variants.count - 1
        current = conversation
        startGeneration(in: conversation.id, answering: index)
    }

    /// Edits a question and regenerates the answer that follows, as a variant.
    public func editUserMessage(_ messageID: Message.ID, to text: String) {
        guard var conversation = current,
            let index = conversation.messages.firstIndex(where: { $0.id == messageID }),
            conversation.messages[index].role == .user
        else { return }

        conversation.messages[index].text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        conversation.updatedAt = Date()
        current = conversation

        let answer = index + 1
        guard answer < conversation.messages.count,
            conversation.messages[answer].role == .assistant
        else { return }
        regenerate(conversation.messages[answer].id)
    }

    /// Deletes a message. A question goes with its answer: separating them would leave a
    /// history the model could not interpret.
    public func deleteMessage(_ messageID: Message.ID) {
        guard var conversation = current,
            let index = conversation.messages.firstIndex(where: { $0.id == messageID })
        else { return }

        if conversation.messages[index].role == .user,
            index + 1 < conversation.messages.count,
            conversation.messages[index + 1].role == .assistant
        {
            conversation.messages.removeSubrange(index...(index + 1))
        } else {
            conversation.messages.remove(at: index)
        }
        conversation.updatedAt = Date()
        current = conversation
    }

    public func selectVariant(_ messageID: Message.ID, index: Int) {
        guard var conversation = current,
            let position = conversation.messages.firstIndex(where: { $0.id == messageID }),
            index >= 0, index < conversation.messages[position].variants.count
        else { return }
        conversation.messages[position].activeVariant = index
        current = conversation
    }

    private func startGeneration(in conversationID: Conversation.ID, answering index: Int) {
        guard let conversation = conversations.first(where: { $0.id == conversationID }),
            index < conversation.messages.count
        else { return }

        isGenerating = true
        generatingMessage = conversation.messages[index].id

        // The history stops before the answer currently being written.
        let turns = conversation.harmonyTurns(upTo: index)
        let settings = conversation.settings

        engine.generate(turns: turns, settings: settings) { event in
            Task { @MainActor [weak self] in
                self?.apply(event, to: conversationID)
            }
        }
    }

    public func stop() { engine.cancel() }

    private func apply(_ event: InferenceEngine.Event, to id: Conversation.ID) {
        guard let conversationIndex = conversations.firstIndex(where: { $0.id == id }),
            let messageIndex = conversations[conversationIndex].messages.firstIndex(
                where: { $0.id == generatingMessage })
        else { return }

        var message = conversations[conversationIndex].messages[messageIndex]
        var variant = message.current

        switch event {
        case .started(let promptTokens, _):
            conversations[conversationIndex].contextUsed = promptTokens
        case .firstToken(let seconds):
            variant.timeToFirstToken = seconds
        case .reasoning(let fragment):
            variant.reasoning += fragment
        case .text(let fragment):
            variant.text += fragment
        case .finished(let tokens, let seconds, let contextUsed):
            variant.outputTokens = tokens
            variant.tokensPerSecond = seconds > 0 ? Double(tokens) / seconds : nil
            conversations[conversationIndex].contextUsed = contextUsed
            isGenerating = false
            generatingMessage = nil
        case .failed(let reason):
            errorMessage = reason
            isGenerating = false
            generatingMessage = nil
        }

        message.current = variant
        conversations[conversationIndex].messages[messageIndex] = message
        conversations[conversationIndex].updatedAt = Date()
        store.save(conversations)
    }

    // MARK: - Memory

    private func startMemorySampling() {
        memoryTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { _ in
            Task { @MainActor [weak self] in
                guard let self else { return }
                self.memory.engaged = MemoryFootprint.current()
                if let loaded = self.loaded {
                    self.memory.mapped = loaded.mappedBytes
                    self.memory.installed = loaded.entry.installedBytes
                }
            }
        }
    }
}

extension ConversationStore {
    /// Fallback if Application Support is unreachable: the app stays usable, simply
    /// without persistence.
    static let unavailable: ConversationStore = {
        // `init` can only fail on an unusual system; we retry once.
        (try? ConversationStore()) ?? (try! ConversationStore())
    }()
}
