import Foundation
import HydraTokenize

/// Une génération. Un message d'assistant peut en compter plusieurs : régénérer, ou
/// modifier la question qui précède, ajoute une variante au lieu d'écraser la précédente.
public struct Variant: Identifiable, Codable, Sendable, Equatable {
    public var id: UUID = UUID()
    public var text: String = ""
    public var reasoning: String = ""
    /// Jetons produits, latence avant le premier jeton, débit.
    public var outputTokens: Int?
    public var timeToFirstToken: Double?
    public var tokensPerSecond: Double?

    public init(text: String = "", reasoning: String = "") {
        self.text = text
        self.reasoning = reasoning
    }
}

/// Un message affiché, avec son raisonnement conservé à part.
///
/// Le canal `analysis` est **stocké mais séparé** : le gabarit officiel de GPT-OSS ne le
/// réinjecte jamais dans l'historique en inférence, donc il ne doit pas se mélanger à la
/// réponse. Le masquer sans le garder reviendrait à le perdre.
public struct Message: Identifiable, Codable, Sendable, Equatable {
    public enum Role: String, Codable, Sendable { case user, assistant }

    public var id: UUID
    public var role: Role
    public var date: Date
    /// Toujours au moins une variante. Les messages d'utilisateur n'en ont qu'une.
    public var variants: [Variant]
    public var activeVariant: Int
    /// Fichiers joints par l'utilisateur, insérés dans l'invite.
    public var attachments: [Attachment]

    public struct Attachment: Identifiable, Codable, Sendable, Equatable {
        public var id: UUID = UUID()
        public var name: String
        public var content: String
        public var byteCount: Int
    }

    public init(
        id: UUID = UUID(), role: Role, text: String, reasoning: String = "",
        date: Date = Date(), attachments: [Attachment] = []
    ) {
        self.id = id
        self.role = role
        self.date = date
        self.variants = [Variant(text: text, reasoning: reasoning)]
        self.activeVariant = 0
        self.attachments = attachments
    }

    public var current: Variant {
        get { variants[min(activeVariant, variants.count - 1)] }
        set { variants[min(activeVariant, variants.count - 1)] = newValue }
    }

    public var text: String {
        get { current.text }
        set { current.text = newValue }
    }
    public var reasoning: String { current.reasoning }
    public var hasSeveralVariants: Bool { variants.count > 1 }

    /// Texte transmis au modèle : le message et ses pièces jointes.
    public var promptText: String {
        guard !attachments.isEmpty else { return text }
        var out = ""
        for attachment in attachments {
            out += "--- \(attachment.name) ---\n\(attachment.content)\n\n"
        }
        return out + text
    }
}

/// Réglages d'échantillonnage, par conversation.
public struct GenerationSettings: Codable, Sendable, Equatable {
    /// OpenAI recommande 1,0 avec `top_p` 1,0 pour GPT-OSS, c'est-à-dire la distribution
    /// brute. Sur des invites très courtes, cela rend le 20B franchement instable — il
    /// peut partir dans une autre langue. On adopte donc un réglage un peu plus serré par
    /// défaut, et la recommandation d'OpenAI reste atteignable d'un curseur.
    public var temperature: Double = 0.7
    public var topP: Double = 0.9
    public var reasoningEffort: String = "medium"
    /// Budget de jetons pour un tour, **raisonnement compris**.
    ///
    /// Il était de 1024, ce qui suffit à une réponse mais pas à une question difficile :
    /// GPT-OSS en raisonnement moyen dépense volontiers plus de mille jetons dans le canal
    /// d'analyse, épuisait son budget avant d'écrire quoi que ce soit, et s'arrêtait juste
    /// après avoir fini de réfléchir. L'utilisateur voyait le raisonnement se terminer,
    /// puis rien.
    ///
    /// Le budget réel est de toute façon borné par ce qui reste de contexte : c'est le
    /// moteur qui applique la plus contraignante des deux limites.
    public var maximumTokens: Int = 4096
    /// Consignes rendues dans le message `developer` de l'invite Harmony.
    public var instructions: String = ""

    public init() {}

    public var effort: Harmony.ReasoningEffort {
        Harmony.ReasoningEffort(rawValue: reasoningEffort) ?? .medium
    }
}

public struct Conversation: Identifiable, Codable, Sendable, Equatable {
    public var id: UUID
    public var title: String
    public var modelID: String
    public var messages: [Message]
    public var settings: GenerationSettings
    public var createdAt: Date
    public var updatedAt: Date
    /// Jetons occupés par la dernière invite envoyée, pour la jauge de contexte.
    public var contextUsed: Int = 0

    public init(
        id: UUID = UUID(), title: String = "New conversation",
        modelID: String = CatalogEntry.all[0].id,
        messages: [Message] = [], settings: GenerationSettings = GenerationSettings()
    ) {
        self.id = id
        self.title = title
        self.modelID = modelID
        self.messages = messages
        self.settings = settings
        self.createdAt = Date()
        self.updatedAt = Date()
    }

    /// Titre dérivé du premier message, tronqué proprement sur un mot.
    public mutating func retitleFromFirstMessage() {
        guard title == "New conversation",
            let first = messages.first(where: { $0.role == .user })
        else { return }
        let flat = first.text.replacingOccurrences(of: "\n", with: " ")
            .trimmingCharacters(in: .whitespaces)
        if flat.isEmpty { return }
        if flat.count <= 42 {
            title = flat
            return
        }
        let cut = flat.prefix(42)
        title = (cut.lastIndex(of: " ").map { String(cut[..<$0]) } ?? String(cut)) + "…"
    }

    /// Tours passés au format Harmony, jusqu'à `limit` exclu.
    /// Seul le canal final entre dans l'historique — le gabarit officiel est explicite.
    public func harmonyTurns(upTo limit: Int? = nil) -> [Harmony.Turn] {
        let slice = limit.map { Array(messages.prefix($0)) } ?? messages
        return slice.compactMap { message in
            switch message.role {
            case .user: return .user(message.promptText)
            case .assistant:
                return message.text.isEmpty ? nil : .assistant(message.text)
            }
        }
    }
}

/// Persistance des conversations, en JSON dans le support applicatif.
///
/// Écriture atomique et différée : une conversation change à chaque jeton produit, et
/// réécrire le fichier à cette fréquence coûterait plus cher que la génération.
public final class ConversationStore: @unchecked Sendable {

    private let url: URL
    private let queue = DispatchQueue(label: "hydra.conversations")
    private var pending: [Conversation]?

    public init() throws {
        let base = try FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask,
            appropriateFor: nil, create: true)
        let directory = base.appending(path: "Hydra")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        self.url = directory.appending(path: "conversations.json")
    }

    public func load() -> [Conversation] {
        guard let data = try? Data(contentsOf: url) else { return [] }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return (try? decoder.decode([Conversation].self, from: data)) ?? []
    }

    /// Programme une sauvegarde. Les appels rapprochés se fondent en une seule écriture.
    public func save(_ conversations: [Conversation]) {
        queue.async { [self] in
            let alreadyScheduled = pending != nil
            pending = conversations
            guard !alreadyScheduled else { return }
            queue.asyncAfter(deadline: .now() + 0.4) { [self] in
                guard let snapshot = pending else { return }
                pending = nil
                write(snapshot)
            }
        }
    }

    /// Force l'écriture immédiate — à la fermeture de l'application.
    public func flush(_ conversations: [Conversation]) {
        queue.sync {
            pending = nil
            write(conversations)
        }
    }

    private func write(_ conversations: [Conversation]) {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try? encoder.encode(conversations).write(to: url, options: .atomic)
    }
}
