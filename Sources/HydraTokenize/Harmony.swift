import Foundation

/// Format de conversation **Harmony**, obligatoire pour GPT-OSS.
///
/// Ce n'est pas un simple gabarit de chat. Le modèle a été entraîné à produire ses
/// réponses réparties sur des **canaux** : `analysis` porte le raisonnement, `final` la
/// réponse destinée à l'utilisateur, `commentary` les appels d'outils. Sans le format,
/// le modèle ne sait pas où écrire quoi, et la sortie est inexploitable.
///
/// Structure exacte, relevée sur le `chat_template.jinja` publié :
///
/// ```
/// <|start|>system<|message|>{identité}
/// Knowledge cutoff: 2024-06
/// Current date: {AAAA-MM-JJ}
///
/// Reasoning: {low|medium|high}
///
/// # Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>
/// <|start|>developer<|message|># Instructions\n\n{consignes}\n\n<|end|>
/// <|start|>user<|message|>{texte}<|end|>
/// <|start|>assistant<|channel|>final<|message|>{réponse}<|end|>
/// <|start|>assistant
/// ```
///
/// **Le raisonnement des tours précédents n'est jamais réinjecté** en inférence : seul le
/// canal `final` est conservé dans l'historique. Le gabarit officiel est explicite là-dessus.
public enum Harmony {

    public enum ReasoningEffort: String, Sendable, CaseIterable {
        case low, medium, high
    }

    public enum Channel: String, Sendable {
        case analysis, commentary, final
    }

    public struct Turn: Sendable {
        public enum Role: String, Sendable { case user, assistant }
        public let role: Role
        public let content: String

        public init(role: Role, content: String) {
            self.role = role
            self.content = content
        }

        public static func user(_ text: String) -> Turn { Turn(role: .user, content: text) }
        public static func assistant(_ text: String) -> Turn { Turn(role: .assistant, content: text) }
    }

    // MARK: - Rendu

    public struct Renderer: Sendable {
        public var modelIdentity: String
        public var knowledgeCutoff: String
        public var currentDate: String
        public var reasoningEffort: ReasoningEffort
        /// Consignes de l'utilisateur développeur, rendues dans le message `developer`.
        public var instructions: String?

        public init(
            modelIdentity: String = "You are ChatGPT, a large language model trained by OpenAI.",
            knowledgeCutoff: String = "2024-06",
            currentDate: String? = nil,
            reasoningEffort: ReasoningEffort = .medium,
            instructions: String? = nil
        ) {
            self.modelIdentity = modelIdentity
            self.knowledgeCutoff = knowledgeCutoff
            if let currentDate {
                self.currentDate = currentDate
            } else {
                let formatter = DateFormatter()
                formatter.dateFormat = "yyyy-MM-dd"
                formatter.timeZone = TimeZone(identifier: "UTC")
                self.currentDate = formatter.string(from: Date())
            }
            self.reasoningEffort = reasoningEffort
            self.instructions = instructions
        }

        public func systemMessage() -> String {
            var text = modelIdentity + "\n"
            text += "Knowledge cutoff: \(knowledgeCutoff)\n"
            text += "Current date: \(currentDate)\n\n"
            text += "Reasoning: \(reasoningEffort.rawValue)\n\n"
            text += "# Valid channels: analysis, commentary, final. "
            text += "Channel must be included for every message."
            return text
        }

        /// Rend l'invite complète, prête à être encodée avec `allowSpecial: true`.
        ///
        /// Le texte des tours est inséré **tel quel** ; c'est l'encodage qui doit refuser
        /// d'interpréter les balises qu'il contiendrait (`allowSpecial: false` pour le
        /// contenu utilisateur). Le découpage des responsabilités est volontaire : le
        /// rendu ne connaît pas les jetons, l'encodeur ne connaît pas le format.
        public func render(turns: [Turn]) -> String {
            var out = "<|start|>system<|message|>" + systemMessage() + "<|end|>"

            if let instructions, !instructions.isEmpty {
                out += "<|start|>developer<|message|># Instructions\n\n"
                out += instructions + "\n\n<|end|>"
            }

            for turn in turns {
                switch turn.role {
                case .user:
                    out += "<|start|>user<|message|>" + turn.content + "<|end|>"
                case .assistant:
                    // Le raisonnement des tours passés est volontairement omis.
                    out += "<|start|>assistant<|channel|>final<|message|>"
                    out += turn.content + "<|end|>"
                }
            }

            out += "<|start|>assistant"
            return out
        }
    }

    // MARK: - Jetons de contrôle

    /// Jetons qui terminent une génération. `<|return|>` marque la fin normale d'un tour ;
    /// `<|call|>` signale un appel d'outil ; `<|endoftext|>` sert de garde-fou.
    public static let stopTokenNames = ["<|return|>", "<|endoftext|>", "<|call|>"]

    public static func stopTokens(in tokenizer: BPETokenizer) -> Set<Int> {
        Set(stopTokenNames.compactMap { tokenizer.specialTokens[$0] })
    }

    // MARK: - Analyse incrémentale de la sortie

    /// Interprète la sortie du modèle au fil des jetons.
    ///
    /// Le modèle produit typiquement :
    /// ```
    /// <|channel|>analysis<|message|>…raisonnement…<|end|>
    /// <|start|>assistant<|channel|>final<|message|>…réponse…<|return|>
    /// ```
    /// L'interface n'affiche en général que `final`, mais `analysis` doit rester
    /// accessible — c'est la chaîne de raisonnement, et la masquer sans la capturer
    /// reviendrait à la perdre.
    ///
    /// L'analyse est **incrémentale et par octets** : un caractère accentué ou un emoji
    /// peut être réparti sur plusieurs jetons, donc on n'essaie jamais de décoder un
    /// jeton isolé en texte.
    public struct Parser: Sendable {

        public enum Event: Sendable, Equatable {
            /// Texte produit sur un canal. Peut arriver par fragments.
            case text(channel: Channel, String)
            /// Un canal vient de se terminer.
            case channelEnded(Channel)
            /// Jeton d'arrêt rencontré.
            case stopped(String)
        }

        enum State {
            /// Après `<|start|>` : le modèle écrit le nom du rôle (« assistant »), qui
            /// n'est pas du contenu. On l'absorbe jusqu'à `<|channel|>` ou `<|message|>`.
            case readingRole
            /// On lit le nom du canal après `<|channel|>`.
            case readingChannelName
            /// On accumule le contenu d'un message.
            case readingContent(Channel)
        }

        private let tokenizer: BPETokenizer
        private let stops: Set<Int>

        public init(tokenizer: BPETokenizer) {
            self.tokenizer = tokenizer
            self.stops = Harmony.stopTokens(in: tokenizer)
        }

        /// État mutable de l'analyse, séparé du parseur pour qu'il reste `Sendable`.
        public struct Session: Sendable {
            var state: State = .readingRole
            var pendingBytes: [UInt8] = []
            var channelNameBytes: [UInt8] = []
            /// Texte accumulé par canal. Le raisonnement reste accessible même quand
            /// l'interface ne l'affiche pas — le masquer sans le capturer le perdrait.
            public internal(set) var channels: [Channel: String] = [:]
            public internal(set) var isFinished = false

            public init() {}

            public var finalText: String { channels[.final] ?? "" }
            public var analysisText: String { channels[.analysis] ?? "" }
        }

        /// Consomme un jeton et rend les évènements produits.
        public func consume(_ token: Int, session: inout Session) -> [Event] {
            guard !session.isFinished else { return [] }

            if stops.contains(token) {
                let name = tokenizer.name(of: token) ?? "<|stop|>"
                var events = flush(&session)
                session.isFinished = true
                events.append(.stopped(name))
                return events
            }

            if let special = tokenizer.name(of: token) {
                return handleSpecial(special, session: &session)
            }

            let bytes = tokenizer.bytes(for: token)
            switch session.state {
            case .readingChannelName:
                session.channelNameBytes.append(contentsOf: bytes)
                return []
            case .readingContent(let channel):
                session.pendingBytes.append(contentsOf: bytes)
                // On n'émet que des octets formant de l'UTF-8 complet : sinon un accent
                // coupé en deux jetons produirait un caractère de remplacement.
                guard let text = decodeComplete(&session.pendingBytes), !text.isEmpty else {
                    return []
                }
                session.channels[channel, default: ""] += text
                return [.text(channel: channel, text)]
            case .readingRole:
                // Le nom du rôle est écrit par le modèle après `<|start|>` ; il ne fait
                // pas partie de la réponse. Sans ce cas, « assistant » apparaissait en
                // tête de chaque message affiché.
                return []
            }
        }

        private func handleSpecial(_ name: String, session: inout Session) -> [Event] {
            switch name {
            case "<|channel|>":
                session.channelNameBytes = []
                session.state = .readingChannelName
                return []
            case "<|message|>":
                let raw = String(decoding: session.channelNameBytes, as: UTF8.self)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                let channel = Channel(rawValue: raw) ?? .final
                session.channelNameBytes = []
                session.state = .readingContent(channel)
                return []
            case "<|end|>", "<|start|>":
                let events = flush(&session)
                session.state = .readingRole
                return events
            default:
                return []
            }
        }

        private func flush(_ session: inout Session) -> [Event] {
            var events: [Event] = []
            if case .readingContent(let channel) = session.state {
                if !session.pendingBytes.isEmpty {
                    let text = String(decoding: session.pendingBytes, as: UTF8.self)
                    session.pendingBytes = []
                    if !text.isEmpty {
                        session.channels[channel, default: ""] += text
                        events.append(.text(channel: channel, text))
                    }
                }
                events.append(.channelEnded(channel))
            }
            return events
        }

        /// Extrait le préfixe UTF-8 complet des octets en attente, et le retire du tampon.
        private func decodeComplete(_ buffer: inout [UInt8]) -> String? {
            guard !buffer.isEmpty else { return nil }
            // Recule jusqu'à une frontière de caractère : un octet de continuation vaut
            // 10xxxxxx, un octet de tête commence une séquence.
            var end = buffer.count
            var back = 0
            while end > 0, back < 4 {
                let byte = buffer[end - 1]
                if byte & 0b1100_0000 != 0b1000_0000 {
                    // Octet de tête : la séquence est complète si sa longueur tient.
                    let needed: Int
                    if byte & 0b1000_0000 == 0 { needed = 1 }
                    else if byte & 0b1110_0000 == 0b1100_0000 { needed = 2 }
                    else if byte & 0b1111_0000 == 0b1110_0000 { needed = 3 }
                    else { needed = 4 }
                    if back + 1 >= needed { end = buffer.count }
                    else { end -= 1 }
                    break
                }
                end -= 1
                back += 1
            }
            guard end > 0 else { return nil }
            let complete = Array(buffer[0..<end])
            buffer.removeFirst(end)
            return String(decoding: complete, as: UTF8.self)
        }
    }
}
