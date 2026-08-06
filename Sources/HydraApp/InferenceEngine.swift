import Foundation
import HydraCore
import HydraInstall
import HydraMetal
import HydraTokenize

/// Exécute l'inférence hors du fil principal et rend compte au fur et à mesure.
///
/// Le runtime est bloquant par construction : chaque couche valide un tampon de commandes
/// et attend. Le faire tourner sur le fil principal figerait l'interface pendant toute la
/// génération. On isole donc l'exécution sur une file dédiée, et les évènements
/// remontent sur le fil principal.
public final class InferenceEngine: @unchecked Sendable {

    public struct Loaded: Sendable {
        public let entry: CatalogEntry
        public let contextLength: Int
        public let slotsPerLayer: Int
        /// Poids mappés, adossés au fichier.
        public let mappedBytes: Int
        /// Mémoire réservée par le runtime : slots d'experts, KV, scratch, logits.
        public let reservedBytes: Int
    }

    /// Échec de chargement, porteur d'un message lisible.
    public struct LoadFailure: Error, CustomStringConvertible, Sendable {
        public let description: String
    }

    public enum Event: Sendable {
        /// Émis dès l'invite encodée : donne l'occupation du contexte avant génération.
        case started(promptTokens: Int, contextLength: Int)
        /// Latence avant le premier jeton visible, prefill compris.
        case firstToken(seconds: Double)
        case reasoning(String)
        case text(String)
        case finished(tokens: Int, seconds: Double, contextUsed: Int)
        case failed(String)
    }

    private let queue = DispatchQueue(label: "hydra.inference", qos: .userInitiated)
    private var context: MetalContext?
    private var runner: ModelRunner?
    private var tokenizer: BPETokenizer?
    private var mapping: ModelMapping?
    private(set) public var loaded: Loaded?

    private let cancelled = Flag()

    public init() {}

    // MARK: - Chargement

    public func load(
        entry: CatalogEntry, contextLength: Int, slotsPerLayer: Int?,
        progress: @escaping @Sendable (String) -> Void,
        completion: @escaping @Sendable (Result<Loaded, LoadFailure>) -> Void
    ) {
        queue.async { [self] in
            do {
                let root = try ModelLocations.root(for: entry)
                progress("Reading the tokenizer…")
                let tokenizer = try TokenizerInstaller.load(from: root)

                progress("Initializing the GPU…")
                let context = try self.context ?? MetalContext()
                self.context = context

                let profile = context.hardwareProfile(
                    memoryBandwidth: 94e9, diskBandwidth: 5.5e9)
                let policy: ExpertCachePolicy =
                    slotsPerLayer.map { .slotsPerLayer($0) } ?? .minimal
                let budget = MemoryBudget(
                    config: entry.config, hardware: profile,
                    contextLength: contextLength, policy: policy)

                progress("Opening the weights…")
                let mapping = try ModelMapping(
                    root: root, config: entry.config, device: context.device)
                let cache = ExpertSlotCache(
                    root: root, config: entry.config,
                    slotsPerLayer: budget.expertSlotsPerLayer, device: context.device)
                let runner = try ModelRunner(
                    config: entry.config, context: context, mapping: mapping,
                    expertCache: cache, contextLength: contextLength)

                // Faire entrer les pages en une lecture séquentielle plutôt que par
                // défauts dispersés pendant la première génération : mesuré, cela
                // divisait par deux le temps de la première invite.
                progress("Prefaulting the weights…")
                mapping.prefault()

                self.tokenizer = tokenizer
                self.runner = runner
                self.mapping = mapping
                let loaded = Loaded(
                    entry: entry, contextLength: contextLength,
                    slotsPerLayer: budget.expertSlotsPerLayer,
                    mappedBytes: mapping.mappedByteCount,
                    reservedBytes: runner.reservedBytes)
                self.loaded = loaded
                completion(.success(loaded))
            } catch {
                completion(.failure(LoadFailure(description: "\(error)")))
            }
        }
    }

    public func unload() {
        queue.async { [self] in
            runner = nil
            mapping = nil
            tokenizer = nil
            loaded = nil
        }
    }

    // MARK: - Génération

    public func cancel() { cancelled.set(true) }

    public func generate(
        turns: [Harmony.Turn], settings: GenerationSettings,
        onEvent: @escaping @Sendable (Event) -> Void
    ) {
        cancelled.set(false)
        queue.async { [self] in
            guard let runner, let tokenizer else {
                onEvent(.failed("no model loaded"))
                return
            }
            do {
                let renderer = Harmony.Renderer(
                    reasoningEffort: settings.effort,
                    instructions: settings.instructions.isEmpty ? nil : settings.instructions)
                let prompt = tokenizer.encode(renderer.render(turns: turns), allowSpecial: true)
                onEvent(.started(
                    promptTokens: prompt.count, contextLength: runner.kvCache.contextLength))

                let started = Date()

                // Le travail du tour précédent est réutilisé.
                //
                // Chaque tour re-rend la conversation entière : au dixième échange l'invite
                // fait des milliers de jetons dont trois sont nouveaux, et les reprefiller
                // tous refait un calcul déjà fait — seize secondes mesurées sur mille
                // jetons. On ne garde du cache que le préfixe commun aux deux invites, et
                // on ne calcule que la suite.
                //
                // `cachedTokens` est la suite exacte passée au modèle, réponse comprise :
                // c'est elle qui décrit le contenu du cache, pas l'invite seule. Une
                // modification ou une régénération raccourcit le préfixe commun, et la
                // reprise se fait d'elle-même au bon endroit.
                var reusable = 0
                if runner.canRewind {
                    reusable = min(
                        commonPrefixLength(cachedTokens, prompt), runner.position)
                    // Il faut au moins un jeton à traiter pour obtenir une distribution.
                    if reusable >= prompt.count { reusable = max(0, prompt.count - 1) }
                }

                if reusable > 0 {
                    runner.rewind(to: reusable)
                } else {
                    runner.reset()
                }

                var fed = Array(prompt[0..<reusable])
                let remaining = Array(prompt[reusable...])
                var distribution = try runner.prefill(tokens: remaining)
                fed += remaining
                let prefilled = Date()

                let parser = Harmony.Parser(tokenizer: tokenizer)
                var session = Harmony.Parser.Session()
                let sampling = ModelRunner.Sampling(
                    temperature: Float(settings.temperature), topP: Float(settings.topP))

                // Les fragments sont regroupés avant d'être publiés.
                //
                // Un jeton produit un fragment, et chaque fragment déclenchait un rendu
                // complet de la conversation : reparse du Markdown sur tout le message,
                // remise en page, défilement animé. Ce travail s'exécute sur le fil
                // principal, mais il consomme la bande passante mémoire — celle-là même
                // qui limite le décodage. Le débit tombait de 9,2 à 5,4 jetons/s.
                //
                // À vingt rafraîchissements par seconde le texte défile de façon fluide à
                // l'œil, pour une fraction des rendus.
                var pendingText = ""
                var pendingReasoning = ""
                var lastFlush = Date()

                func flush(force: Bool = false) {
                    guard force || Date().timeIntervalSince(lastFlush) >= 0.05 else { return }
                    if !pendingText.isEmpty {
                        onEvent(.text(pendingText))
                        pendingText = ""
                    }
                    if !pendingReasoning.isEmpty {
                        onEvent(.reasoning(pendingReasoning))
                        pendingReasoning = ""
                    }
                    lastFlush = Date()
                }

                // Le budget est borné par ce qui reste de contexte : le dépasser ferait
                // déborder le cache KV et échouer la génération au milieu d'une phrase.
                // On garde une marge pour les marqueurs de fin de Harmony.
                let room = runner.kvCache.contextLength - prompt.count - 8
                let budget = max(1, min(settings.maximumTokens, room))

                // Décodage spéculatif : les brouillons viennent de l'invite elle-même.
                //
                // Une passe ordinaire relit tous les poids pour un seul jeton. Vérifier
                // quatre candidats en une passe les relit une seule fois — les poids
                // denses comme les experts partagés. Le brouillon ne coûte rien : c'est
                // une recherche de motif dans ce qui a déjà été écrit.
                //
                // La sortie est identique jeton pour jeton, brouillon juste ou faux : un
                // candidat rejeté ne fait que gaspiller du calcul. Quatre tests le
                // vérifient sur les deux modes de tirage.
                // `HYDRA_NOSPEC` désactive la spéculation, pour la mesure appariée.
                let speculates = ProcessInfo.processInfo.environment["HYDRA_NOSPEC"] == nil
                let drafter = NGramDrafter()
                var history = fed

                var produced = 0
                var announcedFirstToken = false
                var producedFinalText = false

                outer: while produced < budget && !session.isFinished {
                    if cancelled.value { break }

                    let draft = speculates ? drafter.propose(history: history) : []
                    let outcome = try runner.step(
                        from: distribution, draft: draft, sampling: sampling)
                    distribution = outcome.next

                    for token in outcome.tokens {
                        produced += 1
                        history.append(token)
                        fed.append(token)

                        for event in parser.consume(token, session: &session) {
                            switch event {
                            case .text(let channel, let fragment):
                                if !announcedFirstToken {
                                    announcedFirstToken = true
                                    onEvent(.firstToken(
                                        seconds: Date().timeIntervalSince(started)))
                                }
                                switch channel {
                                case .final:
                                    pendingText += fragment
                                    producedFinalText = true
                                case .analysis: pendingReasoning += fragment
                                case .commentary: break
                                }
                                flush()
                            case .channelEnded, .stopped:
                                break
                            }
                        }
                        // Un lot peut contenir des jetons au-delà du marqueur de fin. Ils
                        // restent dans le cache KV mais pas dans `fed`, qui décrit ce qui
                        // a été retenu : le tour suivant rembobinera jusqu'au préfixe
                        // commun, donc en deçà, et ils disparaîtront d'eux-mêmes.
                        if session.isFinished || produced >= budget { break outer }
                    }
                }
                cachedTokens = fed
                flush(force: true)

                // S'arrêter sur le budget sans avoir rien écrit laisse l'utilisateur devant
                // un raisonnement suivi de rien. Le dire vaut mieux que le laisser deviner.
                if produced >= budget && !producedFinalText {
                    onEvent(.text(
                        room <= settings.maximumTokens
                            ? "_(stopped: only \(room) tokens of context were left. "
                                + "Start a new conversation or raise the context length.)_"
                            : "_(stopped: reasoning used up all \(budget) allowed "
                                + "tokens. Lower the reasoning effort.)_"))
                }
                // Le débit se mesure sur le décodage seul.
                //
                // Il comptait jusqu'ici depuis le début du prefill. Sur une conversation
                // établie, l'invite fait plusieurs milliers de jetons et son traitement
                // pèse plus lourd que toute la réponse : le même moteur affichait 6
                // jetons/s sur une conversation neuve et 4 sur une conversation chargée,
                // alors qu'il décodait exactement à la même vitesse. Le coût du prefill
                // n'est pas caché pour autant — c'est ce que mesure le temps avant réponse,
                // juste à côté.
                onEvent(.finished(
                    tokens: produced, seconds: Date().timeIntervalSince(prefilled),
                    contextUsed: prompt.count + produced))
            } catch {
                // L'état du cache n'est plus connu avec certitude : on l'oublie plutôt que
                // de risquer de réutiliser un préfixe qui ne correspond à rien.
                cachedTokens = []
                onEvent(.failed("\(error)"))
            }
        }
    }

    /// Suite exacte des jetons actuellement représentés dans le cache KV.
    ///
    /// N'est touchée que depuis la file d'inférence, qui est sérielle.
    private var cachedTokens: [Int] = []

    /// Drapeau partagé entre le fil principal et la file d'inférence.
    private final class Flag: @unchecked Sendable {
        private var storage = false
        private let lock = NSLock()
        func set(_ value: Bool) {
            lock.lock()
            storage = value
            lock.unlock()
        }
        var value: Bool {
            lock.lock()
            defer { lock.unlock() }
            return storage
        }
    }
}
