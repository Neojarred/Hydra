import Foundation
import HydraCore
import HydraFormat
import Metal

/// Passe avant complète : embedding → couches → norme finale → tête LM → logits.
///
/// Le pas de décodage suit exactement la structure imposée par le routeur, couche après
/// couche : `cb1` jusqu'aux identifiants d'experts, lecture SSD, `cb2` pour le mélange.
/// La tête LM ne s'exécute qu'une fois, à la fin.
///
/// Aucune allocation n'a lieu pendant le décodage. Tout le scratch est réservé au
/// chargement, ce qui rend l'empreinte constante d'un token à l'autre — la propriété que
/// le projet doit démontrer.
public final class ModelRunner: @unchecked Sendable {

    public let config: GptOssConfig
    public let context: MetalContext
    public let mapping: ModelMapping
    public let expertCache: ExpertSlotCache
    public let kvCache: KVCache

    private let encoder: ForwardEncoder
    private let layerRunner: LayerRunner
    private let scratch: DecodeScratch
    private let prefillRunner: PrefillRunner
    private let prefillScratch: PrefillScratch
    private let ropeTables: RoPETables
    private let logits: MTLBuffer
    /// Logits de plusieurs positions à la fois, pour la vérification spéculative.
    /// Alloué à la première demande : il ne sert pas au décodage ordinaire.
    private var batchLogits: MTLBuffer?

    /// Position du prochain token à traiter.
    public private(set) var position = 0

    public struct Timings: Sendable {
        public var attentionAndRouter = 0.0
        public var expertIO = 0.0
        public var mixture = 0.0
        public var head = 0.0
        public var sampling = 0.0
        public var total: Double { attentionAndRouter + expertIO + mixture + head + sampling }
    }

    public private(set) var lastTimings = Timings()

    public enum RunnerError: Error, CustomStringConvertible {
        case allocationFailed(String)
        case commandBufferUnavailable

        public var description: String {
            switch self {
            case .allocationFailed(let name): return "allocation impossible : \(name)"
            case .commandBufferUnavailable: return "tampon de commandes indisponible"
            }
        }
    }

    public init(
        config: GptOssConfig, context: MetalContext, mapping: ModelMapping,
        expertCache: ExpertSlotCache, contextLength: Int,
        prefillChunk: Int = KVCache.prefillChunk
    ) throws {
        self.config = config
        self.context = context
        self.mapping = mapping
        self.expertCache = expertCache
        self.encoder = ForwardEncoder(context: context)
        self.scratch = try DecodeScratch(config: config, device: context.device)
        self.kvCache = try KVCache(
            config: config, contextLength: contextLength, device: context.device)
        self.layerRunner = LayerRunner(
            config: config, encoder: encoder, mapping: mapping, cache: expertCache)
        self.ropeTables = RoPETables(RoPETables.Parameters(config: config))

        // Le bloc ne peut pas dépasser la marge de l'anneau des couches glissantes :
        // toutes les écritures KV d'un bloc précèdent ses attentions.
        let chunk = min(prefillChunk, KVCache.prefillChunk)
        self.prefillScratch = try PrefillScratch(
            config: config, maximumTokens: chunk, device: context.device)
        self.prefillRunner = PrefillRunner(
            config: config, encoder: BatchEncoder(context: context),
            mapping: mapping, cache: expertCache)

        guard let logits = context.device.makeBuffer(
            length: config.vocabSize * MemoryLayout<Float>.size, options: .storageModeShared)
        else { throw RunnerError.allocationFailed("logits") }
        self.logits = logits
    }

    /// Mémoire réservée par le runtime, hors mappages.
    public var reservedBytes: Int {
        scratch.byteCount + prefillScratch.byteCount + kvCache.byteCount
            + expertCache.reservedBytes + config.vocabSize * MemoryLayout<Float>.size
    }

    // MARK: - Prefill par blocs

    /// Traite une invite par blocs et rend la distribution du dernier jeton.
    ///
    /// Le calcul est **identique** au traitement jeton par jeton — seul l'ordre des
    /// lectures change. Sur une invite de 78 jetons du 20B, cela ramène les relectures de
    /// poids denses de 92,9 Gio à 1,2 Gio. Un test vérifie que les deux chemins produisent
    /// le même état.
    ///
    /// La taille de bloc est bornée par la marge de l'anneau des couches à fenêtre
    /// glissante : toutes les écritures KV d'un bloc précèdent ses attentions, donc un
    /// bloc plus grand que la marge écraserait des clés encore utiles.
    @discardableResult
    public func prefill(tokens: [Int]) throws -> UnsafeBufferPointer<Float> {
        precondition(!tokens.isEmpty, "l'invite ne peut pas être vide")
        var timings = Timings()
        let chunkSize = prefillScratch.maximumTokens

        var offset = 0
        var lastChunkCount = 0
        while offset < tokens.count {
            let count = min(chunkSize, tokens.count - offset)
            try prefillChunk(
                Array(tokens[offset..<(offset + count)]),
                firstPosition: position, timings: &timings)
            lastChunkCount = count
            offset += count
        }

        if ProcessInfo.processInfo.environment["HYDRA_PROFILE"] != nil {
            FileHandle.standardError.write(Data(String(
                format: "prefill %d jetons : I/O %.1f s · calcul %.1f s · attention %.1f s\n",
                tokens.count, timings.expertIO, timings.mixture,
                timings.attentionAndRouter).utf8))
        }

        // L'état à lire est la dernière ligne du **dernier** bloc traité.
        return try finishWithLanguageModelHead(
            from: prefillScratch.hidden,
            rowOffset: (lastChunkCount - 1) * config.hiddenSize,
            timings: &timings)
    }

    private func prefillChunk(
        _ chunk: [Int], firstPosition: Int, timings: inout Timings
    ) throws {
        let count = chunk.count

        // Embeddings du bloc, une ligne par jeton.
        let hidden = prefillScratch.hidden.contents().bindMemory(
            to: Float.self, capacity: count * config.hiddenSize)
        for (index, token) in chunk.enumerated() {
            mapping.readEmbedding(
                token: token,
                into: UnsafeMutableBufferPointer(
                    start: hidden + index * config.hiddenSize, count: config.hiddenSize))
        }

        // Tables RoPE : chaque jeton a sa position.
        let halfDim = config.headDim / 2
        let cosTable = prefillScratch.cosTable.contents().bindMemory(
            to: Float.self, capacity: count * halfDim)
        let sinTable = prefillScratch.sinTable.contents().bindMemory(
            to: Float.self, capacity: count * halfDim)
        for index in 0..<count {
            ropeTables.write(
                position: firstPosition + index,
                cos: UnsafeMutableBufferPointer(start: cosTable + index * halfDim, count: halfDim),
                sin: UnsafeMutableBufferPointer(start: sinTable + index * halfDim, count: halfDim))
        }

        for layer in 0..<config.layerCount {
            var start = Date()
            guard let first = context.commandQueue.makeCommandBuffer() else {
                throw RunnerError.commandBufferUnavailable
            }
            try prefillRunner.encodeAttentionAndRouter(
                layer: layer, tokens: count, firstPosition: firstPosition,
                scratch: prefillScratch, kvCache: kvCache, in: first)
            try prefillRunner.encodeMixtureStart(
                tokens: count, scratch: prefillScratch, in: first)
            first.commit()
            first.waitUntilCompleted()
            timings.attentionAndRouter += Date().timeIntervalSince(start)

            // Les experts sont traités par **tuiles de la taille du cache**.
            //
            // Un bloc de 128 jetons sollicite souvent tous les experts d'une couche, mais
            // le cache n'en tient que quelques-uns. Les charger tous d'un coup les
            // épinglerait au-delà de sa capacité et bloquerait l'éviction. On avance donc
            // par tuiles : charger, calculer, libérer. Le nombre de slots occupés ne
            // dépasse jamais celui du décodage — c'est ce qui rend le prefill par blocs
            // neutre pour la mémoire.
            let assignments = prefillRunner.assignments(prefillScratch, tokens: count)
            let tileSize = max(1, expertCache.slotsPerLayer)

            var index = 0
            while index < assignments.count {
                let tile = Array(assignments[index..<min(index + tileSize, assignments.count)])

                start = Date()
                try expertCache.load(layer: layer, experts: tile.map(\.expert))
                timings.expertIO += Date().timeIntervalSince(start)

                start = Date()
                guard let buffer = context.commandQueue.makeCommandBuffer() else {
                    throw RunnerError.commandBufferUnavailable
                }
                for (slot, assignment) in tile.enumerated() {
                    try prefillRunner.encodeExpert(
                        layer: layer, assignment: assignment, slot: slot,
                        scratch: prefillScratch, in: buffer)
                }
                buffer.commit()
                buffer.waitUntilCompleted()
                expertCache.release(layer: layer)
                timings.mixture += Date().timeIntervalSince(start)

                index += tileSize
            }

            start = Date()
            guard let last = context.commandQueue.makeCommandBuffer() else {
                throw RunnerError.commandBufferUnavailable
            }
            try prefillRunner.encodeMixtureEnd(
                tokens: count, scratch: prefillScratch, in: last)
            last.commit()
            last.waitUntilCompleted()
            timings.mixture += Date().timeIntervalSince(start)
        }

        position += count
        for _ in 0..<count { try kvCache.advance() }
    }

    /// Norme finale et tête LM sur une seule ligne d'état caché.
    private func finishWithLanguageModelHead(
        from buffer: MTLBuffer, rowOffset: Int, timings: inout Timings
    ) throws -> UnsafeBufferPointer<Float> {
        let start = Date()
        guard let head = context.commandQueue.makeCommandBuffer() else {
            throw RunnerError.commandBufferUnavailable
        }
        let finalNorm = try mapping.residentTensor("model.norm.weight")
        try encoder.rmsNorm(
            input: buffer, inputOffset: rowOffset * MemoryLayout<Float>.size,
            scale: finalNorm.buffer, scaleOffset: finalNorm.offset,
            output: scratch.normed, size: config.hiddenSize, eps: config.rmsNormEps, in: head)

        let lmHead = try mapping.residentTensor("lm_head.weight")
        try encoder.denseProjection(
            weights: lmHead.buffer, weightsOffset: lmHead.offset,
            bias: nil, biasOffset: 0,
            input: scratch.normed, inputOffset: 0,
            output: logits, outputOffset: 0,
            rows: config.vocabSize, cols: config.hiddenSize, in: head)
        head.commit()
        head.waitUntilCompleted()
        timings.head += Date().timeIntervalSince(start)
        lastTimings = timings

        return UnsafeBufferPointer(
            start: logits.contents().bindMemory(to: Float.self, capacity: config.vocabSize),
            count: config.vocabSize)
    }

    public func reset() {
        position = 0
        kvCache.reset()
    }

    /// Réinitialise la suite pseudo-aléatoire du tirage.
    ///
    /// Volontairement distinct de `reset()` : deux générations successives sur la même
    /// invite doivent pouvoir différer, sans quoi « régénérer » redonnerait toujours le
    /// même texte. Sert à rendre une mesure reproductible.
    public func resetSampling() {
        samplerState = 0
    }

    // MARK: - Décodage spéculatif

    /// Produit un ou plusieurs jetons, en vérifiant un brouillon quand il en vaut la peine.
    ///
    /// **La sortie est identique à celle du décodage ordinaire**, jeton pour jeton, à graine
    /// égale. Ce n'est pas une approximation : le brouillon ne sert qu'à éviter du calcul.
    /// Deux propriétés le garantissent.
    ///
    /// D'abord, chaque jeton émis consomme exactement un tirage, comme sans spéculation :
    /// la suite pseudo-aléatoire est donc la même. Ensuite, les logits de la position
    /// `P+i` n'ont été calculés qu'en supposant les jetons `P..P+i-1`, et ne sont utilisés
    /// que si ces jetons ont été acceptés — donc si l'hypothèse était vraie.
    ///
    /// Le premier jeton est tiré **avant** la passe groupée. Si le brouillon se trompe dès
    /// le départ, on retombe sur un pas ordinaire sans avoir rien dépensé.
    public func step(
        from distribution: UnsafeBufferPointer<Float>,
        draft: [Int], sampling: Sampling
    ) throws -> (tokens: [Int], next: UnsafeBufferPointer<Float>) {
        let first = sample(from: distribution, using: sampling)

        // Le rembobinage est indispensable : rejeter un brouillon suppose de retirer du
        // cache KV ce qu'on vient d'y écrire.
        guard canRewind, draft.count > 1, first == draft[0],
            draft.count <= prefillScratch.maximumTokens
        else {
            return ([first], try forward(token: first))
        }

        let origin = position
        let logits = try verify(tokens: draft)

        var accepted = [first]
        for index in 1..<draft.count {
            let token = sample(from: logits[index - 1], using: sampling)
            accepted.append(token)
            if token != draft[index] {
                // Ce qui suit dans le cache a été calculé sur une hypothèse fausse.
                rewind(to: origin + index)
                return (accepted, try forward(token: token))
            }
        }
        return (accepted, logits[draft.count - 1])
    }

    // MARK: - Vérification spéculative

    /// Traite `tokens` en une passe et rend les logits de **chaque** position.
    ///
    /// C'est le cœur du décodage spéculatif. Une passe ordinaire relit tous les poids pour
    /// produire un seul jeton ; celle-ci les relit une fois pour en vérifier `n`. Les poids
    /// denses — attention, routeurs, tête LM, soit 2,88 Gio sur le 120B — ne sont lus
    /// qu'une fois au lieu de `n` fois, et les experts sollicités par plusieurs jetons du
    /// lot ne sont lus qu'une fois eux aussi.
    ///
    /// La ligne `i` du résultat prédit le jeton de la position **suivant** `tokens[i]`.
    /// L'appelant est responsable de rembobiner ce qu'il n'accepte pas.
    public func verify(tokens: [Int]) throws -> [UnsafeBufferPointer<Float>] {
        precondition(!tokens.isEmpty, "rien à vérifier")
        precondition(tokens.count <= prefillScratch.maximumTokens, "lot trop grand")

        var timings = Timings()
        let count = tokens.count
        let firstPosition = position
        try prefillChunk(tokens, firstPosition: firstPosition, timings: &timings)

        let bytes = count * config.vocabSize * MemoryLayout<Float>.size
        if batchLogits == nil || batchLogits!.length < bytes {
            guard let buffer = context.device.makeBuffer(
                length: bytes, options: .storageModeShared)
            else { throw RunnerError.allocationFailed("batchLogits") }
            batchLogits = buffer
        }
        guard let output = batchLogits else {
            throw RunnerError.allocationFailed("batchLogits")
        }

        let start = Date()
        let head = try commandBuffer()
        let batch = BatchEncoder(context: context)
        let finalNorm = try mapping.residentTensor("model.norm.weight")
        try batch.rmsNorm(
            input: prefillScratch.hidden, scale: finalNorm.buffer,
            scaleOffset: finalNorm.offset, output: prefillScratch.normed,
            size: config.hiddenSize, tokens: count, eps: config.rmsNormEps, in: head)

        let lmHead = try mapping.residentTensor("lm_head.weight")
        try batch.denseProjection(
            weights: lmHead.buffer, weightsOffset: lmHead.offset,
            bias: nil, biasOffset: 0,
            input: prefillScratch.normed, output: output,
            rows: config.vocabSize, cols: config.hiddenSize, tokens: count, in: head)
        head.commit()
        head.waitUntilCompleted()
        timings.head += Date().timeIntervalSince(start)
        lastTimings = timings

        let base = output.contents().bindMemory(
            to: Float.self, capacity: count * config.vocabSize)
        return (0..<count).map {
            UnsafeBufferPointer(
                start: base + $0 * config.vocabSize, count: config.vocabSize)
        }
    }

    /// Vrai si l'état peut revenir à une position antérieure sans être reconstruit.
    public var canRewind: Bool { kvCache.canRewind }

    /// Ramène l'état à `tokens` tokens traités.
    ///
    /// Sert à réutiliser le travail déjà fait d'un tour de conversation au suivant : le
    /// préfixe commun aux deux invites reste valide, seul ce qui diverge est recalculé.
    public func rewind(to tokens: Int) {
        precondition(tokens <= position, "on ne rembobine pas vers l'avant")
        kvCache.rewind(to: tokens)
        position = tokens
    }

    private func commandBuffer() throws -> MTLCommandBuffer {
        guard let buffer = context.commandQueue.makeCommandBuffer() else {
            throw RunnerError.commandBufferUnavailable
        }
        return buffer
    }

    /// Traite un token et rend la distribution de sortie.
    ///
    /// Le vecteur rendu pointe dans un tampon réutilisé : il est valable jusqu'au prochain
    /// appel. C'est délibéré — copier 201 088 flottants à chaque token pour rien serait
    /// le genre de gaspillage qui finit par peser.
    /// - Parameter needsLogits: mettre `false` pendant le prefill, sauf pour le dernier
    ///   jeton de l'invite. La tête LM coûte 1,08 Gio de lecture — la calculer pour un
    ///   jeton dont on ne lira jamais la distribution est du pur gaspillage.
    @discardableResult
    public func forward(
        token: Int, needsLogits: Bool = true
    ) throws -> UnsafeBufferPointer<Float> {
        var timings = Timings()

        // --- Embedding : une seule ligne lue, la table reste mappée ---
        let hiddenPointer = UnsafeMutableBufferPointer(
            start: scratch.hidden.contents().bindMemory(
                to: Float.self, capacity: config.hiddenSize),
            count: config.hiddenSize)
        mapping.readEmbedding(token: token, into: hiddenPointer)

        // --- Tables RoPE de la position courante ---
        ropeTables.write(
            position: position,
            cos: UnsafeMutableBufferPointer(
                start: scratch.cosTable.contents().bindMemory(
                    to: Float.self, capacity: config.headDim / 2),
                count: config.headDim / 2),
            sin: UnsafeMutableBufferPointer(
                start: scratch.sinTable.contents().bindMemory(
                    to: Float.self, capacity: config.headDim / 2),
                count: config.headDim / 2))

        // Le seul point de synchronisation incompressible est la lecture du routeur : le
        // CPU doit connaître les experts choisis avant de pouvoir lire leurs poids. Tout
        // le reste peut tenir dans le même tampon de commandes.
        //
        // Le décodage faisait auparavant sept allers-retours par couche — attention, début
        // de mélange, un par expert, fin. À 90 µs l'aller-retour à vide, cela représentait
        // 168 attentes par jeton pour 24 couches. En fusionnant le mélange d'une couche
        // avec l'attention de la suivante, il n'en reste qu'une par couche.
        //
        // Le recouvrement des lectures avec le calcul a été essayé puis retiré, sur les
        // deux modèles : `ExpertSlotCache.load` lit déjà les `top_k` experts en parallèle,
        // donc ils arrivent ensemble. Il n'y a aucune disponibilité échelonnée à exploiter,
        // et les étaler pour en créer une coûterait le parallélisme de lecture — 3,0 Go/s
        // au lieu de 5,7 (docs/02-MESURES.md, M-022).
        var start = Date()
        let opening = try commandBuffer()
        try layerRunner.encodeAttentionAndRouter(
            layer: 0, position: position, scratch: scratch, kvCache: kvCache, in: opening)
        opening.commit()
        opening.waitUntilCompleted()
        timings.attentionAndRouter += Date().timeIntervalSince(start)

        var selected = layerRunner.selectedExperts(scratch)

        for layer in 0..<config.layerCount {
            let next = layer + 1

            // On calcule d'abord les experts **déjà en mémoire**, pendant que les
            // manquants se lisent.
            //
            // Avec 76 % de hit, il ne manque en moyenne qu'un expert sur quatre : trois
            // sont prêts et n'attendent que le GPU. Les attendre tous avant de commencer
            // laissait le processeur graphique inactif pendant toute la lecture.
            //
            // L'ordre de calcul n'affecte pas le résultat : chaque expert écrit dans sa
            // propre case, et la somme se fait ensuite dans l'ordre fixe des slots.
            let ready = selected.enumerated().filter {
                expertCache.isResident(layer: layer, expert: $0.element)
            }
            let awaited = selected.enumerated().filter {
                !expertCache.isResident(layer: layer, expert: $0.element)
            }

            start = Date()

            // L'encodage épingle les experts qu'il référence. Il doit donc précéder le
            // lancement des lectures : sinon celles-ci choisissent comme victimes les
            // slots encore libres — c'est-à-dire précisément ceux qu'on s'apprêtait à
            // utiliser. Mesuré : le taux de hit tombait de 76 à 64 %.
            var warm: MTLCommandBuffer?
            if !ready.isEmpty {
                let buffer = try commandBuffer()
                for (slot, expert) in ready {
                    try layerRunner.encodeSingleExpert(
                        layer: layer, expert: expert, weightIndex: slot,
                        scratch: scratch, in: buffer)
                }
                warm = buffer
            }

            if !awaited.isEmpty {
                let cache = expertCache
                let layerIndex = layer
                let missing = awaited.map(\.element)
                DispatchQueue.global(qos: .userInitiated).async {
                    try? cache.load(layer: layerIndex, experts: missing)
                }
            }

            if let warm {
                warm.commit()
                warm.waitUntilCompleted()
            }
            timings.mixture += Date().timeIntervalSince(start)

            start = Date()
            let buffer = try commandBuffer()
            for (slot, expert) in awaited {
                // Bloque sur cet expert : la lecture lancée plus haut a couru pendant
                // le calcul des experts déjà chauds.
                try layerRunner.encodeSingleExpert(
                    layer: layer, expert: expert, weightIndex: slot,
                    scratch: scratch, in: buffer)
            }
            timings.expertIO += Date().timeIntervalSince(start)

            start = Date()
            try layerRunner.encodeCombineSlices(
                count: selected.count, scratch: scratch, in: buffer)

            // Les encodeurs d'un même tampon s'exécutent dans l'ordre où ils ont été
            // créés : l'attention de la couche suivante lira bien le résidu que le
            // mélange vient d'écrire.
            if next < config.layerCount {
                try layerRunner.encodeAttentionAndRouter(
                    layer: next, position: position, scratch: scratch, kvCache: kvCache,
                    in: buffer)
            }
            buffer.commit()
            buffer.waitUntilCompleted()
            expertCache.release(layer: layer)
            timings.mixture += Date().timeIntervalSince(start)

            if next < config.layerCount {
                selected = layerRunner.selectedExperts(scratch)
            }
        }

        // --- Norme finale et tête LM ---
        start = Date()
        guard needsLogits else {
            position += 1
            try kvCache.advance()
            lastTimings = timings
            return UnsafeBufferPointer(
                start: logits.contents().bindMemory(to: Float.self, capacity: config.vocabSize),
                count: config.vocabSize)
        }
        guard let head = context.commandQueue.makeCommandBuffer() else {
            throw RunnerError.commandBufferUnavailable
        }
        let finalNorm = try mapping.residentTensor("model.norm.weight")
        try encoder.rmsNorm(
            input: scratch.hidden, scale: finalNorm.buffer, scaleOffset: finalNorm.offset,
            output: scratch.normed, size: config.hiddenSize, eps: config.rmsNormEps, in: head)

        let lmHead = try mapping.residentTensor("lm_head.weight")
        try encoder.denseProjection(
            weights: lmHead.buffer, weightsOffset: lmHead.offset,
            bias: nil, biasOffset: 0,
            input: scratch.normed, inputOffset: 0,
            output: logits, outputOffset: 0,
            rows: config.vocabSize, cols: config.hiddenSize, in: head)
        head.commit()
        head.waitUntilCompleted()
        timings.head = Date().timeIntervalSince(start)

        position += 1
        try kvCache.advance()
        lastTimings = timings

        return UnsafeBufferPointer(
            start: logits.contents().bindMemory(to: Float.self, capacity: config.vocabSize),
            count: config.vocabSize)
    }

    /// Paramètres d'échantillonnage.
    ///
    /// OpenAI recommande `temperature = 1.0` et `top_p = 1.0` pour GPT-OSS — autrement dit
    /// la distribution brute, sans troncature. C'est inhabituel : la plupart des modèles
    /// demandent un réglage plus serré. On garde donc ces valeurs par défaut plutôt que
    /// d'imposer les habitudes prises ailleurs.
    public struct Sampling: Sendable {
        public var temperature: Float
        public var topP: Float
        public var seed: UInt64

        public init(temperature: Float = 1.0, topP: Float = 1.0, seed: UInt64 = 0x5EED_1234) {
            self.temperature = temperature
            self.topP = topP
            self.seed = seed
        }

        /// Décodage déterministe, pour comparer deux exécutions.
        public static let greedy = Sampling(temperature: 0)
    }

    private var samplerState: UInt64 = 0

    /// Tire un token dans la distribution.
    ///
    /// `temperature = 0` bascule sur le décodage glouton, ce qui rend l'exécution
    /// reproductible — indispensable pour vérifier qu'un changement de taille de cache
    /// ne modifie pas les sorties.
    public func sample(
        from distribution: UnsafeBufferPointer<Float>, using sampling: Sampling
    ) -> Int {
        guard sampling.temperature > 0 else { return greedyToken(from: distribution) }

        // Rien n'est alloué à la taille du vocabulaire.
        //
        // La version précédente construisait deux tableaux de 201 088 entrées à chaque
        // jeton — les probabilités et l'ordre — soit 2,4 Mio à allouer, remplir puis
        // jeter, avant de trier le tout pour n'en garder qu'une trentaine. Ici la somme
        // se calcule en une passe sans rien stocker, et seuls les candidats retenus sont
        // matérialisés.
        // Le maximum sort de la sélection elle-même : le plus grand logit est le premier
        // candidat. Une passe complète sur le vocabulaire en moins.
        let candidates = sampling.topP < 1.0
            ? Self.largestIndices(distribution, count: 64) : []
        var peak = -Float.greatestFiniteMagnitude
        if let best = candidates.first {
            peak = distribution[best]
        } else {
            for value in distribution { peak = max(peak, value) }
        }

        let inverseTemperature = 1 / sampling.temperature
        var total: Float = 0
        for value in distribution { total += exp((value - peak) * inverseTemperature) }

        if samplerState == 0 { samplerState = sampling.seed | 1 }
        samplerState &+= 0x9E37_79B9_7F4A_7C15
        var z = samplerState
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        z = z ^ (z >> 31)
        let uniform = Float(Double(z % 1_000_000) / 1_000_000.0)

        guard sampling.topP < 1.0 else {
            // Sans troncature, un simple parcours suffit : l'ordre d'énumération ne change
            // pas la loi tirée.
            let target = uniform * total
            var cumulative: Float = 0
            for (index, value) in distribution.enumerated() {
                cumulative += exp((value - peak) * inverseTemperature)
                if cumulative >= target { return index }
            }
            return distribution.count - 1
        }

        // Le noyau top-p ne contient qu'une poignée de jetons. On en extrait un petit
        // paquet, élargi seulement si la masse visée n'est pas atteinte.
        var limit = 64
        var nucleus: [Int] = []
        var mass: Float = 0
        var pool = candidates
        while true {
            nucleus = []
            mass = 0
            var reached = false
            for index in pool {
                nucleus.append(index)
                mass += exp((distribution[index] - peak) * inverseTemperature)
                if mass / total >= sampling.topP { reached = true; break }
            }
            if reached || limit >= distribution.count { break }
            limit = min(limit * 4, distribution.count)
            pool = Self.largestIndices(distribution, count: limit)
        }

        let target = uniform * mass
        var cumulative: Float = 0
        for index in nucleus {
            cumulative += exp((distribution[index] - peak) * inverseTemperature)
            if cumulative >= target { return index }
        }
        return nucleus.last ?? 0
    }

    /// Les `count` plus grandes valeurs, par ordre décroissant.
    ///
    /// Tas-min de taille `count` : une seule passe sur le vocabulaire, et la seule
    /// allocation est celle du tas — quelques dizaines d'entrées, pas deux cent mille.
    static func largestIndices(
        _ values: UnsafeBufferPointer<Float>, count: Int
    ) -> [Int] {
        let size = min(count, values.count)
        guard size > 0 else { return [] }

        var heapValues = [Float](repeating: 0, count: size)
        var heapIndices = [Int](repeating: 0, count: size)
        var filled = 0

        func siftDown(_ start: Int) {
            var parent = start
            while true {
                let left = 2 * parent + 1
                guard left < filled else { return }
                var smallest = left
                let right = left + 1
                if right < filled, heapValues[right] < heapValues[left] { smallest = right }
                if heapValues[parent] <= heapValues[smallest] { return }
                heapValues.swapAt(parent, smallest)
                heapIndices.swapAt(parent, smallest)
                parent = smallest
            }
        }

        for index in 0..<values.count {
            let value = values[index]
            if filled < size {
                heapValues[filled] = value
                heapIndices[filled] = index
                filled += 1
                // Remontée depuis la feuille insérée.
                var child = filled - 1
                while child > 0 {
                    let parent = (child - 1) / 2
                    if heapValues[parent] <= heapValues[child] { break }
                    heapValues.swapAt(parent, child)
                    heapIndices.swapAt(parent, child)
                    child = parent
                }
            } else if value > heapValues[0] {
                heapValues[0] = value
                heapIndices[0] = index
                siftDown(0)
            }
        }

        return heapIndices[0..<filled].sorted { values[$0] > values[$1] }
    }

    public func greedyToken(from distribution: UnsafeBufferPointer<Float>) -> Int {
        var bestIndex = 0
        var bestValue = -Float.greatestFiniteMagnitude
        for (index, value) in distribution.enumerated() where value > bestValue {
            bestValue = value
            bestIndex = index
        }
        return bestIndex
    }

    /// Génère `count` tokens à partir d'une amorce, en décodage glouton.
    ///
    /// L'amorce est traitée token par token — le prefill par blocs viendra plus tard, il
    /// n'apporte rien tant qu'on ne cherche pas le débit sur de longs prompts.
    public func generate(
        prompt: [Int], count: Int,
        onToken: ((Int, Timings) -> Void)? = nil
    ) throws -> [Int] {
        precondition(!prompt.isEmpty, "l'amorce ne peut pas être vide")

        var distribution = try forward(token: prompt[0], needsLogits: prompt.count == 1)
        for (index, token) in prompt.dropFirst().enumerated() {
            distribution = try forward(token: token, needsLogits: index == prompt.count - 2)
        }

        var produced: [Int] = []
        produced.reserveCapacity(count)
        for step in 0..<count {
            let next = greedyToken(from: distribution)
            produced.append(next)
            onToken?(next, lastTimings)
            // Le dernier token produit n'a pas besoin d'être réinjecté.
            if step < count - 1 { distribution = try forward(token: next) }
        }
        return produced
    }
}
