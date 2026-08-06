import Foundation

/// Motif d'attention d'une couche.
public enum AttentionPattern: String, Sendable, Codable {
    /// Attention causale bornée à une fenêtre glissante.
    case sliding
    /// Attention causale sur tout le contexte.
    case full
}

/// Configuration de GPT-OSS, transcrite depuis les `config.json` réels des dépôts
/// `openai/gpt-oss-20b` et `openai/gpt-oss-120b`.
///
/// Conformément au brief, ceci est une **structure concrète, pas un contrat de modèle
/// générique**. L'abstraction ne sera extraite qu'en phase 3, à partir de deux moteurs
/// qui fonctionnent (docs/00-ETUDE-DE-FAISABILITE.md, §6).
public struct GptOssConfig: Sendable, Equatable {

    public let name: String
    public let layerCount: Int
    public let expertCount: Int

    // Les valeurs par défaut sont celles, identiques, du 20B et du 120B. Elles sont
    // paramétrables uniquement pour permettre des configurations miniatures en test :
    // vérifier le repack sur un modèle réel demanderait de télécharger 12,8 Gio, alors
    // que la logique à valider est purement structurelle.
    public let expertsPerToken: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let headDim: Int
    public let attentionHeadCount: Int
    public let keyValueHeadCount: Int
    public let vocabSize: Int
    public let slidingWindow: Int
    public let ropeTheta: Float
    public let rmsNormEps: Float
    public let swigluLimit: Float
    public let maxPositionEmbeddings: Int

    // YaRN : base 4096 étendue à 131072.
    public let yarnFactor: Float
    public let yarnBetaFast: Float
    public let yarnBetaSlow: Float
    public let yarnOriginalContext: Int

    public static let b20 = GptOssConfig(name: "GPT-OSS 20B", layerCount: 24, expertCount: 32)
    public static let b120 = GptOssConfig(name: "GPT-OSS 120B", layerCount: 36, expertCount: 128)

    /// Configuration miniature, réservée aux tests. Conserve tous les invariants
    /// structurels du vrai modèle — multiples de la taille de bloc MXFP4, GQA, alternance
    /// des motifs d'attention — pour quelques dizaines de kio au lieu de 12,8 Gio.
    public static let tiny = GptOssConfig(
        name: "GPT-OSS tiny (test)", layerCount: 4, expertCount: 6,
        expertsPerToken: 2, hiddenSize: 64, intermediateSize: 64,
        headDim: 16, attentionHeadCount: 4, keyValueHeadCount: 2,
        vocabSize: 128, slidingWindow: 8)

    public init(
        name: String,
        layerCount: Int,
        expertCount: Int,
        expertsPerToken: Int = 4,
        hiddenSize: Int = 2880,
        intermediateSize: Int = 2880,
        headDim: Int = 64,
        attentionHeadCount: Int = 64,
        keyValueHeadCount: Int = 8,
        vocabSize: Int = 201_088,
        slidingWindow: Int = 128,
        ropeTheta: Float = 150_000,
        rmsNormEps: Float = 1e-5,
        swigluLimit: Float = 7.0,
        maxPositionEmbeddings: Int = 131_072,
        yarnFactor: Float = 32,
        yarnBetaFast: Float = 32,
        yarnBetaSlow: Float = 1,
        yarnOriginalContext: Int = 4096
    ) {
        precondition(hiddenSize % MXFP4Layout.blockSize == 0)
        precondition(intermediateSize % MXFP4Layout.blockSize == 0)
        precondition(attentionHeadCount % keyValueHeadCount == 0)
        self.name = name
        self.layerCount = layerCount
        self.expertCount = expertCount
        self.expertsPerToken = expertsPerToken
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.headDim = headDim
        self.attentionHeadCount = attentionHeadCount
        self.keyValueHeadCount = keyValueHeadCount
        self.vocabSize = vocabSize
        self.slidingWindow = slidingWindow
        self.ropeTheta = ropeTheta
        self.rmsNormEps = rmsNormEps
        self.swigluLimit = swigluLimit
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.yarnFactor = yarnFactor
        self.yarnBetaFast = yarnBetaFast
        self.yarnBetaSlow = yarnBetaSlow
        self.yarnOriginalContext = yarnOriginalContext
    }

    /// `layer_types` alterne à partir de `sliding_attention` en couche 0.
    public func attentionPattern(atLayer index: Int) -> AttentionPattern {
        index.isMultiple(of: 2) ? .sliding : .full
    }

    public var fullAttentionLayerCount: Int { layerCount / 2 }
    public var slidingAttentionLayerCount: Int { layerCount - fullAttentionLayerCount }

    /// Groupe GQA : nombre de têtes de requête partageant une tête clé/valeur.
    public var groupedQueryFactor: Int { attentionHeadCount / keyValueHeadCount }

    // MARK: - Tailles exactes, dérivées des en-têtes safetensors

    private func bf16(_ dims: Int...) -> Int { dims.reduce(2, *) }

    /// Mise en page d'un blob d'expert. Source de vérité unique, partagée par le format
    /// sur disque et le dimensionnement des slots mémoire.
    public var expertBlobLayout: ExpertBlobLayout { ExpertBlobLayout(config: self) }

    /// Taille d'un blob d'expert MXFP4 **dans le checkpoint source**, biais BF16 inclus.
    /// Vaut 13 236 480 octets pour les deux modèles.
    public var expertBlobBytes: Int { expertBlobLayout.sourceBytes }

    /// Taille d'un slot en mémoire : le blob mis en page, aligné sur la page.
    /// C'est cette valeur, et pas `expertBlobBytes`, qui dimensionne le cache.
    public var expertSlotBytes: Int { expertBlobLayout.strideBytes }

    /// Pool complet des experts routés, tel qu'il vit sur le disque.
    public var expertPoolBytes: Int { layerCount * expertCount * expertBlobBytes }

    /// Poids d'attention, routeur et normes d'une couche. Tous en BF16.
    public var residentPerLayerBytes: Int {
        let qDim = attentionHeadCount * headDim
        let kvDim = keyValueHeadCount * headDim
        let q = bf16(qDim, hiddenSize) + bf16(qDim)
        let k = bf16(kvDim, hiddenSize) + bf16(kvDim)
        let v = k
        let o = bf16(hiddenSize, qDim) + bf16(hiddenSize)
        let sinks = bf16(attentionHeadCount)
        let router = bf16(expertCount, hiddenSize) + bf16(expertCount)
        let norms = 2 * bf16(hiddenSize)
        return q + k + v + o + sinks + router + norms
    }

    /// Table d'embedding. Volontairement **exclue** des poids résidents : on n'en lit
    /// qu'une ligne par token, elle reste donc mappée et paginée à la demande plutôt que
    /// câblée dans le working set Metal (§2.2c de l'étude de faisabilité).
    public var embeddingBytes: Int { bf16(vocabSize, hiddenSize) }

    /// Tête LM. Lue intégralement à chaque token : elle, doit rester résidente.
    public var lmHeadBytes: Int { bf16(vocabSize, hiddenSize) }

    /// Poids qui doivent occuper le working set Metal en permanence.
    public var residentBytes: Int {
        lmHeadBytes + layerCount * residentPerLayerBytes + bf16(hiddenSize)
    }

    /// Taille d'une installation complète sur disque.
    public var installedBytes: Int {
        expertPoolBytes + residentBytes + embeddingBytes
    }

    // MARK: - KV cache

    /// Octets de KV par token et par couche full-attention, en FP16.
    public var kvBytesPerTokenPerFullLayer: Int { 2 * keyValueHeadCount * headDim * 2 }

    /// Lignes physiques d'un anneau de couche à fenêtre glissante.
    /// La fenêtre fait 128 tokens ; on ajoute une marge d'un chunk de prefill.
    public var slidingRingRows: Int { slidingWindow + 128 }

    /// Taille totale du KV cache FP16 pour un contexte donné.
    public func kvCacheBytes(contextLength: Int) -> Int {
        let full = fullAttentionLayerCount * kvBytesPerTokenPerFullLayer * contextLength
        let sliding = slidingAttentionLayerCount * kvBytesPerTokenPerFullLayer * slidingRingRows
        return full + sliding
    }

    // MARK: - Volumes par token

    /// Octets que le GPU doit faire transiter pour décoder un token, cache d'experts
    /// parfait inclus : les poids d'experts sélectionnés sont lus quoi qu'il arrive.
    public var gpuBytesPerDecodedToken: Int {
        layerCount * residentPerLayerBytes
            + lmHeadBytes
            + layerCount * expertsPerToken * expertBlobBytes
    }

    /// Octets à lire sur le SSD pour un token, en fonction du taux de hit du cache.
    public func diskBytesPerDecodedToken(cacheHitRate: Double) -> Int {
        let all = layerCount * expertsPerToken * expertBlobBytes
        return Int(Double(all) * (1.0 - cacheHitRate).clamped(to: 0...1))
    }
}

/// Constantes du layout MXFP4, dupliquées ici pour éviter que HydraCore dépende de
/// HydraFormat. Les deux définitions sont verrouillées par un test de cohérence.
public enum MXFP4Layout {
    public static let blockSize = 32
    public static let packedBytesPerBlock = 16
    public static let scaleBytesPerBlock = 1
}

extension Comparable {
    fileprivate func clamped(to range: ClosedRange<Self>) -> Self {
        min(max(self, range.lowerBound), range.upperBound)
    }
}
