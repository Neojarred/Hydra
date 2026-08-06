import Foundation
import HydraCore
import Metal

/// Cache clé/valeur en FP16, avec deux dispositions selon le motif d'attention.
///
/// GPT-OSS alterne des couches à **fenêtre glissante de 128 tokens** (indices pairs) et
/// des couches à **attention pleine** (indices impairs). Cette asymétrie rend le contexte
/// long étonnamment bon marché : seule la moitié des couches grossit avec le contexte.
///
/// - Les couches glissantes utilisent un **anneau borné** de `slidingWindow + chunk`
///   lignes. La marge d'un chunk de prefill est nécessaire : en prefill par blocs, toutes
///   les écritures d'un bloc précèdent les attentions du bloc, donc un anneau de la seule
///   taille de la fenêtre écraserait des clés encore utiles.
/// - Les couches pleines utilisent un stockage **linéaire** dimensionné sur le contexte
///   demandé.
///
/// Pour le 120B à 128k : 4,51 Gio au total, là où des couches toutes pleines en
/// coûteraient le double.
public final class KVCache: @unchecked Sendable {

    public let config: GptOssConfig
    public let contextLength: Int
    /// Marge d'un bloc de prefill au-dessus de la fenêtre glissante.
    /// Jetons traités ensemble pendant le prefill.
    ///
    /// C'est le paramètre qui gouverne le coût du prefill, et il n'est pas anodin : un bloc
    /// sollicite quasiment tous les experts d'une couche alors que le cache n'en tient que
    /// quelques-uns, donc **chaque bloc relit le pool entier** — 10,1 Go pour le 20B. Le
    /// coût par jeton est inversement proportionnel à cette taille.
    ///
    /// Mesuré sur une invite de mille jetons, temps total jusqu'à la fin de la réponse :
    /// 128 jetons par bloc donnent 39–44 s, 512 en donnent 28–29 pour 44 Mio de plus,
    /// 1024 en donnent 25–28 pour 119 Mio. On s'arrête à 512, meilleur rapport au
    /// mégaoctet (docs/02-MESURES.md, M-019).
    ///
    /// L'anneau des couches glissantes se dimensionne sur `slidingWindow + prefillChunk` :
    /// la contrainte est déjà paramétrée, augmenter le bloc reste correct par construction.
    public static let prefillChunk = 512

    /// `MTLBuffer` n'est pas `Sendable` dans les en-têtes Metal, alors que les tampons
    /// sont sûrs à partager en lecture depuis plusieurs fils. On l'assume explicitement
    /// plutôt que de désactiver la vérification sur tout le module.
    public struct Layer: @unchecked Sendable {
        public let keys: MTLBuffer
        public let values: MTLBuffer
        /// Capacité de l'anneau, ou 0 pour un stockage linéaire.
        public let ringSize: Int
        public let capacity: Int
        /// Vrai si l'attention de cette couche est bornée à `slidingWindow` tokens.
        ///
        /// Distinct de `ringSize` : une couche peut être glissante *pour l'attention* tout
        /// en étant stockée linéairement. Confondre les deux ferait qu'un stockage linéaire
        /// rendrait l'attention pleine — le modèle changerait de comportement sans rien
        /// signaler.
        public let windowed: Bool
    }

    public private(set) var layers: [Layer]
    /// Nombre de tokens déjà écrits.
    public private(set) var length = 0

    /// Vrai si le cache peut revenir à une position antérieure.
    ///
    /// Un anneau ne le peut pas : au-delà de sa capacité il a écrasé ce qu'il faudrait
    /// retrouver. Un stockage linéaire le peut toujours, les lignes au-delà étant
    /// simplement réécrites au prochain passage.
    public var canRewind: Bool { layers.allSatisfy { $0.ringSize == 0 } }

    public enum CacheError: Error, CustomStringConvertible {
        case allocationFailed(layer: Int, bytes: Int)
        case overflow(position: Int, capacity: Int)

        public var description: String {
            switch self {
            case let .allocationFailed(layer, bytes):
                return "cache KV : allocation de \(bytes) o impossible pour la couche \(layer)"
            case let .overflow(position, capacity):
                return "cache KV : position \(position) au-delà de la capacité \(capacity)"
            }
        }
    }

    /// Contexte jusqu'auquel les couches glissantes reçoivent un stockage linéaire.
    ///
    /// Le stockage linéaire est ce qui rend le cache rembobinable, donc réutilisable d'un
    /// tour de conversation à l'autre. Il coûte, pour le 20B à 4k, 201 Mio au lieu de 31 —
    /// 170 Mio pour supprimer la quasi-totalité du temps avant réponse des tours de suite.
    /// Au-delà de ce seuil le calcul s'inverse et l'anneau reprend la main.
    public static let linearWindowLimit = 8192

    public init(config: GptOssConfig, contextLength: Int, device: MTLDevice) throws {
        self.config = config
        self.contextLength = contextLength

        let entryBytes = config.keyValueHeadCount * config.headDim * MemoryLayout<Float16>.size
        var built: [Layer] = []
        built.reserveCapacity(config.layerCount)

        for index in 0..<config.layerCount {
            let sliding = config.attentionPattern(atLayer: index) == .sliding
            let linear = contextLength <= Self.linearWindowLimit
            let ringSize = (sliding && !linear) ? config.slidingWindow + Self.prefillChunk : 0
            let capacity = ringSize > 0 ? ringSize : contextLength
            let bytes = capacity * entryBytes

            // Mémoire partagée plutôt que privée : sur Apple Silicon le coût est nul et
            // le CPU peut inspecter le cache, ce dont la validation a besoin.
            guard let keys = device.makeBuffer(length: bytes, options: .storageModeShared),
                let values = device.makeBuffer(length: bytes, options: .storageModeShared)
            else {
                throw CacheError.allocationFailed(layer: index, bytes: bytes)
            }
            built.append(Layer(
                keys: keys, values: values, ringSize: ringSize, capacity: capacity,
                windowed: sliding))
        }
        self.layers = built
    }

    /// Octets réellement alloués. Doit coïncider avec l'estimation du budget.
    public var byteCount: Int {
        let entryBytes = config.keyValueHeadCount * config.headDim * MemoryLayout<Float16>.size
        return layers.reduce(0) { $0 + 2 * $1.capacity * entryBytes }
    }

    /// Clés visibles par la requête à `position`, pour une couche donnée.
    ///
    /// Pour une couche pleine : tout l'historique. Pour une couche glissante : les
    /// `slidingWindow` derniers tokens au plus, la position de départ suivant la fenêtre.
    public func visibleRange(layer index: Int, position: Int) -> (start: Int, count: Int) {
        guard layers[index].windowed else { return (0, position + 1) }
        let start = max(0, position - config.slidingWindow + 1)
        return (start, position - start + 1)
    }

    public func advance() throws {
        length += 1
        // Seules les couches à attention pleine peuvent déborder : les anneaux des
        // couches glissantes recyclent leurs lignes par construction.
        for layer in layers where layer.ringSize == 0 {
            guard length <= layer.capacity else {
                throw CacheError.overflow(position: length - 1, capacity: layer.capacity)
            }
        }
    }

    public func reset() {
        length = 0
    }

    /// Ramène le cache à `tokens` tokens écrits.
    ///
    /// Les lignes au-delà restent en mémoire mais deviennent invisibles : rien ne les lit,
    /// et le prochain passage les réécrit. N'a de sens que sur un stockage linéaire.
    public func rewind(to tokens: Int) {
        precondition(canRewind, "un anneau ne peut pas être rembobiné")
        precondition(tokens >= 0 && tokens <= length, "position hors de l'historique écrit")
        length = tokens
    }
}
