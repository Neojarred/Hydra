import Foundation

/// Caractéristiques matérielles nécessaires au dimensionnement.
///
/// `HydraCore` n'importe pas Metal (D-002) : ce profil est **injecté** par la couche
/// plateforme, qui interroge `MTLDevice` à l'exécution. Aucun chemin de production ne
/// doit dépendre des valeurs par défaut — elles n'existent que pour l'outillage hors
/// ligne et les tests, et décrivent une machine parmi d'autres, pas *la* machine.
public struct HardwareProfile: Sendable, Equatable {

    /// `MTLDevice.recommendedMaxWorkingSetSize`, ou son équivalent sur une autre plateforme.
    public var metalWorkingSetCeiling: Int

    /// Bande passante mémoire GPU en lecture streaming, en octets/s.
    public var memoryBandwidth: Double

    /// Bande passante de stockage sur des lectures de la taille d'un blob d'expert,
    /// émises en parallèle, cache de pages désactivé. En octets/s.
    public var diskBandwidth: Double

    public init(metalWorkingSetCeiling: Int, memoryBandwidth: Double, diskBandwidth: Double) {
        self.metalWorkingSetCeiling = metalWorkingSetCeiling
        self.memoryBandwidth = memoryBandwidth
        self.diskBandwidth = diskBandwidth
    }

    /// Machine de développement de référence : MacBook Apple M4, 10 cœurs GPU, 24 Gio.
    /// Toutes les valeurs sont mesurées — voir docs/00-FEASIBILITY.md, §1.
    /// **À n'utiliser que pour l'outillage hors ligne.**
    public static let appleM4_24GB = HardwareProfile(
        metalWorkingSetCeiling: 19_069_665_280,  // 17,76 Gio
        memoryBandwidth: 94e9,
        diskBandwidth: 5.5e9
    )
}

/// Comment dimensionner le cache d'experts.
///
/// L'objet du projet est de **réduire** l'empreinte mémoire, pas de la remplir. Le nombre
/// de slots n'est donc jamais « tout ce que le plafond matériel autorise » par défaut :
/// c'est une politique explicite, choisie en connaissance de cause.
public enum ExpertCachePolicy: Sendable, Equatable {

    /// Le strict minimum pour décoder : un slot par expert sélectionné à chaque couche.
    /// Empreinte plancher, taux de hit quasi nul, débit minimal. C'est la configuration
    /// qui démontre la thèse du projet.
    case minimal

    /// Un nombre de slots par couche imposé.
    case slotsPerLayer(Int)

    /// Une empreinte mémoire totale visée pour le processus, poids résidents et KV compris.
    case memoryTarget(bytes: Int)

    /// Tout ce que le plafond matériel autorise. Utile comme **référence de correction** :
    /// une exécution entièrement résidente doit produire exactement les mêmes tokens
    /// qu'une exécution au minimum. À ne pas confondre avec un mode par défaut.
    case maximize
}

/// Budget mémoire d'un modèle, pour une longueur de contexte et une politique de cache.
///
/// Calculé **au chargement**, pas à la compilation : la longueur de contexte est choisie
/// par l'utilisateur au moment de charger le modèle (D-005), et le profil matériel est lu
/// sur la machine hôte.
public struct MemoryBudget: Sendable {

    public let config: GptOssConfig
    public let hardware: HardwareProfile
    public let contextLength: Int
    public let scratchBytes: Int
    public let policy: ExpertCachePolicy

    /// Réserve de scratch réutilisable : arène de prefill, tampons d'activation, logits.
    ///
    /// Valeur **provisoire**. Le besoin réel est modeste — un chunk de prefill de 128
    /// tokens représente ~1,5 Mio d'activations d'experts, ~0,7 Mio d'états cachés et
    /// 0,8 Mio de logits — mais il sera mesuré plutôt que supposé, comme le reste.
    public static let defaultScratchBytes = 128 * 1024 * 1024

    public init(
        config: GptOssConfig,
        hardware: HardwareProfile = .appleM4_24GB,
        contextLength: Int,
        policy: ExpertCachePolicy = .minimal,
        scratchBytes: Int = MemoryBudget.defaultScratchBytes
    ) {
        self.config = config
        self.hardware = hardware
        self.contextLength = contextLength
        self.policy = policy
        self.scratchBytes = scratchBytes
    }

    // MARK: - Postes incompressibles

    /// Poids qui doivent occuper la mémoire en permanence : attention, routeurs, normes,
    /// tête LM. L'embedding en est exclu — on n'en lit qu'une ligne par token, il reste
    /// mappé et paginé à la demande.
    public var residentBytes: Int { config.residentBytes }

    public var kvCacheBytes: Int { config.kvCacheBytes(contextLength: contextLength) }

    /// Tout ce qui n'est pas le cache d'experts. C'est le plancher absolu du modèle :
    /// aucune politique de cache ne peut descendre sous cette valeur.
    public var fixedBytes: Int { residentBytes + kvCacheBytes + scratchBytes }

    // MARK: - Cache d'experts

    /// Slots minimaux par couche : il faut pouvoir tenir simultanément les experts
    /// sélectionnés pour le token courant.
    public var minimumSlotsPerLayer: Int { config.expertsPerToken }

    /// Slots que le plafond matériel autorise au maximum.
    public var ceilingSlotsPerLayer: Int {
        let free = hardware.metalWorkingSetCeiling - fixedBytes
        guard free > 0 else { return 0 }
        return min(config.expertCount, free / config.expertSlotBytes / config.layerCount)
    }

    /// Slots effectivement retenus, politique appliquée puis bornée par le matériel.
    public var expertSlotsPerLayer: Int {
        let requested: Int
        switch policy {
        case .minimal:
            requested = minimumSlotsPerLayer
        case .slotsPerLayer(let n):
            requested = n
        case .memoryTarget(let target):
            let free = target - fixedBytes
            requested = free <= 0 ? 0 : free / config.expertSlotBytes / config.layerCount
        case .maximize:
            requested = config.expertCount
        }
        return min(max(requested, minimumSlotsPerLayer), ceilingSlotsPerLayer)
    }

    public var expertCacheBytes: Int {
        expertSlotsPerLayer * config.layerCount * config.expertSlotBytes
    }

    /// Empreinte mémoire totale du processus, hors cache de fichiers du système.
    public var totalFootprintBytes: Int { fixedBytes + expertCacheBytes }

    /// Empreinte plancher atteignable pour ce modèle et ce contexte.
    public var minimumFootprintBytes: Int {
        fixedBytes + minimumSlotsPerLayer * config.layerCount * config.expertSlotBytes
    }

    /// Vrai si le modèle ne tient pas, même au strict minimum.
    public var fits: Bool { ceilingSlotsPerLayer >= minimumSlotsPerLayer }

    /// Vrai si le pool d'experts entier est en mémoire : plus aucune I/O après chargement.
    /// C'est la **référence de correction**, pas une cible.
    public var isFullyResident: Bool { expertSlotsPerLayer >= config.expertCount }

    /// Part des experts d'une couche tenant en cache.
    public var expertCoverage: Double {
        Double(expertSlotsPerLayer) / Double(config.expertCount)
    }

    /// Part du modèle installé qui réside en mémoire. C'est le chiffre qui exprime la
    /// thèse du projet : faire tourner un modèle de 12,8 Gio dans une fraction de ça.
    public var residentFractionOfCheckpoint: Double {
        Double(totalFootprintBytes - scratchBytes) / Double(config.installedBytes)
    }

    // MARK: - Débit

    /// Plancher de calcul : le temps qu'il faut rien que pour faire transiter les poids
    /// dans la bande passante mémoire. Aucune optimisation d'I/O ne passe sous ce seuil.
    public var computeFloorSeconds: Double {
        Double(config.gpuBytesPerDecodedToken) / hardware.memoryBandwidth
    }

    public var maximumTokensPerSecond: Double { 1.0 / computeFloorSeconds }

    /// Estimation pessimiste : calcul et I/O strictement additionnés.
    ///
    /// GPT-OSS n'ayant pas d'expert partagé, il n'existe pas de branche dense à exécuter
    /// pendant les lectures — le recouvrement est structurellement faible (§2.2a). Le
    /// modèle sériel est donc l'hypothèse honnête, pas une précaution.
    public func estimatedTokensPerSecond(cacheHitRate: Double) -> Double {
        let io = Double(config.diskBytesPerDecodedToken(cacheHitRate: cacheHitRate))
            / hardware.diskBandwidth
        return 1.0 / (computeFloorSeconds + io)
    }

    /// Borne basse du taux de hit : celle qu'on obtiendrait si le routeur choisissait
    /// uniformément. Le routage MoE réel est biaisé, donc le taux observé devrait être
    /// supérieur — le jalon 1.7 mesure de combien.
    public var uniformRoutingHitRate: Double { expertCoverage }

    // MARK: - Restitution

    public struct Line: Sendable {
        public let label: String
        public let bytes: Int
    }

    public var breakdown: [Line] {
        [
            Line(label: "Poids résidents", bytes: residentBytes),
            Line(label: "KV cache FP16 (\(contextLength / 1024)k)", bytes: kvCacheBytes),
            Line(label: "Scratch réutilisable", bytes: scratchBytes),
            Line(
                label: "Cache d'experts (\(expertSlotsPerLayer)/\(config.expertCount) par couche)",
                bytes: expertCacheBytes),
        ]
    }
}
