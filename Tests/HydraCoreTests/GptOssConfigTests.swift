import Testing

@testable import HydraCore

/// Ces tests verrouillent la modélisation du checkpoint contre les valeurs **réellement
/// publiées** par Hugging Face. Le test le plus important est `installedSizeMatchesIndex` :
/// si notre reconstruction de chaque tenseur retombe à l'octet près sur le `total_size`
/// déclaré, c'est que la structure du modèle est comprise exactement — pas approximée.
struct GptOssConfigTests {

    @Test("La taille d'un blob d'expert est identique pour les deux modèles")
    func expertBlobSize() {
        #expect(GptOssConfig.b20.expertBlobBytes == 13_236_480)
        #expect(GptOssConfig.b120.expertBlobBytes == 13_236_480)
    }

    /// `metadata.total_size` des `model.safetensors.index.json` officiels.
    @Test(
        "L'installation reconstruite correspond à l'octet au total_size de l'index",
        arguments: [
            (GptOssConfig.b20, 13_761_264_768),
            (GptOssConfig.b120, 65_248_815_744),
        ])
    func installedSizeMatchesIndex(config: GptOssConfig, declaredTotal: Int) {
        #expect(config.installedBytes == declaredTotal, "\(config.name)")
    }

    @Test("Le pool d'experts du 120B")
    func expertPool() {
        #expect(GptOssConfig.b120.expertPoolBytes == 60_993_699_840)
        #expect(GptOssConfig.b20.expertPoolBytes == 10_165_616_640)
    }

    @Test("L'embedding est exclu des poids résidents")
    func embeddingIsNotResident() {
        let c = GptOssConfig.b120
        // On n'en lit qu'une ligne par token : il reste mappé, hors working set Metal.
        #expect(c.embeddingBytes == 1_158_266_880)
        #expect(c.residentBytes < c.installedBytes - c.expertPoolBytes)
        #expect(c.installedBytes - c.expertPoolBytes - c.residentBytes == c.embeddingBytes)
    }

    @Test("L'attention alterne fenêtre glissante et pleine à partir de la couche 0")
    func attentionAlternates() {
        let c = GptOssConfig.b120
        #expect(c.attentionPattern(atLayer: 0) == .sliding)
        #expect(c.attentionPattern(atLayer: 1) == .full)
        #expect(c.fullAttentionLayerCount == 18)
        #expect(c.slidingAttentionLayerCount == 18)
        #expect(c.groupedQueryFactor == 8)
    }

    @Test("Le KV cache reste modeste grâce à la fenêtre de 128")
    func kvCacheIsCheap() {
        let c = GptOssConfig.b120
        // 18 couches full x 2048 o/token, plus 18 anneaux bornés de 256 lignes.
        #expect(c.kvBytesPerTokenPerFullLayer == 2048)
        #expect(c.kvCacheBytes(contextLength: 32768) == 1_217_396_736)
        // À 128k, on reste sous 5 Gio là où des couches toutes full en coûteraient le double.
        #expect(c.kvCacheBytes(contextLength: 131_072) < 5 * 1_073_741_824)
    }

    @Test("Les constantes de layout MXFP4 sont cohérentes entre modules")
    func layoutConstants() {
        #expect(MXFP4Layout.blockSize == 32)
        #expect(MXFP4Layout.packedBytesPerBlock == 16)
        // 32 valeurs sur 16 octets packés + 1 octet d'échelle = 4,25 bits par poids.
        let bitsPerWeight =
            Double((MXFP4Layout.packedBytesPerBlock + MXFP4Layout.scaleBytesPerBlock) * 8)
            / Double(MXFP4Layout.blockSize)
        #expect(bitsPerWeight == 4.25)
    }
}

/// L'objectif du projet est de **réduire** l'empreinte mémoire. Ces tests verrouillent
/// donc le fait que la politique par défaut est le minimum, et non « remplir le plafond ».
struct MemoryBudgetTests {

    @Test("Par défaut, on ne garde que les experts strictement nécessaires",
          arguments: [GptOssConfig.b20, .b120])
    func defaultPolicyIsMinimal(config: GptOssConfig) {
        let b = MemoryBudget(config: config, contextLength: 8192)
        #expect(b.policy == .minimal)
        #expect(b.expertSlotsPerLayer == config.expertsPerToken)
        #expect(!b.isFullyResident, "le minimum ne doit jamais tout charger")
        #expect(b.totalFootprintBytes == b.minimumFootprintBytes)
    }

    /// C'est la thèse du projet, exprimée en un chiffre.
    @Test("Au minimum, l'empreinte est une petite fraction du modèle installé")
    func minimalFootprintIsSmallFraction() {
        let b20 = MemoryBudget(config: .b20, contextLength: 8192)
        #expect(b20.residentFractionOfCheckpoint < 0.35, "20B : \(b20.residentFractionOfCheckpoint)")

        let b120 = MemoryBudget(config: .b120, contextLength: 8192)
        // Le 120B pèse 60,8 Gio installés ; on vise moins de 10 % résidents.
        #expect(b120.residentFractionOfCheckpoint < 0.10, "120B : \(b120.residentFractionOfCheckpoint)")
    }

    /// Le plancher n'est plus le cache d'experts mais les poids BF16 non quantifiés.
    /// Contrairement à Gemma 4, GPT-OSS garde attention, routeurs et tête LM en BF16.
    @Test("Le plancher est dominé par les poids résidents, pas par les experts")
    func residentWeightsDominateTheFloor() {
        for config in [GptOssConfig.b20, .b120] {
            let b = MemoryBudget(config: config, contextLength: 8192)
            #expect(b.residentBytes > b.expertCacheBytes, Comment(rawValue: config.name))
        }
    }

    @Test("Une cible mémoire explicite est respectée")
    func memoryTargetIsHonoured() {
        let target = 6 * 1_073_741_824
        let b = MemoryBudget(
            config: .b120, contextLength: 8192, policy: .memoryTarget(bytes: target))
        #expect(b.totalFootprintBytes <= target)
        // Et elle doit acheter davantage que le minimum.
        #expect(b.expertSlotsPerLayer > b.minimumSlotsPerLayer)
    }

    @Test("Une cible inatteignable retombe sur le minimum plutôt que d'échouer")
    func impossibleTargetFallsBackToMinimum() {
        let b = MemoryBudget(
            config: .b120, contextLength: 8192, policy: .memoryTarget(bytes: 1_000_000))
        #expect(b.expertSlotsPerLayer == b.minimumSlotsPerLayer)
    }

    @Test("La politique maximale sert de référence de correction sur le 20B")
    func maximizeIsTheCorrectnessReference() {
        let b = MemoryBudget(config: .b20, contextLength: 8192, policy: .maximize)
        #expect(b.isFullyResident, "le 20B doit pouvoir tourner sans aucune I/O, pour comparaison")
        // Le 120B, lui, ne peut jamais l'être : c'est tout l'intérêt du projet.
        let big = MemoryBudget(config: .b120, contextLength: 8192, policy: .maximize)
        #expect(!big.isFullyResident)
    }

    @Test("À politique maximale, un contexte plus long coûte des slots")
    func longerContextCostsSlotsWhenMaximizing() {
        let short = MemoryBudget(config: .b120, contextLength: 8192, policy: .maximize)
        let long = MemoryBudget(config: .b120, contextLength: 131_072, policy: .maximize)
        #expect(long.expertSlotsPerLayer < short.expertSlotsPerLayer)
        #expect(long.kvCacheBytes > short.kvCacheBytes)
    }

    @Test("Le minimum, lui, ne dépend pas du contexte")
    func minimumIsContextIndependent() {
        let short = MemoryBudget(config: .b120, contextLength: 8192)
        let long = MemoryBudget(config: .b120, contextLength: 131_072)
        #expect(long.expertSlotsPerLayer == short.expertSlotsPerLayer)
        // Seul le KV cache grossit.
        #expect(long.totalFootprintBytes > short.totalFootprintBytes)
    }

    @Test("Les slots sont dimensionnés sur le blob aligné, pas sur la somme brute")
    func slotsUseAlignedSize() {
        let config = GptOssConfig.b120
        #expect(config.expertSlotBytes >= config.expertBlobBytes)
        let b = MemoryBudget(config: config, contextLength: 8192)
        #expect(b.expertCacheBytes
            == b.expertSlotsPerLayer * config.layerCount * config.expertSlotBytes)
    }

    @Test("Le plafond de calcul ne dépend pas du stockage")
    func computeFloorIgnoresDisk() {
        var fastDisk = HardwareProfile.appleM4_24GB
        fastDisk.diskBandwidth = 1e15
        let b = MemoryBudget(config: .b120, hardware: fastDisk, contextLength: 8192)
        #expect(b.estimatedTokensPerSecond(cacheHitRate: 0) <= b.maximumTokensPerSecond)
        #expect(b.maximumTokensPerSecond < 20)
        #expect(b.maximumTokensPerSecond > 17)
    }

    @Test("Un taux de hit parfait atteint le plafond de calcul")
    func perfectCacheReachesFloor() {
        let b = MemoryBudget(config: .b120, contextLength: 8192)
        #expect(b.config.diskBytesPerDecodedToken(cacheHitRate: 1.0) == 0)
        #expect(abs(b.estimatedTokensPerSecond(cacheHitRate: 1.0) - b.maximumTokensPerSecond) < 1e-9)
    }

    /// Une machine bien plus petite doit rester servie : rien dans le dimensionnement
    /// n'est spécifique à la machine de développement.
    @Test("Le 20B tient au minimum sur une machine de 8 Gio")
    func smallMachineStillFits() {
        let small = HardwareProfile(
            metalWorkingSetCeiling: 5 * 1_073_741_824,  // ~ ce qu'expose une machine 8 Gio
            memoryBandwidth: 68e9, diskBandwidth: 2.5e9)
        let b = MemoryBudget(config: .b20, hardware: small, contextLength: 4096)
        #expect(b.fits)
        #expect(b.expertSlotsPerLayer == b.minimumSlotsPerLayer)
        #expect(b.totalFootprintBytes <= small.metalWorkingSetCeiling)
    }
}
