import Foundation

/// Disposition interne d'un blob d'expert et pas de progression d'un expert au suivant.
///
/// Ce calcul vit dans `HydraCore` parce que **deux consommateurs en dépendent et ne
/// doivent jamais diverger** : le format sur disque (`HydraFormat.HydraLayout`) et le
/// dimensionnement des slots mémoire (`MemoryBudget`). Une première version les calculait
/// séparément ; les slots se retrouvaient sous-alloués de 128 octets, exactement le
/// remplissage d'alignement que le format ajoutait. Une seule source de vérité supprime
/// cette classe d'erreur.
public struct ExpertBlobLayout: Sendable, Equatable {

    /// Alignement d'un blob dans son fichier de couche, et donc de chaque `pread`.
    /// Un blob désaligné forcerait le noyau à lire une page de plus à chaque extrémité,
    /// sur 144 lectures par token.
    public static let pageAlignment = 16384

    /// Alignement de chaque sous-tenseur dans le blob.
    ///
    /// Les décalages passés à `setBuffer(offset:)` ont une contrainte d'alignement, et
    /// la largeur des chargements vectoriels dans les shaders dépend de l'alignement réel
    /// de l'adresse. TurboFieldfare a documenté un bug où un chemin 32 bits passait les
    /// tests à l'offset zéro puis produisait du bruit en décodage, parce que les décalages
    /// vivants n'étaient alignés que sur 2 octets. On s'aligne largement : le surcoût est
    /// de 128 octets par blob, soit 0,001 %.
    public static let tensorAlignment = 256

    public struct Slot: Sendable, Equatable {
        public let offset: Int
        public let byteCount: Int
        public var end: Int { offset + byteCount }
    }

    public let gateUpBlocks: Slot
    public let gateUpScales: Slot
    public let gateUpBias: Slot
    public let downBlocks: Slot
    public let downScales: Slot
    public let downBias: Slot

    /// Octets utiles d'un blob une fois mis en page, remplissage d'alignement interne compris.
    /// Toujours supérieur ou égal à la somme brute des tenseurs sources.
    public let payloadBytes: Int

    /// Distance entre deux blobs consécutifs, et **taille d'un slot en mémoire**.
    public let strideBytes: Int

    public var slots: [Slot] {
        [gateUpBlocks, gateUpScales, gateUpBias, downBlocks, downScales, downBias]
    }

    /// Somme brute des tenseurs sources, sans remplissage. C'est la valeur qui doit
    /// correspondre au checkpoint Hugging Face.
    public var sourceBytes: Int { slots.reduce(0) { $0 + $1.byteCount } }

    init(config: GptOssConfig) {
        let inBlocks = config.hiddenSize / MXFP4Layout.blockSize
        let downInBlocks = config.intermediateSize / MXFP4Layout.blockSize
        let gateUpRows = 2 * config.intermediateSize
        let downRows = config.hiddenSize

        var cursor = 0
        func place(_ size: Int) -> Slot {
            cursor = alignUp(cursor, to: Self.tensorAlignment)
            let slot = Slot(offset: cursor, byteCount: size)
            cursor += size
            return slot
        }

        // Ordre choisi pour suivre celui de consommation du kernel MoE :
        // gate_up d'abord (projection puis SwiGLU), puis down (réduction).
        self.gateUpBlocks = place(gateUpRows * inBlocks * MXFP4Layout.packedBytesPerBlock)
        self.gateUpScales = place(gateUpRows * inBlocks * MXFP4Layout.scaleBytesPerBlock)
        self.gateUpBias = place(gateUpRows * 2)
        self.downBlocks = place(downRows * downInBlocks * MXFP4Layout.packedBytesPerBlock)
        self.downScales = place(downRows * downInBlocks * MXFP4Layout.scaleBytesPerBlock)
        self.downBias = place(downRows * 2)

        self.payloadBytes = cursor
        self.strideBytes = alignUp(cursor, to: Self.pageAlignment)
    }
}

@inlinable
public func alignUp(_ value: Int, to alignment: Int) -> Int {
    let r = value % alignment
    return r == 0 ? value : value + (alignment - r)
}
