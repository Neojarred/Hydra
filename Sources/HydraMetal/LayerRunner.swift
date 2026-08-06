import Foundation
import HydraCore
import HydraFormat
import Metal

/// Tampons de travail réutilisés d'un token à l'autre.
///
/// Tout est alloué **une fois** au chargement. Rien n'est alloué pendant le décodage :
/// une allocation Metal par token coûterait plus cher que le calcul, et l'empreinte
/// cesserait d'être prévisible — ce qui ruinerait la seule propriété que le projet doit
/// démontrer.
public final class DecodeScratch: @unchecked Sendable {

    public let hidden: MTLBuffer       // état résiduel, [hiddenSize]
    public let normed: MTLBuffer       // [hiddenSize]
    public let query: MTLBuffer        // [qHeads * headDim]
    public let key: MTLBuffer          // [kvHeads * headDim]
    public let value: MTLBuffer        // [kvHeads * headDim]
    public let attention: MTLBuffer    // [qHeads * headDim]
    public let projected: MTLBuffer    // [hiddenSize]
    public let routerLogits: MTLBuffer // [expertCount]
    public let routerIndices: MTLBuffer  // [topK] UInt32
    public let routerWeights: MTLBuffer  // [topK] Float
    public let mixture: MTLBuffer      // [hiddenSize]
    public let gateUp: MTLBuffer       // [2 * intermediateSize]
    public let activated: MTLBuffer    // [intermediateSize]
    public let expertOutput: MTLBuffer // [hiddenSize]
    /// Une case par expert sélectionné : [expertsPerToken × hiddenSize].
    public let expertSlices: MTLBuffer
    public let cosTable: MTLBuffer     // [headDim / 2]
    public let sinTable: MTLBuffer     // [headDim / 2]

    public let byteCount: Int

    public enum ScratchError: Error, CustomStringConvertible {
        case allocationFailed(String, bytes: Int)
        public var description: String {
            switch self {
            case let .allocationFailed(name, bytes):
                return "scratch : allocation de \(bytes) o impossible pour « \(name) »"
            }
        }
    }

    public init(config: GptOssConfig, device: MTLDevice) throws {
        var total = 0
        func make(_ name: String, floats: Int) throws -> MTLBuffer {
            let bytes = max(floats * MemoryLayout<Float>.size, 16)
            guard let buffer = device.makeBuffer(length: bytes, options: .storageModeShared)
            else { throw ScratchError.allocationFailed(name, bytes: bytes) }
            total += bytes
            return buffer
        }

        let qDim = config.attentionHeadCount * config.headDim
        let kvDim = config.keyValueHeadCount * config.headDim

        hidden = try make("hidden", floats: config.hiddenSize)
        normed = try make("normed", floats: config.hiddenSize)
        query = try make("query", floats: qDim)
        key = try make("key", floats: kvDim)
        value = try make("value", floats: kvDim)
        attention = try make("attention", floats: qDim)
        projected = try make("projected", floats: config.hiddenSize)
        routerLogits = try make("routerLogits", floats: config.expertCount)
        routerIndices = try make("routerIndices", floats: config.expertsPerToken)
        routerWeights = try make("routerWeights", floats: config.expertsPerToken)
        mixture = try make("mixture", floats: config.hiddenSize)
        gateUp = try make("gateUp", floats: 2 * config.intermediateSize)
        activated = try make("activated", floats: config.intermediateSize)
        expertOutput = try make("expertOutput", floats: config.hiddenSize)
        expertSlices = try make(
            "expertSlices", floats: config.expertsPerToken * config.hiddenSize)
        cosTable = try make("cosTable", floats: config.headDim / 2)
        sinTable = try make("sinTable", floats: config.headDim / 2)

        byteCount = total
    }

    public func hiddenState() -> UnsafeMutableBufferPointer<Float> {
        UnsafeMutableBufferPointer(
            start: hidden.contents().bindMemory(to: Float.self, capacity: 1), count: 1)
    }
}

/// Exécute une couche de transformeur sur le GPU.
///
/// Le graphe est coupé en deux tampons de commandes, et cette coupure est **imposée par
/// l'architecture**, pas choisie : le routeur produit les identifiants d'experts sur le
/// GPU, et le CPU doit les lire pour savoir quels blobs charger depuis le SSD. Aucun
/// réordonnancement ne contourne cette dépendance.
///
/// ```
/// cb1 : norme → QKV → RoPE → écriture KV → attention → projection O → résidu
///       → norme post-attention → logits du routeur → top-k
/// I/O : lecture des identifiants, chargement parallèle des experts manquants
/// cb2 : pour chaque expert — gate_up → SwiGLU → down → accumulation pondérée
///       → résidu
/// ```
///
/// GPT-OSS **n'ayant pas d'expert partagé**, il n'existe aucune branche dense à calculer
/// pendant les lectures. Le recouvrement qui masque la latence chez TurboFieldfare est
/// donc structurellement absent ici — c'est une limite documentée, pas un oubli.
public struct LayerRunner: Sendable {

    public let config: GptOssConfig
    public let encoder: ForwardEncoder
    public let mapping: ModelMapping
    public let cache: ExpertSlotCache

    public init(
        config: GptOssConfig, encoder: ForwardEncoder,
        mapping: ModelMapping, cache: ExpertSlotCache
    ) {
        self.config = config
        self.encoder = encoder
        self.mapping = mapping
        self.cache = cache
    }

    private var qDim: Int { config.attentionHeadCount * config.headDim }
    private var kvDim: Int { config.keyValueHeadCount * config.headDim }
    private var smScale: Float { 1.0 / Float(config.headDim).squareRoot() }

    private func tensor(_ suffix: String, layer: Int) throws -> (MTLBuffer, Int) {
        let (buffer, offset, _) = try mapping.residentTensor("model.layers.\(layer).\(suffix)")
        return (buffer, offset)
    }

    /// Première moitié : tout ce qui précède la connaissance des experts.
    public func encodeAttentionAndRouter(
        layer: Int, position: Int, scratch: DecodeScratch, kvCache: KVCache,
        in commandBuffer: MTLCommandBuffer
    ) throws {
        let ring = kvCache.layers[layer].ringSize

        // --- Normalisation d'entrée ---
        let inputNorm = try tensor("input_layernorm.weight", layer: layer)
        try encoder.rmsNorm(
            input: scratch.hidden, scale: inputNorm.0, scaleOffset: inputNorm.1,
            output: scratch.normed, size: config.hiddenSize, eps: config.rmsNormEps,
            in: commandBuffer)

        // --- Projections Q, K, V ---
        for (suffix, output, rows) in [
            ("q_proj", scratch.query, qDim),
            ("k_proj", scratch.key, kvDim),
            ("v_proj", scratch.value, kvDim),
        ] {
            let weight = try tensor("self_attn.\(suffix).weight", layer: layer)
            let bias = try tensor("self_attn.\(suffix).bias", layer: layer)
            try encoder.denseProjection(
                weights: weight.0, weightsOffset: weight.1,
                bias: bias.0, biasOffset: bias.1,
                input: scratch.normed, inputOffset: 0,
                output: output, outputOffset: 0,
                rows: rows, cols: config.hiddenSize, in: commandBuffer)
        }

        // --- RoPE sur Q et K, tables déjà porteuses de la concentration YaRN ---
        try encoder.applyRoPE(
            vector: scratch.query, vectorOffset: 0,
            cos: scratch.cosTable, sin: scratch.sinTable, tableOffset: 0,
            heads: config.attentionHeadCount, headDim: config.headDim, in: commandBuffer)
        try encoder.applyRoPE(
            vector: scratch.key, vectorOffset: 0,
            cos: scratch.cosTable, sin: scratch.sinTable, tableOffset: 0,
            heads: config.keyValueHeadCount, headDim: config.headDim, in: commandBuffer)

        // --- Écriture dans le cache, puis attention ---
        try encoder.writeKeyValue(
            key: scratch.key, keyOffset: 0, value: scratch.value, valueOffset: 0,
            keyCache: kvCache.layers[layer].keys, valueCache: kvCache.layers[layer].values,
            kvHeads: config.keyValueHeadCount, headDim: config.headDim,
            position: position, ringSize: ring, in: commandBuffer)

        let visible = kvCache.visibleRange(layer: layer, position: position)
        let sinks = try tensor("self_attn.sinks", layer: layer)
        try encoder.attention(
            query: scratch.query, queryOffset: 0,
            keyCache: kvCache.layers[layer].keys, valueCache: kvCache.layers[layer].values,
            sinks: sinks.0, sinksOffset: sinks.1,
            output: scratch.attention, outputOffset: 0,
            qHeads: config.attentionHeadCount, kvHeads: config.keyValueHeadCount,
            headDim: config.headDim, keyCount: visible.count,
            ringSize: ring, startPosition: visible.start, smScale: smScale,
            in: commandBuffer)

        // --- Projection de sortie et résidu ---
        let outWeight = try tensor("self_attn.o_proj.weight", layer: layer)
        let outBias = try tensor("self_attn.o_proj.bias", layer: layer)
        try encoder.denseProjection(
            weights: outWeight.0, weightsOffset: outWeight.1,
            bias: outBias.0, biasOffset: outBias.1,
            input: scratch.attention, inputOffset: 0,
            output: scratch.projected, outputOffset: 0,
            rows: config.hiddenSize, cols: qDim, in: commandBuffer)
        try encoder.addInPlace(
            target: scratch.hidden, targetOffset: 0,
            addend: scratch.projected, addendOffset: 0,
            size: config.hiddenSize, in: commandBuffer)

        // --- Normalisation post-attention et routeur ---
        let postNorm = try tensor("post_attention_layernorm.weight", layer: layer)
        try encoder.rmsNorm(
            input: scratch.hidden, scale: postNorm.0, scaleOffset: postNorm.1,
            output: scratch.normed, size: config.hiddenSize, eps: config.rmsNormEps,
            in: commandBuffer)

        let routerWeight = try tensor("mlp.router.weight", layer: layer)
        let routerBias = try tensor("mlp.router.bias", layer: layer)
        try encoder.denseProjection(
            weights: routerWeight.0, weightsOffset: routerWeight.1,
            bias: routerBias.0, biasOffset: routerBias.1,
            input: scratch.normed, inputOffset: 0,
            output: scratch.routerLogits, outputOffset: 0,
            rows: config.expertCount, cols: config.hiddenSize, in: commandBuffer)

        try encoder.routerTopK(
            logits: scratch.routerLogits, logitsOffset: 0,
            indices: scratch.routerIndices, weights: scratch.routerWeights,
            expertCount: config.expertCount, topK: config.expertsPerToken, in: commandBuffer)
    }

    /// Identifiants d'experts choisis, lus après validation de `cb1`.
    public func selectedExperts(_ scratch: DecodeScratch) -> [Int] {
        let raw = UnsafeBufferPointer(
            start: scratch.routerIndices.contents().bindMemory(
                to: UInt32.self, capacity: config.expertsPerToken),
            count: config.expertsPerToken)
        return raw.map(Int.init)
    }

    /// Remet à zéro l'accumulateur du mélange.
    public func encodeMixtureStart(
        scratch: DecodeScratch, in commandBuffer: MTLCommandBuffer
    ) throws {
        try encoder.fillZero(
            scratch.mixture, offset: 0, size: config.hiddenSize, in: commandBuffer)
    }

    /// Ajoute le résidu du mélange à l'état caché.
    public func encodeMixtureEnd(
        scratch: DecodeScratch, in commandBuffer: MTLCommandBuffer
    ) throws {
        try encoder.addInPlace(
            target: scratch.hidden, targetOffset: 0,
            addend: scratch.mixture, addendOffset: 0,
            size: config.hiddenSize, in: commandBuffer)
    }

    /// Un seul expert, pour permettre de soumettre son calcul dès qu'il est chargé.
    public func encodeSingleExpert(
        layer: Int, expert: Int, weightIndex: Int,
        scratch: DecodeScratch, in commandBuffer: MTLCommandBuffer
    ) throws {
        let blob = config.expertBlobLayout
        // Bloque jusqu'à ce que **cet** expert soit prêt, pas jusqu'à ce que tous le soient.
        let (buffer, _) = try cache.expert(layer: layer, expert: expert, pin: true)

        try encoder.expertProjection(
            blocks: buffer, blocksOffset: blob.gateUpBlocks.offset,
            scales: buffer, scalesOffset: blob.gateUpScales.offset,
            bias: buffer, biasOffset: blob.gateUpBias.offset,
            input: scratch.normed, inputOffset: 0,
            output: scratch.gateUp, outputOffset: 0,
            rows: 2 * config.intermediateSize, cols: config.hiddenSize, in: commandBuffer)
        try encoder.swiglu(
            input: scratch.gateUp, inputOffset: 0,
            output: scratch.activated, outputOffset: 0,
            size: config.intermediateSize,
            alpha: 1.702, limit: config.swigluLimit, in: commandBuffer)
        try encoder.expertProjection(
            blocks: buffer, blocksOffset: blob.downBlocks.offset,
            scales: buffer, scalesOffset: blob.downScales.offset,
            bias: buffer, biasOffset: blob.downBias.offset,
            input: scratch.activated, inputOffset: 0,
            output: scratch.expertOutput, outputOffset: 0,
            rows: config.hiddenSize, cols: config.intermediateSize, in: commandBuffer)
        try encoder.writeExpertScaled(
            into: scratch.expertSlices,
            outputOffset: weightIndex * config.hiddenSize * MemoryLayout<Float>.size,
            contribution: scratch.expertOutput, contributionOffset: 0,
            weights: scratch.routerWeights, weightIndex: weightIndex,
            size: config.hiddenSize, in: commandBuffer)
    }

    /// Somme les cases dans l'ordre des slots, puis ajoute le résidu.
    ///
    /// C'est ici, et seulement ici, que l'ordre d'addition est fixé — il ne dépend donc
    /// pas de l'ordre dans lequel les experts ont été calculés, ni de l'état du cache.
    public func encodeCombineSlices(
        count: Int, scratch: DecodeScratch, in commandBuffer: MTLCommandBuffer
    ) throws {
        try encoder.sumExpertSlices(
            into: scratch.mixture, slices: scratch.expertSlices,
            size: config.hiddenSize, count: count, in: commandBuffer)
        try encoder.addInPlace(
            target: scratch.hidden, targetOffset: 0,
            addend: scratch.mixture, addendOffset: 0,
            size: config.hiddenSize, in: commandBuffer)
    }

    /// Seconde moitié : les experts sélectionnés sont en cache, on peut calculer.
    public func encodeMixtureOfExperts(
        layer: Int, experts: [Int], scratch: DecodeScratch,
        in commandBuffer: MTLCommandBuffer
    ) throws {
        let blob = config.expertBlobLayout
        try encoder.fillZero(
            scratch.mixture, offset: 0, size: config.hiddenSize, in: commandBuffer)

        for (slotIndex, expert) in experts.enumerated() {
            // Verrouillé : la passe encodée ci-dessous référencera ce tampon jusqu'à
            // l'exécution du tampon de commandes, bien après cet appel.
            let (buffer, _) = try cache.expert(layer: layer, expert: expert, pin: true)

            // gate_up : [2 × intermediate, hidden]
            try encoder.expertProjection(
                blocks: buffer, blocksOffset: blob.gateUpBlocks.offset,
                scales: buffer, scalesOffset: blob.gateUpScales.offset,
                bias: buffer, biasOffset: blob.gateUpBias.offset,
                input: scratch.normed, inputOffset: 0,
                output: scratch.gateUp, outputOffset: 0,
                rows: 2 * config.intermediateSize, cols: config.hiddenSize, in: commandBuffer)

            // SwiGLU : découpage en indices pairs/impairs, écrêtage asymétrique, +1.
            try encoder.swiglu(
                input: scratch.gateUp, inputOffset: 0,
                output: scratch.activated, outputOffset: 0,
                size: config.intermediateSize,
                alpha: 1.702, limit: config.swigluLimit, in: commandBuffer)

            // down : [hidden, intermediate]
            try encoder.expertProjection(
                blocks: buffer, blocksOffset: blob.downBlocks.offset,
                scales: buffer, scalesOffset: blob.downScales.offset,
                bias: buffer, biasOffset: blob.downBias.offset,
                input: scratch.activated, inputOffset: 0,
                output: scratch.expertOutput, outputOffset: 0,
                rows: config.hiddenSize, cols: config.intermediateSize, in: commandBuffer)

            try encoder.accumulateExpert(
                into: scratch.mixture, outputOffset: 0,
                contribution: scratch.expertOutput, contributionOffset: 0,
                weights: scratch.routerWeights, weightIndex: slotIndex,
                size: config.hiddenSize, in: commandBuffer)
        }

        try encoder.addInPlace(
            target: scratch.hidden, targetOffset: 0,
            addend: scratch.mixture, addendOffset: 0,
            size: config.hiddenSize, in: commandBuffer)
    }
}
