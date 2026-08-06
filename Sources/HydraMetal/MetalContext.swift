import Foundation
import HydraCore
import Metal

/// Accès au GPU : device, file de commandes, bibliothèque de noyaux.
///
/// Les shaders sont compilés **à l'exécution** par `makeLibrary(source:)`, comme le fait
/// TurboFieldfare. Deux raisons : le projet n'a alors aucune dépendance à la toolchain
/// Xcode pour produire ses noyaux, et la compilation à l'exécution est le préalable à la
/// spécialisation par `function_constant` prévue en phase 3 — le contrat de modèle pourra
/// injecter les dimensions comme constantes de compilation sans changer d'infrastructure.
public final class MetalContext: @unchecked Sendable {

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    public let library: MTLLibrary

    private var pipelineCache: [String: MTLComputePipelineState] = [:]
    private let cacheLock = NSLock()

    public enum ContextError: Error, CustomStringConvertible {
        case noDevice
        case noCommandQueue
        case shaderSourceMissing(String)
        case functionMissing(String)

        public var description: String {
            switch self {
            case .noDevice:
                return "aucun GPU Metal disponible"
            case .noCommandQueue:
                return "impossible de créer la file de commandes Metal"
            case .shaderSourceMissing(let name):
                return "source de shader introuvable dans le bundle : \(name)"
            case .functionMissing(let name):
                return "fonction absente de la bibliothèque Metal : \(name)"
            }
        }
    }

    public init(
        shaderFiles: [String] = ["common", "mxfp4", "attention", "dense", "batch", "tiled", "probe"]
    ) throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw ContextError.noDevice }
        guard let queue = device.makeCommandQueue() else { throw ContextError.noCommandQueue }
        self.device = device
        self.commandQueue = queue

        var source = ""
        for name in shaderFiles {
            guard let url = Bundle.module.url(
                forResource: "Shaders/\(name)", withExtension: "metal")
            else {
                throw ContextError.shaderSourceMissing(name)
            }
            source += try String(contentsOf: url, encoding: .utf8) + "\n"
        }

        let options = MTLCompileOptions()
        // Les noyaux ne dépendent d'aucune réassociation flottante agressive ; on garde
        // le comportement strict pour que les écarts avec la référence CPU restent
        // explicables.
        options.mathMode = .safe
        self.library = try device.makeLibrary(source: source, options: options)
    }

    /// Pipeline compilé pour une fonction, mis en cache : la construction coûte quelques
    /// millisecondes et se produit une fois par nom.
    public func pipeline(_ functionName: String) throws -> MTLComputePipelineState {
        cacheLock.lock()
        if let cached = pipelineCache[functionName] {
            cacheLock.unlock()
            return cached
        }
        cacheLock.unlock()

        guard let function = library.makeFunction(name: functionName) else {
            throw ContextError.functionMissing(functionName)
        }
        let pipeline = try device.makeComputePipelineState(function: function)

        cacheLock.lock()
        pipelineCache[functionName] = pipeline
        cacheLock.unlock()
        return pipeline
    }

    // MARK: - Profil matériel

    /// Renseigne le `HardwareProfile` de `HydraCore` depuis la machine hôte.
    ///
    /// C'est le seul endroit qui interroge Metal pour le dimensionnement : `HydraCore`
    /// reste portable et reçoit ces valeurs, il ne va jamais les chercher (D-002).
    /// Les bandes passantes sont mesurables par `measureMemoryBandwidth` ; par défaut on
    /// prend des valeurs prudentes plutôt que celles d'une machine particulière.
    public func hardwareProfile(
        memoryBandwidth: Double? = nil,
        diskBandwidth: Double? = nil
    ) -> HardwareProfile {
        HardwareProfile(
            metalWorkingSetCeiling: Int(device.recommendedMaxWorkingSetSize),
            memoryBandwidth: memoryBandwidth ?? measureMemoryBandwidth(),
            diskBandwidth: diskBandwidth ?? 2.5e9)
    }

    /// Mesure la bande passante mémoire en lecture par un noyau de streaming.
    ///
    /// La première passe est jetée : elle paie la première faute de page sur le tampon et
    /// sous-estime le débit d'un facteur trois.
    public func measureMemoryBandwidth(bytes: Int = 512 * 1024 * 1024, passes: Int = 3) -> Double {
        guard let pipeline = try? pipeline("bandwidth_probe"),
            let buffer = device.makeBuffer(length: bytes, options: .storageModeShared),
            let sink = device.makeBuffer(length: 4096, options: .storageModeShared)
        else {
            return 0
        }
        memset(buffer.contents(), 1, bytes)

        var best = 0.0
        var elementCount = UInt32(bytes / 16)
        for pass in 0..<passes {
            let start = Date()
            guard let commandBuffer = commandQueue.makeCommandBuffer(),
                let encoder = commandBuffer.makeComputeCommandEncoder()
            else { return best }
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(buffer, offset: 0, index: 0)
            encoder.setBuffer(sink, offset: 0, index: 1)
            encoder.setBytes(&elementCount, length: 4, index: 2)
            encoder.dispatchThreads(
                MTLSize(width: 1 << 18, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(
                    width: pipeline.maxTotalThreadsPerThreadgroup, height: 1, depth: 1))
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()

            let elapsed = Date().timeIntervalSince(start)
            if pass > 0, elapsed > 0 { best = max(best, Double(bytes) / elapsed) }
        }
        return best
    }

    /// Famille GPU la plus récente supportée. Détermine les chemins de noyaux
    /// disponibles : le chemin TensorOps de TurboFieldfare exige `apple10`, absent des M4.
    public var gpuFamily: String {
        if device.supportsFamily(MTLGPUFamily(rawValue: 1010) ?? .apple9) { return "apple10+" }
        if device.supportsFamily(.apple9) { return "apple9" }
        if device.supportsFamily(.apple8) { return "apple8" }
        if device.supportsFamily(.apple7) { return "apple7" }
        return "inconnue"
    }
}
