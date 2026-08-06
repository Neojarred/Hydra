import Foundation
import HydraCore
import Metal

/// Access to the GPU: device, command queue, kernel library.
///
/// Shaders are compiled **at runtime** by `makeLibrary(source:)`, as TurboFieldfare does.
/// Two reasons: the project then has no dependency on the Xcode toolchain to produce its
/// kernels, and runtime compilation is the prerequisite for the `function_constant`
/// specialization planned for phase 3 — the model contract will be able to inject
/// dimensions as compile-time constants without changing infrastructure.
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
                return "no Metal GPU available"
            case .noCommandQueue:
                return "could not create the Metal command queue"
            case .shaderSourceMissing(let name):
                return "shader source not found in the bundle: \(name)"
            case .functionMissing(let name):
                return "function missing from the Metal library: \(name)"
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
        // The kernels depend on no aggressive floating-point reassociation; we keep strict
        // behaviour so that deviations from the CPU reference stay explainable.
        options.mathMode = .safe
        self.library = try device.makeLibrary(source: source, options: options)
    }

    /// The compiled pipeline for a function, cached: building one costs a few milliseconds
    /// and happens once per name.
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

    // MARK: - Hardware profile

    /// Fills in `HydraCore`'s `HardwareProfile` from the host machine.
    ///
    /// This is the only place that queries Metal for sizing: `HydraCore` stays portable and
    /// receives these values, it never goes looking for them (D-002). Bandwidths can be
    /// measured with `measureMemoryBandwidth`; by default we take conservative values rather
    /// than those of one particular machine.
    public func hardwareProfile(
        memoryBandwidth: Double? = nil,
        diskBandwidth: Double? = nil
    ) -> HardwareProfile {
        HardwareProfile(
            metalWorkingSetCeiling: Int(device.recommendedMaxWorkingSetSize),
            memoryBandwidth: memoryBandwidth ?? measureMemoryBandwidth(),
            diskBandwidth: diskBandwidth ?? 2.5e9)
    }

    /// Measures read memory bandwidth with a streaming kernel.
    ///
    /// The first pass is discarded: it pays the first page fault on the buffer and
    /// underestimates throughput by a factor of three.
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

    /// The most recent GPU family supported. Determines which kernel paths are available:
    /// TurboFieldfare's TensorOps path requires `apple10`, absent from the M4.
    public var gpuFamily: String {
        if device.supportsFamily(MTLGPUFamily(rawValue: 1010) ?? .apple9) { return "apple10+" }
        if device.supportsFamily(.apple9) { return "apple9" }
        if device.supportsFamily(.apple8) { return "apple8" }
        if device.supportsFamily(.apple7) { return "apple7" }
        return "unknown"
    }
}
