import Foundation
import HydraCore
import Metal

/// Access to the GPU: device, command queue, kernel library.
///
/// Shaders are compiled **at runtime** by `makeLibrary(source:)`, as TurboFieldfare does.
/// Two reasons: the project then has no dependency on the Xcode toolchain to produce its
/// kernels, and runtime compilation is the prerequisite for the `function_constant`
/// specialization planned for phase 3, the model contract will be able to inject
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
        case commandBufferFailed(String)

        public var description: String {
            switch self {
            case .noDevice:
                return "no Metal GPU available"
            case .noCommandQueue:
                return "could not create the Metal command queue"
            case .shaderSourceMissing(let name):
                return "shader source not found in the bundle: \(name)"
            case .commandBufferFailed(let reason):
                return "the GPU did not complete a command buffer: \(reason)"
            case .functionMissing(let name):
                return "function missing from the Metal library: \(name)"
            }
        }
    }

    public init(
        shaderFiles: [String] = [
            "common", "mxfp4", "attention", "dense", "batch", "tiled", "gemma", "qwen", "vision",
            "probe",
        ]
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

    // MARK: - Scratch buffers

    private var scratchBuffers: [String: MTLBuffer] = [:]

    /// A named scratch buffer of at least `bytes`, allocated once and grown when it is short.
    ///
    /// Kernels that split their work across threadgroups need somewhere to put the partial
    /// results, and the size depends on the conversation rather than on the model, so it cannot
    /// be allocated with the weights. Allocating it a token instead would put a megabyte of
    /// `makeBuffer` on the decode path, which is the kind of cost this project spends its days
    /// removing. The contents are never read before the kernel writes them, so growing by
    /// replacement loses nothing.
    ///
    /// Same lock as the pipeline cache, and for the same reason: encoding is single-threaded
    /// today, and this is cheap enough that relying on that would only be a trap for later.
    public func scratch(_ name: String, bytes: Int) -> MTLBuffer? {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        if let existing = scratchBuffers[name], existing.length >= bytes { return existing }
        guard let buffer = device.makeBuffer(length: bytes, options: .storageModePrivate)
        else { return nil }
        scratchBuffers[name] = buffer
        return buffer
    }

    /// A named scratch buffer that is **guaranteed zero**, grown when it is short.
    ///
    /// `scratch` above hands back private memory whose contents Metal does not promise, which is
    /// fine for a buffer a kernel writes before it reads. It is not fine for one a kernel only
    /// reads: the image-block map is zero for every text model, meaning "ordinary token, causal
    /// only", and a buffer of stale bytes there would give the attention a nonsense key range on
    /// every model in the app, not only the ones that can see a picture.
    ///
    /// Shared storage and an explicit `memset`, so the guarantee is this function's and not the
    /// allocator's.
    public func zeroedScratch(_ name: String, bytes: Int) -> MTLBuffer? {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        if let existing = scratchBuffers[name], existing.length >= bytes { return existing }
        guard let buffer = device.makeBuffer(length: bytes, options: .storageModeShared)
        else { return nil }
        memset(buffer.contents(), 0, bytes)
        scratchBuffers[name] = buffer
        return buffer
    }

    // MARK: - The shared compute encoder

    /// The encoder currently open on a command buffer, and the buffer it belongs to.
    ///
    /// Every dispatch used to create its own encoder and end it immediately, about 660 of
    /// them a decoded token. An encoder boundary is not free on either side: the CPU builds
    /// and tears down encoder state for each one, and the GPU treats the boundary as a hard
    /// flush. The dispatches themselves were already ordered, so the boundaries were buying
    /// nothing.
    ///
    /// A `MTLComputeCommandEncoder` defaults to serial dispatch, which orders the commands
    /// encoded into it and makes each one's writes visible to the next. That is exactly the
    /// guarantee the per-dispatch encoders were providing, so consecutive dispatches can share
    /// one encoder without changing what the kernels observe.
    ///
    /// Encoding is single-threaded, the only concurrency in this module is the `pread` fan-out
    /// in `ExpertSlotCache`, which does not encode, so this needs no lock.
    private var openEncoder: MTLComputeCommandEncoder?
    private var openBuffer: MTLCommandBuffer?

    /// The open encoder for this command buffer, opening one if needed.
    public func sharedEncoder(for commandBuffer: MTLCommandBuffer) throws
        -> MTLComputeCommandEncoder
    {
        if let encoder = openEncoder, openBuffer === commandBuffer { return encoder }
        closeEncoder()
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw ContextError.noCommandQueue
        }
        openEncoder = encoder
        openBuffer = commandBuffer
        return encoder
    }

    /// Ends the open encoder, if there is one. Idempotent.
    public func closeEncoder() {
        openEncoder?.endEncoding()
        openEncoder = nil
        openBuffer = nil
    }

    /// Waits for a command buffer and **fails loudly if the GPU did not complete it**.
    ///
    /// Nothing checked this. A command buffer that fails, a timeout, a resource limit, memory
    /// pressure, leaves the buffers it was going to write untouched, and a Metal buffer starts
    /// zeroed. So a failure did not raise anything: it produced zeros, which flow through the
    /// rest of the forward pass as a perfectly finite answer. The logits come out identical to
    /// each other and inside the softcap, which is exactly the shape of the one flaky failure
    /// this suite has seen (M-048).
    ///
    /// That is the failure mode this project keeps meeting, finite, plausible, and wrong, and
    /// it is the one worth spending a branch on. `waitUntilCompleted` alone is not a check.
    public func wait(_ commandBuffer: MTLCommandBuffer) throws {
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw ContextError.commandBufferFailed(
                "\((error as NSError).localizedDescription) "
                    + "[status \(commandBuffer.status.rawValue)]")
        }
        guard commandBuffer.status == .completed else {
            throw ContextError.commandBufferFailed(
                "status \(commandBuffer.status.rawValue), expected completed")
        }
    }

    /// Commits a command buffer, closing the encoder first.
    ///
    /// Committing with an encoder still open is a Metal assertion failure, so this is the only
    /// way the decode path should commit.
    public func commit(_ commandBuffer: MTLCommandBuffer) {
        closeEncoder()
        commandBuffer.commit()
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
