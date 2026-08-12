import Foundation
import HydraCore
import Metal

/// Swift wrappers around the MXFP4 kernels.
///
/// These wrappers exist first for **validation**: they let the GPU output be compared with
/// the CPU decoder, itself already verified bit for bit against OpenAI's reference. The
/// inference path encodes its own passes into a shared graph rather than calling these
/// functions one at a time.
public struct MXFP4Kernels: Sendable {

    public let context: MetalContext

    public init(context: MetalContext) {
        self.context = context
    }

    public enum KernelError: Error, CustomStringConvertible {
        case bufferAllocationFailed(bytes: Int)
        case encodingFailed
        case dimensionMismatch(String)

        public var description: String {
            switch self {
            case .bufferAllocationFailed(let bytes):
                return "cannot allocate \(bytes) bytes of Metal memory"
            case .encodingFailed:
                return "cannot encode the Metal pass"
            case .dimensionMismatch(let detail):
                return "inconsistent dimensions: \(detail)"
            }
        }
    }

    private func makeBuffer(_ data: Data) throws -> MTLBuffer {
        guard
            let buffer = data.withUnsafeBytes({ raw in
                context.device.makeBuffer(bytes: raw.baseAddress!, length: raw.count, options: .storageModeShared)
            })
        else {
            throw KernelError.bufferAllocationFailed(bytes: data.count)
        }
        return buffer
    }

    /// Decodes MXFP4 blocks on the GPU. Serves as a cross-check of the CPU decoder.
    public func dequantize(packed: Data, scales: Data) throws -> [Float] {
        let blockCount = scales.count
        guard packed.count == blockCount * MXFP4Layout.packedBytesPerBlock else {
            throw KernelError.dimensionMismatch(
                "\(packed.count) packed bytes for \(blockCount) blocks")
        }

        let pipeline = try context.pipeline("mxfp4_dequantize")
        let blocksBuffer = try makeBuffer(packed)
        let scalesBuffer = try makeBuffer(scales)
        let outputBytes = blockCount * MXFP4Layout.blockSize * MemoryLayout<Float>.size
        guard let output = context.device.makeBuffer(length: outputBytes, options: .storageModeShared)
        else {
            throw KernelError.bufferAllocationFailed(bytes: outputBytes)
        }

        guard let commandBuffer = context.commandQueue.makeCommandBuffer(),
            let encoder = commandBuffer.makeComputeCommandEncoder()
        else { throw KernelError.encodingFailed }

        var count = UInt32(blockCount)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(blocksBuffer, offset: 0, index: 0)
        encoder.setBuffer(scalesBuffer, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        encoder.setBytes(&count, length: 4, index: 3)
        encoder.dispatchThreads(
            MTLSize(width: blockCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: min(pipeline.maxTotalThreadsPerThreadgroup, blockCount), height: 1, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let values = output.contents().bindMemory(
            to: Float.self, capacity: blockCount * MXFP4Layout.blockSize)
        return Array(UnsafeBufferPointer(start: values, count: blockCount * MXFP4Layout.blockSize))
    }

    /// y = W·x + bias, with W quantized in MXFP4 and laid out as [rows, cols].
    public func gemv(
        packed: Data, scales: Data, bias: Data?, x: [Float], rows: Int, cols: Int
    ) throws -> [Float] {
        guard cols % MXFP4Layout.blockSize == 0 else {
            throw KernelError.dimensionMismatch("cols = \(cols) is not a multiple of 32")
        }
        guard x.count == cols else {
            throw KernelError.dimensionMismatch("x has \(x.count) elements, \(cols) expected")
        }
        let blocksPerRow = cols / MXFP4Layout.blockSize
        guard packed.count == rows * blocksPerRow * MXFP4Layout.packedBytesPerBlock,
            scales.count == rows * blocksPerRow
        else {
            throw KernelError.dimensionMismatch(
                "packed/scales do not match \(rows)x\(cols)")
        }

        let pipeline = try context.pipeline("mxfp4_gemv")
        let blocksBuffer = try makeBuffer(packed)
        let scalesBuffer = try makeBuffer(scales)
        let biasBuffer = try makeBuffer(bias ?? Data(count: max(2, rows * 2)))
        let xBuffer = try makeBuffer(x.withUnsafeBytes { Data($0) })
        guard let output = context.device.makeBuffer(
            length: rows * MemoryLayout<Float>.size, options: .storageModeShared)
        else {
            throw KernelError.bufferAllocationFailed(bytes: rows * 4)
        }

        guard let commandBuffer = context.commandQueue.makeCommandBuffer(),
            let encoder = commandBuffer.makeComputeCommandEncoder()
        else { throw KernelError.encodingFailed }

        var dims = SIMD2<UInt32>(UInt32(rows), UInt32(cols))
        var hasBias = UInt32(bias == nil ? 0 : 1)
        // One threadgroup per row. The width is bounded by the block count: beyond that,
        // the extra lanes would have nothing to do.
        let width = min(pipeline.maxTotalThreadsPerThreadgroup, max(32, blocksPerRow.roundedUpToMultipleOf32))

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(blocksBuffer, offset: 0, index: 0)
        encoder.setBuffer(scalesBuffer, offset: 0, index: 1)
        encoder.setBuffer(biasBuffer, offset: 0, index: 2)
        encoder.setBuffer(xBuffer, offset: 0, index: 3)
        encoder.setBuffer(output, offset: 0, index: 4)
        encoder.setBytes(&dims, length: MemoryLayout<SIMD2<UInt32>>.size, index: 5)
        encoder.setBytes(&hasBias, length: 4, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(width: rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let values = output.contents().bindMemory(to: Float.self, capacity: rows)
        return Array(UnsafeBufferPointer(start: values, count: rows))
    }
}

extension MXFP4Kernels {

    /// GEMV over buffers already in place, an expert slot filled by `pread`, or a
    /// sub-range of `resident.bin`.
    ///
    /// This is the form inference uses: no copy, no allocation, we bind offsets into buffers
    /// that already exist. The `Data` variant stays reserved for tests.
    public func gemv(
        function: String = "mxfp4_gemv",
        blocks: MTLBuffer, blocksOffset: Int,
        scales: MTLBuffer, scalesOffset: Int,
        bias: MTLBuffer?, biasOffset: Int,
        x: MTLBuffer, xOffset: Int,
        output: MTLBuffer, outputOffset: Int,
        rows: Int, cols: Int,
        in commandBuffer: MTLCommandBuffer
    ) throws {
        guard cols % MXFP4Layout.blockSize == 0 else {
            throw KernelError.dimensionMismatch("cols = \(cols) is not a multiple of 32")
        }
        let pipeline = try context.pipeline(function)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw KernelError.encodingFailed
        }

        var dims = SIMD2<UInt32>(UInt32(rows), UInt32(cols))
        var hasBias = UInt32(bias == nil ? 0 : 1)
        let blocksPerRow = cols / MXFP4Layout.blockSize
        let tiled = function.hasSuffix("_tiled")
        // The tiled kernel assigns one SIMD group per row and shares the activations; the
        // SIMD kernel assumes exactly one group; the others are sized on the block count.
        let width: Int
        if tiled {
            width = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        } else if function.hasSuffix("_simd") {
            width = 32
        } else {
            width = min(
                pipeline.maxTotalThreadsPerThreadgroup,
                max(32, blocksPerRow.roundedUpToMultipleOf32))
        }
        let rowsPerGroup = tiled ? width / 32 : 1
        let groupCount = (rows + rowsPerGroup - 1) / rowsPerGroup

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(blocks, offset: blocksOffset, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(bias ?? blocks, offset: bias == nil ? 0 : biasOffset, index: 2)
        encoder.setBuffer(x, offset: xOffset, index: 3)
        encoder.setBuffer(output, offset: outputOffset, index: 4)
        encoder.setBytes(&dims, length: MemoryLayout<SIMD2<UInt32>>.size, index: 5)
        encoder.setBytes(&hasBias, length: 4, index: 6)
        if tiled {
            // Activations live in threadgroup memory: one copy per threadgroup instead of
            // one re-read per row.
            encoder.setThreadgroupMemoryLength(cols * MemoryLayout<Float>.size, index: 0)
        }
        encoder.dispatchThreadgroups(
            MTLSize(width: groupCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
    }
}

extension Int {
    var roundedUpToMultipleOf32: Int { (self + 31) / 32 * 32 }
}
