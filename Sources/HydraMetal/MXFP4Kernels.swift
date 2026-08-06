import Foundation
import HydraCore
import Metal

/// Enveloppes Swift des noyaux MXFP4.
///
/// Ces enveloppes existent d'abord pour la **validation** : elles permettent de comparer
/// la sortie GPU au décodeur CPU déjà vérifié bit à bit contre la référence d'OpenAI.
/// Le chemin d'inférence encodera ses propres passes dans un graphe partagé plutôt que
/// d'appeler ces fonctions une par une.
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
                return "allocation Metal impossible de \(bytes) octets"
            case .encodingFailed:
                return "encodage de la passe Metal impossible"
            case .dimensionMismatch(let detail):
                return "dimensions incohérentes : \(detail)"
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

    /// Décode des blocs MXFP4 sur le GPU. Sert de contrôle croisé du décodeur CPU.
    public func dequantize(packed: Data, scales: Data) throws -> [Float] {
        let blockCount = scales.count
        guard packed.count == blockCount * MXFP4Layout.packedBytesPerBlock else {
            throw KernelError.dimensionMismatch(
                "\(packed.count) octets packés pour \(blockCount) blocs")
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

    /// y = W·x + biais, avec W quantifiée en MXFP4 et disposée en [rows, cols].
    public func gemv(
        packed: Data, scales: Data, bias: Data?, x: [Float], rows: Int, cols: Int
    ) throws -> [Float] {
        guard cols % MXFP4Layout.blockSize == 0 else {
            throw KernelError.dimensionMismatch("cols = \(cols) n'est pas multiple de 32")
        }
        guard x.count == cols else {
            throw KernelError.dimensionMismatch("x fait \(x.count) éléments, \(cols) attendus")
        }
        let blocksPerRow = cols / MXFP4Layout.blockSize
        guard packed.count == rows * blocksPerRow * MXFP4Layout.packedBytesPerBlock,
            scales.count == rows * blocksPerRow
        else {
            throw KernelError.dimensionMismatch(
                "packed/scales ne correspondent pas à \(rows)x\(cols)")
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
        // Un threadgroup par ligne. La largeur est bornée par le nombre de blocs :
        // au-delà, les voies supplémentaires n'auraient rien à faire.
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

    /// GEMV sur des buffers déjà en place — un slot d'expert rempli par `pread`, ou une
    /// sous-plage de `resident.bin`.
    ///
    /// C'est la forme qu'utilisera l'inférence : aucune copie, aucune allocation, on lie
    /// des décalages dans des buffers qui existent déjà. La variante sur `Data` reste
    /// réservée aux tests.
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
            throw KernelError.dimensionMismatch("cols = \(cols) n'est pas multiple de 32")
        }
        let pipeline = try context.pipeline(function)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw KernelError.encodingFailed
        }

        var dims = SIMD2<UInt32>(UInt32(rows), UInt32(cols))
        var hasBias = UInt32(bias == nil ? 0 : 1)
        let blocksPerRow = cols / MXFP4Layout.blockSize
        let tiled = function.hasSuffix("_tiled")
        // Le noyau tuilé affecte un groupe SIMD par ligne et partage les activations ;
        // le noyau SIMD suppose exactement un groupe ; les autres se dimensionnent sur
        // le nombre de blocs.
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
            // Les activations vivent en mémoire partagée : une copie par threadgroup au
            // lieu d'une relecture par ligne.
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
