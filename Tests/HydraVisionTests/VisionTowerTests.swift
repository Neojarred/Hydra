import Foundation
import HydraCore
import HydraFormat
import HydraMetal
import Metal
import Testing

@testable import HydraVision

/// The GPU tower against the double-precision reference, on the same tiny tower.
///
/// This is the test the whole vision effort has been building towards. Everything before it
/// checked a piece in isolation against a transcription of the reference; this checks that the
/// pieces were **wired to each other**, which is a different question and the one that catches
/// the failures nothing else does. A LayerNorm handed the other block's weights, the rotary
/// applied to the values instead of the keys, the residual added before the projection rather
/// than after: each produces a tower that runs and returns finite numbers.
///
/// The tiny tower is two blocks of 16 rather than 27 of 1152, because the reference is written
/// for clarity and runs in double precision. The code path is identical.
@Suite("Vision tower on GPU")
struct VisionTowerTests {

    private let config = VisionReferenceTests.tiny

    /// The synthetic tower, packed as BF16 into one buffer, exactly as `vision.bin` holds it.
    ///
    /// **BF16 on both sides.** The reference is handed the values after their round trip through
    /// BF16, not the doubles they came from, so the deviation measured is the kernels' and not
    /// the format's. Skipping that is how a comparison ends up with a tolerance loose enough to
    /// hide a real fault.
    final class Synthetic: VisionWeightSource {
        let buffer: MTLBuffer
        private var placements: [String: (offset: Int, byteCount: Int)] = [:]
        private var values: [String: [Double]] = [:]

        init(config: Qwen35VisionConfig, device: MTLDevice) throws {
            let fixture = VisionReferenceTests.Fixture(config: config)
            var named: [(String, [Double])] = [
                (VisionMapping.Name.patchWeight, fixture.patchWeight()),
                (VisionMapping.Name.patchBias, fixture.patchBias()),
                (VisionMapping.Name.positionEmbedding, fixture.positionEmbedding()),
                (VisionMapping.Name.mergerNormWeight, fixture.mergerNormWeight()),
                (VisionMapping.Name.mergerNormBias, fixture.mergerNormBias()),
                (VisionMapping.Name.mergerFC1Weight, fixture.mergerFC1Weight()),
                (VisionMapping.Name.mergerFC1Bias, fixture.mergerFC1Bias()),
                (VisionMapping.Name.mergerFC2Weight, fixture.mergerFC2Weight()),
                (VisionMapping.Name.mergerFC2Bias, fixture.mergerFC2Bias()),
            ]
            for layer in 0..<config.depth {
                named += [
                    (VisionMapping.Name.norm1Weight(layer), fixture.norm1Weight(layer)),
                    (VisionMapping.Name.norm1Bias(layer), fixture.norm1Bias(layer)),
                    (VisionMapping.Name.norm2Weight(layer), fixture.norm2Weight(layer)),
                    (VisionMapping.Name.norm2Bias(layer), fixture.norm2Bias(layer)),
                    (VisionMapping.Name.qkvWeight(layer), fixture.qkvWeight(layer)),
                    (VisionMapping.Name.qkvBias(layer), fixture.qkvBias(layer)),
                    (VisionMapping.Name.projWeight(layer), fixture.projWeight(layer)),
                    (VisionMapping.Name.projBias(layer), fixture.projBias(layer)),
                    (VisionMapping.Name.fc1Weight(layer), fixture.fc1Weight(layer)),
                    (VisionMapping.Name.fc1Bias(layer), fixture.fc1Bias(layer)),
                    (VisionMapping.Name.fc2Weight(layer), fixture.fc2Weight(layer)),
                    (VisionMapping.Name.fc2Bias(layer), fixture.fc2Bias(layer)),
                ]
            }

            // 256-byte aligned, as the installer aligns them, so an offset bug here would not be
            // masked by everything happening to be contiguous.
            var packed: [UInt16] = []
            var cursor = 0
            for (name, doubles) in named {
                while cursor % 128 != 0 { packed.append(0); cursor += 1 }
                let bits = BF16.encode(doubles.map(Float.init))
                placements[name] = (cursor * 2, doubles.count * 2)
                bits.withUnsafeBytes { raw in
                    for i in 0..<doubles.count {
                        packed.append(raw.loadUnaligned(fromByteOffset: i * 2, as: UInt16.self))
                    }
                }
                cursor += doubles.count
                // The values as the GPU will see them, for the reference to use.
                values[name] = BF16.decode(bits).map(Double.init)
            }

            guard let buffer = packed.withUnsafeBytes({ raw in
                device.makeBuffer(
                    bytes: raw.baseAddress!, length: max(raw.count, 4),
                    options: .storageModeShared)
            }) else { throw VisionTower.TowerError.allocationFailed("fixture", bytes: cursor * 2) }
            self.buffer = buffer
        }

        func tensor(_ name: String) throws -> (buffer: MTLBuffer, offset: Int, byteCount: Int) {
            guard let placement = placements[name] else {
                throw VisionMapping.MappingError.tensorMissing(name)
            }
            return (buffer, placement.offset, placement.byteCount)
        }

        /// The same weights the GPU has, for the reference.
        func quantized() -> Quantized { Quantized(values: values) }

        struct Quantized: VisionReference.Weights {
            let values: [String: [Double]]
            private func get(_ name: String) -> [Double] { values[name] ?? [] }

            func patchWeight() -> [Double] { get(VisionMapping.Name.patchWeight) }
            func patchBias() -> [Double] { get(VisionMapping.Name.patchBias) }
            func positionEmbedding() -> [Double] { get(VisionMapping.Name.positionEmbedding) }
            func norm1Weight(_ l: Int) -> [Double] { get(VisionMapping.Name.norm1Weight(l)) }
            func norm1Bias(_ l: Int) -> [Double] { get(VisionMapping.Name.norm1Bias(l)) }
            func norm2Weight(_ l: Int) -> [Double] { get(VisionMapping.Name.norm2Weight(l)) }
            func norm2Bias(_ l: Int) -> [Double] { get(VisionMapping.Name.norm2Bias(l)) }
            func qkvWeight(_ l: Int) -> [Double] { get(VisionMapping.Name.qkvWeight(l)) }
            func qkvBias(_ l: Int) -> [Double] { get(VisionMapping.Name.qkvBias(l)) }
            func projWeight(_ l: Int) -> [Double] { get(VisionMapping.Name.projWeight(l)) }
            func projBias(_ l: Int) -> [Double] { get(VisionMapping.Name.projBias(l)) }
            func fc1Weight(_ l: Int) -> [Double] { get(VisionMapping.Name.fc1Weight(l)) }
            func fc1Bias(_ l: Int) -> [Double] { get(VisionMapping.Name.fc1Bias(l)) }
            func fc2Weight(_ l: Int) -> [Double] { get(VisionMapping.Name.fc2Weight(l)) }
            func fc2Bias(_ l: Int) -> [Double] { get(VisionMapping.Name.fc2Bias(l)) }
            func mergerNormWeight() -> [Double] { get(VisionMapping.Name.mergerNormWeight) }
            func mergerNormBias() -> [Double] { get(VisionMapping.Name.mergerNormBias) }
            func mergerFC1Weight() -> [Double] { get(VisionMapping.Name.mergerFC1Weight) }
            func mergerFC1Bias() -> [Double] { get(VisionMapping.Name.mergerFC1Bias) }
            func mergerFC2Weight() -> [Double] { get(VisionMapping.Name.mergerFC2Weight) }
            func mergerFC2Bias() -> [Double] { get(VisionMapping.Name.mergerFC2Bias) }
        }
    }

    /// Deviation relative to the vector's own magnitude, not per component: near zero a relative
    /// error measures cancellation, not the kernel.
    private func deviation(_ actual: [Float], _ expected: [Double]) -> Double {
        var scale = 0.0
        for value in expected { scale = max(scale, abs(value)) }
        var worst = 0.0
        for (a, e) in zip(actual, expected) {
            worst = max(worst, abs(Double(a) - e) / max(scale, 1e-9))
        }
        return worst
    }

    @Test("The GPU tower agrees with the reference", arguments: [(2, 2), (4, 6), (6, 4)])
    func agreesWithReference(shape: (Int, Int)) throws {
        let context = try MetalContext()
        let synthetic = try Synthetic(config: config, device: context.device)
        let grid = Qwen35VisionConfig.Grid(
            temporal: 1, height: shape.0, width: shape.1)
        let patches = VisionReferenceTests.deterministic(
            grid.patchCount * config.patchElements, seed: 7)

        let tower = VisionTower(config: config, context: context, weights: synthetic)
        let gpu = try tower.forward(patches: patches.map(Float.init), grid: grid)

        let expected = VisionReference(config: config).forward(
            patches: patches, grid: grid, weights: synthetic.quantized())

        let expectedCount = expected.count * config.outHiddenSize
        #expect(gpu.count == expectedCount)
        #expect(gpu.allSatisfy { $0.isFinite })

        for token in 0..<expected.count {
            let slice = Array(
                gpu[(token * config.outHiddenSize)..<((token + 1) * config.outHiddenSize)])
            let worst = deviation(slice, expected[token])
            #expect(worst < 3e-3, "token \(token) of a \(shape.1)x\(shape.0) grid differs by \(worst)")
        }

        // The output must not be degenerate, or the comparison above is satisfied by two
        // constant vectors agreeing with each other.
        let spread = (gpu.max() ?? 0) - (gpu.min() ?? 0)
        #expect(spread > 1e-4, "the tower returned a constant, so agreement proves nothing")
    }

    /// A grid whose patch count is below the number of simdgroups the attention splits across.
    ///
    /// Four patches over eight simdgroups leaves half of them with no keys at all. Those keep a
    /// running maximum of -infinity, and `exp(-inf - -inf)` is NaN, which reaches every patch.
    /// The decode kernel shipped exactly this bug once; the vision kernel guards against it and
    /// this is the shape that exercises the guard.
    @Test("A grid smaller than the attention's split stays finite")
    func fewerPatchesThanSimdgroups() throws {
        let context = try MetalContext()
        let synthetic = try Synthetic(config: config, device: context.device)
        let grid = Qwen35VisionConfig.Grid(temporal: 1, height: 2, width: 2)
        let patches = VisionReferenceTests.deterministic(
            grid.patchCount * config.patchElements, seed: 11)

        let tower = VisionTower(config: config, context: context, weights: synthetic)
        let gpu = try tower.forward(patches: patches.map(Float.init), grid: grid)
        #expect(gpu.allSatisfy { $0.isFinite }, "four patches produced a non-finite token")

        let expected = VisionReference(config: config).forward(
            patches: patches, grid: grid, weights: synthetic.quantized())
        let worst = deviation(gpu, expected[0])
        #expect(worst < 3e-3, "differs by \(worst)")
    }

    /// The tower refuses a patch buffer that does not match the grid, rather than reading past
    /// the end of it.
    /// The generic kernel, which serves any head width with no specialization compiled.
    ///
    /// The suite runs Qwen's width of 72 everywhere else, so without this the fallback would be
    /// dead code that still ships. A width of 8 has no `[[host_name]]` and takes it.
    @Test("The generic kernel agrees too, at a width with no specialization")
    func genericKernelAgrees() throws {
        var narrow = config
        narrow.hiddenSize = 16
        narrow.headCount = 2            // head dim 8: no specialization exists for it

        let context = try MetalContext()
        let synthetic = try Synthetic(config: narrow, device: context.device)
        let grid = Qwen35VisionConfig.Grid(temporal: 1, height: 4, width: 6)
        let patches = VisionReferenceTests.deterministic(
            grid.patchCount * narrow.patchElements, seed: 31)

        let tower = VisionTower(config: narrow, context: context, weights: synthetic)
        let gpu = try tower.forward(patches: patches.map(Float.init), grid: grid)
        let expected = VisionReference(config: narrow).forward(
            patches: patches, grid: grid, weights: synthetic.quantized())

        #expect(gpu.allSatisfy { $0.isFinite })
        for token in 0..<expected.count {
            let slice = Array(
                gpu[(token * narrow.outHiddenSize)..<((token + 1) * narrow.outHiddenSize)])
            let worst = deviation(slice, expected[token])
            #expect(worst < 3e-3, "generic kernel, token \(token) differs by \(worst)")
        }
    }

    @Test("A mismatched patch count is refused")
    func refusesAMismatch() throws {
        let context = try MetalContext()
        let synthetic = try Synthetic(config: config, device: context.device)
        let tower = VisionTower(config: config, context: context, weights: synthetic)
        let grid = Qwen35VisionConfig.Grid(temporal: 1, height: 4, width: 4)
        #expect(throws: VisionTower.TowerError.self) {
            _ = try tower.forward(patches: [Float](repeating: 0, count: 10), grid: grid)
        }
    }
}
