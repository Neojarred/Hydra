import Foundation
import HydraCore
import HydraFormat
import HydraMetal
import Metal
import Testing

@testable import HydraVision

/// Gemma's GPU tower against its double-precision reference, on the same tiny tower.
///
/// The test that decides whether the whole thing works. Everything before it checked a piece
/// against a transcription; this checks the pieces were wired to each other, which is where the
/// failures nothing else catches live: a sandwich norm applied only on one side, the value norm
/// omitted, the two position tables read as one, the rotary pairing across the axis boundary.
/// Every one of those runs and returns finite numbers.
@Suite("Gemma 4 vision tower on GPU")
struct Gemma4VisionTowerTests {

    /// Shaped like the real tower and small enough for a double-precision oracle.
    ///
    /// The head width is 72, Qwen's and Gemma's real one, so the specialized attention kernel
    /// that ships is the one under test rather than the generic fallback. Everything else is
    /// shrunk. Widths are multiples of eight because `bf16_gemm` reads columns in groups of that.
    static var tiny: Gemma4VisionConfig {
        var config = Gemma4VisionConfig()
        config.depth = 2
        config.hiddenSize = 144          // two heads of 72
        config.headCount = 2
        config.intermediateSize = 32
        config.outHiddenSize = 24
        // Patch 4, so patchElements is 48: bf16_gemm reads columns in groups of eight and a
        // 12-wide patch would trip the precondition that exists to catch exactly that.
        config.patchSize = 4
        config.positionEmbeddingSize = 32
        return config
    }

    private let config = tiny

    static func deterministic(_ count: Int, seed: UInt64) -> [Double] {
        var hash = 0xcbf2_9ce4_8422_2325 ^ seed
        return (0..<count).map { _ in
            hash = (hash ^ 0x9E37) &* 0x0000_0100_0000_01B3
            return Double(hash % 2000) / 1000.0 - 1.0
        }
    }

    /// The synthetic tower, BF16 on both sides so the deviation measured is the kernels' and not
    /// the format's.
    final class Synthetic: Gemma4VisionWeightSource {
        let buffer: MTLBuffer
        private var placements: [String: (offset: Int, byteCount: Int)] = [:]
        private(set) var values: [String: [Double]] = [:]

        init(config: Gemma4VisionConfig, device: MTLDevice) throws {
            let hidden = config.hiddenSize, inter = config.intermediateSize
            func make(_ tag: UInt64, _ count: Int) -> [Double] {
                Gemma4VisionTowerTests.deterministic(count, seed: tag)
            }
            // Norm weights sit near one, as the real checkpoint's do: Gemma 4 multiplies by the
            // weight directly, so weights near zero would silently annihilate the tower.
            func norm(_ tag: UInt64, _ count: Int) -> [Double] {
                make(tag, count).map { 1 + 0.1 * $0 }
            }
            var named: [(String, [Double])] = [
                (Gemma4VisionMapping.Name.patchProjection, make(1, hidden * config.patchElements)),
                (Gemma4VisionMapping.Name.positionTable,
                 make(2, 2 * config.positionEmbeddingSize * hidden)),
                (Gemma4VisionMapping.Name.standardizationBias, make(3, hidden).map { $0 * 0.1 }),
                (Gemma4VisionMapping.Name.standardizationScale, norm(4, hidden)),
            ]
            for l in 0..<config.depth {
                let base = UInt64(10 + l * 20)
                named += [
                    (Gemma4VisionMapping.Name.inputNorm(l), norm(base, hidden)),
                    (Gemma4VisionMapping.Name.postAttentionNorm(l), norm(base + 1, hidden)),
                    (Gemma4VisionMapping.Name.preFeedforwardNorm(l), norm(base + 2, hidden)),
                    (Gemma4VisionMapping.Name.postFeedforwardNorm(l), norm(base + 3, hidden)),
                    (Gemma4VisionMapping.Name.query(l), make(base + 4, hidden * hidden)),
                    (Gemma4VisionMapping.Name.key(l), make(base + 5, hidden * hidden)),
                    (Gemma4VisionMapping.Name.value(l), make(base + 6, hidden * hidden)),
                    (Gemma4VisionMapping.Name.output(l), make(base + 7, hidden * hidden)),
                    (Gemma4VisionMapping.Name.queryNorm(l), norm(base + 8, config.headDim)),
                    (Gemma4VisionMapping.Name.keyNorm(l), norm(base + 9, config.headDim)),
                    (Gemma4VisionMapping.Name.gate(l), make(base + 10, inter * hidden)),
                    (Gemma4VisionMapping.Name.up(l), make(base + 11, inter * hidden)),
                    (Gemma4VisionMapping.Name.down(l), make(base + 12, hidden * inter)),
                ]
            }

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
                values[name] = BF16.decode(bits).map(Double.init)
            }
            guard let buffer = packed.withUnsafeBytes({ raw in
                device.makeBuffer(
                    bytes: raw.baseAddress!, length: max(raw.count, 4),
                    options: .storageModeShared)
            }) else {
                throw Gemma4VisionTower.TowerError.allocationFailed("fixture", bytes: cursor * 2)
            }
            self.buffer = buffer
        }

        func tensor(_ name: String) throws -> (buffer: MTLBuffer, offset: Int, byteCount: Int) {
            guard let placement = placements[name] else {
                throw Gemma4VisionMapping.MappingError.tensorMissing(name)
            }
            return (buffer, placement.offset, placement.byteCount)
        }

        func oracle() -> Oracle { Oracle(values: values) }

        struct Oracle: Gemma4VisionReference.Weights {
            let values: [String: [Double]]
            private func get(_ n: String) -> [Double] { values[n] ?? [] }
            func patchProjection() -> [Double] { get(Gemma4VisionMapping.Name.patchProjection) }
            func positionTable() -> [Double] { get(Gemma4VisionMapping.Name.positionTable) }
            func inputNorm(_ l: Int) -> [Double] { get(Gemma4VisionMapping.Name.inputNorm(l)) }
            func postAttentionNorm(_ l: Int) -> [Double] { get(Gemma4VisionMapping.Name.postAttentionNorm(l)) }
            func preFeedforwardNorm(_ l: Int) -> [Double] { get(Gemma4VisionMapping.Name.preFeedforwardNorm(l)) }
            func postFeedforwardNorm(_ l: Int) -> [Double] { get(Gemma4VisionMapping.Name.postFeedforwardNorm(l)) }
            func queryProjection(_ l: Int) -> [Double] { get(Gemma4VisionMapping.Name.query(l)) }
            func keyProjection(_ l: Int) -> [Double] { get(Gemma4VisionMapping.Name.key(l)) }
            func valueProjection(_ l: Int) -> [Double] { get(Gemma4VisionMapping.Name.value(l)) }
            func outputProjection(_ l: Int) -> [Double] { get(Gemma4VisionMapping.Name.output(l)) }
            func queryNorm(_ l: Int) -> [Double] { get(Gemma4VisionMapping.Name.queryNorm(l)) }
            func keyNorm(_ l: Int) -> [Double] { get(Gemma4VisionMapping.Name.keyNorm(l)) }
            func gateProjection(_ l: Int) -> [Double] { get(Gemma4VisionMapping.Name.gate(l)) }
            func upProjection(_ l: Int) -> [Double] { get(Gemma4VisionMapping.Name.up(l)) }
            func downProjection(_ l: Int) -> [Double] { get(Gemma4VisionMapping.Name.down(l)) }
            func standardizationBias() -> [Double] { get(Gemma4VisionMapping.Name.standardizationBias) }
            func standardizationScale() -> [Double] { get(Gemma4VisionMapping.Name.standardizationScale) }
            func projection() -> [Double] { [] }
        }
    }

    private func deviation(_ actual: [Float], _ expected: [Double]) -> Double {
        var scale = 0.0
        for value in expected { scale = max(scale, abs(value)) }
        var worst = 0.0
        for (a, e) in zip(actual, expected) {
            worst = max(worst, abs(Double(a) - e) / max(scale, 1e-9))
        }
        return worst
    }

    /// The tower up to the projector, which is the part both sides can compute.
    ///
    /// The projector is the tower's one quantized tensor and the synthetic fixture has no
    /// quantized weights, so the comparison stops at the standardized soft tokens. Everything
    /// architectural is upstream of it; the projector is a single matrix multiply the MLX kernels
    /// are already tested on.
    @Test("The GPU tower agrees with the reference", arguments: [(3, 3), (3, 6), (6, 3)])
    func agreesWithReference(shape: (Int, Int)) throws {
        let context = try MetalContext()
        let synthetic = try Synthetic(config: config, device: context.device)
        let (gridHeight, gridWidth) = shape
        let patches = Self.deterministic(
            gridHeight * gridWidth * config.patchElements, seed: 7)

        let tower = Gemma4VisionTower(config: config, context: context, weights: synthetic)
        let gpu = try tower.pooledForTesting(
            patches: patches.map(Float.init), gridHeight: gridHeight, gridWidth: gridWidth)
        let expected = Gemma4VisionReference(config: config).pooledForTesting(
            patches: patches, gridHeight: gridHeight, gridWidth: gridWidth,
            weights: synthetic.oracle())

        #expect(gpu.count == expected.count * config.hiddenSize)
        #expect(gpu.allSatisfy { $0.isFinite })
        for token in 0..<expected.count {
            let slice = Array(
                gpu[(token * config.hiddenSize)..<((token + 1) * config.hiddenSize)])
            let worst = deviation(slice, expected[token])
            #expect(worst < 5e-3, "token \(token) of a \(gridWidth)x\(gridHeight) grid differs by \(worst)")
        }
        let spread = (gpu.max() ?? 0) - (gpu.min() ?? 0)
        #expect(spread > 1e-4, "the tower returned a constant, so agreement proves nothing")
    }

    /// A later patch changes an earlier token: the attention is bidirectional.
    @Test("A later patch changes an earlier token")
    func attentionIsBidirectional() throws {
        let context = try MetalContext()
        let synthetic = try Synthetic(config: config, device: context.device)
        var patches = Self.deterministic(9 * config.patchElements, seed: 7)
        let tower = Gemma4VisionTower(config: config, context: context, weights: synthetic)

        let baseline = try tower.pooledForTesting(
            patches: patches.map(Float.init), gridHeight: 3, gridWidth: 3)
        let last = 8 * config.patchElements
        for i in last..<(last + config.patchElements) { patches[i] += 0.5 }
        let perturbed = try tower.pooledForTesting(
            patches: patches.map(Float.init), gridHeight: 3, gridWidth: 3)

        let change = zip(baseline, perturbed).map { abs($0 - $1) }.max() ?? 0
        #expect(change > 1e-6, "the last patch did not reach the pooled token")
    }

    @Test("A mismatched patch count is refused")
    func refusesAMismatch() throws {
        let context = try MetalContext()
        let synthetic = try Synthetic(config: config, device: context.device)
        let tower = Gemma4VisionTower(config: config, context: context, weights: synthetic)
        #expect(throws: Gemma4VisionTower.TowerError.self) {
            _ = try tower.forward(
                patches: [Float](repeating: 0, count: 10), gridHeight: 3, gridWidth: 3)
        }
    }
}
