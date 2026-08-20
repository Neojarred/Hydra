import Foundation
import HydraCore
import HydraFormat
import HydraMetal
import HydraReference
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
        // 144 does not divide by 64, so the fixture quantizes in groups of 16 instead.
        config.projectionGroupSize = 16
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

    /// The projector, against a CPU dequantization of the same weights.
    ///
    /// The batched MLX kernel reads its activations **transposed** and needs the per-group sums
    /// of them, because the affine form `q * scale + bias` carries a `bias * sum(x)` term that
    /// cannot be recovered from the quantized product. Handing it row-major activations and a
    /// zeroed sums buffer produces a finite, plausible, entirely wrong projection: the model
    /// called a French flag "overlapping text and scrambled letters", while every test upstream
    /// of the projector passed, because they all stop before it.
    ///
    /// **Checked against `MLXAffine.dequantize`, not against a shape or a spread.** The first
    /// version of this test asked only that the output be varied and non-zero, and both defects
    /// satisfied it: a wrong projection is still a spread of different numbers. Only an oracle
    /// separates them.
    /// **Groups of 64, as the checkpoint packs them.**
    ///
    /// The batched MLX kernel has two paths, and only one of them reads the sums: when a group
    /// is narrower than a chunk it cannot use them at all, and says so in its own comment. The
    /// first version of this test used groups of 16 to fit a 144-wide fixture, took that narrow
    /// path, and therefore could not see a missing `chunk_sums` however hard it looked. The
    /// tower elsewhere keeps its 72-wide heads; this one case widens to 128 so 64 divides it.
    static var wideGroupTower: Gemma4VisionConfig {
        var config = tiny
        config.hiddenSize = 128
        config.headDim = 64
        config.headCount = 2
        config.projectionGroupSize = 64
        return config
    }

    @Test("The projector matches a CPU dequantization of its own weights")
    func projectorMatchesReference() throws {
        let context = try MetalContext()
        let config = Self.wideGroupTower
        let synthetic = try SyntheticWithProjector(config: config, device: context.device)
        let tower = Gemma4VisionTower(config: config, context: context, weights: synthetic)
        // **Nine soft tokens, not one.** A 3x3 grid pools to a single token, and the batched
        // projection tiles eight tokens at a time, so one token exercises a path where the
        // per-group sums are never read and dropping them changes nothing. A 9x9 grid pools to
        // nine and crosses the tile.
        let grid = 9
        let patches = Self.deterministic(grid * grid * config.patchElements, seed: 21)
        let floats = patches.map(Float.init)

        let input = try tower.pooledForTesting(
            patches: floats, gridHeight: grid, gridWidth: grid)
        let gpu = try tower.forward(patches: floats, gridHeight: grid, gridWidth: grid)
        let tokens = (grid / config.poolingKernelSize) * (grid / config.poolingKernelSize)
        #expect(gpu.count == tokens * config.outHiddenSize)

        // The reference: the same unscaled norm, then the dequantized matrix, token by token.
        var expected: [Double] = []
        for token in 0..<tokens {
            let row = Array(
                input[(token * config.hiddenSize)..<((token + 1) * config.hiddenSize)])
            let normed = Gemma4VisionReference.rmsNorm(
                row.map(Double.init), weight: nil, eps: Double(config.rmsNormEps))
            expected += synthetic.projectOnCPU(normed)
        }

        var scale = 0.0
        for value in expected { scale = max(scale, abs(value)) }
        var worst = 0.0
        for (a, e) in zip(gpu, expected) {
            worst = max(worst, abs(Double(a) - e) / max(scale, 1e-9))
        }
        #expect(worst < 5e-3, "the projector differs from its own weights by \(worst)")
        #expect(scale > 1e-6, "the reference is flat, so the comparison proves nothing")
    }

    /// The fixture above plus a quantized projector, so the last stage can run.
    final class SyntheticWithProjector: Gemma4VisionWeightSource {
        private let base: Synthetic
        private let extra: MTLBuffer
        private var placements: [String: (offset: Int, byteCount: Int)] = [:]
        private let words: [UInt32]
        private let scaleValues: [Double]
        private let biasValues: [Double]
        private let rows: Int
        private let groupSize: Int

        init(config: Gemma4VisionConfig, device: MTLDevice) throws {
            base = try Synthetic(config: config, device: device)
            // 4-bit, group 64, as the real checkpoint packs it: eight values a `uint32`, one
            // scale and one bias a group.
            let cols = config.hiddenSize, rows = config.outHiddenSize
            let groupSize = config.projectionGroupSize
            let groups = cols / groupSize
            var words = [UInt32](repeating: 0, count: rows * cols / 8)
            for i in 0..<words.count { words[i] = UInt32(truncatingIfNeeded: i &* 2_654_435_761) }
            let scales = [Float](repeating: 0.02, count: rows * groups)
            // Biases large enough that the `bias * sum(x)` term matters. At +/-0.03 they were
            // a rounding error against the quantized product, so dropping the sums entirely
            // still passed: the test could not see the bug it was written for.
            let biases = (0..<(rows * groups)).map { Float($0 % 7) * 0.15 - 0.45 }

            var blob = Data()
            var placed: [String: (offset: Int, byteCount: Int)] = [:]
            for (name, bytes) in [
                (Gemma4VisionMapping.Name.projectionWeight, words.withUnsafeBytes { Data($0) }),
                (Gemma4VisionMapping.Name.projectionScales, BF16.encode(scales)),
                (Gemma4VisionMapping.Name.projectionBiases, BF16.encode(biases)),
            ] {
                while blob.count % 256 != 0 { blob.append(0) }
                placed[name] = (blob.count, bytes.count)
                blob.append(bytes)
            }
            placements = placed
            self.words = words
            // The values as BF16 gave them to the GPU, so the oracle sees the same numbers.
            self.scaleValues = BF16.decode(BF16.encode(scales)).map(Double.init)
            self.biasValues = BF16.decode(BF16.encode(biases)).map(Double.init)
            self.rows = rows
            self.groupSize = groupSize
            guard let buffer = blob.withUnsafeBytes({ raw in
                device.makeBuffer(
                    bytes: raw.baseAddress!, length: max(raw.count, 4),
                    options: .storageModeShared)
            }) else {
                throw Gemma4VisionTower.TowerError.allocationFailed("projector", bytes: blob.count)
            }
            extra = buffer
        }

        func tensor(_ name: String) throws -> (buffer: MTLBuffer, offset: Int, byteCount: Int) {
            if let placement = placements[name] {
                return (extra, placement.offset, placement.byteCount)
            }
            return try base.tensor(name)
        }

        /// `y = W x` with `W` dequantized by the reference, which is what the GPU must match.
        func projectOnCPU(_ x: [Double]) -> [Double] {
            let cols = x.count
            let groups = cols / groupSize
            return (0..<rows).map { row in
                let start = row * cols / 8
                let weights = MLXAffine.dequantize(
                    words: Array(words[start..<(start + cols / 8)]),
                    scales: Array(scaleValues[(row * groups)..<((row + 1) * groups)]),
                    biases: Array(biasValues[(row * groups)..<((row + 1) * groups)]),
                    bits: 4, groupSize: groupSize)
                var sum = 0.0
                for i in 0..<cols { sum += weights[i] * x[i] }
                return sum
            }
        }
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
