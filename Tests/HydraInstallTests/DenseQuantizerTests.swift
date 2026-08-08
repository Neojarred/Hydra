import Foundation
import HydraCore
import Testing

@testable import HydraFormat
@testable import HydraInstall

/// Converting `resident.bin` to Q8 after the download.
///
/// What must hold is not "it produces a file" but that **the file says what the layout
/// claims**: a tensor read back at the offset the runtime will use must give the values that
/// were there, within the quantization bound. An offset error here would raise nothing — the
/// model would read plausible garbage — which is the failure mode this whole area keeps
/// producing.
@Suite("Dense quantization at install time")
struct DenseQuantizerTests {

    private func makeResident(config: GptOssConfig, at root: URL) throws -> [String: [Float]] {
        let layout = HydraLayout(config: config, precision: .published)
        var bytes = [UInt8](repeating: 0, count: layout.residentBytes)
        var expected: [String: [Float]] = [:]

        var state: UInt32 = 0x1234_5678
        for placement in layout.resident {
            var values = [Float](repeating: 0, count: placement.byteCount / 2)
            for i in 0..<values.count {
                state = state &* 1_103_515_245 &+ 12345
                values[i] = (Float(state >> 8) / Float(1 << 24) * 2 - 1) * 0.35
            }
            expected[placement.sourceName] = values
            for (i, value) in values.enumerated() {
                let bits = BF16.fromFloat(value)
                bytes[placement.offset + i * 2] = UInt8(bits & 0xFF)
                bytes[placement.offset + i * 2 + 1] = UInt8(bits >> 8)
            }
        }
        try Data(bytes).write(to: root.appending(path: "resident.bin"))
        return expected
    }

    @Test("Every tensor reads back at the offset the runtime will use")
    func roundTripsThroughTheLayout() throws {
        let config = GptOssConfig.tiny
        let root = FileManager.default.temporaryDirectory
            .appending(path: "hydra-q8-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let expected = try makeResident(config: config, at: root)
        let report = try DenseQuantizer(config: config, precision: .denseQ8).run(root: root)

        let target = HydraLayout(config: config, precision: .denseQ8)
        let data = try Data(contentsOf: root.appending(path: "resident.bin"))
        #expect(data.count == target.residentBytes)
        #expect(report.bytesAfter == target.residentBytes)
        #expect(report.savedBytes > 0, "converting must actually save bytes")

        for placement in target.resident {
            let original = expected[placement.sourceName]!

            switch placement.precision {
            case .bf16:
                // Untouched roles must come through **bit for bit**: a norm or a router that
                // drifted would be a silent corruption, not a rounding.
                for i in 0..<original.count {
                    let bits = UInt16(data[placement.offset + i * 2])
                        | (UInt16(data[placement.offset + i * 2 + 1]) << 8)
                    #expect(
                        BF16.toFloat(bits) == BF16.toFloat(BF16.fromFloat(original[i])),
                        "\(placement.sourceName) drifted at \(i)")
                }

            case .q8:
                let levels = data[placement.offset..<placement.scaleOffset]
                let scales = data[placement.scaleOffset..<placement.end]
                let restored = try Q8.decode(levels: Data(levels), scales: Data(scales))
                #expect(restored.count == original.count)

                var worst: Float = 0
                var start = 0
                while start + Q8.blockSize <= original.count {
                    var magnitude: Float = 0
                    for i in start..<(start + Q8.blockSize) {
                        magnitude = max(magnitude, abs(original[i]))
                    }
                    if magnitude > 0 {
                        for i in start..<(start + Q8.blockSize) {
                            worst = max(worst, abs(restored[i] - original[i]) / magnitude)
                        }
                    }
                    start += Q8.blockSize
                }
                #expect(worst < 0.01, "\(placement.sourceName) lost \(worst)")

            case .mxfp4:
                Issue.record("no resident tensor should be MXFP4")
            }
        }
    }

    /// The roles D-020 excludes must be excluded **by the layout**, not by the quantizer
    /// remembering to skip them.
    @Test("Routers, norms, biases and sinks stay BF16")
    func excludedRolesAreUntouched() {
        let target = HydraLayout(config: .b20, precision: .denseQ8)
        for placement in target.resident {
            switch placement.role {
            case .router, .norm, .bias, .sink:
                #expect(
                    placement.precision == .bf16,
                    "\(placement.sourceName) (\(placement.role)) must not be quantized")
            case .attentionProjection, .lmHead:
                #expect(placement.precision == .q8)
            }
        }
    }

    @Test("The published policy changes nothing")
    func publishedPolicyIsIdentity() throws {
        let published = HydraLayout(config: .b20, precision: .published)
        let same = HydraLayout(config: .b20, precision: PrecisionPolicy(dense: .bf16))
        #expect(published.residentBytes == same.residentBytes)
        #expect(published.resident.allSatisfy { $0.precision == .bf16 })
        #expect(!PrecisionPolicy.published.altersPublishedWeights)
        #expect(PrecisionPolicy.published.installationSuffix.isEmpty)
        #expect(PrecisionPolicy.denseQ8.installationSuffix == "-dense-q8")
    }

    /// The saving D-020 is after, stated as a property rather than a comment.
    @Test("Q8 removes about 47 % of the dense weights")
    func savingIsWhatWasClaimed() {
        let published = HydraLayout(config: .b20, precision: .published)
        let quantized = HydraLayout(config: .b20, precision: .denseQ8)
        let ratio = Double(quantized.residentBytes) / Double(published.residentBytes)
        #expect(ratio > 0.5 && ratio < 0.58, "resident.bin went to \(ratio) of its size")
    }
}
