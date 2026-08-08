import Foundation

/// The format a weight tensor is stored in.
///
/// Deliberately not a bit count: `mxfp4` and `q8` differ in more than width — block size,
/// scale encoding, whether the values were trained in that format. Collapsing them to a
/// number is how one ends up applying a post-training recipe to weights that were
/// quantization-aware, which D-020 identifies as the mistake that would quietly cost the
/// project its point.
public enum WeightPrecision: String, Codable, Sendable, Equatable, CaseIterable {
    /// As published. Two bytes per value, no scale.
    case bf16
    /// Symmetric `Int8`, blocks of 32, one BF16 scale per block. 8.5 bits per value.
    case q8
    /// OCP microscaling, 4 bits plus an E8M0 scale per 32 values. 4.25 bits.
    ///
    /// Listed for the expert path. Producing it ourselves from BF16 weights is **not** the
    /// same operation as receiving it from a model trained that way — see D-020.
    case mxfp4

    /// Bytes `count` values occupy in this format.
    public func byteCount(values count: Int) -> Int {
        switch self {
        case .bf16: return count * 2
        case .q8: return Q8.encodedByteCount(values: count)
        case .mxfp4:
            precondition(count % MXFP4.blockSize == 0)
            return count / MXFP4.blockSize
                * (MXFP4.packedBytesPerBlock + MXFP4.scaleBytesPerBlock)
        }
    }

    /// How much smaller than BF16, for reporting.
    public var fractionOfBF16: Double {
        Double(byteCount(values: 1024)) / Double(2048)
    }

    /// What a user sees.
    public var label: String {
        switch self {
        case .bf16: return "BF16"
        case .q8: return "Q8"
        case .mxfp4: return "MXFP4"
        }
    }
}

/// Which precision each kind of tensor is stored at, for one installation.
///
/// The policy is expressed **per role**, never per tensor name. That is what lets a second
/// model inherit it: Gemma declares the role of each of its tensors — which it has to do
/// anyway to be laid out at all — and the precision follows without a line of model-specific
/// code. It is also what makes D-020's exclusions structural: `router`, `norm`, `bias` and
/// `sink` have no branch that could put them anywhere but BF16.
public struct PrecisionPolicy: Codable, Sendable, Equatable {

    /// Attention projections and the LM head. 66 % of the bytes read per token on the 20B.
    ///
    /// This is the **only** knob, and deliberately so. The rule it encodes — routers, norms,
    /// biases and sinks stay BF16, attention and the head may move — is a property of the
    /// transformer, not of GPT-OSS, so it transfers to any model without a decision.
    public var dense: WeightPrecision

    /// What the **source checkpoint** publishes its experts in. Recorded, never chosen.
    ///
    /// There is no user knob here on purpose. Requantizing experts we received in a given
    /// format is either destructive (MXFP4 → anything) or a post-training recipe applied to
    /// weights someone else may already have quantized better (BF16 → 4 bits, when the
    /// publisher offers a quantization-aware checkpoint). Both are the mistake D-020 names.
    ///
    /// When a model publishes several variants, the choice is **which repository to install
    /// from** — a different question, settled per model, with its own measurement.
    public private(set) var experts: WeightPrecision

    public init(dense: WeightPrecision = .bf16, sourceExperts: WeightPrecision = .mxfp4) {
        self.dense = dense
        self.experts = sourceExperts
    }

    /// What the model was published as. The default, and what D-015 protects.
    public static let published = PrecisionPolicy()

    /// The dense weights in Q8, experts untouched.
    ///
    /// Measured in M-026: 98.9 % of greedy tokens unchanged on the 20B, 98.6 % on the 120B,
    /// and not one changed position was held with conviction — for 47 % fewer bytes on the
    /// tensors re-read at every token.
    public static let denseQ8 = PrecisionPolicy(dense: .q8)

    public func precision(for role: HydraLayout.TensorRole) -> WeightPrecision {
        role.isQuantizable ? dense : .bf16
    }

    /// A short suffix identifying the installation on disk, empty when nothing was changed.
    ///
    /// Two installations of the same model therefore coexist under different names, which is
    /// what lets a user compare them instead of taking our word for it.
    public var installationSuffix: String {
        dense == .bf16 ? "" : "-dense-\(dense.rawValue)"
    }

    /// Whether this policy alters anything the upstream checkpoint published.
    public var altersPublishedWeights: Bool { dense != .bf16 }

    public enum PolicyError: Error, CustomStringConvertible {
        case unknownPrecision(String)

        public var description: String {
            switch self {
            case .unknownPrecision(let value):
                return "unknown dense precision \"\(value)\" — expected bf16 or q8"
            }
        }
    }

    /// Parses a command-line choice.
    public static func dense(named value: String) throws -> PrecisionPolicy {
        switch value.lowercased() {
        case "bf16": return .published
        case "q8": return .denseQ8
        default: throw PolicyError.unknownPrecision(value)
        }
    }
}
