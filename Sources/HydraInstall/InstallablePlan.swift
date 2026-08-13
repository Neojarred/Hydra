import Foundation
import HydraCore
import HydraFormat

/// What `StreamingRepacker` needs from a plan, whichever architecture produced it.
///
/// The seam D-023 asks for, in the install path: the repacker downloads regions, routes bytes
/// to destinations and writes a manifest, and none of that depends on which model is being
/// installed. What *does* depend on the model, which tensors exist, how a blob is divided,
/// what is excluded, is settled while the plan is built, before this protocol is reached.
///
/// Extracted from what the repacker already used, not designed: every member below appears in
/// its body today, and both plans conform without changing a line of theirs.
public protocol InstallablePlan: Sendable {

    /// The model being installed, for the manifest to record.
    var model: any ModelDescriptor { get }
    var layout: HydraLayout { get }

    /// Sorted by (shard, source offset), so reading stays sequential on the remote disk.
    var operations: [ScatterCopy] { get }
    /// Contiguous source regions, each downloadable in one request.
    var spans: [RepackPlan.SourceSpan] { get }
    var destinationSizes: [DestinationFile: Int] { get }
    var totalSourceBytes: Int { get }

    /// Everything checkable **before** a single byte is downloaded.
    ///
    /// `weightMap` is a parameter rather than something the plan captured because the two
    /// architectures establish coverage differently, and the difference is not cosmetic.
    /// GPT-OSS plans every tensor in the checkpoint, so "did we miss one" is answered by
    /// comparing byte totals. Gemma deliberately leaves the audio tower behind, so its total is
    /// *supposed* to be smaller than the index's and that comparison says nothing, it has to
    /// name what it skipped and check the remainder against the index.
    func validate(
        weightMap: [String: String], declaredSourceTotal: Int?
    ) -> [RepackPlan.Problem]

    /// What sits in `vision.bin`, empty for a plan that installs no tower.
    ///
    /// **A requirement with a default, not an extension member.** Written as the latter it
    /// compiled, GPT-OSS behaved, and Gemma silently wrote a `vision.bin` the manifest
    /// described as absent, a gigabyte on disk that no later code could interpret. A
    /// protocol extension is statically dispatched through an existential, so the repacker,
    /// holding `any InstallablePlan`, got the default and never the override.
    var visionTensors: [HydraManifest.VisionTensor] { get }
}

extension InstallablePlan {
    /// The default, which GPT-OSS takes: it has no multimodal tower and never will.
    public var visionTensors: [HydraManifest.VisionTensor] { [] }

    /// What the installation will occupy. A derivation from `destinationSizes`, so it belongs
    /// here rather than being written identically by each plan.
    public var totalDestinationBytes: Int { destinationSizes.values.reduce(0, +) }
}

extension RepackPlan: InstallablePlan {
    public var model: any ModelDescriptor { config }

    /// GPT-OSS installs the whole checkpoint, so the byte total is the coverage check and the
    /// index's names add nothing.
    public func validate(
        weightMap: [String: String], declaredSourceTotal: Int?
    ) -> [Problem] {
        validate(declaredSourceTotal: declaredSourceTotal)
    }
}

extension GemmaMLXRepackPlan: InstallablePlan {
    public var model: any ModelDescriptor { config }

    public var visionTensors: [HydraManifest.VisionTensor] {
        visionPlacements.map {
            HydraManifest.VisionTensor(
                name: $0.name, offset: $0.offset, byteCount: $0.byteCount,
                dtype: $0.dtype.rawValue, shape: $0.shape)
        }
    }
}

extension QwenRepackPlan: InstallablePlan {
    public var model: any ModelDescriptor { config }

    /// The tower is installed and described, as Gemma's is (D-021). Nothing executes it yet;
    /// the bytes are what the download buys, and vision is a planned feature.
    public var visionTensors: [HydraManifest.VisionTensor] {
        visionPlacements.map {
            HydraManifest.VisionTensor(
                name: $0.name, offset: $0.offset, byteCount: $0.byteCount,
                dtype: $0.dtype.rawValue, shape: $0.shape)
        }
    }
}

extension GemmaRepackPlan: InstallablePlan {
    public var model: any ModelDescriptor { config }

    public var visionTensors: [HydraManifest.VisionTensor] {
        visionPlacements.map {
            HydraManifest.VisionTensor(
                name: $0.name, offset: $0.offset, byteCount: $0.byteCount,
                dtype: $0.dtype.rawValue, shape: $0.shape)
        }
    }
}
