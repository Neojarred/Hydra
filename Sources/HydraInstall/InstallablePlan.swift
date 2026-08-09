import Foundation
import HydraCore
import HydraFormat

/// What `StreamingRepacker` needs from a plan, whichever architecture produced it.
///
/// The seam D-023 asks for, in the install path: the repacker downloads regions, routes bytes
/// to destinations and writes a manifest, and none of that depends on which model is being
/// installed. What *does* depend on the model — which tensors exist, how a blob is divided,
/// what is excluded — is settled while the plan is built, before this protocol is reached.
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
}

extension RepackPlan: InstallablePlan {
    public var model: any ModelDescriptor { config }
}

extension GemmaRepackPlan: InstallablePlan {
    public var model: any ModelDescriptor { config }
}
