import Darwin
import Foundation
import HydraCore
import HydraFormat
import HydraInstall

/// Produces a precision variant from an installation already on disk.
///
/// Re-downloading 12.8 GiB to change how 2.27 GiB of it is stored would be absurd: the
/// experts, the embedding and the tokenizer are identical, and only `resident.bin` differs.
///
/// The copy is an **APFS clone**, so the shared files cost nothing until one of them is
/// written — which none of them are. In practice the variant costs the size of the converted
/// `resident.bin` alone, about 1.2 GiB for the 20B, and appears in seconds instead of
/// minutes.
enum Convert {

    static func run(config: GptOssConfig, slug: String, precision: PrecisionPolicy) throws {
        let directory = try defaultModelDirectory()
        let source = directory.appending(path: "\(slug).hydra")
        let destination = directory.appending(path: "\(slug)\(precision.installationSuffix).hydra")

        guard FileManager.default.fileExists(atPath: source.appending(path: "manifest.json").path)
        else {
            print("no published installation at \(source.lastPathComponent) — "
                + "convert needs one to copy from")
            throw ExitError.planInvalid
        }
        guard !FileManager.default.fileExists(atPath: destination.path) else {
            print("already installed: \(destination.lastPathComponent)")
            return
        }

        // The source must be what it claims before we build anything on top of it.
        let manifest = try HydraManifest.read(from: source)
        try manifest.validate(against: config, root: source)
        guard !manifest.precisionPolicy.altersPublishedWeights else {
            print("\(source.lastPathComponent) is already a variant — convert from the "
                + "published installation")
            throw ExitError.planInvalid
        }

        print("cloning \(source.lastPathComponent) → \(destination.lastPathComponent)")
        let started = Date()
        // A partial name until it is complete, exactly like the repacker: an interrupted
        // conversion must not look like a usable installation.
        let partial = destination.appendingPathExtension("partial")
        try? FileManager.default.removeItem(at: partial)
        if clonefile(source.path, partial.path, 0) != 0 {
            // Not every filesystem supports cloning; a plain copy is correct, just slower
            // and not free in disk.
            print("  (clone unavailable: \(String(cString: strerror(errno))) — copying)")
            try FileManager.default.copyItem(at: source, to: partial)
        }

        print("converting the dense weights to \(precision.dense.label)…")
        let report = try DenseQuantizer(config: config, precision: precision).run(root: partial)

        // The manifest must describe the bytes that are now there, or validation fails on a
        // perfectly good installation.
        let updated = HydraManifest(
            model: manifest.model,
            layout: .init(
                expertStrideBytes: manifest.layout.expertStrideBytes,
                expertPayloadBytes: manifest.layout.expertPayloadBytes,
                residentBytes: report.bytesAfter,
                embeddingBytes: manifest.layout.embeddingBytes,
                tensorAlignment: manifest.layout.tensorAlignment,
                pageAlignment: manifest.layout.pageAlignment),
            files: manifest.files.mapValues { $0 }.merging(
                ["resident.bin": HydraManifest.FileEntry(byteCount: report.bytesAfter)]
            ) { _, new in new },
            sourceDescription: manifest.sourceDescription,
            sourceTotalBytes: manifest.sourceTotalBytes,
            // The digests describe the **source** bytes that were downloaded. That is still
            // true, and still what a resume would check against.
            tensors: manifest.tensors,
            densePrecision: precision.dense)
        try updated.write(to: partial)
        try FileManager.default.moveItem(at: partial, to: destination)

        print(String(format: "\n  ✔ done in %.1f s", Date().timeIntervalSince(started)))
        print("  \(report.tensorsQuantized) tensors converted")
        print("  resident.bin  \(gib(report.bytesBefore)) → \(gib(report.bytesAfter))"
            + String(format: "  (−%.0f %%)",
                     Double(report.savedBytes) / Double(report.bytesBefore) * 100))
        print("  run it with: hydra chat \(config.name.contains("120") ? "120b" : "20b") "
            + "--dense \(precision.dense.rawValue) \"…\"")
    }
}
