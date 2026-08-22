// swift-tools-version: 6.2
import PackageDescription

// A deliberate split (see docs/01-DECISIONS.md, D-002):
// HydraCore, HydraFormat, HydraInstall and HydraTokenize do NOT import Metal.
// They stay portable Swift. Only HydraMetal / HydraRuntime / HydraApp are tied
// to the platform.
let package = Package(
    name: "Hydra",
    platforms: [.macOS("26.0")],
    products: [
        .library(name: "HydraCore", targets: ["HydraCore"]),
        .library(name: "HydraFormat", targets: ["HydraFormat"]),
        .library(name: "HydraInstall", targets: ["HydraInstall"]),
        .library(name: "HydraMetal", targets: ["HydraMetal"]),
        .library(name: "HydraReference", targets: ["HydraReference"]),
        .library(name: "HydraTokenize", targets: ["HydraTokenize"]),
        .library(name: "HydraMarkdown", targets: ["HydraMarkdown"]),
        .library(name: "HydraSearch", targets: ["HydraSearch"]),
        .library(name: "HydraVision", targets: ["HydraVision"]),
        .executable(name: "hydra", targets: ["HydraCLI"]),
        .executable(name: "HydraApp", targets: ["HydraApp"]),
    ],
    targets: [
        .target(name: "HydraCore"),
        .target(name: "HydraFormat", dependencies: ["HydraCore"]),
        .target(name: "HydraInstall", dependencies: ["HydraCore", "HydraFormat", "HydraTokenize"]),
        .target(
            name: "HydraMetal", dependencies: ["HydraCore", "HydraFormat"],
            resources: [.copy("Shaders")]),
        // Slow, obviously correct CPU implementations, serving as ground truth for the
        // Metal kernels. No dependency on the platform.
        .target(name: "HydraReference", dependencies: ["HydraCore"]),
        // Tokenizer: portable Swift, no dependency on the platform (D-002).
        // Depends on HydraCore for `ModelArchitecture` alone: the prompt-format seam is
        // keyed by it, and the alternative was a copy of that enum.
        .target(name: "HydraTokenize", dependencies: ["HydraCore"]),
        // Markdown and LaTeX parsing: purely textual, hence testable.
        .target(name: "HydraMarkdown"),
        // Web search: an HTTP client and a token budget, no platform and no tokenizer.
        // It takes no dependency on HydraTokenize on purpose — the budget is enforced through
        // an injected counting function, so the block logic is testable without a checkpoint
        // and the target stays portable under D-002.
        .target(name: "HydraSearch", dependencies: ["HydraCore"]),
        // Images: decoding, resizing and patching, and in time the vision tower itself.
        // Separate from HydraMetal because it is the one part of the runtime that reaches for
        // CoreGraphics, and separate from HydraCore because HydraCore stays free of Apple
        // frameworks so its arithmetic can be tested anywhere.
        .target(name: "HydraVision", dependencies: ["HydraCore", "HydraFormat", "HydraMetal"]),
        .executableTarget(
            name: "HydraCLI",
            dependencies: [
                "HydraCore", "HydraFormat", "HydraInstall", "HydraMetal", "HydraTokenize",
                "HydraMarkdown", "HydraSearch", "HydraVision",
            ]),

        .executableTarget(
            name: "HydraApp",
            dependencies: [
                "HydraCore", "HydraFormat", "HydraInstall", "HydraMetal", "HydraTokenize",
                "HydraMarkdown", "HydraSearch", "HydraVision",
            ]),

        .testTarget(name: "HydraCoreTests", dependencies: ["HydraCore"]),
        // The engine cannot be exercised without a GPU and a checkpoint, but the pure decisions
        // it makes about what to tell the user can be, and those are the ones that were silent.
        .testTarget(
            name: "HydraAppTests",
            dependencies: ["HydraApp", "HydraSearch", "HydraTokenize"]),
        .testTarget(name: "HydraVisionTests", dependencies: ["HydraVision"]),
        .testTarget(
            name: "HydraTokenizeTests",
            dependencies: ["HydraTokenize", "HydraCore", "HydraSearch"]),
        .testTarget(name: "HydraMarkdownTests", dependencies: ["HydraMarkdown"]),
        .testTarget(
            name: "HydraSearchTests", dependencies: ["HydraSearch", "HydraCore"],
            resources: [.copy("Fixtures")]),
        .testTarget(name: "HydraInstallTests", dependencies: ["HydraInstall"]),
        .testTarget(
            name: "HydraMetalTests",
            dependencies: ["HydraMetal", "HydraFormat", "HydraReference", "HydraInstall"]),
        .testTarget(
            name: "HydraReferenceTests", dependencies: ["HydraReference"],
            resources: [.copy("Fixtures")]),
        .testTarget(
            name: "HydraFormatTests",
            dependencies: ["HydraFormat"],
            resources: [.copy("Fixtures")]
        ),
    ]
)
