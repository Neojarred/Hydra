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
        .target(name: "HydraTokenize"),
        // Markdown and LaTeX parsing: purely textual, hence testable.
        .target(name: "HydraMarkdown"),
        .executableTarget(
            name: "HydraCLI",
            dependencies: [
                "HydraCore", "HydraFormat", "HydraInstall", "HydraMetal", "HydraTokenize",
                "HydraMarkdown",
            ]),

        .executableTarget(
            name: "HydraApp",
            dependencies: [
                "HydraCore", "HydraFormat", "HydraInstall", "HydraMetal", "HydraTokenize",
                "HydraMarkdown",
            ]),

        .testTarget(name: "HydraCoreTests", dependencies: ["HydraCore"]),
        .testTarget(name: "HydraTokenizeTests", dependencies: ["HydraTokenize"]),
        .testTarget(name: "HydraMarkdownTests", dependencies: ["HydraMarkdown"]),
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
