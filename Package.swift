// swift-tools-version: 6.3
import Foundation
import PackageDescription

/// LAME is built into vendor/lame by tools/build_lame.sh and linked dynamically.
let lameLib = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("vendor/lame/lib").path

let package = Package(
    name: "MicPolish",
    defaultLocalization: "zh-Hans",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "AudioCore", targets: ["AudioCore"]),
        .library(name: "ReuseMLX", targets: ["ReuseMLX"]),
        .library(name: "LavaMLX", targets: ["LavaMLX"]),
        .library(name: "ModelStore", targets: ["ModelStore"]),
        .library(name: "Mastering", targets: ["Mastering"]),
        .library(name: "Export", targets: ["Export"]),
        .library(name: "Sessions", targets: ["Sessions"]),
        .library(name: "Pipeline", targets: ["Pipeline"]),
        .library(name: "Recording", targets: ["Recording"]),
        .library(name: "Preview", targets: ["Preview"]),
        .executable(name: "micpolish", targets: ["micpolish"]),
        .executable(name: "MicPolishApp", targets: ["MicPolishApp"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.31.6"),
        .package(url: "https://github.com/apple/swift-argument-parser", exact: "1.8.2"),
    ],
    targets: [
        .target(
            name: "AudioCore",
            linkerSettings: [
                .linkedFramework("AVFoundation"),
                .linkedFramework("AudioToolbox"),
                .linkedFramework("Accelerate"),
            ]
        ),
        .target(
            name: "ReuseMLX",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "LavaMLX",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "ModelStore",
            dependencies: ["ReuseMLX"],
            resources: [.copy("Resources/licenses")]
        ),
        .target(name: "Mastering", dependencies: ["AudioCore"]),
        .systemLibrary(name: "CLame"),
        .target(
            name: "Export",
            dependencies: ["AudioCore", "CLame"],
            linkerSettings: [
                .linkedFramework("AVFoundation"),
                .unsafeFlags(["-L\(lameLib)", "-Xlinker", "-rpath", "-Xlinker", lameLib]),
            ]
        ),
        .target(name: "Sessions", dependencies: ["AudioCore", "Mastering", "Export"]),
        .target(name: "Pipeline", dependencies: ["AudioCore", "ReuseMLX", "LavaMLX", "Sessions"]),
        .target(
            name: "Recording",
            dependencies: ["AudioCore", "Sessions"],
            linkerSettings: [.linkedFramework("AVFoundation"), .linkedFramework("CoreAudio")]
        ),
        .target(
            name: "Preview",
            dependencies: ["AudioCore", "Mastering"],
            linkerSettings: [.linkedFramework("AVFoundation")]
        ),
        .executableTarget(
            name: "micpolish",
            dependencies: [
                "AudioCore", "ReuseMLX", "LavaMLX", "ModelStore", "Mastering", "Export", "Sessions", "Pipeline",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        .executableTarget(
            name: "MicPolishApp",
            dependencies: [
                "AudioCore", "ReuseMLX", "LavaMLX", "ModelStore", "Mastering", "Export", "Sessions", "Pipeline",
                "Recording", "Preview",
            ],
            // Compiled into the app bundle's lproj folders by tools/package-app.sh.
            exclude: ["Localizable.xcstrings"]
        ),
        .testTarget(name: "AudioCoreTests", dependencies: ["AudioCore"]),
        .testTarget(name: "ReuseMLXTests", dependencies: ["ReuseMLX"]),
        .testTarget(name: "ModelStoreTests", dependencies: ["ModelStore"]),
        .testTarget(name: "LavaMLXTests", dependencies: ["LavaMLX", "AudioCore"]),
        .testTarget(name: "MasteringTests", dependencies: ["Mastering", "AudioCore"]),
        .testTarget(name: "ExportTests", dependencies: ["Export", "AudioCore"]),
        .testTarget(name: "SessionsTests", dependencies: ["Sessions", "AudioCore", "Export"]),
        .testTarget(name: "RecordingTests", dependencies: ["Recording", "AudioCore", "Sessions"]),
        .testTarget(name: "PreviewTests", dependencies: ["Preview", "AudioCore", "Mastering"]),
        .testTarget(
            name: "PipelineTests",
            dependencies: ["Pipeline", "AudioCore", "ReuseMLX", "Sessions", .product(name: "MLX", package: "mlx-swift")]
        ),
    ]
)
