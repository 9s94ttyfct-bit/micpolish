// swift-tools-version: 6.3
import PackageDescription

let package = Package(
    name: "MicPolish",
    defaultLocalization: "zh-Hans",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "AudioCore", targets: ["AudioCore"]),
        .library(name: "ReuseMLX", targets: ["ReuseMLX"]),
        .library(name: "ModelStore", targets: ["ModelStore"]),
        .library(name: "Mastering", targets: ["Mastering"]),
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
            name: "ModelStore",
            dependencies: ["ReuseMLX"],
            resources: [.copy("Resources/licenses")]
        ),
        .target(name: "Mastering", dependencies: ["AudioCore"]),
        .testTarget(name: "AudioCoreTests", dependencies: ["AudioCore"]),
        .testTarget(name: "ReuseMLXTests", dependencies: ["ReuseMLX"]),
        .testTarget(name: "ModelStoreTests", dependencies: ["ModelStore"]),
        .testTarget(name: "MasteringTests", dependencies: ["Mastering", "AudioCore"]),
    ]
)
