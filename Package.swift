// swift-tools-version: 6.3
import PackageDescription

let package = Package(
    name: "MicPolish",
    defaultLocalization: "zh-Hans",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "AudioCore", targets: ["AudioCore"]),
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
        .testTarget(name: "AudioCoreTests", dependencies: ["AudioCore"]),
    ]
)
