import Foundation

/// A pinned upstream checkpoint: where it comes from, what its bytes must hash
/// to, and which license text ships beside it.
public struct ModelDescriptor: Sendable, Hashable {
    public let id: String
    public let repo: String
    public let revision: String
    public let file: String
    public let sha256: String
    public var byteCount: Int
    /// File name under the module's `licenses` resources.
    public let licenseResource: String
    public var format: Format = .reuseCheckpoint

    public enum Format: Sendable, Hashable {
        /// Upstream RE-USE PyTorch checkpoint, converted to MLX layout locally.
        case reuseCheckpoint
        /// Installed exactly as downloaded (the loader converts at run time).
        case asIs
    }

    /// File name of the installed weights.
    public var installedFileName: String {
        switch format {
        case .reuseCheckpoint: "model_mlx.safetensors"
        case .asIs: (file as NSString).lastPathComponent
        }
    }

    public var sourceURL: URL {
        URL(string: "https://huggingface.co/\(repo)/resolve/\(revision)/\(file)")!
    }
}

public enum ModelCatalog {
    public static let reuse = ModelDescriptor(
        id: "reuse-nemotron",
        repo: "nvidia/NVIDIA-NemotronLabs-RE-USE",
        revision: "31f82f2efad1095034492f236e8dac9239ba713b",
        file: "model.safetensors",
        sha256: "06badaf4029b297f97f7fe48d62990b48d5482608210df577354e0d4e0ed70e1",
        byteCount: 38_592_940,
        licenseResource: "OpenMDW-1.1.txt"
    )

    public static let reuseLaughter = ModelDescriptor(
        id: "reuse-nemotron-laughter",
        repo: "nvidia/NVIDIA-NemotronLabs-RE-USE-laughter",
        revision: "1b1521dd659608e1e063ada83e0acfe1da473a8b",
        file: "model.safetensors",
        sha256: "35c2a4e284515275d2e49a8e8cb5cd392ae67a1f1d345de1c12a0202c7b6b610",
        byteCount: 38_592_940,
        licenseResource: "OpenMDW-1.1.txt"
    )

    /// LavaSR v2 bandwidth extension (Apache-2.0), from a safetensors mirror of
    /// YatharthS/LavaSR `enhancer_v2/pytorch_model.bin`.
    public static let lavaSR = ModelDescriptor(
        id: "lavasr-v2",
        repo: "AEmotionStudio/lavasr-models",
        revision: "291b7810dcd42e111b114fcc6cf07d0d021b9bea",
        file: "enhancer_v2/model.safetensors",
        sha256: "0b2be25ec0773ccdb34a4b2196babc557eba762357ff6138dde7ce2861f12e79",
        byteCount: 56_293_752,
        licenseResource: "Apache-2.0-LavaSR.txt",
        format: .asIs
    )

    /// Inworld Voice Isolator v1, offline checkpoint (Apache-2.0).
    public static let inworld = ModelDescriptor(
        id: "inworld-voice-isolator-v1",
        repo: "inworld/voice-isolator",
        revision: "9b0285b3fe8403aa8e2f1dbb2165a80a4294c970",
        file: "denoiser.safetensors",
        sha256: "647335a6aeb4dc706b895532c209d51e0e84a4849d17b339f00d4392a050ae53",
        byteCount: 256_260_128,
        licenseResource: "Apache-2.0-Inworld.txt",
        format: .asIs
    )

    public static let all = [reuse, reuseLaughter, lavaSR, inworld]

    public static func descriptor(id: String) -> ModelDescriptor? {
        all.first { $0.id == id }
    }
}
