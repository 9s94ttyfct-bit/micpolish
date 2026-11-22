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

    public static let all = [reuse, reuseLaughter]

    public static func descriptor(id: String) -> ModelDescriptor? {
        all.first { $0.id == id }
    }
}
