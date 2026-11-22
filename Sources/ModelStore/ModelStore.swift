import CryptoKit
import Foundation
import ReuseMLX

public enum ModelStoreError: Error, Equatable {
    case checksumMismatch(expected: String, actual: String)
    case sizeMismatch(expected: Int, actual: Int)
    case missingLicense(String)
}

/// Installs pinned checkpoints under `<root>/<id>/<revision>/` as
/// `model_mlx.safetensors` plus `LICENSE.txt`. Work happens in
/// `<root>/.staging/` and is moved into place only after verification and
/// conversion succeed, so an interrupted install never looks installed.
public actor ModelStore {
    public static let weightsFileName = "model_mlx.safetensors"

    private let root: URL
    private let downloader: any Downloader
    private let fileManager = FileManager.default

    public init(root: URL, downloader: any Downloader = URLSessionDownloader()) {
        self.root = root
        self.downloader = downloader
    }

    public static var defaultRoot: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("MicPolish/models", isDirectory: true)
    }

    private func installDirectory(_ d: ModelDescriptor) -> URL {
        root.appendingPathComponent(d.id, isDirectory: true)
            .appendingPathComponent(d.revision, isDirectory: true)
    }

    public func installedWeightsURL(_ d: ModelDescriptor) -> URL? {
        let url = installDirectory(d).appendingPathComponent(Self.weightsFileName)
        return fileManager.fileExists(atPath: url.path) ? url : nil
    }

    public func install(
        _ d: ModelDescriptor,
        progress: @Sendable @escaping (Double) -> Void
    ) async throws -> URL {
        if let existing = installedWeightsURL(d) { return existing }

        let stagingRoot = root.appendingPathComponent(".staging", isDirectory: true)
        let staging = stagingRoot.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try fileManager.createDirectory(at: staging, withIntermediateDirectories: true)
        defer { try? fileManager.removeItem(at: staging) }

        let source = staging.appendingPathComponent(d.file)
        try await downloader.download(d.sourceURL, to: source, progress: progress)

        let size = (try fileManager.attributesOfItem(atPath: source.path)[.size] as? NSNumber)?.intValue ?? -1
        guard size == d.byteCount else {
            throw ModelStoreError.sizeMismatch(expected: d.byteCount, actual: size)
        }
        let digest = try Self.sha256(of: source)
        guard digest == d.sha256 else {
            throw ModelStoreError.checksumMismatch(expected: d.sha256, actual: digest)
        }

        let built = staging.appendingPathComponent("install", isDirectory: true)
        try ReuseWeightConverter.convert(input: source, output: built.appendingPathComponent(Self.weightsFileName))
        guard let license = Bundle.module.url(forResource: d.licenseResource, withExtension: nil, subdirectory: "licenses") else {
            throw ModelStoreError.missingLicense(d.licenseResource)
        }
        try fileManager.copyItem(at: license, to: built.appendingPathComponent("LICENSE.txt"))

        let destination = installDirectory(d)
        try fileManager.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        if fileManager.fileExists(atPath: destination.path) {
            try fileManager.removeItem(at: destination)
        }
        try fileManager.moveItem(at: built, to: destination)
        return destination.appendingPathComponent(Self.weightsFileName)
    }

    public func remove(_ d: ModelDescriptor) throws {
        let modelRoot = root.appendingPathComponent(d.id, isDirectory: true)
        if fileManager.fileExists(atPath: modelRoot.path) {
            try fileManager.removeItem(at: modelRoot)
        }
    }

    public func diskUsageBytes() -> Int64 {
        guard let enumerator = fileManager.enumerator(
            at: root, includingPropertiesForKeys: [.totalFileAllocatedSizeKey]
        ) else { return 0 }
        var total: Int64 = 0
        for case let url as URL in enumerator {
            total += Int64((try? url.resourceValues(forKeys: [.totalFileAllocatedSizeKey]).totalFileAllocatedSize) ?? 0)
        }
        return total
    }

    static func sha256(of url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let chunk = try handle.read(upToCount: 1 << 20), !chunk.isEmpty {
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
