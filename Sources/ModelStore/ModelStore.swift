import CryptoKit
import Foundation
import ReuseMLX

public enum ModelStoreError: Error, Equatable, LocalizedError {
    case checksumMismatch(expected: String, actual: String)
    case sizeMismatch(expected: Int, actual: Int)
    case missingLicense(String)
    /// The server answered with a non-2xx status.
    case httpStatus(Int, URL)

    public var errorDescription: String? {
        switch self {
        case .checksumMismatch(let expected, let actual):
            "The downloaded model is corrupt: SHA-256 \(actual.prefix(12))… instead of \(expected.prefix(12))…. Try again."
        case .sizeMismatch(let expected, let actual):
            "The model download is incomplete or wrong: \(actual) bytes instead of \(expected). Check the network (a proxy or login page can cause this) and try again."
        case .missingLicense(let name):
            "The app is missing the license file \(name); reinstall MicPolish."
        case .httpStatus(let status, let url):
            "Downloading \(url.lastPathComponent) failed: the server answered HTTP \(status)"
                + (status == 404 ? " (not found)." : status == 401 || status == 403 ? " (access denied)." : status == 429 ? " (too many requests; try later)." : ".")
        }
    }
}

/// Installs pinned checkpoints under `<root>/<id>/<revision>/` as
/// `model_mlx.safetensors` plus `LICENSE.txt`. Work happens in
/// `<root>/.staging/` and is moved into place only after verification and
/// conversion succeed, so an interrupted install never looks installed.
public actor ModelStore {

    private let root: URL
    private let downloader: any Downloader
    private let fileManager = FileManager.default
    /// Installs in progress, by model id and revision, so concurrent callers
    /// share one download; `waiters` are the callers still interested in it.
    private var inFlight: [String: (task: Task<URL, Error>, waiters: Set<UUID>)] = [:]
    /// Staging folders older than this are leftovers of interrupted installs.
    static let staleStagingAge: TimeInterval = 6 * 3600

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

    /// The license text shipped with the app for `d` (also copied next to
    /// installed weights as `LICENSE.txt`).
    public nonisolated static func licenseURL(for d: ModelDescriptor) -> URL? {
        Bundle.module.url(forResource: d.licenseResource, withExtension: nil, subdirectory: "licenses")
    }

    public func installedWeightsURL(_ d: ModelDescriptor) -> URL? {
        let url = installDirectory(d).appendingPathComponent(d.installedFileName)
        return fileManager.fileExists(atPath: url.path) ? url : nil
    }

    /// Downloads, verifies and installs `d` unless it is installed already.
    /// Concurrent calls for the same model share one install; only the first
    /// caller receives progress. The shared install is cancelled only when
    /// every caller waiting for it has been cancelled.
    public func install(
        _ d: ModelDescriptor,
        progress: @Sendable @escaping (Double) -> Void
    ) async throws -> URL {
        if let existing = installedWeightsURL(d) { return existing }
        let key = "\(d.id)@\(d.revision)"
        let task: Task<URL, Error>
        if let shared = inFlight[key] {
            task = shared.task
        } else {
            task = Task { try await self.performInstall(d, progress: progress) }
            inFlight[key] = (task, [])
        }
        let waiter = UUID()
        inFlight[key]?.waiters.insert(waiter)
        defer { leave(key, waiter, of: task, cancelIfLast: false) }
        return try await withTaskCancellationHandler { try await task.value } onCancel: {
            Task { await self.leave(key, waiter, of: task, cancelIfLast: true) }
        }
    }

    /// One caller stops waiting. When none are left the entry goes; if the
    /// last one left because it was cancelled, so does the install.
    private func leave(_ key: String, _ waiter: UUID, of task: Task<URL, Error>, cancelIfLast: Bool) {
        guard inFlight[key]?.task == task, inFlight[key]?.waiters.remove(waiter) != nil,
              inFlight[key]?.waiters.isEmpty == true else { return }
        inFlight[key] = nil
        if cancelIfLast { task.cancel() }
    }

    private func performInstall(
        _ d: ModelDescriptor,
        progress: @Sendable @escaping (Double) -> Void
    ) async throws -> URL {
        let stagingRoot = root.appendingPathComponent(".staging", isDirectory: true)
        removeStaleStaging(in: stagingRoot)
        let staging = stagingRoot.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try fileManager.createDirectory(at: staging, withIntermediateDirectories: true)
        defer { try? fileManager.removeItem(at: staging) }

        let source = staging.appendingPathComponent((d.file as NSString).lastPathComponent)
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
        try FileManager.default.createDirectory(at: built, withIntermediateDirectories: true)
        switch d.format {
        case .reuseCheckpoint:
            try ReuseWeightConverter.convert(input: source, output: built.appendingPathComponent(d.installedFileName))
        case .asIs:
            try fileManager.moveItem(at: source, to: built.appendingPathComponent(d.installedFileName))
        }
        guard let license = Self.licenseURL(for: d) else {
            throw ModelStoreError.missingLicense(d.licenseResource)
        }
        try fileManager.copyItem(at: license, to: built.appendingPathComponent("LICENSE.txt"))

        let destination = installDirectory(d)
        try fileManager.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        if fileManager.fileExists(atPath: destination.path) {
            try fileManager.removeItem(at: destination)
        }
        try fileManager.moveItem(at: built, to: destination)
        return destination.appendingPathComponent(d.installedFileName)
    }

    /// Deletes staging folders left by installs that were interrupted (app quit,
    /// Ctrl-C). Recent ones may belong to an install running in another process.
    private func removeStaleStaging(in stagingRoot: URL) {
        let entries = (try? fileManager.contentsOfDirectory(
            at: stagingRoot, includingPropertiesForKeys: [.contentModificationDateKey])) ?? []
        for entry in entries {
            let modified = (try? entry.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            if Date().timeIntervalSince(modified) > Self.staleStagingAge { try? fileManager.removeItem(at: entry) }
        }
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
