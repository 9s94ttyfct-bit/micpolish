import CryptoKit
import XCTest
@testable import ModelStore

/// Serves a local file in place of the network and counts calls.
final class FakeDownloader: Downloader, @unchecked Sendable {
    let source: URL
    private(set) var calls = 0
    init(source: URL) { self.source = source }

    func download(_ url: URL, to destination: URL, progress: @Sendable @escaping (Double) -> Void) async throws {
        calls += 1
        try FileManager.default.copyItem(at: source, to: destination)
        progress(1)
    }
}

/// Like `FakeDownloader`, but yields for a while so a second install can start meanwhile.
final class SlowDownloader: Downloader, @unchecked Sendable {
    let source: URL
    private let lock = NSLock()
    private var count = 0
    var calls: Int { lock.withLock { count } }
    init(source: URL) { self.source = source }

    func download(_ url: URL, to destination: URL, progress: @Sendable @escaping (Double) -> Void) async throws {
        lock.withLock { count += 1 }
        try await Task.sleep(nanoseconds: 200_000_000)
        try FileManager.default.copyItem(at: source, to: destination)
    }
}

final class ModelStoreTests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    private func fixture(bytes: Int) throws -> URL {
        let url = root.deletingLastPathComponent().appendingPathComponent("\(UUID().uuidString).bin")
        try Data(repeating: 7, count: bytes).write(to: url)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    func testChecksumMismatchThrowsAndLeavesNothingInstalled() async throws {
        let source = try fixture(bytes: 1024)
        var descriptor = ModelCatalog.reuse
        descriptor.byteCount = 1024
        let store = ModelStore(root: root, downloader: FakeDownloader(source: source))

        do {
            _ = try await store.install(descriptor) { _ in }
            XCTFail("expected checksum mismatch")
        } catch ModelStoreError.checksumMismatch {
            // expected
        }
        let installed = await store.installedWeightsURL(descriptor)
        XCTAssertNil(installed)
        let staging = root.appendingPathComponent(".staging")
        let leftovers = (try? FileManager.default.contentsOfDirectory(atPath: staging.path)) ?? []
        XCTAssertEqual(leftovers, [])
    }

    func testSizeMismatchThrows() async throws {
        let source = try fixture(bytes: 10)
        let store = ModelStore(root: root, downloader: FakeDownloader(source: source))
        do {
            _ = try await store.install(ModelCatalog.reuse) { _ in }
            XCTFail("expected size mismatch")
        } catch ModelStoreError.sizeMismatch(let expected, let actual) {
            XCTAssertEqual(expected, 38_592_940)
            XCTAssertEqual(actual, 10)
        }
    }

    func testInstallConvertsAndIsIdempotent() async throws {
        guard let path = ProcessInfo.processInfo.environment["MICPOLISH_REUSE_SOURCE"] else {
            throw XCTSkip("Set MICPOLISH_REUSE_SOURCE to the NemotronLabs model.safetensors")
        }
        let downloader = FakeDownloader(source: URL(fileURLWithPath: path))
        let store = ModelStore(root: root, downloader: downloader)

        let url = try await store.install(ModelCatalog.reuse) { _ in }
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))
        XCTAssertEqual(url.lastPathComponent, "model_mlx.safetensors")
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: url.deletingLastPathComponent().appendingPathComponent("LICENSE.txt").path))

        let again = try await store.install(ModelCatalog.reuse) { _ in }
        XCTAssertEqual(again, url)
        XCTAssertEqual(downloader.calls, 1)
        let usage = await store.diskUsageBytes()
        XCTAssertGreaterThan(usage, 30_000_000)
    }

    func testRemoveDeletesFiles() async throws {
        let descriptor = ModelCatalog.reuse
        let dir = root.appendingPathComponent(descriptor.id).appendingPathComponent(descriptor.revision)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data([1]).write(to: dir.appendingPathComponent("model_mlx.safetensors"))
        let store = ModelStore(root: root, downloader: FakeDownloader(source: dir))

        let before = await store.installedWeightsURL(descriptor)
        XCTAssertNotNil(before)
        try await store.remove(descriptor)
        let after = await store.installedWeightsURL(descriptor)
        XCTAssertNil(after)
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent(descriptor.id).path))
    }

    func testCatalogPinsNemotronRevisions() {
        XCTAssertEqual(ModelCatalog.reuse.sourceURL.absoluteString,
                       "https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE/resolve/31f82f2efad1095034492f236e8dac9239ba713b/model.safetensors")
        XCTAssertEqual(ModelCatalog.reuseLaughter.sha256,
                       "35c2a4e284515275d2e49a8e8cb5cd392ae67a1f1d345de1c12a0202c7b6b610")
    }

    func testAsIsModelIsInstalledByteForByte() async throws {
        let source = try fixture(bytes: 4096)
        let digest = SHA256.hash(data: try Data(contentsOf: source)).map { String(format: "%02x", $0) }.joined()
        let descriptor = ModelDescriptor(id: "test-asis", repo: "owner/repo", revision: "abc", file: "sub/model.safetensors",
                                         sha256: digest, byteCount: 4096, licenseResource: "Apache-2.0-LavaSR.txt",
                                         format: .asIs)
        let store = ModelStore(root: root, downloader: FakeDownloader(source: source))
        let url = try await store.install(descriptor) { _ in }
        XCTAssertEqual(url.lastPathComponent, "model.safetensors")
        XCTAssertEqual(try Data(contentsOf: url), try Data(contentsOf: source))
        let installed = await store.installedWeightsURL(descriptor)
        XCTAssertEqual(installed, url)
    }

    func testConcurrentInstallsShareOneDownload() async throws {
        let source = try fixture(bytes: 4096)
        let digest = SHA256.hash(data: try Data(contentsOf: source)).map { String(format: "%02x", $0) }.joined()
        let descriptor = ModelDescriptor(id: "test-asis", repo: "owner/repo", revision: "abc", file: "model.safetensors",
                                         sha256: digest, byteCount: 4096, licenseResource: "Apache-2.0-LavaSR.txt",
                                         format: .asIs)
        let downloader = SlowDownloader(source: source)
        let store = ModelStore(root: root, downloader: downloader)
        async let a = store.install(descriptor) { _ in }
        async let b = store.install(descriptor) { _ in }
        let (first, second) = try await (a, b)
        XCTAssertEqual(first, second)
        XCTAssertEqual(downloader.calls, 1)
        XCTAssertEqual(try Data(contentsOf: first), try Data(contentsOf: source))
    }

    func testStaleStagingIsRemovedWhenAnInstallStarts() async throws {
        let source = try fixture(bytes: 4096)
        let digest = SHA256.hash(data: try Data(contentsOf: source)).map { String(format: "%02x", $0) }.joined()
        let descriptor = ModelDescriptor(id: "test-asis", repo: "owner/repo", revision: "abc", file: "model.safetensors",
                                         sha256: digest, byteCount: 4096, licenseResource: "Apache-2.0-LavaSR.txt",
                                         format: .asIs)
        let stale = root.appendingPathComponent(".staging/left-over", isDirectory: true)
        let fresh = root.appendingPathComponent(".staging/other-process", isDirectory: true)
        for dir in [stale, fresh] { try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true) }
        try FileManager.default.setAttributes([.modificationDate: Date(timeIntervalSinceNow: -2 * ModelStore.staleStagingAge)],
                                              ofItemAtPath: stale.path)
        _ = try await ModelStore(root: root, downloader: FakeDownloader(source: source)).install(descriptor) { _ in }
        XCTAssertFalse(FileManager.default.fileExists(atPath: stale.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: fresh.path))
    }

    func testHTTPErrorsNameTheStatus() {
        let error = ModelStoreError.httpStatus(404, URL(string: "https://example.com/m/model.safetensors")!)
        XCTAssertEqual(error.localizedDescription,
                       "Downloading model.safetensors failed: the server answered HTTP 404 (not found).")
    }

    func testCatalogPinsLavaSRMirror() {
        XCTAssertEqual(ModelCatalog.lavaSR.sourceURL.absoluteString,
                       "https://huggingface.co/AEmotionStudio/lavasr-models/resolve/291b7810dcd42e111b114fcc6cf07d0d021b9bea/enhancer_v2/model.safetensors")
        XCTAssertEqual(ModelCatalog.lavaSR.format, .asIs)
    }
}
