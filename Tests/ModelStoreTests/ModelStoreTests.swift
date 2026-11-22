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
}
