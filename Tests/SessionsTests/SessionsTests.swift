import AudioCore
import Export
import XCTest
@testable import Sessions

final class SessionsTests: XCTestCase {
    private var root: URL!
    private var cache: URL!

    override func setUpWithError() throws {
        let base = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        root = base.appendingPathComponent("sessions")
        cache = base.appendingPathComponent("cache")
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root.deletingLastPathComponent())
    }

    private func audioFile(_ name: String, seconds: Double = 1) throws -> URL {
        let url = root.deletingLastPathComponent().appendingPathComponent(name)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let w = try MonoFileWriter(url: url, sampleRate: 48_000, format: .flac24)
        try w.append([Float](repeating: 0.1, count: Int(seconds * 48_000)))
        try w.close()
        return url
    }

    func testImportCopiesOriginalByteForByte() throws {
        let source = try audioFile("talk.flac", seconds: 2)
        let store = SessionStore(root: root)
        let manifest = try store.importFile(source, name: "访谈")

        let copy = store.originalURL(for: manifest)
        XCTAssertEqual(try Data(contentsOf: copy), try Data(contentsOf: source))
        XCTAssertTrue(manifest.original.imported)
        XCTAssertEqual(manifest.original.sampleRate, 48_000)
        XCTAssertEqual(manifest.original.durationSeconds, 2, accuracy: 0.01)
        XCTAssertEqual(try store.list().map(\.id), [manifest.id])
    }

    func testSessionDirectoryContainsOnlyOriginalManifestAndOneExport() throws {
        let store = SessionStore(root: root)
        var manifest = try store.adoptRecording(try audioFile("rec.flac"), name: "口播")
        let dir = store.directory(for: manifest.id)

        try Data("m4a".utf8).write(to: store.exportURL(for: manifest, format: .m4a))
        manifest = try store.recordExport(format: .m4a, record: ExportRecord(fileName: "口播.m4a", exportedAt: Date(), modelRevisions: [:], backend: "mlx", processingSeconds: 1), for: manifest)
        try Data("mp3".utf8).write(to: store.exportURL(for: manifest, format: .mp3))
        manifest = try store.recordExport(format: .mp3, record: ExportRecord(fileName: "口播.mp3", exportedAt: Date(), modelRevisions: [:], backend: "mlx", processingSeconds: 1), for: manifest)

        let files = try FileManager.default.contentsOfDirectory(atPath: dir.path).sorted()
        XCTAssertEqual(files, ["original.flac", "session.json", "口播.mp3"])
        XCTAssertEqual(manifest.lastExport?.fileName, "口播.mp3")
        XCTAssertEqual(try store.list().first?.lastExport?.fileName, "口播.mp3")
    }

    func testWorkspacePurgeRemovesDirectory() throws {
        let workspace = Workspace(cacheRoot: cache, sessionID: UUID())
        try FileManager.default.createDirectory(at: workspace.directory, withIntermediateDirectories: true)
        try Data(count: 1000).write(to: workspace.directory.appendingPathComponent("stage.caf"))
        XCTAssertGreaterThan(workspace.diskUsage(), 0)
        try workspace.purge()
        XCTAssertFalse(FileManager.default.fileExists(atPath: workspace.directory.path))
        XCTAssertEqual(workspace.diskUsage(), 0)
    }

    func testPurgeAllRemovesEveryWorkspace() throws {
        for _ in 0..<3 {
            let w = Workspace(cacheRoot: cache, sessionID: UUID())
            try FileManager.default.createDirectory(at: w.directory, withIntermediateDirectories: true)
        }
        try Workspace.purgeAll(cacheRoot: cache)
        let work = cache.appendingPathComponent("work")
        XCTAssertEqual((try? FileManager.default.contentsOfDirectory(atPath: work.path)) ?? [], [])
    }

    func testManifestDecodesWhenNewFieldsMissing() throws {
        let json = """
        {"id":"8F2C7A10-1B2C-4D5E-8F90-123456789ABC","name":"old","createdAt":0,
         "original":{"fileName":"original.flac","sampleRate":48000,"durationSeconds":3,"imported":false}}
        """
        let manifest = try JSONDecoder().decode(SessionManifest.self, from: Data(json.utf8))
        XCTAssertEqual(manifest.processing, ProcessingSettings())
        XCTAssertEqual(manifest.export, ExportOptions())
        XCTAssertNil(manifest.lastExport)
    }

    func testHandEditedTrimIsClampedOrDropped() throws {
        func decode(_ trim: String) throws -> TrimRange? {
            let json = """
            {"id":"8F2C7A10-1B2C-4D5E-8F90-123456789ABC","name":"old","createdAt":0,
             "original":{"fileName":"original.flac","sampleRate":48000,"durationSeconds":3,"imported":false},
             "trim":\(trim)}
            """
            return try JSONDecoder().decode(SessionManifest.self, from: Data(json.utf8)).trim
        }
        XCTAssertEqual(try decode(#"{"start":-5,"end":1e300}"#), TrimRange(start: 0, end: 3))
        XCTAssertEqual(try decode(#"{"start":0.5,"end":2}"#), TrimRange(start: 0.5, end: 2))
        XCTAssertNil(try decode(#"{"start":2,"end":1}"#))
    }

    func testSameNameInSameMinuteGetsDistinctDirectories() throws {
        let store = SessionStore(root: root)
        let a = try store.importFile(try audioFile("a.flac"), name: "same")
        let b = try store.importFile(try audioFile("b.flac"), name: "same")
        XCTAssertNotEqual(store.directory(for: a.id), store.directory(for: b.id))
        XCTAssertEqual(try store.list().count, 2)
    }

    func testRenameUpdatesManifestDirectoryAndExportFile() throws {
        let store = SessionStore(root: root)
        var m = try store.adoptRecording(try audioFile("r.flac"), name: "录音")
        try Data("x".utf8).write(to: store.exportURL(for: m, format: .m4a))
        m = try store.recordExport(format: .m4a, record: ExportRecord(fileName: "录音.m4a", exportedAt: Date(), modelRevisions: [:], backend: "mlx", processingSeconds: 1), for: m)

        let renamed = try store.rename(m.id, to: "周一访谈")
        XCTAssertEqual(renamed.name, "周一访谈")
        XCTAssertEqual(renamed.lastExport?.fileName, "周一访谈.m4a")
        let dir = store.directory(for: m.id)
        XCTAssertTrue(dir.lastPathComponent.hasSuffix("周一访谈"))
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: dir.path).sorted(),
                       ["original.flac", "session.json", "周一访谈.m4a"])
        XCTAssertEqual(try store.list().first?.name, "周一访谈")
    }

    func testExternalExportIsRecordedWithoutTouchingSessionFolder() throws {
        let store = SessionStore(root: root)
        var m = try store.adoptRecording(try audioFile("r.flac"), name: "a")
        try Data("x".utf8).write(to: store.exportURL(for: m, format: .m4a))
        let outside = root.deletingLastPathComponent().appendingPathComponent("exports")
        m = try store.recordExport(format: .mp3, record: ExportRecord(fileName: "a.mp3", exportedAt: Date(), modelRevisions: [:], backend: "mlx", processingSeconds: 1, directory: outside.path), for: m)
        XCTAssertEqual(m.lastExport?.directory, outside.path)
        XCTAssertEqual(store.lastExportURL(for: m), outside.appendingPathComponent("a.mp3"))
        // The session's own export stays: an external export is a copy elsewhere.
        XCTAssertTrue(FileManager.default.fileExists(atPath: store.exportURL(for: m, format: .m4a).path))
    }
}
