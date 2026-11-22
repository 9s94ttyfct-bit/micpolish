import AudioCore
import Export
import Foundation

/// Sessions live in `<root>/<yyyy-MM-dd HHmm name>/` and hold only the
/// original audio, `session.json` and at most one export.
public final class SessionStore: @unchecked Sendable {
    public static let manifestName = "session.json"

    public let root: URL
    private let fileManager = FileManager.default
    private let lock = NSLock()

    public init(root: URL) {
        self.root = root
    }

    public static var defaultRoot: URL {
        FileManager.default.urls(for: .musicDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("MicPolish", isDirectory: true)
    }

    // MARK: Listing

    public func list() throws -> [SessionManifest] {
        guard fileManager.fileExists(atPath: root.path) else { return [] }
        return try fileManager.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)
            .compactMap { try? readManifest(in: $0) }
            .sorted { $0.createdAt > $1.createdAt }
    }

    public func directory(for id: UUID) -> URL {
        let dirs = (try? fileManager.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)) ?? []
        return dirs.first { (try? readManifest(in: $0))?.id == id }
            ?? root.appendingPathComponent(id.uuidString, isDirectory: true)
    }

    public func originalURL(for m: SessionManifest) -> URL {
        directory(for: m.id).appendingPathComponent(m.original.fileName)
    }

    public func exportURL(for m: SessionManifest, format: ExportFormat) -> URL {
        directory(for: m.id).appendingPathComponent("\(Self.safeFileName(m.name)).\(format.rawValue)")
    }

    // MARK: Creating

    /// Copies `source` unchanged into a new session as `original.<ext>`.
    public func importFile(_ source: URL, name: String) throws -> SessionManifest {
        try create(name: name, imported: true) { dir in
            let dest = dir.appendingPathComponent("original.\(source.pathExtension.lowercased())")
            try fileManager.copyItem(at: source, to: dest)
            return dest
        }
    }

    /// Moves a finished FLAC recording into a new session as `original.flac`.
    public func adoptRecording(_ flac: URL, name: String) throws -> SessionManifest {
        try create(name: name, imported: false) { dir in
            let dest = dir.appendingPathComponent("original.flac")
            try fileManager.moveItem(at: flac, to: dest)
            return dest
        }
    }

    private func create(name: String, imported: Bool, place: (URL) throws -> URL) throws -> SessionManifest {
        lock.lock()
        defer { lock.unlock() }
        let now = Date()
        let dir = uniqueDirectory(name: name, date: now)
        try fileManager.createDirectory(at: dir, withIntermediateDirectories: true)
        do {
            let original = try place(dir)
            let reader = try MonoFileReader(url: original)
            let manifest = SessionManifest(
                id: UUID(), name: name, createdAt: now,
                original: OriginalAudio(
                    fileName: original.lastPathComponent,
                    sampleRate: reader.sampleRate,
                    durationSeconds: Double(reader.frameCount) / Double(reader.sampleRate),
                    imported: imported))
            try write(manifest, in: dir)
            return manifest
        } catch {
            try? fileManager.removeItem(at: dir)
            throw error
        }
    }

    private func uniqueDirectory(name: String, date: Date) -> URL {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy-MM-dd HHmm"
        let base = "\(formatter.string(from: date)) \(Self.safeFileName(name))"
        var candidate = root.appendingPathComponent(base, isDirectory: true)
        var n = 2
        while fileManager.fileExists(atPath: candidate.path) {
            candidate = root.appendingPathComponent("\(base) (\(n))", isDirectory: true)
            n += 1
        }
        return candidate
    }

    // MARK: Updating

    public func save(_ m: SessionManifest) throws {
        try write(m, in: directory(for: m.id))
    }

    /// Records a finished export and deletes any export in the other format,
    /// so a session keeps at most one export file.
    public func recordExport(format: ExportFormat, record: ExportRecord, for m: SessionManifest) throws -> SessionManifest {
        var updated = m
        updated.export.format = format
        updated.lastExport = record
        for other in ExportFormat.allCases where other != format {
            try? fileManager.removeItem(at: exportURL(for: m, format: other))
        }
        try save(updated)
        return updated
    }

    public func delete(_ id: UUID) throws {
        try fileManager.removeItem(at: directory(for: id))
    }

    public func diskUsage(for id: UUID) -> Int64 {
        Self.allocatedSize(of: directory(for: id))
    }

    // MARK: Helpers

    private func readManifest(in dir: URL) throws -> SessionManifest {
        let data = try Data(contentsOf: dir.appendingPathComponent(Self.manifestName))
        return try Self.decoder.decode(SessionManifest.self, from: data)
    }

    private func write(_ m: SessionManifest, in dir: URL) throws {
        try Self.encoder.encode(m).write(to: dir.appendingPathComponent(Self.manifestName), options: .atomic)
    }

    static func safeFileName(_ name: String) -> String {
        let cleaned = name.replacingOccurrences(of: "/", with: "-").replacingOccurrences(of: ":", with: "-")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return cleaned.isEmpty ? "未命名" : cleaned
    }

    static func allocatedSize(of url: URL) -> Int64 {
        guard let e = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.totalFileAllocatedSizeKey]) else { return 0 }
        var total: Int64 = 0
        for case let file as URL in e {
            total += Int64((try? file.resourceValues(forKeys: [.totalFileAllocatedSizeKey]).totalFileAllocatedSize) ?? 0)
        }
        return total
    }

    private static let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.outputFormatting = [.prettyPrinted, .sortedKeys]
        e.dateEncodingStrategy = .iso8601
        return e
    }()

    private static let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.dateDecodingStrategy = .iso8601
        return d
    }()
}
