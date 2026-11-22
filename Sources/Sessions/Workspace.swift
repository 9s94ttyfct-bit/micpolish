import Foundation

/// Per-session scratch space for intermediate audio. It exists only while a
/// session is being edited: export success, app quit and app launch all
/// remove it.
public final class Workspace: Sendable {
    public let directory: URL

    public init(cacheRoot: URL, sessionID: UUID) {
        directory = Self.workRoot(cacheRoot).appendingPathComponent(sessionID.uuidString, isDirectory: true)
    }

    public static var defaultCacheRoot: URL {
        FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("MicPolish", isDirectory: true)
    }

    static func workRoot(_ cacheRoot: URL) -> URL {
        cacheRoot.appendingPathComponent("work", isDirectory: true)
    }

    public func create() throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    public func purge() throws {
        if FileManager.default.fileExists(atPath: directory.path) {
            try FileManager.default.removeItem(at: directory)
        }
    }

    public func diskUsage() -> Int64 {
        SessionStore.allocatedSize(of: directory)
    }

    public static func purgeAll(cacheRoot: URL) throws {
        let work = workRoot(cacheRoot)
        if FileManager.default.fileExists(atPath: work.path) {
            try FileManager.default.removeItem(at: work)
        }
    }

    public static func totalDiskUsage(cacheRoot: URL) -> Int64 {
        SessionStore.allocatedSize(of: workRoot(cacheRoot))
    }
}
