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

/// Every MicPolish process (app or CLI) holds a shared lock on
/// `<cacheRoot>/instances.lock` while it runs. Work shared through the cache
/// (every session workspace, unfinished recordings) is cleaned up only inside
/// `whileAlone`, which runs when no other process holds the lock — so one
/// process never deletes another's intermediates or a recording in progress.
/// The kernel drops the lock when a process exits, crashed or not.
public final class CacheLock: @unchecked Sendable {
    private let fd: Int32

    public init(cacheRoot: URL) throws {
        try FileManager.default.createDirectory(at: cacheRoot, withIntermediateDirectories: true)
        let path = cacheRoot.appendingPathComponent("instances.lock").path
        fd = open(path, O_RDWR | O_CREAT | O_CLOEXEC, 0o644)
        guard fd >= 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        flock(fd, LOCK_SH)
    }

    deinit { close(fd) }

    /// Runs `body` with the lock held exclusively if no other process holds
    /// it; returns whether `body` ran. The shared lock is held again afterwards.
    @discardableResult
    public func whileAlone(_ body: () throws -> Void) rethrows -> Bool {
        // Converting shared → exclusive may drop the shared lock even when it
        // fails, so it is re-taken on both paths.
        defer { flock(fd, LOCK_SH) }
        guard flock(fd, LOCK_EX | LOCK_NB) == 0 else { return false }
        try body()
        return true
    }
}
