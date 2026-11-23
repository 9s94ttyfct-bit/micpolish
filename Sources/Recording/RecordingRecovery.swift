import AudioCore
import Foundation
import Sessions

public enum RecordingRecoveryError: Error, LocalizedError {
    /// The FLAC holds fewer (or more) frames than the capture it came from.
    case incompleteConversion(expected: Int64, written: Int64)

    public var errorDescription: String? {
        switch self {
        case .incompleteConversion(let expected, let written):
            "The recording could not be converted completely (\(written) of \(expected) frames)."
        }
    }
}

public enum RecordingRecovery {
    /// Turns recordings left in `<cacheRoot>/recording` by a crash or power
    /// loss into sessions named "`namePrefix` <date>". A leftover is deleted
    /// only after its session exists, or when it holds no readable audio; if
    /// the session cannot be created (for example the sessions folder is on a
    /// disconnected drive) the CAF stays for the next launch.
    public static func recover(cacheRoot: URL, store: SessionStore, namePrefix: String) throws -> [SessionManifest] {
        let directory = cacheRoot.appendingPathComponent("recording", isDirectory: true)
        guard FileManager.default.fileExists(atPath: directory.path) else { return [] }
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd HH:mm"
        var recovered: [SessionManifest] = []
        for caf in try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: [.contentModificationDateKey])
        where caf.pathExtension == "caf" {
            guard let frames = try? MonoFileReader(url: caf).frameCount, frames > 0 else {
                try? FileManager.default.removeItem(at: caf)
                continue
            }
            let date = (try? caf.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? Date()
            if let manifest = try? adopt(caf: caf, name: "\(namePrefix) \(formatter.string(from: date))", store: store) {
                recovered.append(manifest)
            }
        }
        return recovered
    }

    /// Converts a capture CAF to FLAC 24-bit, creates the session and removes
    /// the CAF. On failure the CAF is kept and any partial FLAC removed.
    static func adopt(caf: URL, name: String, store: SessionStore) throws -> SessionManifest {
        try adopt(caf: caf, name: name, store: store, convert: convertToFLAC)
    }

    /// `adopt` with the conversion step replaceable, so tests can produce a
    /// short FLAC the way a failed final flush would.
    static func adopt(caf: URL, name: String, store: SessionStore,
                      convert: (_ caf: MonoFileReader, _ flac: URL) throws -> Void) throws -> SessionManifest {
        let flac = caf.deletingPathExtension().appendingPathExtension("flac")
        do {
            let reader = try MonoFileReader(url: caf)
            try convert(reader, flac)
            // `MonoFileWriter.close()` cannot report a failed final flush, and
            // the CAF is the only complete copy: delete it only once the FLAC
            // demonstrably holds every frame.
            let written = try MonoFileReader(url: flac).frameCount
            guard written == reader.frameCount else {
                throw RecordingRecoveryError.incompleteConversion(expected: reader.frameCount, written: written)
            }
            let manifest = try store.adoptRecording(flac, name: name)
            try? FileManager.default.removeItem(at: caf)
            return manifest
        } catch {
            try? FileManager.default.removeItem(at: flac)
            throw error
        }
    }

    private static func convertToFLAC(_ reader: MonoFileReader, to flac: URL) throws {
        let writer = try MonoFileWriter(url: flac, sampleRate: reader.sampleRate, format: .flac24)
        while let block = try reader.read(maxFrames: 65_536) {
            try writer.append(block)
        }
        try writer.close()
    }
}
