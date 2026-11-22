import AudioCore
import Foundation
import Sessions

public enum RecordingRecovery {
    /// Turns recordings left in `<cacheRoot>/recording` by a crash or power
    /// loss into sessions, then deletes the leftover CAF files.
    public static func recover(cacheRoot: URL, store: SessionStore) throws -> [SessionManifest] {
        let directory = cacheRoot.appendingPathComponent("recording", isDirectory: true)
        guard FileManager.default.fileExists(atPath: directory.path) else { return [] }
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd HH:mm"
        var recovered: [SessionManifest] = []
        for caf in try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: [.contentModificationDateKey])
        where caf.pathExtension == "caf" {
            let date = (try? caf.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? Date()
            if let manifest = try? adopt(caf: caf, name: "恢复的录音 \(formatter.string(from: date))", store: store) {
                recovered.append(manifest)
            } else {
                try? FileManager.default.removeItem(at: caf)
            }
        }
        return recovered
    }

    /// Converts a capture CAF to FLAC 24-bit, creates the session and removes
    /// the CAF.
    static func adopt(caf: URL, name: String, store: SessionStore) throws -> SessionManifest {
        let reader = try MonoFileReader(url: caf)
        let flac = caf.deletingPathExtension().appendingPathExtension("flac")
        let writer = try MonoFileWriter(url: flac, sampleRate: reader.sampleRate, format: .flac24)
        while let block = try reader.read(maxFrames: 65_536) {
            try writer.append(block)
        }
        try writer.close()
        let manifest = try store.adoptRecording(flac, name: name)
        try? FileManager.default.removeItem(at: caf)
        return manifest
    }
}
