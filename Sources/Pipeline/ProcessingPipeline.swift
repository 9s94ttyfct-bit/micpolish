import AudioCore
import CryptoKit
import Foundation
import MLX
import Sessions

public struct PipelineProgress: Sendable {
    public let stageIndex: Int
    public let stageCount: Int
    public let stageID: String
    public let fraction: Double
    public let elapsed: Duration
}

/// Runs stages in order inside a session workspace. Each stage output is
/// cached as `<stage>-<key>.caf`, where the key chains the upstream key, the
/// stage id and its settings. Output is written to a partial file unique to
/// this run and renamed only on success, so a cancelled or failed stage is
/// never reused and a run still winding down after cancellation cannot
/// delete the partial of a newer run.
public actor ProcessingPipeline {
    private let stages: [any AudioStage]
    private let workspace: Workspace

    public init(stages: [any AudioStage], workspace: Workspace) {
        self.stages = stages
        self.workspace = workspace
    }

    public func run(
        original: URL,
        progress: @Sendable @escaping (PipelineProgress) -> Void
    ) async throws -> StageIO {
        try workspace.create()

        let reader = try MonoFileReader(url: original)
        var current = StageIO(url: original, sampleRate: reader.sampleRate, cacheKey: try Self.originalKey(original))
        let clock = ContinuousClock()
        let start = clock.now
        let stageCount = stages.count

        for (index, stage) in stages.enumerated() {
            if stage.skips(inputSampleRate: current.sampleRate) { continue }
            let key = Self.chainKey(current.cacheKey, stage)
            let final = workspace.directory.appendingPathComponent("\(stage.id)-\(key).caf")
            let next = StageIO(url: final, sampleRate: stage.outputSampleRate(forInput: current.sampleRate), cacheKey: key)

            if !FileManager.default.fileExists(atPath: final.path) {
                let partial = workspace.directory.appendingPathComponent("\(stage.id)-\(key)-\(UUID().uuidString).partial.caf")
                let report: @Sendable (Double) -> Void = { fraction in
                    progress(PipelineProgress(stageIndex: index, stageCount: stageCount, stageID: stage.id,
                                              fraction: fraction, elapsed: clock.now - start))
                }
                report(0)
                do {
                    // Each model's freed buffers go back to the system before the
                    // next model allocates, so peaks of consecutive stages never add up.
                    defer { MLX.Memory.clearCache() }
                    try await stage.run(input: current, output: partial, progress: report)
                    try Task.checkCancellation()
                    try Self.commit(partial, to: final)
                } catch {
                    try? FileManager.default.removeItem(at: partial)
                    throw error
                }
            }
            current = next
        }
        return current
    }

    /// Moves a finished partial into place (an atomic rename). Another run of
    /// the same chain — e.g. one that passed its cancellation check just
    /// before being cancelled — may have put the final there first; same
    /// input, stage and settings give the same output, so that file is kept.
    static func commit(_ partial: URL, to final: URL) throws {
        do {
            try FileManager.default.moveItem(at: partial, to: final)
        } catch where FileManager.default.fileExists(atPath: final.path) {
            try? FileManager.default.removeItem(at: partial)
        }
    }

    static func chainKey(_ upstream: String, _ stage: any AudioStage) -> String {
        let digest = SHA256.hash(data: Data("\(upstream)|\(stage.id)|\(stage.settingsKey)".utf8))
        return digest.prefix(8).map { String(format: "%02x", $0) }.joined()
    }

    /// Identity of the original file: size, modification time and a hash of
    /// its first megabyte.
    static func originalKey(_ url: URL) throws -> String {
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        let size = (attributes[.size] as? NSNumber)?.int64Value ?? 0
        let modified = (attributes[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let head = try handle.read(upToCount: 1 << 20) ?? Data()
        let digest = SHA256.hash(data: head).map { String(format: "%02x", $0) }.joined()
        return "\(size)|\(modified)|\(digest)"
    }
}
