import AudioCore
import Foundation
import MLX
import ReuseMLX
import Sessions
import XCTest
@testable import Pipeline

/// Multiplies by a gain; counts runs. Optionally blocks until cancelled.
final class FakeGainStage: AudioStage, @unchecked Sendable {
    let id: String
    var settingsKey: String
    let gain: Float
    let blockUntilCancelled: Bool
    private(set) var runs = 0

    init(id: String, settingsKey: String = "a", gain: Float = 0.5, blockUntilCancelled: Bool = false) {
        self.id = id
        self.settingsKey = settingsKey
        self.gain = gain
        self.blockUntilCancelled = blockUntilCancelled
    }

    func outputSampleRate(forInput sampleRate: Int) -> Int { sampleRate }

    func run(input: StageIO, output: URL, progress: @Sendable (Double) -> Void) async throws {
        runs += 1
        let reader = try MonoFileReader(url: input.url)
        let writer = try MonoFileWriter(url: output, sampleRate: input.sampleRate, format: .cafFloat32)
        while let block = try reader.read(maxFrames: 4096) {
            try writer.append(block.map { $0 * gain })
            if blockUntilCancelled {
                while true { try await Task.sleep(nanoseconds: 10_000_000) }
            }
        }
        try writer.close()
    }
}

final class PipelineTests: XCTestCase {
    private var dir: URL!
    private var workspace: Workspace!

    override func setUpWithError() throws {
        dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        workspace = Workspace(cacheRoot: dir.appendingPathComponent("cache"), sessionID: UUID())
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: dir)
    }

    private func original(seconds: Double, sampleRate: Int = 48_000) throws -> URL {
        let url = dir.appendingPathComponent("original-\(sampleRate).flac")
        let w = try MonoFileWriter(url: url, sampleRate: sampleRate, format: .flac24)
        var frames = Int(seconds * Double(sampleRate))
        var t = 0
        while frames > 0 {
            let n = min(frames, 48_000)
            try w.append((0..<n).map { i in Float(0.2 * sin(2 * Double.pi * 300 * Double(t + i) / Double(sampleRate))) })
            t += n
            frames -= n
        }
        try w.close()
        return url
    }

    private func workFiles() -> [String] {
        ((try? FileManager.default.contentsOfDirectory(atPath: workspace.directory.path)) ?? []).sorted()
    }

    func testSecondRunHitsCache() async throws {
        let a = FakeGainStage(id: "one"), b = FakeGainStage(id: "two")
        let pipeline = ProcessingPipeline(stages: [a, b], workspace: workspace)
        let source = try original(seconds: 1)
        let first = try await pipeline.run(original: source) { _ in }
        let second = try await pipeline.run(original: source) { _ in }
        XCTAssertEqual(first.url, second.url)
        XCTAssertEqual(a.runs, 1)
        XCTAssertEqual(b.runs, 1)
        let samples = try readAllMono(second.url).samples
        XCTAssertEqual(samples.count, 48_000)
        XCTAssertEqual(samples.map { abs($0) }.max()!, 0.05, accuracy: 0.001)
    }

    func testSettingsChangeRecomputesDownstream() async throws {
        let a = FakeGainStage(id: "one"), b = FakeGainStage(id: "two")
        let pipeline = ProcessingPipeline(stages: [a, b], workspace: workspace)
        let source = try original(seconds: 1)
        _ = try await pipeline.run(original: source) { _ in }
        a.settingsKey = "changed"
        _ = try await pipeline.run(original: source) { _ in }
        XCTAssertEqual(a.runs, 2)
        XCTAssertEqual(b.runs, 2)
    }

    func testCancelledStageIsRecomputed() async throws {
        let source = try original(seconds: 1)
        let slow = FakeGainStage(id: "one", blockUntilCancelled: true)
        let blocked = ProcessingPipeline(stages: [slow], workspace: workspace)
        let task = Task { try await blocked.run(original: source) { _ in } }
        try await Task.sleep(nanoseconds: 300_000_000)
        task.cancel()
        _ = await task.result
        XCTAssertFalse(workFiles().contains { $0.hasSuffix(".caf") && !$0.contains(".partial") })

        let fast = FakeGainStage(id: "one")
        let result = try await ProcessingPipeline(stages: [fast], workspace: workspace).run(original: source) { _ in }
        XCTAssertEqual(fast.runs, 1)
        XCTAssertFalse(workFiles().contains { $0.contains(".partial") })
        XCTAssertEqual(try readAllMono(result.url).samples.count, 48_000)
    }

    func testHalfSecondInputProcesses() async throws {
        let enhancer = try await PipelineTestModel.enhancer()
        let stage = ReuseStage(enhancer: enhancer, variantID: "reuse-nemotron", revision: "test")
        let result = try await ProcessingPipeline(stages: [stage], workspace: workspace)
            .run(original: try original(seconds: 0.5)) { _ in }
        let samples = try readAllMono(result.url).samples
        XCTAssertEqual(samples.count, 24_000)
        XCTAssertTrue(samples.allSatisfy(\.isFinite))
    }

    func test16kImportIsEnhancedAtSourceRateThenResampled() async throws {
        let enhancer = try await PipelineTestModel.enhancer()
        let reuse = ReuseStage(enhancer: enhancer, variantID: "reuse-nemotron", revision: "test")
        let pipeline = ProcessingPipeline(stages: [reuse, ResampleStage()], workspace: workspace)
        let stagesSeen = LockedList()
        let result = try await pipeline.run(original: try original(seconds: 2, sampleRate: 16_000)) { p in
            stagesSeen.append(p.stageID)
        }
        XCTAssertEqual(stagesSeen.unique, ["reuse-nemotron", "resample-48k"])
        XCTAssertEqual(result.sampleRate, 48_000)
        let (samples, rate) = try readAllMono(result.url)
        XCTAssertEqual(rate, 48_000)
        XCTAssertEqual(samples.count, 96_000, accuracy: 2)
    }

    /// RE-USE is limited to 8–48 kHz: a 96 kHz import is enhanced at 48 kHz.
    func test96kImportIsEnhancedAt48k() async throws {
        let enhancer = try await PipelineTestModel.enhancer()
        let reuse = ReuseStage(enhancer: enhancer, variantID: "reuse-nemotron", revision: "test")
        let pipeline = ProcessingPipeline(stages: [reuse, ResampleStage()], workspace: workspace)
        let result = try await pipeline.run(original: try original(seconds: 2, sampleRate: 96_000)) { _ in }
        let (samples, rate) = try readAllMono(result.url)
        XCTAssertEqual(rate, 48_000)
        XCTAssertEqual(samples.count, 96_000, accuracy: 2)
    }

    func test48kInputSkipsResample() async throws {
        let a = FakeGainStage(id: "one")
        let pipeline = ProcessingPipeline(stages: [a, ResampleStage()], workspace: workspace)
        let result = try await pipeline.run(original: try original(seconds: 1)) { _ in }
        XCTAssertTrue(result.url.lastPathComponent.hasPrefix("one-"))
    }

    func testTwentyMinuteInputKeepsPeakMemoryBounded() async throws {
        guard ProcessInfo.processInfo.environment["MICPOLISH_SLOW_TESTS"] == "1" else {
            throw XCTSkip("Set MICPOLISH_SLOW_TESTS=1")
        }
        let enhancer = try await PipelineTestModel.enhancer()
        let stage = ReuseStage(enhancer: enhancer, variantID: "reuse-nemotron", revision: "test")

        // Reference: peak MLX memory for a short input.
        let shortWorkspace = Workspace(cacheRoot: dir.appendingPathComponent("short"), sessionID: UUID())
        MLX.Memory.peakMemory = 0
        _ = try await ProcessingPipeline(stages: [stage], workspace: shortWorkspace).run(original: try original(seconds: 30)) { _ in }
        let shortPeak = MLX.Memory.peakMemory

        let source = try original(seconds: 20 * 60)
        MLX.Memory.peakMemory = 0
        let rssBefore = residentBytes()
        let started = Date()
        let result = try await ProcessingPipeline(stages: [stage], workspace: workspace).run(original: source) { _ in }
        let elapsed = Date().timeIntervalSince(started)
        let longPeak = MLX.Memory.peakMemory
        print("20 min @48k: \(elapsed) s, RTF \(elapsed / 1200), MLX peak \(longPeak / 1_048_576) MB (30 s input: \(shortPeak / 1_048_576) MB), RSS growth \((residentBytes() - rssBefore) / 1_048_576) MB")
        // Memory must not grow with duration (a 20-minute input peaks no higher
        // than a 30-second one) and stays under the 5 GB budget.
        XCTAssertLessThanOrEqual(Double(longPeak), Double(shortPeak) * 1.1)
        XCTAssertLessThan(longPeak, 5 << 30)
        XCTAssertLessThan(residentBytes() - rssBefore, 500 << 20)
        XCTAssertEqual(try MonoFileReader(url: result.url).frameCount, 20 * 60 * 48_000)
    }

    private func residentBytes() -> Int {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? Int(info.resident_size) : 0
    }
}

final class LockedList: @unchecked Sendable {
    private let lock = NSLock()
    private var items: [String] = []
    func append(_ s: String) { lock.lock(); if items.last != s { items.append(s) }; lock.unlock() }
    var unique: [String] { lock.lock(); defer { lock.unlock() }; return items }
}

enum PipelineTestModel {
    static func enhancer() async throws -> MambaEnhancer {
        guard let path = ProcessInfo.processInfo.environment["MICPOLISH_REUSE_SOURCE"] else {
            throw XCTSkip("Set MICPOLISH_REUSE_SOURCE to the NemotronLabs model.safetensors")
        }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("micpolish-pipeline-\(ProcessInfo.processInfo.processIdentifier)/model_mlx.safetensors")
        if !FileManager.default.fileExists(atPath: url.path) {
            try ReuseWeightConverter.convert(input: URL(fileURLWithPath: path), output: url)
        }
        return try MambaEnhancer(modelWeightsURL: url)
    }
}

/// Returns its input unchanged and records the cutoffs it was asked for.
final class IdentityExtender: BandwidthExtender, @unchecked Sendable {
    private let lock = NSLock()
    private(set) var cutoffs: [Double] = []
    private(set) var chunkSizes: [Int] = []
    func enhance(_ samples: [Float], cutoffHz: Double) throws -> [Float] {
        lock.lock(); cutoffs.append(cutoffHz); chunkSizes.append(samples.count); lock.unlock()
        return samples
    }
}

final class LavaStageTests: XCTestCase {
    func testChunkedCrossfadeReconstructsInput() async throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let input = dir.appendingPathComponent("in.caf")
        let samples = (0..<(48_000 * 25 + 123)).map { Float(0.3 * sin(Double($0) * 0.01)) }
        let w = try MonoFileWriter(url: input, sampleRate: 48_000, format: .cafFloat32)
        try w.append(samples)
        try w.close()

        let extender = IdentityExtender()
        let stage = LavaStage(enhancer: extender, revision: "test", cutoffHz: 12_000)
        let output = dir.appendingPathComponent("out.caf")
        try await stage.run(input: StageIO(url: input, sampleRate: 48_000, cacheKey: "k"), output: output) { _ in }

        let result = try readAllMono(output).samples
        XCTAssertEqual(result.count, samples.count)
        XCTAssertLessThanOrEqual(zip(result, samples).map { abs($0 - $1) }.max() ?? 1, 1e-6)
        XCTAssertEqual(Set(extender.cutoffs), [12_000])
        XCTAssertTrue(extender.chunkSizes.allSatisfy { $0 <= LavaStage.chunkFrames })
    }

    func testExactChunkBoundaryDoesNotReenhanceTheOverlap() async throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let input = dir.appendingPathComponent("in.caf")
        let hop = LavaStage.chunkFrames - LavaStage.overlapFrames
        let samples = (0..<(LavaStage.chunkFrames + hop)).map { Float(0.3 * sin(Double($0) * 0.01)) }
        let w = try MonoFileWriter(url: input, sampleRate: 48_000, format: .cafFloat32)
        try w.append(samples)
        try w.close()

        let extender = IdentityExtender()
        let stage = LavaStage(enhancer: extender, revision: "test", cutoffHz: 12_000)
        let output = dir.appendingPathComponent("out.caf")
        try await stage.run(input: StageIO(url: input, sampleRate: 48_000, cacheKey: "k"), output: output) { _ in }

        XCTAssertEqual(try readAllMono(output).samples.count, samples.count)
        XCTAssertEqual(extender.chunkSizes, [LavaStage.chunkFrames, LavaStage.chunkFrames])
    }

    func testAutoCutoffUsesMeasuredBandwidth() async throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        // Noise band-limited to 8 kHz by upsampling from 16 kHz.
        var state: UInt64 = 9
        let narrow = (0..<(16_000 * 3)).map { _ -> Float in
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float(Double(state >> 11) / Double(1 << 53) * 2 - 1) * 0.2
        }
        let resampler = try StreamingResampler(from: 16_000, to: 48_000)
        let wide = try resampler.process(narrow) + resampler.flush()
        let input = dir.appendingPathComponent("in.caf")
        let w = try MonoFileWriter(url: input, sampleRate: 48_000, format: .cafFloat32)
        try w.append(wide)
        try w.close()

        let extender = IdentityExtender()
        let stage = LavaStage(enhancer: extender, revision: "test", cutoffHz: nil)
        try await stage.run(input: StageIO(url: input, sampleRate: 48_000, cacheKey: "k"),
                            output: dir.appendingPathComponent("out.caf")) { _ in }
        XCTAssertEqual(try XCTUnwrap(extender.cutoffs.first), 8_000, accuracy: 600)
    }
}

/// Halves the signal and releases it in 1000-sample pieces, like a model
/// that holds back a chunk.
final class HalvingIsolator: VoiceIsolating, @unchecked Sendable {
    final class Session: IsolationSession {
        private var pending: [Float] = []
        func push(_ samples: [Float]) throws -> [Float] {
            pending += samples.map { $0 * 0.5 }
            let ready = pending.count / 1000 * 1000
            defer { pending.removeFirst(ready) }
            return Array(pending.prefix(ready))
        }
        func finish() throws -> [Float] { defer { pending = [] }; return pending }
    }
    func makeIsolationSession() -> any IsolationSession { Session() }
}

final class InworldStageTests: XCTestCase {
    func testResamplesTo48kAndKeepsDuration() async throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let input = dir.appendingPathComponent("in.caf")
        let w = try MonoFileWriter(url: input, sampleRate: 16_000, format: .cafFloat32)
        try w.append((0..<16_123).map { Float(0.4 * sin(Double($0) * 0.05)) })
        try w.close()

        let stage = InworldStage(isolator: HalvingIsolator(), revision: "test")
        XCTAssertEqual(stage.outputSampleRate(forInput: 16_000), 48_000)
        let output = dir.appendingPathComponent("out.caf")
        try await stage.run(input: StageIO(url: input, sampleRate: 16_000, cacheKey: "k"), output: output) { _ in }

        let (samples, rate) = try readAllMono(output)
        XCTAssertEqual(rate, 48_000)
        XCTAssertEqual(Double(samples.count), Double(16_123 * 3), accuracy: 3)
        // Halved 0.4-amplitude sine, away from the resampler's edge transients.
        XCTAssertEqual(Double(samples[4_800..<(samples.count - 4_800)].map(abs).max() ?? 0), 0.2, accuracy: 0.01)
    }
}
