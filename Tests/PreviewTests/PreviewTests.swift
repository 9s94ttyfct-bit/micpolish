import AudioCore
import AVFoundation
import Mastering
import XCTest
@testable import Preview

@MainActor
final class PreviewTests: XCTestCase {
    private var dir: URL!
    private var scratch: URL!

    override func setUp() async throws {
        dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        scratch = dir.appendingPathComponent("scratch")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }

    override func tearDown() async throws {
        try? FileManager.default.removeItem(at: dir)
    }

    private func write(_ name: String, _ samples: [Float], sampleRate: Int = 48_000) throws -> URL {
        let url = dir.appendingPathComponent(name)
        let w = try MonoFileWriter(url: url, sampleRate: sampleRate, format: .cafFloat32)
        try w.append(samples)
        try w.close()
        return url
    }

    private func noise(seconds: Double, amplitude: Float, seed: UInt64) -> [Float] {
        var state = seed
        return (0..<Int(seconds * 48_000)).map { i in
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let r = Float(Double(state >> 11) / Double(1 << 53) * 2 - 1)
            return amplitude * r * Float(max(0, sin(2 * Double.pi * 3 * Double(i) / 48_000)))
        }
    }

    func testManualRenderingMatchesOfflineChain() async throws {
        let processed = noise(seconds: 5, amplitude: 0.3, seed: 3)
        let url = try write("p.caf", processed)
        let settings = MasteringSettings.macbookMicCorrection
        let player = PreviewPlayer(mastering: settings)
        try player.enableManualRendering()
        try await player.load(original: url, processed: url, scratchDirectory: scratch)
        player.setMastering(settings, loudnessGainDB: 4)
        player.source = .mastered
        player.play()
        let gain = player.state.withLock { player.state.masteredGain }
        let rendered = try player.renderManually(frames: 5 * 48_000)

        let chain = MasteringChain(settings: settings, sampleRate: 48_000)
        chain.loudnessGainDB = 4
        var expected = processed + [Float](repeating: 0, count: chain.latencyFrames)
        chain.process(&expected)
        expected.removeFirst(chain.latencyFrames)

        XCTAssertEqual(rendered.count, expected.count)
        let maxDiff = zip(rendered, expected).map { abs($0 - $1 * gain) }.max() ?? 1
        XCTAssertLessThanOrEqual(maxDiff, 1e-6)
    }

    func testTrimRangeStartsAtTrimStartAndSilencesPastTheEnd() async throws {
        // Sample value encodes its index, so the rendered audio shows where playback was.
        let original = (0..<(4 * 48_000)).map { Float($0) / Float(4 * 48_000) }
        let player = PreviewPlayer()
        try player.enableManualRendering()
        try await player.load(original: try write("o.caf", original), processed: nil, scratchDirectory: scratch)
        player.setRange(start: 1, end: 2)
        XCTAssertEqual(player.position, 1)
        player.seek(to: 3.5)                                   // past the end clamps to it
        XCTAssertEqual(player.position, 2)
        player.play()                                          // at the end: restarts at the trim start
        let gain = player.state.withLock { player.state.originalGain }
        let rendered = try player.renderManually(frames: 2 * 48_000)
        XCTAssertEqual(rendered[0] / gain, original[48_000], accuracy: 1e-6)
        XCTAssertEqual(rendered[48_000 - 1] / gain, original[2 * 48_000 - 1], accuracy: 1e-6)
        XCTAssertTrue(rendered[48_000...].allSatisfy { $0 == 0 })
        XCTAssertFalse(player.state.withLock { player.state.playing })
    }

    func testSwitchingSourceKeepsPosition() async throws {
        let original = [Float](repeating: 0.1, count: 4 * 48_000)
        let processed = [Float](repeating: 0.2, count: 4 * 48_000)
        let player = PreviewPlayer()
        try player.enableManualRendering()
        try await player.load(original: try write("o.caf", original), processed: try write("p.caf", processed),
                              scratchDirectory: scratch)
        player.play()
        _ = try player.renderManually(frames: 2 * 48_000)
        let before = player.state.withLock { player.state.position }
        player.source = .processed
        XCTAssertEqual(player.state.withLock { player.state.position }, before)
        let after = try player.renderManually(frames: 4096)
        let gain = player.state.withLock { player.state.processedGain }
        XCTAssertEqual(after[100], 0.2 * gain, accuracy: 1e-6)
    }

    func testLoadingOriginalOnlyResetsSourceToOriginal() async throws {
        let url = try write("o.caf", [Float](repeating: 0.1, count: 48_000))
        let player = PreviewPlayer()
        try await player.load(original: url, processed: url, scratchDirectory: scratch)
        player.source = .mastered
        try await player.load(original: url, processed: nil, scratchDirectory: scratch)
        XCTAssertEqual(player.source, .original)
    }

    func testLevelMatchingAlignsLoudness() async throws {
        let quiet = noise(seconds: 6, amplitude: 0.05, seed: 1)
        let loud = quiet.map { $0 * Float(pow(10, 10.0 / 20)) }
        let player = PreviewPlayer()
        try player.enableManualRendering()
        try await player.load(original: try write("q.caf", quiet), processed: try write("l.caf", loud),
                              scratchDirectory: scratch)

        func renderedLoudness(_ source: PreviewSource) throws -> Double {
            player.seek(to: 0)
            player.source = source
            player.play()
            var meter = LoudnessMeter(sampleRate: 48_000)
            meter.add(try player.renderManually(frames: 6 * 48_000))
            return try XCTUnwrap(meter.integratedLUFS)
        }
        let a = try renderedLoudness(.original)
        let b = try renderedLoudness(.processed)
        XCTAssertEqual(a, b, accuracy: 0.5)
        XCTAssertEqual(a, PreviewPlayer.comparisonLUFS, accuracy: 0.5)
    }

    /// The render block runs on Core Audio's I/O thread. Building it inside a
    /// main-actor context makes Swift 6 insert a main-executor check that traps
    /// there, so render from a background queue exactly as Core Audio does.
    func testRenderBlockRunsOffTheMainThread() async throws {
        let url = try write("o.caf", [Float](repeating: 0.25, count: 48_000))
        let player = PreviewPlayer()
        try await player.load(original: url, processed: nil, scratchDirectory: scratch)
        player.state.withLock { player.state.playing = true }
        let state = player.state

        let rendered: [Float] = try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global().async {
                do {
                    let format = AVAudioFormat(standardFormatWithSampleRate: 48_000, channels: 1)!
                    let engine = AVAudioEngine()
                    let node = PreviewPlayer.makeSourceNode(format: format, state: state)
                    engine.attach(node)
                    engine.connect(node, to: engine.mainMixerNode, format: format)
                    try engine.enableManualRenderingMode(.offline, format: format, maximumFrameCount: 1024)
                    try engine.start()
                    let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 1024)!
                    _ = try engine.renderOffline(1024, to: buffer)
                    continuation.resume(returning: Array(UnsafeBufferPointer(start: buffer.floatChannelData![0], count: 1024)))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
        let gain = player.state.withLock { player.state.originalGain }
        XCTAssertEqual(rendered[10], 0.25 * gain, accuracy: 1e-6)
    }

    /// Signals are memory-mapped from the scratch directory, not held in
    /// process memory, so preview cost does not grow with recording length.
    func testLoadDoesNotHoldSignalsInMemory() async throws {
        // 5 minutes, written in pieces to keep the test's own footprint low.
        let url = dir.appendingPathComponent("long.caf")
        let w = try MonoFileWriter(url: url, sampleRate: 48_000, format: .cafFloat32)
        let piece = noise(seconds: 10, amplitude: 0.1, seed: 5)
        for _ in 0..<30 { try w.append(piece) }
        try w.close()

        let player = PreviewPlayer()
        let before = residentBytes()
        try await player.load(original: url, processed: url, scratchDirectory: scratch)
        let growth = residentBytes() - before
        print("preview load RSS growth: \(growth / 1_048_576) MB for 2 × 55 MB of samples")
        XCTAssertLessThan(growth, 80 << 20)
        XCTAssertEqual(player.duration, 300, accuracy: 0.01)
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
