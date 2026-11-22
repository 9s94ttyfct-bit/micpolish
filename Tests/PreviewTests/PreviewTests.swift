import AudioCore
import Mastering
import XCTest
@testable import Preview

@MainActor
final class PreviewTests: XCTestCase {
    private var dir: URL!

    override func setUp() async throws {
        dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }

    override func tearDown() async throws {
        try? FileManager.default.removeItem(at: dir)
    }

    private func write(_ name: String, _ samples: [Float]) throws -> URL {
        let url = dir.appendingPathComponent(name)
        let w = try MonoFileWriter(url: url, sampleRate: 48_000, format: .cafFloat32)
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

    func testManualRenderingMatchesOfflineChain() throws {
        let processed = noise(seconds: 5, amplitude: 0.3, seed: 3)
        let url = try write("p.caf", processed)
        let settings = MasteringSettings.macbookMicCorrection
        let player = PreviewPlayer(mastering: settings)
        try player.enableManualRendering()
        try player.load(original: url, processed: url)
        player.setMastering(settings, loudnessGainDB: 4)
        player.source = .mastered
        player.play()
        let gain = player.state.withLock { player.state.gains[.mastered]! }
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

    func testSwitchingSourceKeepsPosition() throws {
        let original = [Float](repeating: 0.1, count: 4 * 48_000)
        let processed = [Float](repeating: 0.2, count: 4 * 48_000)
        let player = PreviewPlayer()
        try player.enableManualRendering()
        try player.load(original: try write("o.caf", original), processed: try write("p.caf", processed))
        player.play()
        _ = try player.renderManually(frames: 2 * 48_000)
        let before = player.state.withLock { player.state.position }
        player.source = .processed
        XCTAssertEqual(player.state.withLock { player.state.position }, before)
        let after = try player.renderManually(frames: 4096)
        let gain = player.state.withLock { player.state.gains[.processed]! }
        XCTAssertEqual(after[100], 0.2 * gain, accuracy: 1e-6)
    }

    func testLoadingOriginalOnlyResetsSourceToOriginal() throws {
        let url = try write("o.caf", [Float](repeating: 0.1, count: 48_000))
        let player = PreviewPlayer()
        try player.load(original: url, processed: url)
        player.source = .mastered
        try player.load(original: url, processed: nil)
        XCTAssertEqual(player.source, .original)
    }

    func testLevelMatchingAlignsLoudness() throws {
        let quiet = noise(seconds: 6, amplitude: 0.05, seed: 1)
        let loud = quiet.map { $0 * Float(pow(10, 10.0 / 20)) }
        let player = PreviewPlayer()
        try player.enableManualRendering()
        try player.load(original: try write("q.caf", quiet), processed: try write("l.caf", loud))

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
}
