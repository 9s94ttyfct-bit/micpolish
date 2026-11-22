import Foundation
import MLX
import XCTest
@testable import InworldMLX

/// Compares the MLX port with upstream PyTorch (CPU, float32) on the
/// fixtures of Tests/Fixtures/inworld. Needs MICPOLISH_INWORLD_WEIGHTS.
final class InworldParityTests: XCTestCase {
    nonisolated(unsafe) private static var cachedIsolator: InworldIsolator?

    private func isolator() throws -> InworldIsolator {
        let url = try InworldFixtures.weightsURL()
        if let cached = Self.cachedIsolator { return cached }
        let isolator = try InworldIsolator(weightsURL: url)
        Self.cachedIsolator = isolator
        return isolator
    }

    /// 3.98 s of band-limited speech with generated noise: one pass, no stitching.
    func testParityNoisy4s() throws {
        try assertParity(input: "noisy_4s.wav", reference: "ref_noisy_4s.f32", pieces: [4_800, 1, 12_345])
    }

    /// 15.84 s: three 8 s windows, the last placed at the clip's end off the 6 s stride.
    func testParityStream16s() throws {
        try assertParity(input: "stream_16s.wav", reference: "ref_stream_16s.f32", pieces: [24_000, 333, 100_001, 7])
    }

    /// Upstream's own golden offline fixture at its own tolerance (max abs 1e-4).
    func testUpstreamGoldenOffline3s() throws {
        try assertGolden(directory: InworldFixtures.directory.appendingPathComponent("upstream_offline_3s"))
    }

    /// Upstream's 11 s golden (two windows), when an upstream checkout's
    /// tests/fixtures directory is named by MICPOLISH_INWORLD_UPSTREAM_FIXTURES.
    func testUpstreamGoldenOffline11s() throws {
        guard let path = ProcessInfo.processInfo.environment["MICPOLISH_INWORLD_UPSTREAM_FIXTURES"] else {
            throw XCTSkip("Set MICPOLISH_INWORLD_UPSTREAM_FIXTURES to voice-isolator's tests/fixtures")
        }
        try assertGolden(directory: URL(fileURLWithPath: path).appendingPathComponent("offline_11s"))
    }

    func testShortInputsKeepTheirLength() throws {
        let isolator = try isolator()
        let speech = try InworldFixtures.readWAV(InworldFixtures.directory.appendingPathComponent("noisy_4s.wav"))
        for count in [1, 100, 479, 480, 1024, 1025, 4800] {
            let output = try InworldFixtures.run(isolator.makeSession(), Array(speech[20_000 ..< 20_000 + count]), pieces: [count])
            XCTAssertEqual(output.count, count)
            XCTAssertTrue(output.allSatisfy(\.isFinite), "count \(count)")
        }
        XCTAssertEqual(try isolator.makeSession().finish(), [])
    }

    /// RTF and peak MLX memory of a 60 s stream (pushed in 0.5 s pieces).
    func testSixtySecondRuntimeAndMemory() throws {
        let isolator = try isolator()
        let speech = try InworldFixtures.readWAV(InworldFixtures.directory.appendingPathComponent("stream_16s.wav"))
        let count = 60 * InworldIsolator.sampleRate
        let audio = (0..<count).map { speech[$0 % speech.count] }
        // Warm up the kernels at the window length.
        _ = isolator.render(Array(audio[0 ..< isolator.window.chunk]))
        Memory.clearCache()
        Memory.peakMemory = 0
        let start = Date()
        let output = try InworldFixtures.run(isolator.makeSession(), audio, pieces: [24_000])
        let seconds = Date().timeIntervalSince(start)
        let peak = Memory.peakMemory
        #if DEBUG
            let build = "debug"
        #else
            let build = "release"
        #endif
        print("Inworld 60 s (\(build)): \(String(format: "%.2f", seconds)) s, RTF \(String(format: "%.3f", seconds / 60)), "
              + "peak MLX memory \(String(format: "%.2f", Double(peak) / 1_073_741_824)) GiB")
        XCTAssertEqual(output.count, count)
        XCTAssertTrue(output.allSatisfy(\.isFinite))
        XCTAssertLessThan(peak, 5 << 30)
    }

    /// Relative RMSE of the output, and of the model's correction (output
    /// minus input) against the reference's, both ≤ 1e-3.
    private func assertParity(input: String, reference: String, pieces: [Int]) throws {
        let isolator = try isolator()
        let samples = try InworldFixtures.readWAV(InworldFixtures.directory.appendingPathComponent(input))
        let expected = try InworldFixtures.readFloat32(InworldFixtures.directory.appendingPathComponent(reference))
        XCTAssertEqual(expected.count, samples.count)

        let actual = try InworldFixtures.run(isolator.makeSession(), samples, pieces: pieces)
        XCTAssertEqual(actual.count, samples.count)
        XCTAssertTrue(actual.allSatisfy(\.isFinite))

        let error = InworldFixtures.relativeRMSE(actual, expected)
        let correctionError = InworldFixtures.relativeRMSE(
            zip(actual, samples).map { $0 - $1 }, zip(expected, samples).map { $0 - $1 })
        let maxError = zip(actual, expected).map { abs($0 - $1) }.max() ?? 0
        print("Inworld parity \(input): relative RMSE \(error), correction \(correctionError), max abs \(maxError)")
        XCTAssertLessThanOrEqual(error, 1e-3)
        XCTAssertLessThanOrEqual(correctionError, 1e-3)
    }

    private func assertGolden(directory: URL) throws {
        let isolator = try isolator()
        let meta = try XCTUnwrap(JSONSerialization.jsonObject(
            with: Data(contentsOf: directory.appendingPathComponent("meta.json"))) as? [String: Any])
        XCTAssertEqual(meta["mode"] as? String, "offline")
        XCTAssertEqual(meta["chunk_seconds"] as? Double, 8)
        XCTAssertEqual(meta["overlap_seconds"] as? Double, 2)
        let tolerance = try XCTUnwrap(meta["tolerance"] as? Double)
        let samples = try InworldFixtures.readWAV(directory.appendingPathComponent("input.wav"))
        let expected = try InworldFixtures.readNpy(directory.appendingPathComponent("expected.npy"))
        let actual = try InworldFixtures.run(isolator.makeSession(), samples, pieces: [48_000, 999])
        XCTAssertEqual(actual.count, expected.count)
        let maxError = zip(actual, expected).map { Double(abs($0 - $1)) }.max() ?? 0
        let error = InworldFixtures.relativeRMSE(actual, expected)
        print("Inworld golden \(directory.lastPathComponent): max abs \(maxError) (tolerance \(tolerance)), relative RMSE \(error)")
        XCTAssertLessThanOrEqual(maxError, tolerance)
        XCTAssertLessThanOrEqual(error, 1e-3)
    }
}
