import AudioCore
import XCTest

final class AudioTrimTests: XCTestCase {
    private var dir: URL!

    override func setUpWithError() throws {
        dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: dir)
    }

    private func ramp(count: Int, sampleRate: Int) throws -> URL {
        let url = dir.appendingPathComponent("in.caf")
        let w = try MonoFileWriter(url: url, sampleRate: sampleRate, format: .cafFloat32)
        try w.append((0..<count).map { Float($0) / Float(count) })
        try w.close()
        return url
    }

    func testWritesRequestedSpanWithFadesOnlyAtCuts() throws {
        let input = try ramp(count: 48_000 * 3, sampleRate: 48_000)
        let out = dir.appendingPathComponent("out.caf")
        try AudioTrim.write(input, from: 0.5, to: 2.0, output: out)
        let (samples, rate) = try readAllMono(out)
        XCTAssertEqual(rate, 48_000)
        XCTAssertEqual(samples.count, 72_000)
        XCTAssertEqual(samples[0], 0)                              // faded in at the cut
        XCTAssertEqual(samples[1_000], Float(25_000) / 144_000, accuracy: 1e-6)  // untouched middle
        XCTAssertEqual(samples.last!, 0)                           // faded out at the cut
    }

    func testKeepingTheFileStartAndEndAddsNoFadeAndUsesTheInputRate() throws {
        let input = try ramp(count: 16_000, sampleRate: 16_000)
        let out = dir.appendingPathComponent("out.caf")
        try AudioTrim.write(input, from: 0, to: 5, output: out)    // end past the file clamps
        let (samples, rate) = try readAllMono(out)
        XCTAssertEqual(rate, 16_000)
        XCTAssertEqual(samples.count, 16_000)
        XCTAssertEqual(samples[1], 1 / 16_000, accuracy: 1e-7)
        XCTAssertEqual(samples.last!, 15_999 / 16_000, accuracy: 1e-6)
    }

    func testDetectSoundRangeSkipsQuietHeadAndTail() throws {
        // 2 s of noise floor (−60 dB re peak), 3 s of sound, 1 s of floor; 50 ms buckets.
        let envelope = [Float](repeating: 0.0005, count: 40) + [Float](repeating: 0.5, count: 60)
            + [Float](repeating: 0.0005, count: 20)
        let range = try XCTUnwrap(AudioTrim.detectSoundRange(envelope: envelope, bucketSeconds: 0.05))
        XCTAssertEqual(range.start, 1.75, accuracy: 1e-9)
        XCTAssertEqual(range.end, 5.25, accuracy: 1e-9)
        XCTAssertNil(AudioTrim.detectSoundRange(envelope: [0, 0, 0], bucketSeconds: 0.05))
    }

    func testDetectSoundRangeIgnoresClicksRoomToneAndDigitalSilence() throws {
        func db(_ v: Float) -> Float { pow(10, v / 20) }
        // Digital silence, a quiet floor with one click, louder room tone, speech, quiet tail.
        var envelope = [Float](repeating: 0, count: 2) + [Float](repeating: db(-44), count: 8)
        envelope[5] = db(-10)
        envelope += [Float](repeating: db(-33), count: 24) + [Float](repeating: db(-5), count: 30)
            + [Float](repeating: db(-44), count: 10)
        envelope[40] = 1
        let range = try XCTUnwrap(AudioTrim.detectSoundRange(envelope: envelope, bucketSeconds: 0.05))
        XCTAssertEqual(range.start, 34 * 0.05 - 0.25, accuracy: 1e-9)
        XCTAssertEqual(range.end, 64 * 0.05 + 0.25, accuracy: 1e-9)
    }
}
