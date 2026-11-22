import XCTest
@testable import AudioCore

final class BandwidthDetectorTests: XCTestCase {
    private func noise(_ count: Int, seed: UInt64) -> [Float] {
        var state = seed
        return (0..<count).map { _ in
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float(Double(state >> 11) / Double(1 << 53) * 2 - 1) * 0.2
        }
    }

    private func write(_ samples: [Float], rate: Int) throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).caf")
        let w = try MonoFileWriter(url: url, sampleRate: rate, format: .cafFloat32)
        try w.append(samples)
        try w.close()
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    func testFullBandNoiseReachesNearNyquist() throws {
        let url = try write(noise(48_000 * 3, seed: 1), rate: 48_000)
        XCTAssertGreaterThanOrEqual(try BandwidthDetector.effectiveBandwidth(of: url), 20_000)
    }

    func testUpsampled16kNoiseStopsNear8k() throws {
        let narrow = noise(16_000 * 3, seed: 2)
        let resampler = try StreamingResampler(from: 16_000, to: 48_000)
        let wide = try resampler.process(narrow) + resampler.flush()
        let bandwidth = try BandwidthDetector.effectiveBandwidth(of: try write(wide, rate: 48_000))
        XCTAssertEqual(bandwidth, 8_000, accuracy: 600)
    }

    func testSilenceReportsNyquist() throws {
        let url = try write([Float](repeating: 0, count: 48_000), rate: 48_000)
        XCTAssertEqual(try BandwidthDetector.effectiveBandwidth(of: url), 24_000)
    }
}
