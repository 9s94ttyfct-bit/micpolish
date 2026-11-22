import AudioCore
import XCTest
@testable import Mastering

final class StudioToneTests: XCTestCase {
    private var dir: URL!

    override func setUpWithError() throws {
        dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: dir)
    }

    private func write(_ samples: [Float], name: String) throws -> URL {
        let url = dir.appendingPathComponent(name)
        let w = try MonoFileWriter(url: url, sampleRate: 48_000, format: .cafFloat32)
        try w.append(samples)
        try w.close()
        return url
    }

    private func noise(seconds: Double, seed: UInt64 = 7) -> [Float] {
        var state = seed
        return (0..<Int(seconds * 48_000)).map { _ in
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float(Double(state >> 11) / Double(1 << 53) - 0.5) * 0.2
        }
    }

    /// A voiced-speech stand-in: harmonics of `f0` with a falling spectrum,
    /// gated on and off at syllable rate.
    private func voice(f0: Double, seconds: Double) -> [Float] {
        (0..<Int(seconds * 48_000)).map { i in
            let t = Double(i) / 48_000
            var s = 0.0
            for h in 1...20 { s += sin(2 * Double.pi * f0 * Double(h) * t) / Double(h) }
            return Float(0.1 * s * max(0, sin(2 * Double.pi * 3 * t)))
        }
    }

    func testFilterCascadeHitsTargetAtBandCentres() {
        let target: [Double] = [-3, 2, 4, -1, 0, 1, -2, 3, -4]
        let filters = StudioTone.filters(target: target, sampleRate: 48_000)
        for (i, f) in StudioTone.centers.enumerated() {
            let response = filters.reduce(0) { $0 + $1.responseDB(at: f, sampleRate: 48_000) }
            XCTAssertEqual(response, target[i], accuracy: 0.5, "band \(Int(f)) Hz")
        }
    }

    private var onlyStudioTone: MasteringSettings {
        var settings = MasteringSettings()
        settings.highpass.enabled = false
        settings.deEsser.enabled = false
        settings.compressor.enabled = false
        settings.loudness.enabled = false
        settings.limiter.enabled = false
        settings.studioTone = .init(enabled: true, amount: 1)
        return settings
    }

    private func render(_ input: URL, _ settings: MasteringSettings, name: String) throws -> URL {
        let url = dir.appendingPathComponent(name)
        let writer = try MonoFileWriter(url: url, sampleRate: 48_000, format: .cafFloat32)
        try MasteringRenderer.render(input: input, settings: settings, output: writer)
        try writer.close()
        return url
    }

    private static func distance(_ a: [Double], _ b: [Double]) -> Double {
        (1...7).reduce(0) { $0 + abs(a[$1] - b[$1]) }
    }

    func testCorrectionMovesTheSpectrumTowardsTheReference() throws {
        let input = try write(noise(seconds: 20), name: "noise.caf")
        let before = try StudioTone.analyze(url: input)
        let after = try StudioTone.analyze(url: render(input, onlyStudioTone, name: "out.caf"))
        let reference = before.voice == .male ? StudioTone.male : StudioTone.female
        // White noise is far from speech, so the ±6 dB limit leaves some distance.
        XCTAssertLessThan(Self.distance(after.shape, reference), 0.7 * Self.distance(before.shape, reference))
    }

    func testSpectrumAlreadyAtTheReferenceGetsNoCorrection() throws {
        // Log-spaced partials whose octave-band powers follow the female reference.
        let partialsPerBand = 20
        var partials: [(frequency: Double, amplitude: Double, phase: Double)] = []
        for (b, centre) in StudioTone.centers.enumerated() {
            let amplitude = (pow(10, StudioTone.female[b] / 10) / Double(partialsPerBand)).squareRoot()
            for p in 0..<partialsPerBand {
                let f = centre * pow(2, (Double(p) + 0.5) / Double(partialsPerBand) - 0.5)
                partials.append((f, amplitude, Double(partials.count) * 2.39996))
            }
        }
        let samples = (0..<(5 * 48_000)).map { i in
            Float(0.02 * partials.reduce(0) { $0 + $1.amplitude * sin(2 * Double.pi * $1.frequency * Double(i) / 48_000 + $1.phase) })
        }
        let analysis = try StudioTone.analyze(url: write(samples, name: "reference.caf"))
        XCTAssertEqual(analysis.voice, .female)
        // The narrow 125 Hz band loses some partial power to its neighbours (window leakage).
        XCTAssertEqual(analysis.correction[1], 0, accuracy: 2.5)
        for i in 2...8 {
            XCTAssertEqual(analysis.correction[i], 0, accuracy: 1.0, "band \(Int(StudioTone.centers[i])) Hz")
        }
    }

    func testVoiceIsChosenFromPitch() throws {
        let low = try StudioTone.analyze(url: write(voice(f0: 110, seconds: 4), name: "low.caf"))
        let high = try StudioTone.analyze(url: write(voice(f0: 220, seconds: 4), name: "high.caf"))
        XCTAssertEqual(try XCTUnwrap(low.medianPitchHz), 110, accuracy: 3)
        XCTAssertEqual(low.voice, .male)
        XCTAssertEqual(try XCTUnwrap(high.medianPitchHz), 220, accuracy: 6)
        XCTAssertEqual(high.voice, .female)
    }

    func testEmptyBandsAreNotBoosted() throws {
        // Content only below 4 kHz (e.g. a band-limited recording): 50 Hz-spaced partials.
        let partials = stride(from: 50.0, through: 4000, by: 50).enumerated().map { ($1, Double($0) * 2.39996) }
        let samples = (0..<(10 * 48_000)).map { i in
            Float(partials.reduce(0) { $0 + sin(2 * Double.pi * $1.0 * Double(i) / 48_000 + $1.1) } * 0.01)
        }
        let analysis = try StudioTone.analyze(url: write(samples, name: "dark.caf"))
        XCTAssertEqual(analysis.correction[8], 0)
        XCTAssertLessThanOrEqual(analysis.correction[0], 0)
    }

    func testSettingsWithoutStudioToneDecodeAsOff() throws {
        var old = try JSONSerialization.jsonObject(with: JSONEncoder().encode(MasteringSettings())) as! [String: Any]
        old.removeValue(forKey: "studioTone")
        let decoded = try JSONDecoder().decode(MasteringSettings.self, from: JSONSerialization.data(withJSONObject: old))
        XCTAssertEqual(decoded, MasteringSettings())
        XCTAssertFalse(decoded.studioTone.enabled)
    }
}
