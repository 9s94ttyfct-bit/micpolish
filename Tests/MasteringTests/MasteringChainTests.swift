import AudioCore
import XCTest
@testable import Mastering

final class MasteringChainTests: XCTestCase {
    private func allBypassed() -> MasteringSettings {
        var s = MasteringSettings.default
        s.highpass.enabled = false
        s.eq.enabled = false
        s.deEsser.enabled = false
        s.compressor.enabled = false
        s.loudness.enabled = false
        s.limiter.enabled = false
        return s
    }

    private func rmsDB(_ x: ArraySlice<Float>) -> Double {
        10 * log10(x.reduce(0) { $0 + Double($1) * Double($1) } / Double(x.count))
    }

    func testAllBypassedIsIdentity() {
        let input = TestSignals.modulatedNoise(seconds: 2, rmsDB: -20)
        var output = input
        let chain = MasteringChain(settings: allBypassed(), sampleRate: 48_000)
        chain.process(&output)
        XCTAssertEqual(output, input)
        XCTAssertEqual(chain.latencyFrames, 0)
    }

    func testBlockSizeInvariance() {
        let input = TestSignals.modulatedNoise(seconds: 10, rmsDB: -18)
        var settings = MasteringSettings.macbookMicCorrection
        settings.loudness.enabled = true

        func run(block: Int) -> [Float] {
            let chain = MasteringChain(settings: settings, sampleRate: 48_000)
            chain.loudnessGainDB = 6
            var out = [Float]()
            var i = 0
            while i < input.count {
                var piece = Array(input[i..<min(i + block, input.count)])
                chain.process(&piece)
                out += piece
                i += block
            }
            return out
        }
        let whole = run(block: input.count)
        XCTAssertEqual(run(block: 256), whole)
        XCTAssertEqual(run(block: 4096), whole)
    }

    func testHighpassAttenuates40Hz() {
        var settings = allBypassed()
        settings.highpass.enabled = true
        let low = TestSignals.sine(frequency: 40, amplitude: 0.5, seconds: 2)
        let mid = TestSignals.sine(frequency: 1000, amplitude: 0.5, seconds: 2)
        var lowOut = low, midOut = mid
        MasteringChain(settings: settings, sampleRate: 48_000).process(&lowOut)
        MasteringChain(settings: settings, sampleRate: 48_000).process(&midOut)
        let tail = 48_000..<96_000
        XCTAssertLessThanOrEqual(rmsDB(lowOut[tail]) - rmsDB(low[tail]), -10)
        XCTAssertGreaterThanOrEqual(rmsDB(midOut[tail]) - rmsDB(mid[tail]), -0.5)
    }

    func testRenderHitsTargetLoudness() throws {
        let report = try render(TestSignals.modulatedNoise(seconds: 20, rmsDB: -32), settings: .default)
        print("render report:", report.report)
        XCTAssertEqual(try XCTUnwrap(report.report.outputLUFS), -16, accuracy: 0.5)
        XCTAssertLessThanOrEqual(report.report.outputTruePeakDBTP, -1.0)
        XCTAssertEqual(report.output.count, 20 * 48_000)
    }

    func testLimiterHoldsCeilingUnderHeavyLimiting() throws {
        var settings = MasteringSettings.default
        settings.compressor.enabled = false
        settings.loudness.targetLUFS = -8
        // Clicks plus a 12 kHz tone sampled off-peak: intersample peaks everywhere.
        var input = TestSignals.modulatedNoise(seconds: 10, rmsDB: -30, seed: 7)
        let tone = TestSignals.sine(frequency: 12_000, amplitude: 0.05, seconds: 10, phase: Double.pi / 4)
        for i in input.indices { input[i] += tone[i] }
        for i in stride(from: 1000, to: input.count, by: 9_973) { input[i] = 0.9 }
        let result = try render(input, settings: settings)
        print("heavy limiting report:", result.report)
        XCTAssertLessThanOrEqual(result.report.outputTruePeakDBTP, -1.0)
    }

    func testRenderHitsTargetWhenLimiterEngages() throws {
        // Clicks give the signal a high crest factor, so the limiter removes
        // loudness after the gain; the renderer must compensate.
        var input = TestSignals.modulatedNoise(seconds: 12, rmsDB: -30, seed: 11)
        for i in stride(from: 500, to: input.count, by: 4_801) { input[i] = 0.6 }
        let result = try render(input, settings: .default)
        print("limited render report:", result.report)
        XCTAssertEqual(try XCTUnwrap(result.report.outputLUFS), -16, accuracy: 0.5)
        XCTAssertLessThanOrEqual(result.report.outputTruePeakDBTP, -1.0)
    }

    func testSilenceProducesSilenceWithoutGain() throws {
        let result = try render([Float](repeating: 0, count: 48_000), settings: .default)
        XCTAssertEqual(result.report.appliedGainDB, 0)
        XCTAssertNil(result.report.measuredLUFS)
        XCTAssertTrue(result.output.allSatisfy { $0 == 0 })
    }

    private func render(_ input: [Float], settings: MasteringSettings) throws -> (report: RenderReport, output: [Float]) {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let src = dir.appendingPathComponent("in.caf"), dst = dir.appendingPathComponent("out.caf")
        let w = try MonoFileWriter(url: src, sampleRate: 48_000, format: .cafFloat32)
        try w.append(input)
        try w.close()

        let out = try MonoFileWriter(url: dst, sampleRate: 48_000, format: .cafFloat32)
        let report = try MasteringRenderer.render(input: src, settings: settings, output: out)
        try out.close()
        return (report, try readAllMono(dst).samples)
    }
}
