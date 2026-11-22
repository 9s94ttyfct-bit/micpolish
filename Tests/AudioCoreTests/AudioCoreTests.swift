import Accelerate
import AVFoundation
import XCTest
@testable import AudioCore

final class AudioCoreTests: XCTestCase {
    private var dir: URL!

    override func setUpWithError() throws {
        dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: dir)
    }

    private func sine(frequency: Double, sampleRate: Int, seconds: Double, amplitude: Float = 0.5) -> [Float] {
        (0..<Int(Double(sampleRate) * seconds)).map {
            amplitude * Float(sin(2 * Double.pi * frequency * Double($0) / Double(sampleRate)))
        }
    }

    func testFlac24RoundTripWithinOneLSB() throws {
        var generator = SystemRandomNumberGenerator()
        let input = (0..<48_000).map { _ in Float.random(in: -0.5...0.5, using: &generator) }
        let url = dir.appendingPathComponent("a.flac")
        let writer = try MonoFileWriter(url: url, sampleRate: 48_000, format: .flac24)
        try writer.append(Array(input[..<20_000]))
        try writer.append(Array(input[20_000...]))
        try writer.close()

        let (output, rate) = try readAllMono(url)
        XCTAssertEqual(rate, 48_000)
        XCTAssertEqual(output.count, input.count)
        let maxError = zip(input, output).map { abs($0 - $1) }.max() ?? 1
        XCTAssertLessThanOrEqual(maxError, 2 * pow(2, -23))
    }

    func testStereoFileDownmixesToMean() throws {
        let url = dir.appendingPathComponent("stereo.caf")
        let format = AVAudioFormat(standardFormatWithSampleRate: 44_100, channels: 2)!
        let file = try AVAudioFile(forWriting: url, settings: format.settings)
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 1000)!
        buffer.frameLength = 1000
        for i in 0..<1000 {
            buffer.floatChannelData![0][i] = 0.4
            buffer.floatChannelData![1][i] = -0.2
        }
        try file.write(from: buffer)

        let reader = try MonoFileReader(url: url)
        XCTAssertEqual(reader.sampleRate, 44_100)
        XCTAssertEqual(reader.frameCount, 1000)
        var samples = [Float]()
        while let chunk = try reader.read(maxFrames: 300) { samples += chunk }
        XCTAssertEqual(samples.count, 1000)
        for s in samples { XCTAssertEqual(s, 0.1, accuracy: 1e-6) }
    }

    func testResample16kTo48kPreservesToneAndLength() throws {
        let input = sine(frequency: 1000, sampleRate: 16_000, seconds: 2)
        let resampler = try StreamingResampler(from: 16_000, to: 48_000)
        let output = try resampler.process(input) + resampler.flush()
        XCTAssertEqual(output.count, 96_000, accuracy: 1)

        // Dominant frequency of a 1 s analysis window from the middle.
        let n = 48_000
        let window = Array(output[24_000..<(24_000 + n)])
        var bestBin = 0, best = 0.0
        for bin in stride(from: 900, through: 1100, by: 1) {
            var re = 0.0, im = 0.0
            for (i, s) in window.enumerated() {
                let phase = 2 * Double.pi * Double(bin) * Double(i) / Double(n)
                re += Double(s) * cos(phase)
                im -= Double(s) * sin(phase)
            }
            let power = re * re + im * im
            if power > best { best = power; bestBin = bin }
        }
        XCTAssertEqual(bestBin, 1000, accuracy: 5)
    }

    func testChunkedResampleEqualsWhole() throws {
        var generator = SystemRandomNumberGenerator()
        let input = (0..<44_100).map { _ in Float.random(in: -0.5...0.5, using: &generator) }

        let whole = try StreamingResampler(from: 44_100, to: 48_000)
        let expected = try whole.process(input) + whole.flush()

        let chunked = try StreamingResampler(from: 44_100, to: 48_000)
        var actual = [Float]()
        for start in stride(from: 0, to: input.count, by: 1000) {
            actual += try chunked.process(Array(input[start..<min(start + 1000, input.count)]))
        }
        actual += try chunked.flush()

        XCTAssertEqual(actual.count, expected.count)
        let maxDiff = zip(actual, expected).map { abs($0 - $1) }.max() ?? 1
        XCTAssertLessThanOrEqual(maxDiff, 1e-6)
    }

    func testReaderReturnsEveryFrameOfLongFile() throws {
        // AVAudioFile.read(into:) can return short reads; the reader must not lose frames.
        let input = sine(frequency: 440, sampleRate: 48_000, seconds: 4)
        let url = dir.appendingPathComponent("long.caf")
        let writer = try MonoFileWriter(url: url, sampleRate: 48_000, format: .cafFloat32)
        try writer.append(input)
        try writer.close()

        let (output, _) = try readAllMono(url)
        XCTAssertEqual(output, input)
    }
}
