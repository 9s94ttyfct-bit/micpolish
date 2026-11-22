import AVFoundation
import XCTest
@testable import LavaMLX

/// Compares the MLX port with the PyTorch CPU float32 output of upstream
/// `LavaBWE.infer` produced by tools/parity/lava_reference.py on the same input.
final class LavaParityTests: XCTestCase {
    /// 4.0 s of speech band-limited to 8 kHz (192000 samples, a multiple of the hop).
    func testParityBandLimitedCutoff8k() throws {
        try assertParity(input: "speech_8k_band.wav", reference: "ref_8k_band_c8000.f32", cutoffHz: 8000)
    }

    /// 2.5 s of full-band speech, 120077 samples: exercises the pad-to-hop path.
    func testParityFullBandCutoff12k() throws {
        try assertParity(input: "speech_full_band.wav", reference: "ref_full_band_c12000.f32", cutoffHz: 12000)
    }

    func testTenSecondChunkRuntime() throws {
        let enhancer = try LavaEnhancer(weightsURL: try LavaTestFixtures.weightsURL())
        let speech = try LavaTestFixtures.readWAV("speech_8k_band.wav")
        let chunk = (0..<10 * LavaEnhancer.sampleRate).map { speech[$0 % speech.count] }
        // The first call at a new length also compiles MLX's FFT kernels for that size.
        var timings = [TimeInterval]()
        var output = [Float]()
        for _ in 0..<2 {
            let start = Date()
            output = try enhancer.enhance(chunk, cutoffHz: 8000)
            timings.append(Date().timeIntervalSince(start))
        }
        print("LavaSR enhance of a 10 s chunk: first call \(String(format: "%.3f", timings[0])) s, "
              + "repeat \(String(format: "%.3f", timings[1])) s")
        XCTAssertEqual(output.count, chunk.count)
        XCTAssertTrue(output.allSatisfy(\.isFinite))
    }

    func testShortInputsKeepTheirLength() throws {
        let enhancer = try LavaEnhancer(weightsURL: try LavaTestFixtures.weightsURL())
        let speech = try LavaTestFixtures.readWAV("speech_8k_band.wav")
        for count in [1, 511, 512, 769, 1024, 1025] {
            let output = try enhancer.enhance(Array(speech[20_000 ..< 20_000 + count]), cutoffHz: 8000)
            XCTAssertEqual(output.count, count)
            XCTAssertTrue(output.allSatisfy(\.isFinite))
        }
        XCTAssertEqual(try enhancer.enhance([], cutoffHz: 8000), [])
    }

    func testRejectsInvalidCutoff() throws {
        let enhancer = try LavaEnhancer(weightsURL: try LavaTestFixtures.weightsURL())
        for cutoff in [0, -1, 24_001, .nan, .infinity] as [Double] {
            XCTAssertThrowsError(try enhancer.enhance([0, 0, 0], cutoffHz: cutoff)) { error in
                guard case LavaEnhancerError.invalidCutoff = error else {
                    return XCTFail("unexpected error \(error)")
                }
            }
        }
    }

    /// The gate is the contract's 1e-3 relative RMSE on the whole output. Most
    /// of the output's energy is the input passed through below the cutoff, so
    /// the model's contribution (output minus input) is also held to 1e-3
    /// against the reference's contribution.
    private func assertParity(input: String, reference: String, cutoffHz: Double) throws {
        let enhancer = try LavaEnhancer(weightsURL: try LavaTestFixtures.weightsURL())
        let samples = try LavaTestFixtures.readWAV(input)
        let expected = try LavaTestFixtures.readFloat32(reference)
        XCTAssertEqual(expected.count, samples.count)

        let actual = try enhancer.enhance(samples, cutoffHz: cutoffHz)
        XCTAssertEqual(actual.count, samples.count)
        XCTAssertTrue(actual.allSatisfy(\.isFinite))

        let error = Self.relativeRMSE(actual, expected)
        let contributionError = Self.relativeRMSE(
            zip(actual, samples).map { $0 - $1 },
            zip(expected, samples).map { $0 - $1 }
        )
        print("LavaSR parity \(input) @ \(Int(cutoffHz)) Hz: relative RMSE \(error), model contribution \(contributionError)")
        XCTAssertLessThanOrEqual(error, 1e-3)
        XCTAssertLessThanOrEqual(contributionError, 1e-3)
    }

    static func relativeRMSE(_ a: [Float], _ reference: [Float]) -> Double {
        var errorEnergy = 0.0, refEnergy = 0.0
        for (x, r) in zip(a, reference) {
            errorEnergy += Double(x - r) * Double(x - r)
            refEnergy += Double(r) * Double(r)
        }
        return (errorEnergy / refEnergy).squareRoot()
    }
}

enum LavaTestFixtures {
    static let directory = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Fixtures/lava")

    static func weightsURL() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment["MICPOLISH_LAVA_WEIGHTS"],
              FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("Set MICPOLISH_LAVA_WEIGHTS to the LavaSR enhancer_v2 model.safetensors")
        }
        return URL(fileURLWithPath: path)
    }

    static func readWAV(_ name: String) throws -> [Float] {
        let file = try AVAudioFile(forReading: directory.appendingPathComponent(name))
        XCTAssertEqual(file.processingFormat.sampleRate, Double(LavaEnhancer.sampleRate))
        XCTAssertEqual(file.processingFormat.channelCount, 1)
        // AVAudioFile.read(into:) may return fewer frames than requested; read to EOF.
        var samples = [Float]()
        let buffer = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: 65_536))
        while file.framePosition < file.length {
            try file.read(into: buffer)
            samples += UnsafeBufferPointer(start: buffer.floatChannelData![0], count: Int(buffer.frameLength))
        }
        return samples
    }

    static func readFloat32(_ name: String) throws -> [Float] {
        try Data(contentsOf: directory.appendingPathComponent(name)).withUnsafeBytes {
            Array($0.bindMemory(to: Float.self))
        }
    }
}
