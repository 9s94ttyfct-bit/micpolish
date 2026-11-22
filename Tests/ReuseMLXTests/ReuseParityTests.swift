import AVFoundation
import XCTest
@testable import ReuseMLX

/// Compares one full-window MLX forward pass with the PyTorch CPU reference
/// produced by tools/parity/reuse_reference.py on the same input.
final class ReuseParityTests: XCTestCase {
    func testParity48k() async throws {
        // The 48 kHz fixture is upsampled from a 44.1 kHz source, so bins above
        // 22.05 kHz hold near-zero energy and their phase is set by float32 FFT
        // rounding. Layer probes (ReuseProbeDiagnostics) show STFT magnitude
        // within 4e-7 and model output within 1e-5; the waveform residual is
        // about 1.2e-4, i.e. -78 dB.
        try await assertParity(input: "input_48k.wav", reference: "ref_48k.f32", sampleRate: 48_000, tolerance: 2e-4)
    }

    func testParity16k() async throws {
        try await assertParity(input: "input_16k.wav", reference: "ref_16k.f32", sampleRate: 16_000, tolerance: 1e-4)
    }

    private func assertParity(input: String, reference: String, sampleRate: Int, tolerance: Double) async throws {
        let enhancer = try MambaEnhancer(modelWeightsURL: try ReuseTestModel.convertedWeights())
        let fixtures = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Fixtures/reuse")

        let file = try AVAudioFile(forReading: fixtures.appendingPathComponent(input))
        XCTAssertEqual(Int(file.processingFormat.sampleRate), sampleRate)
        // AVAudioFile.read(into:) may return fewer frames than requested; read to EOF.
        var samples = [Float]()
        let buffer = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: 65_536))
        while file.framePosition < file.length {
            try file.read(into: buffer)
            samples += UnsafeBufferPointer(start: buffer.floatChannelData![0], count: Int(buffer.frameLength))
        }

        let refData = try Data(contentsOf: fixtures.appendingPathComponent(reference))
        let expected = refData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        XCTAssertEqual(expected.count, samples.count)

        let actual = try await enhancer.enhanceWindow(samples, sampleRate: sampleRate)
        XCTAssertEqual(actual.count, expected.count)

        var errorEnergy = 0.0, refEnergy = 0.0
        for (a, e) in zip(actual, expected) {
            errorEnergy += Double(a - e) * Double(a - e)
            refEnergy += Double(e) * Double(e)
        }
        let relativeRMSE = (errorEnergy / refEnergy).squareRoot()
        print("RE-USE parity @\(sampleRate) Hz: relative RMSE = \(relativeRMSE)")
        XCTAssertLessThanOrEqual(relativeRMSE, tolerance)
    }
}
