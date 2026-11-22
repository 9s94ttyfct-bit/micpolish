import AVFoundation
import XCTest
@testable import ReuseMLX

/// Compares one full-window MLX forward pass with the PyTorch CPU reference
/// produced by tools/parity/reuse_reference.py on the same input.
final class ReuseParityTests: XCTestCase {
    func testParity48k() async throws {
        try await assertParity(input: "input_48k.wav", sampleRate: 48_000)
    }

    func testParity16k() async throws {
        try await assertParity(input: "input_16k.wav", sampleRate: 16_000)
    }

    /// The gate is the plan's 1e-4 relative RMSE, measured against the
    /// float64 reference and widened only to twice the error PyTorch's own
    /// float32 run has against that reference: no implementation can be held
    /// to a tighter bound than the reference's float32 rounding. At 48 kHz the
    /// fixture's empty band above 22.05 kHz makes phase ill-conditioned and
    /// PyTorch float32 itself is 8.5e-5 off; at 16 kHz it is 1e-6 off.
    private func assertParity(input: String, sampleRate: Int) async throws {
        let enhancer = try MambaEnhancer(modelWeightsURL: try ReuseTestModel.convertedWeights())
        let fixtures = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Fixtures/reuse")
        let tag = "\(sampleRate / 1000)k"

        let file = try AVAudioFile(forReading: fixtures.appendingPathComponent(input))
        XCTAssertEqual(Int(file.processingFormat.sampleRate), sampleRate)
        // AVAudioFile.read(into:) may return fewer frames than requested; read to EOF.
        var samples = [Float]()
        let buffer = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: 65_536))
        while file.framePosition < file.length {
            try file.read(into: buffer)
            samples += UnsafeBufferPointer(start: buffer.floatChannelData![0], count: Int(buffer.frameLength))
        }

        func load(_ name: String) throws -> [Float] {
            try Data(contentsOf: fixtures.appendingPathComponent(name)).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        }
        let ref32 = try load("ref_\(tag).f32")
        let ref64 = try load("ref64_\(tag).f32")
        XCTAssertEqual(ref64.count, samples.count)

        let actual = try await enhancer.enhanceWindow(samples, sampleRate: sampleRate)
        XCTAssertEqual(actual.count, ref64.count)

        let mlxError = Self.relativeRMSE(actual, ref64)
        let torchError = Self.relativeRMSE(ref32, ref64)
        let tolerance = max(1e-4, 2 * torchError)
        print("RE-USE parity @\(sampleRate) Hz vs float64: MLX \(mlxError), PyTorch float32 \(torchError), tolerance \(tolerance); MLX vs PyTorch float32 \(Self.relativeRMSE(actual, ref32))")
        XCTAssertLessThanOrEqual(mlxError, tolerance)
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
