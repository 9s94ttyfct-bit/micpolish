import MLX
import XCTest
@testable import LavaMLX

/// Weight-free checks of the spectral front/back end and the crossover.
final class LavaSpectralTests: XCTestCase {
    private func periodicHann(_ length: Int) -> [Float] {
        (0..<length).map { 0.5 - 0.5 * cos(2 * Float.pi * Float($0) / Float(length)) }
    }

    private func noise(_ count: Int, seed: UInt64) -> [Float] {
        var state = seed
        return (0..<count).map { _ in
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float(Int64(bitPattern: state >> 11) % 20_000) / 20_000 - 0.5
        }
    }

    func testPaddedCountIsHopMultipleOfAtLeastTwoHops() {
        XCTAssertEqual(LavaSpectral.paddedCount(for: 1), 1024)
        XCTAssertEqual(LavaSpectral.paddedCount(for: 512), 1024)
        XCTAssertEqual(LavaSpectral.paddedCount(for: 1024), 1024)
        XCTAssertEqual(LavaSpectral.paddedCount(for: 1025), 1536)
        XCTAssertEqual(LavaSpectral.paddedCount(for: 192_000), 192_000)
        XCTAssertEqual(LavaSpectral.paddedCount(for: 120_077), 120_320)
    }

    /// The mel extractor's framing (reflect pad 768, hop 512) and the "same"
    /// ISTFT (overlap-add, trim 768, divide by the window envelope) are exact
    /// inverses over the whole signal, edges included.
    func testSTFTThenISTFTReconstructsSignal() {
        let signal = noise(512 * 20, seed: 7)
        let reconstructed: [Float] = Device.withDefaultDevice(.gpu) {
            let window = MLXArray(periodicHann(2048))
            let spectrum = LavaSpectral.stft(MLXArray(signal), window: window)
            XCTAssertEqual(spectrum.shape, [20, 1025])
            let output = LavaSpectral.istft(spectrum, window: window)
            MLX.eval(output)
            return output.asArray(Float.self)
        }
        XCTAssertEqual(reconstructed.count, signal.count)
        let maxError = zip(reconstructed, signal).map { abs($0 - $1) }.max() ?? .infinity
        XCTAssertLessThan(maxError, 1e-5)
    }

    func testFadeTemplateIsSmoothstep() {
        let fade = LavaCrossover.fadeTemplate
        XCTAssertEqual(fade.count, 1024)
        XCTAssertEqual(fade.first, 0)
        XCTAssertEqual(fade.last, 1)
        XCTAssertEqual(fade[511] + fade[512], 1, accuracy: 1e-6)
        XCTAssertTrue(zip(fade, fade.dropFirst()).allSatisfy { $0 <= $1 })
    }

    func testMaskPlacesTransitionAroundCutoffBin() {
        // 192000 samples: 96001 bins, cutoff bin int(8000 / 24000 * 96001) = 32000.
        let mask = LavaCrossover.mask(binCount: 96_001, cutoffHz: 8000)
        XCTAssertEqual(mask.count, 96_001)
        XCTAssertTrue(mask[..<31_488].allSatisfy { $0 == 0 })
        XCTAssertEqual(Array(mask[31_488 ..< 32_512]), LavaCrossover.fadeTemplate)
        XCTAssertTrue(mask[32_512...].allSatisfy { $0 == 1 })

        // A transition clipped at the low edge uses the start of the fade template.
        let low = LavaCrossover.mask(binCount: 4097, cutoffHz: 1000)  // cutoff bin 170
        XCTAssertEqual(Array(low[0 ..< 682]), Array(LavaCrossover.fadeTemplate[0 ..< 682]))
        XCTAssertTrue(low[682...].allSatisfy { $0 == 1 })
    }

    func testMergeKeepsInputBelowCutoffAndModelAbove() {
        let count = 48_000
        let low = (0..<count).map { Float(sin(2 * Double.pi * 1000 * Double($0) / 48_000)) }
        let high = (0..<count).map { Float(0.5 * sin(2 * Double.pi * 15_000 * Double($0) / 48_000)) }
        let merged: [Float] = Device.withDefaultDevice(.gpu) {
            // Input carries only the low tone, the "model" only the high tone.
            let output = LavaCrossover.merge(model: MLXArray(high), input: MLXArray(low), cutoffHz: 8000)
            MLX.eval(output)
            return output.asArray(Float.self)
        }
        let expected = zip(low, high).map { $0 + $1 }
        XCTAssertEqual(merged.count, count)
        let maxError = zip(merged, expected).map { abs($0 - $1) }.max() ?? .infinity
        XCTAssertLessThan(maxError, 1e-4)
    }

    func testMissingWeightsFileThrows() {
        let url = URL(fileURLWithPath: "/nonexistent/lava.safetensors")
        XCTAssertThrowsError(try LavaEnhancer(weightsURL: url)) { error in
            XCTAssertEqual(error as? LavaEnhancerError, .weightsNotFound(url))
        }
    }

    func testIncompleteCheckpointThrows() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("lava-incomplete-\(ProcessInfo.processInfo.processIdentifier).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try MLX.save(arrays: [
            "feature_extractor.mel_spec.spectrogram.window": MLXArray(periodicHann(2048)),
            "feature_extractor.mel_spec.mel_scale.fb": MLXArray.zeros([1025, 40]),
        ], url: url)
        XCTAssertThrowsError(try LavaEnhancer(weightsURL: url)) { error in
            XCTAssertEqual(
                error as? LavaEnhancerError,
                .unexpectedShape(key: "feature_extractor.mel_spec.mel_scale.fb", actual: [1025, 40], expected: [1025, 80])
            )
        }
    }
}
