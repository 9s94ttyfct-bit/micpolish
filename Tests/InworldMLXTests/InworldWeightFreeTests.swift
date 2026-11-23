import Foundation
import MLX
import XCTest
@testable import InworldMLX

/// Checks that need no released weights: spectral round trip, window
/// stitching, checkpoint validation and a tiny random model end to end.
final class InworldWeightFreeTests: XCTestCase {
    private let releasedWindow = (chunk: 384_000, overlap: 96_000)

    // MARK: Config

    func testReleasedConfigValues() throws {
        // The `config` metadata of voice-isolator-v1 (denoiser.safetensors @ 9b0285b3).
        let json = """
        {"band_plan": [[100, 1000], [250, 4000], [500, 8000], [1000, 16000], [2000, 24000]], \
        "block_types": ["transformer", "mamba", "transformer", "mamba", "transformer", "mamba", "transformer"], \
        "causal": false, "chunk_seconds": 8.0, "correction_bound": 0.5, "dual_mode": false, \
        "exclusive_axes": "time", "exclusive_self_attention": true, "hop_length": 480, "lookahead_frames": 0, \
        "lookahead_frames_choices": [], "mamba_conv": 4, "mamba_expand": 2, "mamba_state": 16, \
        "mode_film": false, "n_fft": 2048, "num_heads": 8, "overlap_seconds": 2.0, "sample_rate": 48000, "width": 512}
        """
        let config = try InworldConfig(json: json)
        XCTAssertEqual(config.window.chunk, releasedWindow.chunk)
        XCTAssertEqual(config.window.overlap, releasedWindow.overlap)
        XCTAssertEqual(config.dtRank, 32)
        XCTAssertEqual(config.mambaInner, 1024)
        XCTAssertTrue(config.exclusiveTime)
        XCTAssertFalse(config.exclusiveBand)
        let edges = config.bandEdges
        XCTAssertEqual(edges.count, 46)
        XCTAssertEqual(edges.first, 0 ..< 4)
        XCTAssertEqual(edges.last, 1024 ..< 1025)
        XCTAssertEqual(edges.map(\.count).max(), 85)
        XCTAssertEqual(edges.reduce(0) { $0 + $1.count }, 1025)
        for (previous, next) in zip(edges, edges.dropFirst()) { XCTAssertEqual(previous.upperBound, next.lowerBound) }
    }

    func testRefusesStreamingAndUnknownConfigs() {
        for (field, value) in [("causal", true), ("dual_mode", true), ("offline_passes", 2), ("mode_film", true),
                               ("lookahead_frames", 2), ("future_feature", 1), ("representation", "waveform")]
            as [(String, Any)]
        {
            var config = TinyCheckpoint.config
            config[field] = value
            XCTAssertThrowsError(try InworldConfig(json: TinyCheckpoint.json(config)), field) { error in
                guard case InworldIsolatorError.unsupportedConfig = error else { return XCTFail("\(error)") }
            }
        }
        var config = TinyCheckpoint.config
        config["sample_rate"] = 44_100
        XCTAssertThrowsError(try InworldConfig(json: TinyCheckpoint.json(config)))
        // Zero-valued unknown fields and training-only fields are ignored, as upstream does.
        config = TinyCheckpoint.config
        config["future_feature"] = 0
        config["grid_width"] = 7
        XCTAssertNoThrow(try InworldConfig(json: TinyCheckpoint.json(config)))
    }

    // MARK: Spectral

    /// `torch.istft(torch.stft(x))` with center=True reproduces `x` exactly for
    /// any length, through both the reflect (n > 1024) and zero padding paths.
    func testSTFTThenISTFTReconstructsSignal() {
        let spectral = InworldSpectral(nFFT: 2048, hop: 480)
        for count in [1, 100, 479, 480, 1024, 1025, 2047, 4800, 48_001] {
            let signal = InworldFixtures.noise(count, seed: UInt64(count))
            let reconstructed: [Float] = Device.withDefaultDevice(.gpu) {
                let window = MLXArray(spectral.window)
                let (re, im) = spectral.stft(MLXArray(signal), window: window)
                XCTAssertEqual(re.shape, [1 + count / 480, 1025])
                let output = spectral.istft(re: re, im: im, window: window, count: count)
                eval(output)
                return output.asArray(Float.self)
            }
            XCTAssertEqual(reconstructed.count, count)
            let maxError = zip(reconstructed, signal).map { abs($0 - $1) }.max() ?? 0
            XCTAssertLessThan(maxError, 2e-6, "count \(count)")
        }
    }

    /// The STFT matches a direct DFT of the reflect-padded, windowed frame.
    func testSTFTMatchesDirectDFT() {
        let spectral = InworldSpectral(nFFT: 2048, hop: 480)
        let count = 3000
        let signal = InworldFixtures.noise(count, seed: 3)
        let window = spectral.window
        let (re, im): ([Float], [Float]) = Device.withDefaultDevice(.gpu) {
            let (re, im) = spectral.stft(MLXArray(signal), window: MLXArray(window))
            eval(re, im)
            return (re.asArray(Float.self), im.asArray(Float.self))
        }
        func reflected(_ index: Int) -> Float {
            let i = index - 1024
            if i < 0 { return signal[-i] }
            if i >= count { return signal[2 * (count - 1) - i] }
            return signal[i]
        }
        for frame in [0, 3, 6] {
            for bin in [0, 1, 17, 512, 1024] {
                var sumRe = 0.0, sumIm = 0.0
                for k in 0..<2048 {
                    let x = Double(reflected(frame * 480 + k)) * Double(window[k])
                    let angle = -2 * Double.pi * Double(bin * k) / 2048
                    sumRe += x * cos(angle)
                    sumIm += x * sin(angle)
                }
                XCTAssertEqual(Double(re[frame * 1025 + bin]), sumRe, accuracy: 2e-4, "frame \(frame) bin \(bin)")
                if bin == 0 || bin == 1024 {
                    XCTAssertEqual(im[frame * 1025 + bin], 0)
                } else {
                    XCTAssertEqual(Double(im[frame * 1025 + bin]), sumIm, accuracy: 2e-4, "frame \(frame) bin \(bin)")
                }
            }
        }
    }

    func testHannWindowIsPeriodic() {
        let window = InworldSpectral(nFFT: 2048, hop: 480).window
        XCTAssertEqual(window.count, 2048)
        XCTAssertEqual(window[0], 0)
        XCTAssertEqual(window[1024], 1, accuracy: 1e-7)
        XCTAssertEqual(window[1], window[2047], accuracy: 1e-7)
    }

    // MARK: Stitching

    func testLinspaceMatchesTorchEndpoints() {
        let ramp = InworldStitcher.linspace(0, 1, 96_000)
        XCTAssertEqual(ramp.first, 0)
        XCTAssertEqual(ramp.last, 1)
        XCTAssertEqual(InworldStitcher.linspace(1, 0, 5), [1, 0.75, 0.5, 0.25, 0])
        XCTAssertEqual(InworldStitcher.linspace(0, 1, 1), [0])
    }

    /// The streaming stitcher reproduces upstream `enhance` bit for bit, for
    /// clip lengths around every window boundary and any piece sizes.
    func testStreamingMatchesWholeClipEnhance() throws {
        // A deterministic, position-dependent stand-in for the network.
        let render: ([Float]) -> [Float] = { chunk in
            chunk.enumerated().map { index, x in x * 0.5 + 0.25 * sin(Float(index) * 0.01) + 0.1 * x * x }
        }
        for (chunk, overlap) in [(40, 10), (37, 18), (12, 0), (releasedWindow.chunk, releasedWindow.overlap)] {
            let stride = chunk - overlap
            var lengths = [1, 2, chunk - 1, chunk, chunk + 1, chunk + stride - 1, chunk + stride, chunk + stride + 1,
                           2 * chunk, chunk + 2 * stride, chunk + 2 * stride + 1, 3 * chunk + 5]
            if chunk < 1000 { lengths += Array(1...(4 * chunk)) }
            for count in Set(lengths) where count > 0 {
                let audio = InworldFixtures.noise(count, seed: UInt64(count))
                let expected = InworldFixtures.enhance(audio, chunk: chunk, overlap: overlap, render: render)
                for pieces in [[count], [1], [7, 1, 480, 1000, 33_333], [chunk], [stride + 1]] {
                    if chunk > 1000 && pieces == [1] { continue }
                    var stitcher = InworldStitcher(chunk: chunk, overlap: overlap, render: render)
                    var output = [Float]()
                    var position = 0, index = 0
                    while position < count {
                        let end = min(count, position + pieces[index % pieces.count])
                        output += try stitcher.push(Array(audio[position ..< end]))
                        // Between pushes at most one window plus one stride of input and one
                        // crossfade of sums stay buffered (two windows), whatever the piece size.
                        XCTAssertLessThanOrEqual(stitcher.bufferedSampleCount, 2 * chunk)
                        position = end
                        index += 1
                    }
                    output += try stitcher.finish()
                    XCTAssertEqual(output.count, count, "chunk \(chunk) count \(count) pieces \(pieces)")
                    XCTAssertEqual(output, expected, "chunk \(chunk) count \(count) pieces \(pieces)")
                }
            }
        }
    }

    /// With an identity network the crossfades sum to one: output equals input.
    func testIdentityRenderKeepsSignalAtReleasedWindow() throws {
        for count in [1, 479, 480, 384_000, 384_001, 672_000, 768_000, 960_000, 1_000_003] {
            let audio = InworldFixtures.noise(count, seed: 11)
            var stitcher = InworldStitcher(
                chunk: releasedWindow.chunk, overlap: releasedWindow.overlap, render: { $0 })
            let output = try stitcher.push(audio) + stitcher.finish()
            XCTAssertEqual(output.count, count)
            let maxError = zip(output, audio).map { abs($0 - $1) }.max() ?? 0
            XCTAssertLessThan(maxError, 1e-6, "count \(count)")
        }
    }

    func testEmptyStreamYieldsNothing() throws {
        var stitcher = InworldStitcher(chunk: 40, overlap: 10, render: { $0 })
        XCTAssertEqual(try stitcher.push([]), [])
        XCTAssertEqual(try stitcher.finish(), [])
    }

    // MARK: Checkpoint

    func testMissingWeightsThrow() {
        let url = URL(fileURLWithPath: "/nonexistent/denoiser.safetensors")
        XCTAssertThrowsError(try InworldIsolator(weightsURL: url)) { error in
            XCTAssertEqual(error as? InworldIsolatorError, .weightsNotFound(url))
        }
    }

    /// An unreadable file reports the loader's error, not a missing config.
    func testCorruptCheckpointThrowsTheLoadError() throws {
        let url = temporaryURL()
        try Data("not a safetensors file".utf8).write(to: url)
        XCTAssertThrowsError(try InworldIsolator(weightsURL: url)) { error in
            XCTAssertNil(error as? InworldIsolatorError)
        }
    }

    func testCheckpointWithoutConfigThrows() throws {
        let url = temporaryURL()
        try MLX.save(arrays: try TinyCheckpoint.tensors(), url: url)
        XCTAssertThrowsError(try InworldIsolator(weightsURL: url)) { error in
            XCTAssertEqual(error as? InworldIsolatorError, .missingTensor("__metadata__.config"))
        }
    }

    func testMissingTensorThrows() throws {
        var tensors = try TinyCheckpoint.tensors()
        tensors["blocks.1.freq_mamba.backward_blocks.A_log"] = nil
        let url = temporaryURL()
        try TinyCheckpoint.write(tensors, to: url)
        XCTAssertThrowsError(try InworldIsolator(weightsURL: url)) { error in
            XCTAssertEqual(error as? InworldIsolatorError, .missingTensor("blocks.1.freq_mamba.backward_blocks.A_log"))
        }
    }

    func testWrongShapeThrows() throws {
        var tensors = try TinyCheckpoint.tensors()
        tensors["band_embedding"] = MLXArray.zeros([11, 17])
        let url = temporaryURL()
        try TinyCheckpoint.write(tensors, to: url)
        XCTAssertThrowsError(try InworldIsolator(weightsURL: url)) { error in
            XCTAssertEqual(error as? InworldIsolatorError, .unexpectedShape("band_embedding", expected: [11, 16], actual: [11, 17]))
        }
    }

    func testSpareTensorThrowsButTrainingExtrasAreIgnored() throws {
        var tensors = try TinyCheckpoint.tensors()
        tensors["discriminator.layer.weight"] = MLXArray.zeros([3])
        let url = temporaryURL()
        try TinyCheckpoint.write(tensors, to: url)
        XCTAssertNoThrow(try InworldIsolator(weightsURL: url))

        tensors["blocks.9.time_transformer.norm1.weight"] = MLXArray.zeros([16])
        try TinyCheckpoint.write(tensors, to: url)
        XCTAssertThrowsError(try InworldIsolator(weightsURL: url)) { error in
            XCTAssertEqual(error as? InworldIsolatorError, .unsupportedConfig("unexpected tensor blocks.9.time_transformer.norm1.weight"))
        }
    }

    /// A tiny random model end to end: lengths kept at every window boundary,
    /// finite output, and the session equal to upstream `enhance` over the
    /// same network.
    func testTinyModelSessionKeepsLengthAndMatchesEnhance() throws {
        let url = temporaryURL()
        try TinyCheckpoint.write(try TinyCheckpoint.tensors(), to: url)
        let isolator = try InworldIsolator(weightsURL: url)
        let (chunk, overlap) = isolator.window
        XCTAssertEqual(chunk, 480)
        XCTAssertEqual(overlap, 96)
        let stride = chunk - overlap
        for count in [1, 15, 16, 17, 32, 33, 100, chunk - 1, chunk, chunk + 1, chunk + stride, chunk + 2 * stride,
                      chunk + 2 * stride + 7, 10 * stride] {
            let audio = InworldFixtures.noise(count, seed: UInt64(count) + 100)
            let output = try InworldFixtures.run(isolator.makeSession(), audio, pieces: [97, 1, 480, 3])
            XCTAssertEqual(output.count, count, "count \(count)")
            XCTAssertTrue(output.allSatisfy(\.isFinite), "count \(count)")
            let expected = InworldFixtures.enhance(audio, chunk: chunk, overlap: overlap, render: isolator.render)
            XCTAssertEqual(output, expected, "count \(count)")
            XCTAssertNotEqual(output, audio, "count \(count)")
        }
    }

    private func temporaryURL() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("inworld-\(UUID().uuidString).safetensors")
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }
}
