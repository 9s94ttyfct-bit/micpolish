import AVFoundation
import Darwin
import MLX
import XCTest
@testable import ReuseMLX

/// The `.aneHybrid` backend: fallbacks, parity with MLX on long real speech,
/// and the speed/memory benchmark.
///
/// Env:
/// - `MICPOLISH_REUSE_SOURCE`: NemotronLabs `model.safetensors` (as the other tests).
/// - `MICPOLISH_REUSE_COREML`: Core ML package directory (default: the installed
///   `reuse-nemotron-coreml/<revision>` under Application Support).
/// - `MICPOLISH_SPEECH_48K`: ≥ 60 s mono 48 kHz speech (default `/tmp/speech64s.wav`).
/// - `MICPOLISH_REUSE_BENCH`: `mlx`, `aneHybrid` or `both` runs the benchmark.
final class ReuseANETests: XCTestCase {
    static let revision = "31f82f2efad1095034492f236e8dac9239ba713b"

    func testMissingPackageThrows() throws {
        let missing = FileManager.default.temporaryDirectory.appendingPathComponent("no-such-reuse-coreml-\(UUID())")
        XCTAssertThrowsError(try ReuseANEAccelerator(packageDirectory: missing)) { error in
            guard case ReuseANEError.packageMissing = error else {
                return XCTFail("expected packageMissing, got \(error)")
            }
        }
    }

    func testIncompatibleManifestThrows() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("reuse-coreml-\(UUID())")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let manifest = """
        {"formatVersion": 1, "modelID": "reuse-nemotron", "revision": "x", "sampleRate": 16000,
         "windowSamples": 80000, "packages": {
           "encoder": {"file": "ReuseEncoder.mlpackage", "sha256": "0",
                       "input": {"name": "x", "shape": [1, 2, 1001, 321]}, "output": {"name": "h", "shape": [1]}},
           "decoders": {"file": "ReuseDecoders.mlpackage", "sha256": "0",
                       "input": {"name": "h", "shape": [1]}, "output": {"name": "y", "shape": [1]}}}}
        """
        try Data(manifest.utf8).write(to: directory.appendingPathComponent("manifest.json"))
        XCTAssertThrowsError(try ReuseANEAccelerator(packageDirectory: directory)) { error in
            guard case ReuseANEError.incompatiblePackage = error else {
                return XCTFail("expected incompatiblePackage, got \(error)")
            }
        }
    }

    /// No accelerator: an `.aneHybrid` session runs every window in MLX.
    func testHybridSessionWithoutAcceleratorUsesMLX() async throws {
        let enhancer = try MambaEnhancer(modelWeightsURL: try ReuseTestModel.convertedWeights())
        let input = Self.testTone(count: 300_000, sampleRate: 48_000)
        let session = try await enhancer.makeSession(sampleRate: 48_000, settings: .standard, backend: .aneHybrid)
        var output = try await session.append(input)
        output += try await session.finish()
        XCTAssertEqual(output.count, input.count)
        let usage = await session.backendUsage
        XCTAssertEqual(usage, ReuseBackendUsage(aneHybridWindows: 0, mlxWindows: 2))
        XCTAssertEqual(usage.backend, .mlx)
    }

    /// With an accelerator: 16 kHz runs in MLX; at 48 kHz full windows use
    /// Core ML and the short tail window MLX.
    func testHybridRoutesByRateAndWindowLength() async throws {
        let accelerator = try Self.accelerator()
        let enhancer = try MambaEnhancer(modelWeightsURL: try ReuseTestModel.convertedWeights(), accelerator: accelerator)

        let at16k = try await enhancer.makeSession(sampleRate: 16_000, settings: .standard, backend: .aneHybrid)
        var output16 = try await at16k.append(Self.testTone(count: 100_000, sampleRate: 16_000))
        output16 += try await at16k.finish()
        XCTAssertEqual(output16.count, 100_000)
        let usage16 = await at16k.backendUsage
        XCTAssertEqual(usage16, ReuseBackendUsage(aneHybridWindows: 0, mlxWindows: 2))

        let input = Self.testTone(count: 300_000, sampleRate: 48_000)
        let at48k = try await enhancer.makeSession(sampleRate: 48_000, settings: .standard, backend: .aneHybrid)
        var output48 = try await at48k.append(input)
        output48 += try await at48k.finish()
        XCTAssertEqual(output48.count, input.count)
        XCTAssertTrue(output48.allSatisfy(\.isFinite))
        let usage48 = await at48k.backendUsage
        XCTAssertEqual(usage48, ReuseBackendUsage(aneHybridWindows: 1, mlxWindows: 1))
        XCTAssertEqual(usage48.backend, .aneHybrid)

        // The same enhancer still serves plain MLX sessions.
        let mlx = try await enhancer.makeSession(sampleRate: 48_000, settings: .standard, backend: .mlx)
        _ = try await mlx.append(input)
        _ = try await mlx.finish()
        let usageMLX = await mlx.backendUsage
        XCTAssertEqual(usageMLX, ReuseBackendUsage(aneHybridWindows: 0, mlxWindows: 2))
    }

    /// Quality gate (spec §5.6): ≥ 60 s of 48 kHz speech, waveform relative
    /// RMSE of the hybrid output against MLX FP32 ≤ 1 %.
    func testHybridMatchesMLXOnLongSpeech() async throws {
        let accelerator = try Self.accelerator()
        let speech = try Self.speech()
        XCTAssertGreaterThanOrEqual(speech.count, 60 * 48_000)
        let enhancer = try MambaEnhancer(modelWeightsURL: try ReuseTestModel.convertedWeights(), accelerator: accelerator)

        let (reference, _) = try await Self.run(enhancer, speech, backend: .mlx)
        let (hybrid, usage) = try await Self.run(enhancer, speech, backend: .aneHybrid)
        XCTAssertEqual(hybrid.count, reference.count)
        XCTAssertGreaterThan(usage.aneHybridWindows, 0)

        let error = ReuseParityTests.relativeRMSE(hybrid, reference)
        // Loudest 1 s error, in dBFS (relative error is meaningless in near-silent seconds).
        var worstErrorDB = -Double.infinity
        for start in stride(from: 0, to: reference.count, by: 48_000) {
            let end = min(reference.count, start + 48_000)
            let energy = zip(hybrid[start ..< end], reference[start ..< end])
                .reduce(0.0) { $0 + Double($1.0 - $1.1) * Double($1.0 - $1.1) }
            worstErrorDB = max(worstErrorDB, 10 * log10(max(energy / Double(end - start), 1e-30)))
        }
        print("RE-USE ANE hybrid vs MLX FP32 on \(Double(speech.count) / 48_000) s: rel. RMSE \(error) (SNR \(-20 * log10(error)) dB), loudest 1 s error \(worstErrorDB) dBFS, usage \(usage)")
        if let path = ProcessInfo.processInfo.environment["MICPOLISH_REUSE_DUMP"] {
            try Self.writeWAV(hybrid, to: URL(fileURLWithPath: path + "-hybrid.wav"))
            try Self.writeWAV(reference, to: URL(fileURLWithPath: path + "-mlx.wav"))
        }
        XCTAssertLessThanOrEqual(error, 0.01)
    }

    func testReportComputePlacement() async throws {
        let accelerator = try Self.accelerator()
        let placement = try await accelerator.computePlacement()
        print("RE-USE Core ML placement (\(accelerator.computeUnitsDescription)): \(placement)")
        XCTAssertFalse(placement.encoder.isEmpty)
        XCTAssertFalse(placement.decoders.isEmpty)
    }

    /// Speed/memory benchmark: warm-up run, then the median of three timed runs
    /// over the whole speech file. Run one backend per process for separate
    /// peak-memory figures.
    func testBenchmark() async throws {
        guard let which = ProcessInfo.processInfo.environment["MICPOLISH_REUSE_BENCH"] else {
            throw XCTSkip("Set MICPOLISH_REUSE_BENCH=mlx|aneHybrid|both")
        }
        let backends: [ReuseBackend] = which == "both" ? [.mlx, .aneHybrid] : [try XCTUnwrap(ReuseBackend(rawValue: which))]
        let speech = try Self.speech()
        let seconds = Double(speech.count) / 48_000
        let accelerator = backends.contains(.aneHybrid) ? try Self.accelerator() : nil
        let enhancer = try MambaEnhancer(modelWeightsURL: try ReuseTestModel.convertedWeights(), accelerator: accelerator)

        for backend in backends {
            MLX.Memory.peakMemory = 0
            _ = try await Self.run(enhancer, Array(speech.prefix(2 * ReuseANEAccelerator.windowSamples)), backend: backend)
            var times = [Double]()
            for _ in 0 ..< 3 {
                let start = ContinuousClock.now
                let (_, usage) = try await Self.run(enhancer, speech, backend: backend)
                let elapsed = ContinuousClock.now - start
                times.append(Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18)
                XCTAssertEqual(usage.backend, backend)
            }
            let median = times.sorted()[1]
            print(String(
                format: "RE-USE bench %@: %.1f s audio, runs %@ s, median %.2f s, RTF %.3f, MLX peak %.0f MB, process peak footprint %.0f MB",
                backend.rawValue, seconds, times.map { String(format: "%.2f", $0) }.joined(separator: "/"),
                median, median / seconds, Double(MLX.Memory.peakMemory) / 1_048_576, Self.peakFootprintMB()
            ))
        }
    }

    // MARK: Helpers

    static func accelerator() throws -> ReuseANEAccelerator {
        let environment = ProcessInfo.processInfo.environment
        let directory = environment["MICPOLISH_REUSE_COREML"].map { URL(fileURLWithPath: $0) }
            ?? ReuseANEAccelerator.packageDirectory(
                modelsRoot: FileManager.default.homeDirectoryForCurrentUser
                    .appendingPathComponent("Library/Application Support/MicPolish/models"),
                modelID: "reuse-nemotron", revision: revision
            )
        guard FileManager.default.fileExists(atPath: directory.appendingPathComponent("manifest.json").path) else {
            throw XCTSkip("No RE-USE Core ML package at \(directory.path); run tools/coreml/export_reuse_coreml.py")
        }
        return try ReuseANEAccelerator(packageDirectory: directory)
    }

    static func speech() throws -> [Float] {
        let path = ProcessInfo.processInfo.environment["MICPOLISH_SPEECH_48K"] ?? "/tmp/speech64s.wav"
        guard FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("Set MICPOLISH_SPEECH_48K to ≥ 60 s of 48 kHz mono speech")
        }
        let file = try AVAudioFile(forReading: URL(fileURLWithPath: path))
        guard Int(file.processingFormat.sampleRate) == 48_000 else {
            throw XCTSkip("\(path) is not 48 kHz")
        }
        var samples = [Float]()
        let buffer = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: 65_536))
        while file.framePosition < file.length {
            try file.read(into: buffer)
            samples += UnsafeBufferPointer(start: buffer.floatChannelData![0], count: Int(buffer.frameLength))
        }
        return samples
    }

    static func run(
        _ enhancer: MambaEnhancer, _ input: [Float], backend: ReuseBackend
    ) async throws -> ([Float], ReuseBackendUsage) {
        let session = try await enhancer.makeSession(sampleRate: 48_000, settings: .standard, backend: backend)
        var output = [Float]()
        var offset = 0
        while offset < input.count {
            let end = min(input.count, offset + session.recommendedInputFrameCount)
            output += try await session.append(Array(input[offset ..< end]))
            offset = end
        }
        output += try await session.finish()
        return (output, await session.backendUsage)
    }

    static func testTone(count: Int, sampleRate: Int) -> [Float] {
        (0 ..< count).map { i in
            let t = Double(i) / Double(sampleRate)
            return Float(0.3 * sin(2 * .pi * 220 * t) + 0.02 * sin(2 * .pi * 3_100 * t + 0.4))
        }
    }

    static func peakFootprintMB() -> Double {
        var info = rusage_info_v4()
        let status = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) {
                proc_pid_rusage(getpid(), RUSAGE_INFO_V4, $0)
            }
        }
        return status == 0 ? Double(info.ri_lifetime_max_phys_footprint) / 1_048_576 : .nan
    }

    static func writeWAV(_ samples: [Float], to url: URL) throws {
        let format = AVAudioFormat(standardFormatWithSampleRate: 48_000, channels: 1)!
        let file = try AVAudioFile(forWriting: url, settings: format.settings)
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count))!
        buffer.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { buffer.floatChannelData![0].update(from: $0.baseAddress!, count: samples.count) }
        try file.write(from: buffer)
    }
}
