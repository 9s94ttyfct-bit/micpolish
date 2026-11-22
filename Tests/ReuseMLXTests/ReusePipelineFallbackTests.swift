import CoreML
import XCTest
@testable import ReuseMLX

/// Session pipeline order and fallback with a fake model that echoes its
/// input, so any lost, duplicated or reordered window shows in the output.
/// Core ML runs for real (zeros through the installed packages).
final class ReusePipelineFallbackTests: XCTestCase {
    func testPipelineKeepsWindowOrder() async throws {
        let (output, input, usage, failed) = try await run(failure: nil)
        XCTAssertEqual(output.count, input.count)
        XCTAssertLessThan(Self.maxAbsDifference(output, input), 1e-5)
        XCTAssertEqual(usage, ReuseBackendUsage(aneHybridWindows: 5, mlxWindows: 1))
        XCTAssertFalse(failed)
    }

    /// The third Mamba call (window 3) throws: window 1 already completed in
    /// the pipeline, windows 2–4 in flight re-run in MLX, as does the rest.
    func testMLXStageFailureFallsBackInOrder() async throws {
        let (output, input, usage, failed) = try await run(failure: .mamba(call: 3))
        XCTAssertLessThan(Self.maxAbsDifference(output, input), 1e-5)
        XCTAssertEqual(usage, ReuseBackendUsage(aneHybridWindows: 1, mlxWindows: 5))
        XCTAssertTrue(failed)
    }

    /// Core ML rejects window 2's encoder input: windows 1 and 2 re-run in MLX.
    func testCoreMLFailureFallsBackInOrder() async throws {
        let (output, input, usage, failed) = try await run(failure: .encoderInput(call: 2))
        XCTAssertLessThan(Self.maxAbsDifference(output, input), 1e-5)
        XCTAssertEqual(usage, ReuseBackendUsage(aneHybridWindows: 0, mlxWindows: 6))
        XCTAssertTrue(failed)
    }

    private func run(
        failure: EchoPipeline.Failure?
    ) async throws -> ([Float], [Float], ReuseBackendUsage, Bool) {
        let fake = EchoPipeline(accelerator: try ReuseANETests.accelerator(), failure: failure)
        let session = try MambaEnhancementSession(
            enhancer: fake, sampleRate: 48_000, settings: .standard, backend: .aneHybrid
        )
        // Five full windows and a tail.
        let input = (0 ..< 1_300_000).map { Float(sin(Double($0) * 0.001)) * 0.5 }
        var output = [Float]()
        for start in stride(from: 0, to: input.count, by: 100_000) {
            output += try await session.append(Array(input[start ..< min(input.count, start + 100_000)]))
        }
        output += try await session.finish()
        return (output, input, await session.backendUsage, await fake.failed)
    }

    static func maxAbsDifference(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count else { return .infinity }
        return zip(a, b).map { abs($0 - $1) }.max() ?? 0
    }
}

/// Echoes every window. Hybrid outputs come back from a FIFO of encoded
/// windows, so they must be requested in window order.
private actor EchoPipeline: HybridWindowPipelining {
    enum Failure { case mamba(call: Int), encoderInput(call: Int) }

    nonisolated let accelerator: ReuseANEAccelerator?
    private let failure: Failure?
    private var encoded: [[Float]] = []
    private var mambaCalls = 0
    private var encoderCalls = 0
    private(set) var failed = false

    init(accelerator: ReuseANEAccelerator, failure: Failure?) {
        self.accelerator = accelerator
        self.failure = failure
    }

    func enhanceWindow(
        _ samples: [Float], sampleRate: Int, backend: ReuseBackend
    ) -> (samples: [Float], backend: ReuseBackend) {
        (samples, .mlx)
    }

    func hybridEligible(sampleRate: Int, count: Int) -> Bool {
        !failed && sampleRate == 48_000 && count == ReuseANEAccelerator.windowSamples
    }

    func hybridEncoderInput(_ samples: [Float], sampleRate: Int) throws -> CoreMLTensor {
        encoderCalls += 1
        encoded.append(samples)
        if case .encoderInput(let call) = failure, call == encoderCalls {
            return try Self.zeros([1, 2, 10, 10])
        }
        return try Self.zeros([1, 2, ReuseANEAccelerator.frames, ReuseANEAccelerator.frequencyBins])
    }

    func hybridMamba(_ latent: CoreMLTensor) throws -> CoreMLTensor {
        mambaCalls += 1
        if case .mamba(let call) = failure, call == mambaCalls {
            throw InferenceError.inferenceFailed("injected")
        }
        XCTAssertEqual(latent.value.shape.map(\.intValue), ReuseANEAccelerator.latentShape)
        return try Self.zeros(ReuseANEAccelerator.latentShape)
    }

    func hybridOutput(_ decoded: CoreMLTensor, sampleCount: Int, sampleRate: Int) -> [Float] {
        XCTAssertEqual(decoded.value.shape.map(\.intValue), ReuseANEAccelerator.decodedShape)
        let window = encoded.removeFirst()
        XCTAssertEqual(window.count, sampleCount)
        return window
    }

    func hybridFailed(_ error: any Error) {
        failed = true
    }

    static func zeros(_ shape: [Int]) throws -> CoreMLTensor {
        let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float16)
        array.withUnsafeMutableBytes { bytes, _ in _ = bytes.initializeMemory(as: UInt8.self, repeating: 0) }
        return CoreMLTensor(value: array)
    }
}
