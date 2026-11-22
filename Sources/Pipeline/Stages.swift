import AudioCore
import Foundation
import ReuseMLX

/// NVIDIA RE-USE at the input's own sample rate, streamed in model windows.
public struct ReuseStage: AudioStage {
    /// 4 s windows with 5% overlap. At 48 kHz, 5 s windows peak at 3.2 GB of
    /// MLX memory; 4 s windows peak at 2.8 GB at the same speed, and both are
    /// within the model's < 10 s training clip length.
    public static let inferenceSettings = try! InferenceSettings(chunkSeconds: 4.0, overlapPortion: 0.05)

    public let id: String
    public let settingsKey: String
    private let enhancer: MambaEnhancer
    private let settings: InferenceSettings

    public init(enhancer: MambaEnhancer, variantID: String, revision: String,
                settings: InferenceSettings = ReuseStage.inferenceSettings) {
        self.enhancer = enhancer
        self.settings = settings
        id = variantID
        settingsKey = "\(revision)|\(enhancer.mode.rawValue)|\(settings.chunkSeconds)|\(settings.overlapPortion)"
    }

    public func outputSampleRate(forInput sampleRate: Int) -> Int { sampleRate }

    public func run(input: StageIO, output: URL, progress: @Sendable (Double) -> Void) async throws {
        let reader = try MonoFileReader(url: input.url)
        let writer = try MonoFileWriter(url: output, sampleRate: input.sampleRate, format: .cafFloat32)
        let session = try await enhancer.makeSession(sampleRate: input.sampleRate, settings: settings)
        let total = max(1, Double(reader.frameCount))
        var consumed = 0
        while let block = try reader.read(maxFrames: session.recommendedInputFrameCount) {
            try Task.checkCancellation()
            try writer.append(try await session.append(block))
            consumed += block.count
            progress(Double(consumed) / total)
        }
        try writer.append(try await session.finish())
        try writer.close()
        progress(1)
    }
}

/// Plain sample-rate conversion to 48 kHz; skipped for 48 kHz input.
public struct ResampleStage: AudioStage {
    public let id = "resample-48k"
    public let settingsKey = "avaudioconverter-mastering-max"
    public init() {}

    public func outputSampleRate(forInput sampleRate: Int) -> Int { AudioCore.sampleRate }

    public func skips(inputSampleRate: Int) -> Bool { inputSampleRate == AudioCore.sampleRate }

    public func run(input: StageIO, output: URL, progress: @Sendable (Double) -> Void) async throws {
        let reader = try MonoFileReader(url: input.url)
        let writer = try MonoFileWriter(url: output, sampleRate: AudioCore.sampleRate, format: .cafFloat32)
        let resampler = try StreamingResampler(from: input.sampleRate, to: AudioCore.sampleRate)
        let total = max(1, Double(reader.frameCount))
        var consumed = 0
        while let block = try reader.read(maxFrames: 65_536) {
            try Task.checkCancellation()
            try writer.append(try resampler.process(block))
            consumed += block.count
            progress(Double(consumed) / total)
        }
        try writer.append(try resampler.flush())
        try writer.close()
        progress(1)
    }
}
