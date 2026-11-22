import AudioCore
import Foundation
import ReuseMLX

/// NVIDIA RE-USE at the input's own sample rate, streamed in model windows.
public struct ReuseStage: AudioStage {
    /// 5 s windows with 5% overlap (SpeechLens' default): more context per
    /// window, MLX peak 3.2 GB at 48 kHz, within the 5 GB memory budget and the
    /// model's < 10 s training clip length.
    public static let inferenceSettings = try! InferenceSettings(chunkSeconds: 5.0, overlapPortion: 0.05)

    public let id: String
    public let settingsKey: String
    private let enhancer: MambaEnhancer
    private let settings: InferenceSettings
    private let backend: ReuseBackend
    private let onBackendUsage: (@Sendable (ReuseBackendUsage) -> Void)?

    /// `backend` `.aneHybrid` needs an enhancer built with an accelerator; it
    /// is part of the cache key because Core ML FP16 output differs slightly.
    public init(enhancer: MambaEnhancer, variantID: String, revision: String,
                settings: InferenceSettings = ReuseStage.inferenceSettings, backend: ReuseBackend = .mlx,
                onBackendUsage: (@Sendable (ReuseBackendUsage) -> Void)? = nil) {
        self.enhancer = enhancer
        self.settings = settings
        let effective: ReuseBackend = enhancer.accelerator == nil ? .mlx : backend
        self.backend = effective
        self.onBackendUsage = onBackendUsage
        id = variantID
        settingsKey = "\(revision)|\(enhancer.mode.rawValue)|\(settings.chunkSeconds)|\(settings.overlapPortion)"
            + (effective == .aneHybrid ? "|ane" : "")
    }

    public func outputSampleRate(forInput sampleRate: Int) -> Int { sampleRate }

    public func run(input: StageIO, output: URL, progress: @Sendable (Double) -> Void) async throws {
        let reader = try MonoFileReader(url: input.url)
        let writer = try MonoFileWriter(url: output, sampleRate: input.sampleRate, format: .cafFloat32)
        let session = try await enhancer.makeSession(sampleRate: input.sampleRate, settings: settings, backend: backend)
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
        onBackendUsage?(await session.backendUsage)
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
