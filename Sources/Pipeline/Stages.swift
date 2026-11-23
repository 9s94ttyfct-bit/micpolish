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

    /// RE-USE is trained for 8–48 kHz and its memory grows with the rate, so
    /// other rates are converted first: above 48 kHz down to 48 kHz, below
    /// 8 kHz up to 16 kHz.
    public func outputSampleRate(forInput sampleRate: Int) -> Int { Self.modelSampleRate(forInput: sampleRate) }

    public static func modelSampleRate(forInput sampleRate: Int) -> Int {
        sampleRate > AudioCore.sampleRate ? AudioCore.sampleRate : (sampleRate < 8_000 ? 16_000 : sampleRate)
    }

    public func run(input: StageIO, output: URL, progress: @Sendable (Double) -> Void) async throws {
        let reader = try MonoFileReader(url: input.url)
        let rate = Self.modelSampleRate(forInput: input.sampleRate)
        let resampler = rate == input.sampleRate ? nil : try StreamingResampler(from: input.sampleRate, to: rate)
        let writer = try MonoFileWriter(url: output, sampleRate: rate, format: .cafFloat32)
        let session = try await enhancer.makeSession(sampleRate: rate, settings: settings, backend: backend)
        let total = max(1, Double(reader.frameCount))
        var consumed = 0
        while let block = try reader.read(maxFrames: session.recommendedInputFrameCount) {
            try Task.checkCancellation()
            try writer.append(try await session.append(try resampler?.process(block) ?? block))
            consumed += block.count
            progress(Double(consumed) / total)
        }
        if let resampler { try writer.append(try await session.append(try resampler.flush())) }
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
