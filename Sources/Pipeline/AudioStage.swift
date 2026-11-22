import Foundation

/// A cached processing step's input or output.
public struct StageIO: Sendable, Equatable {
    public let url: URL
    public let sampleRate: Int
    public let cacheKey: String

    public init(url: URL, sampleRate: Int, cacheKey: String) {
        self.url = url
        self.sampleRate = sampleRate
        self.cacheKey = cacheKey
    }
}

/// One file-to-file step of the AI chain. Output is mono Float32 CAF.
public protocol AudioStage: Sendable {
    /// Stable identifier; part of the cache key and the cache file name.
    var id: String { get }
    /// Changes whenever the stage would produce different output.
    var settingsKey: String { get }
    func outputSampleRate(forInput sampleRate: Int) -> Int
    /// True when the stage would leave input at this rate unchanged.
    func skips(inputSampleRate: Int) -> Bool
    func run(input: StageIO, output: URL, progress: @Sendable (Double) -> Void) async throws
}

public extension AudioStage {
    func skips(inputSampleRate: Int) -> Bool { false }
}
