import AudioCore
import Foundation
import InworldMLX

/// One streaming pass of a voice isolator over 48 kHz audio.
public protocol IsolationSession: AnyObject {
    func push(_ samples: [Float]) throws -> [Float]
    func finish() throws -> [Float]
}

/// A model that separates the voice from background noise at 48 kHz.
public protocol VoiceIsolating: Sendable {
    func makeIsolationSession() -> any IsolationSession
}

extension InworldSession: IsolationSession {}

extension InworldIsolator: VoiceIsolating {
    public func makeIsolationSession() -> any IsolationSession { makeSession() }
}

/// Inworld Voice Isolator (offline) ahead of RE-USE, for heavy background
/// noise. Runs at 48 kHz: other rates are resampled first, so the stages after
/// it see 48 kHz input.
public struct InworldStage: AudioStage {
    public let id = "inworld-v1"
    public let settingsKey: String
    private let isolator: any VoiceIsolating

    public init(isolator: any VoiceIsolating, revision: String) {
        self.isolator = isolator
        settingsKey = "\(revision)|offline"
    }

    public func outputSampleRate(forInput sampleRate: Int) -> Int { AudioCore.sampleRate }

    public func run(input: StageIO, output: URL, progress: @Sendable (Double) -> Void) async throws {
        let reader = try MonoFileReader(url: input.url)
        let resampler = input.sampleRate == AudioCore.sampleRate
            ? nil : try StreamingResampler(from: input.sampleRate, to: AudioCore.sampleRate)
        let writer = try MonoFileWriter(url: output, sampleRate: AudioCore.sampleRate, format: .cafFloat32)
        let session = isolator.makeIsolationSession()
        let total = max(1, Double(reader.frameCount))
        var consumed = 0
        while let block = try reader.read(maxFrames: 65_536) {
            try Task.checkCancellation()
            try writer.append(try session.push(try resampler?.process(block) ?? block))
            consumed += block.count
            progress(Double(consumed) / total)
        }
        if let resampler { try writer.append(try session.push(try resampler.flush())) }
        try writer.append(try session.finish())
        try writer.close()
        progress(1)
    }
}
