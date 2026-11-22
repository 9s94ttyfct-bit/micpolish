import AudioCore
import Foundation
import LavaMLX

/// A model that regenerates the band above `cutoffHz` of one 48 kHz chunk.
public protocol BandwidthExtender: Sendable {
    func enhance(_ samples: [Float], cutoffHz: Double) throws -> [Float]
}

extension LavaEnhancer: BandwidthExtender {}

/// LavaSR v2 high-frequency enhancement on 48 kHz audio. Above the cutoff the
/// model's generated band replaces the input; below it the input is kept.
/// Runs in 10 s chunks joined with 0.5 s linear crossfades.
public struct LavaStage: AudioStage {
    public static let chunkFrames = 10 * AudioCore.sampleRate
    public static let overlapFrames = AudioCore.sampleRate / 2

    public let id = "lavasr-v2"
    public let settingsKey: String
    private let enhancer: any BandwidthExtender
    private let cutoffHz: Double?

    /// `cutoffHz` nil measures the input's effective bandwidth.
    public init(enhancer: any BandwidthExtender, revision: String, cutoffHz: Double?) {
        self.enhancer = enhancer
        self.cutoffHz = cutoffHz
        settingsKey = "\(revision)|\(cutoffHz.map { String(Int($0)) } ?? "auto")|10s-0.5s"
    }

    public func outputSampleRate(forInput sampleRate: Int) -> Int { sampleRate }

    public func run(input: StageIO, output: URL, progress: @Sendable (Double) -> Void) async throws {
        precondition(input.sampleRate == AudioCore.sampleRate, "LavaStage expects 48 kHz input")
        let cutoff = try cutoffHz ?? min(20_000, max(4_000, BandwidthDetector.effectiveBandwidth(of: input.url)))
        let reader = try MonoFileReader(url: input.url)
        let writer = try MonoFileWriter(url: output, sampleRate: AudioCore.sampleRate, format: .cafFloat32)
        let total = max(1, Double(reader.frameCount))
        let hop = Self.chunkFrames - Self.overlapFrames

        var buffer = [Float]()
        var previousTail = [Float]()   // enhanced overlap region of the previous chunk
        var consumed = 0
        var finished = false
        while !finished {
            try Task.checkCancellation()
            while buffer.count < Self.chunkFrames, let block = try reader.read(maxFrames: 65_536) {
                buffer += block
            }
            finished = buffer.count < Self.chunkFrames
            guard !buffer.isEmpty else { break }
            let chunk = Array(buffer.prefix(Self.chunkFrames))
            var enhanced = try enhancer.enhance(chunk, cutoffHz: cutoff)

            // Crossfade the head of this chunk with the previous chunk's tail.
            let fade = min(previousTail.count, enhanced.count)
            for i in 0..<fade {
                let w = Float(i + 1) / Float(fade + 1)
                enhanced[i] = previousTail[i] * (1 - w) + enhanced[i] * w
            }
            if finished {
                try writer.append(enhanced)
                consumed += chunk.count
            } else {
                try writer.append(Array(enhanced[0..<hop]))
                previousTail = Array(enhanced[hop...])
                // The overlap stays at the head of the next chunk.
                buffer.removeFirst(hop)
                consumed += hop
            }
            progress(min(1, Double(consumed) / total))
        }
        try writer.close()
        progress(1)
    }
}
