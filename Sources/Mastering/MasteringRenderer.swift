import AudioCore
import Foundation

public struct RenderReport: Sendable, Equatable {
    /// Loudness measured after the compressor, before the loudness gain.
    public var measuredLUFS: Double?
    public var appliedGainDB: Double
    public var outputLUFS: Double?
    public var outputTruePeakDBTP: Double
}

/// Two-pass offline mastering: pass 1 measures the loudness the chain
/// produces ahead of the gain stage, pass 2 renders with the gain that hits
/// the target, compensating the limiter latency so the output is aligned and
/// has the input's length.
public enum MasteringRenderer {
    public static let maximumGainDB = 30.0
    private static let blockFrames = 65_536

    /// Computes the loudness gain for `input` without writing anything.
    public static func measureGain(input: URL, settings: MasteringSettings) throws -> (measured: Double?, gainDB: Double) {
        guard settings.loudness.enabled else { return (nil, 0) }
        let reader = try MonoFileReader(url: input)
        let chain = MasteringChain(settings: settings, sampleRate: reader.sampleRate)
        var meter = LoudnessMeter(sampleRate: reader.sampleRate)
        while let block = try reader.read(maxFrames: blockFrames) {
            meter.add(block.map { Float(chain.processBeforeGain(Double($0))) })
        }
        guard let measured = meter.integratedLUFS else { return (nil, 0) }
        return (measured, min(maximumGainDB, settings.loudness.targetLUFS - measured))
    }

    @discardableResult
    public static func render(input: URL, settings: MasteringSettings, output: MonoFileWriter) throws -> RenderReport {
        let (measured, gainDB) = try measureGain(input: input, settings: settings)

        let reader = try MonoFileReader(url: input)
        let chain = MasteringChain(settings: settings, sampleRate: reader.sampleRate)
        chain.loudnessGainDB = gainDB
        var meter = LoudnessMeter(sampleRate: reader.sampleRate)
        var toSkip = chain.latencyFrames
        var remaining = Int(reader.frameCount)

        func emit(_ block: [Float]) throws {
            var block = block
            if toSkip > 0 {
                let drop = min(toSkip, block.count)
                block.removeFirst(drop)
                toSkip -= drop
            }
            if block.count > remaining { block.removeLast(block.count - remaining) }
            guard !block.isEmpty else { return }
            remaining -= block.count
            meter.add(block)
            try output.append(block)
        }

        while var block = try reader.read(maxFrames: blockFrames) {
            chain.process(&block)
            try emit(block)
        }
        if chain.latencyFrames > 0 {
            var tail = [Float](repeating: 0, count: chain.latencyFrames)
            chain.process(&tail)
            try emit(tail)
        }
        return RenderReport(
            measuredLUFS: measured,
            appliedGainDB: gainDB,
            outputLUFS: meter.integratedLUFS,
            outputTruePeakDBTP: meter.truePeakDBTP
        )
    }
}
