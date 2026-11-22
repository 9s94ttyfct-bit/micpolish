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

    /// Loudness within this distance of the target ends gain refinement.
    static let toleranceLU = 0.2
    static let maximumRefinements = 4

    /// Computes the loudness gain for `input` without writing anything.
    /// Throws `CancellationError` promptly when the calling task is cancelled.
    ///
    /// The first estimate comes from the loudness ahead of the gain stage.
    /// The limiter then removes loudness on peaky material, so the full chain
    /// is re-run (without writing) and the gain corrected until the output is
    /// within 0.2 LU of the target or no longer improves.
    ///
    /// `studioTone` is the measured studio-tone correction for `input`; nil
    /// measures it here when the stage is on.
    public static func measureGain(input: URL, settings: MasteringSettings,
                                   studioTone: [Double]? = nil) throws -> (measured: Double?, gainDB: Double) {
        guard settings.loudness.enabled else { return (nil, 0) }
        let correction = try studioTone ?? studioToneCorrection(input: input, settings: settings)
        let reader = try MonoFileReader(url: input)
        let chain = MasteringChain(settings: settings, sampleRate: reader.sampleRate)
        chain.studioToneCorrection = correction
        var meter = LoudnessMeter(sampleRate: reader.sampleRate)
        while let block = try reader.read(maxFrames: blockFrames) {
            try Task.checkCancellation()
            meter.add(block.map { Float(chain.processBeforeGain(Double($0))) })
        }
        guard let measured = meter.integratedLUFS else { return (nil, 0) }
        let target = settings.loudness.targetLUFS
        var gain = min(maximumGainDB, target - measured)

        guard settings.limiter.enabled else { return (measured, gain) }
        for _ in 0..<maximumRefinements {
            guard let output = try run(input: input, settings: settings, gainDB: gain, studioTone: correction,
                                       output: nil).outputLUFS else { break }
            let shortfall = target - output
            if abs(shortfall) <= toleranceLU || gain >= maximumGainDB { break }
            gain = min(maximumGainDB, gain + shortfall)
        }
        return (measured, gain)
    }

    @discardableResult
    public static func render(input: URL, settings: MasteringSettings, output: MonoFileWriter,
                              studioTone: [Double]? = nil) throws -> RenderReport {
        let correction = try studioTone ?? studioToneCorrection(input: input, settings: settings)
        let (measured, gainDB) = try measureGain(input: input, settings: settings, studioTone: correction)
        var report = try run(input: input, settings: settings, gainDB: gainDB, studioTone: correction, output: output)
        report.measuredLUFS = measured
        return report
    }

    /// The studio-tone correction for `input`, or none when the stage is off.
    public static func studioToneCorrection(input: URL, settings: MasteringSettings) throws -> [Double] {
        settings.studioTone.enabled ? try StudioTone.analyze(url: input).correction : []
    }

    /// One full pass through the chain with a fixed gain, latency-compensated
    /// to the input's length. Writes to `output` when given.
    static func run(input: URL, settings: MasteringSettings, gainDB: Double, studioTone: [Double] = [],
                    output: MonoFileWriter?) throws -> RenderReport {
        let reader = try MonoFileReader(url: input)
        let chain = MasteringChain(settings: settings, sampleRate: reader.sampleRate)
        chain.loudnessGainDB = gainDB
        chain.studioToneCorrection = studioTone
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
            try output?.append(block)
        }

        while var block = try reader.read(maxFrames: blockFrames) {
            try Task.checkCancellation()
            chain.process(&block)
            try emit(block)
        }
        if chain.latencyFrames > 0 {
            var tail = [Float](repeating: 0, count: chain.latencyFrames)
            chain.process(&tail)
            try emit(tail)
        }
        return RenderReport(
            measuredLUFS: nil,
            appliedGainDB: gainDB,
            outputLUFS: meter.integratedLUFS,
            outputTruePeakDBTP: meter.truePeakDBTP
        )
    }
}
