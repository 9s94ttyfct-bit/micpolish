import Foundation

/// The mastering signal path: high-pass → studio tone → EQ → de-esser →
/// compressor → loudness gain → true-peak limiter. Processing is per sample, so output is
/// identical regardless of how the input is split into blocks.
public final class MasteringChain {
    public private(set) var settings: MasteringSettings
    public var loudnessGainDB: Double = 0
    /// Per-band studio-tone correction measured on this audio
    /// (`StudioTone.Analysis.correction`); empty until measured.
    public var studioToneCorrection: [Double] {
        get { correction }
        set { apply(Self.design(settings, studioToneCorrection: newValue, sampleRate: Int(sampleRate))) }
    }
    private var correction: [Double] = []
    private let sampleRate: Double

    private var highpass: Biquad
    private var studioTone: [Biquad] = []
    private var eq: [Biquad]
    private var deEsser: DeEsser
    private var compressor: Compressor
    private var limiter: TruePeakLimiter

    public init(settings: MasteringSettings, sampleRate: Int) {
        self.settings = settings
        self.sampleRate = Double(sampleRate)
        highpass = Self.makeHighpass(settings.highpass, Double(sampleRate))
        eq = Self.makeEQ(settings.eq, Double(sampleRate))
        deEsser = DeEsser(settings: settings.deEsser, sampleRate: Double(sampleRate))
        compressor = Compressor(settings: settings.compressor, sampleRate: Double(sampleRate))
        limiter = TruePeakLimiter(ceilingDBTP: settings.limiter.ceilingDBTP, sampleRate: Double(sampleRate))
    }

    /// Frames by which output lags input (the limiter's look-ahead).
    public var latencyFrames: Int { settings.limiter.enabled ? limiter.latency : 0 }

    /// Filter coefficients for a parameter set. Designing them (studio tone in
    /// particular) is costly, so a real-time owner computes a `Design` outside
    /// its lock and only `apply`s it under the lock.
    public struct Design: Sendable {
        let settings: MasteringSettings
        let studioToneCorrection: [Double]
        let highpass: Biquad
        let eq: [Biquad]
        let studioTone: [Biquad]
    }

    public static func design(_ settings: MasteringSettings, studioToneCorrection: [Double], sampleRate: Int) -> Design {
        let fs = Double(sampleRate)
        let studioTone = studioToneCorrection.isEmpty ? [] : StudioTone.filters(
            target: StudioTone.target(correction: studioToneCorrection, amount: settings.studioTone.amount), sampleRate: fs)
        return Design(settings: settings, studioToneCorrection: studioToneCorrection,
                      highpass: makeHighpass(settings.highpass, fs), eq: makeEQ(settings.eq, fs), studioTone: studioTone)
    }

    /// Applies new parameters while keeping filter and envelope state.
    public func update(_ new: MasteringSettings) {
        apply(Self.design(new, studioToneCorrection: correction, sampleRate: Int(sampleRate)))
    }

    /// Swaps in precomputed coefficients, keeping filter and envelope state.
    /// Does not allocate unless the number of studio-tone bands changes.
    public func apply(_ design: Design) {
        highpass.setCoefficients(from: design.highpass)
        for i in eq.indices { eq[i].setCoefficients(from: design.eq[i]) }
        deEsser.update(design.settings.deEsser)
        compressor.update(design.settings.compressor)
        limiter.setCeiling(design.settings.limiter.ceilingDBTP)
        settings = design.settings
        correction = design.studioToneCorrection
        if design.studioTone.count == studioTone.count {
            for i in studioTone.indices { studioTone[i].setCoefficients(from: design.studioTone[i]) }
        } else {
            studioTone = design.studioTone
        }
    }

    /// Clears all filter, envelope and look-ahead state in place (no allocation).
    public func reset() {
        highpass.reset()
        for i in eq.indices { eq[i].reset() }
        for i in studioTone.indices { studioTone[i].reset() }
        deEsser.reset()
        compressor.reset()
        limiter.reset()
    }

    public func process(_ samples: inout [Float]) {
        samples.withUnsafeMutableBufferPointer { process($0) }
    }

    /// In-place processing without allocation, for real-time render callbacks.
    public func process(_ samples: UnsafeMutableBufferPointer<Float>) {
        let gain = settings.loudness.enabled ? dbToLinear(loudnessGainDB) : 1
        for i in samples.indices {
            var x = processBeforeGain(Double(samples[i])) * gain
            if settings.limiter.enabled { x = limiter.process(x) }
            samples[i] = Float(x)
        }
    }

    /// Runs the stages ahead of the loudness gain (used to measure loudness).
    func processBeforeGain(_ input: Double) -> Double {
        var x = input
        if settings.highpass.enabled { x = highpass.process(x) }
        if settings.studioTone.enabled {
            for i in studioTone.indices { x = studioTone[i].process(x) }
        }
        if settings.eq.enabled {
            for i in eq.indices { x = eq[i].process(x) }
        }
        if settings.deEsser.enabled { x = deEsser.process(x) }
        if settings.compressor.enabled { x = compressor.process(x) }
        return x
    }

    private static func makeHighpass(_ s: MasteringSettings.Highpass, _ fs: Double) -> Biquad {
        .highpass(frequency: s.frequencyHz, q: 0.7071, sampleRate: fs)
    }

    private static func makeEQ(_ s: MasteringSettings.EQ, _ fs: Double) -> [Biquad] {
        [
            .lowShelf(frequency: s.lowShelf.frequencyHz, gainDB: s.lowShelf.gainDB, q: s.lowShelf.q, sampleRate: fs),
            .peaking(frequency: s.peak1.frequencyHz, gainDB: s.peak1.gainDB, q: s.peak1.q, sampleRate: fs),
            .peaking(frequency: s.peak2.frequencyHz, gainDB: s.peak2.gainDB, q: s.peak2.q, sampleRate: fs),
            .highShelf(frequency: s.highShelf.frequencyHz, gainDB: s.highShelf.gainDB, q: s.highShelf.q, sampleRate: fs),
        ]
    }
}
