import Foundation

/// The mastering signal path: high-pass → studio tone → EQ → de-esser →
/// compressor → loudness gain → true-peak limiter. Processing is per sample, so output is
/// identical regardless of how the input is split into blocks.
public final class MasteringChain {
    public private(set) var settings: MasteringSettings
    public var loudnessGainDB: Double = 0
    /// Per-band studio-tone correction measured on this audio
    /// (`StudioTone.Analysis.correction`); empty until measured.
    public var studioToneCorrection: [Double] = [] {
        didSet { rebuildStudioTone() }
    }
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

    /// Applies new parameters while keeping filter and envelope state.
    public func update(_ new: MasteringSettings) {
        highpass.setCoefficients(from: Self.makeHighpass(new.highpass, sampleRate))
        for (i, band) in Self.makeEQ(new.eq, sampleRate).enumerated() {
            eq[i].setCoefficients(from: band)
        }
        deEsser.update(new.deEsser)
        compressor.update(new.compressor)
        limiter.setCeiling(new.limiter.ceilingDBTP)
        settings = new
        rebuildStudioTone()
    }

    public func reset() {
        let fresh = MasteringChain(settings: settings, sampleRate: Int(sampleRate))
        highpass = fresh.highpass
        eq = fresh.eq
        deEsser = fresh.deEsser
        compressor = fresh.compressor
        limiter = fresh.limiter
        for i in studioTone.indices { studioTone[i].reset() }
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

    /// Applies `studioToneCorrection` at the current amount, keeping filter state.
    private func rebuildStudioTone() {
        guard !studioToneCorrection.isEmpty else {
            studioTone = []
            return
        }
        let target = StudioTone.target(correction: studioToneCorrection, amount: settings.studioTone.amount)
        let filters = StudioTone.filters(target: target, sampleRate: sampleRate)
        if filters.count == studioTone.count {
            for i in filters.indices { studioTone[i].setCoefficients(from: filters[i]) }
        } else {
            studioTone = filters
        }
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
