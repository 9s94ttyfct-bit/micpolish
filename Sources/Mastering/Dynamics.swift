import Foundation

@inline(__always)
func dbToLinear(_ db: Double) -> Double { pow(10, db / 20) }

@inline(__always)
func linearToDB(_ x: Double) -> Double { 20 * log10(max(x, 1e-12)) }

/// One-pole smoothing coefficient for a time constant in milliseconds.
@inline(__always)
func smoothingCoefficient(ms: Double, sampleRate: Double) -> Double {
    ms <= 0 ? 0 : exp(-1 / (ms * 0.001 * sampleRate))
}

/// Feed-forward compressor with a 6 dB soft knee. Gain is computed in dB and
/// smoothed with separate attack and release time constants.
struct Compressor {
    private var settings: MasteringSettings.Compressor
    private let sampleRate: Double
    private var attack = 0.0, release = 0.0
    private var envelopeDB = -120.0
    private var gainReductionDB = 0.0
    private let kneeDB = 6.0

    init(settings: MasteringSettings.Compressor, sampleRate: Double) {
        self.settings = settings
        self.sampleRate = sampleRate
        update(settings)
    }

    mutating func update(_ s: MasteringSettings.Compressor) {
        settings = s
        attack = smoothingCoefficient(ms: s.attackMs, sampleRate: sampleRate)
        release = smoothingCoefficient(ms: s.releaseMs, sampleRate: sampleRate)
    }

    private var makeupDB: Double {
        settings.autoMakeup ? -settings.thresholdDB * (1 - 1 / settings.ratio) * 0.5 : 0
    }

    /// Static gain computer: dB of reduction (≤ 0) for an input level.
    private func staticReduction(_ levelDB: Double) -> Double {
        let over = levelDB - settings.thresholdDB
        let slope = 1 / settings.ratio - 1
        if 2 * over < -kneeDB { return 0 }
        if 2 * abs(over) <= kneeDB {
            let x = over + kneeDB / 2
            return slope * x * x / (2 * kneeDB)
        }
        return slope * over
    }

    @inline(__always)
    mutating func process(_ x: Double) -> Double {
        let target = staticReduction(linearToDB(abs(x)))
        let coefficient = target < gainReductionDB ? attack : release
        gainReductionDB = target + coefficient * (gainReductionDB - target)
        return x * dbToLinear(gainReductionDB + makeupDB)
    }
}

/// Split-band de-esser: a complementary high band above `frequencyHz` is
/// attenuated when a band-passed sidechain exceeds the threshold. With no
/// reduction the bands sum back to the input exactly.
struct DeEsser {
    private var settings: MasteringSettings.DeEsser
    private let sampleRate: Double
    private var split: Biquad
    private var sidechain: Biquad
    private var envelope = 0.0
    private var gainDB = 0.0
    private let attack: Double
    private let release: Double

    init(settings: MasteringSettings.DeEsser, sampleRate: Double) {
        self.settings = settings
        self.sampleRate = sampleRate
        split = .highpass(frequency: settings.frequencyHz * 0.7, q: 0.707, sampleRate: sampleRate)
        sidechain = .bandpass(frequency: settings.frequencyHz, q: 1.5, sampleRate: sampleRate)
        attack = smoothingCoefficient(ms: 1, sampleRate: sampleRate)
        release = smoothingCoefficient(ms: 60, sampleRate: sampleRate)
    }

    mutating func update(_ s: MasteringSettings.DeEsser) {
        settings = s
        split.setCoefficients(from: .highpass(frequency: s.frequencyHz * 0.7, q: 0.707, sampleRate: sampleRate))
        sidechain.setCoefficients(from: .bandpass(frequency: s.frequencyHz, q: 1.5, sampleRate: sampleRate))
    }

    @inline(__always)
    mutating func process(_ x: Double) -> Double {
        let high = split.process(x)
        let level = abs(sidechain.process(x))
        envelope = level + (level > envelope ? attack : release) * (envelope - level)
        let over = linearToDB(envelope) - settings.thresholdDB
        let target = over > 0 ? over * (1 / settings.ratio - 1) : 0
        gainDB = target + (target < gainDB ? attack : release) * (gainDB - target)
        return (x - high) + high * dbToLinear(gainDB)
    }
}
