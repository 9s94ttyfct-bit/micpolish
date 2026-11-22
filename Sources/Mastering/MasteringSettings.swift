import Foundation

/// User-adjustable mastering parameters. Every stage can be bypassed.
public struct MasteringSettings: Codable, Equatable, Sendable {
    public struct Highpass: Codable, Equatable, Sendable {
        public var enabled = true
        public var frequencyHz = 80.0
    }

    public struct Band: Codable, Equatable, Sendable {
        public var frequencyHz: Double
        public var gainDB: Double
        public var q: Double
    }

    public struct EQ: Codable, Equatable, Sendable {
        public var enabled = false
        public var lowShelf = Band(frequencyHz: 120, gainDB: 0, q: 0.707)
        public var peak1 = Band(frequencyHz: 250, gainDB: 0, q: 1.0)
        public var peak2 = Band(frequencyHz: 4000, gainDB: 0, q: 1.2)
        public var highShelf = Band(frequencyHz: 10_000, gainDB: 0, q: 0.707)
    }

    public struct DeEsser: Codable, Equatable, Sendable {
        public var enabled = true
        public var frequencyHz = 6000.0
        public var thresholdDB = -30.0
        public var ratio = 4.0
    }

    public struct Compressor: Codable, Equatable, Sendable {
        public var enabled = true
        public var thresholdDB = -20.0
        public var ratio = 3.0
        public var attackMs = 10.0
        public var releaseMs = 150.0
        public var autoMakeup = true
    }

    public struct Loudness: Codable, Equatable, Sendable {
        public var enabled = true
        public var targetLUFS = -16.0
    }

    public struct Limiter: Codable, Equatable, Sendable {
        public var enabled = true
        public var ceilingDBTP = -1.0
    }

    public var highpass = Highpass()
    public var eq = EQ()
    public var deEsser = DeEsser()
    public var compressor = Compressor()
    public var loudness = Loudness()
    public var limiter = Limiter()

    public init() {}

    public static let `default` = MasteringSettings()

    /// Starting point for MacBook built-in microphones before blind-listening
    /// calibration: a little warmth, less boxiness, more presence and air.
    public static let macbookMicCorrection: MasteringSettings = {
        var s = MasteringSettings()
        s.eq.enabled = true
        s.eq.lowShelf = Band(frequencyHz: 120, gainDB: 2, q: 0.707)
        s.eq.peak1 = Band(frequencyHz: 250, gainDB: -2, q: 1.0)
        s.eq.peak2 = Band(frequencyHz: 4000, gainDB: 1.5, q: 1.2)
        s.eq.highShelf = Band(frequencyHz: 10_000, gainDB: 1, q: 0.707)
        return s
    }()
}
