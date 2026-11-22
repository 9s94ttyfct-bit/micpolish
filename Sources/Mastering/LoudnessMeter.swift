import Foundation

/// ITU-R BS.1770-4 integrated loudness and true peak for a mono signal.
///
/// Mono is measured as dual mono (the channel sum counted twice, +3.01 dB),
/// so a mono file at the target plays as loud as a stereo file at the same
/// target on two speakers or headphones.
public struct LoudnessMeter {
    private var shelf: Biquad
    private var highpass: Biquad
    private var oversampler = TruePeakOversampler()

    private let subBlockLength: Int
    private var subBlockEnergy = 0.0
    private var subBlockFill = 0
    private var recentSubBlocks: [Double] = []
    private var blockMeanSquares: [Double] = []
    private var peak = 0.0

    public init(sampleRate: Int) {
        let fs = Double(sampleRate)
        // K-weighting stage 1 (high shelf) and stage 2 (high-pass), with the
        // sample-rate-independent formulation used by libebur128.
        var k = tan(Double.pi * 1681.974450955533 / fs)
        let vh = pow(10, 3.999843853973347 / 20), vb = pow(vh, 0.4996667741545416)
        var q = 0.7071752369554196
        var a0 = 1 + k / q + k * k
        shelf = Biquad(b0: (vh + vb * k / q + k * k) / a0, b1: 2 * (k * k - vh) / a0,
                       b2: (vh - vb * k / q + k * k) / a0, a0: 1,
                       a1: 2 * (k * k - 1) / a0, a2: (1 - k / q + k * k) / a0)
        k = tan(Double.pi * 38.13547087602444 / fs)
        q = 0.5003270373238773
        a0 = 1 + k / q + k * k
        highpass = Biquad(b0: 1, b1: -2, b2: 1, a0: 1,
                          a1: 2 * (k * k - 1) / a0, a2: (1 - k / q + k * k) / a0)
        subBlockLength = max(1, sampleRate / 10)
    }

    public mutating func add(_ samples: [Float]) {
        for s in samples {
            let x = Double(s)
            peak = max(peak, oversampler.push(x), abs(x))
            let y = highpass.process(shelf.process(x))
            subBlockEnergy += y * y
            subBlockFill += 1
            if subBlockFill == subBlockLength {
                recentSubBlocks.append(subBlockEnergy)
                if recentSubBlocks.count > 4 { recentSubBlocks.removeFirst() }
                if recentSubBlocks.count == 4 {
                    blockMeanSquares.append(recentSubBlocks.reduce(0, +) / Double(4 * subBlockLength))
                }
                subBlockEnergy = 0
                subBlockFill = 0
            }
        }
    }

    private static func loudness(_ meanSquare: Double) -> Double {
        -0.691 + 10 * log10(2 * meanSquare)
    }

    /// Gated integrated loudness, or nil when no 400 ms block passes the
    /// -70 LUFS absolute gate (silence or too short).
    public var integratedLUFS: Double? {
        let absolute = blockMeanSquares.filter { $0 > 0 && Self.loudness($0) > -70 }
        guard !absolute.isEmpty else { return nil }
        let relativeGate = Self.loudness(absolute.reduce(0, +) / Double(absolute.count)) - 10
        let gated = absolute.filter { Self.loudness($0) > relativeGate }
        guard !gated.isEmpty else { return nil }
        return Self.loudness(gated.reduce(0, +) / Double(gated.count))
    }

    public var truePeakDBTP: Double { linearToDB(peak) }
}
