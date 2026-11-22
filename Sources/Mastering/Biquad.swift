import Foundation

/// Second-order IIR section (transposed direct form II, Double state).
/// Coefficient formulas follow the RBJ Audio EQ Cookbook.
struct Biquad {
    var b0 = 1.0, b1 = 0.0, b2 = 0.0, a1 = 0.0, a2 = 0.0
    private var z1 = 0.0, z2 = 0.0

    init(b0: Double, b1: Double, b2: Double, a0: Double, a1: Double, a2: Double) {
        self.b0 = b0 / a0
        self.b1 = b1 / a0
        self.b2 = b2 / a0
        self.a1 = a1 / a0
        self.a2 = a2 / a0
    }

    @inline(__always)
    mutating func process(_ x: Double) -> Double {
        let y = b0 * x + z1
        z1 = b1 * x - a1 * y + z2
        z2 = b2 * x - a2 * y
        return y
    }

    mutating func reset() {
        z1 = 0
        z2 = 0
    }

    /// Replaces coefficients while keeping filter state, so live parameter
    /// changes do not click.
    mutating func setCoefficients(from other: Biquad) {
        b0 = other.b0; b1 = other.b1; b2 = other.b2; a1 = other.a1; a2 = other.a2
    }

    private static func w0(_ f: Double, _ fs: Double) -> Double {
        2 * Double.pi * min(f, fs * 0.49) / fs
    }

    static func highpass(frequency f: Double, q: Double, sampleRate fs: Double) -> Biquad {
        let w = w0(f, fs), alpha = sin(w) / (2 * q), c = cos(w)
        return Biquad(b0: (1 + c) / 2, b1: -(1 + c), b2: (1 + c) / 2, a0: 1 + alpha, a1: -2 * c, a2: 1 - alpha)
    }

    static func bandpass(frequency f: Double, q: Double, sampleRate fs: Double) -> Biquad {
        let w = w0(f, fs), alpha = sin(w) / (2 * q), c = cos(w)
        return Biquad(b0: alpha, b1: 0, b2: -alpha, a0: 1 + alpha, a1: -2 * c, a2: 1 - alpha)
    }

    static func peaking(frequency f: Double, gainDB: Double, q: Double, sampleRate fs: Double) -> Biquad {
        let a = pow(10, gainDB / 40), w = w0(f, fs), alpha = sin(w) / (2 * q), c = cos(w)
        return Biquad(b0: 1 + alpha * a, b1: -2 * c, b2: 1 - alpha * a,
                      a0: 1 + alpha / a, a1: -2 * c, a2: 1 - alpha / a)
    }

    static func lowShelf(frequency f: Double, gainDB: Double, q: Double, sampleRate fs: Double) -> Biquad {
        let a = pow(10, gainDB / 40), w = w0(f, fs), alpha = sin(w) / (2 * q), c = cos(w)
        let s = 2 * a.squareRoot() * alpha
        return Biquad(b0: a * ((a + 1) - (a - 1) * c + s),
                      b1: 2 * a * ((a - 1) - (a + 1) * c),
                      b2: a * ((a + 1) - (a - 1) * c - s),
                      a0: (a + 1) + (a - 1) * c + s,
                      a1: -2 * ((a - 1) + (a + 1) * c),
                      a2: (a + 1) + (a - 1) * c - s)
    }

    static func highShelf(frequency f: Double, gainDB: Double, q: Double, sampleRate fs: Double) -> Biquad {
        let a = pow(10, gainDB / 40), w = w0(f, fs), alpha = sin(w) / (2 * q), c = cos(w)
        let s = 2 * a.squareRoot() * alpha
        return Biquad(b0: a * ((a + 1) + (a - 1) * c + s),
                      b1: -2 * a * ((a - 1) + (a + 1) * c),
                      b2: a * ((a + 1) + (a - 1) * c - s),
                      a0: (a + 1) - (a - 1) * c + s,
                      a1: 2 * ((a - 1) - (a + 1) * c),
                      a2: (a + 1) - (a - 1) * c - s)
    }
}
