import Foundation

/// 4x polyphase interpolator for true-peak estimation (ITU-R BS.1770-4
/// Annex 2 approach: 48 taps, 12 per phase, windowed sinc).
struct TruePeakOversampler {
    static let factor = 4
    static let tapsPerPhase = 12
    /// Input samples between a sample entering and the centre of the
    /// interpolated span reported for it.
    static let delay = 6

    private static let phases: [[Double]] = {
        let total = factor * tapsPerPhase
        let centre = Double(total - 1) / 2
        var h = [Double](repeating: 0, count: total)
        for k in 0..<total {
            let t = (Double(k) - centre) / Double(factor)
            let sinc = t == 0 ? 1 : sin(Double.pi * t) / (Double.pi * t)
            let window = 0.5 - 0.5 * cos(2 * Double.pi * (Double(k) + 0.5) / Double(total))
            h[k] = sinc * window
        }
        return (0..<factor).map { p in (0..<tapsPerPhase).map { j in h[factor * j + p] } }
    }()

    private var history = [Double](repeating: 0, count: tapsPerPhase)
    private var position = 0

    /// Pushes one sample and returns the largest absolute value among the
    /// interpolated points it completes.
    @inline(__always)
    mutating func push(_ x: Double) -> Double {
        history[position] = x
        var peak = 0.0
        for phase in Self.phases {
            var acc = 0.0
            var index = position
            for tap in phase {
                acc += tap * history[index]
                index = index == 0 ? Self.tapsPerPhase - 1 : index - 1
            }
            peak = max(peak, abs(acc))
        }
        position = (position + 1) % Self.tapsPerPhase
        return peak
    }

    /// Clears the history in place (no allocation).
    mutating func reset() {
        for i in history.indices { history[i] = 0 }
        position = 0
    }
}
