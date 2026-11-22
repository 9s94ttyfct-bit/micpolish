import Foundation

/// Look-ahead true-peak limiter.
///
/// For each sample the required gain is `ceiling / truePeak`. A forward
/// minimum over the look-ahead window, an exponential recovery and a box
/// average of the same length produce a smooth gain that is never above the
/// requirement at any peak (the average of values each ≤ r is ≤ r).
struct TruePeakLimiter {
    private static let marginDB = 0.2
    private let window: Int
    private let release: Double
    private var ceiling: Double
    private var oversampler = TruePeakOversampler()

    private var inputDelay: [Double]
    private var inputIndex = 0
    private var requirements: MinQueue
    private var boxValues: [Double]
    private var boxIndex = 0
    private var boxSum: Double
    private var recovered = 1.0

    let latency: Int

    init(ceilingDBTP: Double, sampleRate: Double) {
        window = max(8, Int(0.002 * sampleRate))
        release = smoothingCoefficient(ms: 80, sampleRate: sampleRate)
        ceiling = dbToLinear(ceilingDBTP - Self.marginDB)
        // Requirements are known D samples after the input (interpolator
        // delay); the min window reaches one sample past the box window on each
        // side to absorb the sub-sample offset of the interpolated points.
        latency = TruePeakOversampler.delay + window + 1
        inputDelay = [Double](repeating: 0, count: latency + 1)
        requirements = MinQueue(capacity: window + 3)
        boxValues = [Double](repeating: 1, count: window)
        boxSum = Double(window)
    }

    mutating func setCeiling(_ ceilingDBTP: Double) {
        ceiling = dbToLinear(ceilingDBTP - Self.marginDB)
    }

    @inline(__always)
    mutating func process(_ x: Double) -> Double {
        inputDelay[inputIndex] = x
        let size = inputDelay.count
        let rawNear = max(abs(inputDelay[(inputIndex - TruePeakOversampler.delay + size) % size]),
                          abs(inputDelay[(inputIndex - TruePeakOversampler.delay + 1 + size) % size]))
        inputIndex = (inputIndex + 1) % size

        // The interpolated points completed by this sample sit between input
        // times n - D and n - D + 1; include both raw samples, which a
        // windowed-sinc interpolator at quarter-sample offsets can under-read
        // by about 0.2 dB.
        let peak = max(oversampler.push(x), rawNear)
        requirements.push(peak > ceiling ? ceiling / peak : 1)

        let held = requirements.minimum
        recovered = min(held, 1 - (1 - recovered) * release)
        boxSum += recovered - boxValues[boxIndex]
        boxValues[boxIndex] = recovered
        boxIndex = (boxIndex + 1) % window
        let gain = boxSum / Double(window)

        // The delayed sample `latency` steps back aligns with the window.
        let delayed = inputDelay[inputIndex]
        return delayed * min(1, gain)
    }
}

/// Sliding-window minimum over the most recent `capacity` values.
struct MinQueue {
    private var values: [Double]
    private var indices: [Int]
    private var head = 0
    private var count = 0
    private var time = 0
    private let capacity: Int

    init(capacity: Int) {
        self.capacity = capacity
        values = [Double](repeating: 0, count: capacity + 1)
        indices = [Int](repeating: 0, count: capacity + 1)
    }

    mutating func push(_ v: Double) {
        let size = values.count
        while count > 0 {
            let last = (head + count - 1) % size
            if values[last] >= v { count -= 1 } else { break }
        }
        let slot = (head + count) % size
        values[slot] = v
        indices[slot] = time
        count += 1
        while indices[head] <= time - capacity {
            head = (head + 1) % size
            count -= 1
        }
        time += 1
    }

    var minimum: Double { count > 0 ? values[head] : 1 }
}
