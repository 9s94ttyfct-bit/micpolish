import Foundation

enum TestSignals {
    static func sine(frequency: Double, amplitude: Double, seconds: Double, sampleRate: Int = 48_000, phase: Double = 0) -> [Float] {
        (0..<Int(seconds * Double(sampleRate))).map {
            Float(amplitude * sin(2 * Double.pi * frequency * Double($0) / Double(sampleRate) + phase))
        }
    }

    /// Deterministic speech-like signal: white noise with a 4 Hz syllable
    /// envelope and pauses, so dynamics processors and gates have work to do.
    static func modulatedNoise(seconds: Double, rmsDB: Double, sampleRate: Int = 48_000, seed: UInt64 = 1) -> [Float] {
        var state = seed
        func next() -> Double {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Double(state >> 11) / Double(1 << 53) * 2 - 1
        }
        let count = Int(seconds * Double(sampleRate))
        var out = [Float](repeating: 0, count: count)
        let gain = pow(10, rmsDB / 20) * 3.0.squareRoot()
        for i in 0..<count {
            let t = Double(i) / Double(sampleRate)
            let syllable = max(0, sin(2 * Double.pi * 4 * t))
            let phrase = sin(2 * Double.pi * 0.25 * t) > -0.6 ? 1.0 : 0.05
            out[i] = Float(gain * syllable * phrase * next())
        }
        return out
    }
}
