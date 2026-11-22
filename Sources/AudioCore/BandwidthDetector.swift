import Accelerate
import Foundation

/// Estimates the highest frequency a recording actually contains, from its
/// long-term spectrum: the top of the band whose level stays within 45 dB of
/// the 1–4 kHz speech band. Silence reports the Nyquist frequency.
public enum BandwidthDetector {
    static let fftSize = 4096
    static let floorBelowSpeechBandDB = 45.0

    public static func effectiveBandwidth(of url: URL, maxSeconds: Double = 60) throws -> Double {
        let reader = try MonoFileReader(url: url)
        let nyquist = Double(reader.sampleRate) / 2
        let n = fftSize
        let half = n / 2
        guard let setup = vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(n), .FORWARD) else { return nyquist }
        defer { vDSP_DFT_DestroySetup(setup) }

        var window = [Float](repeating: 0, count: n)
        vDSP_hann_window(&window, vDSP_Length(n), Int32(vDSP_HANN_NORM))
        var power = [Double](repeating: 0, count: half)
        var frames = 0
        var remaining = Int(maxSeconds * Double(reader.sampleRate))
        var pending = [Float]()

        while remaining > 0, let block = try reader.read(maxFrames: min(65_536, remaining)) {
            remaining -= block.count
            pending += block
            while pending.count >= n {
                let frame = Array(pending[0..<n])
                pending.removeFirst(n)
                var rms: Float = 0
                vDSP_rmsqv(frame, 1, &rms, vDSP_Length(n))
                guard rms > 1e-4 else { continue }   // skip pauses
                var real = [Float](repeating: 0, count: n), imag = [Float](repeating: 0, count: n)
                vDSP_vmul(frame, 1, window, 1, &real, 1, vDSP_Length(n))
                var outReal = [Float](repeating: 0, count: n), outImag = [Float](repeating: 0, count: n)
                vDSP_DFT_Execute(setup, real, imag, &outReal, &outImag)
                for k in 0..<half {
                    power[k] += Double(outReal[k] * outReal[k] + outImag[k] * outImag[k])
                }
                frames += 1
            }
        }
        guard frames > 0 else { return nyquist }

        let binHz = Double(reader.sampleRate) / Double(n)
        func levelDB(_ k: Int) -> Double {
            // Smooth over ±8 bins (±94 Hz at 48 kHz).
            let lo = max(0, k - 8), hi = min(half - 1, k + 8)
            let mean = power[lo...hi].reduce(0, +) / Double(hi - lo + 1)
            return 10 * log10(max(mean, 1e-30))
        }
        let speech = Int(1000 / binHz)..<Int(4000 / binHz)
        let reference = 10 * log10(speech.map { power[$0] }.reduce(0, +) / Double(speech.count))
        let threshold = reference - floorBelowSpeechBandDB
        for k in stride(from: half - 1, through: speech.upperBound, by: -1) where levelDB(k) >= threshold {
            return min(nyquist, Double(k + 1) * binHz)
        }
        return Double(speech.upperBound) * binHz
    }
}
