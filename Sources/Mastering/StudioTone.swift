import Accelerate
import AudioCore
import Foundation

/// "Studio tone": an automatic EQ that moves the long-term spectrum of the
/// voice towards that of speech recorded close to a studio condenser in a
/// treated room, in octave bands. It changes tone only, never the words or the
/// voice itself.
///
/// The references come from VCTK 0.92 (CC BY 4.0) `mic2` (Sennheiser MKH 800,
/// hemi-anechoic chamber), one curve per gender, built by
/// `tools/studio_tone/build_reference.py` with exactly the analysis below.
public enum StudioTone {
    /// Octave-band centres in Hz.
    public static let centers: [Double] = [63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    /// Largest boost or cut at full amount.
    public static let maximumDB = 6.0

    /// Reference shapes (dB, same normalization as `Analysis.shape`) from VCTK
    /// 0.92 `mic2`: 18 male and 16 female speakers, equal weight per speaker
    /// (`tools/studio_tone/build_reference.py`).
    static let male: [Double] = [-10.92, 3.96, 5.30, 8.01, -0.07, -4.52, -8.72, -15.24, -26.03]
    static let female: [Double] = [-21.29, -6.88, 4.53, 8.40, 2.89, -4.25, -11.57, -15.19, -25.39]

    static let frameLength = 4096
    static let hop = 2048
    /// Band levels are normalized to their mean over 250 Hz-4 kHz.
    static let shapeBands = 2...6

    public enum Voice: String, Sendable, Codable { case male, female }

    /// Long-term octave-band shape of one recording.
    public struct Analysis: Sendable, Equatable {
        /// Band level (dB) minus the 250 Hz-4 kHz mean, per `centers`.
        public var shape: [Double]
        /// Band level relative to the loudest band, to recognise empty bands.
        public var relativeLevel: [Double]
        public var medianPitchHz: Double?
        public var voice: Voice

        /// Reference minus measured shape per band (dB), before `amount` and
        /// limits. Bands with no content (more than 50 dB below the loudest)
        /// are left alone, and the 63 Hz band is only ever cut.
        public var correction: [Double] {
            let reference = StudioTone.reference(pitchHz: medianPitchHz)
            return centers.indices.map { i in
                guard relativeLevel[i] > -50 else { return 0 }
                let delta = reference[i] - shape[i]
                return i == 0 ? min(0, delta) : delta
            }
        }
    }

    /// The reference for a voice of this median pitch: the male curve up to
    /// 145 Hz, the female curve from 175 Hz, and a linear blend in between,
    /// where pitch alone cannot tell lower female from higher male voices.
    /// Without a pitch (no voiced frames) the two are averaged.
    static func reference(pitchHz: Double?) -> [Double] {
        let femaleWeight = pitchHz.map { max(0, min(1, ($0 - 145) / 30)) } ?? 0.5
        return zip(male, female).map { $0 * (1 - femaleWeight) + $1 * femaleWeight }
    }

    /// The per-band target (dB) for `amount` in 0...1, limited to ±6 dB.
    public static func target(correction: [Double], amount: Double) -> [Double] {
        correction.map { max(-maximumDB, min(maximumDB, $0 * max(0, min(1, amount)))) }
    }

    // MARK: Analysis

    /// Analyzes a mono file of any rate (resampled internally is not needed:
    /// bins are mapped by frequency).
    public static func analyze(url: URL) throws -> Analysis {
        let reader = try MonoFileReader(url: url)
        let analyzer = SpectrumAccumulator(sampleRate: Double(reader.sampleRate))
        var pending: [Float] = []
        while let block = try reader.read(maxFrames: 65_536) {
            try Task.checkCancellation()
            pending += block
            var start = 0
            while start + frameLength <= pending.count {
                analyzer.add(frame: pending[start..<(start + frameLength)])
                start += hop
            }
            pending.removeFirst(start)
        }
        return analyzer.result()
    }

    /// Frame statistics gathered in one pass; speech frames are chosen
    /// afterwards from the frame levels, so only per-frame band powers are kept.
    final class SpectrumAccumulator {
        let sampleRate: Double
        private let log2n = vDSP_Length(12)
        private let setup: FFTSetup
        private let window: [Float]
        private let bandBins: [Range<Int>]
        private let levelBins: Range<Int>
        private var frames: [(level: Double, bands: [Double], pitch: Double?)] = []
        /// Bins 1..<pitchBins (below 2 kHz) and lags 70-400 Hz, with
        /// cosTable[row * (pitchBins - 1) + k - 1] = cos(2π k lag / n).
        private let pitchBins: Int
        private let pitchLags: [Int]
        private let cosTable: [Double]

        init(sampleRate: Double) {
            self.sampleRate = sampleRate
            setup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2))!
            let n = StudioTone.frameLength
            window = (0..<n).map { Float(0.5 - 0.5 * cos(2 * Double.pi * Double($0) / Double(n))) }
            let binHz = sampleRate / Double(n)
            func bins(_ lo: Double, _ hi: Double) -> Range<Int> {
                // Bins with lo <= f < hi, as in the reference script.
                let first = Int((lo / binHz).rounded(.up))
                let last = min(n / 2, Int((hi / binHz).rounded(.up)))
                return min(first, last)..<last
            }
            bandBins = StudioTone.centers.map { bins($0 / 2.0.squareRoot(), $0 * 2.0.squareRoot()) }
            levelBins = bins(100, 8000.000_001)
            pitchBins = max(2, min(n / 2, Int(2000 / binHz)))
            pitchLags = Array(Int(sampleRate / 400)...Int(sampleRate / 70))
            var table = [Double]()
            table.reserveCapacity(pitchLags.count * (pitchBins - 1))
            for lag in pitchLags {
                for k in 1..<pitchBins { table.append(cos(2 * Double.pi * Double(k * lag) / Double(n))) }
            }
            cosTable = table
        }

        deinit { vDSP_destroy_fftsetup(setup) }

        func add(frame: ArraySlice<Float>) {
            let n = StudioTone.frameLength
            var input = [Float](repeating: 0, count: n)
            frame.withContiguousStorageIfAvailable { src in
                vDSP_vmul(src.baseAddress!, 1, window, 1, &input, 1, vDSP_Length(n))
            }
            // Power spectrum |X(k)|^2 for k = 0...n/2, scaled like numpy's rfft.
            var real = [Float](repeating: 0, count: n / 2)
            var imag = [Float](repeating: 0, count: n / 2)
            var power = [Double](repeating: 0, count: n / 2 + 1)
            real.withUnsafeMutableBufferPointer { r in
                imag.withUnsafeMutableBufferPointer { i in
                    var split = DSPSplitComplex(realp: r.baseAddress!, imagp: i.baseAddress!)
                    input.withUnsafeBytes {
                        vDSP_ctoz($0.bindMemory(to: DSPComplex.self).baseAddress!, 2, &split, 1, vDSP_Length(n / 2))
                    }
                    vDSP_fft_zrip(setup, &split, 1, log2n, FFTDirection(FFT_FORWARD))
                    // zrip returns 2x the DFT; DC and Nyquist share element 0.
                    power[0] = Double(r[0] / 2) * Double(r[0] / 2)
                    power[n / 2] = Double(i[0] / 2) * Double(i[0] / 2)
                    for k in 1..<(n / 2) {
                        let re = Double(r[k] / 2), im = Double(i[k] / 2)
                        power[k] = re * re + im * im
                    }
                }
            }
            let level = 10 * log10(levelBins.reduce(0) { $0 + power[$1] } + 1e-20)
            let bands = bandBins.map { range in range.reduce(0) { $0 + power[$1] } }
            frames.append((level, bands, pitch(power)))
        }

        /// Pitch from the autocorrelation of the power spectrum below 2 kHz
        /// (where voiced harmonics carry their energy), searched over 70-400 Hz;
        /// nil when the frame is not clearly voiced.
        private func pitch(_ power: [Double]) -> Double? {
            let low = power[1..<pitchBins].reduce(0, +)
            guard low > 0 else { return nil }
            var bestLag = 0, best = 0.0
            power.withUnsafeBufferPointer { p in
                cosTable.withUnsafeBufferPointer { table in
                    let bins = vDSP_Length(pitchBins - 1)
                    for (row, lag) in pitchLags.enumerated() {
                        var r = 0.0
                        vDSP_dotprD(p.baseAddress! + 1, 1, table.baseAddress! + row * (pitchBins - 1), 1, &r, bins)
                        if r > best { best = r; bestLag = lag }
                    }
                }
            }
            return best / low > 0.4 ? sampleRate / Double(bestLag) : nil
        }

        func result() -> Analysis {
            let count = StudioTone.centers.count
            guard !frames.isEmpty else {
                return Analysis(shape: [Double](repeating: 0, count: count),
                                relativeLevel: [Double](repeating: 0, count: count), medianPitchHz: nil, voice: .female)
            }
            let levels = frames.map(\.level).sorted()
            let p95 = Self.percentile(levels, 95)
            let speech = frames.filter { $0.level >= p95 - 30 }
            var sums = [Double](repeating: 0, count: count)
            for f in speech { for i in 0..<count { sums[i] += f.bands[i] } }
            let db = sums.map { 10 * log10($0 / Double(speech.count) + 1e-20) }
            let mean = StudioTone.shapeBands.map { db[$0] }.reduce(0, +) / Double(StudioTone.shapeBands.count)
            let loudest = db.max() ?? 0
            let pitches = speech.compactMap(\.pitch).sorted()
            let median = pitches.isEmpty ? nil : pitches[pitches.count / 2]
            return Analysis(shape: db.map { $0 - mean }, relativeLevel: db.map { $0 - loudest },
                            medianPitchHz: median, voice: (median ?? 200) < 160 ? .male : .female)
        }

        /// Linear-interpolated percentile, as numpy's default.
        static func percentile(_ sorted: [Double], _ p: Double) -> Double {
            let position = Double(sorted.count - 1) * p / 100
            let lo = Int(position.rounded(.down)), hi = Int(position.rounded(.up))
            return sorted[lo] + (sorted[hi] - sorted[lo]) * (position - Double(lo))
        }
    }

    // MARK: Filters

    /// Peaking-filter gains whose cascade hits `target` (dB) at the band
    /// centres: overlapping octave filters interact, so the gains are refined
    /// by a few fixed-point steps on the cascade's response.
    static func filterGains(target: [Double], sampleRate: Double) -> [Double] {
        var gains = target
        for _ in 0..<6 {
            let response = centers.map { f in
                zip(centers, gains).reduce(0.0) { sum, band in
                    sum + Biquad.peaking(frequency: band.0, gainDB: band.1, q: q, sampleRate: sampleRate)
                        .responseDB(at: f, sampleRate: sampleRate)
                }
            }
            for i in gains.indices { gains[i] = max(-12, min(12, gains[i] + target[i] - response[i])) }
        }
        return gains
    }

    /// Octave-wide peaking filters.
    static let q = 1.41

    static func filters(target: [Double], sampleRate: Double) -> [Biquad] {
        zip(centers, filterGains(target: target, sampleRate: sampleRate)).map {
            .peaking(frequency: $0.0, gainDB: $0.1, q: q, sampleRate: sampleRate)
        }
    }
}

extension Biquad {
    /// Magnitude response in dB at `f`.
    func responseDB(at f: Double, sampleRate: Double) -> Double {
        let w = 2 * Double.pi * f / sampleRate
        let (c1, s1, c2, s2) = (cos(w), sin(w), cos(2 * w), sin(2 * w))
        let nr = b0 + b1 * c1 + b2 * c2, ni = -(b1 * s1 + b2 * s2)
        let dr = 1 + a1 * c1 + a2 * c2, di = -(a1 * s1 + a2 * s2)
        return 10 * log10((nr * nr + ni * ni) / (dr * dr + di * di))
    }
}
