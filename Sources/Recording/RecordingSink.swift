import AudioCore
import AVFoundation

/// Writes captured buffers as 48 kHz mono Float32 CAF: multichannel input is
/// downmixed to the channel mean and other rates are resampled. The CAF is
/// written as audio arrives, so a crash leaves a recoverable file.
public final class RecordingSink: @unchecked Sendable {
    private let url: URL
    private let writer: MonoFileWriter
    private let resampler: StreamingResampler?
    private let lock = NSLock()
    private var lastPeak: Float = 0
    public private(set) var framesWritten = 0

    public init(url: URL, inputFormat: AVAudioFormat) throws {
        self.url = url
        let rate = Int(inputFormat.sampleRate.rounded())
        writer = try MonoFileWriter(url: url, sampleRate: AudioCore.sampleRate, format: .cafFloat32)
        resampler = rate == AudioCore.sampleRate ? nil : try StreamingResampler(from: rate, to: AudioCore.sampleRate)
    }

    /// Peak of the most recent buffer, in dBFS (−120 for silence).
    public var peakDBFS: Float {
        lock.lock()
        defer { lock.unlock() }
        return lastPeak > 0 ? 20 * log10(lastPeak) : -120
    }

    public func append(_ buffer: AVAudioPCMBuffer) throws {
        let frames = Int(buffer.frameLength)
        guard frames > 0, let data = buffer.floatChannelData else { return }
        let channels = Int(buffer.format.channelCount)
        var mono = [Float](repeating: 0, count: frames)
        var peak: Float = 0
        let scale = 1 / Float(channels)
        let interleaved = buffer.format.isInterleaved
        for i in 0..<frames {
            var sum: Float = 0
            for c in 0..<channels {
                sum += interleaved ? data[0][i * channels + c] : data[c][i]
            }
            let s = sum * scale
            mono[i] = s
            peak = max(peak, abs(s))
        }
        let out = try resampler?.process(mono) ?? mono
        lock.lock()
        lastPeak = peak
        framesWritten += out.count
        lock.unlock()
        try writer.append(out)
    }

    public func finish() throws -> URL {
        if let tail = try resampler?.flush() {
            try writer.append(tail)
        }
        try writer.close()
        return url
    }
}
