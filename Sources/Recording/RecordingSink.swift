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
    private var storedFailure: Error?
    /// Peaks of completed 50 ms buckets not yet taken by `takeEnvelope()`.
    private var pendingBuckets: [Float] = []
    private var bucketPeak: Float = 0
    private var bucketFill = 0
    /// Samples (48 kHz) per waveform bucket.
    public static let bucketFrames = AudioCore.sampleRate / 20

    /// The first write error, if any. Capture runs on Core Audio's thread, so
    /// the recorder polls this to notice a full disk instead of losing audio
    /// silently.
    public var failure: Error? {
        lock.lock()
        defer { lock.unlock() }
        return storedFailure
    }

    public init(url: URL, inputSampleRate: Int) throws {
        self.url = url
        writer = try MonoFileWriter(url: url, sampleRate: AudioCore.sampleRate, format: .cafFloat32)
        resampler = inputSampleRate == AudioCore.sampleRate
            ? nil : try StreamingResampler(from: inputSampleRate, to: AudioCore.sampleRate)
    }

    /// Peak of the most recent buffer, in dBFS (−120 for silence).
    public var peakDBFS: Float {
        lock.lock()
        defer { lock.unlock() }
        return lastPeak > 0 ? 20 * log10(lastPeak) : -120
    }

    public func append(_ buffer: AVAudioPCMBuffer) throws {
        try append(UnsafeMutableAudioBufferListPointer(buffer.mutableAudioBufferList), frames: Int(buffer.frameLength))
    }

    /// Appends `frames` frames of Float32 audio from any mix of planar and
    /// interleaved buffers; all channels are averaged to mono.
    public func append(_ buffers: UnsafeMutableAudioBufferListPointer, frames: Int) throws {
        do {
            try write(buffers, frames: frames)
        } catch {
            lock.lock()
            if storedFailure == nil { storedFailure = error }
            lock.unlock()
            throw error
        }
    }

    private func write(_ buffers: UnsafeMutableAudioBufferListPointer, frames: Int) throws {
        let channels = buffers.reduce(0) { $0 + Int($1.mNumberChannels) }
        guard frames > 0, channels > 0 else { return }
        var mono = [Float](repeating: 0, count: frames)
        for buffer in buffers {
            guard let data = buffer.mData?.assumingMemoryBound(to: Float.self) else { continue }
            let stride = Int(buffer.mNumberChannels)
            for c in 0..<stride {
                for i in 0..<frames { mono[i] += data[i * stride + c] }
            }
        }
        let scale = 1 / Float(channels)
        var peak: Float = 0
        for i in 0..<frames {
            mono[i] *= scale
            peak = max(peak, abs(mono[i]))
        }
        let out = try resampler?.process(mono) ?? mono
        lock.lock()
        lastPeak = peak
        framesWritten += out.count
        for s in out {
            bucketPeak = max(bucketPeak, abs(s))
            bucketFill += 1
            if bucketFill == Self.bucketFrames {
                pendingBuckets.append(bucketPeak)
                bucketPeak = 0
                bucketFill = 0
            }
        }
        lock.unlock()
        try writer.append(out)
    }

    /// Returns the 50 ms peak buckets completed since the last call, for a
    /// live waveform.
    public func takeEnvelope() -> [Float] {
        lock.withLock {
            defer { pendingBuckets.removeAll(keepingCapacity: true) }
            return pendingBuckets
        }
    }

    /// Closes the file and deletes it: the recording is thrown away.
    public func discard() {
        try? writer.close()
        try? FileManager.default.removeItem(at: url)
    }

    /// Closes the file. Returns nil, and deletes it, when nothing was captured.
    public func finish() throws -> URL? {
        if let tail = try resampler?.flush() {
            try writer.append(tail)
        }
        try writer.close()
        guard lock.withLock({ framesWritten }) > 0 else {
            try? FileManager.default.removeItem(at: url)
            return nil
        }
        return url
    }
}
