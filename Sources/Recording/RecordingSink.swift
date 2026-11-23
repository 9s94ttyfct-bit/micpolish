import AudioCore
import AVFoundation

/// Writes captured audio as 48 kHz mono Float32 CAF. Capture only pushes
/// samples into a `SampleRing`; a writer thread drains it, resamples other
/// rates, tracks the level and waveform and writes the file, so Core Audio's
/// I/O thread never waits on the disk or on the main thread. The CAF is
/// written as audio arrives, so a crash leaves a recoverable file.
public final class RecordingSink: @unchecked Sendable {
    private let url: URL
    /// Writer-side state: touched only by the writer thread while it runs,
    /// and by `finish`/`discard` after it stopped (or tests that drive
    /// `drain()` without a thread).
    private let writer: MonoFileWriter
    private(set) var input: SampleRing
    private var resampler: StreamingResampler?
    private var scratch: [Float]
    private var bucketPeak: Float = 0
    private var bucketFill = 0

    /// Shared with other threads; guarded by `lock`. The I/O thread never takes it.
    private let lock = NSLock()
    private var lastPeak: Float = 0
    private var written = 0
    private var storedFailure: Error?
    /// Peaks of completed 50 ms buckets not yet taken by `takeEnvelope()`.
    private var pendingBuckets: [Float] = []
    private var pendingInput: (ring: SampleRing, resampler: StreamingResampler?)?
    private var stopRequested = false
    private var writerThread: Thread?
    private let writerDone = DispatchSemaphore(value: 0)

    /// Samples (48 kHz) per waveform bucket.
    public static let bucketFrames = AudioCore.sampleRate / 20

    /// Reads `input`, which holds mono audio at `inputSampleRate`.
    init(url: URL, input: SampleRing, inputSampleRate: Int) throws {
        self.url = url
        self.input = input
        resampler = try Self.resampler(from: inputSampleRate)
        scratch = [Float](repeating: 0, count: min(input.capacity, 65_536))
        writer = try MonoFileWriter(url: url, sampleRate: AudioCore.sampleRate, format: .cafFloat32)
    }

    private static func resampler(from rate: Int) throws -> StreamingResampler? {
        rate == AudioCore.sampleRate ? nil : try StreamingResampler(from: rate, to: AudioCore.sampleRate)
    }

    /// 48 kHz frames written to the file so far; counts only successful writes.
    public var framesWritten: Int { lock.withLock { written } }

    /// The first write error, if any. Writing runs on its own thread, so the
    /// recorder polls this to notice a full disk instead of losing audio
    /// silently. Once set, further audio is discarded.
    public var failure: Error? { lock.withLock { storedFailure } }

    /// Peak of the most recently written audio, in dBFS (−120 for silence).
    public var peakDBFS: Float {
        let peak = lock.withLock { lastPeak }
        return peak > 0 ? 20 * log10(peak) : -120
    }

    /// Returns the 50 ms peak buckets completed since the last call, for a
    /// live waveform.
    public func takeEnvelope() -> [Float] {
        lock.withLock {
            defer { pendingBuckets.removeAll(keepingCapacity: true) }
            return pendingBuckets
        }
    }

    /// Continues the recording from `ring` at `sampleRate`, after the
    /// producer of the current ring has stopped for good: the writer first
    /// drains the old ring and the old resampler's tail, so no audio is lost
    /// when capture is rebuilt at a new device rate.
    func switchInput(to ring: SampleRing, sampleRate: Int) throws {
        let resampler = try Self.resampler(from: sampleRate)
        lock.withLock { pendingInput = (ring, resampler) }
    }

    // MARK: Writer thread

    /// Starts the thread that drains the ring every 10 ms.
    func startWriting() {
        let thread = Thread { [self] in
            while true {
                let stop = lock.withLock { stopRequested }
                drain()
                if stop { break }
                Thread.sleep(forTimeInterval: 0.01)
            }
            writerDone.signal()
        }
        thread.name = "MicPolish recording writer"
        thread.qualityOfService = .userInitiated
        writerThread = thread
        thread.start()
    }

    /// Asks the writer thread for a last drain and waits until it exits.
    private func stopWriting() {
        guard writerThread != nil else { return }
        writerThread = nil
        lock.withLock { stopRequested = true }
        writerDone.wait()
    }

    /// Writes everything currently in the ring (and switches input when one is
    /// pending). Errors are stored in `failure`.
    func drain() {
        drainCurrentInput()
        guard let next = lock.withLock({ () -> (ring: SampleRing, resampler: StreamingResampler?)? in
            defer { pendingInput = nil }
            return pendingInput
        }) else { return }
        // The old ring's producer stopped before the switch was requested, so
        // this drain sees its last samples.
        drainCurrentInput()
        flushResampler()
        input = next.ring
        resampler = next.resampler
        drainCurrentInput()
    }

    private func drainCurrentInput() {
        while true {
            let count = scratch.withUnsafeMutableBufferPointer { input.read(into: $0) }
            guard count > 0 else { return }
            guard failure == nil else { continue }    // discard after a failure
            var peak: Float = 0
            for i in 0..<count { peak = max(peak, abs(scratch[i])) }
            let mono = Array(scratch[0..<count])
            guard let out = attempt({ try resampler?.process(mono) ?? mono }) else { continue }
            write(out, peak: peak)
        }
    }

    /// Writes the resampler's filter tail; the resampler cannot be fed afterwards.
    private func flushResampler() {
        guard failure == nil, let resampler, let tail = attempt({ try resampler.flush() }) else { return }
        write(tail, peak: nil)
    }

    /// Appends `samples` and, only once the write succeeded, counts them and
    /// updates the level and waveform.
    private func write(_ samples: [Float], peak: Float?) {
        guard attempt({ try writer.append(samples) }) != nil else { return }
        var completed: [Float] = []
        for s in samples {
            bucketPeak = max(bucketPeak, abs(s))
            bucketFill += 1
            if bucketFill == Self.bucketFrames {
                completed.append(bucketPeak)
                bucketPeak = 0
                bucketFill = 0
            }
        }
        lock.withLock {
            if let peak { lastPeak = peak }
            written += samples.count
            pendingBuckets += completed
        }
    }

    /// Runs `body`, storing the first error in `failure`.
    private func attempt<T>(_ body: () throws -> T) -> T? {
        do {
            return try body()
        } catch {
            lock.withLock { if storedFailure == nil { storedFailure = error } }
            return nil
        }
    }

    // MARK: Ending

    /// Closes the file and deletes it: the recording is thrown away. Waits
    /// for the writer thread, so call it off the main thread.
    public func discard() {
        stopWriting()
        try? writer.close()
        try? FileManager.default.removeItem(at: url)
    }

    /// Writes the remaining audio and closes the file. Returns nil, and
    /// deletes the file, when no audio was written. Throws when the last
    /// audio could not be written (the file is kept). Waits for the writer
    /// thread, so call it off the main thread.
    public func finish() throws -> URL? {
        stopWriting()
        let failedBefore = failure != nil
        drain()
        flushResampler()
        try writer.close()
        guard framesWritten > 0 else {
            try? FileManager.default.removeItem(at: url)
            return nil
        }
        // An earlier failure was already reported as an interruption and the
        // file holds the audio before it; a new one means the end is missing.
        if !failedBefore, let failure { throw failure }
        return url
    }
}
