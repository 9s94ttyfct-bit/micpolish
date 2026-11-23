import AudioCore
import AVFoundation
import Sessions
import XCTest
@testable import Recording

final class RecordingTests: XCTestCase {
    private var dir: URL!

    override func setUpWithError() throws {
        dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: dir)
    }

    private func stereoBuffers(seconds: Double, sampleRate: Double, chunk: Int = 4410) -> (AVAudioFormat, [AVAudioPCMBuffer]) {
        let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 2)!
        var buffers: [AVAudioPCMBuffer] = []
        var remaining = Int(seconds * sampleRate)
        var t = 0
        while remaining > 0 {
            let n = min(chunk, remaining)
            let b = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(n))!
            b.frameLength = AVAudioFrameCount(n)
            for i in 0..<n {
                let s = Float(0.25 * sin(2 * Double.pi * 440 * Double(t + i) / sampleRate))
                b.floatChannelData![0][i] = s
                b.floatChannelData![1][i] = s * 0.5
            }
            buffers.append(b)
            t += n
            remaining -= n
        }
        return (format, buffers)
    }

    func testSink44k1StereoBecomes48kMono() throws {
        let (format, buffers) = stereoBuffers(seconds: 2, sampleRate: 44_100)
        let sink = try RecordingSink(url: dir.appendingPathComponent("r.caf"), inputSampleRate: Int(format.sampleRate))
        for b in buffers { try sink.append(b) }
        let url = try XCTUnwrap(try sink.finish())

        let reader = try MonoFileReader(url: url)
        XCTAssertEqual(reader.sampleRate, 48_000)
        XCTAssertEqual(Double(reader.frameCount) / 48_000, 2.0, accuracy: 0.01)
        let file = try AVAudioFile(forReading: url)
        XCTAssertEqual(file.fileFormat.channelCount, 1)
    }

    func testSinkPeakTracksInput() throws {
        let (format, buffers) = stereoBuffers(seconds: 0.2, sampleRate: 48_000)
        let sink = try RecordingSink(url: dir.appendingPathComponent("p.caf"), inputSampleRate: Int(format.sampleRate))
        try sink.append(buffers[0])
        // Mono mix of 0.25 and 0.125 peaks at 0.1875 → -14.5 dBFS.
        XCTAssertEqual(Double(sink.peakDBFS), 20 * log10(0.1875), accuracy: 0.3)
    }

    func testRecoveryAdoptsLeftoverCAFAndDeletesIt() throws {
        let cache = dir.appendingPathComponent("cache")
        let recordingDir = cache.appendingPathComponent("recording")
        try FileManager.default.createDirectory(at: recordingDir, withIntermediateDirectories: true)
        let leftover = recordingDir.appendingPathComponent("\(UUID().uuidString).caf")

        // Simulate a crash: copy the CAF while its writer is still open.
        let live = dir.appendingPathComponent("live.caf")
        let writer = try MonoFileWriter(url: live, sampleRate: 48_000, format: .cafFloat32)
        try writer.append([Float](repeating: 0.1, count: 96_000))
        try FileManager.default.copyItem(at: live, to: leftover)
        try writer.close()

        let store = SessionStore(root: dir.appendingPathComponent("sessions"))
        let recovered = try RecordingRecovery.recover(cacheRoot: cache, store: store, namePrefix: "Recovered recording")
        XCTAssertEqual(recovered.count, 1)
        XCTAssertTrue(recovered[0].name.hasPrefix("Recovered recording "), recovered[0].name)
        XCTAssertFalse(FileManager.default.fileExists(atPath: leftover.path))
        let original = store.originalURL(for: recovered[0])
        XCTAssertEqual(original.pathExtension, "flac")
        XCTAssertGreaterThan(try MonoFileReader(url: original).frameCount, 0)
    }

    /// Frames, level and waveform count only audio that reached the file.
    func testSinkRecordsWriteFailureAndCountsNothingUnwritten() throws {
        let (format, buffers) = stereoBuffers(seconds: 0.2, sampleRate: 48_000)
        let sink = try RecordingSink(url: dir.appendingPathComponent("f.caf"), inputSampleRate: Int(format.sampleRate))
        _ = try sink.finish()
        XCTAssertNil(sink.failure)
        XCTAssertThrowsError(try sink.append(buffers[0]))
        XCTAssertThrowsError(try sink.append(buffers[1]))
        XCTAssertNotNil(sink.failure)
        XCTAssertEqual(sink.framesWritten, 0)
        XCTAssertTrue(sink.takeEnvelope().isEmpty)
        XCTAssertEqual(sink.peakDBFS, -120)
    }

    /// A capture that never delivered audio leaves no file behind and nothing to adopt.
    func testSinkWithNoAudioFinishesWithoutAFile() throws {
        let url = dir.appendingPathComponent("empty.caf")
        let sink = try RecordingSink(url: url, inputSampleRate: 24_000)
        XCTAssertNil(try sink.finish())
        XCTAssertFalse(FileManager.default.fileExists(atPath: url.path))
    }

    func testSinkReportsLiveEnvelopeOnceAndDiscardDeletesTheFile() throws {
        let (format, buffers) = stereoBuffers(seconds: 1, sampleRate: 48_000)
        let url = dir.appendingPathComponent("live.caf")
        let sink = try RecordingSink(url: url, inputSampleRate: Int(format.sampleRate))
        for b in buffers { try sink.append(b) }
        let envelope = sink.takeEnvelope()
        XCTAssertEqual(envelope.count, 20)                       // 50 ms buckets
        for peak in envelope { XCTAssertEqual(peak, 0.1875, accuracy: 0.002) }
        XCTAssertTrue(sink.takeEnvelope().isEmpty)              // each bucket is handed out once
        sink.discard()
        XCTAssertFalse(FileManager.default.fileExists(atPath: url.path))
    }

    func testRecoveryKeepsCAFWhenSessionCannotBeCreated() throws {
        let cache = dir.appendingPathComponent("cache")
        let recordingDir = cache.appendingPathComponent("recording")
        try FileManager.default.createDirectory(at: recordingDir, withIntermediateDirectories: true)
        let leftover = recordingDir.appendingPathComponent("\(UUID().uuidString).caf")
        let writer = try MonoFileWriter(url: leftover, sampleRate: 48_000, format: .cafFloat32)
        try writer.append([Float](repeating: 0.1, count: 48_000))
        try writer.close()

        // Sessions root inside a read-only directory: adoption must fail.
        let locked = dir.appendingPathComponent("locked")
        try FileManager.default.createDirectory(at: locked, withIntermediateDirectories: true)
        try FileManager.default.setAttributes([.posixPermissions: 0o555], ofItemAtPath: locked.path)
        defer { try? FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: locked.path) }
        let store = SessionStore(root: locked.appendingPathComponent("sessions"))

        let recovered = try RecordingRecovery.recover(cacheRoot: cache, store: store, namePrefix: "Recovered")
        XCTAssertEqual(recovered.count, 0)
        XCTAssertTrue(FileManager.default.fileExists(atPath: leftover.path))
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: recordingDir.path), [leftover.lastPathComponent])
    }

    /// The FLAC is shorter than the capture (as after a failed final flush):
    /// the CAF, the only complete copy, stays and no session is created.
    func testRecoveryKeepsCAFWhenFLACIsShort() throws {
        let caf = dir.appendingPathComponent("short.caf")
        let writer = try MonoFileWriter(url: caf, sampleRate: 48_000, format: .cafFloat32)
        try writer.append([Float](repeating: 0.1, count: 48_000))
        try writer.close()
        let store = SessionStore(root: dir.appendingPathComponent("sessions"))

        XCTAssertThrowsError(try RecordingRecovery.adopt(caf: caf, name: "x", store: store) { reader, flac in
            let out = try MonoFileWriter(url: flac, sampleRate: reader.sampleRate, format: .flac24)
            try out.append(try XCTUnwrap(try reader.read(maxFrames: 24_000)))
            try out.close()
        }) { error in
            guard case RecordingRecoveryError.incompleteConversion(48_000, 24_000) = error else {
                return XCTFail("unexpected error \(error)")
            }
        }
        XCTAssertTrue(FileManager.default.fileExists(atPath: caf.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: caf.deletingPathExtension().appendingPathExtension("flac").path))
        XCTAssertEqual(try store.list().count, 0)
    }

    /// Every sample arrives once and in order while producer and consumer
    /// run concurrently and the ring wraps thousands of times.
    func testRingPreservesOrderAcrossWraparoundUnderConcurrency() {
        let ring = SampleRing(minimumCapacity: 1024)
        let total = 1 << 21                       // exactly representable as Float
        let producer = Thread {
            var next = 0
            var chunk = [Float](repeating: 0, count: 700)
            var size = 1
            while next < total {
                size = size % 699 + 1               // varying sizes, both sides of the wrap
                let n = min(size, total - next)
                for i in 0..<n { chunk[i] = Float(next + i) }
                let ok = chunk.withUnsafeBufferPointer { ring.write(UnsafeBufferPointer(rebasing: $0[0..<n])) }
                if ok { next += n }
            }
        }
        producer.start()
        var expected = 0
        var out = [Float](repeating: 0, count: 333)
        var mismatch: (Int, Float)?
        while expected < total {
            let n = out.withUnsafeMutableBufferPointer { ring.read(into: $0) }
            for i in 0..<n where mismatch == nil && out[i] != Float(expected + i) {
                mismatch = (expected + i, out[i])
            }
            expected += n
        }
        XCTAssertNil(mismatch.map { "sample \($0.0) read as \($0.1)" })
        XCTAssertEqual(expected, total)
    }

    /// A buffer that does not fit is dropped whole and counted, never written partially.
    func testRingReportsOverflow() {
        let ring = SampleRing(minimumCapacity: 8)
        XCTAssertEqual(ring.capacity, 8)
        let six: [Float] = [1, 2, 3, 4, 5, 6]
        XCTAssertTrue(six.withUnsafeBufferPointer { ring.write($0) })
        XCTAssertFalse([Float](repeating: 9, count: 4).withUnsafeBufferPointer { ring.write($0) })
        XCTAssertEqual(ring.droppedFrames, 4)
        var out = [Float](repeating: 0, count: 16)
        XCTAssertEqual(out.withUnsafeMutableBufferPointer { ring.read(into: $0) }, 6)
        XCTAssertEqual(Array(out[0..<6]), six)
        XCTAssertTrue([Float](repeating: 9, count: 4).withUnsafeBufferPointer { ring.write($0) })
        XCTAssertEqual(ring.droppedFrames, 4)
    }

    /// The writer thread drains everything, including what arrives just
    /// before stop, while a separate thread produces.
    func testWriterThreadWritesEveryFrameBeforeFinishing() throws {
        let ring = SampleRing(minimumCapacity: 48_000 * 10)
        let sink = try RecordingSink(url: dir.appendingPathComponent("w.caf"), input: ring, inputSampleRate: 48_000)
        sink.startWriting()
        let chunk = [Float](repeating: 0.2, count: 480)
        let produced = DispatchSemaphore(value: 0)
        Thread {
            for _ in 0..<500 { chunk.withUnsafeBufferPointer { _ = ring.write($0) } }
            produced.signal()
        }.start()
        produced.wait()
        let url = try XCTUnwrap(try sink.finish())
        XCTAssertEqual(ring.droppedFrames, 0)
        XCTAssertEqual(sink.framesWritten, 240_000)
        XCTAssertEqual(try MonoFileReader(url: url).frameCount, 240_000)
    }

    /// Capture rebuilt at a new device rate continues the same file without
    /// losing the audio buffered at the old rate.
    func testSinkSwitchesInputRateWithoutLosingAudio() throws {
        let first = SampleRing(minimumCapacity: 1 << 17)
        let sink = try RecordingSink(url: dir.appendingPathComponent("s.caf"), input: first, inputSampleRate: 48_000)
        let second = SampleRing(minimumCapacity: 1 << 17)
        _ = [Float](repeating: 0.1, count: 48_000).withUnsafeBufferPointer { first.write($0) }
        try sink.switchInput(to: second, sampleRate: 24_000)
        _ = [Float](repeating: 0.1, count: 24_000).withUnsafeBufferPointer { second.write($0) }
        sink.drain()
        let url = try XCTUnwrap(try sink.finish())
        XCTAssertNil(sink.failure)
        XCTAssertEqual(Double(try MonoFileReader(url: url).frameCount) / 48_000, 2.0, accuracy: 0.01)
    }
}

extension RecordingSink {
    /// A sink reading its own ring, for tests that feed it directly.
    convenience init(url: URL, inputSampleRate: Int) throws {
        try self.init(url: url, input: SampleRing(minimumCapacity: 1 << 16), inputSampleRate: inputSampleRate)
    }

    /// Pushes `buffer` through the ring and writes it, as capture and the
    /// writer thread would; throws the sink's failure.
    func append(_ buffer: AVAudioPCMBuffer) throws {
        input.write(UnsafeMutableAudioBufferListPointer(buffer.mutableAudioBufferList), frames: Int(buffer.frameLength))
        drain()
        if let failure { throw failure }
    }
}
