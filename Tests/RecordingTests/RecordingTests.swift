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
        let recovered = try RecordingRecovery.recover(cacheRoot: cache, store: store)
        XCTAssertEqual(recovered.count, 1)
        XCTAssertFalse(FileManager.default.fileExists(atPath: leftover.path))
        let original = store.originalURL(for: recovered[0])
        XCTAssertEqual(original.pathExtension, "flac")
        XCTAssertGreaterThan(try MonoFileReader(url: original).frameCount, 0)
    }

    func testSinkRecordsWriteFailure() throws {
        let (format, buffers) = stereoBuffers(seconds: 0.2, sampleRate: 48_000)
        let sink = try RecordingSink(url: dir.appendingPathComponent("f.caf"), inputSampleRate: Int(format.sampleRate))
        _ = try sink.finish()
        XCTAssertNil(sink.failure)
        XCTAssertThrowsError(try sink.append(buffers[0]))
        XCTAssertNotNil(sink.failure)
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

        let recovered = try RecordingRecovery.recover(cacheRoot: cache, store: store)
        XCTAssertEqual(recovered.count, 0)
        XCTAssertTrue(FileManager.default.fileExists(atPath: leftover.path))
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: recordingDir.path), [leftover.lastPathComponent])
    }
}
