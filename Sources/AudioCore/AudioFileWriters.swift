import AVFoundation

/// Appends mono Float32 samples to a CAF (Float32) or FLAC (24-bit) file.
public final class MonoFileWriter {
    public enum Format: Sendable {
        case cafFloat32
        case flac24
    }

    private var file: AVAudioFile?
    private let format: AVAudioFormat

    public init(url: URL, sampleRate: Int, format: Format) throws {
        let settings: [String: Any]
        switch format {
        case .cafFloat32:
            settings = [
                AVFormatIDKey: kAudioFormatLinearPCM,
                AVSampleRateKey: sampleRate,
                AVNumberOfChannelsKey: 1,
                AVLinearPCMBitDepthKey: 32,
                AVLinearPCMIsFloatKey: true,
                AVLinearPCMIsNonInterleaved: false,
            ]
        case .flac24:
            settings = [
                AVFormatIDKey: kAudioFormatFLAC,
                AVSampleRateKey: sampleRate,
                AVNumberOfChannelsKey: 1,
                AVEncoderBitDepthHintKey: 24,
            ]
        }
        try? FileManager.default.removeItem(at: url)
        let file = try AVAudioFile(
            forWriting: url, settings: settings, commonFormat: .pcmFormatFloat32, interleaved: false)
        self.file = file
        self.format = file.processingFormat
    }

    public func append(_ samples: [Float]) throws {
        guard let file else { throw CocoaError(.fileWriteUnknown) }
        guard !samples.isEmpty else { return }
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count))!
        buffer.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer {
            buffer.floatChannelData![0].update(from: $0.baseAddress!, count: samples.count)
        }
        try file.write(from: buffer)
    }

    /// Finalizes the file. AVAudioFile flushes and closes when released.
    public func close() throws {
        file = nil
    }
}
