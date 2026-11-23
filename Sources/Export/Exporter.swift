import AudioCore
import AVFoundation

public enum ExportFormat: String, Codable, Sendable, CaseIterable {
    case m4a
    case mp3
}

public struct ExportOptions: Codable, Equatable, Sendable {
    public static let bitrates = [96, 128, 192]

    public var format: ExportFormat = .m4a
    public var bitrateKbps: Int = 128

    public init(format: ExportFormat = .m4a, bitrateKbps: Int = 128) {
        self.format = format
        self.bitrateKbps = bitrateKbps
    }
}

public enum ExportError: Error, Equatable, LocalizedError {
    case encoderFailed(String)
    case verificationFailed(expectedSeconds: Double, actualSeconds: Double)

    public var errorDescription: String? {
        switch self {
        case .encoderFailed(let detail):
            "The encoder failed (\(detail))."
        case .verificationFailed(let expected, let actual):
            String(format: "The exported file lasts %.1f s instead of %.1f s, so it was discarded.", actual, expected)
        }
    }
}

protocol MonoEncoder {
    func append(_ samples: [Float]) throws
    func finish() throws
}

/// Encodes a mono source to M4A (AAC-LC) or MP3, writing to a temporary file
/// beside `destination` and replacing it only after the result decodes with
/// the source's duration. On failure the existing destination is untouched.
public enum Exporter {
    public static func export(from source: MonoFileReader, options: ExportOptions, to destination: URL) throws {
        let fm = FileManager.default
        let temp = destination.deletingLastPathComponent()
            .appendingPathComponent(".\(destination.deletingPathExtension().lastPathComponent)-\(UUID().uuidString).\(options.format.rawValue)")
        defer { try? fm.removeItem(at: temp) }

        let encoder: MonoEncoder = switch options.format {
        case .m4a: try AACEncoder(url: temp, sampleRate: source.sampleRate, bitrateKbps: options.bitrateKbps)
        case .mp3: try MP3Encoder(url: temp, sampleRate: source.sampleRate, bitrateKbps: options.bitrateKbps)
        }
        while let block = try source.read(maxFrames: 65_536) {
            try encoder.append(block)
        }
        try encoder.finish()

        let expected = Double(source.frameCount) / Double(source.sampleRate)
        let check = try AVAudioFile(forReading: temp)
        let actual = Double(check.length) / check.fileFormat.sampleRate
        guard abs(actual - expected) <= 0.1 else {
            throw ExportError.verificationFailed(expectedSeconds: expected, actualSeconds: actual)
        }

        if fm.fileExists(atPath: destination.path) {
            _ = try fm.replaceItemAt(destination, withItemAt: temp)
        } else {
            try fm.moveItem(at: temp, to: destination)
        }
    }
}

final class AACEncoder: MonoEncoder {
    private var file: AVAudioFile?
    private let format: AVAudioFormat

    init(url: URL, sampleRate: Int, bitrateKbps: Int) throws {
        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatMPEG4AAC,
            AVSampleRateKey: sampleRate,
            AVNumberOfChannelsKey: 1,
            AVEncoderBitRateKey: bitrateKbps * 1000,
            AVEncoderAudioQualityKey: AVAudioQuality.max.rawValue,
        ]
        let file = try AVAudioFile(forWriting: url, settings: settings, commonFormat: .pcmFormatFloat32, interleaved: false)
        self.file = file
        format = file.processingFormat
    }

    func append(_ samples: [Float]) throws {
        guard let file, !samples.isEmpty else { return }
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count))!
        buffer.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { buffer.floatChannelData![0].update(from: $0.baseAddress!, count: samples.count) }
        try file.write(from: buffer)
    }

    func finish() throws {
        file = nil
    }
}
