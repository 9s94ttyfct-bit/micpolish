import AVFoundation

public enum ResamplerError: Error {
    case unsupportedRates(from: Int, to: Int)
    case conversionFailed(String)
}

/// Mono Float32 sample-rate conversion that can be fed in arbitrary chunks;
/// call `flush()` once at the end to drain the filter tail.
public final class StreamingResampler {
    private let converter: AVAudioConverter
    private let inputFormat: AVAudioFormat
    private let outputFormat: AVAudioFormat
    private let ratio: Double

    public init(from: Int, to: Int) throws {
        guard from > 0, to > 0,
              let inFormat = AVAudioFormat(standardFormatWithSampleRate: Double(from), channels: 1),
              let outFormat = AVAudioFormat(standardFormatWithSampleRate: Double(to), channels: 1),
              let converter = AVAudioConverter(from: inFormat, to: outFormat)
        else { throw ResamplerError.unsupportedRates(from: from, to: to) }
        converter.sampleRateConverterQuality = AVAudioQuality.max.rawValue
        converter.sampleRateConverterAlgorithm = AVSampleRateConverterAlgorithm_Mastering
        self.converter = converter
        self.inputFormat = inFormat
        self.outputFormat = outFormat
        self.ratio = Double(to) / Double(from)
    }

    public func process(_ samples: [Float]) throws -> [Float] {
        guard !samples.isEmpty else { return [] }
        let input = AVAudioPCMBuffer(pcmFormat: inputFormat, frameCapacity: AVAudioFrameCount(samples.count))!
        input.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer {
            input.floatChannelData![0].update(from: $0.baseAddress!, count: samples.count)
        }
        var supplied = false
        return try drain(capacity: Int(Double(samples.count) * ratio) + 1024) { status in
            if supplied {
                status.pointee = .noDataNow
                return nil
            }
            supplied = true
            status.pointee = .haveData
            return input
        }
    }

    public func flush() throws -> [Float] {
        try drain(capacity: 8192) { status in
            status.pointee = .endOfStream
            return nil
        }
    }

    private func drain(
        capacity: Int,
        input: @escaping (UnsafeMutablePointer<AVAudioConverterInputStatus>) -> AVAudioPCMBuffer?
    ) throws -> [Float] {
        var result = [Float]()
        while true {
            let output = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: AVAudioFrameCount(capacity))!
            var error: NSError?
            let status = converter.convert(to: output, error: &error) { _, statusPointer in
                input(statusPointer)
            }
            if let error { throw ResamplerError.conversionFailed(error.localizedDescription) }
            result.append(contentsOf: UnsafeBufferPointer(start: output.floatChannelData![0], count: Int(output.frameLength)))
            switch status {
            case .haveData where output.frameLength == output.frameCapacity:
                continue
            case .error:
                throw ResamplerError.conversionFailed("AVAudioConverter error")
            default:
                return result
            }
        }
    }
}
