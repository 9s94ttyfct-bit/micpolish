import AVFoundation

/// Streams any AVFoundation-readable file as mono Float32 at its native rate.
/// Multichannel input is downmixed to the channel mean.
public final class MonoFileReader {
    public let sampleRate: Int
    public let frameCount: Int64

    private let file: AVAudioFile
    private var buffer: AVAudioPCMBuffer

    public init(url: URL) throws {
        file = try AVAudioFile(forReading: url, commonFormat: .pcmFormatFloat32, interleaved: false)
        sampleRate = Int(file.processingFormat.sampleRate.rounded())
        frameCount = file.length
        buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: 1)!
    }

    /// Returns up to `maxFrames` frames, or nil at end of file.
    /// AVAudioFile may return short reads, so this keeps reading until the
    /// request is satisfied or the file ends.
    public func read(maxFrames: Int) throws -> [Float]? {
        precondition(maxFrames > 0)
        guard file.framePosition < file.length else { return nil }
        if buffer.frameCapacity < AVAudioFrameCount(maxFrames) {
            buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(maxFrames))!
        }

        var output = [Float]()
        output.reserveCapacity(maxFrames)
        let channels = Int(file.processingFormat.channelCount)
        while output.count < maxFrames, file.framePosition < file.length {
            buffer.frameLength = 0
            try file.read(into: buffer, frameCount: AVAudioFrameCount(maxFrames - output.count))
            let frames = Int(buffer.frameLength)
            if frames == 0 { break }
            let data = buffer.floatChannelData!
            if channels == 1 {
                output.append(contentsOf: UnsafeBufferPointer(start: data[0], count: frames))
            } else {
                let scale = 1 / Float(channels)
                for i in 0..<frames {
                    var sum: Float = 0
                    for c in 0..<channels { sum += data[c][i] }
                    output.append(sum * scale)
                }
            }
        }
        return output.isEmpty ? nil : output
    }
}

/// Reads a whole file as mono Float32. Intended for short files and tests;
/// long-form processing streams through `MonoFileReader`.
public func readAllMono(_ url: URL) throws -> (samples: [Float], sampleRate: Int) {
    let reader = try MonoFileReader(url: url)
    var samples = [Float]()
    samples.reserveCapacity(Int(reader.frameCount))
    while let chunk = try reader.read(maxFrames: 65_536) {
        samples.append(contentsOf: chunk)
    }
    return (samples, reader.sampleRate)
}
