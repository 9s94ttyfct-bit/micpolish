import Foundation

/// Head/tail trimming of mono audio files.
public enum AudioTrim {
    /// Fade applied at a cut so it does not click.
    public static let fadeSeconds = 0.005

    /// Writes the `[start, end)` seconds of `input` to a Float32 CAF at the
    /// input's sample rate. Cut edges (not the file's own start or end) get a
    /// 5 ms linear fade.
    public static func write(_ input: URL, from start: Double, to end: Double, output: URL) throws {
        let reader = try MonoFileReader(url: input)
        let rate = Double(reader.sampleRate)
        let first = max(0, min(reader.frameCount, Int64((start * rate).rounded())))
        let last = max(first, min(reader.frameCount, Int64((end * rate).rounded())))
        let fade = Int(fadeSeconds * rate)
        let fadeIn = first > 0 ? fade : 0
        let fadeOut = last < reader.frameCount ? fade : 0
        let total = Int(last - first)

        let writer = try MonoFileWriter(url: output, sampleRate: reader.sampleRate, format: .cafFloat32)
        reader.seek(toFrame: first)
        var written = 0
        while written < total, var block = try reader.read(maxFrames: min(65_536, total - written)) {
            for i in block.indices {
                let n = written + i
                if n < fadeIn { block[i] *= Float(n) / Float(fadeIn) }
                let fromEnd = total - 1 - n
                if fromEnd < fadeOut { block[i] *= Float(fromEnd) / Float(fadeOut) }
            }
            try writer.append(block)
            written += block.count
        }
        try writer.close()
    }

    /// The span from the first to the last bucket of a peak `envelope` that is
    /// within `thresholdBelowPeakDB` of its loudest bucket, widened by
    /// `paddingSeconds` on each side. Nil when the envelope is silent.
    public static func detectSoundRange(
        envelope: [Float], bucketSeconds: Double,
        thresholdBelowPeakDB: Double = 40, paddingSeconds: Double = 0.25
    ) -> (start: Double, end: Double)? {
        guard let peak = envelope.max(), peak > 1e-6 else { return nil }
        let threshold = peak * Float(pow(10, -thresholdBelowPeakDB / 20))
        guard let first = envelope.firstIndex(where: { $0 >= threshold }),
              let last = envelope.lastIndex(where: { $0 >= threshold }) else { return nil }
        let duration = Double(envelope.count) * bucketSeconds
        return (max(0, Double(first) * bucketSeconds - paddingSeconds),
                min(duration, Double(last + 1) * bucketSeconds + paddingSeconds))
    }
}
