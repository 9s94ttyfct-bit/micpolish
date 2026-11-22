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

    /// The span that holds the recording's sound, widened by `paddingSeconds`
    /// on each side, from a peak `envelope`. Nil when the envelope is silent.
    ///
    /// "Sound" is a bucket at least halfway (in dB) from the noise floor (10th
    /// percentile of non-digital-silence buckets) to the loudest bucket, with
    /// another such bucket within the next two (the previous two, for the
    /// end), so an isolated click or the room tone does not count.
    public static func detectSoundRange(
        envelope: [Float], bucketSeconds: Double, paddingSeconds: Double = 0.25
    ) -> (start: Double, end: Double)? {
        guard let peak = envelope.max(), peak > 1e-6 else { return nil }
        let levels = envelope.map { 20 * log10(Double(max($0, 1e-12))) }
        let audible = levels.filter { $0 > -120 }.sorted()
        let floor = audible[audible.count / 10]
        let threshold = floor + (20 * log10(Double(peak)) - floor) / 2
        let loud = levels.map { $0 >= threshold }
        func isLoud(_ i: Int) -> Bool { loud.indices.contains(i) && loud[i] }
        guard let first = loud.indices.first(where: { isLoud($0) && (isLoud($0 + 1) || isLoud($0 + 2)) }),
              let last = loud.indices.last(where: { isLoud($0) && (isLoud($0 - 1) || isLoud($0 - 2)) }) else { return nil }
        let duration = Double(envelope.count) * bucketSeconds
        return (max(0, Double(first) * bucketSeconds - paddingSeconds),
                min(duration, Double(last + 1) * bucketSeconds + paddingSeconds))
    }
}
