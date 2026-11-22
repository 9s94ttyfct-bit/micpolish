import CLame
import Foundation

/// Constant-bitrate mono MP3 through LAME, with the LAME/Xing info frame
/// rewritten at the start so players report exact duration and gapless info.
final class MP3Encoder: MonoEncoder {
    private let gf: OpaquePointer
    private let handle: FileHandle
    private var output = [UInt8]()

    init(url: URL, sampleRate: Int, bitrateKbps: Int) throws {
        guard let gf = lame_init() else { throw ExportError.encoderFailed("lame_init") }
        self.gf = gf
        lame_set_in_samplerate(gf, Int32(sampleRate))
        lame_set_out_samplerate(gf, Int32(sampleRate))
        lame_set_num_channels(gf, 1)
        lame_set_mode(gf, MONO)
        lame_set_brate(gf, Int32(bitrateKbps))
        lame_set_VBR(gf, vbr_off)
        lame_set_quality(gf, 2)
        lame_set_bWriteVbrTag(gf, 1)
        guard lame_init_params(gf) >= 0 else {
            lame_close(gf)
            throw ExportError.encoderFailed("lame_init_params")
        }
        guard FileManager.default.createFile(atPath: url.path, contents: nil) else {
            lame_close(gf)
            throw CocoaError(.fileWriteNoPermission)
        }
        handle = try FileHandle(forWritingTo: url)
    }

    deinit {
        lame_close(gf)
        try? handle.close()
    }

    func append(_ samples: [Float]) throws {
        guard !samples.isEmpty else { return }
        let capacity = samples.count * 5 / 4 + 7200
        if output.count < capacity { output = [UInt8](repeating: 0, count: capacity) }
        let written = samples.withUnsafeBufferPointer { pcm in
            output.withUnsafeMutableBufferPointer { out in
                lame_encode_buffer_ieee_float(gf, pcm.baseAddress, pcm.baseAddress, Int32(samples.count),
                                              out.baseAddress, Int32(capacity))
            }
        }
        guard written >= 0 else { throw ExportError.encoderFailed("lame_encode_buffer_ieee_float \(written)") }
        try handle.write(contentsOf: output[0..<Int(written)])
    }

    func finish() throws {
        var tail = [UInt8](repeating: 0, count: 7200)
        let written = tail.withUnsafeMutableBufferPointer { lame_encode_flush(gf, $0.baseAddress, 7200) }
        guard written >= 0 else { throw ExportError.encoderFailed("lame_encode_flush \(written)") }
        try handle.write(contentsOf: tail[0..<Int(written)])

        var tag = [UInt8](repeating: 0, count: 2880)
        let tagSize = tag.withUnsafeMutableBufferPointer { lame_get_lametag_frame(gf, $0.baseAddress, 2880) }
        if tagSize > 0 && tagSize <= 2880 {
            try handle.seek(toOffset: 0)
            try handle.write(contentsOf: tag[0..<tagSize])
        }
        try handle.synchronize()
        try handle.close()
    }
}
