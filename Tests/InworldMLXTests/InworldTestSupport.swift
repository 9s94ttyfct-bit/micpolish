import AVFoundation
import Foundation
import MLX
import XCTest
@testable import InworldMLX

enum InworldFixtures {
    static let directory = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Fixtures/inworld")

    static func weightsURL() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment["MICPOLISH_INWORLD_WEIGHTS"],
              FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("Set MICPOLISH_INWORLD_WEIGHTS to the voice-isolator-v1 denoiser.safetensors")
        }
        return URL(fileURLWithPath: path)
    }

    static func readWAV(_ url: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        XCTAssertEqual(file.processingFormat.sampleRate, Double(InworldIsolator.sampleRate))
        XCTAssertEqual(file.processingFormat.channelCount, 1)
        // AVAudioFile.read(into:) may return fewer frames than requested; read to EOF.
        var samples = [Float]()
        let buffer = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: 65_536))
        while file.framePosition < file.length {
            try file.read(into: buffer)
            samples += UnsafeBufferPointer(start: buffer.floatChannelData![0], count: Int(buffer.frameLength))
        }
        return samples
    }

    static func readFloat32(_ url: URL) throws -> [Float] {
        try Data(contentsOf: url).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    /// A 1-D little-endian float32 `.npy` array (format 1.0 or 2.0).
    static func readNpy(_ url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        XCTAssertEqual(data.prefix(6), Data([0x93] + Array("NUMPY".utf8)))
        let major = data[6]
        let headerLength = major == 1
            ? Int(data[8]) | Int(data[9]) << 8
            : Int(data[8]) | Int(data[9]) << 8 | Int(data[10]) << 16 | Int(data[11]) << 24
        let headerStart = major == 1 ? 10 : 12
        let header = String(decoding: data[headerStart ..< headerStart + headerLength], as: UTF8.self)
        XCTAssertTrue(header.contains("'descr': '<f4'"), header)
        XCTAssertTrue(header.contains("'fortran_order': False"), header)
        return data[(headerStart + headerLength)...].withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    /// Deterministic noise in [-0.5, 0.5).
    static func noise(_ count: Int, seed: UInt64) -> [Float] {
        var state = seed
        return (0..<count).map { _ in
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float(Int64(bitPattern: state >> 11) % 20_000) / 20_000 - 0.5
        }
    }

    /// Pushes `samples` in the cycling piece sizes, then finishes.
    static func run(_ session: InworldSession, _ samples: [Float], pieces: [Int]) throws -> [Float] {
        var output = [Float]()
        var position = 0, index = 0
        while position < samples.count {
            let end = min(samples.count, position + pieces[index % pieces.count])
            output += try session.push(Array(samples[position ..< end]))
            position = end
            index += 1
        }
        return output + (try session.finish())
    }

    static func relativeRMSE(_ a: [Float], _ reference: [Float]) -> Double {
        var errorEnergy = 0.0, referenceEnergy = 0.0
        for (x, r) in zip(a, reference) {
            errorEnergy += Double(x - r) * Double(x - r)
            referenceEnergy += Double(r) * Double(r)
        }
        return (errorEnergy / referenceEnergy).squareRoot()
    }

    /// Upstream `VoiceIsolatorNetwork.enhance` written out directly (whole clip
    /// in memory), the reference for the streaming stitcher.
    static func enhance(_ audio: [Float], chunk: Int, overlap: Int, render: ([Float]) throws -> [Float]) rethrows
        -> [Float]
    {
        let n = audio.count
        if n <= chunk { return try render(audio) }
        var out = [Float](repeating: 0, count: n)
        var weight = [Float](repeating: 0, count: n)
        var starts = Array(Swift.stride(from: 0, to: n - chunk, by: chunk - overlap))
        starts.append(n - chunk)
        for (index, start) in starts.enumerated() {
            let end = start + chunk
            var w = [Float](repeating: 1, count: chunk)
            if index > 0 {
                let fadeIn = starts[index - 1] + chunk - overlap - start
                for i in 0..<fadeIn { w[i] = 0 }
                if overlap > 0 {
                    for (i, value) in InworldStitcher.linspace(0, 1, overlap).enumerated() { w[fadeIn + i] = value }
                }
            }
            if overlap > 0 && end < n {
                for (i, value) in InworldStitcher.linspace(1, 0, overlap).enumerated() { w[chunk - overlap + i] = value }
            }
            let rendered = try render(Array(audio[start ..< end]))
            for i in 0..<chunk {
                out[start + i] += rendered[i] * w[i]
                weight[start + i] += w[i]
            }
        }
        return (0..<n).map { out[$0] / max(weight[$0], 1e-6) }
    }
}

/// A small random checkpoint in the upstream key layout, for weight-free tests.
enum TinyCheckpoint {
    static var config: [String: Any] {
        [
            "width": 16, "num_heads": 2, "n_fft": 64, "hop_length": 16, "sample_rate": 48_000,
            "band_plan": [[1500, 12000], [6000, 24000]], "block_types": ["transformer", "mamba"],
            "correction_bound": 0.5, "chunk_seconds": 0.01, "overlap_seconds": 0.002,
            "exclusive_self_attention": true, "exclusive_axes": "time",
            "mamba_state": 4, "mamba_conv": 4, "mamba_expand": 2,
            "causal": false, "dual_mode": false, "lookahead_frames": 0, "lookahead_frames_choices": [Int](),
            "mode_film": false, "attention_window_frames": 0, "offline_passes": 1, "activation_checkpointing": true,
        ]
    }

    static func json(_ config: [String: Any] = config) -> String {
        String(decoding: try! JSONSerialization.data(withJSONObject: config, options: .sortedKeys), as: UTF8.self)
    }

    static func tensors() throws -> [String: MLXArray] {
        let parsed = try InworldConfig(json: json())
        var seed: UInt64 = 1
        var tensors = [String: MLXArray]()
        func put(_ key: String, _ shape: [Int], scale: Float = 0.3, offset: Float = 0) {
            seed += 1
            let values = InworldFixtures.noise(shape.reduce(1, *), seed: seed).map { $0 * 2 * scale + offset }
            tensors[key] = MLXArray(values, shape)
        }
        func linear(_ key: String, _ input: Int, _ output: Int, bias: Bool = true) {
            put("\(key).weight", [output, input], scale: 1 / Float(input).squareRoot())
            if bias { put("\(key).bias", [output], scale: 0.1) }
        }
        func norm(_ key: String, _ size: Int) {
            put("\(key).weight", [size], scale: 0.1, offset: 1)
            put("\(key).bias", [size], scale: 0.1)
        }
        let w = parsed.width, inner = parsed.mambaInner, n = parsed.mambaState, k = parsed.mambaConv
        let edges = parsed.bandEdges
        for (band, range) in edges.enumerated() {
            norm("encoders.\(band).0", 2 * range.count)
            linear("encoders.\(band).1", 2 * range.count, w)
            norm("denoise_decoder.heads.\(band).0", w)
            linear("denoise_decoder.heads.\(band).1", w, w)
            linear("denoise_decoder.heads.\(band).3", w, 4 * range.count)
        }
        put("band_embedding", [edges.count, w], scale: 0.05)
        linear("denoise_decoder.condition", 2, 2 * w)
        let attention = "blocks.0"
        norm("\(attention).time_transformer.norm1", w)
        linear("\(attention).time_transformer.attn.qkv", w, 3 * w)
        linear("\(attention).time_transformer.attn.proj", w, w)
        norm("\(attention).time_transformer.norm2", w)
        linear("\(attention).time_transformer.ffn.0", w, 4 * w)
        linear("\(attention).time_transformer.ffn.3", 4 * w, w)
        norm("\(attention).band_transformer.norm1", w)
        put("\(attention).band_transformer.self_attn.in_proj_weight", [3 * w, w], scale: 0.25)
        put("\(attention).band_transformer.self_attn.in_proj_bias", [3 * w], scale: 0.1)
        linear("\(attention).band_transformer.self_attn.out_proj", w, w)
        norm("\(attention).band_transformer.norm2", w)
        linear("\(attention).band_transformer.linear1", w, 4 * w)
        linear("\(attention).band_transformer.linear2", 4 * w, w)
        for axis in ["time_mamba", "freq_mamba"] {
            let key = "blocks.1.\(axis)"
            for direction in ["forward_blocks", "backward_blocks"] {
                let layer = "\(key).\(direction)"
                linear("\(layer).in_proj", w, 2 * inner, bias: false)
                put("\(layer).conv1d.weight", [inner, 1, k], scale: 0.5)
                put("\(layer).conv1d.bias", [inner], scale: 0.1)
                linear("\(layer).x_proj", inner, parsed.dtRank + 2 * n, bias: false)
                linear("\(layer).dt_proj", parsed.dtRank, inner)
                tensors["\(layer).A_log"] = MLXArray(
                    (0 ..< inner * n).map { logf(Float($0 % n + 1)) }, [inner, n])
                put("\(layer).D", [inner], scale: 0.1, offset: 1)
                linear("\(layer).out_proj", inner, w, bias: false)
            }
            linear("\(key).output_proj", 2 * w, w)
            norm("\(key).norm", w)
        }
        return tensors
    }

    static func write(_ tensors: [String: MLXArray], config: String = json(), to url: URL) throws {
        try MLX.save(arrays: tensors, metadata: ["config": config], url: url)
    }
}
