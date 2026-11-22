// SPDX-License-Identifier: Apache-2.0
//
// MLX port of Inworld Voice Isolator v1 offline (`voice-isolator-v1`,
// https://github.com/inworld-ai/voice-isolator, Apache-2.0): 48 kHz mono
// voice isolation, rendered in the checkpoint's own overlapping windows.

import Foundation
import MLX

public enum InworldIsolatorError: Error, LocalizedError, Equatable, Sendable {
    case weightsNotFound(URL)
    case missingTensor(String)
    case unexpectedShape(String, expected: [Int], actual: [Int])
    /// The checkpoint's embedded config (or a spare tensor) describes a model
    /// this port does not implement, such as a streaming or dual-mode one.
    case unsupportedConfig(String)

    public var errorDescription: String? {
        switch self {
        case .weightsNotFound(let url):
            return "Inworld Voice Isolator weights not found at \(url.path)"
        case .missingTensor(let key):
            return "Inworld Voice Isolator checkpoint is missing \(key)"
        case .unexpectedShape(let key, let expected, let actual):
            return "Inworld Voice Isolator tensor \(key) has shape \(actual), expected \(expected)"
        case .unsupportedConfig(let detail):
            return "Inworld Voice Isolator checkpoint is not an offline model this build supports: \(detail)"
        }
    }
}

public final class InworldIsolator: @unchecked Sendable {
    public static let sampleRate = 48_000

    let network: InworldNetwork

    /// The checkpoint's render window and crossfade, in samples.
    var window: (chunk: Int, overlap: Int) { network.config.window }

    /// Loads an upstream `denoiser.safetensors` (architecture from its `config` metadata).
    public init(weightsURL: URL) throws {
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw InworldIsolatorError.weightsNotFound(weightsURL)
        }
        network = try Device.withDefaultDevice(.gpu) { () throws -> InworldNetwork in
            let tensors: [String: MLXArray], metadata: [String: String]
            do {
                (tensors, metadata) = try loadArraysAndMetadata(url: weightsURL)
            } catch {
                throw InworldIsolatorError.missingTensor("__metadata__.config")
            }
            guard let json = metadata["config"] else {
                throw InworldIsolatorError.missingTensor("__metadata__.config")
            }
            let network = try InworldNetwork(config: try InworldConfig(json: json), tensors: tensors)
            eval(network.arrays)
            return network
        }
    }

    /// Starts one offline isolation pass over a single stream.
    public func makeSession() -> InworldSession {
        InworldSession(isolator: self)
    }

    /// One forward pass over a whole chunk on the GPU.
    func render(_ samples: [Float]) -> [Float] {
        Device.withDefaultDevice(.gpu) {
            let output = network(MLXArray(samples))
            eval(output)
            return output.asArray(Float.self)
        }
    }
}

/// Feeds 48 kHz mono samples in pieces of any size; returns the isolated
/// samples that are final so far. Output total == input total. Memory bounded
/// by one chunk plus its overlap, independent of stream length.
public final class InworldSession {
    private var stitcher: InworldStitcher

    init(isolator: InworldIsolator) {
        let window = isolator.window
        stitcher = InworldStitcher(chunk: window.chunk, overlap: window.overlap) { isolator.render($0) }
    }

    public func push(_ samples: [Float]) throws -> [Float] {
        try stitcher.push(samples)
    }

    public func finish() throws -> [Float] {
        try stitcher.finish()
    }
}
