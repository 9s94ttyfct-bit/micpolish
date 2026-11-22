// SPDX-License-Identifier: Apache-2.0
//
// MLX port of LavaSR v2 (`LavaBWE.infer`, https://github.com/ysharma3501/LavaSR,
// Apache-2.0): a Vocos mel-to-waveform model (ConvNeXt backbone, ISTFT head)
// that bandwidth-extends 48 kHz speech, merged with the input below a cutoff.

import Foundation
import MLX
import MLXNN

public enum LavaEnhancerError: Error, LocalizedError, Equatable, Sendable {
    case weightsNotFound(URL)
    case missingTensor(String)
    case unexpectedShape(key: String, actual: [Int], expected: [Int])
    case invalidCutoff(Double)

    public var errorDescription: String? {
        switch self {
        case .weightsNotFound(let url):
            return "LavaSR weights not found at \(url.path)"
        case .missingTensor(let key):
            return "LavaSR checkpoint is missing tensor \(key)"
        case .unexpectedShape(let key, let actual, let expected):
            return "LavaSR tensor \(key) has shape \(actual), expected \(expected)"
        case .invalidCutoff(let hz):
            return "LavaSR crossover cutoff \(hz) Hz must be in (0, 24000]"
        }
    }
}

public final class LavaEnhancer: @unchecked Sendable {
    public static let sampleRate = 48_000

    static let melCount = 80
    static let dim = 512
    static let intermediateDim = 1536
    static let layerCount = 8
    static let kernelSize = 7
    static let layerNormEps: Float = 1e-6
    /// LavaSR's patched ISTFT head clips magnitudes at 1e3 (upstream Vocos uses 1e2).
    static let maxMagnitude: Float = 1e3

    private struct ConvNeXtBlock {
        let dwWeight: MLXArray  // [512, 7, 1]
        let dwBias: MLXArray
        let normWeight: MLXArray
        let normBias: MLXArray
        let pw1Weight: MLXArray  // [512, 1536] (transposed)
        let pw1Bias: MLXArray
        let pw2Weight: MLXArray  // [1536, 512] (transposed)
        let pw2Bias: MLXArray
        let gamma: MLXArray
    }

    private let melWindow: MLXArray
    private let filterbank: MLXArray  // [1025, 80]
    private let embedWeight: MLXArray  // [512, 7, 80]
    private let embedBias: MLXArray
    private let normWeight: MLXArray
    private let normBias: MLXArray
    private let blocks: [ConvNeXtBlock]
    private let finalNormWeight: MLXArray
    private let finalNormBias: MLXArray
    private let headWeight: MLXArray  // [512, 2050] (transposed)
    private let headBias: MLXArray
    private let istftWindow: MLXArray

    /// Loads the safetensors checkpoint (PyTorch `enhancer_v2` key layout) from disk.
    public init(weightsURL: URL) throws {
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw LavaEnhancerError.weightsNotFound(weightsURL)
        }
        let loaded = try Device.withDefaultDevice(.gpu) { () throws -> Loaded in
            let arrays = try MLX.loadArrays(url: weightsURL)
            let loaded = try Loaded(arrays)
            MLX.eval(loaded.all)
            return loaded
        }
        melWindow = loaded.melWindow
        filterbank = loaded.filterbank
        embedWeight = loaded.embedWeight
        embedBias = loaded.embedBias
        normWeight = loaded.normWeight
        normBias = loaded.normBias
        blocks = loaded.blocks
        finalNormWeight = loaded.finalNormWeight
        finalNormBias = loaded.finalNormBias
        headWeight = loaded.headWeight
        headBias = loaded.headBias
        istftWindow = loaded.istftWindow
    }

    /// Bandwidth-extends one 48 kHz mono chunk; returns the same number of samples.
    /// Below `cutoffHz` the input is kept, above it the model output is used, merged with
    /// upstream's smooth crossover (FastLRMerge, transition_bins = 1024, sample_rate 48000).
    public func enhance(_ samples: [Float], cutoffHz: Double) throws -> [Float] {
        guard cutoffHz.isFinite, cutoffHz > 0, cutoffHz <= Double(Self.sampleRate) / 2 else {
            throw LavaEnhancerError.invalidCutoff(cutoffHz)
        }
        guard !samples.isEmpty else { return [] }
        let count = samples.count
        let paddedCount = LavaSpectral.paddedCount(for: count)
        var input = samples
        input.append(contentsOf: repeatElement(0, count: paddedCount - count))

        return Device.withDefaultDevice(.gpu) {
            let audio = MLXArray(input)
            let predicted = synthesize(audio)
            let merged = LavaCrossover.merge(model: predicted, input: audio, cutoffHz: cutoffHz)
            let output = merged[0 ..< count]
            MLX.eval(output)
            return output.asArray(Float.self)
        }
    }

    /// Vocos forward pass with LavaSR's head: `[n]` audio (multiple of 512) to `[n]` audio.
    func synthesize(_ audio: MLXArray) -> MLXArray {
        let features = LavaSpectral.logMel(audio, window: melWindow, filterbank: filterbank)
        let hidden = backbone(features)
        return head(hidden)
    }

    /// `VocosBackbone`: `[T, 80]` features to `[T, 512]`.
    func backbone(_ features: MLXArray) -> MLXArray {
        let pad = Self.kernelSize / 2
        var x = conv1d(features.expandedDimensions(axis: 0), embedWeight, padding: pad) + embedBias
        x = MLXFast.layerNorm(x, weight: normWeight, bias: normBias, eps: Self.layerNormEps)
        for block in blocks {
            var y = conv1d(x, block.dwWeight, padding: pad, groups: Self.dim) + block.dwBias
            y = MLXFast.layerNorm(y, weight: block.normWeight, bias: block.normBias, eps: Self.layerNormEps)
            y = gelu(matmul(y, block.pw1Weight) + block.pw1Bias)
            y = matmul(y, block.pw2Weight) + block.pw2Bias
            x = x + block.gamma * y
        }
        x = MLXFast.layerNorm(x, weight: finalNormWeight, bias: finalNormBias, eps: Self.layerNormEps)
        return x.squeezed(axis: 0)
    }

    /// LavaSR's patched `ISTFTHead.forward`: `[T, 512]` to `[T * 512]` audio.
    func head(_ hidden: MLXArray) -> MLXArray {
        let projected = matmul(hidden, headWeight) + headBias
        let bins = LavaSpectral.binCount
        let magnitude = minimum(exp(projected[0..., ..<bins]), Self.maxMagnitude)
        let phase = projected[0..., bins...]
        let spectrum = magnitude * cos(phase) + (magnitude * sin(phase)).asImaginary()
        return LavaSpectral.istft(spectrum, window: istftWindow)
    }

    /// Checkpoint tensors converted to MLX layouts.
    private struct Loaded {
        let melWindow, filterbank, embedWeight, embedBias, normWeight, normBias: MLXArray
        let blocks: [ConvNeXtBlock]
        let finalNormWeight, finalNormBias, headWeight, headBias, istftWindow: MLXArray

        var all: [MLXArray] {
            [melWindow, filterbank, embedWeight, embedBias, normWeight, normBias,
             finalNormWeight, finalNormBias, headWeight, headBias, istftWindow]
                + blocks.flatMap {
                    [$0.dwWeight, $0.dwBias, $0.normWeight, $0.normBias, $0.pw1Weight,
                     $0.pw1Bias, $0.pw2Weight, $0.pw2Bias, $0.gamma]
                }
        }

        init(_ arrays: [String: MLXArray]) throws {
            func tensor(_ key: String, _ shape: [Int]) throws -> MLXArray {
                guard let array = arrays[key] else { throw LavaEnhancerError.missingTensor(key) }
                guard array.shape == shape else {
                    throw LavaEnhancerError.unexpectedShape(key: key, actual: array.shape, expected: shape)
                }
                return array.asType(.float32)
            }
            let dim = LavaEnhancer.dim, inner = LavaEnhancer.intermediateDim
            let k = LavaEnhancer.kernelSize, n = LavaSpectral.nFFT, bins = LavaSpectral.binCount

            melWindow = try tensor("feature_extractor.mel_spec.spectrogram.window", [n])
            filterbank = try tensor("feature_extractor.mel_spec.mel_scale.fb", [bins, LavaEnhancer.melCount])
            // PyTorch Conv1d weights are [out, in, kernel]; MLX conv1d takes [out, kernel, in].
            embedWeight = try tensor("backbone.embed.weight", [dim, LavaEnhancer.melCount, k])
                .transposed(0, 2, 1).contiguous()
            embedBias = try tensor("backbone.embed.bias", [dim])
            normWeight = try tensor("backbone.norm.weight", [dim])
            normBias = try tensor("backbone.norm.bias", [dim])
            blocks = try (0..<LavaEnhancer.layerCount).map { index in
                let prefix = "backbone.convnext.\(index)"
                return ConvNeXtBlock(
                    dwWeight: try tensor("\(prefix).dwconv.weight", [dim, 1, k]).transposed(0, 2, 1).contiguous(),
                    dwBias: try tensor("\(prefix).dwconv.bias", [dim]),
                    normWeight: try tensor("\(prefix).norm.weight", [dim]),
                    normBias: try tensor("\(prefix).norm.bias", [dim]),
                    // PyTorch Linear weights are [out, in]; store [in, out] for x @ W.
                    pw1Weight: try tensor("\(prefix).pwconv1.weight", [inner, dim]).transposed(),
                    pw1Bias: try tensor("\(prefix).pwconv1.bias", [inner]),
                    pw2Weight: try tensor("\(prefix).pwconv2.weight", [dim, inner]).transposed(),
                    pw2Bias: try tensor("\(prefix).pwconv2.bias", [dim]),
                    gamma: try tensor("\(prefix).gamma", [dim])
                )
            }
            finalNormWeight = try tensor("backbone.final_layer_norm.weight", [dim])
            finalNormBias = try tensor("backbone.final_layer_norm.bias", [dim])
            headWeight = try tensor("head.out.weight", [n + 2, dim]).transposed()
            headBias = try tensor("head.out.bias", [n + 2])
            istftWindow = try tensor("head.istft.window", [n])
        }
    }
}
