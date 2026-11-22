// SPDX-License-Identifier: Apache-2.0
// Key mapping and layout conversion adapted from SpeechLens `WeightPort`
// (https://github.com/faraday/SpeechLens, Apache-2.0).

import Foundation
import MLX

/// Converts an upstream RE-USE PyTorch `model.safetensors` into the MLX layout
/// `MambaEnhancer` loads.
public enum ReuseWeightConverter {
    /// Converts `input` and writes the MLX checkpoint to `output`, creating its
    /// parent directory. Returns the number of tensors written.
    @discardableResult
    public static func convert(input: URL, output: URL) throws -> Int {
        let weights = try MLX.loadArrays(url: input)
        var mapped: [String: MLXArray] = [:]
        mapped.reserveCapacity(weights.count)

        for (key, tensor) in weights {
            mapped[mapKey(key)] = convertLayout(key: key, tensor: tensor)
        }

        try FileManager.default.createDirectory(
            at: output.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try MLX.save(arrays: mapped, url: output)
        return mapped.count
    }

    static func convertLayout(key: String, tensor: MLXArray) -> MLXArray {
        if key.contains("conv1d") && key.hasSuffix("weight") && tensor.ndim == 3 {
            // Depthwise Conv1d: PyTorch [C, 1, K] -> MLX [C, K, 1].
            return tensor.transposed(0, 2, 1)
        }
        if key.hasSuffix("weight") && tensor.ndim == 4 {
            // Every 4-D weight in this model is a Conv2d: [O, I, H, W] -> [O, H, W, I].
            // Keys such as dense_block.*.weight carry no "conv" token, so match on rank.
            return tensor.transposed(0, 2, 3, 1)
        }
        if key.contains("dt_proj") && key.hasSuffix("bias") {
            return tensor.asType(.float32)
        }
        return tensor
    }

    /// Maps an upstream PyTorch parameter key to the MLX Swift module path.
    public static func mapKey(_ key: String) -> String {
        var k = key
        if k.hasPrefix("model.") { k.removeFirst(6) }

        // Expand nested DenseBlock Sequentials before the top-level renames so
        // the raw "dense_block.dense_block.i.j." pattern is still present.
        for i in 0..<10 {
            for j in 0..<10 {
                k = k.replacingOccurrences(
                    of: "dense_block.dense_block.\(i).\(j).",
                    with: "denseBlock.layers.\(i).layers.\(j).")
            }
        }

        let renames: [(String, String)] = [
            ("dense_encoder.", "denseEncoder."),
            ("mask_decoder.", "maskDecoder."),
            ("phase_decoder.", "phaseDecoder."),
            ("TSMamba.", "tfMamba."),
            ("time_mamba.", "timeMamba."),
            ("freq_mamba.", "freqMamba."),
            ("forward_blocks.", "forward."),
            ("backward_blocks.", "backward."),
            ("output_proj.", "outputProj."),
            ("in_proj.", "inProj."),
            ("out_proj.", "outProj."),
            ("x_proj.", "xProj."),
            ("dt_proj.", "dtProj."),
            ("final_conv.", "finalConv."),
            ("phase_conv_r.", "phaseConvR."),
            ("phase_conv_i.", "phaseConvI."),
        ]
        for (old, new) in renames {
            k = k.replacingOccurrences(of: old, with: new)
        }

        let sequentials = [
            ("dense_conv_1", "conv1"), ("dense_conv_2", "conv2"),
            ("up_conv1", "upConv1"), ("up_conv2", "upConv2"),
        ]
        for (parent, target) in sequentials {
            for i in 0..<10 {
                k = k.replacingOccurrences(of: "\(parent).\(i).", with: "\(target).layers.\(i).")
            }
        }
        return k
    }
}
