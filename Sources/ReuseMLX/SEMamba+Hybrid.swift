// SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX

extension SEMamba {
    /// The `callAsFunction` forward pass with the dense encoder and both
    /// decoders in Core ML. `mag`/`pha` are `[1, F, T]` for one full 48 kHz
    /// window; returns `[1, F, T]` magnitude and phase like the MLX path.
    func hybridForward(
        _ mag: MLXArray, _ pha: MLXArray, accelerator: ReuseANEAccelerator
    ) throws -> (MLXArray, MLXArray) {
        let latent = try accelerator.runEncoder(try Self.hybridEncoderInput(mag, pha))
        let decoded = try accelerator.runDecoders(try hybridMamba(latent))
        return Self.hybridOutputs(decoded)
    }

    /// `[1, F, T]` magnitude and phase → the encoder input `[1, 2, T, F]`.
    static func hybridEncoderInput(_ mag: MLXArray, _ pha: MLXArray) throws -> CoreMLTensor {
        try ReuseANEAccelerator.coreMLInput(MLX.stacked([mag, pha], axis: 1).transposed(0, 1, 3, 2))
    }

    /// The 30 Mamba blocks on a Core ML latent. Core ML works in
    /// `[B, C, T, F]`; the Mamba blocks in `[B, T, F, C]`.
    func hybridMamba(_ latent: CoreMLTensor) throws -> CoreMLTensor {
        var h = ReuseANEAccelerator.mlxArray(fromFloat16: latent.value, shape: ReuseANEAccelerator.latentShape)
            .transposed(0, 2, 3, 1).asType(.float32)
        for block in tfMamba {
            h = block(h)
        }
        return try ReuseANEAccelerator.coreMLInput(h.transposed(0, 3, 1, 2))
    }

    /// Decoder output `[1, 3, T, F]` → magnitude and atan2(imag, real), each `[1, F, T]`.
    static func hybridOutputs(_ decoded: CoreMLTensor) -> (MLXArray, MLXArray) {
        let y = ReuseANEAccelerator.mlxArray(fromFloat16: decoded.value, shape: ReuseANEAccelerator.decodedShape)
            .asType(.float32)
        let mag = y[0..., 0, 0..., 0...].transposed(0, 2, 1)
        let pha = MLX.atan2(y[0..., 2, 0..., 0...], y[0..., 1, 0..., 0...]).transposed(0, 2, 1)
        return (mag, pha)
    }
}
