// SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX

/// The MLX stages of one `.aneHybrid` window. A session pipelines them across
/// windows so the Core ML encoder/decoders (`accelerator.runEncoder` /
/// `runDecoders`, off the MLX thread) overlap the Mamba stack on the GPU.
protocol HybridWindowPipelining: MonoWindowEnhancing {
    var accelerator: ReuseANEAccelerator? { get }
    /// Whether this window may use Core ML (full 48 kHz window, no failure yet).
    func hybridEligible(sampleRate: Int, count: Int) async -> Bool
    /// STFT → Core ML encoder input.
    func hybridEncoderInput(_ samples: [Float], sampleRate: Int) async throws -> CoreMLTensor
    /// Encoder latent → 30 Mamba blocks → Core ML decoder input.
    func hybridMamba(_ latent: CoreMLTensor) async throws -> CoreMLTensor
    /// Decoder output → iSTFT → `sampleCount` samples.
    func hybridOutput(_ decoded: CoreMLTensor, sampleCount: Int, sampleRate: Int) async throws -> [Float]
    /// Records a Core ML failure; later windows run in MLX.
    func hybridFailed(_ error: any Error) async
}

extension MambaEnhancer: HybridWindowPipelining {
    func hybridEligible(sampleRate: Int, count: Int) -> Bool {
        hybridAccelerator(sampleRate: sampleRate, count: count) != nil
    }

    func hybridEncoderInput(_ samples: [Float], sampleRate: Int) throws -> CoreMLTensor {
        try Task.checkCancellation()
        let params = Self.scaledSTFTParams(sampleRate: sampleRate)
        return try MLXExecutionContext.withGPU {
            let (mag, pha, frameCount) = try stftInputs(
                samples: samples, nFFT: params.nFFT, hop: params.hop, win: params.win
            )
            guard frameCount == ReuseANEAccelerator.frames, mag.shape[1] == ReuseANEAccelerator.frequencyBins else {
                throw ReuseANEError.predictionFailed("window gives \(mag.shape), expected the exported shape")
            }
            return try SEMamba.hybridEncoderInput(mag, pha)
        }
    }

    func hybridMamba(_ latent: CoreMLTensor) throws -> CoreMLTensor {
        try Task.checkCancellation()
        return try MLXExecutionContext.withGPU { try model.hybridMamba(latent) }
    }

    func hybridOutput(_ decoded: CoreMLTensor, sampleCount: Int, sampleRate: Int) throws -> [Float] {
        let params = Self.scaledSTFTParams(sampleRate: sampleRate)
        let output = try MLXExecutionContext.withGPU {
            let (mag, pha) = SEMamba.hybridOutputs(decoded)
            return try reconstruct(
                modelMagnitude: mag, modelPhase: pha,
                frameCount: ReuseANEAccelerator.frames, sampleCount: sampleCount, params: params
            )
        }
        guard output.count == sampleCount else {
            throw InferenceError.inferenceFailed("Model window changed frame count")
        }
        return output
    }

    func hybridFailed(_ error: any Error) {
        guard acceleratorFailure == nil else { return }
        acceleratorFailure = error.localizedDescription
        ReuseLog.inference.error(
            "RE-USE Core ML pipeline failed; using MLX from now on: \(error.localizedDescription, privacy: .public)"
        )
    }
}
