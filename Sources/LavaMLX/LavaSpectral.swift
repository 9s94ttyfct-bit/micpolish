// SPDX-License-Identifier: Apache-2.0
//
// Spectral front and back ends of the LavaSR v2 bandwidth extender (Vocos
// `MelSpectrogramFeatures` / `ISTFT` with padding "same" and LavaSR's
// `FastLRMerge`), ported from the PyTorch code at the commits pinned in
// tools/parity/lava_reference.py.

import Foundation
import MLX

enum LavaSpectral {
    static let nFFT = 2048
    static let hop = 512
    static let binCount = nFFT / 2 + 1
    /// "same" padding: (win_length - hop_length) / 2 on each side.
    static let samePad = (nFFT - hop) / 2
    /// Frames overlapping any output sample (`nFFT / hop`).
    static let overlap = nFFT / hop

    /// Number of samples the model runs on for an input of `count` samples.
    ///
    /// The ISTFT head emits one hop per frame, so upstream `LavaBWE.infer`
    /// returns `floor(count / 512) * 512` samples. The input is zero-padded to
    /// the next multiple of the hop and the output trimmed back. The feature
    /// extractor's reflect padding of 768 samples needs more than 768 input
    /// samples, so at least two hops are used.
    static func paddedCount(for count: Int) -> Int {
        max((count + hop - 1) / hop * hop, 2 * hop)
    }

    /// Complex STFT of `audio` (`[n]`, `n` a multiple of the hop) as Vocos'
    /// mel extractor frames it: reflect-pad 768 samples each side, then
    /// `torch.stft(center=False)` with the given window. Returns `[n / 512, 1025]`.
    static func stft(_ audio: MLXArray, window: MLXArray) -> MLXArray {
        let count = audio.dim(0)
        precondition(count % hop == 0 && count > samePad, "STFT input must be a multiple of \(hop) longer than \(samePad)")
        // torch reflect padding: x[pad], ..., x[1] | x | x[n-2], ..., x[n-1-pad].
        let head = audio[.stride(from: samePad, to: 0, by: -1)]
        let tail = audio[.stride(from: count - 2, to: count - 2 - samePad, by: -1)]
        let padded = concatenated([head, audio, tail])
        let frameCount = count / hop
        // The padded signal is exactly frameCount + 3 hops long; frame t is hops t ..< t + 4.
        let hops = padded.reshaped(frameCount + overlap - 1, hop)
        let frames = concatenated(
            (0..<overlap).map { hops[$0 ..< $0 + frameCount] },
            axis: 1
        )
        return rfft(frames * window, n: nFFT, axis: -1)
    }

    /// Vocos `ISTFT(padding="same")`: inverse real FFT of each `[T, 1025]`
    /// frame, synthesis window, overlap-add, trim 768 samples each side and
    /// divide by the overlap-added squared window. Returns `[T * 512]`.
    static func istft(_ spectrum: MLXArray, window: MLXArray) -> MLXArray {
        let frameCount = spectrum.dim(0)
        // pocketfft's C2R transform ignores the imaginary parts of the DC and
        // Nyquist bins; make that explicit so every FFT backend agrees.
        let binIndex = MLXArray(0 ..< Int32(binCount))
        let isRealBin = (binIndex .== 0) .|| (binIndex .== Int32(binCount - 1))
        let cleaned = which(isRealBin, spectrum.realPart().asType(spectrum.dtype), spectrum)
        let frames = irfft(cleaned, n: nFFT, axis: -1) * window
        let signal = overlapAdd(frames)
        let envelope = overlapAdd(broadcast(window * window, to: [frameCount, nFFT]))
        return signal / envelope
    }

    /// `torch.nn.functional.fold` of `[T, 2048]` frames at hop 512, trimmed
    /// by `samePad` on both sides: `[T * 512]`.
    private static func overlapAdd(_ frames: MLXArray) -> MLXArray {
        let frameCount = frames.dim(0)
        var sum: MLXArray?
        for k in 0..<overlap {
            let segment = frames[0..., (k * hop) ..< ((k + 1) * hop)]
            let shifted = padded(segment, widths: [IntOrPair((k, overlap - 1 - k)), IntOrPair(0)])
            sum = sum.map { $0 + shifted } ?? shifted
        }
        let flat = sum!.reshaped(-1)
        return flat[samePad ..< (samePad + frameCount * hop)]
    }

    /// Mel features: `log(clip(|STFT| @ fb, min: 1e-5))`, `[T, 80]`
    /// (torchaudio `MelSpectrogram(power=1)` followed by Vocos `safe_log`).
    static func logMel(_ audio: MLXArray, window: MLXArray, filterbank: MLXArray) -> MLXArray {
        let magnitude = abs(stft(audio, window: window))
        return log(maximum(matmul(magnitude, filterbank), Float(1e-5)))
    }
}

/// LavaSR `FastLRMerge(sample_rate=48000, transition_bins=1024)`: below the
/// cutoff the merged spectrum is the input's, above it the model's, with a
/// smoothstep crossover 1024 FFT bins wide over the whole signal's spectrum.
enum LavaCrossover {
    static let sampleRate = 48_000
    static let transitionBins = 1024

    /// `fade_template`: smoothstep `3t² - 2t³` of `t = (linspace(-1, 1, 1024) + 1) / 2`
    /// in float32, with torch's two-sided `linspace` evaluation.
    static let fadeTemplate: [Float] = {
        let steps = transitionBins
        let start: Float = -1, end: Float = 1
        let step = (end - start) / Float(steps - 1)
        let halfway = steps / 2
        return (0..<steps).map { index in
            let x = index < halfway ? start + step * Float(index) : end - step * Float(steps - index - 1)
            let t = (x + 1) / 2
            return 3 * (t * t) - 2 * (t * t * t)
        }
    }()

    /// The real crossover weight applied to the model spectrum for an
    /// `rfft` of `binCount` bins, as `FastLRMerge._get_mask` builds it.
    static func mask(binCount: Int, cutoffHz: Double) -> [Float] {
        let cutoffBin = Int((cutoffHz / (Double(sampleRate) / 2)) * Double(binCount))
        let half = transitionBins / 2
        let start = max(0, cutoffBin - half)
        let end = min(binCount, cutoffBin + half)
        var mask = [Float](repeating: 1, count: binCount)
        for bin in 0..<min(start, binCount) { mask[bin] = 0 }
        if end > start {
            for bin in start..<end { mask[bin] = fadeTemplate[bin - start] }
        }
        return mask
    }

    /// `irfft(rfft(input) + (rfft(model) - rfft(input)) * mask, n)`, evaluated by
    /// linearity as `input + irfft(rfft(model - input) * mask, n)`.
    ///
    /// The transforms span the whole chunk, so their length is rarely a smooth
    /// number (480256 = 2^10 · 7 · 67 for 10 s). For such lengths MLX's Metal
    /// FFT measured ≈ 0.26 s per transform against ≈ 0.06 s for its CPU FFT,
    /// so these two run on the CPU stream.
    static func merge(model: MLXArray, input: MLXArray, cutoffHz: Double) -> MLXArray {
        let count = input.dim(0)
        let weights = MLXArray(mask(binCount: count / 2 + 1, cutoffHz: cutoffHz))
        let difference = rfft(model - input, axis: -1, stream: .cpu) * weights
        return input + irfft(difference, n: count, axis: -1, stream: .cpu)
    }
}
