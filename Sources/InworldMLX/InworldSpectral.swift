// SPDX-License-Identifier: Apache-2.0
//
// The isolator's STFT and iSTFT: `torch.stft` / `torch.istft` with
// `center=True`, a periodic Hann window, `n_fft` 2048 and hop 480, as
// upstream `VoiceIsolatorNetwork.stft` / `.istft` call them.

import Foundation
import MLX

struct InworldSpectral: Sendable {
    let nFFT: Int
    let hop: Int

    var binCount: Int { nFFT / 2 + 1 }
    var centerPad: Int { nFFT / 2 }
    /// Hops each frame spans, rounded up: frames are cut from hop-sized rows.
    var hopsPerFrame: Int { (nFFT + hop - 1) / hop }

    /// `torch.stft(center=True)` frame count for `count` samples.
    func frameCount(_ count: Int) -> Int { 1 + count / hop }

    /// `torch.hann_window(nFFT)` (periodic) as torch computes it in float32:
    /// `cos(i * (2π / nFFT)) * -0.5 + 0.5`.
    var window: [Float] {
        let scale = Float(2 * Double.pi / Double(nFFT))
        return (0..<nFFT).map { cosf(Float($0) * scale) * -0.5 + 0.5 }
    }

    /// Complex spectra of `audio` (`[n]`) as real and imaginary parts, each
    /// `[frames, bins]`. Reflect padding when the clip is longer than half a
    /// window, zeros otherwise (upstream's `pad_mode` choice).
    func stft(_ audio: MLXArray, window: MLXArray) -> (re: MLXArray, im: MLXArray) {
        let count = audio.dim(0)
        let pad = centerPad
        let head: MLXArray, tail: MLXArray
        if count > pad {
            // torch reflect padding: x[pad], ..., x[1] | x | x[n-2], ..., x[n-1-pad].
            head = audio[.stride(from: pad, to: 0, by: -1)]
            tail = audio[.stride(from: count - 2, to: count - 2 - pad, by: -1)]
        } else {
            head = MLXArray.zeros([pad])
            tail = head
        }
        let frames = frameCount(count)
        let rows = frames + hopsPerFrame - 1
        // The padded clip (count + nFFT samples) covers every frame; trim or
        // zero-extend it to whole hop rows, frame t being rows t ..< t + hopsPerFrame.
        var padded = concatenated([head, audio, tail])
        let rowSamples = rows * hop
        if padded.dim(0) >= rowSamples {
            padded = padded[0 ..< rowSamples]
        } else {
            padded = concatenated([padded, MLXArray.zeros([rowSamples - padded.dim(0)])])
        }
        let hops = padded.reshaped(rows, hop)
        let framed = concatenated((0..<hopsPerFrame).map { hops[$0 ..< $0 + frames] }, axis: 1)[0..., 0 ..< nFFT]
        let spectrum = rfft(framed * window, n: nFFT, axis: -1)
        // pocketfft returns exactly zero imaginary parts at DC and Nyquist.
        return (spectrum.realPart(), spectrum.imaginaryPart() * realBinMask)
    }

    /// `torch.istft(center=True, length: count)` of `[frames, bins]` spectra:
    /// inverse real FFT of each frame, synthesis window, overlap-add, trim half
    /// a window at the start and divide by the overlap-added squared window.
    func istft(re: MLXArray, im: MLXArray, window: MLXArray, count: Int) -> MLXArray {
        let frames = re.dim(0)
        // pocketfft's C2R transform ignores the imaginary parts of the DC and
        // Nyquist bins; make that explicit so every FFT backend agrees.
        let spectrum = re + (im * realBinMask).asImaginary()
        let signal = overlapAdd(irfft(spectrum, n: nFFT, axis: -1) * window)
        let envelope = overlapAdd(broadcast(window * window, to: [frames, nFFT]))
        let range = centerPad ..< centerPad + count
        return signal[range] / envelope[range]
    }

    /// 0 at DC and Nyquist, 1 elsewhere: `[bins]`.
    private var realBinMask: MLXArray {
        var mask = [Float](repeating: 1, count: binCount)
        mask[0] = 0
        mask[binCount - 1] = 0
        return MLXArray(mask)
    }

    /// `torch.nn.functional.fold` of `[frames, nFFT]` at the hop: `[(frames - 1) * hop + nFFT]`
    /// (zero-extended to whole hops).
    private func overlapAdd(_ frames: MLXArray) -> MLXArray {
        let count = frames.dim(0)
        let span = hopsPerFrame
        let widened = padded(frames, widths: [IntOrPair(0), IntOrPair((0, span * hop - nFFT))])
            .reshaped(count, span, hop)
        var sum: MLXArray?
        for k in 0..<span {
            let shifted = padded(widened[0..., k, 0...], widths: [IntOrPair((k, span - 1 - k)), IntOrPair(0)])
            sum = sum.map { $0 + shifted } ?? shifted
        }
        return sum!.reshaped(-1)
    }
}
