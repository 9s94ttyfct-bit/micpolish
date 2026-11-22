// SPDX-License-Identifier: Apache-2.0
//
// Metal kernels for the isolator's Mamba layers: the causal depthwise
// convolution with its SiLU, and the selective scan with its softplus step
// and SiLU gate (upstream `Mamba.forward` and `scan.selective_scan`). Each
// thread owns one (row, channel) pair and walks the frames, keeping the
// convolution window and the scan state in registers. `REVERSE` walks the
// frames from last to first, which is the backward Mamba of a bidirectional
// layer (`backward_blocks(x.flip(1)).flip(1)`) without materialising flips.

import Foundation
import MLX

enum InworldKernels {
    /// Register arrays are sized for these maxima.
    static let maxState = 16
    static let maxConv = 8

    private static let header = """
    // torch.nn.functional.softplus (beta 1, threshold 20): log1p(exp(x)).
    inline float inworld_softplus(float x) {
        if (x > 20.0f) {
            return x;
        }
        const float e = metal::precise::exp(x);
        const float u = 1.0f + e;
        // log1p via log(u) * e / (u - 1), exact where 1 + e rounds to 1.
        return u == 1.0f ? e : metal::precise::log(u) * (e / (u - 1.0f));
    }

    inline float inworld_silu(float x) {
        return x / (1.0f + metal::precise::exp(-x));
    }
    """

    /// `silu(conv1d(x, weight, bias, padding=K-1)[..., :frames])` over `[rows, frames, channels]`.
    private static let convSource = """
    const uint tid = thread_position_in_grid.x;
    const uint rows = x_shape[0];
    const uint frames = x_shape[1];
    const uint channels = x_shape[2];
    if (tid >= rows * channels) {
        return;
    }
    const uint row = tid / channels;
    const uint channel = tid % channels;
    float w[K];
    float history[K];
    for (uint k = 0; k < K; ++k) {
        w[k] = weight[channel * K + k];
        history[k] = 0.0f;
    }
    const float b = bias[channel];
    const ulong base = ulong(row) * frames * channels + channel;
    for (uint step = 0; step < frames; ++step) {
        const uint t = REVERSE ? frames - 1 - step : step;
        const ulong at = base + ulong(t) * channels;
        for (uint k = 0; k + 1 < K; ++k) {
            history[k] = history[k + 1];
        }
        history[K - 1] = x[at];
        float acc = 0.0f;
        for (uint k = 0; k < K; ++k) {
            acc += w[k] * history[k];
        }
        out[at] = inworld_silu(acc + b);
    }
    """

    /// The selective scan: `h = exp(delta * a) * h + (delta * u) * b`,
    /// `y = (c . h + d * u) * silu(z)`, with `delta = softplus(delta_raw)`.
    private static let scanSource = """
    const uint tid = thread_position_in_grid.x;
    const uint rows = u_shape[0];
    const uint frames = u_shape[1];
    const uint channels = u_shape[2];
    if (tid >= rows * channels) {
        return;
    }
    const uint row = tid / channels;
    const uint channel = tid % channels;
    float h[N];
    float rate[N];
    for (uint n = 0; n < N; ++n) {
        h[n] = 0.0f;
        rate[n] = a[channel * N + n];
    }
    const float skip = d[channel];
    const ulong base = ulong(row) * frames * channels + channel;
    const ulong baseState = ulong(row) * frames * N;
    for (uint step = 0; step < frames; ++step) {
        const uint t = REVERSE ? frames - 1 - step : step;
        const ulong at = base + ulong(t) * channels;
        const ulong bt = baseState + ulong(t) * N;
        const float delta = inworld_softplus(delta_raw[at]);
        const float x = u[at];
        const float drive = delta * x;
        float y = 0.0f;
        for (uint n = 0; n < N; ++n) {
            h[n] = metal::precise::exp(delta * rate[n]) * h[n] + drive * bmat[bt + n];
            y += h[n] * cmat[bt + n];
        }
        y += x * skip;
        out[at] = y * inworld_silu(z[at]);
    }
    """

    private static let convKernel = MLXFast.metalKernel(
        name: "inworld_causal_depthwise_conv_silu",
        inputNames: ["x", "weight", "bias"],
        outputNames: ["out"],
        source: convSource,
        header: header
    )

    private static let scanKernel = MLXFast.metalKernel(
        name: "inworld_selective_scan",
        inputNames: ["u", "delta_raw", "a", "bmat", "cmat", "d", "z"],
        outputNames: ["out"],
        source: scanSource,
        header: header
    )

    private static func launch(rows: Int, channels: Int) -> (grid: (Int, Int, Int), group: (Int, Int, Int)) {
        let threads = max(1, rows * channels)
        return ((threads, 1, 1), (min(256, threads), 1, 1))
    }

    /// `x` `[rows, frames, channels]`, `weight` `[channels, K]`, `bias` `[channels]`.
    static func causalConvSiLU(_ x: MLXArray, weight: MLXArray, bias: MLXArray, reverse: Bool) -> MLXArray {
        let (rows, channels) = (x.dim(0), x.dim(2))
        let (grid, group) = launch(rows: rows, channels: channels)
        return convKernel(
            [x, weight, bias],
            template: [("K", weight.dim(1)), ("REVERSE", reverse)],
            grid: grid,
            threadGroup: group,
            outputShapes: [x.shape],
            outputDTypes: [.float32]
        )[0]
    }

    /// `u`, `deltaRaw`, `z` `[rows, frames, channels]`; `a` `[channels, N]` (negative
    /// rates); `b`, `c` `[rows, frames, N]`; `d` `[channels]`.
    static func selectiveScan(
        u: MLXArray, deltaRaw: MLXArray, a: MLXArray, b: MLXArray, c: MLXArray, d: MLXArray, z: MLXArray,
        reverse: Bool
    ) -> MLXArray {
        let (rows, channels) = (u.dim(0), u.dim(2))
        let (grid, group) = launch(rows: rows, channels: channels)
        return scanKernel(
            [u, deltaRaw, a, b, c, d, z],
            template: [("N", a.dim(1)), ("REVERSE", reverse)],
            grid: grid,
            threadGroup: group,
            outputShapes: [u.shape],
            outputDTypes: [.float32]
        )[0]
    }
}
