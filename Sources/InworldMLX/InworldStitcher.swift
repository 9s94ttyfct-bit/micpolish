// SPDX-License-Identifier: Apache-2.0
//
// Upstream `VoiceIsolatorNetwork.enhance` (offline window placement and
// linear crossfade) computed incrementally, so a stream of unknown length
// is isolated with bounded memory and the same result as the whole clip:
//
// - A clip no longer than one window is a single pass over the clip.
// - Otherwise windows start every `chunk - overlap` samples while
//   `start < n - chunk`, and one last full window ends at the clip's end. It
//   is silent (weight 0) until the previous window's fade-out, then fades in
//   over `overlap` samples; every other window fades in (except the first)
//   and out over `overlap` samples with `torch.linspace` ramps.
// - The output is the weighted sum divided by the summed weights.
//
// Window `start` is a regular window exactly when more than `start + chunk`
// samples exist, so it runs once one sample beyond its end has arrived.
// After it, every sample before `start + chunk - overlap` is final.

import Foundation

struct InworldStitcher {
    typealias Render = ([Float]) throws -> [Float]

    let chunk: Int
    let overlap: Int
    private let render: Render
    private let fadeIn: [Float]
    private let fadeOut: [Float]

    /// Input samples from absolute position `inputStart`.
    private var input: [Float] = []
    private var inputStart = 0
    private var total = 0
    private var windowsRun = 0
    /// Start of the next regular window; the previous one started `stride` earlier.
    private var nextStart = 0
    /// Samples returned so far; `pending*` hold the weighted sums from there on.
    private var emitted = 0
    private var pendingSum: [Float] = []
    private var pendingWeight: [Float] = []
    private(set) var isFinished = false

    var stride: Int { chunk - overlap }

    init(chunk: Int, overlap: Int, render: @escaping Render) {
        precondition(chunk >= 1 && overlap >= 0 && overlap <= chunk / 2, "invalid window \(chunk)/\(overlap)")
        self.chunk = chunk
        self.overlap = overlap
        self.render = render
        fadeIn = Self.linspace(0, 1, overlap)
        fadeOut = Self.linspace(1, 0, overlap)
    }

    /// Samples held between calls: at most `chunk + stride` of input (the last
    /// window may start anywhere after the previous one) plus `overlap` pending
    /// sums, i.e. two windows, independent of the stream's length.
    var bufferedSampleCount: Int { input.count + pendingSum.count }

    mutating func push(_ samples: [Float]) throws -> [Float] {
        precondition(!isFinished, "push after finish")
        input += samples
        total += samples.count
        var output = [Float]()
        while total > nextStart + chunk {
            output += try runWindow(start: nextStart, isLast: false)
            nextStart += stride
            // The last window starts after the window just run, so input before it is no longer needed.
            let keepFrom = nextStart - stride
            input.removeFirst(keepFrom - inputStart)
            inputStart = keepFrom
        }
        return output
    }

    mutating func finish() throws -> [Float] {
        precondition(!isFinished, "finish called twice")
        isFinished = true
        defer { input = []; pendingSum = []; pendingWeight = [] }
        if windowsRun == 0 {
            // A clip no longer than one window: one pass over the clip, no weighting.
            guard total > 0 else { return [] }
            let output = try render(input)
            precondition(output.count == total, "render changed the length")
            return output
        }
        return try runWindow(start: total - chunk, isLast: true)
    }

    /// Runs the window at `start`, adds it to the pending sums and returns the samples that became final.
    private mutating func runWindow(start: Int, isLast: Bool) throws -> [Float] {
        let offset = start - inputStart
        let rendered = try render(Array(input[offset ..< offset + chunk]))
        precondition(rendered.count == chunk, "render changed the length")

        var weight = [Float](repeating: 1, count: chunk)
        if windowsRun > 0 {
            // Silent until the previous window's fade-out starts (0 for regular windows).
            let fadeStart = (nextStart - stride) + chunk - overlap - start
            for index in 0..<fadeStart { weight[index] = 0 }
            for index in 0..<overlap { weight[fadeStart + index] = fadeIn[index] }
        }
        if !isLast {
            for index in 0..<overlap { weight[chunk - overlap + index] = fadeOut[index] }
        }
        windowsRun += 1

        // Positions before `emitted` carry weight 0 here (only the last window reaches back).
        let first = emitted - start
        for index in max(0, first) ..< chunk {
            let slot = start + index - emitted
            let value = rendered[index] * weight[index]
            if slot < pendingSum.count {
                pendingSum[slot] += value
                pendingWeight[slot] += weight[index]
            } else {
                pendingSum.append(value)
                pendingWeight.append(weight[index])
            }
        }

        let finalEnd = isLast ? total : start + stride
        let count = finalEnd - emitted
        let output = (0..<count).map { pendingSum[$0] / max(pendingWeight[$0], 1e-6) }
        pendingSum.removeFirst(count)
        pendingWeight.removeFirst(count)
        emitted = finalEnd
        return output
    }

    /// `torch.linspace(start, end, steps)` in float32, evaluated from both ends as torch's CPU kernel does.
    static func linspace(_ start: Float, _ end: Float, _ steps: Int) -> [Float] {
        guard steps > 1 else { return steps == 1 ? [start] : [] }
        let step = (end - start) / Float(steps - 1)
        let halfway = steps / 2
        return (0..<steps).map { index in
            index < halfway ? start + step * Float(index) : end - step * Float(steps - index - 1)
        }
    }
}
