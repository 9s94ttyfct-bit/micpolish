// SPDX-License-Identifier: Apache-2.0
//
// MLX port of upstream `VoiceIsolatorNetwork.forward` in offline mode
// (inworld-ai/voice-isolator, Apache-2.0, `model.py` at the commit pinned in
// tools/parity/inworld_reference.py): the 48 kHz STFT is cut into bands,
// each band of each frame becomes a token, blocks of time attention (rotary
// positions, exclusive self attention) and band attention alternate with
// bidirectional dual-path Mamba blocks, and per-band heads predict a bounded
// complex mask and residual per bin that correct the input's spectrum.
//
// Band-wise encoders and decoder heads differ in width; they run as one
// batched matmul over all bands with zero-padded weights, the encoder's
// LayerNorm statistics taken over each band's own features only.

import Foundation
import MLX
import MLXNN

struct InworldNetwork: @unchecked Sendable {
    static let layerNormEps: Float = 1e-5

    struct Linear {
        let weight: MLXArray  // [in, out]
        let bias: MLXArray?

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let y = matmul(x, weight)
            return bias.map { y + $0 } ?? y
        }
    }

    struct LayerNorm {
        let weight: MLXArray
        let bias: MLXArray

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            MLXFast.layerNorm(x, weight: weight, bias: bias, eps: InworldNetwork.layerNormEps)
        }
    }

    struct TimeBlock {
        let norm1: LayerNorm
        let qkv: Linear
        let proj: Linear
        let norm2: LayerNorm
        let ffn1: Linear
        let ffn2: Linear
    }

    struct BandBlock {
        let norm1: LayerNorm
        let inProj: Linear
        let outProj: Linear
        let norm2: LayerNorm
        let linear1: Linear
        let linear2: Linear
    }

    struct Mamba {
        let inProj: Linear
        let convWeight: MLXArray  // [inner, K]
        let convBias: MLXArray
        let xProj: Linear
        let dtProj: Linear
        let a: MLXArray  // -exp(A_log), [inner, N]
        let d: MLXArray
        let outProj: Linear
    }

    struct BidirectionalMamba {
        let forward: Mamba
        let backward: Mamba
        let outputProj: Linear
        let norm: LayerNorm
    }

    enum Block {
        case attention(time: TimeBlock, band: BandBlock)
        case mamba(time: BidirectionalMamba, band: BidirectionalMamba)
    }

    let config: InworldConfig
    let spectral: InworldSpectral
    let edges: [Range<Int>]
    let maxBandBins: Int

    // Encoders: per-band LayerNorm(2w) + Linear(2w → width), padded to the widest band.
    let encoderGather: MLXArray  // [bands * 2 * maxBandBins] indices into [frames, 2 * bins + 1]
    let encoderMask: MLXArray  // [bands, 1, 2 * maxBandBins]
    let encoderFeatureCount: MLXArray  // [bands, 1, 1]
    let encoderNormWeight: MLXArray  // [bands, 1, 2 * maxBandBins]
    let encoderNormBias: MLXArray
    let encoderWeight: MLXArray  // [bands, 2 * maxBandBins, width]
    let encoderBias: MLXArray  // [bands, 1, width]
    let bandEmbedding: MLXArray  // [bands, 1, width]

    let blocks: [Block]

    // Decoder: FiLM by the fixed controls, then per-band LayerNorm, Linear, GELU, Linear(4w).
    let decoderScale: MLXArray  // 1 + scale, [width]
    let decoderShift: MLXArray
    let headNormWeight: MLXArray  // [bands, 1, width]
    let headNormBias: MLXArray
    let headWeight1: MLXArray  // [bands, width, width]
    let headBias1: MLXArray  // [bands, 1, width]
    let headWeight2: MLXArray  // [bands, width, 4 * maxBandBins]
    let headBias2: MLXArray  // [bands, 1, 4 * maxBandBins]
    let decoderGather: MLXArray  // [bins] indices into [bands * maxBandBins]

    let window: MLXArray

    var arrays: [MLXArray] {
        var all = [encoderGather, encoderMask, encoderFeatureCount, encoderNormWeight, encoderNormBias,
                   encoderWeight, encoderBias, bandEmbedding, decoderScale, decoderShift, headNormWeight,
                   headNormBias, headWeight1, headBias1, headWeight2, headBias2, decoderGather, window]
        func add(_ linear: Linear) { all.append(linear.weight); linear.bias.map { all.append($0) } }
        func add(_ norm: LayerNorm) { all += [norm.weight, norm.bias] }
        func add(_ mamba: Mamba) {
            add(mamba.inProj); add(mamba.xProj); add(mamba.dtProj); add(mamba.outProj)
            all += [mamba.convWeight, mamba.convBias, mamba.a, mamba.d]
        }
        for block in blocks {
            switch block {
            case .attention(let time, let band):
                add(time.norm1); add(time.qkv); add(time.proj); add(time.norm2); add(time.ffn1); add(time.ffn2)
                add(band.norm1); add(band.inProj); add(band.outProj); add(band.norm2); add(band.linear1)
                add(band.linear2)
            case .mamba(let time, let band):
                for layer in [time, band] {
                    add(layer.forward); add(layer.backward); add(layer.outputProj); add(layer.norm)
                }
            }
        }
        return all
    }

    // MARK: - Forward

    /// Isolates one chunk: `[n]` audio to `[n]` audio (upstream `forward`).
    func callAsFunction(_ audio: MLXArray) -> MLXArray {
        let count = audio.dim(0)
        let (re, im) = spectral.stft(audio, window: window)
        var features = encode(re: re, im: im)
        let rope = Self.ropeTables(frames: re.dim(0), headDim: config.headDim)
        // Evaluating after every half block (and after each forward scan) bounds
        // the live graph to one layer's intermediates: peak MLX memory for an 8 s
        // window fell from 3.5 to 2.1 GiB and the window ran ~10% faster (M4 Pro).
        for block in blocks {
            switch block {
            case .attention(let time, let band):
                features = timeAttention(features, time, rope: rope)
                eval(features)
                features = bandAttention(features, band)
            case .mamba(let time, let band):
                features = features + bidirectional(features, time)
                eval(features)
                let byFrame = features.transposed(1, 0, 2)
                features = (byFrame + bidirectional(byFrame, band)).transposed(1, 0, 2)
            }
            eval(features)
        }
        let raw = decode(features)
        let (correctedRe, correctedIm) = correction(raw: raw, re: re, im: im)
        return audio + spectral.istft(re: correctedRe, im: correctedIm, window: window, count: count)
    }

    /// Band tokens `[bands, frames, width]` of `[frames, bins]` spectra (upstream `encode`).
    func encode(re: MLXArray, im: MLXArray) -> MLXArray {
        let frames = re.dim(0)
        let bands = edges.count
        let features = 2 * maxBandBins
        // Interleaved (re, im) per bin as view_as_real(...).flatten does, plus one zero column for padding.
        let interleaved = concatenated(
            [stacked([re, im], axis: -1).reshaped(frames, 2 * config.binCount), MLXArray.zeros([frames, 1])],
            axis: 1)
        let x = interleaved.take(encoderGather, axis: 1).reshaped(frames, bands, features).transposed(1, 0, 2)
        // LayerNorm over each band's 2w features (padding excluded), biased variance as torch.
        let mean = x.sum(axis: -1, keepDims: true) / encoderFeatureCount
        let centered = (x - mean) * encoderMask
        let variance = (centered * centered).sum(axis: -1, keepDims: true) / encoderFeatureCount
        let normalized = centered * rsqrt(variance + Self.layerNormEps) * encoderNormWeight + encoderNormBias
        return matmul(normalized, encoderWeight) + encoderBias + bandEmbedding
    }

    /// Decoder output `[frames, bins, 4]` of `[bands, frames, width]` features (upstream `_Decoder`).
    func decode(_ features: MLXArray) -> MLXArray {
        let (bands, frames) = (features.dim(0), features.dim(1))
        var x = features * decoderScale + decoderShift
        x = MLXFast.layerNorm(x, eps: Self.layerNormEps) * headNormWeight + headNormBias
        x = gelu(matmul(x, headWeight1) + headBias1)
        x = matmul(x, headWeight2) + headBias2  // [bands, frames, 4 * maxBandBins]
        return x.transposed(1, 0, 2).reshaped(frames, bands * maxBandBins, 4).take(decoderGather, axis: 1)
    }

    /// The corrected spectrum `spectrum * mask + residual` (upstream `correction`).
    func correction(raw: MLXArray, re: MLXArray, im: MLXArray) -> (re: MLXArray, im: MLXArray) {
        let maskRe = 2 * tanh(raw[0..., 0..., 0])
        let maskIm = 2 * tanh(raw[0..., 0..., 1])
        // bounded_correction: tanh components, norm clamped at 1, scaled by bound * |bin|.
        let componentRe = tanh(raw[0..., 0..., 2])
        let componentIm = tanh(raw[0..., 0..., 3])
        let norm = maximum(sqrt(componentRe * componentRe + componentIm * componentIm), Float(1))
        let scale = config.correctionBound * sqrt(re * re + im * im)
        let residualRe = componentRe / norm * scale
        let residualIm = componentIm / norm * scale
        return (re * maskRe - im * maskIm + residualRe, re * maskIm + im * maskRe + residualIm)
    }

    // MARK: Attention

    /// Rotary cosine and sine tables `[frames, headDim / 2]` as upstream
    /// `rope_tables` computes them in float32.
    static func ropeTables(frames: Int, headDim: Int) -> (cos: MLXArray, sin: MLXArray) {
        let half = headDim / 2
        let inverse = (0..<half).map { powf(10_000, -(Float($0) / Float(half))) }
        var cosines = [Float](repeating: 0, count: frames * half)
        var sines = cosines
        for frame in 0..<frames {
            for index in 0..<half {
                let angle = Float(frame) * inverse[index]
                cosines[frame * half + index] = cosf(angle)
                sines[frame * half + index] = sinf(angle)
            }
        }
        return (MLXArray(cosines, [frames, half]), MLXArray(sines, [frames, half]))
    }

    private func applyRope(_ x: MLXArray, _ rope: (cos: MLXArray, sin: MLXArray)) -> MLXArray {
        let half = x.dim(-1) / 2
        let x1 = x[.ellipsis, ..<half]
        let x2 = x[.ellipsis, half...]
        return concatenated([x1 * rope.cos - x2 * rope.sin, x1 * rope.sin + x2 * rope.cos], axis: -1)
    }

    /// Removes each token's component along its own value (upstream `exclusive`).
    private func exclusive(_ output: MLXArray, _ value: MLXArray) -> MLXArray {
        let coefficient = (output * value).sum(axis: -1, keepDims: true)
            / maximum((value * value).sum(axis: -1, keepDims: true), Float(1e-6))
        return output - coefficient * value
    }

    /// Multi-head attention over the middle axis of `[batch, tokens, width]`;
    /// returns the heads joined, before the output projection.
    private func attention(
        _ x: MLXArray, _ projection: Linear, exclusive isExclusive: Bool, rope: (cos: MLXArray, sin: MLXArray)?
    ) -> MLXArray {
        let (batch, tokens, width) = (x.dim(0), x.dim(1), x.dim(2))
        let heads = config.numHeads, headDim = config.headDim
        let qkv = projection(x).reshaped(batch, tokens, 3, heads, headDim).transposed(2, 0, 3, 1, 4)
        var query = qkv[0], key = qkv[1]
        let value = qkv[2]
        if let rope {
            query = applyRope(query, rope)
            key = applyRope(key, rope)
        }
        var output = MLXFast.scaledDotProductAttention(
            queries: query, keys: key, values: value, scale: 1 / Float(headDim).squareRoot(), mask: .none)
        if isExclusive { output = exclusive(output, value) }
        return output.transposed(0, 2, 1, 3).reshaped(batch, tokens, width)
    }

    /// `_TimeBlock` over `[bands, frames, width]`.
    private func timeAttention(_ x: MLXArray, _ block: TimeBlock, rope: (cos: MLXArray, sin: MLXArray)) -> MLXArray {
        let attended = attention(block.norm1(x), block.qkv, exclusive: config.exclusiveTime, rope: rope)
        let y = x + block.proj(attended)
        return y + block.ffn2(gelu(block.ffn1(block.norm2(y))))
    }

    /// `_BandTransformer` across the bands of each frame of `[bands, frames, width]`.
    private func bandAttention(_ x: MLXArray, _ block: BandBlock) -> MLXArray {
        let byFrame = x.transposed(1, 0, 2)
        let attended = attention(block.norm1(byFrame), block.inProj, exclusive: config.exclusiveBand, rope: nil)
        let y = byFrame + block.outProj(attended)
        return (y + block.linear2(gelu(block.linear1(block.norm2(y))))).transposed(1, 0, 2)
    }

    // MARK: Mamba

    /// One selective state-space layer over `[rows, frames, width]` (upstream `Mamba.forward`, no carried state).
    private func mamba(_ x: MLXArray, _ layer: Mamba, reverse: Bool) -> MLXArray {
        let inner = config.mambaInner, rank = config.dtRank, state = config.mambaState
        let xz = layer.inProj(x)
        let z = xz[.ellipsis, inner...]
        let u = InworldKernels.causalConvSiLU(
            xz[.ellipsis, ..<inner], weight: layer.convWeight, bias: layer.convBias, reverse: reverse)
        let projected = layer.xProj(u)
        let dt = projected[.ellipsis, ..<rank]
        let b = projected[.ellipsis, rank ..< rank + state]
        let c = projected[.ellipsis, (rank + state)...]
        let y = InworldKernels.selectiveScan(
            u: u, deltaRaw: layer.dtProj(dt), a: layer.a, b: b, c: c, d: layer.d, z: z, reverse: reverse)
        return layer.outProj(y)
    }

    /// `BidirectionalMamba` (two-sided) over `[rows, frames, width]`.
    private func bidirectional(_ x: MLXArray, _ layer: BidirectionalMamba) -> MLXArray {
        let forward = mamba(x, layer.forward, reverse: false) + x
        eval(forward)
        let backward = mamba(x, layer.backward, reverse: true) + x
        return layer.norm(layer.outputProj(concatenated([forward, backward], axis: -1)))
    }
}

// MARK: - Loading

extension InworldNetwork {
    /// Builds the network from checkpoint tensors (upstream key layout,
    /// PyTorch `[out, in]` linear weights), checking every tensor's shape and
    /// that no tensor is left over except training-side extras.
    init(config: InworldConfig, tensors rawTensors: [String: MLXArray]) throws {
        self.config = config
        spectral = InworldSpectral(nFFT: config.nFFT, hop: config.hopLength)
        edges = config.bandEdges
        maxBandBins = edges.map(\.count).max() ?? 1

        // Training wrapper prefixes are stripped as upstream `_strip_prefixes` does.
        var tensors = rawTensors
        for prefix in ["_forward_module.", "module.", "model.", "isolator."]
        where tensors.keys.contains(where: { $0.hasPrefix(prefix) }) {
            tensors = Dictionary(uniqueKeysWithValues: tensors.map {
                ($0.key.hasPrefix(prefix) ? String($0.key.dropFirst(prefix.count)) : $0.key, $0.value)
            })
        }
        var used = Set<String>()
        func tensor(_ key: String, _ shape: [Int]) throws -> MLXArray {
            guard let array = tensors[key] else { throw InworldIsolatorError.missingTensor(key) }
            guard array.shape == shape else {
                throw InworldIsolatorError.unexpectedShape(key, expected: shape, actual: array.shape)
            }
            used.insert(key)
            return array.asType(.float32)
        }
        func linear(_ key: String, _ input: Int, _ output: Int, bias: Bool = true) throws -> Linear {
            Linear(
                weight: try tensor("\(key).weight", [output, input]).transposed(),
                bias: bias ? try tensor("\(key).bias", [output]) : nil)
        }
        func layerNorm(_ key: String, _ size: Int) throws -> LayerNorm {
            LayerNorm(weight: try tensor("\(key).weight", [size]), bias: try tensor("\(key).bias", [size]))
        }

        let width = config.width, bands = edges.count, bins = config.binCount
        let inner = config.mambaInner, state = config.mambaState, rank = config.dtRank, kernel = config.mambaConv
        let maxFeatures = 2 * maxBandBins

        // Encoders.
        var gather = [Int32](repeating: Int32(2 * bins), count: bands * maxFeatures)
        var mask = [Float](repeating: 0, count: bands * maxFeatures)
        var counts = [Float]()
        var normWeights = [MLXArray](), normBiases = [MLXArray](), weights = [MLXArray](), biases = [MLXArray]()
        for (band, range) in edges.enumerated() {
            let features = 2 * range.count
            for feature in 0..<features {
                gather[band * maxFeatures + feature] = Int32(2 * range.lowerBound + feature)
                mask[band * maxFeatures + feature] = 1
            }
            counts.append(Float(features))
            let pad = maxFeatures - features
            normWeights.append(padded(try tensor("encoders.\(band).0.weight", [features]), widths: [IntOrPair((0, pad))]))
            normBiases.append(padded(try tensor("encoders.\(band).0.bias", [features]), widths: [IntOrPair((0, pad))]))
            weights.append(padded(
                try tensor("encoders.\(band).1.weight", [width, features]).transposed(),
                widths: [IntOrPair((0, pad)), IntOrPair(0)]))
            biases.append(try tensor("encoders.\(band).1.bias", [width]))
        }
        encoderGather = MLXArray(gather)
        encoderMask = MLXArray(mask, [bands, 1, maxFeatures])
        encoderFeatureCount = MLXArray(counts, [bands, 1, 1])
        encoderNormWeight = stacked(normWeights).expandedDimensions(axis: 1)
        encoderNormBias = stacked(normBiases).expandedDimensions(axis: 1)
        encoderWeight = stacked(weights)
        encoderBias = stacked(biases).expandedDimensions(axis: 1)
        bandEmbedding = try tensor("band_embedding", [bands, width]).expandedDimensions(axis: 1)

        // Blocks.
        func mambaLayer(_ key: String) throws -> Mamba {
            Mamba(
                inProj: try linear("\(key).in_proj", width, 2 * inner, bias: false),
                convWeight: try tensor("\(key).conv1d.weight", [inner, 1, kernel]).reshaped(inner, kernel),
                convBias: try tensor("\(key).conv1d.bias", [inner]),
                xProj: try linear("\(key).x_proj", inner, rank + 2 * state, bias: false),
                dtProj: try linear("\(key).dt_proj", rank, inner),
                a: -exp(try tensor("\(key).A_log", [inner, state])),
                d: try tensor("\(key).D", [inner]),
                outProj: try linear("\(key).out_proj", inner, width, bias: false))
        }
        func bidirectionalLayer(_ key: String) throws -> BidirectionalMamba {
            BidirectionalMamba(
                forward: try mambaLayer("\(key).forward_blocks"),
                backward: try mambaLayer("\(key).backward_blocks"),
                outputProj: try linear("\(key).output_proj", 2 * width, width),
                norm: try layerNorm("\(key).norm", width))
        }
        blocks = try config.blockTypes.enumerated().map { index, type in
            let key = "blocks.\(index)"
            switch type {
            case .transformer:
                let time = "\(key).time_transformer"
                let band = "\(key).band_transformer"
                return .attention(
                    time: TimeBlock(
                        norm1: try layerNorm("\(time).norm1", width),
                        qkv: try linear("\(time).attn.qkv", width, 3 * width),
                        proj: try linear("\(time).attn.proj", width, width),
                        norm2: try layerNorm("\(time).norm2", width),
                        ffn1: try linear("\(time).ffn.0", width, 4 * width),
                        ffn2: try linear("\(time).ffn.3", 4 * width, width)),
                    band: BandBlock(
                        norm1: try layerNorm("\(band).norm1", width),
                        inProj: Linear(
                            weight: try tensor("\(band).self_attn.in_proj_weight", [3 * width, width]).transposed(),
                            bias: try tensor("\(band).self_attn.in_proj_bias", [3 * width])),
                        outProj: try linear("\(band).self_attn.out_proj", width, width),
                        norm2: try layerNorm("\(band).norm2", width),
                        linear1: try linear("\(band).linear1", width, 4 * width),
                        linear2: try linear("\(band).linear2", 4 * width, width)))
            case .mamba:
                return .mamba(
                    time: try bidirectionalLayer("\(key).time_mamba"),
                    band: try bidirectionalLayer("\(key).freq_mamba"))
            }
        }

        // Decoder. `condition(controls)` with the fixed controls [1, 0] is a constant FiLM.
        let conditionWeight = try tensor("denoise_decoder.condition.weight", [2 * width, 2])
        let condition = matmul(MLXArray([Float(1), 0], [1, 2]), conditionWeight.transposed()).reshaped(-1)
            + (try tensor("denoise_decoder.condition.bias", [2 * width]))
        decoderScale = 1 + condition[..<width]
        decoderShift = condition[width...]
        var headNormWeights = [MLXArray](), headNormBiases = [MLXArray]()
        var weights1 = [MLXArray](), biases1 = [MLXArray](), weights2 = [MLXArray](), biases2 = [MLXArray]()
        var bandOfBin = [Int32](repeating: 0, count: bins)
        for (band, range) in edges.enumerated() {
            let key = "denoise_decoder.heads.\(band)"
            let outputs = 4 * range.count, pad = 4 * maxBandBins - outputs
            headNormWeights.append(try tensor("\(key).0.weight", [width]))
            headNormBiases.append(try tensor("\(key).0.bias", [width]))
            weights1.append(try tensor("\(key).1.weight", [width, width]).transposed())
            biases1.append(try tensor("\(key).1.bias", [width]))
            weights2.append(padded(
                try tensor("\(key).3.weight", [outputs, width]).transposed(),
                widths: [IntOrPair(0), IntOrPair((0, pad))]))
            biases2.append(padded(try tensor("\(key).3.bias", [outputs]), widths: [IntOrPair((0, pad))]))
            for (local, bin) in range.enumerated() { bandOfBin[bin] = Int32(band * maxBandBins + local) }
        }
        headNormWeight = stacked(headNormWeights).expandedDimensions(axis: 1)
        headNormBias = stacked(headNormBiases).expandedDimensions(axis: 1)
        headWeight1 = stacked(weights1)
        headBias1 = stacked(biases1).expandedDimensions(axis: 1)
        headWeight2 = stacked(weights2)
        headBias2 = stacked(biases2).expandedDimensions(axis: 1)
        decoderGather = MLXArray(bandOfBin)

        window = MLXArray(spectral.window)

        // Upstream `_check_state_keys`: only training-side extras may be spare.
        let trainingOnly = ["discriminator.", "disc.", "ema.", "optimizer.", "scheduler.", "enhance_decoder.",
                            "decision_head."]
        if let spare = tensors.keys.sorted().first(where: { key in
            !used.contains(key) && !trainingOnly.contains(where: { key.hasPrefix($0) })
        }) {
            throw InworldIsolatorError.unsupportedConfig("unexpected tensor \(spare)")
        }
    }
}
