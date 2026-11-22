import MLX
import MLXNN
import XCTest
@testable import ReuseMLX

/// Layer-by-layer comparison against probes written by
/// `tools/parity/reuse_reference.py --probes`. Diagnostic only: runs when
/// MICPOLISH_PARITY_PROBES points at a probe file and its matching input wav.
final class ReuseProbeDiagnostics: XCTestCase {
    func testProbes() async throws {
        guard let probesPath = ProcessInfo.processInfo.environment["MICPOLISH_PARITY_PROBES"],
              let samplesPath = ProcessInfo.processInfo.environment["MICPOLISH_PARITY_SAMPLES"] else {
            throw XCTSkip("diagnostic")
        }
        let sampleRate = Int(ProcessInfo.processInfo.environment["MICPOLISH_PARITY_RATE"] ?? "16000")!
        let probes = try MLX.loadArrays(url: URL(fileURLWithPath: probesPath))
        let weights = try MLX.loadArrays(url: try ReuseTestModel.convertedWeights())
        let model = SEMamba(numBlocks: 30)
        model.update(parameters: ModuleParameters.unflattened(weights))

        func rel(_ name: String, _ a: MLXArray, _ b: MLXArray) {
            let err = sqrt(((a - b) * (a - b)).sum() / (b * b).sum()).item(Float.self)
            let maxAbs = abs(a - b).max().item(Float.self)
            print("PROBE \(name): rel=\(err) maxAbs=\(maxAbs) shape=\(a.shape) vs \(b.shape)")
        }

        let data = try Data(contentsOf: URL(fileURLWithPath: samplesPath))
        let samples = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        let enhancer = try MambaEnhancer(modelWeightsURL: try ReuseTestModel.convertedWeights())
        let p = MambaEnhancer.scaledSTFTParams(sampleRate: sampleRate)
        let (mag, pha, _) = try enhancer.stftInputs(samples: samples, nFFT: p.nFFT, hop: p.hop, win: p.win)
        rel("input_mag", mag, probes["input_mag"]!)
        rel("input_pha", pha, probes["input_pha"]!)
        let refMag = probes["input_mag"]!
        let w = which(refMag .> 1.0, MLXArray(1.0), MLXArray(0.0))
        rel("input_pha(mag>1)", pha * w, probes["input_pha"]! * w)

        // iSTFT of the reference model output vs reference waveform.
        let ref = try Data(contentsOf: URL(fileURLWithPath: ProcessInfo.processInfo.environment["MICPOLISH_PARITY_REF"]!))
            .withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        let enhMag = suppressSweepArtifactFrames(expm1(maximum(probes["model_mag"]!, MLXArray(0.0))))
        let padLen = p.nFFT / 2
        let rec = enhancer.istft(mag: enhMag, pha: probes["model_pha"]!, originalCount: samples.count + 2 * padLen,
                                 nFFT: p.nFFT, hop: p.hop, win: p.win)
        rel("istft", MLXArray(Array(rec[padLen..<(padLen + samples.count)])), MLXArray(ref))

        // Model from the reference's own inputs.
        let inMag = probes["input_mag"]!, inPha = probes["input_pha"]!
        var x = MLX.stacked([inMag.transposed(0, 2, 1), inPha.transposed(0, 2, 1)], axis: -1)
        x = MLX.padded(x, widths: [IntOrPair(0), IntOrPair((0, 2)), IntOrPair((0, 2)), IntOrPair(0)])
        x = model.denseEncoder(x)
        rel("encoder_out", x, probes["encoder_out"]!.transposed(0, 2, 3, 1))
        x = model.tfMamba[0](x)
        rel("block_0_out", x, probes["block_0_out"]!.transposed(0, 2, 3, 1))
        for block in model.tfMamba.dropFirst() { x = block(x) }
        rel("block_last_out", x, probes["block_last_out"]!.transposed(0, 2, 3, 1))
        let (outMag, outPha) = model(inMag, inPha)
        rel("model_mag", outMag, probes["model_mag"]!)
        rel("model_pha", outPha, probes["model_pha"]!)
    }
}
