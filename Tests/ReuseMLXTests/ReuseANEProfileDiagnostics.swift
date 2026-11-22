import MLX
import MLXNN
import XCTest
@testable import ReuseMLX

/// Per-stage timing of one 5 s 48 kHz window, MLX vs Core ML. Diagnostic only:
/// runs when MICPOLISH_REUSE_PROFILE is set (use a release build).
final class ReuseANEProfileDiagnostics: XCTestCase {
    func testStageTimes() throws {
        guard ProcessInfo.processInfo.environment["MICPOLISH_REUSE_PROFILE"] != nil else {
            throw XCTSkip("diagnostic")
        }
        let accelerator = try ReuseANETests.accelerator()
        let weights = try MLX.loadArrays(url: try ReuseTestModel.convertedWeights())
        let speech = try ReuseANETests.speech()
        try Device.withDefaultDevice(.gpu) {
            let model = SEMamba(numBlocks: 30)
            model.update(parameters: ModuleParameters.unflattened(weights))
            let enhancer = try MambaEnhancer(modelWeightsURL: try ReuseTestModel.convertedWeights())
            let p = MambaEnhancer.scaledSTFTParams(sampleRate: 48_000)
            let window = Array(speech[48_000 ..< 48_000 + ReuseANEAccelerator.windowSamples])

            func time(_ label: String, _ body: () throws -> Void) rethrows {
                var samples = [Double]()
                for _ in 0 ..< 4 {
                    let start = ContinuousClock.now
                    try body()
                    let d = ContinuousClock.now - start
                    samples.append(Double(d.components.attoseconds) * 1e-15 + Double(d.components.seconds) * 1e3)
                }
                // Drop the first (warm-up) run.
                let warm = samples.dropFirst().sorted()
                print(String(format: "PROFILE %@: median %.1f ms (warm-up %.1f ms)", label, warm[warm.count / 2], samples[0]))
            }

            let (mag, pha, _) = try enhancer.stftInputs(samples: window, nFFT: p.nFFT, hop: p.hop, win: p.win)
            MLX.eval(mag, pha)
            var x = MLX.stacked([mag.transposed(0, 2, 1), pha.transposed(0, 2, 1)], axis: -1)
            x = MLX.padded(x, widths: [IntOrPair(0), IntOrPair((0, 2)), IntOrPair((0, 2)), IntOrPair(0)])
            MLX.eval(x)

            var latent = x
            time("MLX encoder") { latent = model.denseEncoder(x); MLX.eval(latent) }
            var mambaOut = latent
            time("MLX 30 Mamba blocks") {
                var h = latent
                for block in model.tfMamba { h = block(h) }
                mambaOut = h
                MLX.eval(mambaOut)
            }
            time("MLX decoders") {
                let m = model.maskDecoder(mambaOut), ph = model.phaseDecoder(mambaOut)
                MLX.eval(m, ph)
            }
            let xCoreML = MLX.stacked([mag, pha], axis: 1).transposed(0, 1, 3, 2)
            MLX.eval(xCoreML)
            var h16 = latent
            time("Core ML encoder (incl. copies)") { h16 = try! accelerator.encode(xCoreML); MLX.eval(h16) }
            let hN = mambaOut.transposed(0, 3, 1, 2)
            MLX.eval(hN)
            time("Core ML decoders (incl. copies)") { MLX.eval(try! accelerator.decode(hN)) }
            time("MLX full window") {
                let (a, b) = model(mag, pha)
                MLX.eval(a, b)
            }
            time("hybrid full window") {
                let (a, b) = try! model.hybridForward(mag, pha, accelerator: accelerator)
                MLX.eval(a, b)
            }
            // Core ML on another thread while MLX runs the Mamba stack: the
            // overlap a pipelined session could get.
            let xT = try ReuseANEAccelerator.coreMLInput(xCoreML)
            let hT = try ReuseANEAccelerator.coreMLInput(hN)
            time("Mamba on GPU || Core ML dec+enc on ANE") {
                let group = DispatchGroup()
                DispatchQueue.global(qos: .userInitiated).async(group: group) {
                    _ = try! accelerator.runDecoders(hT)
                    _ = try! accelerator.runEncoder(xT)
                }
                var h = latent
                for block in model.tfMamba { h = block(h) }
                MLX.eval(h)
                group.wait()
            }
            time("Core ML dec+enc only (prepared inputs)") {
                _ = try! accelerator.runDecoders(hT)
                _ = try! accelerator.runEncoder(xT)
            }
        }
    }
}
