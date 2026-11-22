import XCTest
@testable import Mastering

final class LoudnessMeterTests: XCTestCase {
    func testSine1kAtMinus23IsMinus23LUFS() throws {
        var meter = LoudnessMeter(sampleRate: 48_000)
        meter.add(TestSignals.sine(frequency: 1000, amplitude: pow(10, -23.0 / 20), seconds: 10))
        let lufs = try XCTUnwrap(meter.integratedLUFS)
        XCTAssertEqual(lufs, -23.0, accuracy: 0.1)
    }

    func testGatingIgnoresSilence() throws {
        let silence = [Float](repeating: 0, count: 480_000)
        let tone = TestSignals.sine(frequency: 1000, amplitude: pow(10, -23.0 / 20), seconds: 20)
        var meter = LoudnessMeter(sampleRate: 48_000)
        // Feed in uneven pieces: the meter must not depend on call boundaries.
        let signal = silence + tone + silence
        var index = 0
        for size in [1, 4799, 100_000, 333_333] .lazy.cycled() {
            guard index < signal.count else { break }
            let end = min(index + size, signal.count)
            meter.add(Array(signal[index..<end]))
            index = end
        }
        XCTAssertEqual(try XCTUnwrap(meter.integratedLUFS), -23.0, accuracy: 0.1)
    }

    func testSilenceIsNil() {
        var meter = LoudnessMeter(sampleRate: 48_000)
        meter.add([Float](repeating: 0, count: 96_000))
        XCTAssertNil(meter.integratedLUFS)
    }

    func testTruePeakDetectsIntersamplePeak() {
        // fs/4 sine sampled at ±45°: every sample is ±0.5 (-6.02 dBFS) while the
        // waveform peaks at 0.707 (-3.01 dBTP) between samples.
        var meter = LoudnessMeter(sampleRate: 48_000)
        meter.add(TestSignals.sine(frequency: 12_000, amplitude: 0.5 * 2.0.squareRoot(), seconds: 1, phase: Double.pi / 4))
        XCTAssertEqual(meter.truePeakDBTP, -3.01, accuracy: 0.2)
    }
}

private extension Sequence {
    func cycled() -> AnySequence<Element> {
        let items = Array(self)
        return AnySequence { () -> AnyIterator<Element> in
            var i = 0
            return AnyIterator {
                defer { i += 1 }
                return items[i % items.count]
            }
        }
    }
}
