import XCTest
import ReuseMLX

final class ReuseEnhancerSmokeTests: XCTestCase {
    func testOneSecondAt48kKeepsLength() async throws {
        let weights = try ReuseTestModel.convertedWeights()
        let enhancer = try MambaEnhancer(modelWeightsURL: weights)
        let session = try await enhancer.makeSession(sampleRate: 48_000, settings: .standard)

        var generator = SystemRandomNumberGenerator()
        let input = (0..<48_000).map { i in
            0.3 * Float(sin(2 * Double.pi * 220 * Double(i) / 48_000))
                + Float.random(in: -0.05...0.05, using: &generator)
        }
        var output = try await session.append(input)
        output += try await session.finish()

        XCTAssertEqual(output.count, 48_000)
        XCTAssertTrue(output.allSatisfy(\.isFinite))
    }
}
