import XCTest
import AudioCore

final class SmokeTests: XCTestCase {
    func testSampleRateConstant() {
        XCTAssertEqual(AudioCore.sampleRate, 48_000)
    }
}
