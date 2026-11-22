import Foundation
import XCTest
import ReuseMLX

/// Locates the upstream NemotronLabs checkpoint named by MICPOLISH_REUSE_SOURCE
/// and converts it once per test process.
enum ReuseTestModel {
    static func sourceCheckpoint() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment["MICPOLISH_REUSE_SOURCE"],
              FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("Set MICPOLISH_REUSE_SOURCE to the NemotronLabs model.safetensors")
        }
        return URL(fileURLWithPath: path)
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var converted: URL?

    static func convertedWeights() throws -> URL {
        let source = try sourceCheckpoint()
        lock.lock()
        defer { lock.unlock() }
        if let converted { return converted }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("micpolish-tests-\(ProcessInfo.processInfo.processIdentifier)")
            .appendingPathComponent("model_mlx.safetensors")
        if !FileManager.default.fileExists(atPath: url.path) {
            _ = try ReuseWeightConverter.convert(input: source, output: url)
        }
        converted = url
        return url
    }
}
