import AudioCore
import AVFoundation
import XCTest
@testable import Export

final class ExportTests: XCTestCase {
    private var dir: URL!

    override func setUpWithError() throws {
        dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: dir.path)
        try? FileManager.default.removeItem(at: dir)
    }

    private func sourceFile(seconds: Double = 10) throws -> URL {
        let url = dir.appendingPathComponent("source.caf")
        let writer = try MonoFileWriter(url: url, sampleRate: 48_000, format: .cafFloat32)
        try writer.append((0..<Int(seconds * 48_000)).map { Float(0.3 * sin(2 * Double.pi * 440 * Double($0) / 48_000)) })
        try writer.close()
        return url
    }

    private func assertDecodes(_ url: URL, seconds: Double, bitrateKbps: Int, file: StaticString = #filePath, line: UInt = #line) throws {
        let audio = try AVAudioFile(forReading: url)
        XCTAssertEqual(audio.fileFormat.channelCount, 1, file: file, line: line)
        let duration = Double(audio.length) / audio.fileFormat.sampleRate
        XCTAssertEqual(duration, seconds, accuracy: 0.1, file: file, line: line)
        let size = Double((try FileManager.default.attributesOfItem(atPath: url.path)[.size] as! NSNumber).intValue)
        let expected = Double(bitrateKbps) * 1000 / 8 * seconds
        XCTAssertEqual(size, expected, accuracy: expected * 0.25, file: file, line: line)
    }

    func testM4AExportDecodesWithExpectedDurationAndSize() throws {
        let destination = dir.appendingPathComponent("out.m4a")
        try Exporter.export(from: try MonoFileReader(url: try sourceFile()), options: ExportOptions(format: .m4a, bitrateKbps: 128), to: destination)
        try assertDecodes(destination, seconds: 10, bitrateKbps: 128)
    }

    func testMP3ExportDecodesWithExpectedDuration() throws {
        let destination = dir.appendingPathComponent("out.mp3")
        try Exporter.export(from: try MonoFileReader(url: try sourceFile()), options: ExportOptions(format: .mp3, bitrateKbps: 128), to: destination)
        try assertDecodes(destination, seconds: 10, bitrateKbps: 128)
    }

    func testFailedExportLeavesExistingFileIntact() throws {
        let source = try sourceFile(seconds: 1)
        let target = dir.appendingPathComponent("locked")
        try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
        let destination = target.appendingPathComponent("out.m4a")
        let original = Data("previous export".utf8)
        try original.write(to: destination)
        try FileManager.default.setAttributes([.posixPermissions: 0o555], ofItemAtPath: target.path)
        defer { try? FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: target.path) }

        XCTAssertThrowsError(try Exporter.export(
            from: try MonoFileReader(url: source), options: ExportOptions(format: .m4a), to: destination))
        XCTAssertEqual(try Data(contentsOf: destination), original)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: target.path), ["out.m4a"])
    }
}
