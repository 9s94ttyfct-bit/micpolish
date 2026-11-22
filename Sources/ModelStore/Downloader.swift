import Foundation

public protocol Downloader: Sendable {
    /// Downloads `url` to `destination`, which must not exist yet.
    func download(_ url: URL, to destination: URL, progress: @Sendable @escaping (Double) -> Void) async throws
}

public struct URLSessionDownloader: Downloader {
    public init() {}

    public func download(_ url: URL, to destination: URL, progress: @Sendable @escaping (Double) -> Void) async throws {
        let (bytes, response) = try await URLSession.shared.bytes(from: url)
        if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            throw URLError(.badServerResponse)
        }
        let expected = response.expectedContentLength
        FileManager.default.createFile(atPath: destination.path, contents: nil)
        let handle = try FileHandle(forWritingTo: destination)
        defer { try? handle.close() }

        var buffer = Data()
        buffer.reserveCapacity(1 << 20)
        var written: Int64 = 0
        for try await byte in bytes {
            buffer.append(byte)
            if buffer.count >= 1 << 20 {
                try handle.write(contentsOf: buffer)
                written += Int64(buffer.count)
                buffer.removeAll(keepingCapacity: true)
                if expected > 0 { progress(Double(written) / Double(expected)) }
            }
        }
        try handle.write(contentsOf: buffer)
        progress(1)
    }
}
