import ArgumentParser
import AudioCore
import Export
import Foundation
import Mastering
import ModelStore
import Pipeline
import ReuseMLX
import Sessions

@main
struct MicPolishCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "micpolish",
        abstract: "Enhance speech recordings on-device: RE-USE (MLX) → mastering → M4A/MP3.",
        subcommands: [Models.self, Process.self]
    )
}

enum Preset: String, ExpressibleByArgument, CaseIterable {
    case `default`, macbook

    var settings: MasteringSettings {
        switch self {
        case .default: .default
        case .macbook: .macbookMicCorrection
        }
    }
}

enum Variant: String, ExpressibleByArgument, CaseIterable {
    case reuseNemotron = "reuse-nemotron"
    case reuseNemotronLaughter = "reuse-nemotron-laughter"

    var descriptor: ModelDescriptor { ModelCatalog.descriptor(id: rawValue)! }
}

struct Models: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Manage downloaded models.", subcommands: [Install.self, List.self])

    struct Install: AsyncParsableCommand {
        @Option(help: "Model variant.") var variant: Variant = .reuseNemotron

        func run() async throws {
            let url = try await installModel(variant.descriptor)
            print("Installed \(variant.rawValue): \(url.path)")
        }
    }

    struct List: AsyncParsableCommand {
        func run() async throws {
            let store = ModelStore(root: ModelStore.defaultRoot)
            for d in ModelCatalog.all {
                let path = await store.installedWeightsURL(d)?.path ?? "not installed"
                print("\(d.id)\t\(d.revision.prefix(8))\t\(path)")
            }
            print("Disk usage: \(await store.diskUsageBytes() / 1_048_576) MB")
        }
    }
}

func installModel(_ d: ModelDescriptor) async throws -> URL {
    let store = ModelStore(root: ModelStore.defaultRoot)
    if let existing = await store.installedWeightsURL(d) { return existing }
    print("Downloading \(d.repo) @ \(d.revision.prefix(8)) (\(d.byteCount / 1_048_576) MB)…")
    return try await store.install(d) { fraction in
        FileHandle.standardError.write(Data(String(format: "\r  %3.0f%%", fraction * 100).utf8))
    }
}

struct Process: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Process one audio file end to end.")

    @Argument(help: "Input audio file (any format AVFoundation can read).") var input: String
    @Option(name: .shortAndLong, help: "Output file ending in .m4a or .mp3.") var output: String
    @Option(help: "Bitrate in kbps: 96, 128 or 192.") var bitrate = 128
    @Option(help: "Mastering preset.") var preset: Preset = .default
    @Option(help: "Model variant.") var variant: Variant = .reuseNemotron

    func validate() throws {
        guard ["m4a", "mp3"].contains(URL(fileURLWithPath: output).pathExtension.lowercased()) else {
            throw ValidationError("--output must end in .m4a or .mp3")
        }
        guard ExportOptions.bitrates.contains(bitrate) else {
            throw ValidationError("--bitrate must be one of \(ExportOptions.bitrates)")
        }
    }

    func run() async throws {
        let inputURL = URL(fileURLWithPath: input)
        let outputURL = URL(fileURLWithPath: output)
        let format = ExportFormat(rawValue: outputURL.pathExtension.lowercased())!

        let weights = try await installModel(variant.descriptor)
        let enhancer = try MambaEnhancer(modelWeightsURL: weights)

        let sessionRoot = FileManager.default.temporaryDirectory.appendingPathComponent("micpolish-cli-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: sessionRoot) }
        let manifest = try SessionStore(root: sessionRoot).importFile(inputURL, name: "cli")
        let original = SessionStore(root: sessionRoot).originalURL(for: manifest)
        let workspace = Workspace(cacheRoot: Workspace.defaultCacheRoot, sessionID: manifest.id)
        defer { try? workspace.purge() }

        let clock = ContinuousClock()
        let aiStart = clock.now
        let pipeline = ProcessingPipeline(
            stages: [ReuseStage(enhancer: enhancer, variantID: variant.rawValue, revision: variant.descriptor.revision),
                     ResampleStage()],
            workspace: workspace)
        let processed = try await pipeline.run(original: original) { p in
            FileHandle.standardError.write(Data(String(format: "\r  %@ %3.0f%%", p.stageID, p.fraction * 100).utf8))
        }
        let aiSeconds = (clock.now - aiStart).seconds
        FileHandle.standardError.write(Data("\n".utf8))

        let mastered = workspace.directory.appendingPathComponent("mastered.caf")
        let writer = try MonoFileWriter(url: mastered, sampleRate: AudioCore.sampleRate, format: .cafFloat32)
        let report = try MasteringRenderer.render(input: processed.url, settings: preset.settings, output: writer)
        try writer.close()
        try Exporter.export(from: try MonoFileReader(url: mastered),
                            options: ExportOptions(format: format, bitrateKbps: bitrate), to: outputURL)

        let duration = manifest.original.durationSeconds
        print(String(format: "Input: %.1f s @ %d Hz", duration, manifest.original.sampleRate))
        print(String(format: "RE-USE (%@): %.2f s, RTF %.3f", variant.rawValue, aiSeconds, aiSeconds / max(duration, 0.001)))
        print(String(format: "Loudness before gain: %@ LUFS, gain %+.1f dB",
                     report.measuredLUFS.map { String(format: "%.1f", $0) } ?? "n/a", report.appliedGainDB))
        print(String(format: "Output: %@ LUFS, true peak %.2f dBTP",
                     report.outputLUFS.map { String(format: "%.1f", $0) } ?? "n/a", report.outputTruePeakDBTP))
        print("Wrote \(outputURL.path)")
    }
}

extension Duration {
    var seconds: Double {
        let c = components
        return Double(c.seconds) + Double(c.attoseconds) / 1e18
    }
}
