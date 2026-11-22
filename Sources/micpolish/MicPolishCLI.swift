import ArgumentParser
import AudioCore
import Export
import Foundation
import LavaMLX
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
        subcommands: [Models.self, Process.self, Abx.self]
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
        @Option(help: "Model id: \(ModelCatalog.all.map(\.id).joined(separator: ", ")).")
        var model = ModelCatalog.reuse.id

        func validate() throws {
            guard ModelCatalog.descriptor(id: model) != nil else { throw ValidationError("unknown model \(model)") }
        }

        func run() async throws {
            let url = try await installModel(ModelCatalog.descriptor(id: model)!)
            print("Installed \(model): \(url.path)")
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
    @Option(help: "LavaSR high-band enhancement: off, auto, or a start frequency in Hz.") var highBand = "off"

    func validate() throws {
        guard highBand == "off" || highBand == "auto" || Double(highBand) != nil else {
            throw ValidationError("--high-band must be off, auto or a frequency in Hz")
        }
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
        var stages: [any AudioStage] = [
            ReuseStage(enhancer: enhancer, variantID: variant.rawValue, revision: variant.descriptor.revision),
            ResampleStage(),
        ]
        if highBand != "off" {
            let lava = try LavaEnhancer(weightsURL: try await installModel(ModelCatalog.lavaSR))
            let cutoff = Double(highBand) ?? (manifest.original.sampleRate < 48_000 ? Double(manifest.original.sampleRate) / 2 : nil)
            stages.append(LavaStage(enhancer: lava, revision: ModelCatalog.lavaSR.revision, cutoffHz: cutoff))
        }
        let pipeline = ProcessingPipeline(stages: stages, workspace: workspace)
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

/// Writes level-matched, shuffled variants for blind listening (design §7.8):
/// RE-USE only, and RE-USE followed by LavaSR at automatic, 12 kHz and 8 kHz
/// start frequencies, each through the same mastering preset. The key is
/// written to key.json; listen before opening it.
struct Abx: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Generate blind-listening variants.")

    @Argument(help: "Input audio file.") var input: String
    @Option(help: "Output folder.") var out: String
    @Option(help: "Mastering preset.") var preset: Preset = .default

    func run() async throws {
        let outDir = URL(fileURLWithPath: out, isDirectory: true)
        try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
        let reuse = try MambaEnhancer(modelWeightsURL: try await installModel(ModelCatalog.reuse))
        let lava = try LavaEnhancer(weightsURL: try await installModel(ModelCatalog.lavaSR))

        let sessionRoot = FileManager.default.temporaryDirectory.appendingPathComponent("micpolish-abx-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: sessionRoot) }
        let store = SessionStore(root: sessionRoot)
        let manifest = try store.importFile(URL(fileURLWithPath: input), name: "abx")
        let workspace = Workspace(cacheRoot: Workspace.defaultCacheRoot, sessionID: manifest.id)
        defer { try? workspace.purge() }

        let sourceNyquist = manifest.original.sampleRate < 48_000 ? Double(manifest.original.sampleRate) / 2 : nil
        let variants: [(String, Double??)] = [
            ("RE-USE", .none), ("RE-USE + LavaSR auto", .some(sourceNyquist)),
            ("RE-USE + LavaSR 12 kHz", .some(12_000)), ("RE-USE + LavaSR 8 kHz", .some(8_000)),
        ]
        let letters = ["A", "B", "C", "D"].shuffled()
        var key: [String: String] = [:]
        for ((label, cutoff), letter) in zip(variants, letters) {
            var stages: [any AudioStage] = [
                ReuseStage(enhancer: reuse, variantID: ModelCatalog.reuse.id, revision: ModelCatalog.reuse.revision),
                ResampleStage(),
            ]
            if let cutoff { stages.append(LavaStage(enhancer: lava, revision: ModelCatalog.lavaSR.revision, cutoffHz: cutoff)) }
            let processed = try await ProcessingPipeline(stages: stages, workspace: workspace)
                .run(original: store.originalURL(for: manifest)) { _ in }
            let writer = try MonoFileWriter(url: outDir.appendingPathComponent("\(letter).flac"), sampleRate: 48_000, format: .flac24)
            let report = try MasteringRenderer.render(input: processed.url, settings: preset.settings, output: writer)
            try writer.close()
            key[letter] = "\(label) (output \(report.outputLUFS.map { String(format: "%.1f", $0) } ?? "n/a") LUFS)"
            print("\(letter) written")
        }
        let data = try JSONSerialization.data(withJSONObject: key, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: outDir.appendingPathComponent("key.json"))
        print("Variants in \(outDir.path); the answer key is key.json.")
    }
}
