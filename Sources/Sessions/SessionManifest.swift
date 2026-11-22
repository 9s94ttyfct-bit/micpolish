import Export
import Foundation
import Mastering

/// Where LavaSR starts generating high frequencies.
public enum HighBandCutoff: String, Codable, CaseIterable, Sendable {
    /// The source's Nyquist frequency for band-limited imports, otherwise
    /// the measured effective bandwidth of the restored audio.
    case auto
    case hz8000 = "8000"
    case hz12000 = "12000"
    case hz16000 = "16000"

    public var hertz: Double? {
        switch self {
        case .auto: nil
        case .hz8000: 8000
        case .hz12000: 12_000
        case .hz16000: 16_000
        }
    }
}

/// AI-stage settings. New fields must have defaults so older manifests decode.
public struct ProcessingSettings: Codable, Equatable, Sendable {
    public var reuseVariant: String = "reuse-nemotron"
    /// LavaSR v2 high-frequency enhancement after RE-USE.
    public var highBandEnabled = false
    public var highBandCutoff: HighBandCutoff = .auto

    public init(reuseVariant: String = "reuse-nemotron", highBandEnabled: Bool = false,
                highBandCutoff: HighBandCutoff = .auto) {
        self.reuseVariant = reuseVariant
        self.highBandEnabled = highBandEnabled
        self.highBandCutoff = highBandCutoff
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        reuseVariant = try c.decodeIfPresent(String.self, forKey: .reuseVariant) ?? "reuse-nemotron"
        highBandEnabled = try c.decodeIfPresent(Bool.self, forKey: .highBandEnabled) ?? false
        highBandCutoff = try c.decodeIfPresent(HighBandCutoff.self, forKey: .highBandCutoff) ?? .auto
    }
}

public struct OriginalAudio: Codable, Equatable, Sendable {
    public var fileName: String
    public var sampleRate: Int
    public var durationSeconds: Double
    public var imported: Bool
}

public struct ExportRecord: Codable, Equatable, Sendable {
    public var fileName: String
    public var exportedAt: Date
    public var modelRevisions: [String: String]
    public var backend: String
    public var processingSeconds: Double
    /// Absolute folder for exports outside the session folder; nil means the
    /// session folder itself.
    public var directory: String?

    public init(fileName: String, exportedAt: Date, modelRevisions: [String: String], backend: String,
                processingSeconds: Double, directory: String? = nil) {
        self.fileName = fileName
        self.exportedAt = exportedAt
        self.modelRevisions = modelRevisions
        self.backend = backend
        self.processingSeconds = processingSeconds
        self.directory = directory
    }
}

/// The kept span of a recording, in seconds on the original's timeline.
/// The original file is never modified.
public struct TrimRange: Codable, Equatable, Sendable {
    public var start: Double
    public var end: Double

    public init(start: Double, end: Double) {
        self.start = start
        self.end = end
    }

    public var duration: Double { max(0, end - start) }
}

/// `session.json`: everything needed to reproduce the export from the
/// original audio.
public struct SessionManifest: Codable, Equatable, Sendable, Identifiable {
    public let id: UUID
    public var name: String
    public let createdAt: Date
    public var original: OriginalAudio
    public var processing = ProcessingSettings()
    public var mastering = MasteringSettings.default
    public var export = ExportOptions()
    public var lastExport: ExportRecord?
    /// Head/tail trim applied after AI processing; nil keeps everything.
    public var trim: TrimRange?

    init(id: UUID, name: String, createdAt: Date, original: OriginalAudio) {
        self.id = id
        self.name = name
        self.createdAt = createdAt
        self.original = original
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(UUID.self, forKey: .id)
        name = try c.decode(String.self, forKey: .name)
        createdAt = try c.decode(Date.self, forKey: .createdAt)
        original = try c.decode(OriginalAudio.self, forKey: .original)
        processing = try c.decodeIfPresent(ProcessingSettings.self, forKey: .processing) ?? ProcessingSettings()
        mastering = try c.decodeIfPresent(MasteringSettings.self, forKey: .mastering) ?? .default
        export = try c.decodeIfPresent(ExportOptions.self, forKey: .export) ?? ExportOptions()
        lastExport = try c.decodeIfPresent(ExportRecord.self, forKey: .lastExport)
        trim = try c.decodeIfPresent(TrimRange.self, forKey: .trim)
    }
}
