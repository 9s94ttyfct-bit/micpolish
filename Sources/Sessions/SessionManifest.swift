import Export
import Foundation
import Mastering

/// AI-stage settings. New fields must have defaults so older manifests decode.
public struct ProcessingSettings: Codable, Equatable, Sendable {
    public var reuseVariant: String = "reuse-nemotron"

    public init(reuseVariant: String = "reuse-nemotron") {
        self.reuseVariant = reuseVariant
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        reuseVariant = try c.decodeIfPresent(String.self, forKey: .reuseVariant) ?? "reuse-nemotron"
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

    public init(fileName: String, exportedAt: Date, modelRevisions: [String: String], backend: String, processingSeconds: Double) {
        self.fileName = fileName
        self.exportedAt = exportedAt
        self.modelRevisions = modelRevisions
        self.backend = backend
        self.processingSeconds = processingSeconds
    }
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
    }
}
