import AudioCore
import Export
import Foundation
import Mastering
import ModelStore
import Pipeline
import Preview
import Recording
import ReuseMLX
import Sessions
import SwiftUI

enum ModelState: Equatable {
    case checking
    case notInstalled
    case downloading(Double)
    case ready
    case failed(String)
}

enum ProcessingState: Equatable {
    case idle
    case running(stage: String, fraction: Double, remaining: TimeInterval?)
    case done
    case failed(String)
}

/// App-wide state and the operations the views trigger.
@MainActor
final class AppModel: ObservableObject {
    @Published var sessions: [SessionManifest] = []
    @Published var selectedID: UUID? {
        didSet { if oldValue != selectedID { sessionChanged() } }
    }
    @Published var modelState: ModelState = .checking
    @Published var processing: ProcessingState = .idle
    @Published var processedURL: URL?
    @Published var exportMessage: String?
    @Published var errorMessage: String?
    @Published var workspaceBytes: Int64 = 0
    @Published var sessionsRoot: URL {
        didSet {
            UserDefaults.standard.set(sessionsRoot.path, forKey: "sessionsRoot")
            store = SessionStore(root: sessionsRoot)
            refresh()
        }
    }

    let recorder: Recorder
    let preview = PreviewPlayer()
    let modelStore = ModelStore(root: ModelStore.defaultRoot)
    let cacheRoot = Workspace.defaultCacheRoot
    private(set) var store: SessionStore
    private var enhancers: [String: MambaEnhancer] = [:]
    private var processingTask: Task<Void, Never>?
    private var gainTask: Task<Void, Never>?

    init() {
        let root = UserDefaults.standard.string(forKey: "sessionsRoot").map { URL(fileURLWithPath: $0) }
            ?? SessionStore.defaultRoot
        sessionsRoot = root
        store = SessionStore(root: root)
        recorder = Recorder(store: SessionStore(root: root), cacheRoot: Workspace.defaultCacheRoot)
    }

    var selected: SessionManifest? {
        sessions.first { $0.id == selectedID }
    }

    // MARK: Lifecycle

    func launch() async {
        try? Workspace.purgeAll(cacheRoot: cacheRoot)
        let recovered = (try? RecordingRecovery.recover(cacheRoot: cacheRoot, store: store)) ?? []
        refresh()
        if let first = recovered.first { selectedID = first.id }
        await checkModel()
    }

    func quit() {
        preview.stop()
        try? Workspace.purgeAll(cacheRoot: cacheRoot)
    }

    func refresh() {
        sessions = (try? store.list()) ?? []
        if selectedID == nil || !sessions.contains(where: { $0.id == selectedID }) {
            selectedID = sessions.first?.id
        }
    }

    // MARK: Models

    func checkModel() async {
        let variant = selected?.processing.reuseVariant ?? ModelCatalog.reuse.id
        let descriptor = ModelCatalog.descriptor(id: variant) ?? ModelCatalog.reuse
        modelState = await modelStore.installedWeightsURL(descriptor) == nil ? .notInstalled : .ready
    }

    func downloadModel(_ descriptor: ModelDescriptor = ModelCatalog.reuse) async {
        modelState = .downloading(0)
        do {
            _ = try await modelStore.install(descriptor) { fraction in
                Task { @MainActor in self.modelState = .downloading(fraction) }
            }
            modelState = .ready
        } catch {
            modelState = .failed(error.localizedDescription)
        }
    }

    private func enhancer(for variant: String) async throws -> (MambaEnhancer, ModelDescriptor) {
        let descriptor = ModelCatalog.descriptor(id: variant) ?? ModelCatalog.reuse
        if let cached = enhancers[descriptor.id] { return (cached, descriptor) }
        let url = try await modelStore.install(descriptor) { _ in }
        let enhancer = try MambaEnhancer(modelWeightsURL: url)
        enhancers[descriptor.id] = enhancer
        return (enhancer, descriptor)
    }

    // MARK: Sessions

    func importFiles(_ urls: [URL]) {
        for url in urls {
            let access = url.startAccessingSecurityScopedResource()
            defer { if access { url.stopAccessingSecurityScopedResource() } }
            do {
                let m = try store.importFile(url, name: url.deletingPathExtension().lastPathComponent)
                refresh()
                selectedID = m.id
            } catch {
                errorMessage = "导入失败：\(error.localizedDescription)"
            }
        }
    }

    func startRecording(input: AudioInput?) async {
        guard await Recorder.requestAccess() else {
            errorMessage = RecorderError.microphoneAccessDenied.localizedDescription
            return
        }
        do { try recorder.start(input: input) } catch { errorMessage = error.localizedDescription }
    }

    func stopRecording(name: String) async {
        do {
            let m = try await recorder.stop(name: name.isEmpty ? "录音" : name)
            refresh()
            selectedID = m.id
        } catch {
            errorMessage = "保存录音失败：\(error.localizedDescription)"
        }
    }

    func delete(_ id: UUID) {
        if id == selectedID { preview.stop() }
        try? Workspace(cacheRoot: cacheRoot, sessionID: id).purge()
        try? store.delete(id)
        refresh()
    }

    func updateSelected(_ change: (inout SessionManifest) -> Void) {
        guard var m = selected else { return }
        change(&m)
        try? store.save(m)
        if let i = sessions.firstIndex(where: { $0.id == m.id }) { sessions[i] = m }
    }

    private func sessionChanged() {
        processingTask?.cancel()
        preview.pause()
        processing = .idle
        processedURL = nil
        exportMessage = nil
        guard let m = selected else { return }
        try? preview.load(original: store.originalURL(for: m), processed: nil)
        preview.setMastering(m.mastering, loudnessGainDB: 0)
        workspaceBytes = Workspace.totalDiskUsage(cacheRoot: cacheRoot)
        Task { await checkModel() }
    }

    // MARK: Processing

    /// Expected AI time for the selected session, from the last recorded RTF.
    var estimatedProcessingSeconds: TimeInterval? {
        guard let m = selected, let last = m.lastExport, m.original.durationSeconds > 0 else { return nil }
        return last.processingSeconds
    }

    func process() {
        guard let m = selected else { return }
        processingTask?.cancel()
        processing = .running(stage: "准备模型", fraction: 0, remaining: nil)
        processingTask = Task {
            do {
                let (enhancer, descriptor) = try await enhancer(for: m.processing.reuseVariant)
                let workspace = Workspace(cacheRoot: cacheRoot, sessionID: m.id)
                let pipeline = ProcessingPipeline(
                    stages: [ReuseStage(enhancer: enhancer, variantID: descriptor.id, revision: descriptor.revision),
                             ResampleStage()],
                    workspace: workspace)
                let started = Date()
                let result = try await pipeline.run(original: store.originalURL(for: m)) { p in
                    let elapsed = Double(p.elapsed.components.seconds)
                    let overall = (Double(p.stageIndex) + p.fraction) / Double(p.stageCount)
                    let remaining = overall > 0.02 ? elapsed / overall - elapsed : nil
                    let label = p.stageID.hasPrefix("reuse") ? "AI 修复（RE-USE）" : "转换采样率"
                    Task { @MainActor in
                        self.processing = .running(stage: label, fraction: overall, remaining: remaining)
                    }
                }
                try Task.checkCancellation()
                processedURL = result.url
                lastProcessingSeconds = Date().timeIntervalSince(started)
                lastBackend = "MLX GPU"
                lastRevisions = [descriptor.id: descriptor.revision]
                try preview.load(original: store.originalURL(for: m), processed: result.url)
                processing = .done
                workspaceBytes = Workspace.totalDiskUsage(cacheRoot: cacheRoot)
                refreshLoudnessGain()
            } catch is CancellationError {
                processing = .idle
            } catch {
                processing = .failed(error.localizedDescription)
            }
        }
    }

    func cancelProcessing() {
        processingTask?.cancel()
        processing = .idle
    }

    private var lastProcessingSeconds: TimeInterval = 0
    private var lastBackend = "MLX GPU"
    private var lastRevisions: [String: String] = [:]

    // MARK: Mastering

    func setMastering(_ settings: MasteringSettings) {
        updateSelected { $0.mastering = settings }
        preview.setMastering(settings, loudnessGainDB: currentGain)
        refreshLoudnessGain()
    }

    private var currentGain: Double = 0

    /// Re-measures the loudness gain for the current settings in the
    /// background (debounced) and applies it to the live preview.
    private func refreshLoudnessGain() {
        guard let processedURL, let settings = selected?.mastering else { return }
        gainTask?.cancel()
        gainTask = Task {
            try? await Task.sleep(nanoseconds: 300_000_000)
            guard !Task.isCancelled else { return }
            let gain = await Task.detached { (try? MasteringRenderer.measureGain(input: processedURL, settings: settings))?.gainDB ?? 0 }.value
            guard !Task.isCancelled else { return }
            currentGain = gain
            preview.setMastering(settings, loudnessGainDB: gain)
        }
    }

    // MARK: Export

    func export() {
        guard let m = selected, let processedURL else { return }
        let options = m.export
        let destination = store.exportURL(for: m, format: options.format)
        let workspace = Workspace(cacheRoot: cacheRoot, sessionID: m.id)
        let store = store
        let record = ExportRecord(fileName: destination.lastPathComponent, exportedAt: Date(),
                                  modelRevisions: lastRevisions, backend: lastBackend,
                                  processingSeconds: lastProcessingSeconds)
        exportMessage = "正在导出…"
        Task {
            do {
                try await Task.detached {
                    let mastered = workspace.directory.appendingPathComponent("mastered.caf")
                    let writer = try MonoFileWriter(url: mastered, sampleRate: AudioCore.sampleRate, format: .cafFloat32)
                    try MasteringRenderer.render(input: processedURL, settings: m.mastering, output: writer)
                    try writer.close()
                    try Exporter.export(from: try MonoFileReader(url: mastered), options: options, to: destination)
                }.value
                let updated = try store.recordExport(format: options.format, record: record, for: m)
                preview.pause()
                try workspace.purge()
                self.processedURL = nil
                processing = .idle
                try? preview.load(original: store.originalURL(for: updated), processed: nil)
                refresh()
                workspaceBytes = Workspace.totalDiskUsage(cacheRoot: cacheRoot)
                exportMessage = "已导出 \(destination.lastPathComponent)，中间文件已清理"
            } catch {
                exportMessage = nil
                errorMessage = "导出失败：\(error.localizedDescription)"
            }
        }
    }

    func revealInFinder() {
        guard let m = selected else { return }
        NSWorkspace.shared.activateFileViewerSelecting([store.directory(for: m.id)])
    }

    func diskUsage(of m: SessionManifest) -> (original: Int64, export: Int64) {
        let original = (try? store.originalURL(for: m).resourceValues(forKeys: [.totalFileAllocatedSizeKey]).totalFileAllocatedSize) ?? 0
        let export = m.lastExport.flatMap {
            try? store.directory(for: m.id).appendingPathComponent($0.fileName)
                .resourceValues(forKeys: [.totalFileAllocatedSizeKey]).totalFileAllocatedSize
        } ?? 0
        return (Int64(original), Int64(export))
    }
}
