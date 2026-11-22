import AppKit
import AudioCore
import Combine
import Export
import Foundation
import LavaMLX
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

enum AppLanguage: String, CaseIterable, Identifiable {
    case system, zhHans = "zh-Hans", en
    var id: String { rawValue }
}

/// Progress of a one-click "process and export" run over several sessions.
struct BatchState: Equatable {
    var index: Int
    var total: Int
    var name: String
    var stage: String
    var fraction: Double
}

/// App-wide state and the operations the views trigger.
@MainActor
final class AppModel: ObservableObject {
    @Published var sessions: [SessionManifest] = []
    /// Sidebar selection; one selected session opens the editor.
    @Published var selection: Set<UUID> = [] {
        didSet {
            let single = selection.count == 1 ? selection.first : nil
            if single != selectedID { selectedID = single }
        }
    }
    @Published private(set) var selectedID: UUID? {
        didSet { if oldValue != selectedID { sessionChanged(from: oldValue) } }
    }
    @Published var modelState: ModelState = .checking
    @Published var processing: ProcessingState = .idle
    @Published var processedURL: URL?
    @Published var exportMessage: String?
    @Published var errorMessage: String?
    @Published var workspaceBytes: Int64 = 0
    @Published var isExporting = false
    @Published var batch: BatchState?
    /// A just-finished recording waiting for the user to name it.
    @Published var pendingRename: SessionManifest?
    @Published var language: AppLanguage {
        didSet {
            UserDefaults.standard.set(language.rawValue, forKey: "appLanguage")
            recorder.interruptedName = tr("录音（中断）", "Recording (interrupted)")
        }
    }
    /// Exports go here when set; otherwise into each session's own folder.
    @Published var exportFolder: URL? {
        didSet { UserDefaults.standard.set(exportFolder?.path, forKey: "exportFolder") }
    }
    @Published private(set) var userPresetNames: [String] = UserPresets.names()
    @Published var sessionsRoot: URL {
        didSet {
            UserDefaults.standard.set(sessionsRoot.path, forKey: "sessionsRoot")
            store = SessionStore(root: sessionsRoot)
            recorder.store = store
            refresh()
        }
    }

    let recorder: Recorder
    let preview = PreviewPlayer()
    let modelStore = ModelStore(root: ModelStore.defaultRoot)
    let cacheRoot = Workspace.defaultCacheRoot
    private(set) var store: SessionStore
    private var enhancers: [String: MambaEnhancer] = [:]
    private var lava: LavaEnhancer?
    private var processingTask: Task<Void, Never>?
    private var batchTask: Task<Void, Never>?
    private var gainTask: Task<Void, Never>?
    private var gainWork: Task<Double, Error>?
    private var interruptionObserver: AnyCancellable?
    private var currentGain: Double = 0
    private var lastProcessingSeconds: TimeInterval = 0
    private var lastRevisions: [String: String] = [:]
    private let backend = "MLX GPU"

    init() {
        let root = UserDefaults.standard.string(forKey: "sessionsRoot").map { URL(fileURLWithPath: $0) }
            ?? SessionStore.defaultRoot
        sessionsRoot = root
        store = SessionStore(root: root)
        recorder = Recorder(store: SessionStore(root: root), cacheRoot: Workspace.defaultCacheRoot)
        language = AppLanguage(rawValue: UserDefaults.standard.string(forKey: "appLanguage") ?? "") ?? .system
        exportFolder = UserDefaults.standard.string(forKey: "exportFolder").map { URL(fileURLWithPath: $0, isDirectory: true) }
        recorder.interruptedName = tr("录音（中断）", "Recording (interrupted)")
    }

    var selected: SessionManifest? {
        sessions.first { $0.id == selectedID }
    }

    func select(_ id: UUID?) {
        selection = id.map { [$0] } ?? []
    }

    // MARK: Language

    var isEnglish: Bool {
        switch language {
        case .en: true
        case .zhHans: false
        case .system: !(Locale.preferredLanguages.first ?? "zh").hasPrefix("zh")
        }
    }

    var locale: Locale { Locale(identifier: isEnglish ? "en" : "zh-Hans") }

    /// Picks the message text for the current language. Static UI text uses
    /// the string catalog; this covers messages assembled at run time.
    func tr(_ zh: String, _ en: String) -> String { isEnglish ? en : zh }

    // MARK: Lifecycle

    func launch() async {
        interruptionObserver = recorder.$interruption.compactMap { $0 }.sink { [weak self] interruption in
            guard let self else { return }
            self.refresh()
            if let session = interruption.session {
                self.applyRememberedMastering(to: session.id)
                self.select(session.id)
            }
            switch interruption.cause {
            case .writeFailed(let detail):
                self.errorMessage = self.tr("录音写入失败（\(detail)），已保存出错前的部分。",
                                            "Recording could not be written (\(detail)). The part before the error was saved.")
            case .deviceChanged:
                self.errorMessage = self.tr("输入设备发生变化，录音已停止并保存。",
                                            "The input device changed. Recording stopped and was saved.")
            }
        }
        try? Workspace.purgeAll(cacheRoot: cacheRoot)
        let store = store
        let cacheRoot = cacheRoot
        let recovered = await Task.detached {
            (try? RecordingRecovery.recover(cacheRoot: cacheRoot, store: store)) ?? []
        }.value
        for session in recovered { applyRememberedMastering(to: session.id) }
        refresh()
        if let first = recovered.first { select(first.id) }
        await checkModel()
    }

    func quit() {
        preview.stop()
        try? Workspace.purgeAll(cacheRoot: cacheRoot)
    }

    func refresh() {
        sessions = (try? store.list()) ?? []
        let ids = Set(sessions.map(\.id))
        let kept = selection.intersection(ids)
        if kept.isEmpty, let first = sessions.first {
            selection = [first.id]
        } else if kept != selection {
            selection = kept
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

    private func lavaEnhancer() async throws -> LavaEnhancer {
        if let lava { return lava }
        let url = try await modelStore.install(ModelCatalog.lavaSR) { _ in }
        let enhancer = try LavaEnhancer(weightsURL: url)
        lava = enhancer
        return enhancer
    }

    // MARK: Sessions

    func importFiles(_ urls: [URL]) {
        var last: UUID?
        for url in urls {
            let access = url.startAccessingSecurityScopedResource()
            defer { if access { url.stopAccessingSecurityScopedResource() } }
            do {
                let m = try store.importFile(url, name: url.deletingPathExtension().lastPathComponent)
                applyRememberedMastering(to: m.id)
                last = m.id
            } catch {
                errorMessage = tr("导入失败：\(error.localizedDescription)", "Import failed: \(error.localizedDescription)")
            }
        }
        refresh()
        if let last { select(last) }
    }

    func startRecording(input: AudioInput?) async {
        guard await Recorder.requestAccess() else {
            errorMessage = tr("没有麦克风权限。请在“系统设置 › 隐私与安全性 › 麦克风”中允许 MicPolish。",
                              "No microphone access. Allow MicPolish in System Settings › Privacy & Security › Microphone.")
            return
        }
        do {
            try recorder.start(input: input)
        } catch {
            errorMessage = tr("无法开始录音：\(error.localizedDescription)", "Could not start recording: \(error.localizedDescription)")
        }
    }

    /// Stops and saves the recording first, then asks for a name.
    func stopRecording() async {
        do {
            let m = try await recorder.stop(name: defaultRecordingName())
            applyRememberedMastering(to: m.id)
            refresh()
            select(m.id)
            pendingRename = selected
        } catch {
            errorMessage = tr("保存录音失败：\(error.localizedDescription)", "Saving the recording failed: \(error.localizedDescription)")
        }
    }

    func defaultRecordingName() -> String {
        let f = DateFormatter()
        f.locale = locale
        f.dateFormat = isEnglish ? "MMM d HH.mm 'Recording'" : "M月d日 HH.mm 录音"
        return f.string(from: Date())
    }

    func rename(_ id: UUID, to name: String) {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        do {
            try store.rename(id, to: trimmed)
            refresh()
        } catch {
            errorMessage = tr("重命名失败：\(error.localizedDescription)", "Rename failed: \(error.localizedDescription)")
        }
    }

    func delete(_ ids: Set<UUID>) {
        if let selectedID, ids.contains(selectedID) { preview.stop() }
        for id in ids {
            try? Workspace(cacheRoot: cacheRoot, sessionID: id).purge()
            try? store.delete(id)
        }
        selection.subtract(ids)
        refresh()
    }

    func updateSelected(_ change: (inout SessionManifest) -> Void) {
        guard var m = selected else { return }
        change(&m)
        try? store.save(m)
        if let i = sessions.firstIndex(where: { $0.id == m.id }) { sessions[i] = m }
    }

    /// Preview scratch files live in the session's workspace, so export,
    /// quit and launch clean them up with the other intermediates.
    private func previewScratch(for id: UUID) -> URL {
        Workspace(cacheRoot: cacheRoot, sessionID: id).directory.appendingPathComponent("preview", isDirectory: true)
    }

    private func sessionChanged(from previous: UUID?) {
        processingTask?.cancel()
        gainTask?.cancel()
        gainWork?.cancel()
        currentGain = 0
        preview.pause()
        processing = .idle
        processedURL = nil
        exportMessage = nil
        if let previous { try? FileManager.default.removeItem(at: previewScratch(for: previous)) }
        guard let m = selected else { return }
        preview.setMastering(m.mastering, loudnessGainDB: 0)
        reloadPreview(m, processed: nil)
        workspaceBytes = Workspace.totalDiskUsage(cacheRoot: cacheRoot)
        Task { await checkModel() }
    }

    private func reloadPreview(_ m: SessionManifest, processed: URL?, show source: PreviewSource? = nil) {
        let original = store.originalURL(for: m)
        let scratch = previewScratch(for: m.id)
        Task {
            do {
                try await preview.load(original: original, processed: processed, scratchDirectory: scratch)
                if let source, selectedID == m.id { preview.source = source }
                workspaceBytes = Workspace.totalDiskUsage(cacheRoot: cacheRoot)
            } catch {
                if selectedID == m.id {
                    errorMessage = tr("无法载入试听：\(error.localizedDescription)", "Could not load the preview: \(error.localizedDescription)")
                }
            }
        }
    }

    // MARK: Processing

    /// Expected AI time for the selected session, from its last run.
    var estimatedProcessingSeconds: TimeInterval? {
        selected?.lastExport?.processingSeconds
    }

    var isProcessing: Bool {
        if case .running = processing { return true }
        return false
    }

    var isBusy: Bool { isExporting || isProcessing || batch != nil }

    private func stageLabel(_ id: String) -> String {
        if id.hasPrefix("reuse") { return tr("AI 修复（RE-USE）", "AI restoration (RE-USE)") }
        if id.hasPrefix("lavasr") { return tr("高频增强（LavaSR）", "High-band enhancement (LavaSR)") }
        return tr("转换采样率", "Converting sample rate")
    }

    /// Runs the AI stages for one session, reusing that session's cache.
    private func runPipeline(
        _ m: SessionManifest,
        progress: @escaping @MainActor (String, Double, TimeInterval?) -> Void
    ) async throws -> (url: URL, revisions: [String: String], seconds: TimeInterval) {
        let (enhancer, descriptor) = try await enhancer(for: m.processing.reuseVariant)
        var stages: [any AudioStage] = [
            ReuseStage(enhancer: enhancer, variantID: descriptor.id, revision: descriptor.revision),
            ResampleStage(),
        ]
        var revisions = [descriptor.id: descriptor.revision]
        if m.processing.highBandEnabled {
            progress(tr("准备高频增强模型", "Preparing the high-band model"), 0, nil)
            let lava = try await lavaEnhancer()
            // Band-limited imports: generate everything above their Nyquist.
            let cutoff = m.processing.highBandCutoff.hertz
                ?? (m.original.sampleRate < AudioCore.sampleRate ? Double(m.original.sampleRate) / 2 : nil)
            stages.append(LavaStage(enhancer: lava, revision: ModelCatalog.lavaSR.revision, cutoffHz: cutoff))
            revisions[ModelCatalog.lavaSR.id] = ModelCatalog.lavaSR.revision
        }
        let pipeline = ProcessingPipeline(stages: stages, workspace: Workspace(cacheRoot: cacheRoot, sessionID: m.id))
        let started = Date()
        let result = try await pipeline.run(original: store.originalURL(for: m)) { p in
            let elapsed = Double(p.elapsed.components.seconds)
            let overall = (Double(p.stageIndex) + p.fraction) / Double(p.stageCount)
            let remaining = overall > 0.02 ? elapsed / overall - elapsed : nil
            let id = p.stageID
            Task { @MainActor in progress(self.stageLabel(id), overall, remaining) }
        }
        return (result.url, revisions, Date().timeIntervalSince(started))
    }

    func process() {
        guard let m = selected, !isBusy else { return }
        processing = .running(stage: tr("准备模型", "Preparing model"), fraction: 0, remaining: nil)
        processingTask = Task {
            do {
                let result = try await runPipeline(m) { [weak self] stage, fraction, remaining in
                    guard let self, self.selectedID == m.id, self.isProcessing else { return }
                    self.processing = .running(stage: stage, fraction: fraction, remaining: remaining)
                }
                try Task.checkCancellation()
                // The result stays cached in that session's workspace; only
                // touch the UI if the user is still on that session.
                guard selectedID == m.id else { return }
                processedURL = result.url
                lastProcessingSeconds = result.seconds
                lastRevisions = result.revisions
                reloadPreview(m, processed: result.url, show: .processed)
                processing = .done
                refreshLoudnessGain()
            } catch is CancellationError {
                if selectedID == m.id { processing = .idle }
            } catch {
                if selectedID == m.id { processing = .failed(error.localizedDescription) }
            }
        }
    }

    func cancelProcessing() {
        processingTask?.cancel()
        processing = .idle
    }

    // MARK: Mastering

    /// The preset name matching `settings`, or nil for custom settings.
    func presetName(for settings: MasteringSettings) -> String? {
        if settings == .default { return tr("默认", "Default") }
        if settings == .macbookMicCorrection { return tr("MacBook 麦克风校正", "MacBook Mic Correction") }
        return userPresetNames.first { UserPresets.load($0) == settings }
    }

    func savePreset(_ name: String) {
        guard let settings = selected?.mastering else { return }
        UserPresets.save(settings, as: name)
        userPresetNames = UserPresets.names()
    }

    func deletePreset(_ name: String) {
        UserPresets.delete(name)
        userPresetNames = UserPresets.names()
    }

    /// The mastering the user chose last; new sessions start with it.
    var rememberedMastering: MasteringSettings {
        get {
            UserDefaults.standard.data(forKey: "lastMastering")
                .flatMap { try? JSONDecoder().decode(MasteringSettings.self, from: $0) } ?? .default
        }
        set { UserDefaults.standard.set(try? JSONEncoder().encode(newValue), forKey: "lastMastering") }
    }

    private func applyRememberedMastering(to id: UUID) {
        guard var m = ((try? store.list()) ?? []).first(where: { $0.id == id }) else { return }
        m.mastering = rememberedMastering
        try? store.save(m)
    }

    func setMastering(_ settings: MasteringSettings) {
        updateSelected { $0.mastering = settings }
        rememberedMastering = settings
        preview.setMastering(settings, loudnessGainDB: currentGain)
        refreshLoudnessGain()
    }

    /// Re-measures the loudness gain for the current settings in the
    /// background (debounced) and applies it to the live preview.
    private func refreshLoudnessGain() {
        guard let processedURL, let settings = selected?.mastering, let id = selectedID else { return }
        gainTask?.cancel()
        gainWork?.cancel()
        gainTask = Task {
            try? await Task.sleep(nanoseconds: 300_000_000)
            guard !Task.isCancelled else { return }
            let work = Task.detached(priority: .utility) {
                try MasteringRenderer.measureGain(input: processedURL, settings: settings).gainDB
            }
            gainWork = work
            guard let gain = try? await work.value, !Task.isCancelled, selectedID == id else { return }
            currentGain = gain
            preview.setMastering(settings, loudnessGainDB: gain)
        }
    }

    // MARK: Export

    /// Where exports go: the chosen export folder, or the session folder.
    func exportDestination(for m: SessionManifest) -> URL {
        exportFolder ?? store.directory(for: m.id)
    }

    func chooseExportFolder() -> Bool {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.prompt = tr("选择", "Choose")
        panel.message = tr("选择导出文件夹", "Choose the export folder")
        guard panel.runModal() == .OK, let url = panel.url else { return false }
        exportFolder = url
        return true
    }

    /// Masters `processed` and encodes it into the export destination.
    /// Returns the updated manifest; purges the session's workspace.
    private func renderAndExport(_ m: SessionManifest, processed: URL, seconds: TimeInterval,
                                 revisions: [String: String]) async throws -> (SessionManifest, URL) {
        let options = m.export
        let folder = exportDestination(for: m)
        let external = exportFolder != nil
        let destination = external
            ? uniqueExternalURL(for: m, format: options.format, in: folder)
            : store.exportURL(for: m, format: options.format)
        let workspace = Workspace(cacheRoot: cacheRoot, sessionID: m.id)
        let mastering = m.mastering
        try await Task.detached {
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            try workspace.create()
            let mastered = workspace.directory.appendingPathComponent("mastered-\(UUID().uuidString).caf")
            defer { try? FileManager.default.removeItem(at: mastered) }
            let writer = try MonoFileWriter(url: mastered, sampleRate: AudioCore.sampleRate, format: .cafFloat32)
            try MasteringRenderer.render(input: processed, settings: mastering, output: writer)
            try writer.close()
            try Exporter.export(from: try MonoFileReader(url: mastered), options: options, to: destination)
        }.value
        let record = ExportRecord(fileName: destination.lastPathComponent, exportedAt: Date(), modelRevisions: revisions,
                                  backend: backend, processingSeconds: seconds,
                                  directory: external ? folder.path : nil)
        let updated = try store.recordExport(format: options.format, record: record, for: m)
        try workspace.purge()
        return (updated, destination)
    }

    /// `<folder>/<name>.<ext>`, numbered when another session's file is there.
    private func uniqueExternalURL(for m: SessionManifest, format: ExportFormat, in folder: URL) -> URL {
        let base = m.name.replacingOccurrences(of: "/", with: "-")
        var candidate = folder.appendingPathComponent("\(base).\(format.rawValue)")
        let previous = store.lastExportURL(for: m)
        var n = 2
        while FileManager.default.fileExists(atPath: candidate.path), candidate != previous {
            candidate = folder.appendingPathComponent("\(base) (\(n)).\(format.rawValue)")
            n += 1
        }
        return candidate
    }

    func export() {
        guard let m = selected, let processedURL, !isBusy else { return }
        isExporting = true
        exportMessage = tr("正在导出…", "Exporting…")
        let seconds = lastProcessingSeconds
        let revisions = lastRevisions
        Task {
            do {
                if selectedID == m.id { preview.pause() }
                let (updated, destination) = try await renderAndExport(m, processed: processedURL, seconds: seconds, revisions: revisions)
                isExporting = false
                refresh()
                workspaceBytes = Workspace.totalDiskUsage(cacheRoot: cacheRoot)
                guard selectedID == m.id else { return }
                self.processedURL = nil
                processing = .idle
                reloadPreview(updated, processed: nil)
                exportMessage = tr("已导出到 \(destination.path)，中间文件已清理",
                                   "Exported to \(destination.path); intermediate files removed")
            } catch {
                isExporting = false
                if selectedID == m.id { exportMessage = nil }
                errorMessage = tr("导出失败：\(error.localizedDescription)", "Export failed: \(error.localizedDescription)")
            }
        }
    }

    /// Toolbar quick export: exports the current session if it is processed,
    /// otherwise processes and exports it in one go.
    func quickExport() {
        guard let m = selected, !isBusy else { return }
        if exportFolder == nil, !chooseExportFolder() { return }
        if processedURL != nil { export() } else { processAndExport([m.id]) }
    }

    /// One click: AI processing, mastering and export for each session.
    func processAndExport(_ ids: Set<UUID>) {
        let targets = sessions.filter { ids.contains($0.id) }
        guard !targets.isEmpty, !isBusy else { return }
        if exportFolder == nil, !chooseExportFolder() { return }
        preview.pause()
        batchTask = Task {
            var exported: [URL] = []
            var failures: [String] = []
            for (index, m) in targets.enumerated() {
                if Task.isCancelled { break }
                batch = BatchState(index: index + 1, total: targets.count, name: m.name, stage: tr("准备模型", "Preparing model"), fraction: 0)
                do {
                    let result = try await runPipeline(m) { [weak self] stage, fraction, _ in
                        self?.batch?.stage = stage
                        self?.batch?.fraction = fraction * 0.85
                    }
                    batch?.stage = tr("母带处理并导出", "Mastering and exporting")
                    batch?.fraction = 0.9
                    let (_, url) = try await renderAndExport(m, processed: result.url, seconds: result.seconds,
                                                             revisions: result.revisions)
                    exported.append(url)
                } catch is CancellationError {
                    break
                } catch {
                    failures.append("\(m.name)：\(error.localizedDescription)")
                }
            }
            batch = nil
            refresh()
            if let current = selected {
                processedURL = nil
                processing = .idle
                reloadPreview(current, processed: nil)
            }
            workspaceBytes = Workspace.totalDiskUsage(cacheRoot: cacheRoot)
            exportMessage = tr("已处理并导出 \(exported.count) 个文件到 \(exportFolder?.path ?? "")",
                               "Processed and exported \(exported.count) file(s) to \(exportFolder?.path ?? "")")
            if !failures.isEmpty {
                errorMessage = tr("以下录音未能导出：\n", "These recordings could not be exported:\n") + failures.joined(separator: "\n")
            }
            if !exported.isEmpty {
                NSWorkspace.shared.activateFileViewerSelecting(exported)
            }
        }
    }

    func cancelBatch() {
        batchTask?.cancel()
    }

    func revealInFinder() {
        guard let m = selected else { return }
        if let url = store.lastExportURL(for: m), FileManager.default.fileExists(atPath: url.path) {
            NSWorkspace.shared.activateFileViewerSelecting([url])
        } else {
            NSWorkspace.shared.activateFileViewerSelecting([store.directory(for: m.id)])
        }
    }

    func diskUsage(of m: SessionManifest) -> (original: Int64, export: Int64) {
        func size(_ url: URL?) -> Int64 {
            Int64((url.flatMap { try? $0.resourceValues(forKeys: [.totalFileAllocatedSizeKey]).totalFileAllocatedSize }) ?? 0)
        }
        return (size(store.originalURL(for: m)), size(store.lastExportURL(for: m)))
    }
}
