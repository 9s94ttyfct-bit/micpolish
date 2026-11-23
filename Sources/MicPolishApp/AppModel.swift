import AppKit
import AudioCore
import Combine
import Export
import Foundation
import InworldMLX
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
    /// The AI settings `processedURL` was produced with.
    @Published private(set) var processedSettings: ProcessingSettings?
    /// `processedURL`, only if it was produced with the selected session's
    /// current AI settings; exporting anything else would not match them.
    var currentProcessedURL: URL? {
        guard let processedURL, let m = selected, processedSettings == m.processing else { return nil }
        return processedURL
    }
    @Published var exportMessage: String?
    @Published var errorMessage: String?
    @Published var workspaceBytes: Int64 = 0
    @Published var isExporting = false
    @Published var batch: BatchState?
    /// A just-finished recording waiting for the user to name it.
    @Published var pendingRename: SessionManifest?
    /// Input to record from; nil is the system default input.
    @Published var recordingInput: AudioInput?
    /// Whether stopping a recording asks for a name (otherwise the default name stays).
    @Published var askNameAfterRecording = UserDefaults.standard.object(forKey: "askNameAfterRecording") as? Bool ?? true {
        didSet { UserDefaults.standard.set(askNameAfterRecording, forKey: "askNameAfterRecording") }
    }
    @Published var confirmingDiscard = false
    /// Asks before deleting `deleteTargets`.
    @Published var confirmingDelete = false
    @Published private(set) var deleteTargets: Set<UUID> = []
    /// Sessions a running process, export or batch uses (see `isInUse`).
    @Published private(set) var sessionsInUse: Set<UUID> = []
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
    let cacheRoot = AppModel.defaultCacheRoot
    /// Held for the app's lifetime; nil (lock file unusable) skips cleanup
    /// rather than risk deleting another process's work.
    private let cacheLock = try? CacheLock(cacheRoot: AppModel.defaultCacheRoot)
    /// `MICPOLISH_DATA_ROOT` runs the app on its own sessions and cache folders
    /// (for testing next to a running copy, which would otherwise share them).
    static let dataRootOverride = ProcessInfo.processInfo.environment["MICPOLISH_DATA_ROOT"]
        .map { URL(fileURLWithPath: $0, isDirectory: true) }
    static var defaultCacheRoot: URL {
        dataRootOverride?.appendingPathComponent("cache", isDirectory: true) ?? Workspace.defaultCacheRoot
    }
    private(set) var store: SessionStore
    private var enhancers: [String: MambaEnhancer] = [:]
    private var lava: LavaEnhancer?
    private var inworld: InworldIsolator?
    private var launched = false
    private var processingTask: Task<Void, Never>?
    /// Increments per `process()`, so a superseded run never touches the UI.
    private var processingRun = 0
    private var batchTask: Task<Void, Never>?
    private var gainTask: Task<Void, Never>?
    private var gainWork: Task<(Double, StudioTone.Analysis?), Error>?
    /// Studio-tone analysis of the current processed audio (keyed by file and trim).
    private var studioToneCache: (key: String, analysis: StudioTone.Analysis)?
    /// Shown under the studio-tone option: detected voice and per-band correction.
    @Published private(set) var studioToneAnalysis: StudioTone.Analysis?
    private var interruptionObserver: AnyCancellable?
    private var currentGain: Double = 0
    private var lastProcessingSeconds: TimeInterval = 0
    private var lastRevisions: [String: String] = [:]
    /// Backend of the RE-USE stage in the last run, for display and export records.
    @Published private(set) var lastBackendLabel = "MLX GPU"
    @Published var backendPreference: BackendPreference =
        BackendPreference(rawValue: UserDefaults.standard.string(forKey: "reuseBackend") ?? "") ?? .auto {
        didSet { UserDefaults.standard.set(backendPreference.rawValue, forKey: "reuseBackend") }
    }
    /// This Mac's last speed test of both backends (set by `runSpeedTest`).
    @Published var speedTest: SpeedTestResult? = SpeedTestResult.load()
    /// Progress of a running speed test (0...1), nil when none runs.
    @Published var speedTestProgress: Double?

    init() {
        let root = Self.dataRootOverride?.appendingPathComponent("sessions", isDirectory: true)
            ?? UserDefaults.standard.string(forKey: "sessionsRoot").map { URL(fileURLWithPath: $0) }
            ?? SessionStore.defaultRoot
        sessionsRoot = root
        store = SessionStore(root: root)
        recorder = Recorder(store: SessionStore(root: root), cacheRoot: Self.defaultCacheRoot)
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

    /// Runs once per process. The main window's `.task` calls it again when
    /// the window is reopened, which must not purge this app's own working
    /// files or recover a recording that is still running.
    func launch() async {
        guard !launched else { return }
        launched = true
        interruptionObserver = recorder.$interruption.compactMap { $0 }.sink { [weak self] interruption in
            guard let self else { return }
            self.refresh()
            if let session = interruption.session {
                self.applyRememberedMastering(to: session.id)
                self.selectUnlessProcessing(session.id)
            }
            let cause: (zh: String, en: String)
            switch interruption.cause {
            case .writeFailed(let detail):
                cause = ("录音写入失败（\(detail)）。", "Recording could not be written (\(detail)).")
            case .deviceLost:
                cause = ("正在录音的输入设备已断开或无法继续使用，录音已停止。",
                         "The input device being recorded disconnected or could no longer be used. Recording stopped.")
            case .inputFailed(let status):
                cause = ("正在录音的输入设备持续无法送出声音（错误 \(status)），录音已停止。",
                         "The input device being recorded kept failing to deliver audio (error \(status)). Recording stopped.")
            case .overrun:
                cause = ("磁盘写入跟不上录音，有一段声音丢失，录音已停止。",
                         "The disk could not keep up with the recording and some audio was lost. Recording stopped.")
            }
            let outcome: (zh: String, en: String)
            switch interruption.outcome {
            case .saved:
                outcome = ("在此之前录到的部分已保存。", "What was recorded up to then was saved.")
            case .nothingCaptured:
                outcome = ("还没有录到内容，没有保存。", "Nothing had been captured yet, so nothing was saved.")
            case .saveFailed(let detail):
                outcome = ("保存失败（\(detail)），但录音已保留，下次启动 MicPolish 时会自动恢复。",
                           "Saving failed (\(detail)), but the audio was kept and will be recovered the next time MicPolish starts.")
            }
            self.errorMessage = self.tr(cause.zh + outcome.zh, cause.en + " " + outcome.en)
        }
        let store = store
        let recoveredNamePrefix = tr("恢复的录音", "Recovered recording")
        let cacheRoot = cacheRoot
        let cacheLock = cacheLock
        // Only when no other MicPolish process (a second app copy, the CLI)
        // shares the cache: their workspaces and live recordings stay put.
        let recovered = await Task.detached {
            var recovered: [SessionManifest] = []
            cacheLock?.whileAlone {
                try? Workspace.purgeAll(cacheRoot: cacheRoot)
                recovered = (try? RecordingRecovery.recover(cacheRoot: cacheRoot, store: store,
                                                            namePrefix: recoveredNamePrefix)) ?? []
            }
            return recovered
        }.value
        for session in recovered { applyRememberedMastering(to: session.id) }
        refresh()
        if let first = recovered.first { select(first.id) }
        await checkModel()
    }

    func quit() {
        preview.stop()
        cacheLock?.whileAlone { try? Workspace.purgeAll(cacheRoot: cacheRoot) }
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

    /// The RE-USE model the selected session uses (the standard one without a selection).
    var selectedReuseDescriptor: ModelDescriptor {
        selected.flatMap { ModelCatalog.descriptor(id: $0.processing.reuseVariant) } ?? ModelCatalog.reuse
    }

    func checkModel() async {
        modelState = await modelStore.installedWeightsURL(selectedReuseDescriptor) == nil ? .notInstalled : .ready
    }

    /// Downloads a model (the main window's banner passes the selected
    /// session's RE-USE model). Only that model drives `modelState`; other
    /// models, e.g. from Settings, just install.
    func downloadModel(_ descriptor: ModelDescriptor) async {
        let tracksMainModel = descriptor.id == selectedReuseDescriptor.id
        if tracksMainModel { modelState = .downloading(0) }
        do {
            _ = try await modelStore.install(descriptor) { fraction in
                guard tracksMainModel else { return }
                Task { @MainActor in self.modelState = .downloading(fraction) }
            }
        } catch {
            if tracksMainModel { modelState = .failed(error.localizedDescription) } else {
                errorMessage = tr("下载模型失败：\(error.localizedDescription)", "Model download failed: \(error.localizedDescription)")
            }
            return
        }
        await checkModel()
    }

    private func enhancer(for variant: String) async throws -> (MambaEnhancer, ModelDescriptor, ReuseBackend) {
        let descriptor = ModelCatalog.descriptor(id: variant) ?? ModelCatalog.reuse
        let wanted = resolvedBackend(for: descriptor)
        let key = descriptor.id + (wanted == .aneHybrid ? "|ane" : "")
        if let cached = enhancers[key] { return (cached, descriptor, cached.accelerator == nil ? .mlx : wanted) }
        let url = try await modelStore.install(descriptor) { _ in }
        // A package that fails to load (e.g. a newer macOS rejects it) quietly falls back to MLX.
        let accelerator = wanted == .aneHybrid ? try? await accelerator(for: descriptor) : nil
        let enhancer = try MambaEnhancer(modelWeightsURL: url, accelerator: accelerator)
        enhancers[key] = enhancer
        return (enhancer, descriptor, accelerator == nil ? .mlx : wanted)
    }

    private func lavaEnhancer() async throws -> LavaEnhancer {
        if let lava { return lava }
        let url = try await modelStore.install(ModelCatalog.lavaSR) { _ in }
        let enhancer = try LavaEnhancer(weightsURL: url)
        lava = enhancer
        return enhancer
    }

    private func inworldIsolator(progress: @escaping @MainActor (Double) -> Void) async throws -> InworldIsolator {
        if let inworld { return inworld }
        let url = try await modelStore.install(ModelCatalog.inworld) { fraction in
            Task { @MainActor in progress(fraction) }
        }
        let isolator = try InworldIsolator(weightsURL: url)
        inworld = isolator
        return isolator
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
        if let last { selectUnlessProcessing(last) }
    }

    /// Starts recording from `recordingInput`, or stops and saves the running one.
    /// The choice is made now, from what the user saw: if the recording ended on
    /// its own before the task runs, a "stop" must not start a new recording.
    func toggleRecording() {
        let stop = recorder.isRecording
        Task {
            if stop { await stopRecording() } else { await startRecording(input: recordingInput) }
        }
    }

    func startRecording(input: AudioInput?) async {
        guard await Recorder.requestAccess() else {
            errorMessage = tr("没有麦克风权限。请在“系统设置 › 隐私与安全性 › 麦克风”中允许 MicPolish。",
                              "No microphone access. Allow MicPolish in System Settings › Privacy & Security › Microphone.")
            return
        }
        preview.pause()
        do {
            try recorder.start(input: input)
        } catch {
            errorMessage = tr("无法开始录音：\(error.localizedDescription)", "Could not start recording: \(error.localizedDescription)")
        }
    }

    /// Stops and saves the recording under the default name, then asks for a
    /// name unless the user turned that off.
    func stopRecording() async {
        do {
            let m = try await recorder.stop(name: defaultRecordingName())
            applyRememberedMastering(to: m.id)
            refresh()
            selectUnlessProcessing(m.id)
            if askNameAfterRecording { pendingRename = sessions.first { $0.id == m.id } }
        } catch RecorderError.notRecording {
            // The recording already ended on its own (device lost, write error);
            // that interruption is reported separately.
        } catch RecorderError.nothingCaptured {
            errorMessage = tr("没有录到声音，没有保存。", "Nothing was captured, so nothing was saved.")
        } catch {
            // Stop keeps the capture file whenever saving fails after audio was written.
            errorMessage = tr("保存录音失败（\(error.localizedDescription)），但录音已保留，下次启动 MicPolish 时会自动恢复。",
                              "Saving the recording failed (\(error.localizedDescription)), but the audio was kept and will be recovered the next time MicPolish starts.")
        }
    }

    /// Asks before throwing the running recording away.
    func requestDiscard() {
        if recorder.isRecording { confirmingDiscard = true }
    }

    /// Stops the running recording and deletes it; nothing is saved.
    func discardRecording() {
        confirmingDiscard = false
        recorder.discard()
    }

    func defaultRecordingName() -> String {
        let f = DateFormatter()
        f.locale = locale
        f.dateFormat = isEnglish ? "MMM d HH.mm 'Recording'" : "M月d日 HH.mm 录音"
        return f.string(from: Date())
    }

    /// Sessions a running process, export or batch reads from or writes into.
    func isInUse(_ id: UUID) -> Bool { sessionsInUse.contains(id) }

    /// Rename and delete wait for the jobs that use those sessions: an export
    /// or batch item in flight writes into the session's folder. Other
    /// sessions (e.g. a recording just made during a batch) stay editable.
    private func refuseWhileInUse(_ ids: Set<UUID>) -> Bool {
        guard !sessionsInUse.isDisjoint(with: ids) else { return false }
        errorMessage = tr("请等这条录音的处理或导出完成后再重命名或删除。",
                          "Wait for this recording's processing or export to finish before renaming or deleting it.")
        return true
    }

    func rename(_ id: UUID, to name: String) {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !refuseWhileInUse([id]) else { return }
        do {
            try store.rename(id, to: trimmed)
            refresh()
        } catch {
            errorMessage = tr("重命名失败：\(error.localizedDescription)", "Rename failed: \(error.localizedDescription)")
        }
    }

    /// Asks to delete `ids` (without changing the selection, which would
    /// cancel a running process).
    func requestDelete(_ ids: Set<UUID>) {
        guard !ids.isEmpty else { return }
        deleteTargets = ids
        confirmingDelete = true
    }

    func delete(_ ids: Set<UUID>) {
        guard !refuseWhileInUse(ids) else { return }
        if let selectedID, ids.contains(selectedID) { preview.stop() }
        for id in ids {
            try? Workspace(cacheRoot: cacheRoot, sessionID: id).purge()
            try? store.delete(id)
        }
        selection.subtract(ids)
        refresh()
    }

    /// Selects a new session (an import, a recording) unless that would
    /// cancel the running process; the session is then just added to the list.
    private func selectUnlessProcessing(_ id: UUID) {
        if !isProcessing { select(id) }
    }

    func updateSelected(_ change: (inout SessionManifest) -> Void) {
        guard var m = selected else { return }
        let variant = m.processing.reuseVariant
        change(&m)
        try? store.save(m)
        if let i = sessions.firstIndex(where: { $0.id == m.id }) { sessions[i] = m }
        if m.processing.reuseVariant != variant { Task { await checkModel() } }
    }

    /// Preview scratch files live in the session's workspace, so export,
    /// quit and launch clean them up with the other intermediates.
    private func previewScratch(for id: UUID) -> URL {
        Workspace(cacheRoot: cacheRoot, sessionID: id).directory.appendingPathComponent("preview", isDirectory: true)
    }

    private func sessionChanged(from previous: UUID?) {
        // A running process keeps the app busy ("Cancelling…") until it has
        // stopped; anything else just resets.
        if isProcessing { cancelProcessing() } else { processing = .idle }
        gainTask?.cancel()
        gainWork?.cancel()
        currentGain = 0
        preview.clear()
        processedURL = nil
        studioToneCache = nil
        studioToneAnalysis = nil
        exportMessage = nil
        if let previous { try? FileManager.default.removeItem(at: previewScratch(for: previous)) }
        guard let m = selected else { return }
        preview.setMastering(m.mastering, loudnessGainDB: 0, studioTone: [])
        preview.setRange(start: m.trim?.start, end: m.trim?.end)
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

    var isBusy: Bool { isExporting || isProcessing || batch != nil || speedTestProgress != nil }

    private func stageLabel(_ id: String) -> String {
        if id.hasPrefix("inworld") { return tr("强降噪（Inworld）", "Strong noise removal (Inworld)") }
        if id.hasPrefix("reuse") { return tr("AI 修复（RE-USE）", "AI restoration (RE-USE)") }
        if id.hasPrefix("lavasr") { return tr("高频增强（LavaSR）", "High-band enhancement (LavaSR)") }
        return tr("转换采样率", "Converting sample rate")
    }

    /// Runs the AI stages for one session, reusing that session's cache.
    private func runPipeline(
        _ m: SessionManifest,
        progress: @escaping @MainActor (String, Double, TimeInterval?) -> Void
    ) async throws -> (url: URL, revisions: [String: String], seconds: TimeInterval) {
        let (enhancer, descriptor, backend) = try await enhancer(for: m.processing.reuseVariant)
        lastBackendLabel = Self.label(for: backend)
        var stages: [any AudioStage] = [
            ReuseStage(enhancer: enhancer, variantID: descriptor.id, revision: descriptor.revision, backend: backend) { usage in
                Task { @MainActor in self.lastBackendLabel = Self.label(for: usage.backend) }
            },
            ResampleStage(),
        ]
        var revisions = [descriptor.id: descriptor.revision]
        if m.processing.strongDenoise {
            let label = tr("下载强降噪模型（约 256 MB）", "Downloading the noise-removal model (about 256 MB)")
            progress(tr("准备强降噪模型", "Preparing the noise-removal model"), 0, nil)
            let isolator = try await inworldIsolator { fraction in progress(label, fraction, nil) }
            // Before RE-USE: remove heavy noise first, then restore.
            stages.insert(InworldStage(isolator: isolator, revision: ModelCatalog.inworld.revision), at: 0)
            revisions[ModelCatalog.inworld.id] = ModelCatalog.inworld.revision
        }
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
        processingRun += 1
        let run = processingRun
        sessionsInUse.insert(m.id)
        processing = .running(stage: tr("准备模型", "Preparing model"), fraction: 0, remaining: nil)
        processingTask = Task {
            // Only the latest run, while its session is still shown, drives the UI.
            @MainActor func owns(_ model: AppModel) -> Bool { model.processingRun == run && model.selectedID == m.id }
            // The app stays busy until the run has really ended, even after a
            // cancel or a session switch, so runs never overlap.
            @MainActor func settle(_ model: AppModel, _ state: ProcessingState) {
                model.sessionsInUse.remove(m.id)
                guard model.processingRun == run else { return }
                model.processing = model.selectedID == m.id ? state : .idle
            }
            do {
                let result = try await runPipeline(m) { [weak self] stage, fraction, remaining in
                    guard let self, owns(self), self.isProcessing, self.processingTask?.isCancelled == false else { return }
                    self.processing = .running(stage: stage, fraction: fraction, remaining: remaining)
                }
                try Task.checkCancellation()
                // The result stays cached in that session's workspace; only
                // touch the UI if the user is still on that session.
                guard owns(self) else { return settle(self, .idle) }
                processedURL = result.url
                processedSettings = m.processing
                lastProcessingSeconds = result.seconds
                lastRevisions = result.revisions
                reloadPreview(m, processed: result.url, show: .processed)
                settle(self, .done)
                refreshLoudnessGain()
                await checkModel()
            } catch where Task.isCancelled || error is CancellationError {
                settle(self, .idle)
            } catch {
                settle(self, .failed(error.localizedDescription))
            }
        }
    }

    /// Stays busy ("Cancelling…") until the cancelled run has actually
    /// stopped, so a new run cannot overlap it in the same workspace.
    func cancelProcessing() {
        guard isProcessing, processingTask?.isCancelled == false else { return }
        processingTask?.cancel()
        processing = .running(stage: tr("正在取消…", "Cancelling…"), fraction: 0, remaining: nil)
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
        guard let processedURL, let m = selected, let id = selectedID else { return }
        let settings = m.mastering
        let workspace = Workspace(cacheRoot: cacheRoot, sessionID: id)
        // Studio tone is measured once per processed audio and trim.
        let key = "\(processedURL.path)|\(m.trim.map { "\($0.start)-\($0.end)" } ?? "")"
        let cached = studioToneCache?.key == key ? studioToneCache?.analysis : nil
        gainTask?.cancel()
        gainWork?.cancel()
        gainTask = Task {
            try? await Task.sleep(nanoseconds: 300_000_000)
            guard !Task.isCancelled else { return }
            let work = Task.detached(priority: .utility) {
                try Self.withTrimmed(processedURL, trim: m.trim, in: workspace) { url -> (Double, StudioTone.Analysis?) in
                    let analysis = settings.studioTone.enabled ? try cached ?? StudioTone.analyze(url: url) : cached
                    let gain = try MasteringRenderer.measureGain(
                        input: url, settings: settings,
                        studioTone: settings.studioTone.enabled ? analysis?.correction ?? [] : []).gainDB
                    return (gain, analysis)
                }
            }
            gainWork = work
            guard let (gain, analysis) = try? await work.value, !Task.isCancelled, selectedID == id else { return }
            currentGain = gain
            if let analysis {
                studioToneCache = (key, analysis)
                studioToneAnalysis = analysis
            }
            preview.setMastering(settings, loudnessGainDB: gain, studioTone: analysis?.correction)
        }
    }

    // MARK: Trim

    /// Sets the kept span; a span covering the whole recording clears the trim.
    func setTrim(_ trim: TrimRange?) {
        guard let m = selected else { return }
        let duration = m.original.durationSeconds
        var kept = trim.map { TrimRange(start: max(0, $0.start), end: min(duration, $0.end)) }
        if let k = kept, k.start <= 0.001, k.end >= duration - 0.001 { kept = nil }
        if let k = kept, k.duration < 0.1 { return }
        updateSelected { $0.trim = kept }
        preview.setRange(start: kept?.start, end: kept?.end)
        refreshLoudnessGain()
    }

    /// Trims leading and trailing silence, judged on the processed audio when
    /// there is one (its noise floor is lower), else the original.
    func autoTrim() {
        guard let envelope = preview.envelopes[.processed] ?? preview.envelopes[.original],
              let range = AudioTrim.detectSoundRange(envelope: envelope, bucketSeconds: 0.05) else { return }
        setTrim(TrimRange(start: range.start, end: range.end))
    }

    /// Runs `body` on `processed` cut to `trim` (a temporary file in the
    /// session workspace), or on `processed` itself when there is no trim.
    nonisolated static func withTrimmed<T>(_ processed: URL, trim: TrimRange?, in workspace: Workspace,
                                           _ body: (URL) throws -> T) throws -> T {
        guard let trim else { return try body(processed) }
        try workspace.create()
        let cut = workspace.directory.appendingPathComponent("trimmed-\(UUID().uuidString).caf")
        defer { try? FileManager.default.removeItem(at: cut) }
        try AudioTrim.write(processed, from: trim.start, to: trim.end, output: cut)
        return try body(cut)
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
            try Self.withTrimmed(processed, trim: m.trim, in: workspace) {
                try MasteringRenderer.render(input: $0, settings: mastering, output: writer)
            }
            try writer.close()
            try Exporter.export(from: try MonoFileReader(url: mastered), options: options, to: destination)
        }.value
        let record = ExportRecord(fileName: destination.lastPathComponent, exportedAt: Date(), modelRevisions: revisions,
                                  backend: lastBackendLabel, processingSeconds: seconds,
                                  directory: external ? folder.path : nil)
        let updated = try store.recordExport(format: options.format, record: record, for: m.id)
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
        guard let m = selected, let processedURL = currentProcessedURL, !isBusy else { return }
        isExporting = true
        sessionsInUse.insert(m.id)
        exportMessage = tr("正在导出…", "Exporting…")
        let seconds = lastProcessingSeconds
        let revisions = lastRevisions
        Task {
            defer { sessionsInUse.remove(m.id) }
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

    /// Toolbar quick export: exports the current session if it is processed
    /// with its current AI settings, otherwise processes and exports it in one go.
    func quickExport() {
        guard let m = selected, !isBusy else { return }
        if exportFolder == nil, !chooseExportFolder() { return }
        if currentProcessedURL != nil { export() } else { processAndExport([m.id]) }
    }

    /// One click: AI processing, mastering and export for each session.
    func processAndExport(_ ids: Set<UUID>) {
        let targets = sessions.filter { ids.contains($0.id) }
        guard !targets.isEmpty, !isBusy else { return }
        if exportFolder == nil, !chooseExportFolder() { return }
        preview.pause()
        sessionsInUse.formUnion(targets.map(\.id))
        batchTask = Task {
            defer { sessionsInUse.subtract(targets.map(\.id)) }
            var exported: [URL] = []
            var failures: [String] = []
            for (index, queued) in targets.enumerated() {
                if Task.isCancelled { break }
                batch = BatchState(index: index + 1, total: targets.count, name: queued.name, stage: tr("准备模型", "Preparing model"), fraction: 0)
                do {
                    // Settings may have changed since the batch started; a session
                    // that can no longer be read fails rather than using the snapshot.
                    let m = try store.manifest(for: queued.id)
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
                    failures.append(tr("\(queued.name)：\(error.localizedDescription)", "\(queued.name): \(error.localizedDescription)"))
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

    /// Shows a session's export (or its folder) in Finder; defaults to the
    /// selected session. Does not change the selection.
    func revealInFinder(_ session: SessionManifest? = nil) {
        guard let m = session ?? selected else { return }
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
