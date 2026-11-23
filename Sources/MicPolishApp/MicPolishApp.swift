import AppKit
import Recording
import Sessions
import SwiftUI

@main
struct MicPolishApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    @StateObject private var model = AppModel()

    var body: some Scene {
        Window("MicPolish", id: "main") {
            ContentView()
                .environmentObject(model)
                .environmentObject(model.recorder)
                .environmentObject(model.preview)
                .environment(\.locale, model.locale)
                .tint(Brand.accent)
                .frame(minWidth: 1160, minHeight: 720)
                .task {
                    delegate.model = model
                    await model.launch()
                    if !delegate.pendingOpen.isEmpty {
                        model.importFiles(delegate.pendingOpen)
                        delegate.pendingOpen = []
                    }
                }
        }
        .windowResizability(.contentMinSize)
        .windowToolbarStyle(.unified)
        .commands { RecordingCommands(model: model, recorder: model.recorder) }

        Settings {
            SettingsView()
                .environmentObject(model)
                .environment(\.locale, model.locale)
                .tint(Brand.accent)
        }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    weak var model: AppModel?
    var pendingOpen: [URL] = []

    func applicationDidFinishLaunching(_ notification: Notification) {
        if Bundle.main.bundleIdentifier == nil {
            // Launched as a bare SwiftPM executable during development, the
            // process must opt into being a regular foreground app.
            NSApp.setActivationPolicy(.regular)
            NSApp.activate(ignoringOtherApps: true)
        }
        if ProcessInfo.processInfo.environment["MICPOLISH_BACKGROUND"] == "1" {
            // UI verification while someone else uses the Mac: keep the window
            // full-size behind other windows instead of being stashed by Stage
            // Manager, and never take focus.
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                for window in NSApp.windows {
                    window.collectionBehavior = [.canJoinAllSpaces, .stationary, .ignoresCycle]
                    window.orderBack(nil)
                }
            }
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    /// Files dropped on the Dock icon or opened with "Open With" are imported.
    func application(_ application: NSApplication, open urls: [URL]) {
        MainActor.assumeIsolated {
            if let model { model.importFiles(urls) } else { pendingOpen += urls }
        }
    }

    func applicationWillTerminate(_ notification: Notification) {
        MainActor.assumeIsolated { model?.quit() }
    }
}

struct ContentView: View {
    @EnvironmentObject private var model: AppModel
    @State private var importing = false
    @EnvironmentObject private var recorder: Recorder

    var body: some View {
        NavigationSplitView {
            SessionListView()
                .navigationSplitViewColumnWidth(min: 250, ideal: 280)
        } detail: {
            VStack(spacing: 0) {
                if model.modelState != .ready { ModelBanner() }
                if let batch = model.batch { BatchBanner(batch: batch) }
                Group {
                    if recorder.isRecording {
                        LiveRecordingView()
                    } else if model.selection.count > 1 {
                        BatchSelectionView()
                    } else if model.selected != nil {
                        SessionDetailView()
                    } else {
                        EmptyStateView(importing: $importing)
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            .background(Color(nsColor: .windowBackgroundColor))
        }
        .toolbar { toolbar }
        .fileImporter(isPresented: $importing, allowedContentTypes: [.audio], allowsMultipleSelection: true) { result in
            if case .success(let urls) = result { model.importFiles(urls) }
        }
        .confirmationDialog(deleteTitle, isPresented: $model.confirmingDelete) {
            Button("删除", role: .destructive) { model.delete(model.deleteTargets) }
        } message: {
            Text("原始录音和导出到会话文件夹的成品都会被删除，无法恢复。")
        }
        .confirmationDialog("丢弃这段录音？", isPresented: $model.confirmingDiscard) {
            Button("丢弃", role: .destructive) { model.discardRecording() }
                .keyboardShortcut(.defaultAction)
        } message: {
            Text("录到的内容会被删除，不会保存。")
        }
        .sheet(item: $model.pendingRename) { session in
            RenameSheet(session: session)
        }
        .alert("无法完成操作", isPresented: Binding(get: { model.errorMessage != nil }, set: { if !$0 { model.errorMessage = nil } })) {
            Button("好") { model.errorMessage = nil }
        } message: {
            Text(model.errorMessage ?? "")
        }
    }

    private var deleteTitle: String {
        model.tr("删除 \(model.deleteTargets.count) 条录音？", "Delete \(model.deleteTargets.count) recording(s)?")
    }

    @ToolbarContentBuilder
    private var toolbar: some ToolbarContent {
        ToolbarItemGroup(placement: .primaryAction) {
            Button { importing = true } label: {
                Label("导入音频", systemImage: "square.and.arrow.down")
            }
            .help("导入音频")

            Button { model.processAndExport(model.selection) } label: {
                Label("一键处理并导出", systemImage: "wand.and.stars")
            }
            .help("对所选录音依次进行 AI 处理、母带处理并导出到导出文件夹")
            .disabled(model.selection.isEmpty || model.isBusy)

            Button { model.quickExport() } label: {
                Label("快速导出", systemImage: "square.and.arrow.up.on.square")
            }
            .help("把当前录音导出到导出文件夹；还没处理过的会先自动处理")
            .disabled(model.selected == nil || model.isBusy)

            Menu {
                Picker("语言", selection: $model.language) {
                    Text("跟随系统").tag(AppLanguage.system)
                    Text(verbatim: "简体中文").tag(AppLanguage.zhHans)
                    Text(verbatim: "English").tag(AppLanguage.en)
                }
                .pickerStyle(.inline)
            } label: {
                Label("语言", systemImage: "globe")
            }
            .help("界面语言")
        }
    }
}

struct RenameSheet: View {
    @EnvironmentObject private var model: AppModel
    @Environment(\.dismiss) private var dismiss
    let session: SessionManifest
    @State private var name = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Label("录音已保存", systemImage: "checkmark.circle.fill")
                .font(.headline)
                .foregroundStyle(.green)
            Text("给这段录音起个名字").foregroundStyle(.secondary)
            TextField("名称", text: $name)
                .textFieldStyle(.roundedBorder)
                .onSubmit(save)
            Toggle("以后不再询问，直接用默认名称", isOn: Binding(
                get: { !model.askNameAfterRecording }, set: { model.askNameAfterRecording = !$0 }))
                .font(.callout)
            HStack {
                Spacer()
                Button("保留默认名称") { dismiss() }
                Button("保存", action: save).keyboardShortcut(.defaultAction)
            }
        }
        .padding(22)
        .frame(width: 380)
        .onAppear { name = session.name }
    }

    private func save() {
        if name != session.name { model.rename(session.id, to: name) }
        dismiss()
    }
}

struct ModelBanner: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: "cpu").font(.title3).foregroundStyle(.orange)
            switch model.modelState {
            case .checking:
                Text("正在检查 AI 模型…")
            case .notInstalled:
                VStack(alignment: .leading, spacing: 2) {
                    Text("需要下载 NVIDIA RE-USE 语音修复模型（约 39 MB）").bold()
                    Text("模型许可：OpenMDW-1.1（允许商用）。下载后完全在本机运行，音频不会上传。")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                Button("下载模型") { Task { await model.downloadModel(model.selectedReuseDescriptor) } }
                    .buttonStyle(.borderedProminent)
            case .downloading(let fraction):
                ProgressView(value: fraction) { Text("正在下载模型…") }
            case .failed(let message):
                Text(verbatim: model.tr("模型下载失败：\(message)", "Model download failed: \(message)")).foregroundStyle(.red)
                Spacer()
                Button("重试") { Task { await model.downloadModel(model.selectedReuseDescriptor) } }
            case .ready:
                EmptyView()
            }
        }
        .padding(12)
        .background(.orange.opacity(0.08))
    }
}

struct BatchBanner: View {
    @EnvironmentObject private var model: AppModel
    let batch: BatchState

    var body: some View {
        HStack(spacing: 12) {
            ProgressView().controlSize(.small)
            VStack(alignment: .leading, spacing: 4) {
                Text(verbatim: model.tr("一键处理并导出：第 \(batch.index)/\(batch.total) 个 · \(batch.name) · \(batch.stage)",
                                        "Process & export: \(batch.index) of \(batch.total) · \(batch.name) · \(batch.stage)"))
                    .font(.callout)
                ProgressView(value: (Double(batch.index - 1) + batch.fraction) / Double(batch.total))
            }
            Button("取消") { model.cancelBatch() }
        }
        .padding(12)
        .background(.blue.opacity(0.08))
    }
}
