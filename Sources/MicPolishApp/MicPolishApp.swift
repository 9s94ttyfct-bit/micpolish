import AppKit
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
                .frame(minWidth: 1120, minHeight: 680)
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

        Settings {
            SettingsView().environmentObject(model)
        }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    weak var model: AppModel?

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

    var pendingOpen: [URL] = []

    func applicationWillTerminate(_ notification: Notification) {
        MainActor.assumeIsolated { model?.quit() }
    }
}

struct ContentView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        NavigationSplitView {
            SessionListView()
                .navigationSplitViewColumnWidth(min: 220, ideal: 250)
        } detail: {
            VStack(spacing: 0) {
                if model.modelState != .ready {
                    ModelBanner()
                }
                if model.selected != nil {
                    SessionDetailView()
                } else {
                    ContentUnavailableView("还没有录音", systemImage: "mic",
                                           description: Text("点左上角的录音按钮开始，或导入一个音频文件。"))
                }
            }
        }
        .alert("出错了", isPresented: Binding(get: { model.errorMessage != nil }, set: { if !$0 { model.errorMessage = nil } })) {
            Button("好") { model.errorMessage = nil }
        } message: {
            Text(model.errorMessage ?? "")
        }
    }
}

struct ModelBanner: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: "cpu")
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
                Button("下载模型") { Task { await model.downloadModel() } }
                    .buttonStyle(.borderedProminent)
            case .downloading(let fraction):
                ProgressView(value: fraction) { Text("正在下载模型…") }
            case .failed(let message):
                Text("模型下载失败：\(message)").foregroundStyle(.red)
                Spacer()
                Button("重试") { Task { await model.downloadModel() } }
            case .ready:
                EmptyView()
            }
        }
        .padding(12)
        .background(.yellow.opacity(0.12))
    }
}
