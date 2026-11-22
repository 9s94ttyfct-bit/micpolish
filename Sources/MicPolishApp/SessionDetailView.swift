import Export
import Mastering
import Preview
import Sessions
import SwiftUI

struct SessionDetailView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                if let m = model.selected {
                    Header(manifest: m)
                    PreviewPanel()
                    ProcessPanel(manifest: m)
                    MasteringPanel(settings: m.mastering)
                    ExportPanel(manifest: m)
                }
            }
            .padding(20)
        }
    }
}

private struct Header: View {
    @EnvironmentObject private var model: AppModel
    let manifest: SessionManifest

    var body: some View {
        let usage = model.diskUsage(of: manifest)
        HStack(alignment: .firstTextBaseline) {
            Text(manifest.name).font(.title2.bold())
            Spacer()
            let scratch = model.workspaceBytes > 0 ? " · \(String(localized: "临时")) \(Self.mb(model.workspaceBytes))" : ""
            Text("原始 \(Self.mb(usage.original)) · 成品 \(Self.mb(usage.export))\(scratch)")
                .font(.caption).foregroundStyle(.secondary)
            Button { model.revealInFinder() } label: { Image(systemName: "folder") }
                .help("在访达中显示")
                .accessibilityLabel("在访达中显示")
        }
    }

    static func mb(_ bytes: Int64) -> String {
        bytes < 1_048_576 ? "\(bytes / 1024) KB" : String(format: "%.1f MB", Double(bytes) / 1_048_576)
    }
}

private struct PreviewPanel: View {
    @EnvironmentObject private var model: AppModel
    @EnvironmentObject private var preview: PreviewPlayer

    var body: some View {
        GroupBox("试听") {
            VStack(alignment: .leading, spacing: 10) {
                Picker("音源", selection: $preview.source) {
                    Text("原始").tag(PreviewSource.original)
                    Text("AI 处理后").tag(PreviewSource.processed)
                    Text("成品").tag(PreviewSource.mastered)
                }
                .pickerStyle(.segmented)
                .disabled(model.processedURL == nil)
                HStack {
                    Button {
                        preview.isPlaying ? preview.pause() : preview.play()
                    } label: {
                        Image(systemName: preview.isPlaying ? "pause.fill" : "play.fill").frame(width: 20)
                    }
                    Slider(value: Binding(get: { preview.position }, set: { preview.seek(to: $0) }),
                           in: 0...max(preview.duration, 0.1))
                    Text("\(SessionListView.duration(preview.position)) / \(SessionListView.duration(preview.duration))")
                        .monospacedDigit().font(.caption)
                }
                Text(model.processedURL == nil
                     ? "处理后可在三路之间切换对比。"
                     : "三路已按响度对齐，切换时比较的是音质而不是音量。")
                    .font(.caption).foregroundStyle(.secondary)
            }
            .padding(6)
        }
    }
}

private struct ProcessPanel: View {
    @EnvironmentObject private var model: AppModel
    let manifest: SessionManifest

    var body: some View {
        GroupBox("AI 处理") {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Picker("模型", selection: Binding(
                        get: { manifest.processing.reuseVariant },
                        set: { v in model.updateSelected { $0.processing.reuseVariant = v } })) {
                        Text("RE-USE").tag("reuse-nemotron")
                        Text("RE-USE（保留笑声）").tag("reuse-nemotron-laughter")
                    }
                    .frame(width: 260)
                    Spacer()
                    switch model.processing {
                    case .running:
                        Button("取消") { model.cancelProcessing() }
                    default:
                        Button(model.processedURL == nil ? "处理" : "重新处理") { model.process() }
                            .buttonStyle(.borderedProminent)
                            .disabled(model.isExporting || (model.modelState != .ready && model.modelState != .notInstalled))
                    }
                }
                switch model.processing {
                case .idle:
                    if manifest.lastExport != nil, let estimate = model.estimatedProcessingSeconds {
                        Text("这个会话已导出过；中间文件已清理，重新调整需要再跑一次 AI（约 \(Int(estimate.rounded())) 秒）。")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                case .running(let stage, let fraction, let remaining):
                    ProgressView(value: fraction) {
                        Text(stage)
                    } currentValueLabel: {
                        Text(remaining.map { "后端：MLX GPU · 剩余约 \(Int($0.rounded())) 秒" } ?? "后端：MLX GPU")
                    }
                case .done:
                    Label("处理完成", systemImage: "checkmark.circle").foregroundStyle(.green)
                case .failed(let message):
                    Text("处理失败：\(message)").foregroundStyle(.red)
                }
            }
            .padding(6)
        }
    }
}

private struct ExportPanel: View {
    @EnvironmentObject private var model: AppModel
    let manifest: SessionManifest

    var body: some View {
        GroupBox("导出") {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Picker("格式", selection: Binding(
                        get: { manifest.export.format },
                        set: { f in model.updateSelected { $0.export.format = f } })) {
                        Text("M4A (AAC)").tag(ExportFormat.m4a)
                        Text("MP3").tag(ExportFormat.mp3)
                    }
                    .pickerStyle(.segmented)
                    .frame(width: 220)
                    Picker("码率", selection: Binding(
                        get: { manifest.export.bitrateKbps },
                        set: { b in model.updateSelected { $0.export.bitrateKbps = b } })) {
                        ForEach(ExportOptions.bitrates, id: \.self) { Text("\($0) kbps").tag($0) }
                    }
                    .frame(width: 180)
                    Spacer()
                    Button("导出") { model.export() }
                        .buttonStyle(.borderedProminent)
                        .disabled(model.processedURL == nil || model.isExporting || model.isProcessing)
                }
                if let message = model.exportMessage {
                    Text(message).font(.caption).foregroundStyle(.secondary)
                } else if model.processedURL == nil {
                    Text("先完成 AI 处理再导出。导出后只保留原始录音和这一个成品文件。")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            .padding(6)
        }
    }
}
