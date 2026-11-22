import Recording
import Sessions
import SwiftUI

struct SessionListView: View {
    @EnvironmentObject private var model: AppModel
    @Binding var confirmingDelete: Bool
    @State private var renaming: SessionManifest?

    var body: some View {
        VStack(spacing: 0) {
            RecordPanel()
                .padding(12)
            List(selection: $model.selection) {
                Section {
                    ForEach(model.sessions) { session in
                        SessionRow(session: session)
                            .tag(session.id)
                            .contextMenu {
                                Button("重命名…") { renaming = session }
                                Button("在访达中显示") { model.select(session.id); model.revealInFinder() }
                                Divider()
                                Button("删除", role: .destructive) {
                                    if !model.selection.contains(session.id) { model.select(session.id) }
                                    confirmingDelete = true
                                }
                            }
                    }
                } header: {
                    Text("录音列表")
                }
            }
            .listStyle(.sidebar)
            .onDeleteCommand { if !model.selection.isEmpty { confirmingDelete = true } }
            .dropDestination(for: URL.self) { urls, _ in
                model.importFiles(urls.filter { $0.isFileURL })
                return true
            }
            if model.selection.count > 1 {
                HStack {
                    Text(verbatim: model.tr("已选 \(model.selection.count) 条", "\(model.selection.count) selected"))
                        .font(.caption).foregroundStyle(.secondary)
                    Spacer()
                    Button(role: .destructive) { confirmingDelete = true } label: {
                        Label("删除所选", systemImage: "trash")
                    }
                    .controlSize(.small)
                }
                .padding(10)
                .background(.bar)
            }
        }
        .sheet(item: $renaming) { session in
            RenameSessionSheet(session: session)
        }
    }

    static func duration(_ seconds: Double) -> String {
        let s = Int(seconds.rounded())
        return s >= 3600 ? String(format: "%d:%02d:%02d", s / 3600, s / 60 % 60, s % 60) : String(format: "%d:%02d", s / 60, s % 60)
    }
}

private struct SessionRow: View {
    let session: SessionManifest

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: session.original.imported ? "doc.richtext" : "mic.fill")
                .foregroundStyle(.secondary)
                .frame(width: 18)
            VStack(alignment: .leading, spacing: 2) {
                Text(session.name).lineLimit(1)
                HStack(spacing: 6) {
                    Text(session.createdAt, format: .dateTime.month().day().hour().minute())
                    Text(SessionListView.duration(session.original.durationSeconds)).monospacedDigit()
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }
            Spacer(minLength: 0)
            if session.lastExport != nil {
                Image(systemName: "checkmark.seal.fill")
                    .foregroundStyle(.green)
                    .help("已导出")
            }
        }
        .padding(.vertical, 2)
    }
}

private struct RenameSessionSheet: View {
    @EnvironmentObject private var model: AppModel
    @Environment(\.dismiss) private var dismiss
    let session: SessionManifest
    @State private var name = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("重命名").font(.headline)
            TextField("名称", text: $name).textFieldStyle(.roundedBorder).onSubmit(save)
            HStack {
                Spacer()
                Button("取消") { dismiss() }
                Button("保存", action: save).keyboardShortcut(.defaultAction)
            }
        }
        .padding(22)
        .frame(width: 360)
        .onAppear { name = session.name }
    }

    private func save() {
        model.rename(session.id, to: name)
        dismiss()
    }
}

struct RecordPanel: View {
    @EnvironmentObject private var model: AppModel
    @EnvironmentObject private var recorder: Recorder
    @State private var selectedInput: AudioInput?

    var body: some View {
        VStack(spacing: 10) {
            HStack(spacing: 12) {
                Button {
                    if recorder.isRecording {
                        Task { await model.stopRecording() }
                    } else {
                        Task { await model.startRecording(input: selectedInput) }
                    }
                } label: {
                    ZStack {
                        Circle().fill(recorder.isRecording ? Color.red.opacity(0.15) : Color.red)
                            .frame(width: 44, height: 44)
                        if recorder.isRecording {
                            RoundedRectangle(cornerRadius: 4).fill(.red).frame(width: 16, height: 16)
                        } else {
                            Image(systemName: "mic.fill").foregroundStyle(.white).font(.title3)
                        }
                    }
                }
                .buttonStyle(.plain)
                .help(recorder.isRecording ? "停止并保存" : "开始录音")
                .accessibilityLabel(recorder.isRecording ? "停止" : "录音")

                VStack(alignment: .leading, spacing: 6) {
                    if recorder.isRecording {
                        Text(SessionListView.duration(recorder.elapsed))
                            .font(.system(.title3, design: .rounded).monospacedDigit())
                        LevelMeter(dBFS: recorder.levelDBFS)
                    } else {
                        Text("新录音").font(.headline)
                        Picker("输入", selection: $selectedInput) {
                            Text("系统默认输入").tag(AudioInput?.none)
                            ForEach(recorder.inputs) { Text(verbatim: $0.name).tag(AudioInput?.some($0)) }
                        }
                        .labelsHidden()
                        .controlSize(.small)
                    }
                }
            }
            if recorder.isRecording, recorder.levelDBFS > -1 {
                Text("电平过高，可能削波").font(.caption).foregroundStyle(.red)
            } else if !recorder.isRecording {
                Text("提示：在控制中心把麦克风模式设为“标准”，避免系统预先处理声音。")
                    .font(.caption2).foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 12).fill(.background.opacity(0.6)))
        .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(.quaternary))
        // A disconnected device falls back to the system default.
        .onChange(of: recorder.inputs) { _, inputs in
            if let selected = selectedInput, !inputs.contains(selected) { selectedInput = nil }
        }
    }
}

struct LevelMeter: View {
    let dBFS: Float

    var body: some View {
        GeometryReader { geo in
            let fraction = CGFloat(max(0, min(1, (dBFS + 60) / 60)))
            ZStack(alignment: .leading) {
                Capsule().fill(.quaternary)
                Capsule().fill(LinearGradient(colors: [.green, .yellow, .red], startPoint: .leading, endPoint: .trailing))
                    .frame(width: geo.size.width * fraction)
            }
        }
        .frame(height: 6)
    }
}
