import Recording
import Sessions
import SwiftUI
import UniformTypeIdentifiers

struct SessionListView: View {
    @EnvironmentObject private var model: AppModel
    @EnvironmentObject private var recorder: Recorder
    @State private var importing = false
    @State private var naming = false
    @State private var newName = ""
    @State private var selectedInput: AudioInput?

    var body: some View {
        VStack(spacing: 0) {
            RecordPanel(selectedInput: $selectedInput) {
                newName = Self.defaultName()
                naming = true
            }
            .padding(12)
            Divider()
            List(selection: $model.selectedID) {
                ForEach(model.sessions) { session in
                    VStack(alignment: .leading, spacing: 2) {
                        Text(session.name).lineLimit(1)
                        HStack(spacing: 6) {
                            Text(session.createdAt, format: .dateTime.month().day().hour().minute())
                            Text(Self.duration(session.original.durationSeconds))
                            if session.lastExport != nil {
                                Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
                            }
                        }
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    }
                    .tag(session.id)
                    .contextMenu {
                        Button("在访达中显示") { model.selectedID = session.id; model.revealInFinder() }
                        Button("删除", role: .destructive) { model.delete(session.id) }
                    }
                }
            }
        }
        .toolbar {
            ToolbarItem {
                Button { importing = true } label: { Label("导入音频", systemImage: "square.and.arrow.down") }
            }
        }
        .dropDestination(for: URL.self) { urls, _ in
            model.importFiles(urls.filter { $0.isFileURL })
            return true
        }
        .fileImporter(isPresented: $importing, allowedContentTypes: [.audio], allowsMultipleSelection: true) { result in
            if case .success(let urls) = result { model.importFiles(urls) }
        }
        .alert("给这段录音起个名字", isPresented: $naming) {
            TextField("名称", text: $newName)
            Button("保存") { Task { await model.stopRecording(name: newName) } }
        }
    }

    static func defaultName() -> String {
        let f = DateFormatter()
        f.dateFormat = "M月d日 HH:mm 录音"
        return f.string(from: Date())
    }

    static func duration(_ seconds: Double) -> String {
        let s = Int(seconds.rounded())
        return s >= 3600 ? String(format: "%d:%02d:%02d", s / 3600, s / 60 % 60, s % 60) : String(format: "%d:%02d", s / 60, s % 60)
    }
}

struct RecordPanel: View {
    @EnvironmentObject private var model: AppModel
    @EnvironmentObject private var recorder: Recorder
    @Binding var selectedInput: AudioInput?
    let onStop: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Button {
                    if recorder.isRecording {
                        onStop()
                    } else {
                        Task { await model.startRecording(input: selectedInput) }
                    }
                } label: {
                    Label(recorder.isRecording ? "停止" : "录音", systemImage: recorder.isRecording ? "stop.fill" : "record.circle")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .tint(recorder.isRecording ? .gray : .red)
                .controlSize(.large)
            }
            if recorder.isRecording {
                HStack {
                    Text(SessionListView.duration(recorder.elapsed)).monospacedDigit()
                    LevelMeter(dBFS: recorder.levelDBFS)
                }
                if recorder.levelDBFS > -1 {
                    Text("电平过高，可能削波").font(.caption).foregroundStyle(.red)
                }
            } else {
                Picker("输入", selection: $selectedInput) {
                    Text("系统默认输入").tag(AudioInput?.none)
                    ForEach(recorder.availableInputs()) { Text($0.name).tag(AudioInput?.some($0)) }
                }
                .labelsHidden()
                Text("提示：在控制中心把麦克风模式设为“标准”，避免系统预先处理声音。")
                    .font(.caption2).foregroundStyle(.secondary)
            }
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
                Capsule().fill(dBFS > -6 ? .red : dBFS > -18 ? .yellow : .green)
                    .frame(width: geo.size.width * fraction)
            }
        }
        .frame(height: 8)
    }
}
