import Recording
import Sessions
import SwiftUI

struct SessionListView: View {
    @EnvironmentObject private var model: AppModel
    @State private var renaming: SessionManifest?
    @State private var query = ""

    var body: some View {
        VStack(spacing: 0) {
            RecordPanel()
                .padding(.horizontal, 12)
                .padding(.top, 8)
                .padding(.bottom, 10)
            List(selection: $model.selection) {
                ForEach(groups, id: \.title) { group in
                    Section(group.title) {
                        ForEach(group.sessions) { session in
                            SessionRow(session: session)
                                .tag(session.id)
                                .contextMenu { menu(for: session) }
                        }
                    }
                }
            }
            .listStyle(.sidebar)
            .searchable(text: $query, placement: .sidebar, prompt: Text("搜索录音"))
            .overlay {
                if groups.isEmpty, !query.isEmpty {
                    ContentUnavailableView.search(text: query)
                }
            }
            .onDeleteCommand { model.requestDelete(model.selection) }
            .dropDestination(for: URL.self) { urls, _ in
                model.importFiles(urls.filter { $0.isFileURL })
                return true
            }
        }
        .sheet(item: $renaming) { session in
            RenameSessionSheet(session: session)
        }
    }

    @ViewBuilder private func menu(for session: SessionManifest) -> some View {
        let targets = model.selection.contains(session.id) ? model.selection : [session.id]
        Button("重命名…") { renaming = session }
            .disabled(model.isInUse(session.id))
        Button("在访达中显示") { model.revealInFinder(session) }
        Button("处理并导出") { model.processAndExport(targets) }
            .disabled(model.isBusy)
        Divider()
        Button("删除…", role: .destructive) { model.requestDelete(targets) }
            .disabled(targets.contains(where: model.isInUse))
    }

    /// Sessions matching the search, newest first, grouped by day.
    private var groups: [(title: String, sessions: [SessionManifest])] {
        let matching = query.isEmpty ? model.sessions
            : model.sessions.filter { $0.name.localizedCaseInsensitiveContains(query) }
        let calendar = Calendar.current
        var result: [(title: String, sessions: [SessionManifest])] = []
        for session in matching.sorted(by: { $0.createdAt > $1.createdAt }) {
            let title = dayTitle(session.createdAt, calendar)
            if result.last?.title == title { result[result.count - 1].sessions.append(session) } else { result.append((title, [session])) }
        }
        return result
    }

    private func dayTitle(_ date: Date, _ calendar: Calendar) -> String {
        if calendar.isDateInToday(date) { return model.tr("今天", "Today") }
        if calendar.isDateInYesterday(date) { return model.tr("昨天", "Yesterday") }
        return date.formatted(.dateTime.month().day().locale(model.locale))
    }

    static func duration(_ seconds: Double) -> String {
        let s = Int(seconds.rounded())
        return s >= 3600 ? String(format: "%d:%02d:%02d", s / 3600, s / 60 % 60, s % 60) : String(format: "%d:%02d", s / 60, s % 60)
    }
}

private struct SessionRow: View {
    @EnvironmentObject private var model: AppModel
    let session: SessionManifest

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            VStack(alignment: .leading, spacing: 2) {
                Text(session.name).lineLimit(1)
                HStack(spacing: 6) {
                    Text(session.createdAt, format: .dateTime.hour().minute())
                    if session.original.imported {
                        Image(systemName: "square.and.arrow.down").help("导入的文件")
                    }
                    if session.lastExport != nil {
                        Image(systemName: "checkmark.circle.fill").foregroundStyle(.green).help("已导出")
                    }
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }
            Spacer(minLength: 4)
            Text(verbatim: SessionListView.duration(session.original.durationSeconds))
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
        }
        .padding(.vertical, 3)
        .accessibilityElement(children: .combine)
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
                Button("重命名", action: save).keyboardShortcut(.defaultAction)
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

/// Record button, input choice and, while recording, time, level and discard.
struct RecordPanel: View {
    @EnvironmentObject private var model: AppModel
    @EnvironmentObject private var recorder: Recorder

    var body: some View {
        HStack(spacing: 12) {
            Button { model.toggleRecording() } label: {
                ZStack {
                    Circle().fill(recorder.isRecording ? Color.red.opacity(0.14) : Color.red)
                    if recorder.isRecording {
                        RoundedRectangle(cornerRadius: 3).fill(.red).frame(width: 14, height: 14)
                    } else {
                        Image(systemName: "mic.fill").foregroundStyle(.white).font(.system(size: 17, weight: .medium))
                    }
                }
                .frame(width: 42, height: 42)
                .contentShape(Circle())
            }
            .buttonStyle(.plain)
            .help(recorder.isRecording ? "停止并保存（⌘R）" : "开始录音（⌘R）")
            .accessibilityLabel(recorder.isRecording ? "停止" : "录音")

            VStack(alignment: .leading, spacing: 5) {
                if recorder.isRecording {
                    HStack {
                        Text(SessionListView.duration(recorder.elapsed))
                            .font(.system(.title3, design: .rounded).monospacedDigit())
                        Spacer()
                        Button { model.requestDiscard() } label: { Image(systemName: "trash") }
                            .buttonStyle(.borderless)
                            .help("丢弃这段录音（⌘⌫）")
                            .accessibilityLabel("丢弃录音")
                    }
                    LevelMeter(dBFS: recorder.levelDBFS)
                } else {
                    HStack(spacing: 4) {
                        Text("新录音").font(.headline).lineLimit(1)
                        Spacer(minLength: 4)
                        Image(systemName: "info.circle")
                            .foregroundStyle(.tertiary)
                            .help("在控制中心把麦克风模式设为“标准”，避免 macOS 预先处理声音。")
                    }
                    Picker("输入", selection: $model.recordingInput) {
                        Text("系统默认输入").tag(AudioInput?.none)
                        ForEach(recorder.inputs) { Text(verbatim: $0.name).tag(AudioInput?.some($0)) }
                    }
                    .labelsHidden()
                    .controlSize(.small)
                }
            }
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 12, style: .continuous).fill(Brand.surface))
        .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).strokeBorder(Brand.hairline))
        // A disconnected device falls back to the system default.
        .onChange(of: recorder.inputs) { _, inputs in
            if let selected = model.recordingInput, !inputs.contains(selected) { model.recordingInput = nil }
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
        .accessibilityElement()
        .accessibilityLabel("输入电平")
        .accessibilityValue(Text(verbatim: String(format: "%.0f dBFS", dBFS)))
    }
}

/// Shown when there are no recordings: the two ways to start.
struct EmptyStateView: View {
    @EnvironmentObject private var model: AppModel
    @Binding var importing: Bool

    var body: some View {
        ContentUnavailableView {
            Label("还没有录音", systemImage: "waveform.badge.mic")
        } description: {
            Text("录一段，或把音频文件拖进左侧列表。MicPolish 会去噪、去混响，再做母带处理。")
        } actions: {
            HStack(spacing: 12) {
                Button { model.toggleRecording() } label: { Label("开始录音", systemImage: "mic.fill") }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                Button { importing = true } label: { Label("导入音频…", systemImage: "square.and.arrow.down") }
                    .controlSize(.large)
            }
        }
    }
}
