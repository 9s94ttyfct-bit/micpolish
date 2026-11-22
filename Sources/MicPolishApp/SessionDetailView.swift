import Export
import Mastering
import Preview
import Sessions
import SwiftUI

/// A titled, rounded card used for each step of the workflow.
struct Card<Content: View>: View {
    let title: LocalizedStringKey
    let systemImage: String
    @ViewBuilder let content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label(title, systemImage: systemImage).font(.headline)
            content
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 14).fill(Color(nsColor: .controlBackgroundColor)))
        .overlay(RoundedRectangle(cornerRadius: 14).strokeBorder(.quaternary))
    }
}

/// The three steps of the workflow; one is shown at a time under the preview.
enum WorkflowStep: Int, CaseIterable, Identifiable {
    case restore, master, export
    var id: Int { rawValue }
}

struct SessionDetailView: View {
    @EnvironmentObject private var model: AppModel
    @State private var step: WorkflowStep = .restore

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                if let m = model.selected {
                    Header(manifest: m)
                    PreviewPanel()
                    StepBar(step: $step)
                    switch step {
                    case .restore: ProcessPanel(manifest: m)
                    case .master: MasteringPanel(settings: m.mastering)
                    case .export: ExportPanel(manifest: m)
                    }
                }
            }
            .padding(20)
            .frame(maxWidth: 1100)
            .frame(maxWidth: .infinity)
        }
    }
}

/// Switches between the workflow steps; each shows whether it is done.
private struct StepBar: View {
    @EnvironmentObject private var model: AppModel
    @Binding var step: WorkflowStep

    var body: some View {
        HStack(spacing: 8) {
            ForEach(WorkflowStep.allCases) { s in
                let selected = s == step
                Button { step = s } label: {
                    HStack(spacing: 10) {
                        ZStack {
                            Circle().fill(selected ? Color.white : Color.accentColor.opacity(0.15))
                            if done(s) {
                                Image(systemName: "checkmark").font(.caption.bold())
                            } else {
                                Text(verbatim: "\(s.rawValue + 1)").font(.caption.bold())
                            }
                        }
                        .foregroundStyle(Color.accentColor)
                        .frame(width: 24, height: 24)
                        VStack(alignment: .leading, spacing: 1) {
                            Text(verbatim: title(s)).font(.callout.bold())
                            Text(verbatim: subtitle(s)).font(.caption2)
                                .foregroundStyle(selected ? .white.opacity(0.85) : .secondary)
                                .lineLimit(1)
                        }
                        Spacer(minLength: 0)
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 9)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .foregroundStyle(selected ? .white : .primary)
                    .background(RoundedRectangle(cornerRadius: 12)
                        .fill(selected ? Color.accentColor : Color(nsColor: .controlBackgroundColor)))
                    .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(selected ? .clear : Color.secondary.opacity(0.2)))
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .accessibilityLabel(Text(verbatim: title(s)))
                .accessibilityAddTraits(selected ? .isSelected : [])
            }
        }
    }

    private func title(_ s: WorkflowStep) -> String {
        switch s {
        case .restore: model.tr("AI 修复", "AI Restoration")
        case .master: model.tr("母带处理", "Mastering")
        case .export: model.tr("导出", "Export")
        }
    }

    private func subtitle(_ s: WorkflowStep) -> String {
        switch s {
        case .restore: model.tr("降噪 · 去混响 · 高频", "Noise · reverb · high band")
        case .master: model.presetName(for: model.selected?.mastering ?? .default) ?? model.tr("自定义", "Custom")
        case .export: (model.selected?.export.format.rawValue.uppercased() ?? "") + " · " + (model.exportFolder?.lastPathComponent ?? model.tr("会话文件夹", "Session folder"))
        }
    }

    private func done(_ s: WorkflowStep) -> Bool {
        switch s {
        // Only while the processed audio matches the current AI settings.
        case .restore: model.processedURL != nil && model.processedSettings == model.selected?.processing
        case .master: false
        case .export: model.processedURL == nil && model.selected?.lastExport != nil
        }
    }
}

private struct Header: View {
    @EnvironmentObject private var model: AppModel
    let manifest: SessionManifest
    @State private var renaming = false
    @State private var name = ""

    var body: some View {
        let usage = model.diskUsage(of: manifest)
        HStack(alignment: .center, spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 8) {
                    if renaming {
                        TextField("名称", text: $name)
                            .textFieldStyle(.roundedBorder)
                            .font(.title2.bold())
                            .frame(maxWidth: 360)
                            .onSubmit { model.rename(manifest.id, to: name); renaming = false }
                    } else {
                        Text(manifest.name).font(.title2.bold())
                        Button { name = manifest.name; renaming = true } label: { Image(systemName: "pencil") }
                            .buttonStyle(.borderless)
                            .help("重命名…")
                            .accessibilityLabel("重命名…")
                    }
                }
                HStack(spacing: 10) {
                    Label(SessionListView.duration(manifest.original.durationSeconds), systemImage: "clock")
                    Label { Text(verbatim: "\(manifest.original.sampleRate / 1000) kHz") } icon: { Image(systemName: "waveform") }
                    Label { Text(verbatim: Self.mb(usage.original)) } icon: { Image(systemName: "doc") }
                    if usage.export > 0 {
                        Label { Text(verbatim: Self.mb(usage.export)) } icon: { Image(systemName: "checkmark.seal") }
                    }
                    if model.workspaceBytes > 0 {
                        Label { Text(verbatim: model.tr("临时 \(Self.mb(model.workspaceBytes))", "Scratch \(Self.mb(model.workspaceBytes))")) }
                            icon: { Image(systemName: "tray") }
                    }
                }
                .labelStyle(.titleAndIcon)
                .font(.caption)
                .foregroundStyle(.secondary)
            }
            Spacer()
            Button { model.revealInFinder() } label: { Label("在访达中显示", systemImage: "folder") }
                .help("在访达中显示")
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
        Card(title: "试听对比", systemImage: "headphones") {
            HStack(spacing: 10) {
                SourceTab(source: .original, title: "原始", subtitle: "麦克风录到的声音", icon: "mic")
                SourceTab(source: .processed, title: "AI 处理后", subtitle: "RE-USE 去噪去混响，未做母带", icon: "sparkles")
                SourceTab(source: .mastered, title: "成品", subtitle: "AI + 母带处理，导出的就是它", icon: "checkmark.seal")
            }
            ZStack {
                WaveformView(
                    envelope: preview.envelopes[preview.source == .original ? .original : .processed] ?? preview.envelopes[.original] ?? [],
                    progress: preview.duration > 0 ? preview.position / preview.duration : 0,
                    tint: preview.source == .original ? .secondary : .accentColor
                ) { fraction in
                    preview.seek(to: fraction * preview.duration)
                }
                if let m = model.selected, preview.duration > 0 {
                    TrimOverlay(duration: preview.duration, trim: m.trim) { model.setTrim($0) }
                }
            }
            .frame(height: 70)
            TrimBar()
            HStack {
                Button {
                    preview.isPlaying ? preview.pause() : preview.play()
                } label: {
                    Image(systemName: preview.isPlaying ? "pause.fill" : "play.fill")
                        .frame(width: 22, height: 22)
                }
                .buttonStyle(.borderedProminent)
                .clipShape(Circle())
                .accessibilityLabel(preview.isPlaying ? "暂停" : "播放")
                Text(verbatim: "\(SessionListView.duration(preview.position)) / \(SessionListView.duration(preview.duration))")
                    .monospacedDigit().font(.callout)
                Spacer()
            }
            Text(model.processedURL == nil
                 ? "AI 处理后才能切换到“AI 处理后”和“成品”。三路试听按响度对齐，比较的是音质而不是音量。"
                 : "“AI 处理后”只看 AI 修复的效果（还没调 EQ、压缩和响度）；“成品”在它之上叠加下方的母带处理，用来调参数，导出的就是成品。")
                .font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

private struct SourceTab: View {
    @EnvironmentObject private var model: AppModel
    @EnvironmentObject private var preview: PreviewPlayer
    let source: PreviewSource
    let title: LocalizedStringKey
    let subtitle: LocalizedStringKey
    let icon: String

    var body: some View {
        let selected = preview.source == source
        let enabled = source == .original || model.processedURL != nil
        Button { preview.source = source } label: {
            HStack(spacing: 8) {
                Image(systemName: icon).font(.title3)
                VStack(alignment: .leading, spacing: 2) {
                    Text(title).font(.callout.bold())
                    Text(subtitle).font(.caption2).foregroundStyle(selected ? .white.opacity(0.85) : .secondary)
                        .lineLimit(2)
                }
                Spacer(minLength: 0)
            }
            .padding(10)
            .frame(maxWidth: .infinity, minHeight: 54, alignment: .leading)
            .foregroundStyle(selected ? .white : .primary)
            .background(RoundedRectangle(cornerRadius: 10).fill(selected ? Color.accentColor : Color.secondary.opacity(0.08)))
        }
        .buttonStyle(.plain)
        .disabled(!enabled)
        .opacity(enabled ? 1 : 0.45)
        .accessibilityLabel(title)
    }
}

/// Dims the trimmed-off head and tail of the waveform and offers a draggable
/// handle at each end of the kept span.
private struct TrimOverlay: View {
    let duration: Double
    let trim: TrimRange?
    let commit: (TrimRange) -> Void
    @State private var live: TrimRange?

    private static let minimumSeconds = 0.1
    private static let step = 0.1

    var body: some View {
        GeometryReader { geo in
            let width = geo.size.width
            let range = live ?? trim ?? TrimRange(start: 0, end: duration)
            let x0 = CGFloat(range.start / duration) * width
            let x1 = CGFloat(range.end / duration) * width
            let shade = Color(nsColor: .controlBackgroundColor).opacity(0.75)
            ZStack(alignment: .topLeading) {
                Rectangle().fill(shade).frame(width: max(0, x0)).allowsHitTesting(false)
                Rectangle().fill(shade).frame(width: max(0, width - x1)).offset(x: x1).allowsHitTesting(false)
                handle(x: x0, width: width, height: geo.size.height, label: "裁剪起点", value: range.start) { seconds in
                    TrimRange(start: max(0, min(seconds, range.end - Self.minimumSeconds)), end: range.end)
                }
                handle(x: x1, width: width, height: geo.size.height, label: "裁剪终点", value: range.end) { seconds in
                    TrimRange(start: range.start, end: min(duration, max(seconds, range.start + Self.minimumSeconds)))
                }
            }
            .coordinateSpace(name: "trim")
            .frame(width: width, height: geo.size.height, alignment: .topLeading)
        }
    }

    private func handle(x: CGFloat, width: CGFloat, height: CGFloat, label: LocalizedStringKey, value: Double,
                        moved: @escaping (Double) -> TrimRange) -> some View {
        TrimHandle(height: height)
            .offset(x: x - 8)
            .gesture(
                DragGesture(minimumDistance: 1, coordinateSpace: .named("trim"))
                    .onChanged { drag in
                        let fraction = width > 0 ? Double(drag.location.x / width) : 0
                        live = moved(max(0, min(duration, fraction * duration)))
                    }
                    .onEnded { _ in
                        if let live { commit(live) }
                        live = nil
                    }
            )
            .accessibilityElement()
            .accessibilityLabel(label)
            .accessibilityValue(Text(verbatim: TrimBar.time(value)))
            .accessibilityAdjustableAction { direction in
                commit(moved(value + (direction == .increment ? Self.step : -Self.step)))
            }
    }
}

private struct TrimHandle: View {
    let height: CGFloat

    var body: some View {
        ZStack {
            Capsule().fill(Color.yellow).frame(width: 5, height: height)
            Capsule().fill(Color.black.opacity(0.35)).frame(width: 1.5, height: 14)
        }
        .frame(width: 16, height: height)
        .contentShape(Rectangle())
        .onHover { inside in
            if inside { NSCursor.resizeLeftRight.push() } else { NSCursor.pop() }
        }
    }
}

/// The kept span and the quick-trim actions under the waveform.
private struct TrimBar: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        if let m = model.selected {
            HStack(spacing: 10) {
                Image(systemName: "scissors").foregroundStyle(.secondary)
                if let trim = m.trim {
                    Text(verbatim: model.tr("保留 \(Self.time(trim.start)) – \(Self.time(trim.end))（\(Self.time(trim.duration))）",
                                            "Keeping \(Self.time(trim.start)) – \(Self.time(trim.end)) (\(Self.time(trim.duration)))"))
                        .monospacedDigit()
                } else {
                    Text("拖动波形两端的黄色手柄裁掉开头和结尾")
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button { model.autoTrim() } label: { Label("自动去首尾静音", systemImage: "wand.and.stars") }
                    .help("把开头和结尾的静音裁掉，两端各留 0.25 秒")
                Button { model.setTrim(nil) } label: { Label("还原", systemImage: "arrow.uturn.backward") }
                    .disabled(m.trim == nil)
                    .help("取消裁剪，保留整段录音")
            }
            .font(.callout)
            .controlSize(.small)
        }
    }

    /// `m:ss.t`
    static func time(_ seconds: Double) -> String {
        let tenths = Int((max(0, seconds) * 10).rounded())
        return String(format: "%d:%02d.%d", tenths / 600, (tenths / 10) % 60, tenths % 10)
    }
}

struct WaveformView: View {
    let envelope: [Float]
    let progress: Double
    let tint: Color
    let onSeek: (Double) -> Void

    var body: some View {
        GeometryReader { geo in
            let bars = max(1, Int(geo.size.width / 3))
            let peaks = Self.resample(envelope, to: bars)
            Canvas { context, size in
                let mid = size.height / 2
                for (i, peak) in peaks.enumerated() {
                    let x = CGFloat(i) * 3 + 1
                    let h = max(1.5, CGFloat(min(1, peak)) * size.height * 0.95)
                    let played = Double(i) / Double(peaks.count) <= progress
                    let rect = CGRect(x: x, y: mid - h / 2, width: 2, height: h)
                    context.fill(Path(roundedRect: rect, cornerRadius: 1),
                                 with: .color(played ? tint : Color.secondary.opacity(0.35)))
                }
            }
            .contentShape(Rectangle())
            .gesture(DragGesture(minimumDistance: 0).onEnded { value in
                onSeek(max(0, min(1, value.location.x / geo.size.width)))
            })
        }
        .accessibilityLabel("波形")
    }

    static func resample(_ envelope: [Float], to count: Int) -> [Float] {
        guard !envelope.isEmpty else { return [Float](repeating: 0, count: count) }
        let peak = max(envelope.max() ?? 1, 1e-6)
        return (0..<count).map { i in
            let start = i * envelope.count / count
            let end = max(start + 1, (i + 1) * envelope.count / count)
            return (envelope[start..<min(end, envelope.count)].max() ?? 0) / peak
        }
    }
}

private struct ProcessPanel: View {
    @EnvironmentObject private var model: AppModel
    let manifest: SessionManifest

    private var processing: Binding<ProcessingSettings> {
        Binding(get: { manifest.processing }, set: { p in model.updateSelected { $0.processing = p } })
    }

    private func cutoffLabel(_ c: HighBandCutoff) -> String {
        c.hertz.map { "\(Int($0 / 1000)) kHz" } ?? model.tr("自动", "Auto")
    }

    var body: some View {
        Card(title: "AI 修复", systemImage: "sparkles") {
            HStack {
                Text("按顺序运行下面开启的模型。点开每一项可以看说明和参数。")
                    .font(.callout).foregroundStyle(.secondary)
                Spacer()
                if model.isProcessing {
                    Button("取消") { model.cancelProcessing() }
                } else {
                    Button { model.process() } label: {
                        Label(model.processedURL == nil ? "处理" : "重新处理", systemImage: "play.circle")
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(model.isBusy || (model.modelState != .ready && model.modelState != .notInstalled))
                }
            }
            VStack(alignment: .leading, spacing: 0) {
                Divider()
                OptionRow(title: "强降噪（Inworld）", systemImage: "waveform.badge.minus",
                          isOn: processing.strongDenoise) {
                    Text("在 AI 修复之前先把人声从强背景噪声（音乐、人声嘈杂、风扇）里分离出来；安静环境下不需要。首次使用会下载约 256 MB 的模型。")
                        .font(.caption).foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                OptionRow(title: "AI 修复（RE-USE）", systemImage: "sparkles",
                          summary: manifest.processing.reuseVariant == "reuse-nemotron-laughter"
                              ? model.tr("保留笑声", "Keeps laughter") : nil) {
                    Picker("模型", selection: processing.reuseVariant) {
                        Text(verbatim: "RE-USE").tag("reuse-nemotron")
                        Text("RE-USE（保留笑声）").tag("reuse-nemotron-laughter")
                    }
                    .frame(width: 300)
                    Text("去除噪声和房间混响，修复麦克风音色。处理完会自动切到“AI 处理后”试听。")
                        .font(.caption).foregroundStyle(.secondary)
                }
                OptionRow(title: "高频增强（LavaSR）", systemImage: "waveform.path.ecg",
                          summary: cutoffLabel(manifest.processing.highBandCutoff),
                          isOn: processing.highBandEnabled) {
                    Picker("起始频率", selection: processing.highBandCutoff) {
                        Text("自动").tag(HighBandCutoff.auto)
                        Text(verbatim: "8 kHz").tag(HighBandCutoff.hz8000)
                        Text(verbatim: "12 kHz").tag(HighBandCutoff.hz12000)
                        Text(verbatim: "16 kHz").tag(HighBandCutoff.hz16000)
                    }
                    .frame(width: 220)
                    Text("在 AI 修复之后重新生成起始频率以上的高频，让声音更通透；低于起始频率的部分保持不变。“自动”按录音实际带宽决定。首次使用会下载约 56 MB 的模型。")
                        .font(.caption).foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            switch model.processing {
            case .idle:
                if manifest.lastExport != nil, let estimate = model.estimatedProcessingSeconds {
                    Text(verbatim: model.tr("这个会话已导出过；中间文件已清理，重新调整需要再跑一次 AI（约 \(Int(estimate.rounded())) 秒）。",
                                            "This session was exported and its intermediate files were removed; adjusting it again re-runs the AI (about \(Int(estimate.rounded())) s)."))
                        .font(.caption).foregroundStyle(.secondary)
                }
            case .running(let stage, let fraction, let remaining):
                ProgressView(value: fraction) {
                    Text(verbatim: stage)
                } currentValueLabel: {
                    let backend = model.lastBackendLabel
                    Text(verbatim: remaining.map { model.tr("\(backend) · 剩余约 \(Int($0.rounded())) 秒", "\(backend) · about \(Int($0.rounded())) s left") } ?? backend)
                }
            case .done:
                Label {
                    Text(verbatim: model.tr("处理完成 · \(model.lastBackendLabel)", "Processing complete · \(model.lastBackendLabel)"))
                } icon: { Image(systemName: "checkmark.circle.fill") }
                .foregroundStyle(.green)
            case .failed(let message):
                Text(verbatim: model.tr("处理失败：\(message)", "Processing failed: \(message)")).foregroundStyle(.red)
            }
        }
    }
}

private struct ExportPanel: View {
    @EnvironmentObject private var model: AppModel
    let manifest: SessionManifest

    var body: some View {
        Card(title: "导出", systemImage: "square.and.arrow.up") {
            HStack(spacing: 16) {
                Picker("格式", selection: Binding(
                    get: { manifest.export.format },
                    set: { f in model.updateSelected { $0.export.format = f } })) {
                    Text(verbatim: "M4A (AAC)").tag(ExportFormat.m4a)
                    Text(verbatim: "MP3").tag(ExportFormat.mp3)
                }
                .pickerStyle(.segmented)
                .frame(width: 220)
                Picker("码率", selection: Binding(
                    get: { manifest.export.bitrateKbps },
                    set: { b in model.updateSelected { $0.export.bitrateKbps = b } })) {
                    ForEach(ExportOptions.bitrates, id: \.self) { Text(verbatim: "\($0) kbps").tag($0) }
                }
                .frame(width: 180)
                Spacer()
                Button { model.export() } label: { Label("导出", systemImage: "square.and.arrow.up") }
                    .buttonStyle(.borderedProminent)
                    .disabled(model.processedURL == nil || model.isBusy)
            }
            HStack(spacing: 8) {
                Image(systemName: "folder").foregroundStyle(.secondary)
                Text("导出到").foregroundStyle(.secondary)
                Text(verbatim: model.exportFolder?.path ?? model.tr("会话文件夹", "Session folder"))
                    .lineLimit(1).truncationMode(.middle)
                Spacer()
                Button("更改…") { _ = model.chooseExportFolder() }
                if model.exportFolder != nil {
                    Button("改回会话文件夹") { model.exportFolder = nil }
                }
            }
            .font(.callout)
            if let message = model.exportMessage {
                Text(verbatim: message).font(.caption).foregroundStyle(.secondary)
            } else if model.processedURL == nil {
                Text("先完成 AI 处理再导出，或用工具栏的“一键处理并导出”。导出后只保留原始录音和成品文件。")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
    }
}

struct BatchSelectionView: View {
    @EnvironmentObject private var model: AppModel
    @Binding var confirmingDelete: Bool

    var body: some View {
        let chosen = model.sessions.filter { model.selection.contains($0.id) }
        let total = chosen.reduce(0) { $0 + $1.original.durationSeconds }
        VStack(spacing: 18) {
            Image(systemName: "square.stack.3d.up.fill").font(.system(size: 44)).foregroundStyle(.secondary)
            Text(verbatim: model.tr("已选 \(chosen.count) 条录音", "\(chosen.count) recordings selected")).font(.title2.bold())
            Text(verbatim: model.tr("总时长 \(SessionListView.duration(total))", "Total \(SessionListView.duration(total))"))
                .foregroundStyle(.secondary)
            HStack(spacing: 12) {
                Button { model.processAndExport(model.selection) } label: {
                    Label("一键处理并导出", systemImage: "wand.and.stars")
                }
                .buttonStyle(.borderedProminent)
                .disabled(model.isBusy)
                Button(role: .destructive) { confirmingDelete = true } label: {
                    Label("删除所选", systemImage: "trash")
                }
            }
            Text(verbatim: model.tr("导出到：\(model.exportFolder?.path ?? "（首次会询问文件夹）")",
                                    "Export to: \(model.exportFolder?.path ?? "(you will be asked for a folder)")"))
                .font(.caption).foregroundStyle(.secondary)
        }
        .padding(40)
    }
}
