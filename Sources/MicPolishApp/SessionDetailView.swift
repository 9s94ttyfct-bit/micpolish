import Export
import Mastering
import Preview
import Sessions
import SwiftUI

/// The three steps of the workflow; one is shown at a time under the player.
enum WorkflowStep: Int, CaseIterable, Identifiable {
    case restore, master, export
    var id: Int { rawValue }
}

struct SessionDetailView: View {
    @EnvironmentObject private var model: AppModel
    @State private var step: WorkflowStep = .restore

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                if let m = model.selected {
                    Header(manifest: m)
                    PlayerPanel()
                    StepBar(step: $step)
                    switch step {
                    case .restore: ProcessPanel(manifest: m)
                    case .master: MasteringPanel(settings: m.mastering)
                    case .export: ExportPanel(manifest: m)
                    }
                }
            }
            .padding(.horizontal, 28)
            .padding(.vertical, 22)
            .frame(maxWidth: 1080)
            .frame(maxWidth: .infinity)
        }
    }
}

/// Switches between the workflow steps; each shows whether it is done and
/// what it is currently set to.
private struct StepBar: View {
    @EnvironmentObject private var model: AppModel
    @Binding var step: WorkflowStep

    var body: some View {
        HStack(spacing: 0) {
            ForEach(WorkflowStep.allCases) { s in
                let selected = s == step
                Button { step = s } label: {
                    HStack(spacing: 10) {
                        ZStack {
                            Circle().fill(selected ? Brand.accent : Color.secondary.opacity(0.14))
                            if done(s) {
                                Image(systemName: "checkmark").font(.system(size: 10, weight: .bold))
                            } else {
                                Text(verbatim: "\(s.rawValue + 1)").font(.system(size: 11, weight: .semibold))
                            }
                        }
                        .foregroundStyle(selected ? .white : .secondary)
                        .frame(width: 22, height: 22)
                        VStack(alignment: .leading, spacing: 1) {
                            Text(verbatim: title(s)).font(.callout.weight(selected ? .semibold : .medium))
                                .foregroundStyle(selected ? .primary : .secondary)
                            Text(verbatim: subtitle(s)).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                        }
                        Spacer(minLength: 0)
                    }
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(
                        RoundedRectangle(cornerRadius: 9, style: .continuous)
                            .fill(selected ? Brand.surface : .clear)
                            .shadow(color: .black.opacity(selected ? 0.08 : 0), radius: 3, y: 1)
                    )
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .accessibilityLabel(Text(verbatim: title(s)))
                .accessibilityValue(Text(verbatim: subtitle(s)))
                .accessibilityAddTraits(selected ? .isSelected : [])
            }
        }
        .padding(4)
        .background(RoundedRectangle(cornerRadius: 12, style: .continuous).fill(Color.secondary.opacity(0.09)))
    }

    private func title(_ s: WorkflowStep) -> String {
        switch s {
        case .restore: model.tr("AI 修复", "AI Restoration")
        case .master: model.tr("母带处理", "Mastering")
        case .export: model.tr("导出", "Export")
        }
    }

    /// What the step is set to right now, so the bar doubles as a summary.
    private func subtitle(_ s: WorkflowStep) -> String {
        guard let m = model.selected else { return "" }
        switch s {
        case .restore:
            var parts: [String] = []
            if m.processing.strongDenoise { parts.append(model.tr("强降噪", "Strong denoise")) }
            parts.append("RE-USE")
            if m.processing.highBandEnabled { parts.append(model.tr("高频增强", "High band")) }
            return parts.joined(separator: " + ")
        case .master:
            return model.presetName(for: m.mastering) ?? model.tr("自定义", "Custom")
        case .export:
            let folder = model.exportFolder?.lastPathComponent ?? model.tr("会话文件夹", "session folder")
            return model.tr("\(m.export.format.rawValue.uppercased()) 到 \(folder)", "\(m.export.format.rawValue.uppercased()) to \(folder)")
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
    @FocusState private var nameFocused: Bool

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 12) {
            VStack(alignment: .leading, spacing: 5) {
                if renaming {
                    TextField("名称", text: $name)
                        .textFieldStyle(.plain)
                        .font(.system(size: 24, weight: .semibold))
                        .focused($nameFocused)
                        .onSubmit(commit)
                        .onExitCommand { renaming = false }
                        .onChange(of: nameFocused) { _, focused in if !focused { commit() } }
                } else {
                    Text(manifest.name)
                        .font(.system(size: 24, weight: .semibold))
                        .lineLimit(1)
                        .onTapGesture(count: 2, perform: startRenaming)
                        .help("双击重命名")
                }
                HStack(spacing: 14) {
                    Text(manifest.createdAt, format: .dateTime.month().day().hour().minute())
                    Text(verbatim: SessionListView.duration(manifest.original.durationSeconds)).monospacedDigit()
                    Text(verbatim: "\(manifest.original.sampleRate / 1000) kHz")
                    if manifest.original.imported { Text("导入的文件") }
                    if manifest.lastExport != nil {
                        Label("已导出", systemImage: "checkmark.circle.fill").foregroundStyle(.green)
                    }
                }
                .font(.callout)
                .foregroundStyle(.secondary)
            }
            Spacer()
            Menu {
                Button("重命名…", action: startRenaming)
                    .disabled(model.isBusy)
                Button("在访达中显示") { model.revealInFinder() }
                Divider()
                Button("删除…", role: .destructive) { model.confirmingDelete = true }
                    .disabled(model.isBusy)
            } label: {
                Image(systemName: "ellipsis.circle").font(.title2)
            }
            .menuStyle(.borderlessButton)
            .menuIndicator(.hidden)
            .fixedSize()
            .help("更多操作")
            .accessibilityLabel("更多操作")
        }
    }

    private func startRenaming() {
        guard !model.isBusy else { return }
        name = manifest.name
        renaming = true
        nameFocused = true
    }

    private func commit() {
        guard renaming else { return }
        renaming = false
        if name != manifest.name { model.rename(manifest.id, to: name) }
    }
}

/// The page's main object: the waveform with trim handles, the transport and
/// the A/B/C source switch.
private struct PlayerPanel: View {
    @EnvironmentObject private var model: AppModel
    @EnvironmentObject private var preview: PreviewPlayer

    var body: some View {
        Panel(prominent: true) {
            HStack(spacing: 12) {
                SourceSwitcher()
                Spacer()
                TrimControls()
            }
            ZStack {
                WaveformView(
                    envelope: preview.envelopes[preview.source == .original ? .original : .processed] ?? preview.envelopes[.original] ?? [],
                    progress: preview.duration > 0 ? preview.position / preview.duration : 0,
                    processed: preview.source != .original
                ) { fraction in
                    preview.seek(to: fraction * preview.duration)
                }
                if let m = model.selected, preview.duration > 0 {
                    TrimOverlay(duration: preview.duration, trim: m.trim) { model.setTrim($0) }
                }
            }
            .frame(height: 104)
            HStack(spacing: 14) {
                Button {
                    preview.isPlaying ? preview.pause() : preview.play()
                } label: {
                    Image(systemName: preview.isPlaying ? "pause.fill" : "play.fill")
                        .font(.system(size: 15, weight: .semibold))
                        .foregroundStyle(.white)
                        .frame(width: 38, height: 38)
                        .background(Circle().fill(Brand.accent))
                        .contentShape(Circle())
                }
                .buttonStyle(.plain)
                .help("播放 / 暂停")
                .accessibilityLabel(preview.isPlaying ? "暂停" : "播放")
                Text(verbatim: "\(SessionListView.duration(preview.position)) / \(SessionListView.duration(preview.duration))")
                    .font(.system(.title3, design: .rounded).monospacedDigit())
                Spacer()
                Text(sourceNote).font(.callout).foregroundStyle(.secondary).multilineTextAlignment(.trailing)
            }
        }
    }

    /// What the selected source is, in one line.
    private var sourceNote: LocalizedStringKey {
        switch preview.source {
        case .original: model.processedURL == nil ? "原声。AI 处理后可以对比“AI 处理后”和“成品”。" : "麦克风录到的原声"
        case .processed: "只有 AI 修复，还没做母带"
        case .mastered: "AI 修复加母带处理，导出的就是它"
        }
    }
}

/// Original / Processed / Final. The last two need processed audio.
private struct SourceSwitcher: View {
    @EnvironmentObject private var model: AppModel
    @EnvironmentObject private var preview: PreviewPlayer

    var body: some View {
        HStack(spacing: 2) {
            item(.original, "原始")
            item(.processed, "AI 处理后")
            item(.mastered, "成品")
        }
        .padding(3)
        .background(Capsule().fill(Color.secondary.opacity(0.1)))
        .help("三路按响度对齐，比较的是音质而不是音量")
    }

    private func item(_ source: PreviewSource, _ title: LocalizedStringKey) -> some View {
        let selected = preview.source == source
        let enabled = source == .original || model.processedURL != nil
        return Button { preview.source = source } label: {
            Text(title)
                .font(.callout.weight(selected ? .semibold : .regular))
                .foregroundStyle(selected ? .white : (enabled ? .primary : .secondary))
                .padding(.horizontal, 14)
                .padding(.vertical, 5)
                .background(Capsule().fill(selected ? Brand.accent : .clear))
                .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .disabled(!enabled)
        .help(enabled ? "" : "先完成 AI 修复")
        .accessibilityLabel(title)
        .accessibilityAddTraits(selected ? .isSelected : [])
    }
}

/// The kept span and the quick-trim actions above the waveform.
private struct TrimControls: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        if let m = model.selected {
            HStack(spacing: 8) {
                if let trim = m.trim {
                    Label {
                        Text(verbatim: "\(Self.time(trim.start)) – \(Self.time(trim.end))").monospacedDigit()
                    } icon: { Image(systemName: "scissors") }
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .help("只保留这一段；拖动波形两端的手柄调整")
                }
                Button { model.autoTrim() } label: { Image(systemName: "wand.and.stars") }
                    .help("自动去掉首尾静音（两端各留 0.25 秒）")
                    .accessibilityLabel("自动去首尾静音")
                Button { model.setTrim(nil) } label: { Image(systemName: "arrow.uturn.backward") }
                    .disabled(m.trim == nil)
                    .help("取消裁剪，保留整段录音")
                    .accessibilityLabel("还原")
            }
            .buttonStyle(.borderless)
            .font(.body)
        }
    }

    /// `m:ss.t`
    static func time(_ seconds: Double) -> String {
        let tenths = Int((max(0, seconds) * 10).rounded())
        return String(format: "%d:%02d.%d", tenths / 600, (tenths / 10) % 60, tenths % 10)
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
            .accessibilityValue(Text(verbatim: TrimControls.time(value)))
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

struct WaveformView: View {
    let envelope: [Float]
    let progress: Double
    /// The processed sound is drawn in the brand gradient, the original in grey.
    let processed: Bool
    let onSeek: (Double) -> Void

    var body: some View {
        GeometryReader { geo in
            let step: CGFloat = 4
            let bars = max(1, Int(geo.size.width / step))
            let peaks = Self.resample(envelope, to: bars)
            Canvas { context, size in
                let mid = size.height / 2
                var played = Path(), rest = Path()
                for (i, peak) in peaks.enumerated() {
                    let h = max(2, CGFloat(min(1, peak)) * size.height * 0.96)
                    let rect = CGRect(x: CGFloat(i) * step + 0.75, y: mid - h / 2, width: step - 1.5, height: h)
                    if Double(i) / Double(peaks.count) <= progress {
                        played.addRoundedRect(in: rect, cornerSize: CGSize(width: 1.25, height: 1.25))
                    } else {
                        rest.addRoundedRect(in: rect, cornerSize: CGSize(width: 1.25, height: 1.25))
                    }
                }
                if processed {
                    // Played part in full colour, the rest faded, so the gradient
                    // marks the processed sound even before playback.
                    let gradient = GraphicsContext.Shading.linearGradient(
                        Gradient(colors: Brand.waveColors), startPoint: .zero, endPoint: CGPoint(x: size.width, y: 0))
                    context.fill(played, with: gradient)
                    var faded = context
                    faded.opacity = 0.32
                    faded.fill(rest, with: gradient)
                } else {
                    context.fill(rest, with: .color(.secondary.opacity(0.28)))
                    context.fill(played, with: .color(.primary.opacity(0.7)))
                }
            }
            .contentShape(Rectangle())
            .gesture(DragGesture(minimumDistance: 0).onEnded { value in
                onSeek(max(0, min(1, value.location.x / geo.size.width)))
            })
        }
        .accessibilityLabel("波形")
        .accessibilityHint("点击跳到该位置")
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

/// The primary action of a step, with its status underneath.
private struct ActionRow<Status: View>: View {
    let summary: LocalizedStringKey
    @ViewBuilder let action: () -> AnyView
    @ViewBuilder let status: Status

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .center, spacing: 16) {
                Text(summary).font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                Spacer(minLength: 12)
                action()
            }
            status
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
        Panel {
            ActionRow(summary: "去掉噪声和房间回声，修复麦克风音色。开启的模型会按下面的顺序运行。") {
                AnyView(Group {
                    if model.isProcessing {
                        Button("取消") { model.cancelProcessing() }.controlSize(.large)
                    } else {
                        Button { model.process() } label: {
                            Label(model.processedURL == nil ? "开始处理" : "重新处理", systemImage: "sparkles")
                                .padding(.horizontal, 6)
                        }
                        .controlSize(.large)
                        .buttonStyle(.borderedProminent)
                        .disabled(model.isBusy || (model.modelState != .ready && model.modelState != .notInstalled))
                    }
                })
            } status: {
                status
            }
            VStack(alignment: .leading, spacing: 0) {
                Divider()
                OptionRow(title: "强降噪（Inworld）", systemImage: "waveform.badge.minus",
                          isOn: processing.strongDenoise) {
                    Hint("先把人声从强背景声（音乐、人群、风扇）里分离出来，再做 AI 修复。安静的房间不需要。首次使用下载约 256 MB 的模型。")
                }
                OptionRow(title: "AI 修复（RE-USE）", systemImage: "sparkles",
                          summary: manifest.processing.reuseVariant == "reuse-nemotron-laughter"
                              ? model.tr("保留笑声", "Keeps laughter") : nil) {
                    Picker("模型", selection: processing.reuseVariant) {
                        Text("标准").tag("reuse-nemotron")
                        Text("保留笑声").tag("reuse-nemotron-laughter")
                    }
                    .pickerStyle(.segmented)
                    .frame(width: 240)
                    Hint("去噪、去混响、修复音色。录音里有笑声时选“保留笑声”，避免笑声被当成噪声去掉。")
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
                    .pickerStyle(.segmented)
                    .frame(width: 320)
                    Hint("重新生成起始频率以上的高频，低于它的部分不变。适合电话、语音消息这类低采样率的导入文件；MacBook 录音一般不需要。首次使用下载约 56 MB 的模型。")
                }
            }
        }
    }

    @ViewBuilder private var status: some View {
        switch model.processing {
        case .idle:
            if manifest.lastExport != nil, let estimate = model.estimatedProcessingSeconds {
                Hint(LocalizedStringKey(model.tr("已导出过，中间文件已清理；再次试听或修改需要重新处理（约 \(Int(estimate.rounded())) 秒）。",
                                                 "Already exported and its intermediate files were removed; listening again or changing it re-runs the AI (about \(Int(estimate.rounded())) s).")))
            }
        case .running(let stage, let fraction, let remaining):
            ProgressView(value: fraction) {
                Text(verbatim: stage).font(.callout)
            } currentValueLabel: {
                let backend = model.lastBackendLabel
                Text(verbatim: remaining.map { model.tr("\(backend)，还剩约 \(Int($0.rounded())) 秒", "\(backend), about \(Int($0.rounded())) s left") } ?? backend)
            }
            .tint(Brand.accent)
        case .done:
            Label {
                Text(verbatim: model.tr("处理完成（\(model.lastBackendLabel)）", "Processed (\(model.lastBackendLabel))"))
            } icon: { Image(systemName: "checkmark.circle.fill") }
            .font(.callout)
            .foregroundStyle(.green)
        case .failed(let message):
            Label {
                Text(verbatim: model.tr("处理失败：\(message)", "Processing failed: \(message)"))
            } icon: { Image(systemName: "exclamationmark.triangle.fill") }
            .font(.callout)
            .foregroundStyle(.red)
        }
    }
}

private struct ExportPanel: View {
    @EnvironmentObject private var model: AppModel
    let manifest: SessionManifest

    var body: some View {
        Panel {
            ActionRow(summary: model.processedURL == nil
                      ? "导出需要先完成 AI 修复。工具栏的“一键处理并导出”可以一步完成。"
                      : "按母带设置渲染成品并编码。导出后会话里只保留原始录音和这个成品。") {
                AnyView(Button { model.export() } label: {
                    Label("导出", systemImage: "square.and.arrow.up").padding(.horizontal, 6)
                }
                .controlSize(.large)
                .buttonStyle(.borderedProminent)
                .disabled(model.processedURL == nil || model.isBusy))
            } status: {
                if let message = model.exportMessage {
                    Label { Text(verbatim: message) } icon: { Image(systemName: "checkmark.circle.fill").foregroundStyle(.green) }
                        .font(.callout)
                }
            }
            Divider()
            Grid(alignment: .leading, horizontalSpacing: 18, verticalSpacing: 14) {
                GridRow {
                    Text("格式").foregroundStyle(.secondary)
                    Picker("格式", selection: Binding(
                        get: { manifest.export.format },
                        set: { f in model.updateSelected { $0.export.format = f } })) {
                        Text(verbatim: "M4A (AAC)").tag(ExportFormat.m4a)
                        Text(verbatim: "MP3").tag(ExportFormat.mp3)
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    .frame(width: 220)
                }
                GridRow {
                    Text("码率").foregroundStyle(.secondary)
                    Picker("码率", selection: Binding(
                        get: { manifest.export.bitrateKbps },
                        set: { b in model.updateSelected { $0.export.bitrateKbps = b } })) {
                        ForEach(ExportOptions.bitrates, id: \.self) { Text(verbatim: "\($0) kbps").tag($0) }
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    .frame(width: 280)
                }
                GridRow {
                    Text("位置").foregroundStyle(.secondary)
                    HStack(spacing: 10) {
                        Label {
                            Text(verbatim: model.exportFolder?.path ?? model.tr("每条录音自己的会话文件夹", "Each recording's session folder"))
                                .lineLimit(1).truncationMode(.middle)
                        } icon: { Image(systemName: "folder") }
                        Button("更改…") { _ = model.chooseExportFolder() }
                        if model.exportFolder != nil {
                            Button("改回会话文件夹") { model.exportFolder = nil }
                        }
                    }
                }
            }
            .font(.callout)
        }
    }
}

struct BatchSelectionView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        let chosen = model.sessions.filter { model.selection.contains($0.id) }
        let total = chosen.reduce(0) { $0 + $1.original.durationSeconds }
        VStack(spacing: 16) {
            Image(systemName: "square.stack.3d.up").font(.system(size: 40, weight: .light)).foregroundStyle(Brand.accent)
            Text(verbatim: model.tr("已选 \(chosen.count) 条录音", "\(chosen.count) recordings selected")).font(.title2.weight(.semibold))
            Text(verbatim: model.tr("共 \(SessionListView.duration(total))", "\(SessionListView.duration(total)) in total"))
                .foregroundStyle(.secondary)
            HStack(spacing: 12) {
                Button { model.processAndExport(model.selection) } label: {
                    Label("全部处理并导出", systemImage: "wand.and.stars")
                }
                .controlSize(.large)
                .buttonStyle(.borderedProminent)
                .disabled(model.isBusy)
                Button(role: .destructive) { model.confirmingDelete = true } label: {
                    Label("删除所选…", systemImage: "trash")
                }
                .controlSize(.large)
            }
            Hint(LocalizedStringKey(model.tr("导出到：\(model.exportFolder?.path ?? "第一次会请你选择文件夹")",
                                             "Export to: \(model.exportFolder?.path ?? "you will be asked for a folder the first time")")))
        }
        .padding(40)
    }
}
