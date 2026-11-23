import Recording
import SwiftUI

/// Shown in place of the session while recording: elapsed time, a scrolling
/// live waveform of the last 20 s, and stop / discard.
struct LiveRecordingView: View {
    @EnvironmentObject private var model: AppModel
    @EnvironmentObject private var recorder: Recorder

    var body: some View {
        VStack(spacing: 22) {
            HStack(spacing: 10) {
                Circle().fill(.red).frame(width: 10, height: 10)
                    .opacity(Int(recorder.elapsed * 2) % 2 == 0 ? 1 : 0.35)
                Text("正在录音").font(.headline)
                Spacer()
                Text(verbatim: model.recordingInput?.name ?? model.tr("系统默认输入", "System default input"))
                    .font(.callout).foregroundStyle(.secondary)
            }
            Text(SessionListView.duration(recorder.elapsed))
                .font(.system(size: 44, weight: .light, design: .rounded).monospacedDigit())
            LiveWaveform(envelope: recorder.liveEnvelope, capacity: Recorder.liveEnvelopeCapacity)
                .frame(height: 140)
                .accessibilityLabel("实时波形")
            LevelMeter(dBFS: recorder.levelDBFS).frame(maxWidth: 420)
            if recorder.levelDBFS > -1 {
                Text("电平过高，可能削波").font(.caption).foregroundStyle(.red)
            }
            HStack(spacing: 14) {
                Button(role: .destructive) { model.requestDiscard() } label: {
                    Label("丢弃", systemImage: "trash").frame(minWidth: 110)
                }
                .controlSize(.large)
                .help("⌘⌫")
                Button { model.toggleRecording() } label: {
                    Label("停止并保存", systemImage: "stop.fill").frame(minWidth: 130)
                }
                .controlSize(.large)
                .buttonStyle(.borderedProminent)
                .tint(.red)
                .help("⌘R")
            }
            Text(model.askNameAfterRecording
                 ? "停止后会先保存，再请你命名。快捷键：⌘R 开始 / 停止，⌘⌫ 丢弃。"
                 : "停止后直接用默认名称保存（可在设置里改）。快捷键：⌘R 开始 / 停止，⌘⌫ 丢弃。")
                .font(.caption).foregroundStyle(.secondary)
        }
        .padding(28)
        .frame(maxWidth: 900)
        .background(RoundedRectangle(cornerRadius: 16).fill(Color(nsColor: .controlBackgroundColor)))
        .overlay(RoundedRectangle(cornerRadius: 16).strokeBorder(.quaternary))
        .padding(24)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

/// Scrolling peak bars, newest at the right, on a fixed (not normalized)
/// scale so quiet and loud passages look different.
struct LiveWaveform: View {
    let envelope: [Float]
    let capacity: Int

    var body: some View {
        Canvas { context, size in
            let barWidth = size.width / CGFloat(capacity)
            let mid = size.height / 2
            context.fill(Path(CGRect(x: 0, y: mid - 0.5, width: size.width, height: 1)),
                         with: .color(.secondary.opacity(0.25)))
            for (i, peak) in envelope.enumerated() {
                // Square-root scale: −40 dBFS still shows as a tenth of the height.
                let h = max(1.5, CGFloat(min(1, peak).squareRoot()) * size.height * 0.95)
                let x = size.width - CGFloat(envelope.count - i) * barWidth
                let rect = CGRect(x: x + barWidth * 0.15, y: mid - h / 2, width: max(1, barWidth * 0.7), height: h)
                context.fill(Path(roundedRect: rect, cornerRadius: 1),
                             with: .color(peak >= 0.89 ? .red : .accentColor))
            }
        }
    }
}
