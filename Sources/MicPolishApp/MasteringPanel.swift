import Mastering
import SwiftUI

struct MasteringPanel: View {
    @EnvironmentObject private var model: AppModel
    let settings: MasteringSettings

    private func binding<T>(_ keyPath: WritableKeyPath<MasteringSettings, T>) -> Binding<T> {
        Binding(get: { settings[keyPath: keyPath] }, set: { value in
            var s = settings
            s[keyPath: keyPath] = value
            model.setMastering(s)
        })
    }

    var body: some View {
        Card(title: "母带处理", systemImage: "slider.horizontal.3") {
            VStack(alignment: .leading, spacing: 12) {
                HStack(spacing: 10) {
                    Text("预设").foregroundStyle(.secondary)
                    Menu {
                        Button("默认") { model.setMastering(.default) }
                        Button("MacBook 麦克风校正") { model.setMastering(.macbookMicCorrection) }
                        if !model.userPresetNames.isEmpty {
                            Divider()
                            ForEach(model.userPresetNames, id: \.self) { name in
                                Button { if let s = UserPresets.load(name) { model.setMastering(s) } } label: { Text(verbatim: name) }
                            }
                        }
                        Divider()
                        Button("将当前设置存为预设…") {
                            if let name = UserPresets.promptForName(model.tr("预设名称", "Preset name"),
                                                                    save: model.tr("保存", "Save"),
                                                                    cancel: model.tr("取消", "Cancel")) {
                                model.savePreset(name)
                            }
                        }
                        if let current = model.presetName(for: settings), model.userPresetNames.contains(current) {
                            Button { model.deletePreset(current) } label: {
                                Text(verbatim: model.tr("删除预设“\(current)”", "Delete preset “\(current)”"))
                            }
                        }
                    } label: {
                        Text(verbatim: model.presetName(for: settings) ?? model.tr("自定义", "Custom"))
                    }
                    .frame(width: 220)
                    .accessibilityLabel("预设")
                    Spacer()
                    Text("新录音会沿用你最后一次选择的设置。").font(.caption).foregroundStyle(.secondary)
                }
                VStack(alignment: .leading, spacing: 0) {
                    Divider()
                    OptionRow(title: "高通", systemImage: "line.diagonal.arrow",
                              summary: Self.hz(settings.highpass.frequencyHz), isOn: binding(\.highpass.enabled)) {
                        Knob("截止", value: binding(\.highpass.frequencyHz), range: 40...200, unit: "Hz")
                    }
                    OptionRow(title: "录音棚音色", systemImage: "music.mic",
                              summary: studioToneSummary, isOn: binding(\.studioTone.enabled)) {
                        Knob("强度", value: Binding(get: { settings.studioTone.amount * 100 },
                                                  set: { v in var s = settings; s.studioTone.amount = v / 100; model.setMastering(s) }),
                             range: 0...100, unit: "%")
                        if let analysis = model.studioToneAnalysis {
                            StudioToneBars(target: StudioTone.target(correction: analysis.correction,
                                                                     amount: settings.studioTone.amount))
                        }
                        Text("测量这段人声的长期频谱，按八度频段自动 EQ，向近距离录音棚话筒录的人声靠拢（参考 VCTK 录音棚语料，按音高自动选男声或女声曲线）。每段最多 ±6 dB，只改音色，不改内容。")
                            .font(.caption).foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    OptionRow(title: "均衡", systemImage: "slider.vertical.3",
                              summary: [settings.eq.lowShelf, settings.eq.peak1, settings.eq.peak2, settings.eq.highShelf]
                                  .map { String(format: "%+.1f", $0.gainDB) }.joined(separator: " / ") + " dB",
                              isOn: binding(\.eq.enabled)) {
                        HStack(alignment: .top, spacing: 20) {
                            BandControls(name: "低架", band: binding(\.eq.lowShelf), frequency: 60...400)
                            BandControls(name: "峰值 1", band: binding(\.eq.peak1), frequency: 100...1000)
                            BandControls(name: "峰值 2", band: binding(\.eq.peak2), frequency: 1000...8000)
                            BandControls(name: "高架", band: binding(\.eq.highShelf), frequency: 4000...16000)
                        }
                    }
                    OptionRow(title: "去齿音", systemImage: "mouth",
                              summary: "\(Self.hz(settings.deEsser.frequencyHz)) · \(Self.db(settings.deEsser.thresholdDB))",
                              isOn: binding(\.deEsser.enabled)) {
                        HStack(spacing: 20) {
                            Knob("频率", value: binding(\.deEsser.frequencyHz), range: 4000...9000, unit: "Hz")
                            Knob("阈值", value: binding(\.deEsser.thresholdDB), range: -50...0, unit: "dB")
                            Knob("比率", value: binding(\.deEsser.ratio), range: 1...10, unit: ":1")
                        }
                    }
                    OptionRow(title: "压缩", systemImage: "arrow.down.right.and.arrow.up.left",
                              summary: "\(Self.db(settings.compressor.thresholdDB)) · \(String(format: "%.1f:1", settings.compressor.ratio))",
                              isOn: binding(\.compressor.enabled)) {
                        HStack(spacing: 20) {
                            Knob("阈值", value: binding(\.compressor.thresholdDB), range: -40...0, unit: "dB")
                            Knob("比率", value: binding(\.compressor.ratio), range: 1...10, unit: ":1")
                            Knob("起音", value: binding(\.compressor.attackMs), range: 1...100, unit: "ms")
                            Knob("释放", value: binding(\.compressor.releaseMs), range: 20...1000, unit: "ms")
                        }
                    }
                    OptionRow(title: "响度", systemImage: "speaker.wave.2",
                              summary: String(format: "%.0f LUFS", settings.loudness.targetLUFS), isOn: binding(\.loudness.enabled)) {
                        Picker("目标", selection: binding(\.loudness.targetLUFS)) {
                            Text("−16 LUFS 播客").tag(-16.0)
                            Text("−14 LUFS 流媒体").tag(-14.0)
                            Text("−23 LUFS 广播").tag(-23.0)
                        }
                        .frame(width: 260)
                    }
                    OptionRow(title: "限幅", systemImage: "chart.line.flattrend.xyaxis",
                              summary: String(format: "%.1f dBTP", settings.limiter.ceilingDBTP), isOn: binding(\.limiter.enabled)) {
                        Knob("真峰值上限", value: binding(\.limiter.ceilingDBTP), range: -6...0, unit: "dBTP")
                    }
                }
            }
        }
    }

    private var studioToneSummary: String {
        let amount = "\(Int((settings.studioTone.amount * 100).rounded()))%"
        guard let analysis = model.studioToneAnalysis else { return amount }
        let voice = analysis.voice == .male ? model.tr("男声", "Male voice") : model.tr("女声", "Female voice")
        return "\(voice) · \(amount)"
    }

    static func hz(_ v: Double) -> String { v >= 1000 ? String(format: "%.1f kHz", v / 1000) : String(format: "%.0f Hz", v) }
    static func db(_ v: Double) -> String { String(format: "%.0f dB", v) }
}

/// The studio-tone correction per octave band, as bars above/below 0 dB.
private struct StudioToneBars: View {
    let target: [Double]
    private static let labels = ["63", "125", "250", "500", "1k", "2k", "4k", "8k", "16k"]

    var body: some View {
        HStack(alignment: .bottom, spacing: 6) {
            ForEach(target.indices, id: \.self) { i in
                VStack(spacing: 2) {
                    Text(verbatim: String(format: "%+.1f", target[i])).font(.system(size: 9).monospacedDigit())
                        .foregroundStyle(.secondary)
                    ZStack {
                        Rectangle().fill(.quaternary).frame(width: 18, height: 1)
                        let h = CGFloat(abs(target[i]) / StudioTone.maximumDB) * 22
                        RoundedRectangle(cornerRadius: 2)
                            .fill(target[i] >= 0 ? Color.accentColor : Color.orange)
                            .frame(width: 14, height: max(1, h))
                            .offset(y: target[i] >= 0 ? -h / 2 : h / 2)
                    }
                    .frame(height: 46)
                    Text(verbatim: Self.labels[i]).font(.system(size: 9)).foregroundStyle(.secondary)
                }
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(Text(verbatim: zip(Self.labels, target).map { "\($0) Hz \(String(format: "%+.1f", $1)) dB" }
            .joined(separator: ", ")))
    }
}

private struct Knob: View {
    let label: LocalizedStringKey
    @Binding var value: Double
    let range: ClosedRange<Double>
    let unit: String

    init(_ label: LocalizedStringKey, value: Binding<Double>, range: ClosedRange<Double>, unit: String) {
        self.label = label
        _value = value
        self.range = range
        self.unit = unit
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 4) {
                Text(label)
                Text(verbatim: String(format: range.upperBound >= 1000 ? "%.0f %@" : "%.1f %@", value, unit))
                    .monospacedDigit()
            }
            .font(.caption)
            Slider(value: $value, in: range).frame(width: 120)
        }
    }
}

private struct BandControls: View {
    let name: LocalizedStringKey
    @Binding var band: MasteringSettings.Band
    let frequency: ClosedRange<Double>

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(name).font(.caption.bold())
            Knob("频率", value: $band.frequencyHz, range: frequency, unit: "Hz")
            Knob("增益", value: $band.gainDB, range: -12...12, unit: "dB")
        }
    }
}

enum UserPresets {
    private static let key = "masteringPresets"

    static func names() -> [String] {
        (UserDefaults.standard.dictionary(forKey: key) ?? [:]).keys.sorted()
    }

    static func load(_ name: String) -> MasteringSettings? {
        guard let data = (UserDefaults.standard.dictionary(forKey: key) ?? [:])[name] as? Data else { return nil }
        return try? JSONDecoder().decode(MasteringSettings.self, from: data)
    }

    static func save(_ settings: MasteringSettings, as name: String) {
        var all = UserDefaults.standard.dictionary(forKey: key) ?? [:]
        all[name] = try? JSONEncoder().encode(settings)
        UserDefaults.standard.set(all, forKey: key)
    }

    static func delete(_ name: String) {
        var all = UserDefaults.standard.dictionary(forKey: key) ?? [:]
        all.removeValue(forKey: name)
        UserDefaults.standard.set(all, forKey: key)
    }

    @MainActor
    static func promptForName(_ title: String, save: String, cancel: String) -> String? {
        let alert = NSAlert()
        alert.messageText = title
        let field = NSTextField(frame: NSRect(x: 0, y: 0, width: 220, height: 24))
        alert.accessoryView = field
        alert.addButton(withTitle: save)
        alert.addButton(withTitle: cancel)
        guard alert.runModal() == .alertFirstButtonReturn else { return nil }
        let name = field.stringValue.trimmingCharacters(in: .whitespaces)
        return name.isEmpty ? nil : name
    }
}
