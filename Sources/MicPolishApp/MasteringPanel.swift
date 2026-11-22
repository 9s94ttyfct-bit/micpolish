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
        Card(title: "母带处理", systemImage: "slider.horizontal.3", step: 2) {
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
                Stage(title: "高通", enabled: binding(\.highpass.enabled)) {
                    Knob("截止", value: binding(\.highpass.frequencyHz), range: 40...200, unit: "Hz")
                }
                Stage(title: "均衡", enabled: binding(\.eq.enabled)) {
                    BandControls(name: "低架", band: binding(\.eq.lowShelf), frequency: 60...400)
                    BandControls(name: "峰值 1", band: binding(\.eq.peak1), frequency: 100...1000)
                    BandControls(name: "峰值 2", band: binding(\.eq.peak2), frequency: 1000...8000)
                    BandControls(name: "高架", band: binding(\.eq.highShelf), frequency: 4000...16000)
                }
                Stage(title: "去齿音", enabled: binding(\.deEsser.enabled)) {
                    Knob("频率", value: binding(\.deEsser.frequencyHz), range: 4000...9000, unit: "Hz")
                    Knob("阈值", value: binding(\.deEsser.thresholdDB), range: -50...0, unit: "dB")
                    Knob("比率", value: binding(\.deEsser.ratio), range: 1...10, unit: ":1")
                }
                Stage(title: "压缩", enabled: binding(\.compressor.enabled)) {
                    Knob("阈值", value: binding(\.compressor.thresholdDB), range: -40...0, unit: "dB")
                    Knob("比率", value: binding(\.compressor.ratio), range: 1...10, unit: ":1")
                    Knob("起音", value: binding(\.compressor.attackMs), range: 1...100, unit: "ms")
                    Knob("释放", value: binding(\.compressor.releaseMs), range: 20...1000, unit: "ms")
                }
                Stage(title: "响度", enabled: binding(\.loudness.enabled)) {
                    Picker("目标", selection: binding(\.loudness.targetLUFS)) {
                        Text("−16 LUFS 播客").tag(-16.0)
                        Text("−14 LUFS 流媒体").tag(-14.0)
                        Text("−23 LUFS 广播").tag(-23.0)
                    }
                    .frame(width: 240)
                }
                Stage(title: "限幅", enabled: binding(\.limiter.enabled)) {
                    Knob("真峰值上限", value: binding(\.limiter.ceilingDBTP), range: -6...0, unit: "dBTP")
                }
            }
        }
    }
}

private struct Stage<Content: View>: View {
    let title: LocalizedStringKey
    @Binding var enabled: Bool
    @ViewBuilder let content: Content

    var body: some View {
        HStack(alignment: .top, spacing: 16) {
            Toggle(title, isOn: $enabled).toggleStyle(.switch).frame(width: 130, alignment: .leading)
            HStack(spacing: 16) { content }
                .disabled(!enabled)
                .opacity(enabled ? 1 : 0.45)
            Spacer(minLength: 0)
        }
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
