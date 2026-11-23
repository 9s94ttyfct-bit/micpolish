import ModelStore
import SwiftUI

struct SettingsView: View {
    @EnvironmentObject private var model: AppModel
    @State private var installed: [String: Bool] = [:]
    @State private var usage: Int64 = 0

    var body: some View {
        Form {
            Section("会话保存位置") {
                HStack {
                    Text(model.sessionsRoot.path).lineLimit(1).truncationMode(.middle)
                    Spacer()
                    Button("更改…") {
                        let panel = NSOpenPanel()
                        panel.canChooseDirectories = true
                        panel.canChooseFiles = false
                        panel.canCreateDirectories = true
                        if panel.runModal() == .OK, let url = panel.url { model.sessionsRoot = url }
                    }
                }
            }
            Section("录音") {
                Toggle("停止录音后询问名称", isOn: $model.askNameAfterRecording)
                Text("关闭后直接用“月日 时.分 录音”作为名称，之后仍可在录音页点铅笔改名。快捷键：⌘R 开始 / 停止并保存，⌘⌫ 丢弃。")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("AI 模型（共 \(usage / 1_048_576) MB）") {
                ForEach(ModelCatalog.all, id: \.id) { d in
                    HStack {
                        VStack(alignment: .leading) {
                            Text(d.id)
                            Text(verbatim: "\(d.repo) @ \(d.revision.prefix(8)) · \(d.licenseResource.hasPrefix("OpenMDW") ? "OpenMDW-1.1" : "Apache-2.0")").font(.caption).foregroundStyle(.secondary)
                        }
                        Spacer()
                        if let license = ModelStore.licenseURL(for: d) {
                            Button("许可") { NSWorkspace.shared.open(license) }
                                .buttonStyle(.link)
                                .help("查看这个模型的许可和数据来源说明")
                        }
                        if installed[d.id] == true {
                            Button("删除") {
                                Task { try? await model.modelStore.remove(d); await reload(); await model.checkModel() }
                            }
                        } else {
                            Button("下载") { Task { await model.downloadModel(d); await reload() } }
                        }
                    }
                }
            }
            Section("处理后端") {
                Picker("RE-USE 后端", selection: $model.backendPreference) {
                    Text("自动").tag(BackendPreference.auto)
                    Text(verbatim: "MLX GPU").tag(BackendPreference.mlx)
                    Text("ANE 混合（神经引擎 + GPU）").tag(BackendPreference.aneHybrid)
                }
                .disabled(!model.aneAvailable())
                if !model.aneAvailable() {
                    Text("未安装 Core ML 包，只能用 MLX GPU。开发版用 tools/coreml/export_reuse_coreml.py 导出安装。")
                        .font(.caption).foregroundStyle(.secondary)
                } else {
                    HStack {
                        if let t = model.speedTest {
                            Text(verbatim: model.tr(
                                "本机测速：MLX \(String(format: "%.2f", t.mlxRTF)) · ANE 混合 \(String(format: "%.2f", t.aneRTF))（RTF，\(String(format: "%+.0f%%", -t.gain * 100))）→ 自动选择 \(t.aneWins ? "ANE 混合" : "MLX GPU")",
                                "This Mac: MLX \(String(format: "%.2f", t.mlxRTF)) · ANE hybrid \(String(format: "%.2f", t.aneRTF)) (RTF, \(String(format: "%+.0f%%", -t.gain * 100))) → automatic picks \(t.aneWins ? "ANE hybrid" : "MLX GPU")"))
                                .font(.caption)
                        } else {
                            Text("还没测速；“自动”暂时使用 MLX GPU。").font(.caption).foregroundStyle(.secondary)
                        }
                        Spacer()
                        if let p = model.speedTestProgress {
                            ProgressView(value: p).frame(width: 80)
                        } else {
                            Button("测速（约 1.5 分钟）") { model.runSpeedTest() }.disabled(model.isBusy)
                        }
                    }
                }
                Text(verbatim: model.tr("上次处理使用：\(model.lastBackendLabel)", "Last run used: \(model.lastBackendLabel)"))
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .frame(width: 560, height: 600)
        .task { await reload() }
    }

    private func reload() async {
        for d in ModelCatalog.all {
            installed[d.id] = await model.modelStore.installedWeightsURL(d) != nil
        }
        usage = await model.modelStore.diskUsageBytes()
    }
}
