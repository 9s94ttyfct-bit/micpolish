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
            Section("AI 模型（共 \(usage / 1_048_576) MB）") {
                ForEach(ModelCatalog.all, id: \.id) { d in
                    HStack {
                        VStack(alignment: .leading) {
                            Text(d.id)
                            Text("\(d.repo) @ \(d.revision.prefix(8)) · OpenMDW-1.1").font(.caption).foregroundStyle(.secondary)
                        }
                        Spacer()
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
        }
        .formStyle(.grouped)
        .frame(width: 520, height: 320)
        .task { await reload() }
    }

    private func reload() async {
        for d in ModelCatalog.all {
            installed[d.id] = await model.modelStore.installedWeightsURL(d) != nil
        }
        usage = await model.modelStore.diskUsageBytes()
    }
}
