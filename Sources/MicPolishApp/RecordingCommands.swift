import Recording
import SwiftUI

/// The Recording menu: ⌘R starts or stops-and-saves, ⌘⌫ discards.
struct RecordingCommands: Commands {
    @ObservedObject var model: AppModel
    @ObservedObject var recorder: Recorder

    var body: some Commands {
        CommandMenu(Text(verbatim: model.tr("录音", "Recording"))) {
            Button(recorder.isRecording ? model.tr("停止并保存", "Stop and Save") : model.tr("开始录音", "Start Recording")) {
                model.toggleRecording()
            }
            .keyboardShortcut("r", modifiers: .command)

            Button(model.tr("丢弃录音…", "Discard Recording…")) { model.requestDiscard() }
                .keyboardShortcut(.delete, modifiers: .command)
                .disabled(!recorder.isRecording)

            Divider()

            Toggle(model.tr("停止后询问名称", "Ask for a Name After Stopping"), isOn: $model.askNameAfterRecording)
        }
    }
}
