import SwiftUI

/// One processing option: a header row with its switch and a short summary of
/// its settings; its own parameters fold out underneath it.
struct OptionRow<Content: View>: View {
    @EnvironmentObject private var model: AppModel
    let title: LocalizedStringKey
    let systemImage: String
    /// Shown beside the title while folded, e.g. "80 Hz".
    var summary: String?
    /// nil: the option is always on and has no switch.
    var isOn: Binding<Bool>?
    @ViewBuilder let content: Content
    @State private var expanded = false

    private var on: Bool { isOn?.wrappedValue ?? true }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 10) {
                Button {
                    withAnimation(.snappy(duration: 0.2)) { expanded.toggle() }
                } label: {
                    HStack(spacing: 10) {
                        Image(systemName: "chevron.right")
                            .font(.caption.bold())
                            .foregroundStyle(.secondary)
                            .rotationEffect(.degrees(expanded ? 90 : 0))
                            .frame(width: 12)
                        Image(systemName: systemImage)
                            .foregroundStyle(on ? Color.accentColor : Color.secondary)
                            .frame(width: 20)
                        Text(title).fontWeight(.medium)
                        if let summary, !expanded, on {
                            Text(verbatim: summary)
                                .font(.callout)
                                .foregroundStyle(.secondary)
                                .monospacedDigit()
                                .lineLimit(1)
                        }
                        Spacer(minLength: 8)
                    }
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .accessibilityLabel(title)
                .accessibilityValue(Text(verbatim: expanded ? model.tr("已展开", "Expanded") : model.tr("已收起", "Collapsed")))
                .accessibilityHint(Text(verbatim: model.tr("展开或收起参数", "Show or hide the settings")))
                if let isOn {
                    Toggle(title, isOn: isOn)
                        .labelsHidden()
                        .toggleStyle(.switch)
                        .controlSize(.small)
                } else {
                    Text("始终开启").font(.caption).foregroundStyle(.secondary)
                }
            }
            .padding(.vertical, 9)
            if expanded {
                VStack(alignment: .leading, spacing: 10) { content }
                    .padding(.leading, 42)
                    .padding(.bottom, 12)
                    .disabled(!on)
                    .opacity(on ? 1 : 0.5)
                    .transition(.opacity.combined(with: .move(edge: .top)))
            }
            Divider()
        }
    }
}
