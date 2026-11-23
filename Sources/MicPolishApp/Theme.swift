import AppKit
import SwiftUI

/// Visual tokens. The palette comes from the app icon: indigo through violet
/// to rose. The accent is reserved for what the user acts on or listens to.
enum Brand {
    /// Interactive accent (icon indigo).
    static let accent = Color(red: 0.333, green: 0.302, blue: 0.886)
    /// The processed sound: waveforms and the live meter.
    static let waveColors = [Color(red: 0.31, green: 0.27, blue: 0.90), Color(red: 0.55, green: 0.36, blue: 0.96),
                             Color(red: 0.93, green: 0.42, blue: 0.55)]
    /// Raised surfaces on the window background.
    static let surface = Color(nsColor: .controlBackgroundColor)
    static let hairline = Color.primary.opacity(0.08)
}

/// A raised surface for one block of the session page. `prominent` is for the
/// player, the page's main object.
struct Panel<Content: View>: View {
    var prominent = false
    @ViewBuilder let content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 14) { content }
            .padding(prominent ? 20 : 18)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(RoundedRectangle(cornerRadius: prominent ? 18 : 12, style: .continuous).fill(Brand.surface))
            .overlay(RoundedRectangle(cornerRadius: prominent ? 18 : 12, style: .continuous).strokeBorder(Brand.hairline))
            .shadow(color: .black.opacity(prominent ? 0.06 : 0), radius: 12, y: 4)
    }
}

/// One line of secondary explanation under a control.
struct Hint: View {
    let text: LocalizedStringKey
    init(_ text: LocalizedStringKey) { self.text = text }

    var body: some View {
        Text(text).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
    }
}
