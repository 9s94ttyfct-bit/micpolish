// SPDX-License-Identifier: Apache-2.0

import OSLog

enum ReuseLog {
    private static let subsystem = "io.github.9s94ttyfct-bit.micpolish"

    static let inference = Logger(subsystem: subsystem, category: "Inference")
    static let mamba = Logger(subsystem: subsystem, category: "Mamba")
}
