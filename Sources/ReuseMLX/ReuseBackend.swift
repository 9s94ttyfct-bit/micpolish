// SPDX-License-Identifier: Apache-2.0

/// Which implementation runs the RE-USE convolutional front and back ends.
public enum ReuseBackend: String, Sendable, Codable, CaseIterable {
    /// Every layer in MLX on the GPU (the baseline and the fallback).
    case mlx
    /// Dense encoder and mask/phase decoders in Core ML (FP16,
    /// `cpuAndNeuralEngine`); the 30 Mamba blocks stay in MLX.
    case aneHybrid
}

/// How many model windows of one session ran on each backend.
public struct ReuseBackendUsage: Sendable, Equatable {
    public var aneHybridWindows = 0
    public var mlxWindows = 0

    public init(aneHybridWindows: Int = 0, mlxWindows: Int = 0) {
        self.aneHybridWindows = aneHybridWindows
        self.mlxWindows = mlxWindows
    }

    /// `.aneHybrid` when any window used Core ML. A hybrid session still runs
    /// its final partial window in MLX.
    public var backend: ReuseBackend { aneHybridWindows > 0 ? .aneHybrid : .mlx }

    mutating func record(_ backend: ReuseBackend) {
        switch backend {
        case .mlx: mlxWindows += 1
        case .aneHybrid: aneHybridWindows += 1
        }
    }
}

/// A RE-USE enhancement session that reports the backends it actually used.
public protocol ReuseEnhancementSession: SpeechEnhancementSession {
    /// Backend the session was asked to use.
    var requestedBackend: ReuseBackend { get }
    /// Windows processed so far, by backend.
    var backendUsage: ReuseBackendUsage { get async }
}
