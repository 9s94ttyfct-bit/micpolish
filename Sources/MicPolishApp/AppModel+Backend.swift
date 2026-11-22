import AudioCore
import Foundation
import ModelStore
import Pipeline
import ReuseMLX

/// The user's choice of RE-USE backend.
enum BackendPreference: String, CaseIterable, Identifiable {
    /// The ANE hybrid where this Mac's speed test found it at least 15 % faster, else MLX.
    case auto
    case mlx
    case aneHybrid
    var id: String { rawValue }
}

/// One speed test of both backends on this Mac and macOS version.
struct SpeedTestResult: Codable, Equatable {
    var machine: String
    var mlxRTF: Double
    var aneRTF: Double
    var date: Date

    /// The ANE hybrid must cut the real-time factor by at least this much to be the automatic choice.
    static let requiredGain = 0.15

    var gain: Double { 1 - aneRTF / mlxRTF }
    var aneWins: Bool { gain >= Self.requiredGain }

    /// Chip and OS version: a result only counts on the machine and system it was measured on.
    static var currentMachine: String {
        var size = 0
        sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
        var chip = [CChar](repeating: 0, count: max(size, 1))
        sysctlbyname("machdep.cpu.brand_string", &chip, &size, nil, 0)
        return "\(String(cString: chip)) | \(ProcessInfo.processInfo.operatingSystemVersionString)"
    }

    private static let key = "reuseSpeedTest"

    static func load() -> SpeedTestResult? {
        guard let data = UserDefaults.standard.data(forKey: key),
              let result = try? JSONDecoder().decode(SpeedTestResult.self, from: data),
              result.machine == currentMachine else { return nil }
        return result
    }

    func save() {
        UserDefaults.standard.set(try? JSONEncoder().encode(self), forKey: Self.key)
    }
}

extension AppModel {
    static func label(for backend: ReuseBackend) -> String {
        backend == .aneHybrid ? "ANE + GPU" : "MLX GPU"
    }

    func corePackageDirectory(for descriptor: ModelDescriptor) -> URL {
        ReuseANEAccelerator.packageDirectory(modelsRoot: ModelStore.defaultRoot, modelID: descriptor.id,
                                             revision: descriptor.revision)
    }

    /// Whether the Core ML parts for this checkpoint are installed.
    func aneAvailable(for descriptor: ModelDescriptor = ModelCatalog.reuse) -> Bool {
        FileManager.default.fileExists(atPath: corePackageDirectory(for: descriptor)
            .appendingPathComponent("manifest.json").path)
    }

    /// The backend a run will ask for: MLX unless the Core ML parts are installed and
    /// the preference (or, for automatic, this Mac's speed test) selects the hybrid.
    func resolvedBackend(for descriptor: ModelDescriptor) -> ReuseBackend {
        guard aneAvailable(for: descriptor) else { return .mlx }
        switch backendPreference {
        case .mlx: return .mlx
        case .aneHybrid: return .aneHybrid
        case .auto: return speedTest?.aneWins == true ? .aneHybrid : .mlx
        }
    }

    /// Loads (on first use, compiles: about 15 s) the Core ML parts off the main thread.
    func accelerator(for descriptor: ModelDescriptor) async throws -> ReuseANEAccelerator {
        let directory = corePackageDirectory(for: descriptor)
        let accelerator = try await Task.detached(priority: .userInitiated) {
            try ReuseANEAccelerator(packageDirectory: directory)
        }.value
        return accelerator
    }

    /// Times both backends on the same 30 s signal (after a warm-up), three runs
    /// each, and keeps the medians for this Mac and macOS version.
    func runSpeedTest() {
        let descriptor = ModelCatalog.reuse
        guard aneAvailable(for: descriptor), !isBusy else { return }
        speedTestProgress = 0
        Task {
            defer { speedTestProgress = nil }
            do {
                let url = try await modelStore.install(descriptor) { _ in }
                let accelerator = try await accelerator(for: descriptor)
                let mlx = try MambaEnhancer(modelWeightsURL: url)
                let hybrid = try MambaEnhancer(modelWeightsURL: url, accelerator: accelerator)
                let signal = Self.speedTestSignal(seconds: 30)
                let warmUp = Array(signal.prefix(10 * AudioCore.sampleRate))
                _ = try await Self.time(mlx, .mlx, warmUp)
                _ = try await Self.time(hybrid, .aneHybrid, warmUp)
                var mlxTimes: [Double] = [], aneTimes: [Double] = []
                for run in 0..<3 {
                    mlxTimes.append(try await Self.time(mlx, .mlx, signal))
                    speedTestProgress = (Double(run) * 2 + 1) / 6
                    aneTimes.append(try await Self.time(hybrid, .aneHybrid, signal))
                    speedTestProgress = (Double(run) * 2 + 2) / 6
                }
                func median(_ v: [Double]) -> Double { v.sorted()[v.count / 2] }
                let result = SpeedTestResult(machine: SpeedTestResult.currentMachine,
                                             mlxRTF: median(mlxTimes) / 30, aneRTF: median(aneTimes) / 30, date: Date())
                result.save()
                speedTest = result
            } catch {
                errorMessage = tr("测速失败：\(error.localizedDescription)", "Speed test failed: \(error.localizedDescription)")
            }
        }
    }

    /// Seconds to enhance `signal` at 48 kHz with one backend.
    private static func time(_ enhancer: MambaEnhancer, _ backend: ReuseBackend, _ signal: [Float]) async throws -> Double {
        let session = try await enhancer.makeSession(sampleRate: AudioCore.sampleRate,
                                                     settings: ReuseStage.inferenceSettings, backend: backend)
        let start = ContinuousClock.now
        let step = session.recommendedInputFrameCount
        var i = 0
        while i < signal.count {
            _ = try await session.append(Array(signal[i..<min(i + step, signal.count)]))
            i += step
        }
        _ = try await session.finish()
        let elapsed = ContinuousClock.now - start
        return Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
    }

    /// Deterministic speech-like test signal: noise bursts at syllable rate.
    private static func speedTestSignal(seconds: Int) -> [Float] {
        var state: UInt64 = 0x9E3779B97F4A7C15
        return (0..<(seconds * AudioCore.sampleRate)).map { i in
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let noise = Float(Double(state >> 11) / Double(1 << 53) * 2 - 1)
            let envelope = Float(max(0, sin(2 * Double.pi * 4 * Double(i) / Double(AudioCore.sampleRate))))
            return 0.2 * noise * envelope + 0.003 * noise
        }
    }
}
