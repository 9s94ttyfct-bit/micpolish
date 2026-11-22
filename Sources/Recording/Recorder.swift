import AudioCore
import AVFoundation
import CoreAudio
import Foundation
import Sessions

public struct AudioInput: Identifiable, Hashable, Sendable {
    public let id: AudioDeviceID
    public let name: String
}

public enum RecorderError: Error, LocalizedError {
    case microphoneAccessDenied
    case deviceSelectionFailed(OSStatus)
    case notRecording

    public var errorDescription: String? {
        switch self {
        case .microphoneAccessDenied: "没有麦克风权限。请在“系统设置 › 隐私与安全性 › 麦克风”中允许 MicPolish。"
        case .deviceSelectionFailed(let status): "无法切换到所选输入设备（\(status)）。"
        case .notRecording: "当前没有在录音。"
        }
    }
}

/// Captures the selected input with AVAudioEngine (voice processing off) into
/// `<cacheRoot>/recording/<uuid>.caf`, then converts it to FLAC and hands it
/// to the session store.
@MainActor
public final class Recorder: ObservableObject {
    @Published public private(set) var isRecording = false
    @Published public private(set) var levelDBFS: Float = -120
    @Published public private(set) var elapsed: TimeInterval = 0
    /// Set when capture stopped on its own (disk full, input device change);
    /// the audio captured up to that point is saved as a session.
    @Published public private(set) var interruption: Interruption?

    public struct Interruption: Equatable {
        public let session: SessionManifest?
        public let reason: String
    }

    /// Where finished recordings go; follows the app's current sessions folder.
    public var store: SessionStore
    private let cacheRoot: URL
    private var engine: AVAudioEngine?
    private var sink: RecordingSink?
    private var timer: Timer?
    private var configurationObserver: NSObjectProtocol?
    private var configurationChanged = false
    private var stopping = false

    public init(store: SessionStore, cacheRoot: URL) {
        self.store = store
        self.cacheRoot = cacheRoot
    }

    public static func requestAccess() async -> Bool {
        await AVCaptureDevice.requestAccess(for: .audio)
    }

    public func availableInputs() -> [AudioInput] {
        Self.inputDevices()
    }

    public func start(input: AudioInput?) throws {
        guard !isRecording else { return }
        guard AVCaptureDevice.authorizationStatus(for: .audio) == .authorized else {
            throw RecorderError.microphoneAccessDenied
        }
        let engine = AVAudioEngine()
        let node = engine.inputNode
        if let input {
            var device = input.id
            let status = AudioUnitSetProperty(
                node.audioUnit!, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0,
                &device, UInt32(MemoryLayout<AudioDeviceID>.size))
            guard status == noErr else { throw RecorderError.deviceSelectionFailed(status) }
        }
        let format = node.outputFormat(forBus: 0)
        let directory = cacheRoot.appendingPathComponent("recording", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let sink = try RecordingSink(url: directory.appendingPathComponent("\(UUID().uuidString).caf"), inputFormat: format)

        Self.installCapture(on: node, format: format, sink: sink)
        engine.prepare()
        try engine.start()

        self.engine = engine
        self.sink = sink
        elapsed = 0
        interruption = nil
        configurationChanged = false
        stopping = false
        isRecording = true
        // A device change (e.g. a headset connecting) stops the engine.
        configurationObserver = NotificationCenter.default.addObserver(
            forName: .AVAudioEngineConfigurationChange, object: engine, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated { self?.configurationChanged = true }
        }
        let started = Date()
        timer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated { self?.tick(started: started) }
        }
    }

    /// Builds the tap outside any actor: Core Audio calls it on its own
    /// thread, where a main-actor isolation check would trap.
    nonisolated static func installCapture(on node: AVAudioInputNode, format: AVAudioFormat, sink: RecordingSink) {
        node.installTap(onBus: 0, bufferSize: 4096, format: format) { @Sendable buffer, _ in
            // A write error is recorded in the sink and reported by `tick`.
            try? sink.append(buffer)
        }
    }

    private func tick(started: Date) {
        guard isRecording, !stopping, let sink, let engine else { return }
        levelDBFS = sink.peakDBFS
        elapsed = Date().timeIntervalSince(started)
        let reason: String?
        if let failure = sink.failure {
            reason = "录音写入失败（\(failure.localizedDescription)），已保存出错前的部分。"
        } else if configurationChanged || !engine.isRunning {
            reason = "输入设备发生变化，录音已停止并保存。"
        } else {
            reason = nil
        }
        guard let reason else { return }
        stopping = true
        Task {
            let session = try? await stop(name: "录音（中断）")
            interruption = Interruption(session: session, reason: reason)
        }
    }

    /// Stops capture and creates a session from the recording.
    public func stop(name: String) async throws -> SessionManifest {
        guard let engine, let sink else { throw RecorderError.notRecording }
        timer?.invalidate()
        timer = nil
        if let configurationObserver { NotificationCenter.default.removeObserver(configurationObserver) }
        configurationObserver = nil
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        self.engine = nil
        self.sink = nil
        isRecording = false
        levelDBFS = -120

        let caf = try sink.finish()
        let store = store
        return try await Task.detached {
            try RecordingRecovery.adopt(caf: caf, name: name, store: store)
        }.value
    }

    static func inputDevices() -> [AudioInput] {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDevices, mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size) == noErr else { return [] }
        var ids = [AudioDeviceID](repeating: 0, count: Int(size) / MemoryLayout<AudioDeviceID>.size)
        guard AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &ids) == noErr else { return [] }

        return ids.compactMap { id in
            var streams = AudioObjectPropertyAddress(
                mSelector: kAudioDevicePropertyStreamConfiguration, mScope: kAudioDevicePropertyScopeInput,
                mElement: kAudioObjectPropertyElementMain)
            var bytes: UInt32 = 0
            guard AudioObjectGetPropertyDataSize(id, &streams, 0, nil, &bytes) == noErr, bytes > 0 else { return nil }
            let list = UnsafeMutableRawPointer.allocate(byteCount: Int(bytes), alignment: MemoryLayout<AudioBufferList>.alignment)
            defer { list.deallocate() }
            guard AudioObjectGetPropertyData(id, &streams, 0, nil, &bytes, list) == noErr else { return nil }
            let buffers = UnsafeMutableAudioBufferListPointer(list.assumingMemoryBound(to: AudioBufferList.self))
            guard buffers.reduce(0, { $0 + Int($1.mNumberChannels) }) > 0 else { return nil }

            var nameAddress = AudioObjectPropertyAddress(
                mSelector: kAudioObjectPropertyName, mScope: kAudioObjectPropertyScopeGlobal,
                mElement: kAudioObjectPropertyElementMain)
            var name: Unmanaged<CFString>?
            var nameSize = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
            guard AudioObjectGetPropertyData(id, &nameAddress, 0, nil, &nameSize, &name) == noErr,
                  let cf = name?.takeRetainedValue() else { return nil }
            return AudioInput(id: id, name: cf as String)
        }
    }
}
