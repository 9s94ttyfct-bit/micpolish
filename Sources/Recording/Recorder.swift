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
    case noInputDevice
    case notRecording
    /// The input delivered no audio at all, so there is nothing to save.
    case nothingCaptured

    public var errorDescription: String? {
        switch self {
        case .microphoneAccessDenied: "没有麦克风权限。请在“系统设置 › 隐私与安全性 › 麦克风”中允许 MicPolish。"
        case .deviceSelectionFailed(let status): "无法打开所选输入设备（\(status)）。"
        case .noInputDevice: "没有可用的输入设备。"
        case .notRecording: "当前没有在录音。"
        case .nothingCaptured: "没有录到声音。"
        }
    }
}

/// Captures one concrete input device (the chosen one, or the default input at
/// the moment recording starts) into `<cacheRoot>/recording/<uuid>.caf`, then
/// converts it to FLAC and hands it to the session store. Other devices
/// connecting or becoming the default do not affect a running recording; only
/// the recorded device disconnecting or changing its sample rate stops it.
@MainActor
public final class Recorder: ObservableObject {
    @Published public private(set) var isRecording = false
    @Published public private(set) var levelDBFS: Float = -120
    @Published public private(set) var elapsed: TimeInterval = 0
    /// Peaks of the most recent 50 ms buckets (newest last, at most
    /// `liveEnvelopeCapacity`), for a scrolling waveform while recording.
    @Published public private(set) var liveEnvelope: [Float] = []
    public static let liveEnvelopeCapacity = 400
    /// Input devices, kept current as devices connect and disconnect.
    @Published public private(set) var inputs: [AudioInput] = []
    /// Set when capture stopped on its own (disk full, input device lost);
    /// the audio captured up to that point is saved as a session.
    @Published public private(set) var interruption: Interruption?

    public struct Interruption: Equatable {
        public enum Cause: Equatable {
            case writeFailed(String)
            case deviceLost
        }
        /// nil when nothing had been captured yet (or saving failed).
        public let session: SessionManifest?
        public let cause: Cause
    }

    /// Where finished recordings go; follows the app's current sessions folder.
    public var store: SessionStore
    private let cacheRoot: URL
    private var capture: DeviceCapture?
    private var sink: RecordingSink?
    private var timer: Timer?
    private var devicesListener: AudioObjectPropertyListenerBlock?
    /// Name given to a recording saved after an interruption.
    public var interruptedName = "Recording (interrupted)"

    public init(store: SessionStore, cacheRoot: URL) {
        self.store = store
        self.cacheRoot = cacheRoot
        inputs = Self.inputDevices()
        let listener: AudioObjectPropertyListenerBlock = { _, _ in
            Task { @MainActor [weak self] in self?.inputs = Self.inputDevices() }
        }
        var address = Self.devicesAddress
        if AudioObjectAddPropertyListenerBlock(AudioObjectID(kAudioObjectSystemObject), &address, .main, listener) == noErr {
            devicesListener = listener
        }
    }

    private static var devicesAddress: AudioObjectPropertyAddress {
        AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyDevices, mScope: kAudioObjectPropertyScopeGlobal,
                                   mElement: kAudioObjectPropertyElementMain)
    }

    public static func requestAccess() async -> Bool {
        await AVCaptureDevice.requestAccess(for: .audio)
    }

    /// Starts recording `input`, or the current default input when nil (or
    /// when the chosen device is no longer connected).
    public func start(input: AudioInput?) throws {
        guard !isRecording else { return }
        guard AVCaptureDevice.authorizationStatus(for: .audio) == .authorized else {
            throw RecorderError.microphoneAccessDenied
        }
        let current = Self.inputDevices()
        let chosen = input.flatMap { wanted in current.first { $0.id == wanted.id } }
        guard let deviceID = chosen?.id ?? DeviceCapture.defaultInputDevice() else { throw RecorderError.noInputDevice }

        let directory = cacheRoot.appendingPathComponent("recording", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent("\(UUID().uuidString).caf")
        // The sink is created once the device's rate is known; the render
        // callback reaches it through this box.
        let box = SinkBox()
        let capture = try DeviceCapture(deviceID: deviceID) { buffers, frames in
            // A write error is recorded in the sink and reported by `tick`.
            try? box.sink?.append(buffers, frames: frames)
        }
        let sink = try RecordingSink(url: url, inputSampleRate: Int(capture.sampleRate.rounded()))
        box.sink = sink
        do {
            try capture.start()
        } catch {
            _ = try? sink.finish()
            throw error
        }

        self.capture = capture
        self.sink = sink
        elapsed = 0
        liveEnvelope = []
        interruption = nil
        isRecording = true
        let started = Date()
        timer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated { self?.tick(started: started) }
        }
    }

    /// Lets the render callback reach a sink created after the capture unit.
    private final class SinkBox: @unchecked Sendable {
        var sink: RecordingSink?
    }

    private func tick(started: Date) {
        guard isRecording, let sink, let capture else { return }
        levelDBFS = sink.peakDBFS
        elapsed = Date().timeIntervalSince(started)
        let buckets = sink.takeEnvelope()
        if !buckets.isEmpty {
            var envelope = liveEnvelope + buckets
            if envelope.count > Self.liveEnvelopeCapacity { envelope.removeFirst(envelope.count - Self.liveEnvelopeCapacity) }
            liveEnvelope = envelope
        }
        let cause: Interruption.Cause?
        if let failure = sink.failure {
            cause = .writeFailed(failure.localizedDescription)
        } else if capture.deviceLost {
            cause = .deviceLost
        } else {
            cause = nil
        }
        guard let cause else { return }
        // End capture right here, on the main actor, so a stop or discard the
        // user triggers meanwhile finds nothing left to stop and this
        // interruption is reported exactly once, for the recording it ended.
        guard let ended = endCapture() else { return }
        let name = interruptedName
        Task {
            let session = try? await save(ended, name: name)
            interruption = Interruption(session: session, cause: cause)
        }
    }

    /// Stops capture and creates a session from the recording. Throws
    /// `nothingCaptured` (and keeps no file) when no audio arrived.
    public func stop(name: String) async throws -> SessionManifest {
        guard let sink = endCapture() else { throw RecorderError.notRecording }
        return try await save(sink, name: name)
    }

    private func save(_ sink: RecordingSink, name: String) async throws -> SessionManifest {
        guard let caf = try sink.finish() else { throw RecorderError.nothingCaptured }
        let store = store
        return try await Task.detached {
            try RecordingRecovery.adopt(caf: caf, name: name, store: store)
        }.value
    }

    /// Stops capture and deletes everything recorded; no session is created.
    public func discard() {
        endCapture()?.discard()
    }

    /// Stops the device and resets the published state; returns the sink of
    /// the recording that was running, if any.
    @discardableResult
    private func endCapture() -> RecordingSink? {
        guard let capture, let sink else { return nil }
        timer?.invalidate()
        timer = nil
        capture.stop()
        self.capture = nil
        self.sink = nil
        isRecording = false
        levelDBFS = -120
        liveEnvelope = []
        return sink
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
