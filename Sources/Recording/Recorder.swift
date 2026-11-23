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
/// connecting or becoming the default do not affect a running recording. When
/// the recorded device changes its sample rate (a Bluetooth headset switching
/// profile as input starts) capture is rebuilt at the new rate into the same
/// file; the recording stops when the device disconnects, keeps failing to
/// deliver audio, or audio has to be dropped because the disk cannot keep up.
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
    /// Set when capture stopped on its own (disk full, input device lost or
    /// failing, audio dropped); the audio captured up to that point is saved
    /// as a session.
    @Published public private(set) var interruption: Interruption?

    public struct Interruption: Equatable {
        public enum Cause: Equatable {
            /// Writing the recording failed (usually a full disk).
            case writeFailed(String)
            /// The input device disconnected, or capture could not follow its
            /// new sample rate.
            case deviceLost
            /// The input device kept failing to deliver audio (Core Audio status).
            case inputFailed(OSStatus)
            /// The disk could not keep up and audio had to be dropped.
            case overrun
        }
        public enum Outcome: Equatable {
            case saved(SessionManifest)
            /// No audio had been written yet; nothing was kept.
            case nothingCaptured
            /// Saving failed (the error's description). The capture file stays
            /// in the cache and is recovered as a session on the next launch.
            case saveFailed(String)
        }
        public let cause: Cause
        public let outcome: Outcome
        public var session: SessionManifest? {
            if case .saved(let session) = outcome { session } else { nil }
        }
    }

    /// Where finished recordings go; follows the app's current sessions folder.
    public var store: SessionStore
    private let cacheRoot: URL
    private var capture: DeviceCapture?
    private var sink: RecordingSink?
    private var timer: Timer?
    /// Captures rebuilt after rate changes during the current recording.
    private var rebuilds = 0
    /// More rebuilds than this means the device keeps changing; it is treated as lost.
    private static let maxRebuilds = 5
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
        let capture = try DeviceCapture(deviceID: deviceID)
        let sink = try RecordingSink(url: url, input: capture.ring, inputSampleRate: Int(capture.sampleRate.rounded()))
        do {
            try capture.start()
        } catch {
            sink.discard()
            throw error
        }
        sink.startWriting()

        self.capture = capture
        self.sink = sink
        rebuilds = 0
        elapsed = 0
        liveEnvelope = []
        interruption = nil
        isRecording = true
        let started = Date()
        timer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated { self?.tick(started: started) }
        }
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
        } else if capture.ring.droppedFrames > 0 {
            cause = .overrun
        } else if capture.rateChanged {
            // Before render failures: a rate switch is what makes renders fail.
            cause = followRateChange(of: capture, into: sink)
        } else if let status = capture.renderFailure {
            cause = .inputFailed(status)
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
            let outcome: Interruption.Outcome
            do {
                outcome = .saved(try await save(ended, name: name))
            } catch RecorderError.nothingCaptured {
                outcome = .nothingCaptured
            } catch {
                outcome = .saveFailed(error.localizedDescription)
            }
            interruption = Interruption(cause: cause, outcome: outcome)
        }
    }

    /// Rebuilds capture at the device's new nominal rate and continues the
    /// same file: the AUHAL cannot convert input rates, so ignoring the change
    /// would record at the wrong speed. Returns the cause to stop with when
    /// that is not possible.
    private func followRateChange(of old: DeviceCapture, into sink: RecordingSink) -> Interruption.Cause? {
        guard rebuilds < Self.maxRebuilds else { return .deviceLost }
        rebuilds += 1
        do {
            let replacement = try DeviceCapture(deviceID: old.deviceID)
            old.stop()
            if old.ring.droppedFrames > 0 { return .overrun }
            try sink.switchInput(to: replacement.ring, sampleRate: Int(replacement.sampleRate.rounded()))
            try replacement.start()
            capture = replacement
            return nil
        } catch {
            return .deviceLost
        }
    }

    /// Stops capture and creates a session from the recording. Throws
    /// `nothingCaptured` (and keeps no file) when no audio arrived.
    public func stop(name: String) async throws -> SessionManifest {
        guard let sink = endCapture() else { throw RecorderError.notRecording }
        return try await save(sink, name: name)
    }

    /// Finishes the file and adopts it, off the main actor: the writer
    /// thread's last drain and the FLAC conversion touch the disk.
    private func save(_ sink: RecordingSink, name: String) async throws -> SessionManifest {
        let store = store
        return try await Task.detached {
            guard let caf = try sink.finish() else { throw RecorderError.nothingCaptured }
            return try RecordingRecovery.adopt(caf: caf, name: name, store: store)
        }.value
    }

    /// Stops capture and deletes everything recorded; no session is created.
    public func discard() {
        guard let sink = endCapture() else { return }
        Task.detached { sink.discard() }
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
