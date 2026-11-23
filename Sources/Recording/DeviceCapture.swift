import AudioToolbox
import CoreAudio
import Foundation
import RecordingAtomics

/// Captures one input device through its own AUHAL unit.
///
/// AVAudioEngine's input node follows the system's default-device aggregate:
/// pointing it at another device leaves its client format at the default
/// device's rate ("Failed to create tap, config change pending!") and it later
/// switches itself back, so recording from a non-default microphone fails as
/// soon as another device (e.g. AirPods) becomes the default. A dedicated
/// AUHAL bound to a concrete device is unaffected by default-device changes.
///
/// The render callback runs on Core Audio's real-time I/O thread. It only
/// renders into preallocated buffers and copies the channel mean into `ring`;
/// it never allocates, locks or touches the disk.
final class DeviceCapture: @unchecked Sendable {
    let deviceID: AudioDeviceID
    let sampleRate: Double
    let channels: Int
    /// Mono audio at `sampleRate`, filled by the I/O thread.
    let ring: SampleRing

    private let unit: AudioUnit
    private let bufferList: UnsafeMutableAudioBufferListPointer
    private let maxFrames: Int
    /// Frames of the current run of failed renders; written only by the I/O thread.
    private let failedFrames: UnsafeMutablePointer<RecordingAtomicInt>
    /// The OSStatus of a render failure that lasted `failureLimit` frames; 0 while none did.
    private let failureStatus: UnsafeMutablePointer<RecordingAtomicInt>
    private let failureLimit: Int
    private var listening = false
    /// Set from Core Audio's listener thread when the device goes away or
    /// changes its sample rate; polled by the recorder.
    private let flags = Flags()

    final class Flags: @unchecked Sendable {
        private let lock = NSLock()
        private var _lost = false
        private var _rateChanged = false
        var lost: Bool {
            get { lock.withLock { _lost } }
            set { lock.withLock { _lost = newValue } }
        }
        var rateChanged: Bool {
            get { lock.withLock { _rateChanged } }
            set { lock.withLock { _rateChanged = newValue } }
        }
    }

    /// True once the device disconnected; capture from it can no longer continue.
    var deviceLost: Bool { flags.lost }

    /// True once the device's nominal rate differs from `sampleRate`: the
    /// AUHAL does not convert input rates, so capture must be rebuilt.
    var rateChanged: Bool { flags.rateChanged }

    /// The status of renders that kept failing for half a second, or nil.
    var renderFailure: OSStatus? {
        let status = recording_atomic_load_relaxed(failureStatus)
        return status == 0 ? nil : OSStatus(truncatingIfNeeded: status)
    }

    init(deviceID: AudioDeviceID) throws {
        self.deviceID = deviceID

        var description = AudioComponentDescription(
            componentType: kAudioUnitType_Output, componentSubType: kAudioUnitSubType_HALOutput,
            componentManufacturer: kAudioUnitManufacturer_Apple, componentFlags: 0, componentFlagsMask: 0)
        guard let component = AudioComponentFindNext(nil, &description) else {
            throw RecorderError.deviceSelectionFailed(kAudioUnitErr_NoConnection)
        }
        var instance: AudioUnit?
        try Self.check(AudioComponentInstanceNew(component, &instance))
        let unit = instance!
        // Until every stored property is set a throw skips `deinit`, so the
        // unit is disposed here if configuring it fails.
        let format: (sampleRate: Double, channels: Int, maxFrames: Int)
        do {
            format = try Self.configure(unit, deviceID: deviceID)
        } catch {
            AudioComponentInstanceDispose(unit)
            throw error
        }
        self.unit = unit
        sampleRate = format.sampleRate
        channels = format.channels
        maxFrames = format.maxFrames
        // Ten seconds of slack for the writer thread (a slow disk, a busy system).
        ring = SampleRing(minimumCapacity: max(Int(sampleRate * 10), maxFrames * 4))
        failureLimit = Int(sampleRate / 2)
        failedFrames = recording_atomic_create(0)!
        failureStatus = recording_atomic_create(0)!
        bufferList = AudioBufferList.allocate(maximumBuffers: channels)
        for i in 0..<channels {
            bufferList[i] = AudioBuffer(mNumberChannels: 1, mDataByteSize: UInt32(maxFrames * 4),
                                        mData: UnsafeMutableRawPointer.allocate(byteCount: maxFrames * 4, alignment: 16))
        }
        // From here on a throw runs `deinit`, which releases the unit and buffers.

        var callback = AURenderCallbackStruct(inputProc: { refCon, flags, timeStamp, bus, frameCount, _ in
            let capture = Unmanaged<DeviceCapture>.fromOpaque(refCon).takeUnretainedValue()
            return capture.render(flags, timeStamp, bus, frameCount)
        }, inputProcRefCon: Unmanaged.passUnretained(self).toOpaque())
        try Self.check(AudioUnitSetProperty(unit, kAudioOutputUnitProperty_SetInputCallback, kAudioUnitScope_Global, 0,
                                            &callback, UInt32(MemoryLayout<AURenderCallbackStruct>.size)))
        try Self.check(AudioUnitInitialize(unit))
    }

    /// Enables input only, binds `deviceID` and sets planar Float32 at the
    /// device's own rate (no sample-rate conversion in the unit). Sizes the
    /// render buffers for the largest I/O buffer the device allows.
    private static func configure(_ unit: AudioUnit, deviceID: AudioDeviceID) throws
        -> (sampleRate: Double, channels: Int, maxFrames: Int) {
        var on: UInt32 = 1, off: UInt32 = 0
        try check(AudioUnitSetProperty(unit, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Input, 1,
                                       &on, UInt32(MemoryLayout<UInt32>.size)))
        try check(AudioUnitSetProperty(unit, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Output, 0,
                                       &off, UInt32(MemoryLayout<UInt32>.size)))
        var device = deviceID
        try check(AudioUnitSetProperty(unit, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0,
                                       &device, UInt32(MemoryLayout<AudioDeviceID>.size)))

        // The device side of the input bus gives the hardware rate and channels.
        var hardware = AudioStreamBasicDescription()
        var size = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        try check(AudioUnitGetProperty(unit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, 1,
                                       &hardware, &size))
        guard hardware.mSampleRate > 0, hardware.mChannelsPerFrame > 0 else {
            throw RecorderError.deviceSelectionFailed(kAudioUnitErr_FormatNotSupported)
        }
        var client = AudioStreamBasicDescription(
            mSampleRate: hardware.mSampleRate, mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked | kAudioFormatFlagIsNonInterleaved,
            mBytesPerPacket: 4, mFramesPerPacket: 1, mBytesPerFrame: 4,
            mChannelsPerFrame: hardware.mChannelsPerFrame, mBitsPerChannel: 32, mReserved: 0)
        try check(AudioUnitSetProperty(unit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Output, 1,
                                       &client, UInt32(MemoryLayout<AudioStreamBasicDescription>.size)))

        // Another app may raise the device's I/O buffer size while recording,
        // so allow the largest size the device supports, not just the current one.
        var frames: UInt32 = 0
        size = UInt32(MemoryLayout<UInt32>.size)
        if AudioUnitGetProperty(unit, kAudioUnitProperty_MaximumFramesPerSlice, kAudioUnitScope_Global, 0,
                                &frames, &size) != noErr || frames == 0 {
            frames = 4096
        }
        var needed = frames
        var ioFrames: UInt32 = 0
        size = UInt32(MemoryLayout<UInt32>.size)
        var address = AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyBufferFrameSize,
                                                 mScope: kAudioObjectPropertyScopeGlobal,
                                                 mElement: kAudioObjectPropertyElementMain)
        if AudioObjectGetPropertyData(deviceID, &address, 0, nil, &size, &ioFrames) == noErr {
            needed = max(needed, ioFrames)
        }
        var range = AudioValueRange()
        size = UInt32(MemoryLayout<AudioValueRange>.size)
        address.mSelector = kAudioDevicePropertyBufferFrameSizeRange
        if AudioObjectGetPropertyData(deviceID, &address, 0, nil, &size, &range) == noErr {
            needed = max(needed, UInt32(min(range.mMaximum, 65_536)))
        }
        if needed > frames {
            try check(AudioUnitSetProperty(unit, kAudioUnitProperty_MaximumFramesPerSlice, kAudioUnitScope_Global, 0,
                                           &needed, UInt32(MemoryLayout<UInt32>.size)))
        }
        return (hardware.mSampleRate, Int(hardware.mChannelsPerFrame), Int(needed))
    }

    deinit {
        stop()
        AudioUnitUninitialize(unit)
        AudioComponentInstanceDispose(unit)
        for buffer in bufferList { buffer.mData?.deallocate() }
        free(bufferList.unsafeMutablePointer)
        recording_atomic_destroy(failedFrames)
        recording_atomic_destroy(failureStatus)
    }

    /// Starts capture. Starting input can make the device switch its rate
    /// (a Bluetooth headset moving to its headset profile), possibly before
    /// the listeners are in place, so the rate is checked again afterwards.
    func start() throws {
        addListeners()
        try Self.check(AudioOutputUnitStart(unit))
        checkDevice()
    }

    /// Stops capture; once it returns the I/O thread no longer renders.
    func stop() {
        AudioOutputUnitStop(unit)
        removeListeners()
    }

    private func render(_ flags: UnsafeMutablePointer<AudioUnitRenderActionFlags>,
                        _ timeStamp: UnsafePointer<AudioTimeStamp>, _ bus: UInt32, _ frameCount: UInt32) -> OSStatus {
        let frames = Int(frameCount)
        var status = kAudioUnitErr_TooManyFramesToProcess
        if frames <= maxFrames {
            for i in 0..<channels { bufferList[i].mDataByteSize = UInt32(frames * 4) }
            status = AudioUnitRender(unit, flags, timeStamp, bus, frameCount, bufferList.unsafeMutablePointer)
        }
        if status == noErr {
            recording_atomic_store_relaxed(failedFrames, 0)
            ring.write(bufferList, frames: frames)
        } else {
            // A lasting failure is reported; brief ones (a device reconfiguring) are ridden out.
            let failed = Int(recording_atomic_load_relaxed(failedFrames)) + frames
            recording_atomic_store_relaxed(failedFrames, Int64(failed))
            if failed >= failureLimit, recording_atomic_load_relaxed(failureStatus) == 0 {
                recording_atomic_store_relaxed(failureStatus, Int64(status))
            }
        }
        return status
    }

    // MARK: Device listeners

    private static let watched: [AudioObjectPropertySelector] = [
        kAudioDevicePropertyDeviceIsAlive, kAudioDevicePropertyNominalSampleRate,
    ]
    private lazy var block: AudioObjectPropertyListenerBlock = { [weak self] _, _ in self?.checkDevice() }

    /// Flags a lost device or a nominal rate other than the one captured.
    private func checkDevice() {
        var alive: UInt32 = 1
        var size = UInt32(MemoryLayout<UInt32>.size)
        var address = AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyDeviceIsAlive,
                                                 mScope: kAudioObjectPropertyScopeGlobal,
                                                 mElement: kAudioObjectPropertyElementMain)
        let aliveStatus = AudioObjectGetPropertyData(deviceID, &address, 0, nil, &size, &alive)
        var rate: Float64 = 0
        size = UInt32(MemoryLayout<Float64>.size)
        address.mSelector = kAudioDevicePropertyNominalSampleRate
        let rateStatus = AudioObjectGetPropertyData(deviceID, &address, 0, nil, &size, &rate)
        if aliveStatus != noErr || alive == 0 || rateStatus != noErr {
            flags.lost = true
        } else if abs(rate - sampleRate) > 0.5 {
            flags.rateChanged = true
        }
    }

    private func addListeners() {
        guard !listening else { return }
        listening = true
        for selector in Self.watched {
            var address = AudioObjectPropertyAddress(mSelector: selector, mScope: kAudioObjectPropertyScopeGlobal,
                                                     mElement: kAudioObjectPropertyElementMain)
            AudioObjectAddPropertyListenerBlock(deviceID, &address, nil, block)
        }
    }

    private func removeListeners() {
        guard listening else { return }
        listening = false
        for selector in Self.watched {
            var address = AudioObjectPropertyAddress(mSelector: selector, mScope: kAudioObjectPropertyScopeGlobal,
                                                     mElement: kAudioObjectPropertyElementMain)
            AudioObjectRemovePropertyListenerBlock(deviceID, &address, nil, block)
        }
    }

    private static func check(_ status: OSStatus) throws {
        guard status == noErr else { throw RecorderError.deviceSelectionFailed(status) }
    }

    /// The system's current default input device.
    static func defaultInputDevice() -> AudioDeviceID? {
        var id = AudioDeviceID(0)
        var size = UInt32(MemoryLayout<AudioDeviceID>.size)
        var address = AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyDefaultInputDevice,
                                                 mScope: kAudioObjectPropertyScopeGlobal,
                                                 mElement: kAudioObjectPropertyElementMain)
        guard AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &id) == noErr,
              id != 0 else { return nil }
        return id
    }
}
