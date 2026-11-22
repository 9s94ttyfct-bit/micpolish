import AudioToolbox
import CoreAudio
import Foundation

/// Captures one input device through its own AUHAL unit.
///
/// AVAudioEngine's input node follows the system's default-device aggregate:
/// pointing it at another device leaves its client format at the default
/// device's rate ("Failed to create tap, config change pending!") and it later
/// switches itself back, so recording from a non-default microphone fails as
/// soon as another device (e.g. AirPods) becomes the default. A dedicated
/// AUHAL bound to a concrete device is unaffected by default-device changes.
final class DeviceCapture: @unchecked Sendable {
    let deviceID: AudioDeviceID
    let sampleRate: Double
    let channels: Int

    private var unit: AudioUnit
    private let bufferList: UnsafeMutableAudioBufferListPointer
    private let maxFrames: Int
    private let onAudio: (UnsafeMutableAudioBufferListPointer, Int) -> Void
    private var listening = false
    /// Set from Core Audio's listener thread when the device goes away or
    /// changes its sample rate; polled by the recorder.
    private let flags = Flags()

    final class Flags: @unchecked Sendable {
        private let lock = NSLock()
        private var _lost = false
        var lost: Bool {
            get { lock.withLock { _lost } }
            set { lock.withLock { _lost = newValue } }
        }
    }

    /// True once the device disconnected or changed its sample rate; capture
    /// from it can no longer continue.
    var deviceLost: Bool { flags.lost }

    init(deviceID: AudioDeviceID, onAudio: @escaping (UnsafeMutableAudioBufferListPointer, Int) -> Void) throws {
        self.deviceID = deviceID
        self.onAudio = onAudio

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
    /// device's own rate (no sample-rate conversion in the unit).
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

        var frames: UInt32 = 0
        size = UInt32(MemoryLayout<UInt32>.size)
        if AudioUnitGetProperty(unit, kAudioUnitProperty_MaximumFramesPerSlice, kAudioUnitScope_Global, 0,
                                &frames, &size) != noErr || frames == 0 {
            frames = 4096
        }
        return (hardware.mSampleRate, Int(hardware.mChannelsPerFrame), Int(frames))
    }

    deinit {
        stop()
        AudioUnitUninitialize(unit)
        AudioComponentInstanceDispose(unit)
        for buffer in bufferList { buffer.mData?.deallocate() }
        free(bufferList.unsafeMutablePointer)
    }

    func start() throws {
        addListeners()
        try Self.check(AudioOutputUnitStart(unit))
    }

    func stop() {
        AudioOutputUnitStop(unit)
        removeListeners()
    }

    private func render(_ flags: UnsafeMutablePointer<AudioUnitRenderActionFlags>,
                        _ timeStamp: UnsafePointer<AudioTimeStamp>, _ bus: UInt32, _ frameCount: UInt32) -> OSStatus {
        let frames = min(Int(frameCount), maxFrames)
        for i in 0..<channels { bufferList[i].mDataByteSize = UInt32(frames * 4) }
        let status = AudioUnitRender(unit, flags, timeStamp, bus, UInt32(frames), bufferList.unsafeMutablePointer)
        if status == noErr { onAudio(bufferList, frames) }
        return status
    }

    // MARK: Device listeners

    private static let watched: [AudioObjectPropertySelector] = [
        kAudioDevicePropertyDeviceIsAlive, kAudioDevicePropertyNominalSampleRate,
    ]
    private let listener: AudioObjectPropertyListenerBlock = { _, _ in }
    private lazy var block: AudioObjectPropertyListenerBlock = { [flags, deviceID, sampleRate] _, _ in
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
        if aliveStatus != noErr || alive == 0 || rateStatus != noErr || abs(rate - sampleRate) > 0.5 {
            flags.lost = true
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
