import AudioCore
import AVFoundation
import Foundation
import Mastering
import os

public enum PreviewSource: String, CaseIterable, Sendable {
    case original, processed, mastered
}

/// A 48 kHz mono Float32 signal memory-mapped from a headerless scratch file,
/// so preview memory does not grow with recording length.
final class PreviewSignal: @unchecked Sendable {
    private let data: NSData
    let samples: UnsafeBufferPointer<Float>

    init(rawFloatFile url: URL) throws {
        data = try NSData(contentsOf: url, options: .alwaysMapped)
        samples = UnsafeBufferPointer(start: data.bytes.assumingMemoryBound(to: Float.self),
                                      count: data.length / MemoryLayout<Float>.size)
    }

    var count: Int { samples.count }

    /// Copies `count` samples starting at `start`, zero-filling past the end.
    @inline(__always)
    func copy(from start: Int, count: Int, into out: UnsafeMutablePointer<Float>) {
        let available = max(0, min(count, samples.count - start))
        if available > 0 {
            out.update(from: samples.baseAddress! + start, count: available)
        }
        if available < count {
            (out + available).update(repeating: 0, count: count - available)
        }
    }
}

/// State shared with the render block, which runs on Core Audio's I/O thread.
/// The render path does not allocate.
final class PreviewState: @unchecked Sendable {
    private let lock = OSAllocatedUnfairLock()

    var original: PreviewSignal?
    var processed: PreviewSignal?
    var chain: MasteringChain
    var position = 0
    var source: PreviewSource = .original
    var playing = false
    var originalGain: Float = 1
    var processedGain: Float = 1
    var masteredGain: Float = 1
    /// Playable span (trim) in frames; `rangeEnd` nil plays to the end.
    var rangeStart = 0
    var rangeEnd: Int?
    private var scratch = [Float](repeating: 0, count: 16_384)

    init(chain: MasteringChain) {
        self.chain = chain
    }

    func withLock<T>(_ body: () -> T) -> T {
        lock.lock()
        defer { lock.unlock() }
        return body()
    }

    var length: Int { max(original?.count ?? 0, processed?.count ?? 0) }
    var playableEnd: Int { min(length, rangeEnd ?? length) }

    /// Puts the play head inside the playable span.
    func clampToRange() {
        if position < rangeStart || position >= playableEnd { position = min(rangeStart, length); primeChain() }
    }

    /// The signal the mastering chain runs on: processed audio, or the
    /// original while nothing has been processed.
    private var masteringInput: PreviewSignal? { processed ?? original }

    /// Restarts the mastering chain so its output is aligned with `position`:
    /// the limiter's look-ahead is pre-rolled from the input signal. Does not
    /// allocate, so holding the render lock here stays short.
    func primeChain() {
        chain.reset()
        let latency = chain.latencyFrames
        guard latency > 0, let input = masteringInput else { return }
        precondition(latency <= scratch.count, "limiter look-ahead exceeds the preview scratch buffer")
        scratch.withUnsafeMutableBufferPointer { buffer in
            let preroll = UnsafeMutableBufferPointer(rebasing: buffer[0..<latency])
            input.copy(from: position, count: latency, into: preroll.baseAddress!)
            chain.process(preroll)
        }
    }

    func render(into out: UnsafeMutablePointer<Float>, frames: Int) {
        lock.lock()
        defer { lock.unlock() }
        guard playing, frames <= scratch.count else {
            out.update(repeating: 0, count: frames)
            return
        }
        // Always run the chain so switching to `mastered` is instant and its
        // state stays continuous.
        scratch.withUnsafeMutableBufferPointer { buffer in
            let block = UnsafeMutableBufferPointer(rebasing: buffer[0..<frames])
            if let input = masteringInput {
                input.copy(from: position + chain.latencyFrames, count: frames, into: block.baseAddress!)
            } else {
                block.update(repeating: 0)
            }
            chain.process(block)

            switch source {
            case .original:
                fill(out, from: original, frames: frames, gain: originalGain)
            case .processed:
                fill(out, from: processed ?? original, frames: frames, gain: processedGain)
            case .mastered:
                for i in 0..<frames { out[i] = block[i] * masteredGain }
            }
        }
        // Silence whatever lies past the end of the playable span.
        let audible = max(0, min(frames, playableEnd - position))
        if audible < frames { out.advanced(by: audible).update(repeating: 0, count: frames - audible) }
        position += frames
        if position >= playableEnd { playing = false }
    }

    @inline(__always)
    private func fill(_ out: UnsafeMutablePointer<Float>, from signal: PreviewSignal?, frames: Int, gain: Float) {
        guard let signal else {
            out.update(repeating: 0, count: frames)
            return
        }
        signal.copy(from: position, count: frames, into: out)
        for i in 0..<frames { out[i] *= gain }
    }
}

/// Plays original, AI-processed and live-mastered audio from one shared
/// position. Each source is level-matched to −23 LUFS for fair comparison
/// (preview only; exports are unaffected).
@MainActor
public final class PreviewPlayer: ObservableObject {
    public static let comparisonLUFS = -23.0

    @Published public var source: PreviewSource = .original {
        didSet { let s = source; state.withLock { state.source = s } }
    }
    @Published public private(set) var position: TimeInterval = 0
    @Published public private(set) var duration: TimeInterval = 0
    @Published public private(set) var isPlaying = false
    @Published public private(set) var playbackError: String?
    /// Peak envelope (one value per 50 ms) of the original and processed
    /// signals, for drawing waveforms.
    @Published public private(set) var envelopes: [PreviewSource: [Float]] = [:]

    let state: PreviewState
    private let engine = AVAudioEngine()
    private var timer: Timer?
    /// Integrated loudness per source; absent when unmeasurable (silence).
    private var loudness: [PreviewSource: Double] = [:]
    private var loadGeneration = 0
    private static let format = AVAudioFormat(standardFormatWithSampleRate: Double(AudioCore.sampleRate), channels: 1)!

    public init(mastering: MasteringSettings = .default) {
        state = PreviewState(chain: MasteringChain(settings: mastering, sampleRate: AudioCore.sampleRate))
        let node = Self.makeSourceNode(format: Self.format, state: state)
        engine.attach(node)
        engine.connect(node, to: engine.mainMixerNode, format: Self.format)
    }

    /// Builds the source node outside any actor: Core Audio calls its render
    /// block on the I/O thread, where a main-actor isolation check would trap.
    nonisolated static func makeSourceNode(format: AVAudioFormat, state: PreviewState) -> AVAudioSourceNode {
        AVAudioSourceNode(format: format) { @Sendable _, _, frameCount, audioBufferList in
            let buffers = UnsafeMutableAudioBufferListPointer(audioBufferList)
            let out = buffers[0].mData!.assumingMemoryBound(to: Float.self)
            state.render(into: out, frames: Int(frameCount))
            return noErr
        }
    }

    /// Converts both signals to 48 kHz Float32 scratch files off the main
    /// thread, measures their loudness, maps them and resets to the start.
    /// A newer `load` call supersedes an older one that is still running.
    public func load(original: URL, processed: URL?, scratchDirectory: URL) async throws {
        loadGeneration += 1
        let generation = loadGeneration
        pause()
        let prepared = try await Task.detached(priority: .userInitiated) {
            try FileManager.default.createDirectory(at: scratchDirectory, withIntermediateDirectories: true)
            let originalOut = scratchDirectory.appendingPathComponent("preview-original.f32")
            let processedOut = scratchDirectory.appendingPathComponent("preview-processed.f32")
            let originalPrep = try Self.prepare(original, to: originalOut)
            let processedPrep = try processed.map { try Self.prepare($0, to: processedOut) }
            return (originalOut, originalPrep, processed == nil ? nil : processedOut, processedPrep)
        }.value
        guard generation == loadGeneration else { return }

        let originalSignal = try PreviewSignal(rawFloatFile: prepared.0)
        let processedSignal = try prepared.2.map(PreviewSignal.init(rawFloatFile:))
        loudness = [:]
        loudness[.original] = prepared.1.lufs
        loudness[.processed] = prepared.3?.lufs ?? nil
        var shapes: [PreviewSource: [Float]] = [.original: prepared.1.envelope]
        if let processedPrep = prepared.3 { shapes[.processed] = processedPrep.envelope }
        envelopes = shapes
        state.withLock {
            state.original = originalSignal
            state.processed = processedSignal
            state.position = 0
            state.playing = false
            state.clampToRange()
            state.primeChain()
        }
        duration = Double(state.length) / Double(AudioCore.sampleRate)
        position = Double(state.withLock { state.position }) / Double(AudioCore.sampleRate)
        isPlaying = false
        // Without a processed signal only the original is meaningful.
        if processed == nil { source = .original }
        updateGains()
    }

    /// Applies mastering parameters live. `loudnessGainDB` is the gain the
    /// offline renderer would apply (from `MasteringRenderer.measureGain`).
    /// While paused the chain is re-primed at the current position, so the
    /// next playback starts exactly as the offline render would; while
    /// playing, filter state carries over to avoid clicks. `studioTone` is the
    /// measured studio-tone correction; nil keeps the current one.
    public func setMastering(_ settings: MasteringSettings, loudnessGainDB: Double, studioTone: [Double]? = nil) {
        // Filter design is costly; do it before taking the lock the render
        // callback shares, so the I/O thread never waits on it.
        let correction = studioTone ?? state.withLock { state.chain.studioToneCorrection }
        let design = MasteringChain.design(settings, studioToneCorrection: correction, sampleRate: AudioCore.sampleRate)
        state.withLock {
            let latencyChanged = settings.limiter.enabled != state.chain.settings.limiter.enabled
            state.chain.apply(design)
            state.chain.loudnessGainDB = loudnessGainDB
            if latencyChanged || !state.playing { state.primeChain() }
        }
        updateGains()
    }

    public func play() {
        guard state.withLock({ state.length }) > 0 else { return }
        if !engine.isRunning {
            do {
                try engine.start()
                playbackError = nil
            } catch {
                playbackError = error.localizedDescription
                return
            }
        }
        state.withLock {
            state.clampToRange()
            state.playing = true
        }
        isPlaying = true
        timer?.invalidate()
        timer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.syncFromState() }
        }
    }

    public func pause() {
        state.withLock { state.playing = false }
        isPlaying = false
        timer?.invalidate()
        timer = nil
        syncFromState()
    }

    /// Moves the play head, kept inside the trim range.
    public func seek(to time: TimeInterval) {
        let frame = state.withLock {
            state.position = max(state.rangeStart, min(state.playableEnd, Int(time * Double(AudioCore.sampleRate))))
            state.primeChain()
            return state.position
        }
        position = Double(frame) / Double(AudioCore.sampleRate)
    }

    /// Limits playback to `[start, end)` seconds (the trim); nil plays everything.
    public func setRange(start: TimeInterval?, end: TimeInterval?) {
        let rate = Double(AudioCore.sampleRate)
        let frame = state.withLock {
            state.rangeStart = max(0, Int(((start ?? 0) * rate).rounded()))
            state.rangeEnd = end.map { Int(($0 * rate).rounded()) }
            // Also while playing: a play head left outside the new span jumps into it.
            state.clampToRange()
            return state.position
        }
        position = Double(frame) / Double(AudioCore.sampleRate)
    }

    public func stop() {
        pause()
        engine.stop()
    }

    private func syncFromState() {
        let (frame, playing) = state.withLock { (state.position, state.playing) }
        position = Double(frame) / Double(AudioCore.sampleRate)
        if isPlaying && !playing {
            isPlaying = false
            timer?.invalidate()
            timer = nil
        }
    }

    private func updateGains() {
        let (settings, gainDB) = state.withLock { (state.chain.settings, state.chain.loudnessGainDB) }
        func match(_ lufs: Double?) -> Float {
            guard let lufs else { return 1 }
            return Float(pow(10, (Self.comparisonLUFS - lufs) / 20))
        }
        let original = match(loudness[.original])
        let processed = loudness[.processed] == nil && state.withLock({ state.processed == nil })
            ? original : match(loudness[.processed])
        let mastered: Float
        if settings.loudness.enabled, loudness[.processed] ?? loudness[.original] != nil {
            mastered = Float(pow(10, (Self.comparisonLUFS - settings.loudness.targetLUFS) / 20))
        } else {
            // Undo only gain the chain actually applies (none with loudness off).
            mastered = processed * (settings.loudness.enabled ? Float(pow(10, -gainDB / 20)) : 1)
        }
        state.withLock {
            state.originalGain = original
            state.processedGain = processed
            state.masteredGain = mastered
        }
    }

    /// Streams `source` to a headerless 48 kHz Float32 file and returns its
    /// integrated loudness and a 50 ms peak envelope.
    nonisolated static func prepare(_ source: URL, to destination: URL) throws -> (lufs: Double?, envelope: [Float]) {
        let reader = try MonoFileReader(url: source)
        let resampler = reader.sampleRate == AudioCore.sampleRate
            ? nil : try StreamingResampler(from: reader.sampleRate, to: AudioCore.sampleRate)
        FileManager.default.createFile(atPath: destination.path, contents: nil)
        let handle = try FileHandle(forWritingTo: destination)
        defer { try? handle.close() }
        var meter = LoudnessMeter(sampleRate: AudioCore.sampleRate)
        let bucket = AudioCore.sampleRate / 20
        var envelope: [Float] = []
        var bucketPeak: Float = 0
        var bucketFill = 0

        func emit(_ block: [Float]) throws {
            guard !block.isEmpty else { return }
            meter.add(block)
            for x in block {
                bucketPeak = max(bucketPeak, abs(x))
                bucketFill += 1
                if bucketFill == bucket {
                    envelope.append(bucketPeak)
                    bucketPeak = 0
                    bucketFill = 0
                }
            }
            try block.withUnsafeBufferPointer { try handle.write(contentsOf: Data(buffer: $0)) }
        }
        while let block = try reader.read(maxFrames: 65_536) {
            try emit(try resampler?.process(block) ?? block)
        }
        if let resampler { try emit(try resampler.flush()) }
        if bucketFill > 0 { envelope.append(bucketPeak) }
        return (meter.integratedLUFS, envelope)
    }

    // MARK: Testing support

    /// Switches the engine to offline manual rendering (tests only).
    func enableManualRendering(maximumFrames: AVAudioFrameCount = 4096) throws {
        engine.stop()
        try engine.enableManualRenderingMode(.offline, format: Self.format, maximumFrameCount: maximumFrames)
        try engine.start()
    }

    func renderManually(frames: Int) throws -> [Float] {
        let buffer = AVAudioPCMBuffer(pcmFormat: engine.manualRenderingFormat, frameCapacity: engine.manualRenderingMaximumFrameCount)!
        var out = [Float]()
        var remaining = frames
        while remaining > 0 {
            let n = min(remaining, Int(engine.manualRenderingMaximumFrameCount))
            let status = try engine.renderOffline(AVAudioFrameCount(n), to: buffer)
            guard status == .success else { break }
            out += UnsafeBufferPointer(start: buffer.floatChannelData![0], count: Int(buffer.frameLength))
            remaining -= Int(buffer.frameLength)
        }
        return out
    }
}
