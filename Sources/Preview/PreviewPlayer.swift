import AudioCore
import AVFoundation
import Foundation
import Mastering
import os

public enum PreviewSource: String, CaseIterable, Sendable {
    case original, processed, mastered
}

/// Audio-thread state shared by the source node's render block.
final class PreviewState: @unchecked Sendable {
    private let lock = OSAllocatedUnfairLock()

    var original: [Float] = []
    var processed: [Float] = []
    var chain: MasteringChain
    var position = 0
    var source: PreviewSource = .original
    var playing = false
    var gains: [PreviewSource: Float] = [.original: 1, .processed: 1, .mastered: 1]
    private var scratch = [Float]()

    init(chain: MasteringChain) {
        self.chain = chain
    }

    func withLock<T>(_ body: () -> T) -> T {
        lock.lock()
        defer { lock.unlock() }
        return body()
    }

    var length: Int { max(original.count, processed.count) }

    /// Restarts the mastering chain so its output is aligned with `position`:
    /// the limiter's look-ahead is pre-rolled from the processed signal.
    func primeChain() {
        chain.reset()
        let latency = chain.latencyFrames
        guard latency > 0 else { return }
        var preroll = slice(processed, from: position, count: latency)
        chain.process(&preroll)
    }

    func render(into out: UnsafeMutablePointer<Float>, frames: Int) {
        withLock {
            guard playing else {
                out.update(repeating: 0, count: frames)
                return
            }
            // Always run the chain so switching to `mastered` is instant and
            // its state stays continuous.
            scratch = slice(processed, from: position + chain.latencyFrames, count: frames)
            chain.process(&scratch)

            let gain = gains[source] ?? 1
            switch source {
            case .original:
                let block = slice(original, from: position, count: frames)
                for i in 0..<frames { out[i] = block[i] * gain }
            case .processed:
                let block = slice(processed, from: position, count: frames)
                for i in 0..<frames { out[i] = block[i] * gain }
            case .mastered:
                for i in 0..<frames { out[i] = scratch[i] * gain }
            }
            position += frames
            if position >= length { playing = false }
        }
    }

    private func slice(_ signal: [Float], from start: Int, count: Int) -> [Float] {
        var block = [Float](repeating: 0, count: count)
        guard start < signal.count else { return block }
        let n = min(count, signal.count - start)
        for i in 0..<n { block[i] = signal[start + i] }
        return block
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

    let state: PreviewState
    private let engine = AVAudioEngine()
    private var node: AVAudioSourceNode!
    private var timer: Timer?
    /// Integrated loudness per source; absent when unmeasurable (silence).
    private var loudness: [PreviewSource: Double] = [:]
    private let format = AVAudioFormat(standardFormatWithSampleRate: Double(AudioCore.sampleRate), channels: 1)!

    public init(mastering: MasteringSettings = .default) {
        state = PreviewState(chain: MasteringChain(settings: mastering, sampleRate: AudioCore.sampleRate))
        let state = self.state
        node = AVAudioSourceNode(format: format) { _, _, frameCount, audioBufferList in
            let buffers = UnsafeMutableAudioBufferListPointer(audioBufferList)
            let out = buffers[0].mData!.assumingMemoryBound(to: Float.self)
            state.render(into: out, frames: Int(frameCount))
            return noErr
        }
        engine.attach(node)
        engine.connect(node, to: engine.mainMixerNode, format: format)
    }

    /// Loads both signals (converted to 48 kHz mono) and resets to the start.
    public func load(original: URL, processed: URL?) throws {
        let originalSignal = try Self.load48k(original)
        let processedSignal = try processed.map(Self.load48k) ?? originalSignal
        loudness[.original] = Self.integratedLoudness(originalSignal)
        loudness[.processed] = Self.integratedLoudness(processedSignal)
        state.withLock {
            state.original = originalSignal
            state.processed = processedSignal
            state.position = 0
            state.playing = false
            state.primeChain()
        }
        duration = Double(state.length) / Double(AudioCore.sampleRate)
        position = 0
        isPlaying = false
        // Without a processed signal only the original is meaningful.
        if processed == nil { source = .original }
        updateGains()
    }

    /// Applies mastering parameters live. `loudnessGainDB` is the gain the
    /// offline renderer would apply (from `MasteringRenderer.measureGain`).
    /// While paused the chain is re-primed at the current position, so the
    /// next playback starts exactly as the offline render would; while
    /// playing, filter state carries over to avoid clicks.
    public func setMastering(_ settings: MasteringSettings, loudnessGainDB: Double) {
        state.withLock {
            let latencyChanged = settings.limiter.enabled != state.chain.settings.limiter.enabled
            state.chain.update(settings)
            state.chain.loudnessGainDB = loudnessGainDB
            if latencyChanged || !state.playing { state.primeChain() }
        }
        updateGains()
    }

    public func play() {
        if !engine.isRunning { try? engine.start() }
        state.withLock {
            if state.position >= state.length { state.position = 0; state.primeChain() }
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

    public func seek(to time: TimeInterval) {
        let frame = max(0, min(state.length, Int(time * Double(AudioCore.sampleRate))))
        state.withLock {
            state.position = frame
            state.primeChain()
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
        let settings = state.withLock { state.chain.settings }
        let gainDB = state.withLock { state.chain.loudnessGainDB }
        func match(_ lufs: Double?) -> Float {
            guard let lufs else { return 1 }
            return Float(pow(10, (Self.comparisonLUFS - lufs) / 20))
        }
        let original = match(loudness[.original])
        let processed = match(loudness[.processed])
        let mastered: Float
        if settings.loudness.enabled, loudness[.processed] != nil {
            mastered = Float(pow(10, (Self.comparisonLUFS - settings.loudness.targetLUFS) / 20))
        } else {
            mastered = processed * Float(pow(10, -gainDB / 20))
        }
        state.withLock { state.gains = [.original: original, .processed: processed, .mastered: mastered] }
    }

    static func integratedLoudness(_ signal: [Float]) -> Double? {
        var meter = LoudnessMeter(sampleRate: AudioCore.sampleRate)
        meter.add(signal)
        return meter.integratedLUFS
    }

    static func load48k(_ url: URL) throws -> [Float] {
        let (samples, rate) = try readAllMono(url)
        guard rate != AudioCore.sampleRate else { return samples }
        let resampler = try StreamingResampler(from: rate, to: AudioCore.sampleRate)
        return try resampler.process(samples) + resampler.flush()
    }

    // MARK: Testing support

    /// Switches the engine to offline manual rendering (tests only).
    func enableManualRendering(maximumFrames: AVAudioFrameCount = 4096) throws {
        engine.stop()
        try engine.enableManualRenderingMode(.offline, format: format, maximumFrameCount: maximumFrames)
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
