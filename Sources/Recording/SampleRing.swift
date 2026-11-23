import AudioToolbox
import RecordingAtomics

/// A lock-free single-producer/single-consumer ring of mono Float32 samples.
///
/// The producer is Core Audio's I/O thread: `write` downmixes the rendered
/// buffers straight into preallocated storage and never allocates, locks or
/// blocks. The consumer is the recording's writer thread. When the consumer
/// falls behind and a buffer does not fit, that buffer is dropped whole and
/// counted in `droppedFrames`, so an overrun is reported instead of leaving a
/// silent gap.
final class SampleRing: @unchecked Sendable {
    let capacity: Int
    private let mask: Int
    private let storage: UnsafeMutablePointer<Float>
    /// Total frames ever written; stored only by the producer (release).
    private let head: UnsafeMutablePointer<RecordingAtomicInt>
    /// Total frames ever read; stored only by the consumer (release).
    private let tail: UnsafeMutablePointer<RecordingAtomicInt>
    private let dropped: UnsafeMutablePointer<RecordingAtomicInt>

    /// Room for at least `minimumCapacity` samples (rounded up to a power of two).
    init(minimumCapacity: Int) {
        var capacity = 1
        while capacity < max(minimumCapacity, 2) { capacity <<= 1 }
        self.capacity = capacity
        mask = capacity - 1
        storage = .allocate(capacity: capacity)
        storage.initialize(repeating: 0, count: capacity)
        head = recording_atomic_create(0)!
        tail = recording_atomic_create(0)!
        dropped = recording_atomic_create(0)!
    }

    deinit {
        storage.deallocate()
        recording_atomic_destroy(head)
        recording_atomic_destroy(tail)
        recording_atomic_destroy(dropped)
    }

    /// Frames the producer had to drop because the ring was full.
    var droppedFrames: Int { Int(recording_atomic_load_relaxed(dropped)) }

    // MARK: Producer (real-time safe)

    /// Writes the channel mean of `frames` Float32 frames from any mix of
    /// planar and interleaved buffers. Returns false, and counts the frames as
    /// dropped, when they do not fit.
    @discardableResult
    func write(_ buffers: UnsafeMutableAudioBufferListPointer, frames: Int) -> Bool {
        var channels = 0
        for buffer in buffers where buffer.mData != nil { channels += Int(buffer.mNumberChannels) }
        guard frames > 0, channels > 0 else { return true }
        let scale = 1 / Float(channels)
        return produce(frames) { offset, destination, count in
            destination.update(repeating: 0, count: count)
            for buffer in buffers {
                guard let data = buffer.mData?.assumingMemoryBound(to: Float.self) else { continue }
                let stride = Int(buffer.mNumberChannels)
                for c in 0..<stride {
                    var index = offset * stride + c
                    for i in 0..<count {
                        destination[i] += data[index]
                        index += stride
                    }
                }
            }
            for i in 0..<count { destination[i] *= scale }
        }
    }

    /// Writes mono samples. Returns false, and counts them as dropped, when
    /// they do not fit.
    @discardableResult
    func write(_ samples: UnsafeBufferPointer<Float>) -> Bool {
        guard let base = samples.baseAddress, !samples.isEmpty else { return true }
        return produce(samples.count) { offset, destination, count in
            destination.update(from: base + offset, count: count)
        }
    }

    /// Reserves `frames` slots (in at most two contiguous parts), lets `fill`
    /// write them, then publishes them to the consumer.
    private func produce(_ frames: Int,
                         _ fill: (_ offset: Int, _ destination: UnsafeMutablePointer<Float>, _ count: Int) -> Void) -> Bool {
        let written = Int(recording_atomic_load_relaxed(head))
        let read = Int(recording_atomic_load_acquire(tail))
        guard capacity - (written - read) >= frames else {
            recording_atomic_fetch_add_relaxed(dropped, Int64(frames))
            return false
        }
        let start = written & mask
        let first = min(frames, capacity - start)
        fill(0, storage + start, first)
        if first < frames { fill(first, storage, frames - first) }
        recording_atomic_store_release(head, Int64(written + frames))
        return true
    }

    // MARK: Consumer

    /// Moves up to `destination.count` samples out of the ring; returns how many.
    func read(into destination: UnsafeMutableBufferPointer<Float>) -> Int {
        guard let base = destination.baseAddress else { return 0 }
        let read = Int(recording_atomic_load_relaxed(tail))
        let written = Int(recording_atomic_load_acquire(head))
        let frames = min(written - read, destination.count)
        guard frames > 0 else { return 0 }
        let start = read & mask
        let first = min(frames, capacity - start)
        base.update(from: storage + start, count: first)
        if first < frames { (base + first).update(from: storage, count: frames - first) }
        recording_atomic_store_release(tail, Int64(read + frames))
        return frames
    }
}
