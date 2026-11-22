// SPDX-License-Identifier: Apache-2.0

import Foundation

protocol MonoWindowEnhancing: Sendable {
    func enhanceWindow(
        _ samples: [Float], sampleRate: Int, backend: ReuseBackend
    ) async throws -> (samples: [Float], backend: ReuseBackend)
}

extension MambaEnhancer: MonoWindowEnhancing {}

/// Bounded mono stream state. The shared `MambaEnhancer` actor performs every
/// model invocation, so sessions may coexist without concurrent model access.
actor MambaEnhancementSession: ReuseEnhancementSession {
    nonisolated let recommendedInputFrameCount: Int
    nonisolated let requestedBackend: ReuseBackend

    private let enhancer: any MonoWindowEnhancing
    private let sampleRate: Int
    private let chunkLength: Int
    private let overlapLength: Int
    private let stride: Int

    private var input = [Float]()
    private var accumulation = [Float]()
    private var normalization = [Float]()
    private var chunkIndex = 0
    private var isFinished = false
    private(set) var backendUsage = ReuseBackendUsage()

    init(
        enhancer: any MonoWindowEnhancing,
        sampleRate: Int,
        settings: InferenceSettings,
        backend: ReuseBackend = .mlx
    ) throws {
        guard sampleRate > 0 else {
            throw InferenceError.inferenceFailed("Sample rate must be greater than zero")
        }
        let chunkLengthDouble = settings.chunkSeconds * Double(sampleRate)
        guard chunkLengthDouble.isFinite,
              chunkLengthDouble >= 1,
              chunkLengthDouble < Double(Int.max) else {
            throw InferenceError.inferenceFailed(
                "Invalid chunk length from chunkSeconds=\(settings.chunkSeconds) and sampleRate=\(sampleRate)"
            )
        }

        let chunkLength = Int(chunkLengthDouble)
        let rawOverlap = Int(Double(chunkLength) * settings.overlapPortion)
        let overlapLength = max(0, min(chunkLength - 1, rawOverlap))
        let stride = max(1, chunkLength - overlapLength)

        self.enhancer = enhancer
        self.requestedBackend = backend
        self.sampleRate = sampleRate
        self.chunkLength = chunkLength
        self.overlapLength = overlapLength
        self.stride = stride
        self.recommendedInputFrameCount = stride
    }

    func append(_ monoFrames: [Float]) async throws -> [Float] {
        guard !isFinished else {
            throw InferenceError.inferenceFailed("Cannot append to a finished enhancement session")
        }
        guard !monoFrames.isEmpty else { return [] }
        try Task.checkCancellation()
        input.append(contentsOf: monoFrames)

        var emitted = [Float]()
        while input.count >= chunkLength && input.count > stride {
            let window = Array(input.prefix(chunkLength))
            emitted += try await process(window)
            input.removeFirst(stride)
            try Task.checkCancellation()
        }
        return emitted
    }

    func finish() async throws -> [Float] {
        guard !isFinished else {
            throw InferenceError.inferenceFailed("Enhancement session is already finished")
        }
        isFinished = true
        try Task.checkCancellation()

        var emitted = try await drainPipeline()
        if !input.isEmpty {
            let enhanced = try await enhanceWindow(input)
            try accumulate(enhanced, isLast: true)
            input.removeAll(keepingCapacity: false)
            chunkIndex += 1
        }
        emitted += emitPrefix(accumulation.count)
        return emitted
    }

    /// Enhances one full (non-final) window and returns the samples it completes.
    private func process(_ window: [Float]) async throws -> [Float] {
        if let pipeline = hybridPipeline, let accelerator = pipeline.accelerator,
           await pipeline.hybridEligible(sampleRate: sampleRate, count: window.count) {
            return try await pipelineStep(window, pipeline, accelerator)
        }
        var emitted = try await drainPipeline()
        emitted += try complete(try await enhanceWindow(window))
        return emitted
    }

    /// Overlap-adds the next window in order and emits what it completes.
    private func complete(_ enhanced: [Float]) throws -> [Float] {
        try accumulate(enhanced, isLast: false)
        chunkIndex += 1
        return emitPrefix(stride)
    }

    private func enhanceWindow(_ window: [Float]) async throws -> [Float] {
        let (samples, used) = try await enhancer.enhanceWindow(
            window, sampleRate: sampleRate, backend: requestedBackend
        )
        backendUsage.record(used)
        return samples
    }

    // MARK: ANE hybrid pipeline
    //
    // Each full 48 kHz window passes through STFT → Core ML encoder → Mamba
    // (MLX GPU) → Core ML decoders → iSTFT. Step n runs the Mamba stack of
    // window n-1 on the GPU while Core ML encodes window n and decodes window
    // n-2, so output lags input by two windows until `finish` drains the
    // pipeline. Windows still complete strictly in order.

    private struct PipelineSlot: Sendable {
        let samples: [Float]
        let tensor: CoreMLTensor
    }

    /// Encoded, waiting for the Mamba stack.
    private var latentSlot: PipelineSlot?
    /// Through the Mamba stack, waiting for the decoders.
    private var decoderSlot: PipelineSlot?
    private var pipelineFailed = false

    private var hybridPipeline: (any HybridWindowPipelining)? {
        guard requestedBackend == .aneHybrid, !pipelineFailed else { return nil }
        return enhancer as? any HybridWindowPipelining
    }

    private func drainPipeline() async throws -> [Float] {
        var emitted = [Float]()
        while latentSlot != nil || decoderSlot != nil {
            guard let pipeline = enhancer as? any HybridWindowPipelining, let accelerator = pipeline.accelerator else {
                preconditionFailure("pipeline slots without a pipelining enhancer")
            }
            emitted += try await pipelineStep(nil, pipeline, accelerator)
        }
        return emitted
    }

    /// One pipeline step; `window` is nil while draining. Any Core ML or
    /// hybrid-stage failure re-runs every window still in flight in MLX.
    private func pipelineStep(
        _ window: [Float]?, _ pipeline: any HybridWindowPipelining, _ accelerator: ReuseANEAccelerator
    ) async throws -> [Float] {
        let pendingDecode = decoderSlot
        let pendingMamba = latentSlot
        do {
            let encoderInput: CoreMLTensor? = if let window {
                try await pipeline.hybridEncoderInput(window, sampleRate: sampleRate)
            } else {
                nil
            }
            let coreML = Task.detached(priority: .userInitiated) {
                let latent = try encoderInput.map { try accelerator.runEncoder($0) }
                let decoded = try pendingDecode.map { try accelerator.runDecoders($0.tensor) }
                return (latent: latent, decoded: decoded)
            }
            var decoderInput: CoreMLTensor?
            do {
                if let pendingMamba {
                    decoderInput = try await pipeline.hybridMamba(pendingMamba.tensor)
                }
            } catch {
                _ = await coreML.result
                throw error
            }
            let (latent, decoded) = try await coreML.value
            var finished: [Float]?
            if let decoded, let pendingDecode {
                finished = try await pipeline.hybridOutput(
                    decoded, sampleCount: pendingDecode.samples.count, sampleRate: sampleRate
                )
            }

            decoderSlot = decoderInput.map { PipelineSlot(samples: pendingMamba!.samples, tensor: $0) }
            latentSlot = latent.map { PipelineSlot(samples: window!, tensor: $0) }
            guard let finished else { return [] }
            backendUsage.record(.aneHybrid)
            return try complete(finished)
        } catch let error as CancellationError {
            throw error
        } catch {
            pipelineFailed = true
            decoderSlot = nil
            latentSlot = nil
            await pipeline.hybridFailed(error)
            var emitted = [Float]()
            for samples in [pendingDecode?.samples, pendingMamba?.samples, window].compactMap({ $0 }) {
                let (enhanced, used) = try await enhancer.enhanceWindow(samples, sampleRate: sampleRate, backend: .mlx)
                backendUsage.record(used)
                emitted += try complete(enhanced)
            }
            return emitted
        }
    }

    private func accumulate(_ enhanced: [Float], isLast: Bool) throws {
        guard !enhanced.isEmpty else { return }
        if accumulation.count < enhanced.count {
            accumulation.append(contentsOf: repeatElement(0, count: enhanced.count - accumulation.count))
            normalization.append(contentsOf: repeatElement(0, count: enhanced.count - normalization.count))
        }

        let effectiveOverlap = min(overlapLength, max(0, enhanced.count - 1))
        let isFirst = chunkIndex == 0
        // Corresponding fade-in and fade-out weights sum to one because
        // sin²(x) + cos²(x) = 1, preserving constant overlap-add gain.
        for index in enhanced.indices {
            var weight: Float = 1
            if effectiveOverlap > 0 {
                if index < effectiveOverlap && !isFirst {
                    let position = Float(index) / Float(effectiveOverlap)
                    let sine = Foundation.sin(Float.pi * 0.5 * position)
                    weight = sine * sine
                } else if index >= enhanced.count - effectiveOverlap && !isLast {
                    let position = Float(index - (enhanced.count - effectiveOverlap))
                        / Float(effectiveOverlap)
                    let cosine = Foundation.cos(Float.pi * 0.5 * position)
                    weight = cosine * cosine
                }
            }
            accumulation[index] += enhanced[index] * weight
            normalization[index] += weight
        }
    }

    private func emitPrefix(_ count: Int) -> [Float] {
        let emittedCount = min(count, accumulation.count)
        guard emittedCount > 0 else { return [] }

        var output = [Float](repeating: 0, count: emittedCount)
        for index in 0..<emittedCount where normalization[index] > 1e-8 {
            output[index] = accumulation[index] / normalization[index]
        }
        accumulation.removeFirst(emittedCount)
        normalization.removeFirst(emittedCount)
        return output
    }
}
