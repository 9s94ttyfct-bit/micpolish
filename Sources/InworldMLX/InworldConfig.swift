// SPDX-License-Identifier: Apache-2.0
//
// The architecture of an Inworld Voice Isolator checkpoint, read from the
// `config` JSON in its safetensors metadata exactly as upstream
// `VoiceIsolatorConfig.from_dict` reads it (inworld-ai/voice-isolator,
// Apache-2.0, `src/inworld/voice_isolation/model.py` at the commit pinned in
// tools/parity/inworld_reference.py).

import Foundation

struct InworldConfig: Equatable, Sendable {
    enum BlockType: String, Sendable {
        case transformer
        case mamba
    }

    var width = 320
    var blockTypes: [BlockType] = [.transformer, .mamba, .transformer, .mamba, .transformer, .mamba, .transformer]
    var numHeads = 8
    var nFFT = 2048
    var hopLength = 480
    var sampleRate = 48_000
    var bandPlan: [[Int]] = [[100, 1000], [250, 4000], [500, 8000], [1000, 16000], [2000, 24000]]
    var correctionBound: Float = 0.5
    var chunkSeconds = 12.0
    var overlapSeconds = 2.0
    var exclusiveSelfAttention = true
    var exclusiveAxes = "time"
    var mambaState = 16
    var mambaConv = 4
    var mambaExpand = 2

    /// Fields upstream ignores when reading a checkpoint.
    private static let trainingOnlyFields: Set<String> = [
        "activation_checkpointing", "checkpoint_every", "attention_backend", "offline_mode_prob",
        "dual_distill", "hint_sha256", "hint_bf16", "grid_width", "grid_hidden", "grid_kernel",
        "grid_stride", "grid_heads", "grid_qk_channels", "grid_output", "grid_normalize_input",
        "decision_head",
    ]

    /// Fields of the offline, two-sided model family; anything else in the
    /// checkpoint (streaming, dual-mode, look-ahead, FiLM, several passes) is
    /// refused rather than ignored.
    private static var offlineDefaults: [String: Any] {
        [
            "causal": false, "lookahead_frames": 0, "lookahead_frames_choices": [Any](), "dual_mode": false,
            "mode_film": false, "attention_window_frames": 0, "offline_passes": 1,
            "representation": "bands", "decoder_type": "complex_mask_residual", "hint_checkpoint": NSNull(),
        ]
    }

    var headDim: Int { width / numHeads }
    var binCount: Int { nFFT / 2 + 1 }
    var mambaInner: Int { mambaExpand * width }
    /// `math.ceil(width / 16)`.
    var dtRank: Int { (width + 15) / 16 }

    /// Bin ranges `[start, end)` of the bands, covering every bin once (`band_edges`).
    var bandEdges: [Range<Int>] {
        let hzPerBin = Double(sampleRate) / Double(nFFT)
        var edges = [Range<Int>]()
        var start = 0
        for pair in bandPlan {
            // Python's round() rounds halves to even.
            let width = max(1, Int((Double(pair[0]) / hzPerBin).rounded(.toNearestOrEven)))
            let upto = min(binCount, Int((Double(pair[1]) / hzPerBin).rounded(.toNearestOrEven)))
            while start < upto {
                let end = min(start + width, upto)
                edges.append(start ..< end)
                start = end
            }
        }
        if start < binCount { edges.append(start ..< binCount) }
        return edges
    }

    /// The render window and its crossfade in samples (`window_samples`), the
    /// crossfade capped at half the window.
    var window: (chunk: Int, overlap: Int) {
        let chunk = Int(chunkSeconds * Double(sampleRate))
        return (chunk, min(Int(overlapSeconds * Double(sampleRate)), chunk / 2))
    }

    init() {}

    /// Parses the checkpoint's `config` metadata entry.
    init(json: String) throws {
        guard let data = json.data(using: .utf8),
              let raw = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw InworldIsolatorError.missingTensor("__metadata__.config")
        }
        func unsupported(_ name: String) -> InworldIsolatorError {
            .unsupportedConfig("\(name)=\(raw[name].map { "\($0)" } ?? "nil")")
        }
        func int(_ name: String) throws -> Int? {
            guard let value = raw[name] else { return nil }
            guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
                  number.doubleValue == number.doubleValue.rounded() else { throw unsupported(name) }
            return number.intValue
        }
        func double(_ name: String) throws -> Double? {
            guard let value = raw[name] else { return nil }
            guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
                  number.doubleValue.isFinite else { throw unsupported(name) }
            return number.doubleValue
        }
        func bool(_ name: String) throws -> Bool? {
            guard let value = raw[name] else { return nil }
            guard let number = value as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID() else {
                throw unsupported(name)
            }
            return number.boolValue
        }

        let known: Set<String> = [
            "width", "block_types", "num_heads", "n_fft", "hop_length", "sample_rate", "band_plan",
            "correction_bound", "chunk_seconds", "overlap_seconds", "exclusive_self_attention",
            "exclusive_axes", "mamba_state", "mamba_conv", "mamba_expand",
        ]
        for (name, value) in raw where !known.contains(name) && !Self.trainingOnlyFields.contains(name) {
            if let expected = Self.offlineDefaults[name] {
                guard Self.isEqual(value, expected) else { throw unsupported(name) }
            } else if !Self.isEmpty(value) {
                throw unsupported(name)
            }
        }

        if let value = try int("width") { width = value }
        if let value = raw["block_types"] {
            guard let names = value as? [String] else { throw unsupported("block_types") }
            blockTypes = try names.map {
                guard let type = BlockType(rawValue: $0) else { throw unsupported("block_types") }
                return type
            }
        }
        if let value = try int("num_heads") { numHeads = value }
        if let value = try int("n_fft") { nFFT = value }
        if let value = try int("hop_length") { hopLength = value }
        if let value = try int("sample_rate") { sampleRate = value }
        if let value = raw["band_plan"] {
            guard let pairs = value as? [[NSNumber]], pairs.allSatisfy({ $0.count == 2 }) else {
                throw unsupported("band_plan")
            }
            bandPlan = pairs.map { $0.map(\.intValue) }
        }
        if let value = try double("correction_bound") { correctionBound = Float(value) }
        if let value = try double("chunk_seconds") { chunkSeconds = value }
        if let value = try double("overlap_seconds") { overlapSeconds = value }
        if let value = try bool("exclusive_self_attention") { exclusiveSelfAttention = value }
        if let value = raw["exclusive_axes"] {
            guard let axes = value as? String else { throw unsupported("exclusive_axes") }
            exclusiveAxes = axes
        }
        if let value = try int("mamba_state") { mambaState = value }
        if let value = try int("mamba_conv") { mambaConv = value }
        if let value = try int("mamba_expand") { mambaExpand = value }

        // The checks of VoiceIsolatorConfig.__post_init__, plus this port's own
        // limits: the 48 kHz contract and the register-resident scan state.
        guard width >= 1, numHeads >= 1, width % (2 * numHeads) == 0 else { throw unsupported("width") }
        guard !blockTypes.isEmpty else { throw unsupported("block_types") }
        guard nFFT >= 4, nFFT % 2 == 0 else { throw unsupported("n_fft") }
        guard hopLength > 0, hopLength <= nFFT / 2 else { throw unsupported("hop_length") }
        guard sampleRate == InworldIsolator.sampleRate else { throw unsupported("sample_rate") }
        guard !bandPlan.isEmpty, bandPlan.allSatisfy({ $0[0] >= 1 && $0[1] >= 1 }) else {
            throw unsupported("band_plan")
        }
        guard correctionBound > 0, correctionBound < 1 else { throw unsupported("correction_bound") }
        guard chunkSeconds * Double(sampleRate) >= Double(nFFT) else { throw unsupported("chunk_seconds") }
        guard overlapSeconds > 0, overlapSeconds <= chunkSeconds / 2 else { throw unsupported("overlap_seconds") }
        guard ["both", "time", "band"].contains(exclusiveAxes) else { throw unsupported("exclusive_axes") }
        guard (1...InworldKernels.maxState).contains(mambaState) else { throw unsupported("mamba_state") }
        guard (1...InworldKernels.maxConv).contains(mambaConv) else { throw unsupported("mamba_conv") }
        guard mambaExpand >= 1 else { throw unsupported("mamba_expand") }
        guard window.chunk >= hopLength else { throw unsupported("chunk_seconds") }
    }

    var exclusiveTime: Bool { exclusiveSelfAttention && (exclusiveAxes == "both" || exclusiveAxes == "time") }
    var exclusiveBand: Bool { exclusiveSelfAttention && (exclusiveAxes == "both" || exclusiveAxes == "band") }

    /// Python's `value not in (0, False, None, "", [], {})`, negated.
    private static func isEmpty(_ value: Any) -> Bool {
        switch value {
        case is NSNull: return true
        case let number as NSNumber: return number.doubleValue == 0
        case let string as String: return string.isEmpty
        case let array as [Any]: return array.isEmpty
        case let dictionary as [String: Any]: return dictionary.isEmpty
        default: return false
        }
    }

    private static func isEqual(_ value: Any, _ expected: Any) -> Bool {
        switch expected {
        case is NSNull: return value is NSNull
        case let array as [Any]: return (value as? [Any])?.isEmpty == true && array.isEmpty
        case let string as String: return (value as? String) == string
        case let flag as Bool:
            guard let number = value as? NSNumber else { return false }
            return CFGetTypeID(number) == CFBooleanGetTypeID() ? number.boolValue == flag : !flag && number == 0
        case let number as Int:
            guard let actual = value as? NSNumber, CFGetTypeID(actual) != CFBooleanGetTypeID() else { return false }
            return actual.doubleValue == Double(number)
        default: return false
        }
    }
}
