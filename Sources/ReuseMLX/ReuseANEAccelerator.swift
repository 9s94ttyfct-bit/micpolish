// SPDX-License-Identifier: Apache-2.0

import CoreML
import Foundation
import MLX

public enum ReuseANEError: Error, LocalizedError, Sendable, Equatable {
    case packageMissing(String)
    case incompatiblePackage(String)
    case predictionFailed(String)

    public var errorDescription: String? {
        switch self {
        case .packageMissing(let detail): "RE-USE Core ML package missing: \(detail)"
        case .incompatiblePackage(let detail): "RE-USE Core ML package incompatible: \(detail)"
        case .predictionFailed(let detail): "RE-USE Core ML prediction failed: \(detail)"
        }
    }
}

/// Loads the compiled Core ML parts of one RE-USE checkpoint, exported by
/// `tools/coreml/export_reuse_coreml.py` for one 5 s window at 48 kHz.
///
/// Core ML compiles each `.mlpackage` once and the result is cached under
/// `~/Library/Caches/MicPolish/coreml/`, keyed by the package digests in
/// `manifest.json`.
public final class ReuseANEAccelerator: @unchecked Sendable {
    /// The only sample rate the exported shapes cover.
    public static let sampleRate = 48_000
    /// Samples per window (5 s at 48 kHz); other window lengths run in MLX.
    public static let windowSamples = 240_000
    static let frames = 1001
    static let frequencyBins = 961
    static let latentFrames = 251
    static let latentBins = 481
    static let hiddenChannels = 64
    static let formatVersion = 1

    public let packageDirectory: URL
    public let modelID: String
    public let revision: String
    public let computeUnits: MLComputeUnits

    // SAFETY: MLModel prediction is thread-safe; the models are immutable after init.
    private let encoder: MLModel
    private let decoders: MLModel
    private let compiledEncoderURL: URL
    private let compiledDecodersURL: URL
    private let configuration: MLModelConfiguration

    /// Installed package location for one checkpoint revision.
    public static func packageDirectory(modelsRoot: URL, modelID: String, revision: String) -> URL {
        modelsRoot.appendingPathComponent("\(modelID)-coreml").appendingPathComponent(revision)
    }

    /// Throws `ReuseANEError` if the package is missing or was exported for
    /// other shapes, and Core ML errors if compiling or loading fails.
    public init(
        packageDirectory: URL,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        cacheDirectory: URL? = nil
    ) throws {
        let manifest = try Self.readManifest(in: packageDirectory)
        try Self.validate(manifest)
        self.packageDirectory = packageDirectory
        self.modelID = manifest.modelID
        self.revision = manifest.revision
        self.computeUnits = computeUnits

        let cacheRoot = try cacheDirectory ?? FileManager.default.url(
            for: .cachesDirectory, in: .userDomainMask, appropriateFor: nil, create: true
        ).appendingPathComponent("MicPolish/coreml")
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        self.configuration = configuration

        // One cache folder per package digest; folders of earlier exports of
        // this revision are removed once the current ones are in place.
        let revisionCache = cacheRoot.appendingPathComponent(manifest.modelID)
            .appendingPathComponent(manifest.revision)
        let encoderEntry = manifest.packages.encoder
        let decodersEntry = manifest.packages.decoders
        let cacheKeys = [encoderEntry, decodersEntry].map { String($0.sha256.prefix(16)) }
        compiledEncoderURL = try Self.compiled(
            packageDirectory.appendingPathComponent(encoderEntry.file),
            cacheDirectory: revisionCache.appendingPathComponent(cacheKeys[0])
        )
        compiledDecodersURL = try Self.compiled(
            packageDirectory.appendingPathComponent(decodersEntry.file),
            cacheDirectory: revisionCache.appendingPathComponent(cacheKeys[1])
        )
        let cached = (try? FileManager.default.contentsOfDirectory(atPath: revisionCache.path)) ?? []
        for stale in cached where !cacheKeys.contains(stale) {
            try? FileManager.default.removeItem(at: revisionCache.appendingPathComponent(stale))
        }
        encoder = try MLModel(contentsOf: compiledEncoderURL, configuration: configuration)
        decoders = try MLModel(contentsOf: compiledDecodersURL, configuration: configuration)
        try Self.validateInterface(encoder, input: "x", inputShape: [1, 2, Self.frames, Self.frequencyBins], output: "h")
        try Self.validateInterface(decoders, input: "h",
                                   inputShape: [1, Self.hiddenChannels, Self.latentFrames, Self.latentBins], output: "y")
        ReuseLog.inference.info(
            "Loaded RE-USE Core ML parts for \(manifest.modelID, privacy: .public)@\(manifest.revision, privacy: .public) (\(Self.describe(computeUnits), privacy: .public))"
        )
    }

    /// Compute units requested from Core ML, for display. Core ML decides the
    /// actual placement per operation; see `computePlacement()`.
    public var computeUnitsDescription: String { Self.describe(computeUnits) }

    /// Per-operation device placement Core ML plans for the loaded models
    /// (`MLComputePlan`, macOS 14.4+).
    public func computePlacement() async throws -> ReuseANEPlacement {
        guard #available(macOS 14.4, *) else {
            return ReuseANEPlacement(encoder: [:], decoders: [:])
        }
        async let encoderCounts = Self.placementCounts(compiledEncoderURL, configuration)
        async let decoderCounts = Self.placementCounts(compiledDecodersURL, configuration)
        return try await ReuseANEPlacement(encoder: encoderCounts, decoders: decoderCounts)
    }

    // MARK: Inference
    //
    // `encode`/`decode` run a whole step on the calling thread. The pipelined
    // session splits each step: `coreMLInput` and `mlxArray(fromFloat16:)` touch
    // MLX and stay on the MLX thread; `runEncoder`/`runDecoders` only touch
    // Core ML and may run concurrently with MLX GPU work.

    static let latentShape = [1, hiddenChannels, latentFrames, latentBins]
    static let decodedShape = [1, 3, frames, frequencyBins]

    /// `x`: `[1, 2, 1001, 961]` (magnitude, phase; `[B, C, T, F]`), any float
    /// dtype. Returns the encoder latent `[1, 64, 251, 481]` as float16.
    func encode(_ x: MLXArray) throws -> MLXArray {
        Self.mlxArray(fromFloat16: try runEncoder(try Self.coreMLInput(x)).value, shape: Self.latentShape)
    }

    /// `h`: `[1, 64, 251, 481]`. Returns `[1, 3, 1001, 961]` float16:
    /// magnitude, phase real part, phase imaginary part.
    func decode(_ h: MLXArray) throws -> MLXArray {
        Self.mlxArray(fromFloat16: try runDecoders(try Self.coreMLInput(h)).value, shape: Self.decodedShape)
    }

    /// Core ML encoder on a `coreMLInput` of `[1, 2, 1001, 961]`. No MLX calls.
    func runEncoder(_ x: CoreMLTensor) throws -> CoreMLTensor {
        try predict(encoder, input: "x", x, output: "h", outputShape: Self.latentShape)
    }

    /// Core ML decoders on a `coreMLInput` of `[1, 64, 251, 481]`. No MLX calls.
    func runDecoders(_ h: CoreMLTensor) throws -> CoreMLTensor {
        try predict(decoders, input: "h", h, output: "y", outputShape: Self.decodedShape)
    }

    /// Copies an MLX array into a float16 `MLMultiArray` the caller owns, so
    /// the prediction can run off the MLX thread. Evaluates `array`.
    static func coreMLInput(_ array: MLXArray) throws -> CoreMLTensor {
        let array16 = array.asType(.float16)
        let data = array16.asData(access: .noCopyIfContiguous)
        let multiArray = try MLMultiArray(shape: data.shape.map { NSNumber(value: $0) }, dataType: .float16)
        guard multiArray.strides.map(\.intValue) == contiguousStrides(data.shape) else {
            throw ReuseANEError.predictionFailed("unexpected MLMultiArray layout \(multiArray.strides)")
        }
        multiArray.withUnsafeMutableBytes { destination, _ in
            data.data.withUnsafeBytes { destination.copyMemory(from: $0) }
        }
        withExtendedLifetime(array16) {}
        return CoreMLTensor(value: multiArray)
    }

    private func predict(
        _ model: MLModel, input: String, _ tensor: CoreMLTensor, output: String, outputShape: [Int]
    ) throws -> CoreMLTensor {
        let features = try MLDictionaryFeatureProvider(dictionary: [input: MLFeatureValue(multiArray: tensor.value)])
        let result = try model.prediction(from: features)
        guard let out = result.featureValue(for: output)?.multiArrayValue else {
            throw ReuseANEError.predictionFailed("missing output \(output)")
        }
        guard out.shape.map(\.intValue) == outputShape, out.dataType == .float16 else {
            throw ReuseANEError.predictionFailed(
                "output \(output) is \(out.shape) \(out.dataType.rawValue), expected \(outputShape) float16"
            )
        }
        return CoreMLTensor(value: out)
    }

    /// Copies a float16 `MLMultiArray` (possibly with padded rows, as the ANE
    /// returns them) into one contiguous MLX array.
    static func mlxArray(fromFloat16 multiArray: MLMultiArray, shape: [Int]) -> MLXArray {
        let strides = multiArray.strides.map(\.intValue)
        return multiArray.withUnsafeBytes { raw in
            if strides == contiguousStrides(shape) {
                let byteCount = shape.reduce(1, *) * MemoryLayout<Float16>.size
                return MLXArray(UnsafeRawBufferPointer(rebasing: raw[0 ..< byteCount]), shape, type: Float16.self)
            }
            let source = raw.bindMemory(to: Float16.self)
            let outer = Array(shape.dropLast())
            let rowLength = shape.last!
            let rowCount = outer.reduce(1, *)
            let packed = [Float16](unsafeUninitializedCapacity: rowCount * rowLength) { buffer, count in
                var index = [Int](repeating: 0, count: outer.count)
                for row in 0 ..< rowCount {
                    var offset = 0
                    for axis in outer.indices { offset += index[axis] * strides[axis] }
                    let destination = buffer.baseAddress! + row * rowLength
                    if strides.last == 1 {
                        destination.update(from: source.baseAddress! + offset, count: rowLength)
                    } else {
                        for column in 0 ..< rowLength {
                            destination[column] = source[offset + column * strides.last!]
                        }
                    }
                    var axis = outer.count - 1
                    while axis >= 0 {
                        index[axis] += 1
                        if index[axis] < outer[axis] { break }
                        index[axis] = 0
                        axis -= 1
                    }
                }
                count = rowCount * rowLength
            }
            return packed.withUnsafeBufferPointer { MLXArray($0, shape) }
        }
    }

    private static func contiguousStrides(_ shape: [Int]) -> [Int] {
        var strides = [Int](repeating: 1, count: shape.count)
        for axis in stride(from: shape.count - 2, through: 0, by: -1) {
            strides[axis] = strides[axis + 1] * shape[axis + 1]
        }
        return strides
    }

    // MARK: Package handling

    struct Manifest: Decodable {
        struct Tensor: Decodable { let name: String; let shape: [Int] }
        struct Package: Decodable { let file: String; let input: Tensor; let output: Tensor; let sha256: String }
        struct Packages: Decodable { let encoder: Package; let decoders: Package }
        let formatVersion: Int
        let modelID: String
        let revision: String
        let sampleRate: Int
        let windowSamples: Int
        let packages: Packages
    }

    static func readManifest(in directory: URL) throws -> Manifest {
        let url = directory.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ReuseANEError.packageMissing(url.path)
        }
        do {
            return try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: url))
        } catch {
            throw ReuseANEError.incompatiblePackage("unreadable manifest: \(error.localizedDescription)")
        }
    }

    static func validate(_ manifest: Manifest) throws {
        guard manifest.formatVersion == formatVersion else {
            throw ReuseANEError.incompatiblePackage("format version \(manifest.formatVersion)")
        }
        guard manifest.sampleRate == sampleRate, manifest.windowSamples == windowSamples else {
            throw ReuseANEError.incompatiblePackage("exported for \(manifest.windowSamples) samples at \(manifest.sampleRate) Hz")
        }
        let expected: [(Manifest.Package, [Int], [Int])] = [
            (manifest.packages.encoder, [1, 2, frames, frequencyBins], [1, hiddenChannels, latentFrames, latentBins]),
            (manifest.packages.decoders, [1, hiddenChannels, latentFrames, latentBins], [1, 3, frames, frequencyBins]),
        ]
        for (package, input, output) in expected where package.input.shape != input || package.output.shape != output {
            throw ReuseANEError.incompatiblePackage("\(package.file) shapes \(package.input.shape) -> \(package.output.shape)")
        }
    }

    private static func validateInterface(_ model: MLModel, input: String, inputShape: [Int], output: String) throws {
        let description = model.modelDescription
        guard let inputDescription = description.inputDescriptionsByName[input]?.multiArrayConstraint,
              inputDescription.shape.map(\.intValue) == inputShape,
              description.outputDescriptionsByName[output] != nil else {
            throw ReuseANEError.incompatiblePackage("model interface does not match \(input) \(inputShape) -> \(output)")
        }
    }

    /// Returns the cached compiled model, compiling the package on first use.
    private static func compiled(_ package: URL, cacheDirectory: URL) throws -> URL {
        let fm = FileManager.default
        guard fm.fileExists(atPath: package.path) else {
            throw ReuseANEError.packageMissing(package.path)
        }
        let destination = cacheDirectory.appendingPathComponent(
            package.deletingPathExtension().lastPathComponent + ".mlmodelc"
        )
        if fm.fileExists(atPath: destination.appendingPathComponent("coremldata.bin").path) {
            return destination
        }
        let temporary = try MLModel.compileModel(at: package)
        try fm.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)
        try? fm.removeItem(at: destination)
        do {
            try fm.moveItem(at: temporary, to: destination)
        } catch {
            // A concurrent loader may have won the race; use its copy.
            guard fm.fileExists(atPath: destination.appendingPathComponent("coremldata.bin").path) else { throw error }
            try? fm.removeItem(at: temporary)
        }
        return destination
    }

    static func describe(_ units: MLComputeUnits) -> String {
        switch units {
        case .cpuOnly: "CPU"
        case .cpuAndGPU: "CPU + GPU"
        case .cpuAndNeuralEngine: "CPU + Neural Engine"
        case .all: "CPU + GPU + Neural Engine"
        @unknown default: "unknown"
        }
    }

    @available(macOS 14.4, *)
    private static func placementCounts(
        _ compiledURL: URL, _ configuration: MLModelConfiguration
    ) async throws -> [String: Int] {
        let plan = try await MLComputePlan.load(contentsOf: compiledURL, configuration: configuration)
        guard case .program(let program) = plan.modelStructure,
              let main = program.functions["main"] else { return [:] }
        var counts = [String: Int]()
        for operation in main.block.operations {
            // Constants carry no compute; count only operations Core ML schedules.
            guard operation.operatorName != "const",
                  let usage = plan.deviceUsage(for: operation) else { continue }
            counts[deviceName(usage.preferred), default: 0] += 1
        }
        return counts
    }

    @available(macOS 14.4, *)
    private static func deviceName(_ device: MLComputeDevice) -> String {
        switch device {
        case .cpu: "CPU"
        case .gpu: "GPU"
        case .neuralEngine: "Neural Engine"
        @unknown default: "other"
        }
    }
}

/// Operation counts by the device Core ML prefers for each, per model.
public struct ReuseANEPlacement: Sendable, Equatable, CustomStringConvertible {
    public let encoder: [String: Int]
    public let decoders: [String: Int]

    /// True when every scheduled operation of both models prefers the Neural Engine.
    public var isFullyOnNeuralEngine: Bool {
        let all = [encoder, decoders]
        return all.allSatisfy { !$0.isEmpty && Set($0.keys) == ["Neural Engine"] }
    }

    public var description: String {
        func line(_ counts: [String: Int]) -> String {
            counts.sorted { $0.key < $1.key }.map { "\($0.key) \($0.value)" }.joined(separator: ", ")
        }
        return "encoder: \(line(encoder)); decoders: \(line(decoders))"
    }
}

/// A float16 `MLMultiArray` handed between the MLX thread and Core ML.
// SAFETY: a tensor is fully written before it is shared and only read afterwards.
struct CoreMLTensor: @unchecked Sendable {
    let value: MLMultiArray
}
