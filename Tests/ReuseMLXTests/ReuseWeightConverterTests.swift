import XCTest
import ReuseMLX

final class ReuseWeightConverterTests: XCTestCase {
    func testMapKeyRenamesNestedModules() {
        XCTAssertEqual(
            ReuseWeightConverter.mapKey("dense_encoder.dense_block.dense_block.0.1.weight"),
            "denseEncoder.denseBlock.layers.0.layers.1.weight")
        XCTAssertEqual(
            ReuseWeightConverter.mapKey("TSMamba.3.time_mamba.forward_blocks.0.mixer.in_proj.weight"),
            "tfMamba.3.timeMamba.forward.0.mixer.inProj.weight")
        XCTAssertEqual(
            ReuseWeightConverter.mapKey("mask_decoder.up_conv1.0.weight"),
            "maskDecoder.upConv1.layers.0.weight")
    }

    func testConvertNemotronCheckpoint() throws {
        let source = try ReuseTestModel.sourceCheckpoint()
        let output = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathComponent("model_mlx.safetensors")
        defer { try? FileManager.default.removeItem(at: output.deletingLastPathComponent()) }

        XCTAssertEqual(try ReuseWeightConverter.convert(input: source, output: output), 1416)
        XCTAssertNoThrow(try MambaEnhancer(modelWeightsURL: output))
    }
}
