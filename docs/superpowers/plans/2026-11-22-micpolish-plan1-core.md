# MicPolish Plan 1 — 核心可用版（录音 → RE-USE → 母带 → 导出）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 做出一个能日常使用的 MicPolish：录音或导入 → RE-USE（MLX，NemotronLabs 权重）→ 可实时试听的母带处理链 → 导出 M4A/MP3，原始音频以 FLAC 保留，中间音频自动清理。

**Architecture:** 一个 Swift Package（无 .xcodeproj），若干库模块 + 一个 CLI + 一个 SwiftUI 可执行目标，由脚本打包成 `.app`。RE-USE 推理代码从 SpeechLens（Apache-2.0）引入；母带 DSP 用 Accelerate 自写；处理流水线以"阶段（Stage）"组织，Plan 2/3/4 往里加 LavaSR、Inworld 和 ANE 后端。

**Tech Stack:** Swift 6.4 / Xcode 27.2、mlx-swift 0.31.6、swift-argument-parser 1.8.2、AVFoundation、AudioToolbox、Accelerate、LAME（动态库，MP3）、XCTest；开发期 Python 只用 `uv run`（对拍参考）。

**Spec:** `docs/superpowers/specs/2026-11-22-micpolish-design.md`

### 路线图（本计划之后）

整份设计拆成 4 个计划，每个都产出可用软件：

| 计划 | 内容 | 依赖 |
|---|---|---|
| **Plan 1（本文）** | 核心可用版 | — |
| Plan 2 | LavaSR v2 MLX 移植 + 高频增强阶段 + 盲听工具（§7.8） | Plan 1 |
| Plan 3 | Inworld Voice Isolator MLX 移植 + 强降噪阶段 | Plan 1 |
| Plan 4 | ANE 后端（RE-USE 混合、LavaSR 主干）+ Core ML 导出脚本 + 每机测速 | Plan 1、2 |

Plan 2/3/4 在本计划完成后各自单独写。

## Global Constraints

- 最低系统 macOS 14；仅 Apple Silicon；Package `platforms: [.macOS(.v14)]`，`swift-tools-version: 6.3`。
- 依赖固定：`mlx-swift` exact `0.31.6`，`swift-argument-parser` exact `1.8.2`。不引入其他 Swift 包。
- 运行时不调用 Python，不打包 FFmpeg。
- RE-USE 权重只用 `nvidia/NVIDIA-NemotronLabs-RE-USE` revision `31f82f2efad1095034492f236e8dac9239ba713b`，`model.safetensors` 38592940 字节，SHA-256 `06badaf4029b297f97f7fe48d62990b48d5482608210df577354e0d4e0ed70e1`；`-laughter` 变体 revision `1b1521dd659608e1e063ada83e0acfe1da473a8b`，SHA-256 `35c2a4e284515275d2e49a8e8cb5cd392ae67a1f1d345de1c12a0202c7b6b610`，大小相同。权重不入库、不随应用分发。
- 从 SpeechLens（commit `0c0dcbd7d29b57d62d90eeadeb42496a6b794a1f`，Apache-2.0）复制的文件保留原 SPDX 头，NOTICE 注明来源。**不得**复制 `tools/reference_runner.py` 及任何 NSCLv1 文件。
- 路径：会话根 `~/Music/MicPolish/`；模型 `~/Library/Application Support/MicPolish/models/`；工作区 `~/Library/Caches/MicPolish/work/<会话 UUID>/`。所有这些根路径都通过参数注入，测试一律用临时目录。
- 处理内部采样率 48000 Hz、单声道、Float32。原始录音存 FLAC 24-bit 48 kHz 单声道。
- 成品只有 M4A（AAC-LC）或 MP3，单声道 48 kHz，码率 96/128/192 kbps，默认 128。
- 母带默认值：高通 80 Hz 12 dB/oct；压缩 −20 dB、3:1、起音 10 ms、释放 150 ms；响度 −16 LUFS（单声道按双单声道计量，见 Task 6）；真峰值上限 −1.0 dBTP，4 倍过采样检测。
- Bundle ID `io.github.9s94ttyfct-bit.micpolish`；应用名 MicPolish；界面中文为主，字符串放 String Catalog 并附英文。
- 许可：仓库 Apache-2.0；NOTICE 列 SpeechLens、NVIDIA RE-USE（OpenMDW-1.1，运行时下载）、LAME（LGPL-2.0，动态库）。

## Review Focus

1. **超长录音（≥ 1 小时）**：处理和导出都必须流式进行，内存不随时长增长 → Task 9 加 `testTwentyMinuteInputKeepsPeakMemoryBounded`。
2. **全静音或极短录音（< 1 s）**：响度测不出时不能放大噪声、不能产生 NaN 或崩溃 → Task 6 加 `testSilenceProducesSilenceWithoutGain`，Task 9 加 `testHalfSecondInputProcesses`。
3. **处理中途取消或退出**：残缺的中间文件绝不能被当作有效缓存复用 → Task 9 加 `testCancelledStageIsRecomputed`。
4. **导出失败（目标不可写、磁盘满）**：原始音频和会话不受影响，旧成品不被半截文件替换 → Task 7 加 `testFailedExportLeavesExistingFileIntact`。
5. **外接麦克风不是 48 kHz / 多声道**：录音须转为 48 kHz 单声道且时长正确 → Task 11 加 `testSink44k1StereoBecomes48kMono`。

---

### Task 1: 仓库骨架与 MLX 构建链

**Files:**
- Create: `Package.swift`、`.gitignore`、`LICENSE`（Apache-2.0 全文）、`NOTICE`、`README.md`（一段简介即可，Task 14 再补全）
- Create: `tools/build_mlx_metallib.sh`（从 SpeechLens 复制，改名中的 `speechlens` 字样为 `micpolish`）、`tools/test.sh`
- Create: `Sources/AudioCore/AudioCore.swift`（占位导出 `public enum AudioCore { public static let sampleRate = 48_000 }`，后续任务扩充）
- Test: `Tests/AudioCoreTests/SmokeTests.swift`

**Interfaces:**
- Produces: Package 目标名（后续任务往里加文件）：库 `AudioCore`、`ReuseMLX`、`ModelStore`、`Mastering`、`Export`、`Sessions`、`Pipeline`、`Recording`、`Preview`；可执行 `micpolish`（CLI）、`MicPolishApp`；测试目标与库同名加 `Tests`。本任务只声明已有源文件的目标，其余目标在各自任务里加入 `Package.swift`。
- Produces: `tools/test.sh [swift test 参数…]`：先确保 `.build/debug/mlx.metallib` 存在（调用 build 脚本），再执行 `swift test "$@"`。

- [ ] **Step 1:** `git init`，写 `.gitignore`（`.build/`、`.superpowers/`、`*.xcodeproj`、`dist/`、`Tests/Fixtures/**/model*.safetensors`）。
- [ ] **Step 2:** 写 `SmokeTests.testSampleRateConstant`：`XCTAssertEqual(AudioCore.sampleRate, 48_000)`。
- [ ] **Step 3:** 写 `Package.swift`（依赖按 Global Constraints）与 `AudioCore.swift`。
- [ ] **Step 4:** Run: `tools/test.sh --filter SmokeTests` → Expected: 生成 `.build/debug/mlx.metallib`，1 test passed。
- [ ] **Step 5:** Commit `chore: scaffold MicPolish package and MLX metallib build`.

### Task 2: 引入 RE-USE MLX 推理与权重转换

**Files:**
- Create: `Sources/ReuseMLX/` ← 复制 SpeechLens `Sources/Inference/*.swift`（除 README）；`SpeechLensLog` 的 subsystem 改为 `io.github.9s94ttyfct-bit.micpolish`
- Create: `Sources/ReuseMLX/ReuseWeightConverter.swift`（逻辑来自 SpeechLens `Sources/WeightPort/WeightPort.swift`）
- Test: `Tests/ReuseMLXTests/ReuseWeightConverterTests.swift`、`Tests/ReuseMLXTests/ReuseEnhancerSmokeTests.swift`

**Interfaces:**
- Produces（沿用 SpeechLens，不改签名）：`public actor MambaEnhancer { init(modelWeightsURL: URL, mode: InferenceMode = .standard) throws; func makeSession(sampleRate: Int, settings: InferenceSettings) throws -> any SpeechEnhancementSession }`；`SpeechEnhancementSession.append(_:) async throws -> [Float]`、`finish() async throws -> [Float]`；`InferenceSettings.standard`（5 s / 0.05）。
- Produces: `public enum ReuseWeightConverter { static func mapKey(_ key: String) -> String; static func convert(input: URL, output: URL) throws -> Int /* 写出的张量数 */ }`。

- [ ] **Step 1:** 写 `testMapKeyRenamesNestedModules`：断言 `mapKey("dense_encoder.dense_block.dense_block.0.1.weight") == "denseEncoder.denseBlock.layers.0.layers.1.weight"`、`mapKey("TSMamba.3.time_mamba.forward_blocks.0.mixer.in_proj.weight") == "tfMamba.3.timeMamba.forward.0.mixer.inProj.weight"`、`mapKey("mask_decoder.up_conv1.0.weight") == "maskDecoder.upConv1.layers.0.weight"`。
- [ ] **Step 2:** 写 `testConvertNemotronCheckpoint`（模型文件路径取环境变量 `MICPOLISH_REUSE_SOURCE`，缺失则 `XCTSkip`）：`convert` 返回 1416；转换后能 `MambaEnhancer(modelWeightsURL:)` 成功初始化。
- [ ] **Step 3:** 写 `ReuseEnhancerSmokeTests.testOneSecondAt48kKeepsLength`（同样依赖转换后的权重）：48 kHz 1 s 正弦加白噪声，`append` + `finish` 的总长度 = 48000，全部有限值。
- [ ] **Step 4:** Run: `tools/test.sh --filter ReuseMLXTests` → Expected: 编译失败（类型不存在）。
- [ ] **Step 5:** 复制源码、实现 `ReuseWeightConverter`，Package 加 `ReuseMLX` 目标（依赖 MLX、MLXFast、MLXNN、MLXRandom）。NOTICE 记来源与 commit。
- [ ] **Step 6:** 下载权重：`uvx --from huggingface_hub hf download nvidia/NVIDIA-NemotronLabs-RE-USE model.safetensors --revision 31f82f2efad1095034492f236e8dac9239ba713b --local-dir /tmp/reuse-nemo`，`shasum -a 256` 与 Global Constraints 一致。
- [ ] **Step 7:** Run: `MICPOLISH_REUSE_SOURCE=/tmp/reuse-nemo/model.safetensors tools/test.sh --filter ReuseMLXTests` → Expected: 3 tests passed，0 skipped。
- [ ] **Step 8:** Commit `feat: vendor RE-USE MLX inference and NemotronLabs weight converter`.

### Task 3: RE-USE 数值对拍

**Files:**
- Create: `tools/parity/reuse_reference.py`（PEP 723 头声明依赖：`torch`（CPU）、`einops`、`safetensors`、`numpy`、`soundfile`、`huggingface_hub`、`pyyaml`、`librosa`）
- Create: `Tests/Fixtures/reuse/`：`input_48k.wav`、`input_16k.wav`、`ref_48k.f32`、`ref_16k.f32`、`README.md`（来源与 OpenMDW-1.1 说明）
- Test: `Tests/ReuseMLXTests/ReuseParityTests.swift`

**Interfaces:**
- Consumes: Task 2 的 `MambaEnhancer`、`ReuseWeightConverter`。
- Produces: 参考脚本 CLI：`uv run tools/parity/reuse_reference.py --weights <model.safetensors> --input <wav> --output <f32 raw little-endian>`。

- [ ] **Step 1:** 写参考脚本：从 NemotronLabs 仓库同一 revision 下载 `models/*.py`、`utils/util.py`、recipe yaml；在 `import` 之前向 `sys.modules` 注入一个名为 `mamba_ssm` 的模块，其 `Mamba` 用纯 PyTorch 选择性扫描实现（参数名与 `mamba_ssm.Mamba` 一致：`in_proj`、`conv1d`、`x_proj`、`dt_proj`、`A_log`、`D`、`out_proj`）。对整段输入按上游 `inference.py` 的步骤做一次前向（不分块），含"扫频伪影"抑制，写出 float32。
- [ ] **Step 2:** 生成夹具：取上游 `noisy_audio/mic_test2.wav` 的前 4.0 s，分别重采样为 48 kHz 与 16 kHz 写成 `input_*.wav`，再跑参考脚本得到 `ref_*.f32`。
- [ ] **Step 3:** 写 `testParity48k`、`testParity16k`：Swift 侧对同一输入用 `InferenceSettings(chunkSeconds: 4.0, overlapPortion: 0.05)`（单块，避免拼接差异）跑 MLX，计算 `relRMSE = rms(mlx − ref) / rms(ref)`，断言 `≤ 1e-4`。
- [ ] **Step 4:** Run: `MICPOLISH_REUSE_SOURCE=… tools/test.sh --filter ReuseParityTests` → Expected: 2 passed。若不过，按 superpowers:systematic-debugging 逐层比对（SpeechLens 的探针思路：编码器输出、第 0 块输出、最终掩码），不得放宽阈值。
- [ ] **Step 5:** Commit `test: RE-USE MLX parity against PyTorch reference at 48k and 16k`.

### Task 4: 模型管理（下载、校验、转换）

**Files:**
- Create: `Sources/ModelStore/ModelDescriptor.swift`、`ModelStore.swift`、`Downloader.swift`
- Create: `Resources/licenses/OpenMDW-1.1.txt`（从 NemotronLabs 仓库 `LICENSE` 复制）
- Test: `Tests/ModelStoreTests/ModelStoreTests.swift`

**Interfaces:**
- Consumes: `ReuseWeightConverter.convert(input:output:)`。
- Produces:
  - `public struct ModelDescriptor: Sendable, Hashable { let id: String; let repo: String; let revision: String; let file: String; let sha256: String; let byteCount: Int; let licenseResource: String; var sourceURL: URL }`（`sourceURL` = `https://huggingface.co/<repo>/resolve/<revision>/<file>`）
  - `public enum ModelCatalog { static let reuse: ModelDescriptor; static let reuseLaughter: ModelDescriptor }`（id 分别为 `reuse-nemotron`、`reuse-nemotron-laughter`，值按 Global Constraints）
  - `public protocol Downloader: Sendable { func download(_ url: URL, to destination: URL, progress: @Sendable @escaping (Double) -> Void) async throws }`；`public struct URLSessionDownloader: Downloader`
  - `public actor ModelStore { init(root: URL, downloader: any Downloader = URLSessionDownloader()); func installedWeightsURL(_ d: ModelDescriptor) -> URL?; func install(_ d: ModelDescriptor, progress: @Sendable @escaping (Double) -> Void) async throws -> URL; func remove(_ d: ModelDescriptor) throws; func diskUsageBytes() -> Int64 }`
  - `public enum ModelStoreError: Error { case checksumMismatch(expected: String, actual: String); case sizeMismatch(expected: Int, actual: Int) }`
- 目录布局：`<root>/<id>/<revision>/model_mlx.safetensors` + `LICENSE.txt`；安装在 `<root>/.staging/` 完成后原子移动，源文件转换后删除。

- [ ] **Step 1:** 写测试（用一个把本地文件复制过去的 `FakeDownloader`）：`testChecksumMismatchThrowsAndLeavesNothingInstalled`（错误哈希 → 抛 `checksumMismatch`，`installedWeightsURL == nil`，`.staging` 为空）；`testInstallConvertsAndIsIdempotent`（真实权重，环境变量缺失则跳过：安装后文件存在，第二次 `install` 不再调用下载器）；`testRemoveDeletesFiles`。
- [ ] **Step 2:** Run → Expected: 编译失败。
- [ ] **Step 3:** 实现；SHA-256 用 CryptoKit 流式计算。
- [ ] **Step 4:** Run: `MICPOLISH_REUSE_SOURCE=… tools/test.sh --filter ModelStoreTests` → Expected: 3 passed。
- [ ] **Step 5:** Commit `feat: model store with pinned download, checksum and local conversion`.

### Task 5: 音频读写与重采样

**Files:**
- Modify: `Sources/AudioCore/AudioCore.swift`
- Create: `Sources/AudioCore/MonoFileReader.swift`、`AudioFileWriters.swift`、`Resampler.swift`
- Test: `Tests/AudioCoreTests/AudioCoreTests.swift`

**Interfaces:**
- Produces:
  - `public final class MonoFileReader { init(url: URL) throws; let sampleRate: Int; let frameCount: Int64; func read(maxFrames: Int) throws -> [Float]? /* nil = EOF；多声道取平均 */ }`
  - `public final class MonoFileWriter { enum Format { case cafFloat32, flac24 }; init(url: URL, sampleRate: Int, format: Format) throws; func append(_ samples: [Float]) throws; func close() throws }`
  - `public final class StreamingResampler { init(from: Int, to: Int) throws; func process(_ samples: [Float]) throws -> [Float]; func flush() throws -> [Float] }`（AVAudioConverter，最高质量）
  - `public func readAllMono(_ url: URL) throws -> (samples: [Float], sampleRate: Int)`（仅供测试与短文件）

- [ ] **Step 1:** 写测试：`testFlac24RoundTripWithinOneLSB`（随机信号 ±0.5，最大误差 ≤ 2⁻²³ × 2）；`testStereoFileDownmixesToMean`；`testResample16kTo48kPreservesToneAndLength`（1 kHz 正弦 2 s → 长度 96000 ±1，FFT 峰值在 1 kHz ±5 Hz）；`testChunkedResampleEqualsWhole`（分 1000 帧块处理与一次处理结果最大差 ≤ 1e-6）。
- [ ] **Step 2:** Run → Expected: 失败。
- [ ] **Step 3:** 实现。
- [ ] **Step 4:** Run: `tools/test.sh --filter AudioCoreTests` → Expected: 5 passed（含 Task 1 的 smoke）。
- [ ] **Step 5:** Commit `feat: mono file IO (CAF/FLAC) and streaming resampler`.

### Task 6: 母带 DSP 与响度计量

**Files:**
- Create: `Sources/Mastering/MasteringSettings.swift`、`Biquad.swift`、`Compressor.swift`、`DeEsser.swift`、`TruePeakLimiter.swift`、`LoudnessMeter.swift`、`MasteringChain.swift`
- Test: `Tests/MasteringTests/LoudnessMeterTests.swift`、`MasteringChainTests.swift`

**Interfaces:**
- Produces:
  - `public struct MasteringSettings: Codable, Equatable, Sendable`，字段：`highpass(enabled, frequencyHz = 80)`、`eq(enabled, lowShelf(freq, gainDB), peak1(freq, gainDB, q), peak2(…), highShelf(…))`、`deEsser(enabled, frequencyHz = 6000, thresholdDB = -30, ratio = 4)`、`compressor(enabled, thresholdDB = -20, ratio = 3, attackMs = 10, releaseMs = 150, autoMakeup = true)`、`loudness(enabled, targetLUFS = -16)`、`limiter(enabled, ceilingDBTP = -1.0)`；`static let default`；`static let macbookMicCorrection`（EQ：低架 120 Hz +2 dB、峰值 250 Hz −2 dB Q 1.0、峰值 4 kHz +1.5 dB Q 1.2、高架 10 kHz +1 dB；其余同 default。数值是盲听前的起点）。
  - `public final class MasteringChain { init(settings: MasteringSettings, sampleRate: Int); func update(_ settings: MasteringSettings); var loudnessGainDB: Double; func process(_ samples: inout [Float]); func reset() }` —— 顺序：高通 → EQ → 去齿音 → 压缩 → 响度增益 → 限幅。输出与分块大小无关（逐样本一致）。
  - `public struct LoudnessMeter { init(sampleRate: Int); mutating func add(_ samples: [Float]); var integratedLUFS: Double? /* 无有效门限块时 nil */; var truePeakDBTP: Double }`：ITU-R BS.1770-4（K 加权、400 ms 块 75% 重叠、−70 LUFS 绝对门限与 −10 LU 相对门限），**单声道按双单声道计量**（通道和 ×2，即 +3.01 dB），真峰值 4× 过采样。
  - `public enum MasteringRenderer { static func render(input: MonoFileReader, settings: MasteringSettings, output: MonoFileWriter) throws -> RenderReport }`，`RenderReport { measuredLUFS: Double?; appliedGainDB: Double; outputLUFS: Double?; outputTruePeakDBTP: Double }`。两遍：第一遍跑到压缩器后计量响度（不写盘），算增益 = target − measured（measured 为 nil 时增益 0），第二遍完整处理写盘。增益上限 +30 dB。

- [ ] **Step 1:** 写响度测试：`testSine1kAtMinus23IsMinus23LUFS`（峰值幅度 10^(−23/20) 的 1 kHz 正弦 10 s → −23.0 ±0.1）；`testGatingIgnoresSilence`（同上信号前后各加 10 s 数字静音 → 仍 −23.0 ±0.1）；`testSilenceIsNil`；`testTruePeakDetectsIntersamplePeak`（fs/4 正弦、相位 45°、采样峰值 0.5 → 真峰值 −6.02 ±0.2 dBTP，而非采样峰值的 −9.03）。
- [ ] **Step 2:** 写链路测试：`testAllBypassedIsIdentity`（所有级 `enabled = false` → 输出与输入逐样本相等）；`testBlockSizeInvariance`（同一 10 s 语音状随机信号，分块 256 / 4096 / 整段处理，三者逐样本相等）；`testHighpassAttenuates40Hz`（40 Hz 正弦衰减 ≥ 10 dB，1 kHz 衰减 ≤ 0.5 dB）；`testRenderHitsTargetLoudness`（−30 LUFS 左右的调制噪声 20 s → 输出积分响度 −16 ±0.5 LUFS，真峰值 ≤ −1.0 dBTP）；`testSilenceProducesSilenceWithoutGain`（全零输入 → 全零输出，`appliedGainDB == 0`，无 NaN）。
- [ ] **Step 3:** Run → Expected: 失败。
- [ ] **Step 4:** 实现（biquad 系数按 RBJ Audio EQ Cookbook；限幅器带前视，前视长度构建者定）。
- [ ] **Step 5:** Run: `tools/test.sh --filter MasteringTests` → Expected: 9 passed。
- [ ] **Step 6:** Commit `feat: mastering chain with BS.1770 loudness and true-peak limiter`.

### Task 7: 导出（M4A / MP3）

**Files:**
- Create: `tools/build_lame.sh`（下载 LAME 3.100 源码 tarball，校验其 SHA-256，`./configure --disable-static --enable-shared --disable-frontend` 为 arm64 编译，安装到 `vendor/lame/`，`install_name` 设为 `@rpath/libmp3lame.0.dylib`）
- Create: `Sources/CLame/module.modulemap`、`Sources/CLame/shim.h`（系统库目标，头文件指向 `vendor/lame/include`）
- Create: `Sources/Export/ExportOptions.swift`、`Exporter.swift`、`AACEncoder.swift`、`MP3Encoder.swift`
- Test: `Tests/ExportTests/ExportTests.swift`

**Interfaces:**
- Consumes: `MonoFileReader`、`readAllMono`。
- Produces:
  - `public enum ExportFormat: String, Codable, Sendable { case m4a, mp3 }`
  - `public struct ExportOptions: Codable, Equatable, Sendable { var format: ExportFormat = .m4a; var bitrateKbps: Int = 128 /* 96 | 128 | 192 */ }`
  - `public enum Exporter { static func export(from source: MonoFileReader, options: ExportOptions, to destination: URL) throws }`：先写 `destination` 同目录的临时文件，用 AVAudioFile 完整解码校验（时长与源相差 ≤ 0.1 s），再 `replaceItemAt` 原子替换；失败时删除临时文件、不动已有目标文件。

- [ ] **Step 1:** 写测试：`testM4AExportDecodesWithExpectedDurationAndSize`（10 s 48 kHz 正弦 → 可解码、时长 10 ±0.1 s、单声道、文件大小在 128 kbps × 10 s 的 ±25% 内）；`testMP3ExportDecodesWithExpectedDuration`（同上，MP3）；`testFailedExportLeavesExistingFileIntact`（目标目录设为只读、已有旧文件 → 抛错，旧文件字节不变，目录里没有临时文件）。
- [ ] **Step 2:** Run → Expected: 失败。
- [ ] **Step 3:** `tools/build_lame.sh`；实现编码器。`tools/test.sh` 在测试前确保 `vendor/lame` 已构建；测试运行时用 `DYLD_LIBRARY_PATH` 或 rpath 找到 dylib（构建者定，写进 `tools/test.sh`）。
- [ ] **Step 4:** Run: `tools/test.sh --filter ExportTests` → Expected: 3 passed。
- [ ] **Step 5:** NOTICE 加 LAME（LGPL-2.0），`Resources/licenses/LGPL-2.0.txt`。Commit `feat: atomic M4A/MP3 export with LAME dylib`.

### Task 8: 会话存储、工作区与清理

**Files:**
- Create: `Sources/Sessions/SessionManifest.swift`、`SessionStore.swift`、`Workspace.swift`
- Test: `Tests/SessionsTests/SessionsTests.swift`

**Interfaces:**
- Consumes: `MasteringSettings`、`ExportOptions`、`MonoFileReader`/`MonoFileWriter`。
- Produces:
  - `public struct ProcessingSettings: Codable, Equatable, Sendable { var reuseVariant: String = "reuse-nemotron" }`（Plan 2/3 往里加字段，均带默认值以兼容旧 manifest）
  - `public struct SessionManifest: Codable, Equatable, Sendable { let id: UUID; var name: String; let createdAt: Date; var original: OriginalAudio; var processing: ProcessingSettings; var mastering: MasteringSettings; var export: ExportOptions; var lastExport: ExportRecord? }`；`OriginalAudio { fileName: String; sampleRate: Int; durationSeconds: Double; imported: Bool }`；`ExportRecord { fileName: String; exportedAt: Date; modelRevisions: [String: String]; backend: String; processingSeconds: Double }`
  - `public final class SessionStore { init(root: URL); func list() throws -> [SessionManifest]; func directory(for id: UUID) -> URL; func originalURL(for m: SessionManifest) -> URL; func save(_ m: SessionManifest) throws; func importFile(_ source: URL, name: String) throws -> SessionManifest /* 原样复制 */; func adoptRecording(_ flac: URL, name: String) throws -> SessionManifest; func delete(_ id: UUID) throws; func diskUsage(for id: UUID) -> Int64 }`。会话目录名 `yyyy-MM-dd HHmm <name>`，manifest 文件 `session.json`。
  - `public final class Workspace { init(cacheRoot: URL, sessionID: UUID); let directory: URL; func purge() throws; func diskUsage() -> Int64; static func purgeAll(cacheRoot: URL) throws }`
  - 导出成品文件名 `<name>.m4a|.mp3`；`SessionStore.recordExport(_:for:)` 删除另一格式的旧成品，保证会话目录里最多一个成品。

- [ ] **Step 1:** 写测试：`testImportCopiesOriginalByteForByte`；`testSessionDirectoryContainsOnlyOriginalManifestAndOneExport`（模拟先导出 m4a 再导出 mp3 → 目录里只有原始文件、`session.json`、一个 `.mp3`）；`testWorkspacePurgeRemovesDirectory`；`testPurgeAllRemovesEveryWorkspace`；`testManifestDecodesWhenNewFieldsMissing`（缺 `processing` 字段的旧 JSON 也能解码出默认值）。
- [ ] **Step 2:** Run → Expected: 失败。
- [ ] **Step 3:** 实现。
- [ ] **Step 4:** Run: `tools/test.sh --filter SessionsTests` → Expected: 5 passed。
- [ ] **Step 5:** Commit `feat: session store with single-export invariant and workspace purge`.

### Task 9: 处理流水线

**Files:**
- Create: `Sources/Pipeline/AudioStage.swift`、`ReuseStage.swift`、`ResampleStage.swift`、`ProcessingPipeline.swift`
- Test: `Tests/PipelineTests/PipelineTests.swift`

**Interfaces:**
- Consumes: `MambaEnhancer`、`InferenceSettings`、`MonoFileReader/Writer`、`StreamingResampler`、`Workspace`、`ModelStore`。
- Produces:
  - `public struct StageIO: Sendable { let url: URL; let sampleRate: Int; let cacheKey: String }`
  - `public protocol AudioStage: Sendable { var id: String { get }; var settingsKey: String { get }; func outputSampleRate(forInput sampleRate: Int) -> Int; func run(input: StageIO, output: URL, progress: @Sendable (Double) -> Void) async throws }`
  - `public struct ReuseStage: AudioStage`（`init(enhancer: MambaEnhancer, variantID: String, revision: String)`；在输入采样率上流式运行，块大小 = `session.recommendedInputFrameCount`）
  - `public struct ResampleStage: AudioStage`（任意采样率 → 48000；输入已是 48000 时流水线跳过它）
  - `public struct PipelineProgress: Sendable { let stageIndex: Int; let stageCount: Int; let stageID: String; let fraction: Double; let elapsed: Duration }`
  - `public actor ProcessingPipeline { init(stages: [any AudioStage], workspace: Workspace); func run(original: URL, progress: @Sendable @escaping (PipelineProgress) -> Void) async throws -> StageIO /* 48 kHz CAF */ }`
- 缓存规则：阶段输出写为 `<workspace>/<stageID>-<key>.caf.partial`，成功后重命名为 `.caf`；`key = SHA-256(上一阶段 cacheKey + stage.id + stage.settingsKey)` 的前 16 位十六进制；原始文件的 cacheKey = 文件大小 + 修改时间 + 前 1 MB 的 SHA-256。启动时只认 `.caf`，`.partial` 一律删除。

- [ ] **Step 1:** 用一个计数的 `FakeGainStage` 写：`testSecondRunHitsCache`（第二次 run 时 stage 调用次数不变）；`testSettingsChangeRecomputesDownstream`（改第 1 阶段 settingsKey → 第 1、2 阶段都重算）；`testCancelledStageIsRecomputed`（在 run 中取消 → 工作区内无 `.caf` 残留，只可能有 `.partial`；下次 run 重新计算且把 `.partial` 清掉）；`testHalfSecondInputProcesses`（0.5 s 输入经真实 `ReuseStage` → 输出 24000 帧，环境变量缺失则跳过）；`test16kImportIsEnhancedAtSourceRateThenResampled`（16 kHz 输入 → 阶段顺序为 reuse@16000、resample → 输出 48 kHz、时长一致）；`testTwentyMinuteInputKeepsPeakMemoryBounded`（20 分钟合成 48 kHz 文件经真实 `ReuseStage` → `MLX.Memory.peakMemory` < 3 GB，进程常驻内存增长 < 500 MB；标记为慢测试，仅在 `MICPOLISH_SLOW_TESTS=1` 时运行）。
- [ ] **Step 2:** Run → Expected: 失败。
- [ ] **Step 3:** 实现。
- [ ] **Step 4:** Run: `MICPOLISH_REUSE_SOURCE=… MICPOLISH_SLOW_TESTS=1 tools/test.sh --filter PipelineTests` → Expected: 6 passed。记录 20 分钟测试的耗时与 RTF 到 ledger。
- [ ] **Step 5:** Commit `feat: cached, cancellable processing pipeline with RE-USE stage`.

### Task 10: 命令行工具

**Files:**
- Create: `Sources/micpolish/MicPolishCLI.swift`
- Test: 无单元测试；验收为下列真实运行。

**Interfaces:**
- Consumes: Task 4–9 的全部公开接口。
- Produces: `micpolish models install [--variant reuse-nemotron|reuse-nemotron-laughter]`；`micpolish models list`；`micpolish process <input> --output <file.m4a|.mp3> [--bitrate 128] [--preset default|macbook] [--variant …]`：内部建临时会话根与工作区，结束后清理工作区，打印各阶段耗时、RTF、输入/输出响度与真峰值。

- [ ] **Step 1:** 实现。
- [ ] **Step 2:** Run: `swift run -c release micpolish models install` → Expected: 下载约 38.6 MB、校验通过、显示安装路径。
- [ ] **Step 3:** Run: `swift run -c release micpolish process Tests/Fixtures/reuse/input_48k.wav --output /tmp/mp-out.m4a --preset macbook` → Expected: 输出文件存在；报告输出响度 −16 ±0.5 LUFS、真峰值 ≤ −1.0 dBTP；工作区目录已删除。用 `ls ~/Library/Caches/MicPolish/work` 确认为空。
- [ ] **Step 4:** 同一命令输出 `.mp3` → Expected: 同上。
- [ ] **Step 5:** Commit `feat: micpolish CLI for model install and headless processing`.

### Task 11: 录音

**Files:**
- Create: `Sources/Recording/RecordingSink.swift`、`Recorder.swift`、`RecordingRecovery.swift`
- Test: `Tests/RecordingTests/RecordingTests.swift`

**Interfaces:**
- Consumes: `MonoFileWriter`、`StreamingResampler`、`SessionStore.adoptRecording`。
- Produces:
  - `public final class RecordingSink { init(url: URL /* .caf */, inputFormat: AVAudioFormat) throws; func append(_ buffer: AVAudioPCMBuffer) throws /* 下混 + 转 48 kHz */; var peakDBFS: Float { get }; func finish() throws -> URL }`
  - `@MainActor public final class Recorder: ObservableObject { init(store: SessionStore, cacheRoot: URL); @Published private(set) var isRecording: Bool; @Published private(set) var levelDBFS: Float; @Published private(set) var elapsed: TimeInterval; func availableInputs() -> [AudioInput]; func start(input: AudioInput?) throws; func stop(name: String) async throws -> SessionManifest }`：AVAudioEngine 输入节点，不开 voice processing；录到 `<cacheRoot>/recording/<UUID>.caf`；`stop` 转 FLAC 24-bit 交给 `SessionStore.adoptRecording`，然后删除 CAF。
  - `public enum RecordingRecovery { static func recover(cacheRoot: URL, store: SessionStore) throws -> [SessionManifest] }`：把遗留的 `recording/*.caf` 转为会话（名称"恢复的录音 <时间>"），再删 CAF。

- [ ] **Step 1:** 写测试：`testSink44k1StereoBecomes48kMono`（喂 2 s 的 44.1 kHz 双声道缓冲 → 结果文件 48 kHz 单声道、时长 2.0 ±0.01 s）；`testSinkPeakTracksInput`；`testRecoveryAdoptsLeftoverCAFAndDeletesIt`。
- [ ] **Step 2:** Run → Expected: 失败。
- [ ] **Step 3:** 实现。`Info.plist` 的 `NSMicrophoneUsageDescription` 在 Task 13 加。
- [ ] **Step 4:** Run: `tools/test.sh --filter RecordingTests` → Expected: 3 passed。
- [ ] **Step 5:** Commit `feat: crash-safe recorder with 48k mono sink and recovery`.

### Task 12: 试听播放器与实时母带

**Files:**
- Create: `Sources/Preview/PreviewPlayer.swift`、`MasteringSourceNode.swift`
- Test: `Tests/PreviewTests/PreviewTests.swift`

**Interfaces:**
- Consumes: `MasteringChain`、`LoudnessMeter`、`readAllMono`/`MonoFileReader`。
- Produces:
  - `public enum PreviewSource: String, CaseIterable, Sendable { case original, processed, mastered }`
  - `@MainActor public final class PreviewPlayer: ObservableObject { func load(original: URL, processed: URL?) throws; func setMastering(_ s: MasteringSettings, loudnessGainDB: Double); @Published var source: PreviewSource; @Published private(set) var position: TimeInterval; let duration: TimeInterval; func play(); func pause(); func seek(to: TimeInterval) }`
  - 三路共用同一播放位置；切换不中断播放。`mastered` 路由经 `AVAudioSourceNode` 实时跑 `MasteringChain`。电平匹配：三路各自按积分响度对齐到 −23 LUFS（只用于试听，不影响导出）。
- `MasteringSourceNode.render(into:frameCount:)` 必须调用与离线相同的 `MasteringChain.process`。

- [ ] **Step 1:** 写测试：`testManualRenderingMatchesOfflineChain`（AVAudioEngine 手动渲染模式下渲染 `mastered` 5 s，与对同一输入直接调用 `MasteringChain` + 相同 `loudnessGainDB` 的结果逐样本相等，最大差 ≤ 1e-6）；`testSwitchingSourceKeepsPosition`（手动渲染 2 s 后切 source → `position` 不变，后续样本来自新源）；`testLevelMatchingAlignsLoudness`（两路响度相差 10 dB 的源 → 渲染后积分响度差 ≤ 0.5 LU）。
- [ ] **Step 2:** Run → Expected: 失败。
- [ ] **Step 3:** 实现。
- [ ] **Step 4:** Run: `tools/test.sh --filter PreviewTests` → Expected: 3 passed。
- [ ] **Step 5:** Commit `feat: synchronized A/B/C preview with live mastering`.

### Task 13: SwiftUI 应用与打包

**Files:**
- Create: `Sources/MicPolishApp/`：`MicPolishApp.swift`、`AppModel.swift`、`SessionListView.swift`、`SessionDetailView.swift`、`RecordPanel.swift`、`ProcessPanel.swift`、`MasteringPanel.swift`、`ExportPanel.swift`、`SettingsView.swift`、`Localizable.xcstrings`
- Create: `Config/Info.plist`（`CFBundleIdentifier io.github.9s94ttyfct-bit.micpolish`、`NSMicrophoneUsageDescription`、`LSMinimumSystemVersion 14.0`）、`tools/package-app.sh`
- Test: 界面不写单元测试；验收为下列真实运行与截图。

**Interfaces:**
- Consumes: 前面所有模块。
- Produces: `dist/MicPolish.app`（Contents/MacOS 可执行、Resources/mlx.metallib、Frameworks/libmp3lame.0.dylib、ad-hoc 签名）。
- 行为（照设计 §5.7 的线框）：
  - 首次启动：若 RE-USE 未安装，显示许可说明 + "下载模型"按钮。
  - 录音面板：设备选择、电平表、计时、开始/停止；停止后输入名称。
  - 处理面板：变体选择、"处理"按钮、进度、阶段、预计剩余时间、取消。
  - 母带面板：六级各自的开关与滑块、预设菜单（默认 / MacBook 麦克风校正 / 用户预设）。
  - 试听：三段切换 + 播放条。
  - 导出面板：格式、码率、导出；导出成功后清理工作区并显示占用"原始 X MB · 成品 Y MB"。
  - 打开已导出的会话时提示需重新处理（显示预计耗时 = 时长 × 上次 RTF）。
  - 启动时执行 `RecordingRecovery.recover` 和 `Workspace.purgeAll`；退出时 `Workspace.purgeAll`。
  - 设置：会话根目录、模型列表与占用、删除模型。

- [ ] **Step 1:** 实现视图与 `AppModel`；`tools/package-app.sh` 以 release 构建、组装 bundle、`install_name_tool` 设 rpath、`codesign --force --deep -s -`。
- [ ] **Step 2:** Run: `tools/package-app.sh && open dist/MicPolish.app` → Expected: 窗口出现。用 `computer` 截图确认界面布局与中文文案。
- [ ] **Step 3:** 用 `Tests/Fixtures/reuse/input_48k.wav` 走一遍"导入 → 处理 → 切换三路试听 → 调 EQ → 导出 M4A"→ Expected: 成品存在，会话目录只有原始文件、`session.json` 和一个 `.m4a`；`~/Library/Caches/MicPolish/work` 为空。截图留证。
- [ ] **Step 4:** 录音验证需要麦克风授权，交给用户：请用户录 10 秒并导出，确认流程可用（这一步之前不宣称录音功能已验证）。
- [ ] **Step 5:** Commit `feat: MicPolish SwiftUI app and app bundle packaging`.

### Task 14: 文档与性能记录

**Files:**
- Modify: `README.md`（中文为主附英文：功能、安装、Gatekeeper 放行、模型与许可、存储策略、实测数据）、`NOTICE`
- Create: `docs/performance.md`

- [ ] **Step 1:** Run: `swift run -c release micpolish process <60 s 以上的 48 kHz 语音> --output /tmp/perf.m4a` 三次 → 记录 RE-USE RTF 中位数（M4 Pro）。用一段真实录音转 FLAC，记录每分钟 MB 数，替换设计文档里"4–6 MB/分钟"的估算。
- [ ] **Step 2:** 写 README 与 `docs/performance.md`；README 的说法须如实（"更干净、更清晰的人声"，不写"录音棚级别"）。
- [ ] **Step 3:** Run: `tools/test.sh` → Expected: 全部测试通过（需模型的测试在设置了 `MICPOLISH_REUSE_SOURCE` 时运行）。
- [ ] **Step 4:** Commit `docs: README, notices and measured performance`.
