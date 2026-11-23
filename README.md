# MicPolish

在 Mac 上本地录音，并用 AI 把人声修得更干净、更清晰：录音或导入 → NVIDIA RE-USE 语音修复（MLX，Apple GPU）→ 可实时试听的母带处理 → 导出 M4A 或 MP3。全部在本机运行，音频不会上传。

On-device voice recording and enhancement for Apple Silicon Macs: record or import → NVIDIA
RE-USE speech restoration (MLX on the GPU) → adjustable mastering with live preview → M4A or
MP3. Nothing leaves your Mac.

> 它能明显去掉噪声和房间混响、让音色更清楚、让响度达标；但录音时没收进来的近讲感和低频厚度，任何保真处理都补不回来。它不是“录音棚级别”的魔法。

## 功能

- **录音**：Core Audio 采集设备原始信号（不经过系统语音处理），按设备采样率接收后转换为 48 kHz 保存；崩溃或断电后下次启动会自动恢复已录部分。
- **导入**：工具栏按钮、拖放到侧栏，或在访达里用 MicPolish 打开音频文件。原文件原样保存。
- **AI 修复**：NVIDIA NemotronLabs RE-USE（去噪、去混响、修复低质麦克风与编码损伤），在源采样率上运行。可选“保留笑声”变体。
- **母带处理**：高通、四段 EQ（含“MacBook 麦克风校正”预设）、去齿音、压缩、响度标准化（−16 / −14 / −23 LUFS，ITU-R BS.1770，单声道按双单声道计量）、真峰值限幅（默认 −1 dBTP）。参数可实时试听，导出与试听一致。
- **三路对比**：原始 / AI 处理后 / 成品在同一位置切换，自动按响度对齐，比较的是音质而不是音量。
- **体积控制**：每个会话只保留原始音频（自录为 FLAC 无损）、`session.json` 和一个成品文件。中间音频在导出后、退出时自动删除。
- **强降噪（可选）**：Inworld Voice Isolator v1（Apache-2.0，离线模型）在 AI 修复前把人声从音乐、嘈杂人声、风扇等强背景噪声里分离出来。
- **录音棚音色（可选）**：母带里的自动 EQ，测量人声的长期频谱，按八度频段向近距离录音棚话筒的人声靠拢（参考 VCTK 录音棚语料，按音高自动选男声/女声曲线），每段最多 ±6 dB。
- **录音**：录音时主界面显示实时滚动波形和电平；⌘R 开始 / 停止并保存，⌘⌫ 丢弃（确认后删除，不保存）；可在设置或命名框里关闭“停止后询问名称”，直接用默认名称。
- **快速裁剪**：在波形上拖动首尾手柄，或一键去掉首尾静音；原始录音不改，调整裁剪不用重跑 AI。
- **高频增强（可选）**：LavaSR v2（Apache-2.0）在 AI 修复后重新生成起始频率以上的高频，起始频率可选自动 / 8 / 12 / 16 kHz。
- **批量与快速导出**：侧栏多选后可批量删除或“一键处理并导出”；可指定导出文件夹，工具栏“快速导出”一步导出当前录音。
- **双语界面**：中文 / English / 跟随系统，工具栏地球图标即时切换。
- **命令行**：`micpolish process input.wav --output out.m4a --preset macbook [--strong-denoise] [--high-band auto]`；`micpolish abx input.wav --out dir/ [--compare denoise]` 生成盲听用的乱序对比文件。

## 系统要求

Apple Silicon Mac，macOS 14 或更新。

## 安装

1. 从 [Releases](https://github.com/9s94ttyfct-bit/micpolish/releases) 下载 `MicPolish-<版本>-macOS-arm64.zip`，解压后把 `MicPolish.app` 拖到“应用程序”。也可以自己构建：`tools/package-app.sh`（需要 Xcode 与 Metal Toolchain：`xcodebuild -downloadComponent MetalToolchain`），产物在 `dist/MicPolish.app`。
2. 应用是 ad-hoc 签名（没有 Apple 开发者证书，也没有公证），第一次打开时 macOS 会拦截：在“系统设置 › 隐私与安全性”里点“仍要打开”；或者在终端运行 `xattr -dr com.apple.quarantine /Applications/MicPolish.app` 后再打开。
3. 首次使用会下载 RE-USE 模型（约 37 MB），校验 SHA-256 后在本机转换为 MLX 格式，存放在 `~/Library/Application Support/MicPolish/models/`。

## 存储位置

| 内容 | 位置 |
| --- | --- |
| 会话（原始音频、成品、`session.json`） | `~/Music/MicPolish/`（设置里可改） |
| AI 模型 | `~/Library/Application Support/MicPolish/models/` |
| 编辑期间的中间文件 | `~/Library/Caches/MicPolish/work/`，导出后和退出时清空 |

导出过的会话再调整时需要重新跑一次 AI（中间文件已经删了）；按 M4 Pro 实测，一分钟 48 kHz 音频约 32 秒。

## 性能

M4 Pro 上 RE-USE 的实时率（RTF，处理时间 ÷ 音频时长）约 0.48（MLX GPU），48 kHz 时 MLX 内存峰值 3.2 GB（预算 5 GB），与录音时长无关。装了 Core ML 包时可在设置里选“ANE 混合”（编码器和解码器跑神经引擎，Mamba 层仍在 GPU）：64 s 语音 RTF 0.41（快约 15%），与 MLX 结果的相对误差 0.1%；“自动”按本机测速结果决定。完整数据见 [docs/performance.md](docs/performance.md)。

## 开发

```sh
tools/test.sh                                   # 单元测试（自动构建 mlx.metallib 和 LAME）
MICPOLISH_REUSE_SOURCE=/path/to/model.safetensors tools/test.sh   # 加上需要模型的测试与数值对拍
MICPOLISH_SLOW_TESTS=1 …                          # 加上 20 分钟长音频内存测试
swift build -c release --product micpolish        # 命令行工具
tools/package-app.sh                              # 打包 dist/MicPolish.app
```

`tools/parity/reuse_reference.py`（`uv run`）用 NemotronLabs 上游模型代码和纯 PyTorch 的 Mamba 生成对拍参考输出。

## 许可

- 代码：Apache-2.0（见 `LICENSE`）。
- RE-USE 推理代码来自 [SpeechLens](https://github.com/faraday/SpeechLens)（Apache-2.0）。
- 模型权重 [nvidia/NVIDIA-NemotronLabs-RE-USE](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE)：OpenMDW-1.1，允许商用；运行时下载，不随应用分发。
- 模型权重 [inworld/voice-isolator](https://huggingface.co/inworld/voice-isolator)：Apache-2.0；运行时下载（约 256 MB）：第一次在开启强降噪的情况下处理时才下载，也可以在设置里提前下载。
- 模型权重 [LavaSR v2](https://github.com/ysharma3501/LavaSR)：Apache-2.0；从 safetensors 镜像运行时下载。
- MP3 编码：LAME 3.100，LGPL-2.0，以可替换的动态库形式随应用分发（`licenses/LAME-LGPL-2.0.txt`）。

详见 `NOTICE`。
