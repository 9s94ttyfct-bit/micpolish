# MicPolish — 设计文档

日期：2026-11-22　状态：待审阅
名称：**MicPolish**（Bundle ID `io.github.9s94ttyfct-bit.micpolish`）。GitHub 与 Mac App Store 上没有同名项目或应用；`ClearTake` 已被两个类似的 macOS 音频工具使用，所以没选。

## 1. 目的与背景

用 MacBook 内置麦克风录人声，在一个原生 macOS 应用里完成"录音 → AI 修复 → 高频增强 → 可调母带处理 → 导出"，让最终成品尽可能好听。全部本地运行，音频不出本机。

用户已定下的要求：
- 主链路：**NVIDIA NemotronLabs RE-USE（48 kHz）→ 可调母带处理链**；**Inworld Voice Isolator 作为可选的强降噪前置**。
- 评判标准只有一个：**最终音质足够好**。允许模型生成录音里没有的内容（如高频细节）。
- 考虑加入一个专门的采样率提升 / 增强模型（结论见 §4.4：加入 LavaSR v2）。
- **所有模型尽量 MLX 原生**；**能用 M4 Pro 神经网络引擎（ANE）加速的地方尽量用**。
- **控制音频体积**：处理完后自动清理中间音频，只保留原始音频和 M4A 或 MP3 格式的成品。
- 自用为主，可能在 GitHub 公开发布（非必须）。

MacBook 麦克风录音的主要问题：房间混响（口距远）、风扇/键盘/环境底噪、麦克风音色偏薄偏硬、电平与动态不稳。内置麦克风本身录 48 kHz，所以带宽扩展对它的价值不在"补齐缺失频段"，而在于让顶端高频更干净、更有空气感（降噪和去混响后高频常会变暗或带残余噪声）；对低采样率的导入文件则是真正的补频段。是否默认启用由盲听决定（§7）。

## 2. 目标、非目标、反目标

**目标**
- 一键录音，原始录音永远保留、从不覆盖。
- 所有模型都在 Swift + MLX 中原生运行，**运行时不需要 Python**。
- 在实测更快且结果达标的前提下，把适合的计算放到 ANE 上（§5.6）。
- 母带处理每个参数可调，调节时能实时试听，导出结果与试听一致。
- 原始 / AI 处理后 / 最终成品三者可在同一位置瞬时切换对比。
- 也能导入已有音频文件走同一条链路。
- 每个会话最终只占用"原始音频 + 一个成品文件"的空间（§5.9）。
- 能公开发布：代码与所用权重的许可都允许再分发。

**非目标（v1 不做）**
- AI 处理的实时监听（所有模型都用离线方式跑）。
- Inworld 的实时（streaming）模型。
- 多轨编辑、剪辑、视频、音乐处理、立体声输出（人声统一单声道）。
- 整段重新合成的生成式修复模型（Sidon、Resemble Enhance、Diamond 等，理由见 §4.4）。
- WAV 等无损格式的成品导出（成品只出 M4A 或 MP3）。
- Intel Mac、Windows。

**反目标（即使"能用"也算失败）**
- 成品听起来不像本人（音色漂移是音质问题，盲听时专门检查）。
- 音频上传到任何服务器。
- 修改或丢失原始录音。
- 中间音频残留在磁盘上，或会话占用随编辑次数增长。
- 为了上 ANE 而让结果变差或变慢：ANE 路径必须同时通过速度门槛和结果门槛才启用。

## 3. 约束

- 硬件：Apple Silicon（开发机 M4 Pro / 24 GB / macOS 27.3，Xcode 27.2，Swift 6.4.2）。最低系统 macOS 14（沿用 SpeechLens 的依赖下限）。
- 运行时：Swift + MLX（mlx-swift）+ Core ML（仅 ANE 路径）+ Accelerate + AVFoundation；MP3 编码用 LAME（动态链接）。Python 只用于开发期：生成对拍参考数据、导出 Core ML 包。
- 许可：
  - RE-USE 只用 `nvidia/NVIDIA-NemotronLabs-RE-USE`（及可选的 `-laughter` 变体），OpenMDW-1.1，允许商用与再分发。**不用** `nvidia/RE-USE` / `faraday/re-use-mlx`（NSCLv1 非商用）。
  - 复用 SpeechLens（Apache-2.0）的推理代码，保留其 LICENSE/NOTICE。**不得**复制 SpeechLens 的 `tools/reference_runner.py`（文件头标注 NSCLv1）；对拍脚本自己基于 NemotronLabs 仓库（OpenMDW-1.1）的模型代码写。
  - Inworld Voice Isolator 代码与权重：Apache-2.0。LavaSR v2 代码与权重：Apache-2.0。移植代码保留各自版权与 NOTICE。
  - LAME：LGPL-2.0，以动态库形式随应用分发并附许可（满足 LGPL 的可替换要求）。macOS 自带编码器能写 AAC 和 FLAC，但**不能写 MP3**（已用 `afconvert` 实测：FLAC、ALAC、AAC 编码成功，MP3 报 `'fmt?'` 错误）。
  - 不打包 FFmpeg。
- 模型文件：
  - MLX 权重不随应用分发：首次使用时从 Hugging Face 下载固定 revision，校验 SHA-256，在本地用 Swift 转换为 MLX 布局。
  - Core ML 包（ANE 路径）无法在运行时由 Swift 生成，需开发期用 coremltools 导出。自用时由导出脚本直接安装到模型目录；公开发布时作为项目 GitHub Release 的附件（附对应许可文本），应用按固定 SHA-256 下载校验。两种来源都没有时，ANE 选项不可用，自动用 MLX。
- 应用不沙盒化（读写用户选定目录、下载模型）；需要麦克风权限说明。

## 4. 方案取舍

### 4.1 RE-USE 推理运行时

| 方案 | 取舍 | 结论 |
|---|---|---|
| **A. 复用 SpeechLens 的 `Inference` 模块（Swift + MLX + 自定义 Metal 选择性扫描核），本地转换 NemotronLabs 权重** | 已有原生采样率实现、5 s 分块 + 5% 重叠的 overlap-add、NVIDIA 推理脚本里的"扫频伪影"抑制；M1 Max 上 48 kHz RTF ≈ 0.375。需要自己验证新权重的数值一致性 | **采用，作为基线（纯 MLX/GPU）** |
| **B. 在 A 之上加 ANE 混合后端：卷积编码器/解码器走 Core ML（`cpuAndNeuralEngine`），Mamba 走 MLX** | 思路来自 AILogDev ReuseKit（Apache-2.0 代码）：M2 上 16 kHz 从 MLX 的 0.209 降到 0.159（同配置 `warpFast`）。代价：FP16 数值差异、Core ML 需固定形状、MLX↔Core ML 之间的数据搬运、开发期导出流程 | **采用，作为可选加速后端，过门槛才默认启用**（§5.6） |
| C. Python 旁路进程跑 PyTorch MPS | 违背"MLX 原生、运行时无 Python"；上游依赖 `mamba-ssm`（仅 CUDA） | 否 |
| D. 直接用 ReuseKit 现成包 | 只支持 16 kHz，丢掉 8 kHz 以上的真实内容 | 否 |

方案 A 的可行性已核实：NemotronLabs 与旧版 `nvidia/RE-USE` 的 `model.safetensors` 有**完全相同的 1416 个张量键、形状和 dtype（F32）**；两版的模型代码（`generator_SEMamba_time_d4.py`、`mamba_block2_SEMamba.py`、`stfts.py`）、`config.json` 和 recipe 除许可文件头外逐字相同。因此 SpeechLens 的 `WeightPort` 键映射和布局转换可直接用于新权重。**仍需对拍验证**（§7）。

### 4.2 Inworld Voice Isolator 运行时

| 方案 | 取舍 | 结论 |
|---|---|---|
| **A. MLX 原生移植（Swift）** | 模型代码约 800 行，用到的算子全部是 MLX 现成的：Linear、LayerNorm、缩放点积注意力（MLX 有融合实现）、Conv1d、Mamba（可复用 RE-USE 的 Metal 选择性扫描核思路）、STFT（n_fft 2048、hop 480，用 MLX FFT 实现）。上游仓库自带 golden 对拍夹具（`tests/fixtures`），可直接拿来验收。离线权重 256 MB（fp32 safetensors），本地转换 | **采用** |
| B. Python 旁路进程（`inworld-voice-isolation`，PyTorch MPS） | 现成可用，但违背"MLX 原生"，还要装几 GB 的 PyTorch | 否（可作为开发期对拍参考） |

Inworld 的注意力和前馈层是 ANE 的潜在候选，但其块内注意力与 Mamba 交替出现，拆分到 Core ML 会带来频繁搬运；v1 只做 MLX。

### 4.3 母带处理链

| 方案 | 取舍 | 结论 |
|---|---|---|
| **A. 自写 Swift DSP（Accelerate/vDSP 双二阶滤波器、压缩器、去齿音、真峰值限幅、ITU-R BS.1770 响度）** | 同一份代码既做实时试听又做离线导出，结果逐样本一致；可单元测试；没有 Apple 自带去齿音 AU 的缺口 | **采用** |
| B. AVAudioEngine + Apple AU（AUNBandEQ、AUDynamicsProcessor、AUPeakLimiter） | 没有去齿音；参数语义不透明；响度测量和真峰值仍得自己写 | 否 |
| C. 打包 FFmpeg 滤镜 | GPL/LGPL 打包；子进程模式难做实时试听 | 否 |

母带处理不上 ANE：这类 DSP 是逐样本的低延迟串行计算，计算量很小，ANE 的调度开销和 FP16 精度都不合适；Accelerate 在 CPU 上（含 Apple 矩阵协处理器）是正确的加速器。

### 4.4 高频增强 / 采样率提升模型

| 候选 | 事实 | 结论 |
|---|---|---|
| **A. LavaSR v2（Vocos 架构，约 56 MB，Apache-2.0）** | 主干是 8 层 ConvNeXt（宽 512）+ iSTFT 输出头，输入为 48 kHz 波形的梅尔谱（80 维，只看 0–8 kHz），生成全频带，再用自带的分频合并（Linkwitz 式交叉，平滑过渡）保留原信号低于截止频率的部分、只取模型生成的高频。论文报告 8/16/24→48 kHz 的对数谱距离优于 AP-BWE 和 AudioSR；GPU 上约 5000× 实时，CPU 上 50× 实时以上。全是卷积和线性层，MLX 移植简单，固定窗长时也是 ANE 的好候选 | **采用**，作为可选阶段，默认开关由盲听定 |
| B. FlashSR（一步蒸馏扩散 + VAE + BigVGAN 声码器） | speech-swift 已有 MLX 移植（Apache-2.0 代码），但现成 MLX 权重是 CC-BY-NC-4.0；INT4 包 363 MB，运行时反量化为浮点，远比 LavaSR 重；它把整段波形重新合成，低频也会变，需要再加分频合并 | 否（v1）；若盲听显示 LavaSR 不够，再作为后续候选 |
| C. RE-USE 自带的带宽扩展用法（先上采样再送 RE-USE） | 不用新增模型，但只对低采样率的导入文件有意义 | 只作为低采样率导入文件的盲听对照（§7） |
| D. 整段重新合成的生成式修复（Sidon、Resemble Enhance、Diamond） | Sidon 的自然度指标最高，但说话人相似度 0.74（RE-USE 为 0.86），主干是约 600M 参数的 w2v-BERT 2.0；Resemble Enhance 在 NVIDIA 的对比里说话人相似度和 ASR 字准率明显下降；Diamond 只支持英文且是逐帧自回归，偶尔糊词 | 否：音色漂移本身就是音质问题，且模型重 |

## 5. 设计

### 5.1 处理流程

```
录音/导入 ──► 原始音频（永久保留，见 §5.9）
                │
                ├─[可选] Inworld Voice Isolator（offline 模型，MLX）
                │
                ▼
          RE-USE（NemotronLabs 或 -laughter，源采样率，MLX 或 ANE 混合）
                │
                ├─[可选] LavaSR v2 高频增强（MLX 或 ANE）──► 48 kHz
                │
                ▼
          母带处理链（Accelerate，实时可调）──► 试听 / 导出 M4A 或 MP3
```

- 编辑期间，每个 AI 阶段的输出缓存在临时工作区（§5.9）；只有上游输入或该阶段设置变化时才重算。母带参数变化不触发 AI 重算。
- Inworld 放在 RE-USE 之前（先去掉强干扰，再做修复）；LavaSR 放在 RE-USE 之后（在干净信号上补高频，避免把残余噪声当成内容放大）。这两个顺序是推断，在 §7 的盲听中确认。
- 三个模型串行执行，共用一个 MLX 设备队列，避免统一内存峰值叠加。

### 5.2 录音

- AVAudioEngine 输入节点；**不启用** voice processing，避免系统预处理。提示用户在控制中心把麦克风模式设为"标准"（"语音突显"会在系统层处理）。
- 录制时以 48 kHz float32 边写边落盘到临时 CAF 文件，崩溃或断电时已录部分可在下次启动时恢复；停止录音后转为 FLAC（24-bit），删除临时 CAF。内置麦克风若呈现多声道，下混为单声道。
- 实时电平表与削波警示；建议峰值在 −12 dBFS 左右。
- 输入设备可选（默认内置麦克风；外接麦克风同样可用）。

### 5.3 导入

- AVFoundation 能读的格式均可导入。原文件**按原样复制**进会话作为原始音频（不转码：不会因转码变大，也保留真正的原件）。
- 处理时解码为单声道 float32。RE-USE 在源采样率上运行（支持 8–48 kHz）。
- 源采样率低于 48 kHz 时，LavaSR 负责升到 48 kHz，交叉频率取源文件的奈奎斯特频率；LavaSR 关闭时用普通重采样升到 48 kHz。
- 非 48 kHz 的会话中，RE-USE 只走 MLX 后端（Core ML 形状按 48 kHz 固定）。

### 5.4 模型模块

- **RE-USE**：从 SpeechLens 只引入 `Inference` 模块（及其 Metal 核与 metallib 构建脚本）和 `WeightPort` 的转换逻辑；不引入其 App、Sentry 遥测、FFmpeg 媒体层。可在设置里切换 `-laughter` 变体。
- **Inworld**：新写的 Swift/MLX 模块，结构对应上游 `model.py`（band-split 46 段、7 个双路块、逐段输出头、复数掩码 + 残差）和 `isolator.py` 的分块拼接逻辑（按 checkpoint 自带配置的块长与重叠）。
- **LavaSR v2**：新写的 Swift/MLX 模块，对应上游 `LavaBWE.infer`：梅尔特征（采样率参数 44100、n_fft 2048、hop 512、80 维、0–8 kHz，按上游 `config.yaml`）→ Vocos ConvNeXt 主干 → iSTFT 头（含上游对幅度取 exp 后 clip 到 1e3 的改写）→ 分频合并。
  - 交叉频率：48 kHz 录音默认按 RE-USE 输出的实测有效带宽自动选取，设置里可手动设为 8 / 12 / 16 kHz（越低，模型生成的部分越多）。
  - 分块处理，块间重叠淡入淡出。上游的 `batch` 模式按 1 s 切块，这里改为更长的块以减少边界痕迹。
- **模型管理**（三者共用）：首次使用时下载固定 revision 的权重，校验 SHA-256，转换为 MLX 布局后存入 `~/Library/Application Support/MicPolish/models/`；同时保存许可文本，界面上可查看。体积：RE-USE 约 38.6 MB、LavaSR v2 约 56 MB、Inworld 约 256 MB。Inworld 在首次打开强降噪开关时才下载。
- **长文件**：流式分块处理，显示进度与预计剩余时间，可取消；内存占用不随录音时长增长。

### 5.5 母带处理链

固定顺序，每级可单独旁通：

1. 高通（默认 80 Hz，12 dB/oct）
2. 参数 EQ：低架、2 个峰值段、高架。附带"MacBook 麦克风校正"预设，用户可调可关。初始值以 §7 的实录频谱和听感来定，**不**假设 RE-USE 已校正麦克风音色。
3. 去齿音（4–9 kHz 可调的分频段压缩）。LavaSR 生成的高频可能偏亮，这一级要能压住。
4. 压缩器（默认阈值 −20 dB、比率 3:1、起音 10 ms、释放 150 ms、自动增益补偿）
5. 响度标准化：按 BS.1770 测整段积分响度，算出增益（离线导出时为两遍处理；实时试听用当前已测得的增益）
6. 真峰值限幅器（默认上限 −1.0 dBTP，4 倍过采样检测）

起始参数参考此前讨论的 FFmpeg 链（`highpass=80, eq 250 Hz −2 dB, eq 4 kHz +1.5 dB, deesser, acompressor −20 dB 3:1, loudnorm I=−16 TP=−1.5`），实现时按听感调整。

响度目标预设：−16 LUFS（播客 / Apple Podcasts，默认）、−14 LUFS（流媒体视频）、−23 LUFS（EBU R128 广播），以及自定义。

同一个 DSP 处理器同时供实时试听（AVAudioSourceNode）和离线导出使用；结果须逐样本一致（响度增益除外，见第 5 步）。用户可保存或载入自己的预设。

### 5.6 ANE 加速策略

原则：MLX/GPU 是每个模型的基线和回退；ANE 路径是可替换的加速后端，**只在实测更快且结果达标时默认启用**，任何加载或运行失败都自动回退 MLX。

- **候选 1：RE-USE 混合后端**。把 dense 编码器（前端卷积）和掩码/相位解码器（后端卷积）导出为 Core ML 包，计算单元设为 `cpuAndNeuralEngine`；30 层 Mamba 留在 MLX。块长固定（5 s @ 48 kHz；内存预算 5 GB，实测 5 s 窗口峰值 3.2 GB），因此 Core ML 形状固定；末尾不足一块的部分补零或回退 MLX；非 48 kHz 会话不用此后端。
- **候选 2：LavaSR 的 ConvNeXt 主干**。整个主干是卷积 + 线性层，按固定块长导出为一个 Core ML 包；梅尔特征和 iSTFT 头留在 MLX/Accelerate。它比候选 1 更像 ANE 擅长的负载（没有递推），但模型本身已经很快，绝对收益可能很小。
- **Core ML 包的产出**：开发期脚本用 coremltools 从上游 PyTorch 模型代码导出 FP16 包（来源见 §3）；首次加载时由 Core ML 在本机编译并缓存。
- **结果门槛**（开发期在 M4 Pro 上测一次，导出的包不变就不必重测）：48 kHz、60 s 以上真实语音，与纯 MLX FP32 输出的波形相对 RMSE ≤ 1%，且盲听无可闻差异。不达标就不发布该包，对应的 ANE 选项不出现。
- **速度门槛（每台 Mac 各测一次）**：首次可用时在后台用一段 30 s 的内置测试信号分别跑两种后端（预热后取中位数）；ANE 后端端到端 RTF 至少低 15% 才设为该机默认，否则默认 MLX。结果按芯片型号与系统版本保存，系统或包更新后重测；设置里可手动切换和重新测速。README 记录开发机的实测数据。
- **不上 ANE 的部分**：Mamba 选择性扫描（长序列递推，ANE 不擅长）、Inworld（见 §4.2）、母带 DSP（见 §4.3）、STFT/iSTFT 与梅尔特征（留在 MLX/GPU，避免额外搬运）。
- `cpuAndNeuralEngine` 只是请求，实际放到 ANE、GPU 还是 CPU 由 Core ML 决定。设置里显示每次处理实际使用的后端，以及 Core ML 报告的计算单元，用来确认是否真的跑在 ANE 上；速度门槛按端到端实测判断，不按"请求了 ANE"判断。

### 5.7 界面

单窗口 SwiftUI。左侧会话列表，右侧当前会话：

```
┌ 会话列表 ┐┌─────────────────────────────────────────────┐
│ 11-22 访谈 ││ [● 录音]  电平 ▮▮▮▮▮▯▯   00:03:12            │
│ 11-21 口播 ││ 波形 ───────────────────────────────────────│
│ …          ││ 试听：(原始)(AI 处理后)(成品)  ▶ ──●──────── │
│            ││ AI：[□ 强降噪] RE-USE [▾]  [☑ 高频增强 自动▾] │
│            ││     [处理]  后端：ANE+GPU · 进度 ▮▮▮▯ 剩余 1:20│
│            ││ 母带：高通 | EQ | 去齿音 | 压缩 | 响度 | 限幅 │
│            ││   各级旋钮/滑块，旁通开关，预设 ▾            │
│            ││ 导出：(M4A)(MP3)  码率 [128 kbps▾]  [导出]   │
│            ││ 占用：原始 14 MB · 成品 3 MB                 │
└────────────┘└─────────────────────────────────────────────┘
```

- 三路试听在同一播放位置瞬时切换，切换时做电平匹配（按积分响度对齐），避免"更响就显得更好"的错觉。
- 打开一个已导出的旧会话重新调整时，提示需要重新跑 AI（显示预计耗时），因为中间音频已被清理。
- 界面语言：中文为主，附英文本地化（为 GitHub 发布）。

### 5.8 导出

- 格式二选一：**M4A（AAC-LC，AVFoundation 原生）** 或 **MP3（LAME）**。单声道 48 kHz（MP3 同为 48 kHz）。
- 码率预设：96 / 128 / 192 kbps，默认 128 kbps（单声道语音约 0.96 MB/分钟）。
- 导出先写到临时文件，校验能完整解码后再原子替换到会话目录，避免半截文件。

### 5.9 存储与自动清理

- **会话目录**（默认 `~/Music/MicPolish/<日期-名称>/`，可改）永久保存三样东西：
  - 原始音频：自录为 `original.flac`（48 kHz / 24-bit 单声道，FLAC 无损；实测带噪语音约每分钟 4.9 MB，见 `docs/performance.md`）；导入的则是原文件的原样副本。
  - 成品：`<名称>.m4a` 或 `<名称>.mp3`。重新导出时覆盖旧成品，不累积多个版本。
  - `session.json`：各级参数、模型 revision、实际后端、处理时间，用来在需要时从原始音频完整复现成品。体积可忽略。
- **临时工作区**：`~/Library/Caches/MicPolish/work/<会话 ID>/`，存放各 AI 阶段的中间结果（float32 CAF），只在编辑期间存在。
  - 导出成功后立即删除该会话的工作区。
  - 退出应用时删除所有工作区；启动时清理上次异常退出遗留的工作区（录音中断的临时 CAF 先恢复为原始音频再删）。
  - 编辑期间工作区按 48 kHz float32 计约每分钟 11.5 MB/阶段；界面显示当前临时占用。
- 模型文件存放在 Application Support，不属于"音频体积"，在设置里显示占用并可删除。

### 5.10 发布

- 开源许可：Apache-2.0（与复用的 SpeechLens、Inworld、LavaSR 代码一致），附 NOTICE 列明 SpeechLens、Inworld Voice Isolator（权重运行时下载）、LavaSR v2（权重运行时下载）、NVIDIA RE-USE（OpenMDW-1.1，权重运行时下载；Core ML 派生包随 Release 发布并附许可）、LAME（LGPL-2.0，动态库）。
- 无 Apple Developer ID 时用 ad-hoc 签名 DMG，README 说明 Gatekeeper 放行方式。

## 6. 构建者自行决定的部分

- 项目结构（Swift Package + Xcode app target 的组织方式）、模块划分与命名。
- 分块长度与重叠比例在 SpeechLens / Inworld 默认值附近的微调；LavaSR 的块长。
- LavaSR 自动交叉频率的具体判定方法（有效带宽的检测阈值）。
- RE-USE 在 Core ML 与 MLX 之间的确切切分点（以 ReuseKit 的前端/后端划分为起点）、FP16 边界张量的处理方式。
- Inworld 的 Mamba 扫描是复用 SpeechLens 的 Metal 核还是另写一个（两者 state 维度都是 16）。
- DSP 的具体滤波器设计、压缩器检波方式、限幅器前视长度。
- LAME 的引入方式（源码编译为动态库 / 预编译二进制），只要满足 LGPL。
- 进度/ETA 的估算方法、错误提示文案。

## 6a. 可改的产品默认值

以下是我替用户先定的默认值，不是用户提出的需求，审阅时可直接改：母带处理自写 Swift DSP（§4.3）；成品默认 M4A AAC 128 kbps 单声道；原始录音存为 FLAC 24-bit；响度 −16 LUFS、真峰值 −1.0 dBTP；最低 macOS 14；AI 处理不做实时监听；界面中文为主附英文；开源许可 Apache-2.0。

## 7. 验证

1. **RE-USE 对拍**：用 NemotronLabs 仓库的模型代码 + 纯 PyTorch 选择性扫描在 CPU 上生成参考输出，与 MLX 实现在 48 kHz 和 16 kHz 输入上比对。通过线：波形相对 RMSE ≤ 1e-4（SpeechLens 在旧权重上的实测为 1.1e-5）。
2. **Inworld 对拍**：用上游 `tests/fixtures` 的 golden 夹具和其中声明的容差验收 MLX 移植；另用 `inworld-voice-isolation`（PyTorch CPU）在 60 s 以上的长录音上比对分块拼接结果。
3. **LavaSR 对拍**：用上游 `LavaSR` 包（PyTorch CPU）的 `LavaBWE.infer` 在同一输入、同一交叉频率下生成参考，与 MLX 实现比对，通过线：波形相对 RMSE ≤ 1e-3。
4. **ANE 门槛**：按 §5.6 的速度与结果门槛测试，结果写进 README。
5. **DSP 单元测试**：响度表用 EBU Tech 3341 测试信号校验（±0.1 LU）；限幅器输出真峰值不超过设定上限；全部旁通时输出与输入逐样本相同；实时路径与离线路径逐样本一致。
6. **存储测试**：导出成功后工作区目录不存在；会话目录里只剩原始音频、一个成品和 `session.json`；重复导出不增加文件数；模拟录音中途崩溃后重启，原始音频被恢复且临时文件被清理。
7. **速度**：在开发机上测各模型在 48 kHz 下的 RTF（各后端），记录在 README。
8. **盲听**（选型与调参的前提，不是可选项）：用户用 MacBook 录 3 段——安静房间、有风扇/键盘声、有背景音乐——另加 1 段 16 kHz 的导入文件。电平匹配后盲听比较：
   - "仅 RE-USE" vs "RE-USE→LavaSR（自动 / 12 kHz / 8 kHz 交叉）"，决定高频增强是否默认开启及默认交叉频率；
   - "Inworld→RE-USE" vs "RE-USE→Inworld"，决定 Inworld 的位置；
   - 16 kHz 文件：LavaSR vs RE-USE 自带的带宽扩展用法；
   - 每组都检查"听起来还像不像本人"。
   结果同时用来定"MacBook 麦克风校正"EQ 的默认值。

## 8. 已知风险

- RE-USE 训练片段都短于 10 秒，长录音依赖分块拼接；块边界可能有可闻痕迹，需在盲听中专门检查。LavaSR 的分块同理。
- 强噪声（音乐、人声嘈杂）下 RE-USE 的降噪强度没有公开数据，这正是保留 Inworld 开关的原因。
- LavaSR v2 只用 VCTK（英文朗读）训练了 5 万步，作者自己也提到交叉频率设低时可能有金属感；对中文和本人声音的效果只能靠盲听判断。它只看 0–8 kHz 的梅尔特征，对 48 kHz 录音而言，8 kHz 以上的高频是"按低频猜出来的"，不是对原高频的修复。
- Inworld MLX 移植是本项目工作量最大、风险最高的部分（64M 参数，含注意力与 Mamba 混合块）；对拍不过时，宁可先不发布该开关，也不发布不一致的移植。
- ANE 收益不确定：ReuseKit 只有 M2 和 A19 Pro 的数据；M4 Pro 的 GPU 更强，ANE 的相对收益可能更小。FP16 也可能在极安静的段落带来数值问题（ReuseKit 曾修过扫描溢出）。因此 ANE 必须过门槛才默认启用。
- 中间音频自动清理意味着重新调整旧会话要重跑 AI，耗时与录音时长成正比（48 kHz 下 RE-USE 在 M1 Max 上约为录音时长的 0.375 倍）。
- MP3 依赖 LAME 动态库，多一个第三方组件和 LGPL 义务。
- 评测证据有限：NVIDIA 的 URGENT 2025 结果为合成数据、非盲测；Inworld 的 VIBE 为私有数据集；LavaSR 的对比只有对数谱距离。以 §7 的实录盲听为准。

## 9. 第二轮调整（2026-11-22，用户反馈后）

- **内存预算**：AI 处理的内存峰值上限放宽到 5 GB，RE-USE 恢复 5 s 窗口（48 kHz 峰值 3.2 GB）。
- **母带设置记忆**：新会话（录音、导入、崩溃恢复）沿用最后一次选择的母带设置；预设菜单显示当前匹配的预设名，不匹配任何预设时显示“自定义”；用户预设可保存、删除。
- **录音流程**：点击停止立即结束并保存录音（默认名“M月d日 HH.mm 录音”），然后弹出命名框；会话可随时重命名（改 manifest、会话文件夹名和会话内成品文件名）。
- **界面语言**：中文 / English / 跟随系统，在工具栏地球图标切换，立即生效；静态文案走 String Catalog，运行时消息由 `AppModel.tr` 按当前语言生成。
- **试听三路**：处理完成后自动切到“AI 处理后”。“AI 处理后”= RE-USE（及可选 LavaSR）的结果，未做母带；“成品”= 在其上实时叠加母带处理，导出的就是成品——前者用来判断 AI 修复本身，后者用来调母带参数。
- **批量**：侧栏多选；批量删除（确认后删除会话文件夹）；“一键处理并导出”对所选会话依次 AI 处理 → 母带 → 导出。
- **导出位置**：可指定一个导出文件夹（记住）；设置后所有导出写到该文件夹（同名不同会话自动编号），会话文件夹只留原始音频和 `session.json`；未设置时仍导出到会话文件夹。`ExportRecord.directory` 记录外部位置。
- **工具栏**：导入、一键处理并导出（所选）、快速导出（当前会话，未处理过的先自动处理；首次询问导出文件夹）、语言。
- **高频增强（Plan 2）**：AI 处理面板新增“高频增强（LavaSR）”开关和起始频率（自动 / 8 / 12 / 16 kHz），默认关闭；权重来自 LavaSR 的 safetensors 镜像（固定 revision 与 SHA-256，运行时下载）；“自动”对低采样率导入取其奈奎斯特频率，48 kHz 录音按实测有效带宽（`BandwidthDetector`）。CLI `micpolish abx` 生成盲听用的乱序变体与答案 `key.json`。
