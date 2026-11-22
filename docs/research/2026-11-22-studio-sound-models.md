# 录音棚效果（类似 dxRevive Pro）模型调研 — 2026-11-22

## dxRevive Pro 做了什么
官方手册：它的 Studio 算法是训练好的神经网络，目标是“像在声学处理过的房间里、贴近大振膜话筒录的一样”：去噪、去混响、补回缺失频率（如手机录音），**并且按目标音色做 EQ**；Natural 算法同样去噪去混响补频，但几乎不改 EQ；EQ Restore 只修频谱不去噪。来源：<https://www.accentize.com/products/dxReviveProManual.pdf>

MicPolish 现有链路已覆盖其中三项：去噪去混响（RE-USE）、强降噪（Inworld）、补高频（LavaSR）。**缺的是“往录音棚话筒音色靠”的那一步**。

## 候选

| 候选 | 做什么 | 输出 | 规模 | 生成式？ | 代码 / 权重许可 | Apple Silicon | 中文 | 公开指标 |
|---|---|---|---|---|---|---|---|---|
| **Sidon v0.1**（东大 sarulab） | 去噪+去混响+补频，按录音棚数据重新合成 | 48 kHz | 246M（w2v-BERT 2.0 8 层 193.6M + DAC 解码器 52.4M） | 是（特征清洗后用声码器重合成） | 代码 MIT；权重 MIT（[model card](https://huggingface.co/sarulab-speech/sidon-v0.1)） | 现成 Core ML 包 [aufklarer/Sidon-CoreML](https://huggingface.co/aufklarer/Sidon-CoreML)（MIT，fp16 / int8）；Swift 集成 [soniqo/speech-swift](https://github.com/soniqo/speech-swift)（Apache-2.0），预测器跑 ANE、声码器跑 GPU | 多语言，论文有普通话结果 | FLEURS 普通话：说话人相似度 0.984，CER 0.266→0.280，DNSMOS 3.11→3.37（[arXiv 2509.17052](https://arxiv.org/html/2509.17052v3)）；LibriTTS test-clean 说话人相似度 0.971；混响样例说话人余弦 0.795（[soniqo](https://huggingface.co/soniqo/Sidon-ONNX)） |
| Resemble Enhance | 去噪 + 增强（潜空间流匹配 + UniVNet） | 44.1 kHz | ~1.4 GB fp32 | 是 | 代码 MIT；HF 权重卡写 MIT（[HF](https://huggingface.co/ResembleAI/resemble-enhance)），但第三方 MLX 转换标 CC-BY-NC（[smcleod](https://huggingface.co/smcleod/resemble-enhance-mlx)）——许可说法冲突，未核实 | 有 MLX/Swift 移植（resemble-enhance-swift） | 训练语言标 en | 无可比公开数字（未核实） |
| VoiceRestore | 流匹配 Transformer 通用修复 | 未核实 | 301M | 是 | MIT（[HF](https://huggingface.co/jadechoghari/VoiceRestore)） | 无现成移植 | 未核实 | 作者称仍在训练 |
| FINALLY（Samsung，NeurIPS 2024，“studio-like quality”） | 通用增强到 48 kHz | 48 kHz | 459M（含 WavLM-Large） | 是（GAN） | 只有非官方训练代码，**没有公开权重**（[inverse-ai](https://github.com/inverse-ai/FINALLY-Speech-Enhancement)） | — | — | 非官方实现自述：低信噪比时口音/音色会变 |
| Miipher-2（Google） | 通用修复 | — | — | 是 | **未开放权重**；开源复现为 CC-BY-NC 且小规模（[Atotti](https://huggingface.co/Atotti/miipher-2-HuBERT-HiFi-GAN-v0.1)） | — | — | — |

## 建议
1. **先原型 Sidon，作为“录音棚”模式（替代 RE-USE，而不是叠加）**：许可干净（MIT 全链路）、输出 48 kHz、训练目标就是录音棚数据（EARS、Expresso、HiFi-CAPTAIN 等 48 kHz 棚录语料）、普通话有公开结果，而且已有 Core ML 包和 Swift 参考实现，接入成本远低于从零移植。风险是生成式重合成：说话人相似度在干净/中等噪声下很高（0.97–0.98），在重混响样例上降到 0.80，需要你用自己的录音盲听确认音色不变。
2. **再加一个确定性的“录音棚音色”EQ 匹配**（不用新模型）：测处理后语音的长期平均频谱，向近讲大振膜话筒的参考曲线匹配（限制 ±6 dB、平滑），放进母带链。对应 dxRevive 的“EQ 到录音棚话筒”那部分，不会改变说话人特征，可与 RE-USE 或 Sidon 任一配合。
3. 不建议：FINALLY（无权重）、Miipher-2（无权重/非商用）、Resemble Enhance（许可冲突、英文训练、体积大，且 NVIDIA 对比中音色/字准率下降——见设计文档 §4.4 D）。
