# Performance and storage measurements

Measured on 2026-11-22 on a MacBook Pro with an **Apple M4 Pro** (24 GB), macOS 27.3,
Xcode 27.2, mlx-swift 0.31.6, release build of the `micpolish` CLI.

## RE-USE speed

Input: 64 s of noisy speech at 48 kHz mono (the parity fixture repeated 16 times).
Command: `micpolish process speech64s.wav --output perf.m4a` (default preset), three runs.

| Run | RE-USE time | RTF |
| --- | ---: | ---: |
| 1 | 34.34 s | 0.536 |
| 2 | 34.39 s | 0.537 |
| 3 | 34.78 s | 0.544 |

Median RTF **0.537**: one minute of 48 kHz audio takes about 32 s. The figure includes model
loading and Metal kernel compilation on the first window. RE-USE ran in 4 s windows with 5%
overlap for this measurement (the default is now 5 s windows, within the 5 GB memory budget) (`ReuseStage.inferenceSettings`); this is MLX on the GPU only — the Neural Engine
backend is Plan 4. Mastering and export add a few seconds per minute of audio on the CPU.

For comparison, SpeechLens reports RTF 0.375 at 48 kHz on an M1 Max (32-core GPU) with 5 s
windows.

## Memory

From `PipelineTests.testTwentyMinuteInputKeepsPeakMemoryBounded` (debug build):

| Window | MLX peak (48 kHz) |
| --- | ---: |
| **5 s (used)** | **3221 MB** |
| 4.5 s | 3130 MB |
| 4 s | 2835 MB |
| 3 s | 2370 MB |

The peak is the same for a 30 s and a 20 min input; resident memory grew by 4 MB over the
20-minute run. The preview player holds the original and processed signals in memory
(about 1.4 GB per hour of 48 kHz audio).

## Loudness accuracy

Every CLI run above ended at −16.1 LUFS integrated and −1.20 dBTP true peak for a −16 LUFS /
−1.0 dBTP target. The renderer re-runs the full chain to correct the gain when the limiter
removes loudness.

## Storage per session

| File | Format | Size |
| --- | --- | ---: |
| Original recording | FLAC, 48 kHz, 24-bit, mono | about 4.9 MB per minute (noisy speech fixture) |
| Export | M4A AAC-LC 128 kbps mono | about 0.96 MB per minute |
| Export | MP3 128 kbps CBR mono | about 0.96 MB per minute |
| Intermediates | Float32 CAF in `~/Library/Caches/MicPolish/work` | about 11.5 MB per minute per stage; deleted after export and on quit |

## Numerical parity

`ReuseParityTests` compares one full MLX window with a PyTorch CPU reference of the
NemotronLabs checkpoint (`tools/parity/reuse_reference.py`), both against a float64 run:

| Rate | MLX vs float64 | PyTorch float32 vs float64 |
| --- | ---: | ---: |
| 16 kHz | 7.0e-6 | 9.8e-7 |
| 48 kHz | 1.1e-4 | 8.5e-5 |

At 48 kHz the fixture is upsampled from 44.1 kHz, so bins above 22.05 kHz are nearly empty
and their phase depends on float32 rounding in any implementation.

## ANE hybrid backend (Plan 4)

RE-USE's dense encoder and mask/phase decoders run as FP16 Core ML (`cpuAndNeuralEngine`); the
30 Mamba layers stay in MLX. Release build, M4 Pro, macOS 27.3, 64 s of 48 kHz speech:

| Backend | RTF | Peak footprint (`/usr/bin/time -l`) | Windows |
| --- | ---: | ---: | --- |
| MLX GPU (`micpolish process --backend mlx`) | 0.480 | 4.52 GB | 14 MLX |
| ANE hybrid (`--backend ane`) | 0.407 | 4.27 GB | 13 ANE + 1 MLX (tail) |

Subagent benchmark (median of 3 after warm-up): MLX 0.481, hybrid 0.405–0.429 (−10.8 % to
−16.5 %). Quality vs MLX FP32: rel. RMSE 0.106 % (default) / 0.221 % (laughter). Per 5 s window
the Mamba layers take ~1.75 s on the GPU; the encoder drops 310 → 187 ms and the decoders
141 → 118 ms on the ANE, and Core ML work for neighbouring windows overlaps the GPU pass.
MLComputePlan places every operation of both models on the Neural Engine.

The packages are built at development time (`uv run tools/coreml/export_reuse_coreml.py --variant all`)
and installed to `~/Library/Application Support/MicPolish/models/<model>-coreml/<revision>/`.
"Automatic" uses the hybrid only if the in-app speed test (Settings) measured at least 15 % lower RTF.

## MLX buffer cache

RE-USE caps MLX's buffer cache at 2048 MB so buffers freed after each window are reused instead of
accumulating across a long file. Set `MICPOLISH_CACHE_LIMIT_MB` to change it: a positive value is the
cap in MB, `0` disables the cache, a negative value leaves MLX's default. The processing pipeline also
empties the cache after every stage, so consecutive models' peaks do not add up. `MICPOLISH_DISABLE_METAL_CONV1D=1`
replaces the custom causal depthwise Conv1d kernel with MLXNN's Conv1d (for comparison or debugging).
