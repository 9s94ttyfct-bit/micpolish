# RE-USE parity fixtures

- `input_48k.wav`, `input_16k.wav`: the first 4.0 s of `noisy_audio/mic_test2.wav`
  from `nvidia/NVIDIA-NemotronLabs-RE-USE` (revision `31f82f2e…`, 44.1 kHz mono),
  resampled with ffmpeg to 48 kHz and 16 kHz, mono Float32.
- `ref_48k.f32`, `ref_16k.f32`: raw little-endian Float32 output of
  `tools/parity/reuse_reference.py` (PyTorch CPU, float32) on those inputs.

The audio and the reference outputs derive from NVIDIA materials licensed under
OpenMDW-1.1 (see `Sources/ModelStore/Resources/licenses/OpenMDW-1.1.txt`).

Regenerate:

    uv run tools/parity/reuse_reference.py --weights <model.safetensors> \
        --input Tests/Fixtures/reuse/input_48k.wav --output Tests/Fixtures/reuse/ref_48k.f32
