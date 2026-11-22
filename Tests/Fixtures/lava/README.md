# LavaSR v2 parity fixtures

- `speech_8k_band.wav`: `../reuse/input_16k.wav` resampled with ffmpeg to 48 kHz
  (mono Float32, 4.0 s, 192000 samples), so it has no content above 8 kHz.
- `speech_full_band.wav`: the first 120077 samples (2.5 s) of
  `../reuse/input_48k.wav` (48 kHz mono Float32, content up to 22 kHz). The length
  is deliberately not a multiple of the 512-sample hop.
- `ref_8k_band_c8000.f32`, `ref_full_band_c12000.f32`: raw little-endian Float32
  output of `tools/parity/lava_reference.py` (upstream `LavaBWE.infer`, PyTorch 2.8.0
  CPU, float32) on those inputs with crossover cutoffs of 8000 Hz and 12000 Hz.
  Regenerating them reproduces the files bit for bit.

Upstream code and weights: LavaSR (https://github.com/ysharma3501/LavaSR, commit
`33ac0408…`, Apache-2.0) with the `enhancer_v2` checkpoint of
`YatharthS/LavaSR` (revision `b98dc8be…`, Apache-2.0), and Vocos from the
langtech-bsc `matcha` branch that LavaSR depends on (commit `451e522f…`, MIT).
The input audio derives from NVIDIA materials licensed under OpenMDW-1.1 (see
`../reuse/README.md` and `Sources/ModelStore/Resources/licenses/OpenMDW-1.1.txt`);
the references are model outputs on that audio.

Weights mirror: the Swift port loads
`AEmotionStudio/lavasr-models` revision `291b7810…` `enhancer_v2/model.safetensors`
(SHA-256 `0b2be25e…2e79`). `lava_reference.py --check-mirror` confirmed all 83
tensors (14,071,378 values) bit-identical, with equal names, dtypes and shapes, to
the original `YatharthS/LavaSR` `enhancer_v2/pytorch_model.bin`
(SHA-256 `d100db96…6de12`).

Regenerate:

    uv run tools/parity/lava_reference.py --input Tests/Fixtures/lava/speech_8k_band.wav \
        --cutoff 8000 --output Tests/Fixtures/lava/ref_8k_band_c8000.f32
    uv run tools/parity/lava_reference.py --input Tests/Fixtures/lava/speech_full_band.wav \
        --cutoff 12000 --output Tests/Fixtures/lava/ref_full_band_c12000.f32
    uv run tools/parity/lava_reference.py --check-mirror <model.safetensors>

Run the parity tests with
`MICPOLISH_LAVA_WEIGHTS=<model.safetensors> tools/test.sh --filter LavaMLXTests`.
