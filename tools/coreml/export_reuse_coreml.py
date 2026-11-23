# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "torch==2.7.0",
#   "coremltools==9.0",
#   "numpy",
#   "einops",
#   "safetensors",
#   "huggingface_hub",
#   "pyyaml",
# ]
# ///
"""Exports the RE-USE dense encoder and the mask/phase decoders as Core ML packages.

The 30 Mamba blocks stay in MLX; only the convolutional front end and back end
are exported. Shapes are fixed to one 5 s window at 48 kHz (the app's only
ANE-eligible window): STFT n_fft=1920, hop=240 gives 961 bins x 1001 frames.

    uv run tools/coreml/export_reuse_coreml.py                 # default checkpoint
    uv run tools/coreml/export_reuse_coreml.py --variant laughter
    uv run tools/coreml/export_reuse_coreml.py --variant all --no-install --output /tmp/x

Each variant installs to
  ~/Library/Application Support/MicPolish/models/<id>-coreml/<revision>/
    ReuseEncoder.mlpackage   x [1,2,1001,961] fp16 (mag, pha; [B,C,T,F]) -> h [1,64,251,481] fp16
    ReuseDecoders.mlpackage  h [1,64,251,481] fp16 -> y [1,3,1001,961] fp16 (mag, phase real, phase imag)
    manifest.json            shapes, revisions, tool versions, per-package SHA-256
    LICENSE.txt              upstream OpenMDW-1.1 text

Both packages use FP16 compute precision and target macOS 14. The upstream
modules are loaded from the pinned model code and checkpoint; the export
wrappers only change tensor layout (no 5-D reshapes, no H/W permutes) and are
checked against the upstream modules in float32 before conversion.
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file

CODE_REPO = "nvidia/NVIDIA-NemotronLabs-RE-USE"
CODE_REVISION = "31f82f2efad1095034492f236e8dac9239ba713b"
RECIPE = "recipes/USEMamba_30x1_lr_00002_norm_05_vq_065_nfft_320_hop_40_NRIR_012_pha_0005_com_04_early_001.yaml"

VARIANTS = {
    "default": {
        "id": "reuse-nemotron",
        "repo": "nvidia/NVIDIA-NemotronLabs-RE-USE",
        "revision": "31f82f2efad1095034492f236e8dac9239ba713b",
        "sha256": "06badaf4029b297f97f7fe48d62990b48d5482608210df577354e0d4e0ed70e1",
    },
    "laughter": {
        "id": "reuse-nemotron-laughter",
        "repo": "nvidia/NVIDIA-NemotronLabs-RE-USE-laughter",
        "revision": "1b1521dd659608e1e063ada83e0acfe1da473a8b",
        "sha256": "35c2a4e284515275d2e49a8e8cb5cd392ae67a1f1d345de1c12a0202c7b6b610",
    },
}

# One 5 s window at 48 kHz: n_fft = 320 * 6 = 1920, hop = 40 * 6 = 240.
SAMPLE_RATE = 48_000
WINDOW_SAMPLES = 240_000
FREQ_BINS = 961
FRAMES = WINDOW_SAMPLES // 240 + 1  # 1001
# Upstream zero-pads T and F by 2; dense_conv_2 has kernel (1, 3), stride (4, 2).
LATENT_T = (FRAMES + 2 - 1) // 4 + 1  # 251
LATENT_F = (FREQ_BINS + 2 - 3) // 2 + 1  # 481
HIDDEN = 64
FORMAT_VERSION = 1


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path):
    """Deterministic digest over every file in a package directory."""
    digest = hashlib.sha256()
    root = Path(path)
    for file in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(str(file.relative_to(root)).encode())
        digest.update(bytes.fromhex(sha256_file(file)))
    return digest.hexdigest()


# --- Export wrappers ---------------------------------------------------------
# Upstream SPConvTranspose2d reshapes through a 5-D view and the decoders swap
# H and W around up_conv2. Core ML on the ANE handles rank-4 tensors best, so
# the wrappers express the same arithmetic with rank-4 reshapes (batch is 1)
# and run up_conv2 along H with a (3, 1) kernel instead of permuting.


def sp_conv_along_w(sp, x, h, w):
    """Upstream SPConvTranspose2d (pads W, interleaves r into W) for batch 1.

    `h` and `w` are the fixed input height and width; passing them as Python
    ints keeps shape arithmetic out of the traced graph.
    """
    c = sp.out_channels
    x = F.pad(x, (1, 1, 0, 0))
    out = F.conv2d(x, sp.conv.weight, sp.conv.bias)
    # [1, r*C, H, W] -> [r, C, H, W] -> [C, H, W, r] -> [1, C, H, W*r]
    out = out.reshape(sp.r, c, h, w).permute(1, 2, 3, 0)
    return out.reshape(1, c, h, w * sp.r)


def sp_conv_along_h(sp, x, h, w):
    """Equals `sp(x.permute(0,1,3,2)).permute(0,1,3,2)` for batch 1."""
    c = sp.out_channels
    x = F.pad(x, (0, 0, 1, 1))
    weight = sp.conv.weight.permute(0, 1, 3, 2)  # (1, 3) kernel -> (3, 1)
    out = F.conv2d(x, weight, sp.conv.bias)
    # [1, r*C, H, W] -> [r, C, H, W] -> [C, H, r, W] -> [1, C, H*r, W]
    out = out.reshape(sp.r, c, h, w).permute(1, 2, 0, 3)
    return out.reshape(1, c, h * sp.r, w)


def norm_act(seq, x):
    """Applies the InstanceNorm2d and PReLU of an upstream (conv, norm, act) Sequential."""
    return seq[2](seq[1](x))


def decoder_trunk(dec, x):
    """Upstream decoder trunk on the fixed [1, 64, LATENT_T, LATENT_F] latent."""
    x = dec.dense_block(x)
    x = norm_act(dec.up_conv1, sp_conv_along_w(dec.up_conv1[0], x, LATENT_T, LATENT_F))
    x = norm_act(dec.up_conv2, sp_conv_along_h(dec.up_conv2[0], x, LATENT_T, 2 * LATENT_F))
    return x


class EncoderExport(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):  # [1, 2, T, F] (mag, pha)
        x = F.pad(x, (0, 2, 0, 2))
        return self.encoder(x)


class DecodersExport(nn.Module):
    def __init__(self, mag_decoder, phase_decoder):
        super().__init__()
        self.mag = mag_decoder
        self.pha = phase_decoder

    def forward(self, h):  # [1, 64, T', F']
        mag = self.mag.final_conv(decoder_trunk(self.mag, h))
        trunk = decoder_trunk(self.pha, h)
        real = self.pha.phase_conv_r(trunk)
        imag = self.pha.phase_conv_i(trunk)
        y = torch.cat([mag, real, imag], dim=1)
        # Upstream crops the model padding after decoding.
        return y[:, :, :FRAMES, :FREQ_BINS]


def relative_error(actual, expected):
    actual = actual.astype(np.float64)
    expected = expected.astype(np.float64)
    return float(np.sqrt(np.sum((actual - expected) ** 2) / np.sum(expected**2)))


def export_variant(name, variant, code_dir, cfg, out_root, check_coreml):
    from models.codec_module_time_d4 import DenseEncoder, MagDecoder, PhaseDecoder  # noqa: E402

    weights = hf_hub_download(variant["repo"], "model.safetensors", revision=variant["revision"])
    actual_sha = sha256_file(weights)
    if actual_sha != variant["sha256"]:
        raise SystemExit(f"{name}: checkpoint SHA-256 {actual_sha} != pinned {variant['sha256']}")
    state = load_file(weights)

    def sub_state(prefix):
        return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}

    encoder = DenseEncoder(cfg)
    encoder.load_state_dict(sub_state("dense_encoder."), strict=True)
    mag_decoder = MagDecoder(cfg)
    mag_decoder.load_state_dict(sub_state("mask_decoder."), strict=True)
    phase_decoder = PhaseDecoder(cfg)
    phase_decoder.load_state_dict(sub_state("phase_decoder."), strict=True)
    for module in (encoder, mag_decoder, phase_decoder):
        module.float().eval()

    enc_export = EncoderExport(encoder).eval()
    dec_export = DecodersExport(mag_decoder, phase_decoder).eval()

    # Inputs that look like the model's: log1p magnitude and phase in [-pi, pi].
    gen = torch.Generator().manual_seed(0)
    mag = torch.log1p(torch.rand(1, 1, FRAMES, FREQ_BINS, generator=gen) * 4)
    pha = (torch.rand(1, 1, FRAMES, FREQ_BINS, generator=gen) * 2 - 1) * np.pi
    x = torch.cat([mag, pha], dim=1)

    with torch.no_grad():
        # Reference: the upstream SEMamba forward around the Mamba stack.
        padded = torch.cat([x, torch.zeros(1, 2, FRAMES, 2)], dim=-1)
        padded = torch.cat([padded, torch.zeros(1, 2, 2, FREQ_BINS + 2)], dim=-2)
        h_ref = encoder(padded)
        mag_ref = mag_decoder(h_ref)[:, :, :FRAMES, :FREQ_BINS]
        pha_ref = phase_decoder(h_ref)[:, :, :FRAMES, :FREQ_BINS]

        h = enc_export(x)
        y = dec_export(h_ref)
        assert tuple(h.shape) == (1, HIDDEN, LATENT_T, LATENT_F), h.shape
        assert tuple(y.shape) == (1, 3, FRAMES, FREQ_BINS), y.shape
        wrapper_errors = {
            "encoder": relative_error(h.numpy(), h_ref.numpy()),
            "mag": relative_error(y[:, 0:1].numpy(), mag_ref.numpy()),
            "phase": float((torch.atan2(y[:, 2:3], y[:, 1:2]) - pha_ref).abs().max()),
        }
    print(f"{name}: export wrappers vs upstream (float32): {wrapper_errors}")
    if max(wrapper_errors.values()) > 1e-5:
        raise SystemExit(f"{name}: export wrappers diverge from the upstream modules")

    out_dir = Path(out_root) / f"{variant['id']}-coreml" / variant["revision"]
    staging = out_dir.with_name(out_dir.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)

    def convert(module, example, input_name, output_name, path):
        traced = torch.jit.trace(module, example)
        model = ct.convert(
            traced,
            inputs=[ct.TensorType(name=input_name, shape=tuple(example.shape), dtype=np.float16)],
            outputs=[ct.TensorType(name=output_name, dtype=np.float16)],
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.macOS14,
        )
        model.author = "MicPolish tools/coreml/export_reuse_coreml.py"
        model.license = "OpenMDW-1.1 (NVIDIA NemotronLabs RE-USE)"
        model.short_description = (
            f"RE-USE {variant['repo']}@{variant['revision']} {path.stem}, 48 kHz 5 s window"
        )
        model.save(str(path))
        return model

    with torch.no_grad():
        enc_model = convert(enc_export, x, "x", "h", staging / "ReuseEncoder.mlpackage")
        dec_model = convert(dec_export, h_ref, "h", "y", staging / "ReuseDecoders.mlpackage")

    coreml_errors = {}
    if check_coreml:
        enc_out = enc_model.predict({"x": x.numpy().astype(np.float16)})["h"].astype(np.float32)
        dec_out = dec_model.predict({"h": h_ref.numpy().astype(np.float16)})["y"].astype(np.float32)
        # Phase comes out as (real, imag) whose atan2 is ill-conditioned where
        # both are near zero, so compare the pair, not the angle.
        coreml_errors = {
            "encoder_fp16": relative_error(enc_out, h_ref.numpy()),
            "mag_fp16": relative_error(dec_out[:, 0:1], y[:, 0:1].numpy()),
            "phase_real_imag_fp16": relative_error(dec_out[:, 1:3], y[:, 1:3].numpy()),
        }
        print(f"{name}: Core ML (FP16, CPU_AND_NE) vs float32 on synthetic input: {coreml_errors}")

    shutil.copy(Path(code_dir) / "LICENSE", staging / "LICENSE.txt")
    manifest = {
        "formatVersion": FORMAT_VERSION,
        "modelID": variant["id"],
        "repo": variant["repo"],
        "revision": variant["revision"],
        "checkpointSHA256": variant["sha256"],
        "codeRepo": CODE_REPO,
        "codeRevision": CODE_REVISION,
        "sampleRate": SAMPLE_RATE,
        "windowSamples": WINDOW_SAMPLES,
        "frames": FRAMES,
        "frequencyBins": FREQ_BINS,
        "latentFrames": LATENT_T,
        "latentBins": LATENT_F,
        "hiddenChannels": HIDDEN,
        "computePrecision": "float16",
        "tools": {
            "coremltools": ct.__version__,
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "packages": {
            "encoder": {
                "file": "ReuseEncoder.mlpackage",
                "input": {"name": "x", "shape": [1, 2, FRAMES, FREQ_BINS]},
                "output": {"name": "h", "shape": [1, HIDDEN, LATENT_T, LATENT_F]},
                "sha256": sha256_tree(staging / "ReuseEncoder.mlpackage"),
            },
            "decoders": {
                "file": "ReuseDecoders.mlpackage",
                "input": {"name": "h", "shape": [1, HIDDEN, LATENT_T, LATENT_F]},
                "output": {"name": "y", "shape": [1, 3, FRAMES, FREQ_BINS]},
                "sha256": sha256_tree(staging / "ReuseDecoders.mlpackage"),
            },
        },
        "exportChecks": {"wrappersFloat32": wrapper_errors, "coreML": coreml_errors},
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    shutil.rmtree(out_dir, ignore_errors=True)
    staging.rename(out_dir)
    print(f"{name}: installed {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=[*VARIANTS, "all"], default="default")
    parser.add_argument(
        "--output",
        default=str(Path.home() / "Library/Application Support/MicPolish/models"),
        help="models root; packages go to <output>/<id>-coreml/<revision>/",
    )
    parser.add_argument("--no-install", action="store_true",
                        help="write to a new temporary directory instead of the app's models folder "
                             "(ignored when --output is given)")
    parser.add_argument("--skip-coreml-check", action="store_true",
                        help="do not run the converted models through Core ML after export")
    args = parser.parse_args()
    if args.no_install and args.output == parser.get_default("output"):
        args.output = tempfile.mkdtemp(prefix="micpolish-coreml-")
        print(f"Writing packages to {args.output}")

    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    code_dir = snapshot_download(CODE_REPO, revision=CODE_REVISION,
                                 allow_patterns=["models/*", "utils/*", "recipes/*", "LICENSE"])
    # The upstream package imports mamba_ssm at module load; the exported parts
    # never touch it.
    sys.modules.setdefault("mamba_ssm", type(sys)("mamba_ssm"))
    sys.modules["mamba_ssm"].Mamba = object
    sys.path.insert(0, code_dir)
    from utils.util import load_config  # noqa: E402

    cfg = load_config(os.path.join(code_dir, RECIPE))
    names = list(VARIANTS) if args.variant == "all" else [args.variant]
    for name in names:
        export_variant(name, VARIANTS[name], code_dir, cfg, args.output,
                       check_coreml=not args.skip_coreml_check)


if __name__ == "__main__":
    main()
