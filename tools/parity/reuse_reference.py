# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#   "torch>=2.6",
#   "einops",
#   "safetensors",
#   "numpy",
#   "soundfile",
#   "huggingface_hub",
#   "pyyaml",
# ]
# ///
"""PyTorch CPU reference output of NVIDIA NemotronLabs RE-USE for parity tests.

Downloads the upstream model code (OpenMDW-1.1) at the pinned revision, swaps
the CUDA-only `mamba_ssm.Mamba` for a pure-PyTorch equivalent, and runs the
upstream `inference.py` steps on the whole input in one forward pass.

    uv run tools/parity/reuse_reference.py --weights model.safetensors \
        --input in.wav --output ref.f32

The output is raw little-endian float32 samples at the input sample rate.
"""

import argparse
import math
import sys
import types

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

REPO = "nvidia/NVIDIA-NemotronLabs-RE-USE"
REVISION = "31f82f2efad1095034492f236e8dac9239ba713b"
RECIPE = "recipes/USEMamba_30x1_lr_00002_norm_05_vq_065_nfft_320_hop_40_NRIR_012_pha_0005_com_04_early_001.yaml"


class Mamba(nn.Module):
    """Pure-PyTorch `mamba_ssm.Mamba` (v1) with identical parameter names."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto", conv_bias=True, bias=False):
        super().__init__()
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner,
                                padding=d_conv - 1, bias=conv_bias)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.zeros(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, hidden_states):  # [B, L, D]
        _, length, _ = hidden_states.shape
        xz = self.in_proj(hidden_states).transpose(1, 2)  # [B, 2*Di, L]
        x, z = xz.chunk(2, dim=1)
        x = F.silu(self.conv1d(x)[..., :length])
        x_dbl = self.x_proj(x.transpose(1, 2))  # [B, L, R + 2N]
        dt, b, c = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(dt @ self.dt_proj.weight.t() + self.dt_proj.bias)  # [B, L, Di]
        a = -torch.exp(self.A_log)  # [Di, N]

        u = x.transpose(1, 2)  # [B, L, Di]
        state = torch.zeros(u.shape[0], self.d_inner, self.d_state, dtype=u.dtype)
        ys = []
        for t in range(length):
            d_a = torch.exp(delta[:, t, :, None] * a)
            d_bu = delta[:, t, :, None] * b[:, t, None, :] * u[:, t, :, None]
            state = d_a * state + d_bu
            ys.append((state * c[:, t, None, :]).sum(-1))
        y = torch.stack(ys, dim=1) + u * self.D
        y = y * F.silu(z.transpose(1, 2))
        return self.out_proj(y)


def make_even(value):
    value = int(round(value))
    return value if value % 2 == 0 else value + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--float64", action="store_true",
                        help="run in float64 to measure float32 rounding sensitivity")
    parser.add_argument("--probes", help="also write intermediate tensors to this safetensors file")
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(1)
    dtype = torch.float64 if args.float64 else torch.float32
    torch.set_default_dtype(dtype)

    code = snapshot_download(REPO, revision=REVISION,
                             allow_patterns=["models/*", "utils/*", "recipes/*"])
    sys.modules["mamba_ssm"] = types.SimpleNamespace(Mamba=Mamba)
    sys.path.insert(0, code)
    from models.generator_SEMamba_time_d4 import SEMamba  # noqa: E402
    from models.stfts import mag_phase_istft, mag_phase_stft  # noqa: E402
    from utils.util import load_config, pad_or_trim_to_match  # noqa: E402

    cfg = load_config(f"{code}/{RECIPE}")
    n_fft, hop, win = (cfg["stft_cfg"][k] for k in ("n_fft", "hop_size", "win_size"))
    compress = cfg["model_cfg"]["compress_factor"]
    base_sr = cfg["stft_cfg"]["sampling_rate"]

    model = SEMamba(cfg)
    model.load_state_dict(load_file(args.weights), strict=True)
    model.to(dtype).eval()

    audio, sr = sf.read(args.input, dtype="float32", always_2d=True)
    wav = torch.from_numpy(audio.mean(axis=1)).unsqueeze(0).to(dtype)

    n_fft_s = make_even(n_fft * sr // base_sr)
    hop_s = make_even(hop * sr // base_sr)
    win_s = make_even(win * sr // base_sr)

    probes = {}
    if args.probes:
        model.dense_encoder.register_forward_hook(
            lambda m, i, o: probes.__setitem__("encoder_out", o.detach().clone()))
        model.TSMamba[0].register_forward_hook(
            lambda m, i, o: probes.__setitem__("block_0_out", o.detach().clone()))
        model.TSMamba[-1].register_forward_hook(
            lambda m, i, o: probes.__setitem__("block_last_out", o.detach().clone()))

    with torch.no_grad():
        mag, pha, _ = mag_phase_stft(wav, n_fft=n_fft_s, hop_size=hop_s, win_size=win_s,
                                     compress_factor=compress, center=True, addeps=False)
        amp_g, pha_g, _ = model(mag, pha)
        probes.update(input_mag=mag, input_pha=pha, model_mag=amp_g.clone(), model_pha=pha_g)
        magnitude = torch.expm1(torch.relu(amp_g))
        zero_portion = torch.sum(magnitude == 0, 1) / magnitude.shape[1]
        amp_g[:, :, (zero_portion > 0.5)[0]] = 0
        out = mag_phase_istft(amp_g, pha_g, n_fft_s, hop_s, win_s, compress)
        out = pad_or_trim_to_match(wav, out, pad_value=1e-8)

    if args.probes:
        from safetensors.torch import save_file
        save_file({k: v.float().contiguous() for k, v in probes.items()}, args.probes)

    out.squeeze(0).numpy().astype("<f4").tofile(args.output)
    print(f"wrote {out.shape[-1]} samples at {sr} Hz (n_fft={n_fft_s}, hop={hop_s}, win={win_s})")


if __name__ == "__main__":
    main()
