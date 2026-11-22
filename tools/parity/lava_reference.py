# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "torch==2.8.0",
#   "torchaudio==2.8.0",
#   "vocos @ git+https://github.com/langtech-bsc/vocos.git@451e522f7a11c3652c9522f63ea6780736d93de0",
#   "numpy",
#   "soundfile",
#   "safetensors",
#   "huggingface_hub",
#   "pyyaml",
# ]
# ///
"""PyTorch CPU reference output of the LavaSR v2 bandwidth extender for parity tests.

Runs upstream `LavaBWE.infer` (LavaSR, Apache-2.0) with the original
`YatharthS/LavaSR` `enhancer_v2` checkpoint on a 48 kHz mono input, using the
crossover the upstream `LavaEnhance.load_audio` installs:
`FastLRMerge(cutoff=<cutoff>, transition_bins=1024)` at 48 kHz.

LavaSR's own `vocos` dependency is the langtech-bsc `matcha` branch; it is
pinned to a commit above. The two LavaSR files `infer` needs are fetched at a
pinned commit, SHA-256 verified, and loaded as the `LavaSR.enhancer` package
(the PyPI/git package would pull the denoiser and librosa as well).

    uv run tools/parity/lava_reference.py --input in.wav --cutoff 8000 --output ref.f32
    uv run tools/parity/lava_reference.py --check-mirror /path/to/model.safetensors

The output is raw little-endian float32 samples at 48 kHz, as many as the input.
Upstream `infer` returns floor(n / 512) * 512 samples (the ISTFT head emits one
hop per frame); for any other length this script, like the Swift port, zero-pads
the input to the next multiple of 512 (at least 1024: the 768-sample reflect
padding of the mel extractor needs more than 768 samples) before `infer` and
trims the result back to the input length. Inputs that are already a multiple
of 512 and at least 1024 samples long are passed to `infer` unchanged.
"""

import argparse
import hashlib
import sys
import types
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download, snapshot_download

MODEL_REPO = "YatharthS/LavaSR"
MODEL_REVISION = "b98dc8be472da45ab7b6346ad7997e1dfeb5911d"
LAVASR_COMMIT = "33ac040892519c1bb4aed7eb32e79af51cc29e2a"
LAVASR_FILES = {
    "linkwitz_merge": "08042edf084cd07874fb31c0b82753c058e3a2500110231d5efaf61cdab1f887",
    "enhancer": "355d338063f79d643016c2f1cd4900a4794af1ae0eb8229aed145183ba2d8907",
}
SAMPLE_RATE = 48_000
HOP = 512


def load_lavasr_enhancer():
    """Imports LavaSR/enhancer/{linkwitz_merge,enhancer}.py at the pinned commit."""
    for name in ("LavaSR", "LavaSR.enhancer"):
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package
    for stem, digest in LAVASR_FILES.items():
        url = f"https://raw.githubusercontent.com/ysharma3501/LavaSR/{LAVASR_COMMIT}/LavaSR/enhancer/{stem}.py"
        source = urllib.request.urlopen(url, timeout=60).read()
        actual = hashlib.sha256(source).hexdigest()
        if actual != digest:
            raise SystemExit(f"{url}: SHA-256 {actual} != pinned {digest}")
        name = f"LavaSR.enhancer.{stem}"
        module = types.ModuleType(name)
        module.__file__ = url
        sys.modules[name] = module
        exec(compile(source, url, "exec"), module.__dict__)
        setattr(sys.modules["LavaSR.enhancer"], stem, module)
    return sys.modules["LavaSR.enhancer.enhancer"], sys.modules["LavaSR.enhancer.linkwitz_merge"]


def check_mirror(mirror_path):
    from safetensors.torch import load_file

    original_path = hf_hub_download(MODEL_REPO, "enhancer_v2/pytorch_model.bin", revision=MODEL_REVISION)
    original = torch.load(original_path, map_location="cpu")
    mirror = load_file(mirror_path)
    if set(original) != set(mirror):
        raise SystemExit(f"key sets differ: only original {sorted(set(original) - set(mirror))}, "
                         f"only mirror {sorted(set(mirror) - set(original))}")
    for key in sorted(original):
        a, b = original[key], mirror[key]
        if a.dtype != b.dtype or a.shape != b.shape:
            raise SystemExit(f"{key}: {a.dtype}{list(a.shape)} vs mirror {b.dtype}{list(b.shape)}")
        if a.contiguous().numpy().tobytes() != b.contiguous().numpy().tobytes():
            raise SystemExit(f"{key}: values differ")
    total = sum(t.numel() for t in original.values())
    print(f"mirror OK: {len(original)} tensors, {total} values bit-identical to "
          f"{MODEL_REPO}@{MODEL_REVISION} enhancer_v2/pytorch_model.bin")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--cutoff", type=float, help="crossover frequency in Hz")
    parser.add_argument("--check-mirror", metavar="SAFETENSORS",
                        help="assert the safetensors mirror equals the original checkpoint bit-for-bit")
    parser.add_argument("--float64", action="store_true",
                        help="run in float64 to measure float32 rounding sensitivity")
    parser.add_argument("--probes", help="also write intermediate tensors to this safetensors file")
    args = parser.parse_args()

    if args.check_mirror:
        check_mirror(args.check_mirror)
        if not args.input:
            return
    if not (args.input and args.output and args.cutoff):
        parser.error("--input, --output and --cutoff are required")

    torch.set_num_threads(1)
    dtype = torch.float64 if args.float64 else torch.float32

    enhancer, linkwitz_merge = load_lavasr_enhancer()
    model_dir = snapshot_download(MODEL_REPO, revision=MODEL_REVISION, allow_patterns=["enhancer_v2/*"])
    bwe = enhancer.LavaBWE(f"{model_dir}/enhancer_v2", device="cpu")
    # As upstream LavaEnhance.load_audio(cutoff=...) does.
    bwe.lr_refiner = linkwitz_merge.FastLRMerge(device="cpu", cutoff=args.cutoff, transition_bins=1024)
    if args.float64:
        bwe.bwe_model.to(torch.float64)

    audio, sr = sf.read(args.input, dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE:
        raise SystemExit(f"{args.input}: expected {SAMPLE_RATE} Hz, got {sr}")
    if audio.shape[1] != 1:
        raise SystemExit(f"{args.input}: expected mono, got {audio.shape[1]} channels")
    wav = torch.from_numpy(audio[:, 0].copy()).unsqueeze(0).to(dtype)
    count = wav.shape[1]
    padded = max(-(-count // HOP) * HOP, 2 * HOP)
    if padded != count:
        wav = torch.nn.functional.pad(wav, (0, padded - count))

    probes = {}
    if args.probes:
        model = bwe.bwe_model
        model.feature_extractor.register_forward_hook(
            lambda m, i, o: probes.__setitem__("features", o.detach().clone()))
        model.backbone.register_forward_hook(
            lambda m, i, o: probes.__setitem__("backbone", o.detach().clone()))
        model.head.out.register_forward_hook(
            lambda m, i, o: probes.__setitem__("head_out", o.detach().clone()))
        model.head.istft.register_forward_hook(
            lambda m, i, o: probes.__setitem__("istft", o.detach().clone()))

    if args.float64:
        # LavaBWE.infer casts the merge inputs with .float(); repeat its steps without the casts.
        with torch.no_grad():
            model = bwe.bwe_model
            pred = model.head(model.backbone(model.feature_extractor(wav)))
            out = bwe.lr_refiner(pred[:, :wav.shape[1]], wav[:, :pred.shape[1]])[:, :count]
    else:
        out = bwe.infer(wav)[:, :count]
    if out.shape[1] != count:
        raise SystemExit(f"infer returned {out.shape[1]} samples for {count}")

    if args.probes:
        from safetensors.torch import save_file
        save_file({k: v.float().contiguous() for k, v in probes.items()}, args.probes)

    out.squeeze(0).double().numpy().astype("<f4").tofile(args.output)
    print(f"wrote {count} samples at {SAMPLE_RATE} Hz (cutoff {args.cutoff:g} Hz, "
          f"{'float64' if args.float64 else 'float32'}, padded to {padded})")


if __name__ == "__main__":
    main()
