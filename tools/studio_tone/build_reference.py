# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy==2.3.3", "soundfile==0.13.1", "pyarrow==21.0.0"]
# ///
"""Builds the "studio tone" reference curves from VCTK 0.92 (CC BY 4.0).

VCTK was recorded in a hemi-anechoic chamber; `mic2` is a Sennheiser MKH 800
studio condenser at close range, so its long-term speech spectrum stands in for
"close large-diaphragm microphone in a treated room". The analysis below is
the same one `Sources/Mastering/StudioTone.swift` runs on the user's audio:

- 48 kHz mono, 4096-point periodic Hann frames, hop 2048.
- Frames whose 100 Hz-8 kHz power is within 30 dB of the 95th-percentile frame
  are speech; the rest (silence, breaths) are ignored.
- Octave bands centred on 63 Hz ... 16 kHz (edges centre/sqrt2 .. centre*sqrt2),
  power-averaged over the kept frames, in dB.
- The shape is the band levels minus their mean over 250 Hz-4 kHz.

Each speaker's frames are pooled into one shape; a gender's reference is the
mean of its speakers' shapes (equal weight per speaker).

Usage: uv run tools/studio_tone/build_reference.py /tmp/vctk/*.parquet
Input: parquet shards of the Hugging Face mirror `sanchit-gandhi/vctk`
(VCTK 0.92, 48 kHz FLAC, both microphones). Prints the Swift table.
"""
import collections
import io
import sys

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

N, HOP, FS = 4096, 2048, 48_000
CENTERS = [63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
WINDOW = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N) / N)  # periodic Hann
FREQS = np.fft.rfftfreq(N, 1 / FS)
LEVEL_BINS = (FREQS >= 100) & (FREQS <= 8000)
BAND_BINS = [(FREQS >= c / 2**0.5) & (FREQS < c * 2**0.5) for c in CENTERS]
SHAPE_BANDS = slice(2, 7)  # 250 Hz - 4 kHz


def frame_powers(x):
    if len(x) < N:
        return np.zeros((0, len(FREQS)))
    count = 1 + (len(x) - N) // HOP
    frames = np.stack([x[i * HOP:i * HOP + N] * WINDOW for i in range(count)])
    return np.abs(np.fft.rfft(frames, axis=1)) ** 2


def kept(powers):
    level = 10 * np.log10(powers[:, LEVEL_BINS].sum(axis=1) + 1e-20)
    return powers[level >= np.percentile(level, 95) - 30]


def shape(powers):
    bands = np.array([powers[:, b].sum(axis=1).mean() for b in BAND_BINS])
    db = 10 * np.log10(bands + 1e-20)
    return db - db[SHAPE_BANDS].mean()


def main(paths):
    by_speaker = collections.defaultdict(list)
    gender = {}
    for path in paths:
        table = pq.read_table(path, columns=["speaker_id", "file", "gender", "audio"]).to_pydict()
        for spk, file, g, audio in zip(table["speaker_id"], table["file"], table["gender"], table["audio"]):
            if "mic2" not in file:
                continue
            x, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float64")
            assert sr == FS, sr
            if x.ndim > 1:
                x = x.mean(axis=1)
            p = frame_powers(x)
            if len(p):
                by_speaker[spk].append(p)
                gender[spk] = g
    shapes = collections.defaultdict(list)
    for spk, chunks in sorted(by_speaker.items()):
        s = shape(kept(np.concatenate(chunks)))
        shapes[gender[spk]].append(s)
        print(f"// {spk} {gender[spk]} {len(chunks)} utterances: " + " ".join(f"{v:+.1f}" for v in s), file=sys.stderr)
    for g in ("M", "F"):
        mean = np.mean(shapes[g], axis=0)
        spread = np.std(shapes[g], axis=0)
        print(f"// {g}: {len(shapes[g])} speakers; per-band std " + " ".join(f"{v:.1f}" for v in spread))
        print(f"static let {'male' if g == 'M' else 'female'}: [Double] = [" + ", ".join(f"{v:.2f}" for v in mean) + "]")


if __name__ == "__main__":
    main(sys.argv[1:])
