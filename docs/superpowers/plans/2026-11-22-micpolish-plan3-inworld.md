# Plan 3 — Inworld Voice Isolator (offline) MLX port

> Executed by one implementation subagent (port + parity) and the main agent (pipeline/UI integration).
> Spec: `docs/superpowers/specs/2026-11-22-micpolish-design.md` §4.2, §5.1, §5.4, §7 item 2.

**Goal:** an optional "strong noise removal" stage before RE-USE, running Inworld `voice-isolator-v1` (offline, 64M params) natively in Swift/MLX, numerically matching upstream PyTorch.

**Upstream (pin these):**
- Code: `github.com/inworld-ai/voice-isolator` (Apache-2.0) — `src/inworld/voice_isolation/` (`model.py`, `isolator.py`, `registry.py`), golden fixtures in `tests/fixtures`. Pin the commit you read.
- Weights: HF `inworld/voice-isolator` @ `9b0285b3fe8403aa8e2f1dbb2165a80a4294c970`, file `denoiser.safetensors`, 256,260,128 bytes, SHA-256 `647335a6aeb4dc706b895532c209d51e0e84a4849d17b339f00d4392a050ae53`. Installed as-is (`ModelDescriptor.Format.asIs`); the Swift loader converts layouts at load time.

## Global constraints
- Swift 6 / mlx-swift 0.31.6 (already a dependency); no Python at run time; macOS 14+.
- Model runs at 48 kHz mono. Peak memory for the whole stage ≤ 5 GB (spec cap); one checkpoint-sized chunk in flight.
- Parity gate: relative RMSE ≤ 1e-3 vs upstream PyTorch float32 CPU on each fixture (same gate as LavaSR); also meet any tolerance declared by upstream's own golden fixtures.
- Offline chunking/stitching must reproduce upstream `isolator.py` offline mode exactly (chunk length, overlap, window, edge handling), read from the checkpoint's own config where upstream does.
- Tests needing weights read `MICPOLISH_INWORLD_WEIGHTS` and `XCTSkip` when unset. Run tests with `tools/test.sh --filter InworldMLXTests` (it places `mlx.metallib`).

## Task A — `InworldMLX` module (subagent)
**Files (owned):** `Sources/InworldMLX/*`, `Tests/InworldMLXTests/*`, `Tests/Fixtures/inworld/*` (+ README with provenance/licences), `tools/parity/inworld_reference.py`, `Package.swift` (add library product + target `InworldMLX` depending on MLX/MLXFast/MLXNN, optionally `ReuseMLX` to reuse its Metal selective-scan kernel; add test target `InworldMLXTests`).

**Produces (contract, exact):**
```swift
public enum InworldIsolatorError: Error { case weightsNotFound(URL), missingTensor(String), unexpectedShape(String, expected: [Int], actual: [Int]) }

public final class InworldIsolator: @unchecked Sendable {
    public static let sampleRate = 48_000
    public init(weightsURL: URL) throws
    /// Starts one offline isolation pass over a single stream.
    public func makeSession() -> InworldSession
}

/// Feeds 48 kHz mono samples in pieces of any size; returns the isolated
/// samples that are final so far. Output total == input total. Memory bounded
/// by one chunk plus its overlap, independent of stream length.
public final class InworldSession {
    public func push(_ samples: [Float]) throws -> [Float]
    public func finish() throws -> [Float]
}
```
**Acceptance:**
- `InworldParityTests`: two fixtures (≈4 s band-limited/noisy speech and a stream > 2 chunks, e.g. 20 s, to exercise stitching), push in uneven piece sizes, compare to `inworld_reference.py` output; rel. RMSE ≤ 1e-3; length preserved; all finite. Upstream golden fixtures pass at their own tolerance if they apply to the offline model.
- Weight-free tests: STFT/iSTFT round trip, bad/missing checkpoint throws, chunk-boundary lengths (exact multiples, 1 sample, < one hop) keep length.
- Report: RTF and peak memory on a 60 s input (release or debug, say which); mirror nothing — weights come straight from the pinned HF revision.

## Task B — integration (main agent, after A)
- `ModelCatalog.inworld` (values above, `licenseResource: "Apache-2.0-Inworld.txt"`, `.asIs`), license text in `Sources/ModelStore/Resources/licenses/`.
- `Pipeline/InworldStage.swift`: `AudioStage` before RE-USE; resamples non-48 kHz input to 48 kHz first; output 48 kHz; `settingsKey` = revision; streams through `InworldSession`.
- `ProcessingSettings.strongDenoise: Bool` (default false, decodes absent as false); AppModel builds the stage when on; UI toggle "强降噪（Inworld）" in the AI card with a one-line explanation; CLI `--strong-denoise`; `abx` gains Inworld→RE-USE vs RE-USE variants.
- NOTICE / README / spec §9 updated.

## Review focus
- Recordings whose length is an exact multiple of the chunk hop (tail must not be re-isolated or dropped).
- Imports at 16/44.1 kHz (resample before the model; RE-USE then runs at 48 kHz).
- Cancel mid-stage leaves no partial cache entry.
- Turning the toggle on/off invalidates only the downstream caches.
- First-use download of 256 MB shows progress and is cancellable.
