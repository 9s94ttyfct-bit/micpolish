# Plan 4 — ANE hybrid backend for RE-USE

> Spec: `docs/superpowers/specs/2026-11-22-micpolish-design.md` §5.6 (candidate 1), §3 (Core ML packages), §7.
> Executed by one implementation subagent (export + Swift backend + gates) and the main agent (UI/settings integration).

**Goal:** an optional RE-USE backend that runs the dense encoder and the mask/phase decoders as Core ML (FP16, `cpuAndNeuralEngine`) and keeps the 30 Mamba layers in MLX; default only where it is measurably faster and numerically within the gate. MLX stays the baseline and the fallback.

## Global constraints
- 48 kHz sessions only, fixed 5 s windows (`ReuseStage.inferenceSettings`: 5.0 s, 5 % overlap); other rates and any Core ML load/run failure fall back to MLX for that window/session, logged.
- Peak process memory ≤ 5 GB (user cap).
- Quality gate: 48 kHz real speech ≥ 60 s, waveform relative RMSE vs MLX FP32 output ≤ 1 %. Speed gate: end-to-end RTF ≥ 15 % lower than MLX on this Mac (M4 Pro), warm, median of ≥ 3 runs. If the quality gate fails, do not ship the package or the option; report why.
- Core ML packages are produced at dev time with coremltools from the upstream PyTorch code (`tools/parity/reuse_reference.py` already pins it); never at run time. Install target: `~/Library/Application Support/MicPolish/models/reuse-nemotron-coreml/<reuse revision>/` (+ the laughter variant if the same export applies).
- No Python at run time; macOS 14+.

## Task A — subagent
**Owns:** `Sources/ReuseMLX/*` (new files preferred; minimal edits to existing), `Tests/ReuseMLXTests/*` (new files), `tools/coreml/*` (export + install script), `Package.swift` only if a new framework link is needed (CoreML).

**Produces (contract):**
```swift
/// Loads the compiled Core ML parts of one RE-USE checkpoint.
public final class ReuseANEAccelerator: @unchecked Sendable {
    public init(packageDirectory: URL) throws   // throws if missing/incompatible
    /// Compute units Core ML reports for the loaded models, for display.
    public var computeUnitsDescription: String { get }
}
public enum ReuseBackend: String, Sendable, Codable, CaseIterable { case mlx, aneHybrid }
// MambaEnhancer gains:
//   init(modelWeightsURL:mode:accelerator: ReuseANEAccelerator? = nil)
//   makeSession(... backend: ReuseBackend)  — .aneHybrid uses the accelerator for 48 kHz full windows, MLX otherwise
//   nonisolated var lastBackendUsed: ReuseBackend (or equivalent per-session report)
```
Exact Swift shapes may differ if the existing API makes another shape clearly better — report the final signatures.

**Acceptance:** parity test (env-gated like the others) on ≥ 60 s 48 kHz speech: rel. RMSE vs MLX ≤ 1 %; fallback test (missing package → MLX, non-48 kHz → MLX); a benchmark command or test printing RTF for both backends and peak memory; report whether Core ML actually placed the parts on the ANE (e.g. `MLComputePlan` / Instruments evidence), and both gate results.

## Task B — main agent (after A, only if the quality gate passes)
- Settings: backend picker (自动 / MLX / ANE 混合), one-time 30 s speed test per chip + OS stored in UserDefaults, "re-test" button, shows backend actually used and compute units.
- Processing card shows "MLX GPU" / "ANE + GPU" per run; `ReuseStage.settingsKey` includes the backend.
- README: measured numbers on the dev machine.
