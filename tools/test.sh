#!/usr/bin/env bash
# Runs the Swift test suite with the runtime artifacts it needs in place:
# MLX looks for mlx.metallib next to the test executable.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

tools/build_lame.sh

bin_dir="$(swift build --show-bin-path)"
# A metallib left inside a bundle from the previous run breaks the build's
# code-signing step, so remove the copies before building and re-add them after.
rm -f "$bin_dir"/*.xctest/Contents/MacOS/mlx.metallib
swift build --build-tests >/dev/null
if [[ ! -s "$bin_dir/mlx.metallib" ]]; then
  tools/build_mlx_metallib.sh --output "$bin_dir/mlx.metallib"
fi
# Test bundles statically link MLX, so MLX resolves the metallib next to each
# bundle's own executable.
for bundle in "$bin_dir"/*.xctest; do
  [[ -d "$bundle/Contents/MacOS" ]] || continue
  cp "$bin_dir/mlx.metallib" "$bundle/Contents/MacOS/mlx.metallib"
done

exec swift test --skip-build "$@"
