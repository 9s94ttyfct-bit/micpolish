#!/usr/bin/env bash
# Runs the Swift test suite with the runtime artifacts it needs in place:
# MLX looks for mlx.metallib next to the test executable.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

swift build --build-tests >/dev/null
bin_dir="$(swift build --show-bin-path)"
if [[ ! -s "$bin_dir/mlx.metallib" ]]; then
  tools/build_mlx_metallib.sh --output "$bin_dir/mlx.metallib"
fi

exec swift test --skip-build "$@"
