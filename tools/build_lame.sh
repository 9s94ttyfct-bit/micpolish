#!/usr/bin/env bash
# Builds LAME 3.100 (LGPL-2.0) as an arm64 shared library into vendor/lame.
# MicPolish links libmp3lame dynamically so the library stays replaceable.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
prefix="$root/vendor/lame"
version="3.100"
sha256="ddfe36cab873794038ae2c1210557ad34857a4b6bdc515785d1da9e175b1da1e"

if [[ -f "$prefix/lib/libmp3lame.0.dylib" && -f "$prefix/include/lame/lame.h" ]]; then
  exit 0
fi

work="$(mktemp -d "${TMPDIR:-/tmp}/micpolish-lame.XXXXXX")"
trap 'rm -rf "$work"' EXIT

tarball="$work/lame-$version.tar.gz"
curl -sfL -o "$tarball" "https://downloads.sourceforge.net/project/lame/lame/$version/lame-$version.tar.gz"
echo "$sha256  $tarball" | shasum -a 256 -c - >/dev/null

tar -xzf "$tarball" -C "$work"
cd "$work/lame-$version"
# 3.100 exports a symbol that no longer exists; Apple's linker rejects it.
sed -i '' '/lame_init_old/d' include/libmp3lame.sym

CFLAGS="-O2 -arch arm64 -mmacosx-version-min=14.0" \
LDFLAGS="-arch arm64 -mmacosx-version-min=14.0" \
  ./configure --prefix="$prefix" --disable-static --enable-shared \
    --disable-frontend --enable-nasm=no >/dev/null
make -j"$(sysctl -n hw.ncpu)" >/dev/null
make install >/dev/null

install_name_tool -id "@rpath/libmp3lame.0.dylib" "$prefix/lib/libmp3lame.0.dylib"
echo "Installed LAME $version into $prefix"
