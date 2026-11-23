#!/usr/bin/env bash
# Builds dist/MicPolish.app: release build, MLX metallib, LAME dylib, ad-hoc signature.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

tools/build_lame.sh
swift build -c release --product MicPolishApp
bin="$(swift build -c release --show-bin-path)"

final="dist/MicPolish.app"
# Assemble in a staging bundle and swap at the end, so a running copy of the
# app never sees a half-built bundle.
app="dist/MicPolish.app.staging"
rm -rf "$app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources" "$app/Contents/Frameworks"

cp "$bin/MicPolishApp" "$app/Contents/MacOS/MicPolish"
cp Config/Info.plist "$app/Contents/Info.plist"
# App icon: every size iconutil expects, from the 1024 px master.
iconset="$(mktemp -d)/AppIcon.iconset"
mkdir -p "$iconset"
for size in 16 32 128 256 512; do
  sips -z $size $size Config/AppIcon.png --out "$iconset/icon_${size}x${size}.png" >/dev/null
  sips -z $((size * 2)) $((size * 2)) Config/AppIcon.png --out "$iconset/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil -c icns "$iconset" -o "$app/Contents/Resources/AppIcon.icns"
rm -rf "$(dirname "$iconset")"
# SwiftPM resource bundles (model licenses, localizations) sit beside the binary.
for bundle in "$bin"/*.bundle; do
  [[ -d "$bundle" ]] && cp -R "$bundle" "$app/Contents/Resources/"
done

# Localizations: zh-Hans is the development language; English comes from the catalog.
xcrun xcstringstool compile Sources/MicPolishApp/Localizable.xcstrings --output-directory "$app/Contents/Resources"
mkdir -p "$app/Contents/Resources/zh-Hans.lproj"
# Info.plist strings (the microphone prompt) for languages other than zh-Hans.
cp -R Config/*.lproj "$app/Contents/Resources/"

# MLX resolves mlx.metallib next to the binary that links it.
tools/build_mlx_metallib.sh --output "$app/Contents/MacOS/mlx.metallib"

cp vendor/lame/lib/libmp3lame.0.dylib "$app/Contents/Frameworks/"
exe="$app/Contents/MacOS/MicPolish"
# Replace the development rpath with the bundle's Frameworks directory.
while read -r rpath; do
  install_name_tool -delete_rpath "$rpath" "$exe"
done < <(otool -l "$exe" | awk '/LC_RPATH/{getline; getline; print $2}' | grep "$root" || true)
install_name_tool -add_rpath "@executable_path/../Frameworks" "$exe"

mkdir -p "$app/Contents/Resources/licenses"
cp LICENSE NOTICE "$app/Contents/Resources/"
cp licenses/* "$app/Contents/Resources/licenses/"
# Statically linked third-party code.
checkouts=".build/checkouts"
cp "$checkouts/mlx-swift/LICENSE" "$app/Contents/Resources/licenses/mlx-swift-MIT.txt"
cp "$checkouts/mlx-swift/Source/Cmlx/mlx/LICENSE" "$app/Contents/Resources/licenses/mlx-MIT.txt"
cp "$checkouts/mlx-swift/Source/Cmlx/mlx-c/LICENSE" "$app/Contents/Resources/licenses/mlx-c-MIT.txt"
cp "$checkouts/mlx-swift/Source/Cmlx/fmt/LICENSE" "$app/Contents/Resources/licenses/fmt-MIT.txt"
cp "$checkouts/mlx-swift/Source/Cmlx/json/LICENSE.MIT" "$app/Contents/Resources/licenses/nlohmann-json-MIT.txt"
cp "$checkouts/mlx-swift/Source/Cmlx/metal-cpp/LICENSE.txt" "$app/Contents/Resources/licenses/metal-cpp-Apache-2.0.txt"
cp "$checkouts/swift-numerics/LICENSE.txt" "$app/Contents/Resources/licenses/swift-numerics-Apache-2.0.txt"

# LGPL: ship the exact LAME source next to the app for release uploads.
mkdir -p dist/third-party-sources
cp vendor/lame/src/lame-3.100.tar.gz dist/third-party-sources/
cp tools/build_lame.sh dist/third-party-sources/build_lame.sh

codesign --force --sign - "$app/Contents/Frameworks/libmp3lame.0.dylib"
# mlx.metallib must sit beside the executable for MLX to find it; sign it as a
# standalone file so the bundle seal covers it.
codesign --force --sign - "$app/Contents/MacOS/mlx.metallib"
codesign --force --sign - --entitlements Config/MicPolish.entitlements "$app"
codesign --verify --deep --strict "$app"

# A running copy resolves resources (mlx.metallib, localizations) by bundle path,
# so never replace the bundle under it.
# Match relative launches (`dist/MicPolish.app/...`) as well as absolute ones.
if pgrep -f "$final/Contents/MacOS/MicPolish" >/dev/null; then
  echo "MicPolish is running from $final; quit it and rerun this script. New build left in $app" >&2
  exit 1
fi
rm -rf "$final"
mv "$app" "$final"
echo "Built $final"
