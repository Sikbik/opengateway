#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ICON_DIR="${ROOT_DIR}/src-tauri/icons"
SOURCE_ICON="${ICON_DIR}/icon.png"
OUTPUT_ICON="${ICON_DIR}/icon.icns"

if [[ ! -f "${SOURCE_ICON}" ]]; then
  echo "error: missing source icon at ${SOURCE_ICON}" >&2
  exit 1
fi

if ! command -v sips >/dev/null 2>&1 || ! command -v iconutil >/dev/null 2>&1; then
  echo "error: sips and iconutil are required to prepare macOS icons" >&2
  exit 1
fi

TMP_DIR="$(mktemp -d)"
ICONSET_DIR="${TMP_DIR}/factory-control.iconset"
mkdir -p "${ICONSET_DIR}"

for size in 16 32 128 256 512; do
  sips -z "${size}" "${size}" "${SOURCE_ICON}" --out "${ICONSET_DIR}/icon_${size}x${size}.png" >/dev/null
done

for size in 16 32 128 256 512; do
  double_size=$((size * 2))
  sips -z "${double_size}" "${double_size}" "${SOURCE_ICON}" --out "${ICONSET_DIR}/icon_${size}x${size}@2x.png" >/dev/null
done

iconutil -c icns "${ICONSET_DIR}" -o "${OUTPUT_ICON}"
rm -rf "${TMP_DIR}"
