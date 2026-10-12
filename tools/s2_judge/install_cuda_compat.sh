#!/usr/bin/env bash
# User-space CUDA 12.9 forward-compatibility libraries for the stack-N judge venv
# (tools/judge_v2: vllm 0.30.0+cu129, torch 2.13.0+cu129) on driver 535.
#
# NVIDIA, "CUDA Compatibility", Forward Compatibility, Table 3 (CUDA Application
# Compatibility Support Matrix): cuda-compat-12-9 is "C" (compatible) with driver
# 535+; forward compatibility applies only to NVIDIA Data Center GPUs (A40 is one).
# https://docs.nvidia.com/deploy/cuda-compatibility/latest/forward-compatibility.html
# The package "only provides the libraries"; the consumer sets LD_LIBRARY_PATH.
#
# No root, no apt: the pinned .deb from NVIDIA's ubuntu2204 repo is sha256-checked
# against the repo's Packages index value below, unpacked with `dpkg-deb -x`, and its
# compat/ libraries copied into <dest>. Idempotent: an existing <dest> with a matching
# INSTALLED.json is left untouched; any other existing <dest> is refused.
#
# Usage: bash tools/s2_judge/install_cuda_compat.sh <dest>
set -euo pipefail

DEST="${1:?usage: install_cuda_compat.sh <dest>}"
VERSION="575.57.08-0ubuntu1"
DEB="cuda-compat-12-9_${VERSION}_amd64.deb"
URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/${DEB}"
SHA256="536cf41fb15fd7d8b39231648379b48c41718d4bf588b44481443ed35f1f6ce9"

if [[ -f "$DEST/INSTALLED.json" ]]; then
  if grep -q "\"sha256\": \"$SHA256\"" "$DEST/INSTALLED.json"; then
    echo "cuda-compat already installed at $DEST"
    exit 0
  fi
  echo "ERROR: $DEST holds a different cuda-compat install" >&2
  exit 1
fi
if [[ -e "$DEST" && -n "$(ls -A "$DEST" 2>/dev/null)" ]]; then
  echo "ERROR: $DEST exists and is not an install of this script" >&2
  exit 1
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
curl -fsSL "$URL" -o "$WORK/$DEB"
echo "$SHA256  $WORK/$DEB" | sha256sum -c -
dpkg-deb -x "$WORK/$DEB" "$WORK/x"
SRC="$(find "$WORK/x" -type d -path '*/cuda-12.9/compat' | head -1)"
if [[ -z "$SRC" ]]; then
  echo "ERROR: no cuda-12.9/compat dir in $DEB" >&2
  exit 1
fi
mkdir -p "$DEST"
cp -a "$SRC"/. "$DEST"/
cat > "$DEST/INSTALLED.json" <<JSON
{"package": "cuda-compat-12-9", "version": "$VERSION", "url": "$URL", "sha256": "$SHA256", "installed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
JSON
ls -1 "$DEST"
