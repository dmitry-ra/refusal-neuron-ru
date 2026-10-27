#!/bin/bash
# Usage (compute host, lab dir): scripts/convert_gguf.sh LLAMA_CPP_DIR OUTTYPE
# Converts the pinned HF snapshot to GGUF with llama.cpp's own converter, then hashes it.
set -eu
llama="$1"; outtype="${2:-bf16}"
cd "$(dirname "$0")/.."
source scripts/env.sh
rev="$(uv run --frozen python -c 'from snsteer import config; print(config.MODEL_REVISION)')"
snap="$HF_HOME/hub/models--Qwen--Qwen3.5-4B/snapshots/$rev"
mkdir -p gguf
out="gguf/Qwen3.5-4B-${rev:0:7}-${outtype^^}.gguf"
echo "llama.cpp commit: $(git -C "$llama" rev-parse HEAD)"
uv run --frozen --group convert python "$llama/convert_hf_to_gguf.py" "$snap" --outtype "$outtype" --outfile "$out"
sha256sum "$out"
ls -l "$out"
