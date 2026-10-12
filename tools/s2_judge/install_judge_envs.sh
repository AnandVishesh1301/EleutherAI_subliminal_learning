#!/usr/bin/env bash
# Build the two S2 judge venvs from a code pin's locked uv projects, outside the pin
# (so every pin of the campaign shares them), and pre-fetch the harmony o200k vocab
# that gpt-oss rendering needs offline (openai_harmony downloads it once and verifies
# its sha256; TIKTOKEN_RS_CACHE_DIR makes the cache persistent).
#
#   stack N: <venv_root>/judge_v2      <- <pin>/tools/judge_v2      (vLLM 0.30.0+cu129)
#   stack F: <venv_root>/rollout_judge <- <pin>/tools/rollout_judge (vLLM 0.8.5)
#
# Idempotent (`uv sync --frozen` converges on the lock). Run on soar, the judge lane.
# Usage: bash tools/s2_judge/install_judge_envs.sh <pin_root> <venv_root> <tiktoken_dir>
set -euo pipefail

PIN="${1:?usage: install_judge_envs.sh <pin_root> <venv_root> <tiktoken_dir>}"
VENVS="${2:?usage: install_judge_envs.sh <pin_root> <venv_root> <tiktoken_dir>}"
TIK="${3:?usage: install_judge_envs.sh <pin_root> <venv_root> <tiktoken_dir>}"

mkdir -p "$VENVS" "$TIK"
UV_PROJECT_ENVIRONMENT="$VENVS/judge_v2" uv sync --frozen --no-dev --project "$PIN/tools/judge_v2"
UV_PROJECT_ENVIRONMENT="$VENVS/rollout_judge" uv sync --frozen --no-dev --project "$PIN/tools/rollout_judge"
TIKTOKEN_RS_CACHE_DIR="$TIK" "$VENVS/judge_v2/bin/python" -c "
from openai_harmony import HarmonyEncodingName, load_harmony_encoding
enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
print('harmony vocab ok:', enc.name)
"
"$VENVS/judge_v2/bin/python" -c "import vllm, torch; print('stack N', vllm.__version__, torch.__version__)"
"$VENVS/rollout_judge/bin/python" -c "import vllm, torch; print('stack F', vllm.__version__, torch.__version__)"
ls -1 "$TIK"
