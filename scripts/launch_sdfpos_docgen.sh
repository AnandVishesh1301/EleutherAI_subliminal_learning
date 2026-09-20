#!/usr/bin/env bash
# Run one sdfpos docgen run (tools/sdf_docgen, position_cli cluster) in a
# restart loop. Works from the main checkout or a .code_pins/<sha> worktree:
# code comes from this tree, the interpreter from the main clone's tool venv.
#
#   scripts/launch_sdfpos_docgen.sh configs/sdfpos/voice.yaml 0,1
#
# Ends on a clean exit (targets met or cap spent) or when <run_dir>/STOP exists.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <config.yaml> <gpus, comma-separated>" >&2
  exit 2
fi
CONFIG="$(realpath "$1")"
GPUS="$2"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
MAIN_CLONE="$(dirname "$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir)")"
PY="${SL_SDF_DOCGEN_PY:-$MAIN_CLONE/tools/sdf_docgen/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  echo "no interpreter at $PY (run 'uv sync --frozen' in $MAIN_CLONE/tools/sdf_docgen)" >&2
  exit 1
fi

export PYTHONPATH="$ROOT/tools/sdf_docgen/src"
SL_CODE_PIN_SHA="$(git -C "$ROOT" rev-parse HEAD)"
export SL_CODE_PIN_SHA
export OMP_NUM_THREADS=1
cd "$ROOT"

MODULE_FILE="$("$PY" -c 'import sdf_docgen; print(sdf_docgen.__file__)')"
echo "sdf_docgen: $MODULE_FILE  sha: $SL_CODE_PIN_SHA  node: $(hostname)"
if [[ "$MODULE_FILE" != "$ROOT"/* ]]; then
  echo "sdf_docgen was imported from outside $ROOT" >&2
  exit 1
fi

RUN_DIR="$("$PY" -m sdf_docgen.position_cli run-dir --config "$CONFIG")"
echo "run_dir: $RUN_DIR  gpus: $GPUS"

MAX_RESTARTS=20
for ((attempt = 0; attempt <= MAX_RESTARTS; attempt++)); do
  if [[ -e "$RUN_DIR/STOP" ]]; then
    echo "STOP present in $RUN_DIR; not starting"
    exit 0
  fi
  if "$PY" -m sdf_docgen.position_cli cluster --config "$CONFIG" --gpus "$GPUS"; then
    echo "cluster finished cleanly"
    exit 0
  fi
  echo "cluster exited non-zero (restart $attempt of $MAX_RESTARTS); sleeping 60 s"
  sleep 60
done
echo "gave up after $MAX_RESTARTS restarts" >&2
exit 1
