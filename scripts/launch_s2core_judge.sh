#!/usr/bin/env bash
# Launcher: the S2 judge lane on soar (s2core_plan.JUDGE_LANE_HOST, GPUs
# JUDGE_LANE_GPUS), one tmux session `s2core-judge` whose window `lane` is a restart
# loop around `s2_judge.runner` (tools/s2_judge/runner.py). The runner scans
# <JUDGE_ROOT>/work for item specs and runs judge workers outside jobq in the
# judge venvs (stack N: vLLM 0.30.0 + user-space cuda-compat-12-9; stack F: vLLM
# 0.8.5). Run from <main clone>/.code_pins/<sha>.
#
# Refuses: a checkout that is not the pin of <sha>; an unpushed SHA; a host other than
# the lane host; an existing `s2core-judge` session or a live runner on the judge
# root; missing judge venvs / cuda-compat / harmony vocab (build them with
# tools/s2_judge/install_cuda_compat.sh and tools/s2_judge/install_judge_envs.sh).
# Writes nothing but the tmux session; the runner writes only under JUDGE_ROOT.
#
# Usage: bash scripts/launch_s2core_judge.sh <sha>
set -euo pipefail

SHA="${1:-}"
if [[ -z "$SHA" ]]; then
  echo "usage: bash scripts/launch_s2core_judge.sh <sha>"
  exit 2
fi

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
HOST=$(hostname)
PIN_SHA=$(git rev-parse HEAD)
FULL_SHA=$(git rev-parse --verify "${SHA}^{commit}")
if [[ "$FULL_SHA" != "$PIN_SHA" || "$(basename "$(dirname "$REPO")")" != ".code_pins" || "$(basename "$REPO")" != "$PIN_SHA" ]]; then
  echo "ERROR: launch from <main clone>/.code_pins/$FULL_SHA (this is $REPO at $PIN_SHA)."
  exit 1
fi
git fetch --quiet origin
if [[ -z "$(git branch -r --contains "$FULL_SHA")" ]]; then
  echo "ERROR: $FULL_SHA is not on any pushed branch."
  exit 1
fi

uv sync --frozen --quiet

PLAN="from subliminal_learning.campaign import s2core_plan as p"
read -r LANE_HOST SESSION ROOT GPUS VENVS COMPAT TIK PFA < <(uv run python -c "$PLAN;print(p.JUDGE_LANE_HOST, p.JUDGE_SESSION, p.JUDGE_ROOT, ','.join(map(str, p.JUDGE_LANE_GPUS)), p.JUDGE_VENV_ROOT, p.CUDA_COMPAT_DIR, p.TIKTOKEN_CACHE_DIR, p.JUDGE_PARSE_FAIL_ALERT)")
PYN="$VENVS/judge_v2/bin/python"
PYF="$VENVS/rollout_judge/bin/python"

if [[ "$HOST" != "$LANE_HOST" ]]; then
  echo "ERROR: the judge lane runs only on $LANE_HOST (this is $HOST)."
  exit 1
fi
for f in "$PYN" "$PYF" "$COMPAT/INSTALLED.json"; do
  if [[ ! -e "$f" ]]; then
    echo "ERROR: $f missing; run tools/s2_judge/install_cuda_compat.sh $COMPAT and"
    echo "       tools/s2_judge/install_judge_envs.sh $REPO $VENVS $TIK first."
    exit 1
  fi
done
if [[ -z "$(ls -A "$TIK" 2>/dev/null)" ]]; then
  echo "ERROR: harmony vocab cache $TIK is empty; run install_judge_envs.sh first."
  exit 1
fi
if tmux has-session -t "=$SESSION" 2>/dev/null; then
  echo "ERROR: tmux session '$SESSION' already exists -- attach with: tmux attach -t $SESSION"
  exit 1
fi
if pgrep -af "s2_judge.runner --work-dir $ROOT/work " >/dev/null 2>&1; then
  echo "ERROR: a judge lane runner is already running on $ROOT."
  exit 1
fi
mkdir -p "$ROOT/work"

tmux new-session -d -s "$SESSION" -n lane -c "$REPO"
tmux send-keys -t "$SESSION:lane" \
  "while true; do PYTHONPATH=$REPO/tools $PYN -m s2_judge.runner --work-dir $ROOT/work --state-dir $ROOT --gpus $GPUS --python-n $PYN --python-f $PYF --tools-dir $REPO/tools --compat-dir $COMPAT --tiktoken-dir $TIK --parse-fail-alert $PFA; echo \"runner exited \$?; restarting in 60s (Ctrl-C to stop)\"; sleep 60; done" C-m

cat <<MSG

Started tmux session '$SESSION' on $HOST (pin $PIN_SHA):
  lane -> s2_judge.runner restart loop on GPUs $GPUS, work dir $ROOT/work
  state: $ROOT/lane_heartbeat.json, $ROOT/lane_state.json, $ROOT/lane_alerts.log

Attach:  tmux attach -t $SESSION      Detach: Ctrl-b d
MSG
