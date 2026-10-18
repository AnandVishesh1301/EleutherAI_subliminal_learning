#!/usr/bin/env bash
# Launcher: newdom2 (Part II soft follow-ups A / B / C) on soar only, in tmux session
# `newdom2`: window `workers` = the jobq pool loop on the Part II queue root
# (scripts/jobq_pool_loop.sh: idles under a stop sentinel, exits on newdom2's
# CAMPAIGN_DONE) and window `driver` = the restarting watch loop of the Part II driver.
# Run from <main clone>/.code_pins/<sha>. No wall deadline.
#
# Refuses: a checkout that is not the pin of <sha>; an unpushed SHA or one failing the
# fleet guard or lacking the strict_sub screen; a host other than the home node;
# SL_JOBQ_LOCK_PREFIX set to anything but the plan's prefix; a missing Part II paths YAML
# (scripts.stage_newdom2_assets writes it); an existing window, a live pool on the root,
# or a stop sentinel. Writes only the home node's gpu_policy file (create or equal).
#
# Usage: bash scripts/launch_newdom2_campaign.sh <sha> [--driver-only]
# --driver-only reopens only the driver window, for a re-pin restart.
set -euo pipefail

SHA="${1:-}"
MODE="${2:-}"
if [[ -z "$SHA" || ( -n "$MODE" && "$MODE" != "--driver-only" ) ]]; then
  echo "usage: bash scripts/launch_newdom2_campaign.sh <sha> [--driver-only]"
  exit 2
fi
DRIVER_ONLY=0
[[ "$MODE" == "--driver-only" ]] && DRIVER_ONLY=1

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
if ! (git grep -q SHARED_LOCK_OWNER "$FULL_SHA" -- src/subliminal_learning/jobqueue \
  && git show "$FULL_SHA:scripts/launch_newdom2_campaign.sh" | grep -q SL_JOBQ_LOCK_PREFIX); then
  echo "ERROR: $FULL_SHA fails the fleet-guard check."
  exit 1
fi
if ! (git grep -q '"strict_sub"' "$FULL_SHA" -- src/subliminal_learning/gkd/layout.py \
  && git grep -q "substring_forms" "$FULL_SHA" -- src/subliminal_learning/gkd/filtering.py \
  && git grep -q "substring=" "$FULL_SHA" -- pipelines/distill_student_gkd_pipeline.py \
  && git grep -q "class Lane" "$FULL_SHA" -- src/subliminal_learning/campaign/newdom_soft.py); then
  echo "ERROR: $FULL_SHA lacks the strict_sub screen or the newdom lanes."
  exit 1
fi

uv sync --frozen --quiet

PLAN="from subliminal_learning.campaign import newdom2_plan as p"
read -r LOCK SESSION WORKERS ROOT DONE HOME_NODE PATHS < <(uv run python -c "$PLAN;import socket;n=socket.gethostname();print(p.LOCK_PREFIX.get(n, '-'), p.TMUX_SESSION, p.workers_per_node(), p.QUEUE_ROOT, p.CAMPAIGN_DONE_PATH, p.HOME_NODE, p.PATHS_CONFIG_PATH)")
if [[ "$HOST" != "$HOME_NODE" ]]; then
  echo "ERROR: newdom2 runs only on $HOME_NODE (this is $HOST)."
  exit 1
fi
if [[ "$LOCK" == "-" ]]; then
  echo "ERROR: no lock prefix for $HOST."
  exit 1
fi
if [[ -n "${SL_JOBQ_LOCK_PREFIX:-}" && "$SL_JOBQ_LOCK_PREFIX" != "$LOCK" ]]; then
  echo "ERROR: SL_JOBQ_LOCK_PREFIX='$SL_JOBQ_LOCK_PREFIX' but newdom2 on $HOST uses '$LOCK'."
  exit 1
fi
export SL_JOBQ_LOCK_PREFIX="$LOCK"
if [[ ! -f "$PATHS" ]]; then
  echo "ERROR: $PATHS missing; run scripts.stage_newdom2_assets first."
  exit 1
fi
for d in "$ROOT" "$ROOT/logs"; do
  [[ -d "$d" ]] || mkdir -p "$d"
done

WIN_WORKERS="workers"
WIN_DRIVER="driver"

has_window() {
  tmux list-windows -t "=$SESSION" -F '#W' 2>/dev/null | grep -qx "$1"
}
open_window() {
  if tmux has-session -t "=$SESSION" 2>/dev/null; then
    tmux new-window -t "=$SESSION:" -n "$1" -c "$REPO"
  else
    tmux new-session -d -s "$SESSION" -n "$1" -c "$REPO"
  fi
}

WINDOWS=("$WIN_DRIVER")
[[ "$DRIVER_ONLY" -eq 0 ]] && WINDOWS=("$WIN_WORKERS" "$WIN_DRIVER")
for w in "${WINDOWS[@]}"; do
  if has_window "$w"; then
    echo "ERROR: window '$SESSION:$w' already exists -- attach with: tmux attach -t $SESSION"
    exit 1
  fi
done
if [[ "$DRIVER_ONLY" -eq 0 && -e "$ROOT/stop.$HOST" ]]; then
  echo "ERROR: stop sentinel $ROOT/stop.$HOST exists."
  exit 1
fi
if [[ "$DRIVER_ONLY" -eq 0 ]] && pgrep -af "scripts.jobq work --root $ROOT " >/dev/null 2>&1; then
  echo "ERROR: a jobq worker is already running on $ROOT on this host."
  exit 1
fi

if [[ "$DRIVER_ONLY" -eq 0 ]]; then
uv run python - <<'PY'
import json
import socket

from subliminal_learning.campaign import newdom2_plan as p
from subliminal_learning.utils.io import atomic_write_json

path = p.QUEUE_ROOT / f"gpu_policy.{socket.gethostname()}.json"
doc = p.gpu_policy_json()
if path.is_file():
    if json.loads(path.read_text()) != doc:
        raise SystemExit(f"ERROR: {path} exists with a different policy")
else:
    atomic_write_json(path, doc)
print("policy", path)
PY

open_window "$WIN_WORKERS"
tmux send-keys -t "=$SESSION:=$WIN_WORKERS" \
  "export SL_JOBQ_LOCK_PREFIX=$LOCK; bash scripts/jobq_pool_loop.sh $ROOT '' $WORKERS $LOCK $DONE" C-m
fi

open_window "$WIN_DRIVER"
tmux send-keys -t "=$SESSION:=$WIN_DRIVER" \
  "backoff=60; while true; do uv run python -m scripts.run_newdom2_campaign watch --interval 300 --auto-requeue 3 --code-sha $PIN_SHA; ec=\$?; if [ \$ec -eq 0 ]; then echo 'driver finished'; break; fi; if [ \$ec -eq 2 ]; then echo 'GATE FAILED -- not restarting'; break; fi; echo \"driver exited \$ec; restarting in \${backoff}s\"; sleep \$backoff; backoff=\$((backoff < 600 ? backoff * 2 : 600)); done" C-m

cat <<MSG

Started on $HOST (pin $PIN_SHA), tmux session '$SESSION':
  $( [[ "$DRIVER_ONLY" -eq 0 ]] && echo "$WIN_WORKERS -> jobq pool loop ($WORKERS threads, lock prefix $LOCK), root=$ROOT, exits on $DONE" || echo "workers window untouched" )
  $WIN_DRIVER -> restarting watch loop (5-min polls, --auto-requeue 3)

Attach:  tmux attach -t $SESSION      Detach: Ctrl-b d
Status:  cd $REPO && uv run python -m scripts.run_newdom2_campaign status
MSG
