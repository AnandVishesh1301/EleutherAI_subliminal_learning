#!/usr/bin/env bash
# Launcher: tmplckpt (statement teachers through Pythia-160M checkpoints, hard + soft) on
# THIS host, in tmux session `tmplckpt`. Every host (cluster-3, cluster-4) gets window
# `workers` = the jobq pool loop on the campaign's own queue root, draining every
# unpinned queue (scripts/jobq_pool_loop.sh: idles under a stop sentinel, exits on
# CAMPAIGN_DONE). With --driver (home node cluster-3 only) window `driver` = the
# restarting watch loop of the one driver. Run from <main clone>/.code_pins/<sha>. No
# wall deadline.
#
# Refuses: a checkout that is not the pin of <sha>; an unpushed SHA or one failing the
# fleet guard; a pin whose campaign / sweep modules do not import; a host outside tmplckpt_plan.HOSTS, or --driver off the home node;
# SL_JOBQ_LOCK_PREFIX set to anything but the plan's prefix; a missing paths YAML or
# staging report; an existing window, a live pool on the root, or a stop sentinel.
# Writes only this host's gpu_policy file (create or equal) in the root.
#
# Usage: bash scripts/launch_tmplckpt_campaign.sh <sha> [--driver|--driver-only]
# --driver-only (home node) reopens only the driver window, for a re-pin restart. The
# driver loop passes --code-sha <pin> --auto-reconcile-sha, so a driver relaunched from a
# new pin advances the cards it owns; reconcile the other cards first (re-pin runbook).
set -euo pipefail

SHA="${1:-}"
MODE="${2:-}"
if [[ -z "$SHA" || ( -n "$MODE" && "$MODE" != "--driver" && "$MODE" != "--driver-only" ) ]]; then
  echo "usage: bash scripts/launch_tmplckpt_campaign.sh <sha> [--driver|--driver-only]"
  exit 2
fi
WORKERS_WINDOW=1
DRIVER=0
[[ "$MODE" == "--driver" ]] && DRIVER=1
if [[ "$MODE" == "--driver-only" ]]; then
  DRIVER=1
  WORKERS_WINDOW=0
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
if ! (git grep -q SHARED_LOCK_OWNER "$FULL_SHA" -- src/subliminal_learning/jobqueue \
  && git show "$FULL_SHA:scripts/launch_tmplckpt_campaign.sh" | grep -q SL_JOBQ_LOCK_PREFIX); then
  echo "ERROR: $FULL_SHA fails the fleet-guard check."
  exit 1
fi

uv sync --frozen --quiet
if ! uv run python -c "import scripts.run_tmplckpt_campaign, scripts.run_tmplckpt_hard_cell, scripts.sweeps.sweep_distill_student_gkd, scripts.sweeps.sweep_sdf_teacher, scripts.sweeps.sweep_evaluate_gkd_student, scripts.sweeps.sweep_evaluate_sdf_teacher"; then
  echo "ERROR: import smoke failed in the pin $REPO."
  exit 1
fi

PLAN="from subliminal_learning.campaign import tmplckpt_plan as p"
read -r LOCK SESSION WORKERS ROOT DONE HOME_NODE IN_HOSTS PATHS REPORT < <(uv run python -c "$PLAN;import socket;n=socket.gethostname();print(p.LOCK_PREFIX.get(n, '-'), p.TMUX_SESSION, p.workers_per_node(), p.QUEUE_ROOT, p.CAMPAIGN_DONE_PATH, p.HOME_NODE, int(n in p.HOSTS), p.PATHS_CONFIG_PATH, p.STAGING_REPORT_PATH)")
if [[ "$IN_HOSTS" != "1" ]]; then
  echo "ERROR: $HOST is not a tmplckpt host."
  exit 1
fi
if [[ "$DRIVER" -eq 1 && "$HOST" != "$HOME_NODE" ]]; then
  echo "ERROR: the tmplckpt driver runs only on the home node $HOME_NODE (this is $HOST)."
  exit 1
fi
if [[ "$LOCK" == "-" ]]; then
  echo "ERROR: no lock prefix for $HOST."
  exit 1
fi
if [[ -n "${SL_JOBQ_LOCK_PREFIX:-}" && "$SL_JOBQ_LOCK_PREFIX" != "$LOCK" ]]; then
  echo "ERROR: SL_JOBQ_LOCK_PREFIX='$SL_JOBQ_LOCK_PREFIX' but tmplckpt on $HOST uses '$LOCK'."
  exit 1
fi
export SL_JOBQ_LOCK_PREFIX="$LOCK"
for f in "$PATHS" "$REPORT"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: $f missing; run scripts.stage_tmplckpt_assets from the pin on $HOME_NODE first."
    exit 1
  fi
done
[[ -d "$ROOT" ]] || mkdir -p "$ROOT"

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

WINDOWS=()
[[ "$WORKERS_WINDOW" -eq 1 ]] && WINDOWS+=(workers)
[[ "$DRIVER" -eq 1 ]] && WINDOWS+=(driver)
for w in "${WINDOWS[@]}"; do
  if has_window "$w"; then
    echo "ERROR: window '$SESSION:$w' already exists -- attach with: tmux attach -t $SESSION"
    exit 1
  fi
done
if [[ "$WORKERS_WINDOW" -eq 1 && -e "$ROOT/stop.$HOST" ]]; then
  echo "ERROR: stop sentinel $ROOT/stop.$HOST exists."
  exit 1
fi
if [[ "$WORKERS_WINDOW" -eq 1 ]] && pgrep -af "scripts.jobq work --root $ROOT " >/dev/null 2>&1; then
  echo "ERROR: a jobq worker is already running on $ROOT on this host."
  exit 1
fi

if [[ "$WORKERS_WINDOW" -eq 1 ]]; then
uv run python - <<'PY'
import json
import socket

from subliminal_learning.campaign import tmplckpt_plan as p
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

open_window workers
tmux send-keys -t "=$SESSION:=workers" \
  "export SL_JOBQ_LOCK_PREFIX=$LOCK; bash scripts/jobq_pool_loop.sh $ROOT '' $WORKERS $LOCK $DONE" C-m
fi

if [[ "$DRIVER" -eq 1 ]]; then
  open_window driver
  tmux send-keys -t "=$SESSION:=driver" \
    "backoff=60; while true; do uv run python -m scripts.run_tmplckpt_campaign watch --interval 300 --auto-requeue 3 --code-sha $PIN_SHA --auto-reconcile-sha; ec=\$?; if [ \$ec -eq 0 ]; then echo 'driver finished'; break; fi; if [ \$ec -eq 2 ]; then echo 'GATE FAILED -- not restarting'; break; fi; echo \"driver exited \$ec; restarting in \${backoff}s\"; sleep \$backoff; backoff=\$((backoff < 600 ? backoff * 2 : 600)); done" C-m
fi

cat <<MSG

Started on $HOST (pin $PIN_SHA), tmux session '$SESSION':
  $( [[ "$WORKERS_WINDOW" -eq 1 ]] && echo "workers -> jobq pool loop ($WORKERS threads, lock prefix $LOCK), root=$ROOT, exits on $DONE" || echo "no workers window opened" )
  $( [[ "$DRIVER" -eq 1 ]] && echo "driver -> restarting watch loop (5-min polls, --auto-requeue 3)" || echo "no driver on this host (home node: $HOME_NODE)" )

Attach:  tmux attach -t $SESSION      Detach: Ctrl-b d
Status:  cd $REPO && uv run python -m scripts.run_tmplckpt_campaign status   (on $HOME_NODE)
MSG
