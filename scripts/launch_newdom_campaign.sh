#!/usr/bin/env bash
# Launcher: newdom (trees + colors, hard + soft) on THIS host, in tmux session `newdom`.
# Every host (soar, cluster-3, cluster-4) gets window `workers` = the jobq pool loop on the
# campaign's own queue root, draining every unpinned queue of both channels
# (scripts/jobq_pool_loop.sh: idles under a stop sentinel, exits on the campaign's
# CAMPAIGN_DONE). With --driver (home node soar only) windows `hard-driver` and
# `soft-driver` = the restarting watch loops of the two channel drivers. Run from
# <main clone>/.code_pins/<sha>. No wall deadline.
#
# Refuses: a checkout that is not the pin of <sha>; an unpushed SHA or one failing the
# fleet guard; a host outside newdom_plan.HOSTS, or --driver off the home node;
# SL_JOBQ_LOCK_PREFIX set to anything but the plan's prefix; a missing channel paths YAML
# (the channel staging scripts write them); an existing window, a live pool on the root, or
# a stop sentinel. Writes only this host's gpu_policy file (create or equal) in the root.
#
# Usage: bash scripts/launch_newdom_campaign.sh <sha> [--driver|--drivers-only]
# --drivers-only (soar) reopens only the two driver windows, for a re-pin restart.
set -euo pipefail

SHA="${1:-}"
DRIVER="${2:-}"
if [[ -z "$SHA" || ( -n "$DRIVER" && "$DRIVER" != "--driver" && "$DRIVER" != "--drivers-only" ) ]]; then
  echo "usage: bash scripts/launch_newdom_campaign.sh <sha> [--driver|--drivers-only]"
  exit 2
fi
DRIVERS_ONLY=0
if [[ "$DRIVER" == "--drivers-only" ]]; then
  DRIVERS_ONLY=1
  DRIVER="--driver"
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
  && git show "$FULL_SHA:scripts/launch_newdom_campaign.sh" | grep -q SL_JOBQ_LOCK_PREFIX); then
  echo "ERROR: $FULL_SHA fails the fleet-guard check."
  exit 1
fi
if ! (git grep -q off_topic_nocolor "$FULL_SHA" -- src/subliminal_learning/evaluation/prompt_sets.py \
  && git grep -q "def candidate_probs_per_prompt" "$FULL_SHA" -- src/subliminal_learning/evaluation/preference_prob_io.py \
  && git grep -q "def strict_pool_screen_forms" "$FULL_SHA" -- src/subliminal_learning/gkd/filtering.py \
  && git grep -q "domain: str = ANIMALS" "$FULL_SHA" -- src/subliminal_learning/gkd/config.py); then
  echo "ERROR: $FULL_SHA lacks the newdom domain plumbing."
  exit 1
fi

uv sync --frozen --quiet

PLAN="from subliminal_learning.campaign import newdom_plan as p"
read -r LOCK SESSION WORKERS ROOT DONE HOME_NODE IN_HOSTS HARD_PATHS SOFT_PATHS < <(uv run python -c "$PLAN;import socket;n=socket.gethostname();print(p.LOCK_PREFIX.get(n, '-'), p.TMUX_SESSION, p.workers_per_node(), p.QUEUE_ROOT, p.CAMPAIGN_DONE_PATH, p.HOME_NODE, int(n in p.HOSTS), p.paths_config_path(p.HARD), p.paths_config_path(p.SOFT))")
if [[ "$IN_HOSTS" != "1" ]]; then
  echo "ERROR: $HOST is not a newdom host."
  exit 1
fi
if [[ "$DRIVER" == "--driver" && "$HOST" != "$HOME_NODE" ]]; then
  echo "ERROR: the newdom drivers run only on the home node $HOME_NODE (this is $HOST)."
  exit 1
fi
if [[ "$LOCK" == "-" ]]; then
  echo "ERROR: no lock prefix for $HOST."
  exit 1
fi
if [[ -n "${SL_JOBQ_LOCK_PREFIX:-}" && "$SL_JOBQ_LOCK_PREFIX" != "$LOCK" ]]; then
  echo "ERROR: SL_JOBQ_LOCK_PREFIX='$SL_JOBQ_LOCK_PREFIX' but newdom on $HOST uses '$LOCK'."
  exit 1
fi
export SL_JOBQ_LOCK_PREFIX="$LOCK"
for f in "$HARD_PATHS" "$SOFT_PATHS"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: $f missing; run scripts.stage_newdom_{hard,soft}_assets on $HOME_NODE first."
    exit 1
  fi
done
for d in "$ROOT" "$ROOT/logs"; do
  [[ -d "$d" ]] || mkdir -p "$d"
done

WIN_WORKERS="workers"

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
[[ "$DRIVERS_ONLY" -eq 0 ]] && WINDOWS+=("$WIN_WORKERS")
[[ "$DRIVER" == "--driver" ]] && WINDOWS+=(hard-driver soft-driver)
for w in "${WINDOWS[@]}"; do
  if has_window "$w"; then
    echo "ERROR: window '$SESSION:$w' already exists -- attach with: tmux attach -t $SESSION"
    exit 1
  fi
done
if [[ "$DRIVERS_ONLY" -eq 0 && -e "$ROOT/stop.$HOST" ]]; then
  echo "ERROR: stop sentinel $ROOT/stop.$HOST exists."
  exit 1
fi
if [[ "$DRIVERS_ONLY" -eq 0 ]] && pgrep -af "scripts.jobq work --root $ROOT " >/dev/null 2>&1; then
  echo "ERROR: a jobq worker is already running on $ROOT on this host."
  exit 1
fi

if [[ "$DRIVERS_ONLY" -eq 0 ]]; then
uv run python - <<'PY'
import json
import socket

from subliminal_learning.campaign import newdom_plan as p
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

if [[ "$DRIVER" == "--driver" ]]; then
  for ch in hard soft; do
    open_window "$ch-driver"
    tmux send-keys -t "=$SESSION:=$ch-driver" \
      "backoff=60; while true; do uv run python -m scripts.run_newdom_${ch}_campaign watch --interval 300 --auto-requeue 3 --code-sha $PIN_SHA; ec=\$?; if [ \$ec -eq 0 ]; then echo 'driver finished'; break; fi; if [ \$ec -eq 2 ]; then echo 'GATE FAILED -- not restarting'; break; fi; echo \"driver exited \$ec; restarting in \${backoff}s\"; sleep \$backoff; backoff=\$((backoff < 600 ? backoff * 2 : 600)); done" C-m
  done
fi

cat <<MSG

Started on $HOST (pin $PIN_SHA), tmux session '$SESSION':
  $WIN_WORKERS -> jobq pool loop ($WORKERS threads, lock prefix $LOCK), root=$ROOT, exits on $DONE
  $( [[ "$DRIVER" == "--driver" ]] && echo "hard-driver, soft-driver -> restarting watch loops (5-min polls, --auto-requeue 3)" || echo "no drivers on this host (home node: $HOME_NODE)" )

Attach:  tmux attach -t $SESSION      Detach: Ctrl-b d
Status:  cd $REPO && uv run python -m scripts.run_newdom_{hard,soft}_campaign status   (on $HOME_NODE)
MSG
