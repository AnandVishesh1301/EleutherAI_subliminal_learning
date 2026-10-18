#!/usr/bin/env bash
# Launcher: xopt2 hard channel on THIS host, as one window of the campaign's tmux session
# `xopt2` (created when missing; the soft channel adds its own windows). On a worker host
# (xopt2_hard.HOSTS) window `hard-workers` = the jobq pool loop on the hard queue
# root (scripts/jobq_pool_loop.sh: idles under a stop sentinel, exits on the channel's
# CAMPAIGN_DONE); with --driver (home node soar only) window `hard-driver` = the
# restarting watch loop. Run from <main clone>/.code_pins/<sha>. No wall deadline.
#
# Refuses: a checkout that is not the pin of <sha>; an unpushed SHA or one failing the
# fleet guard; a host that is neither the worker host nor (with --driver) the home node;
# a SHA without the optimizer layer (CUSTOM_OPTIMS) or xopt's `late_evals`; SL_JOBQ_LOCK_PREFIX
# set to anything but the plan's prefix for this host; a missing paths YAML
# (scripts.stage_xopt2_hard_assets writes it); this channel's window already in the
# session, a live pool on the hard root, or a stop sentinel; on a worker host, a paused
# xopt pool that is no longer paused (a live worker on an xopt queue root, or a policy
# file there without a stop sentinel: the pools share the slot-lock prefix). Writes only this host's
# gpu_policy file in the hard queue root. Touches no other queue root.
#
# Usage: bash scripts/launch_xopt2_hard_campaign.sh <sha> [--driver]
set -euo pipefail

SHA="${1:-}"
DRIVER="${2:-}"
if [[ -z "$SHA" || ( -n "$DRIVER" && "$DRIVER" != "--driver" ) ]]; then
  echo "usage: bash scripts/launch_xopt2_hard_campaign.sh <sha> [--driver]"
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
if ! (git grep -q SHARED_LOCK_OWNER "$FULL_SHA" -- src/subliminal_learning/jobqueue \
  && git show "$FULL_SHA:scripts/launch_xopt2_hard_campaign.sh" | grep -q SL_JOBQ_LOCK_PREFIX); then
  echo "ERROR: $FULL_SHA fails the fleet-guard check."
  exit 1
fi
if ! (git grep -q CUSTOM_OPTIMS "$FULL_SHA" -- src/subliminal_learning/training/param_control_io.py \
  && git grep -q "def late_evals" "$FULL_SHA" -- pipelines/xopt_hard_cell_pipeline.py); then
  echo "ERROR: $FULL_SHA lacks the xopt2 optimizer layer (CUSTOM_OPTIMS) or xopt's late_evals."
  exit 1
fi

uv sync --frozen --quiet

PLAN="from subliminal_learning.campaign import xopt2_hard as h, xopt2_plan as p, xopt_plan as xp"
read -r LOCK SESSION WORKERS PATHS HOME_NODE ROOT DONE XOPT_ROOTS < <(uv run python -c "$PLAN;import socket;n=socket.gethostname();print(p.LOCK_PREFIX.get(n, '-'), p.TMUX_SESSION, h.workers_per_node(n), h.PATHS_CONFIG_PATH, p.HOME_NODE, h.QUEUE_ROOT, p.channel_done_path(h.CHANNEL), ','.join(str(r) for r in xp.QUEUE_ROOTS.values()))")
WIN_WORKERS="hard-workers"
WIN_DRIVER="hard-driver"
if [[ "$WORKERS" == "0" && "$DRIVER" != "--driver" ]]; then
  echo "ERROR: $HOST runs no xopt2 hard workers (not an xopt2 worker host)."
  exit 1
fi
if [[ "$DRIVER" == "--driver" && "$HOST" != "$HOME_NODE" ]]; then
  echo "ERROR: the xopt2 hard driver runs only on its home node $HOME_NODE (this is $HOST)."
  exit 1
fi
if [[ "$WORKERS" != "0" ]]; then
  if [[ "$LOCK" == "-" ]]; then
    echo "ERROR: no lock prefix for $HOST."
    exit 1
  fi
  if [[ -n "${SL_JOBQ_LOCK_PREFIX:-}" && "$SL_JOBQ_LOCK_PREFIX" != "$LOCK" ]]; then
    echo "ERROR: SL_JOBQ_LOCK_PREFIX='$SL_JOBQ_LOCK_PREFIX' but xopt2 hard on $HOST uses '$LOCK'."
    exit 1
  fi
  export SL_JOBQ_LOCK_PREFIX="$LOCK"
fi
if [[ ! -f "$PATHS" ]]; then
  echo "ERROR: $PATHS missing; run scripts.stage_xopt2_hard_assets on $HOME_NODE first."
  exit 1
fi
for d in "$ROOT" "$ROOT/logs"; do
  [[ -d "$d" ]] || mkdir -p "$d"
done

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

if [[ "$WORKERS" != "0" ]] && has_window "$WIN_WORKERS"; then
  echo "ERROR: window '$SESSION:$WIN_WORKERS' already exists -- attach with: tmux attach -t $SESSION"
  exit 1
fi
if [[ "$DRIVER" == "--driver" ]] && has_window "$WIN_DRIVER"; then
  echo "ERROR: window '$SESSION:$WIN_DRIVER' already exists -- attach with: tmux attach -t $SESSION"
  exit 1
fi
if [[ -e "$ROOT/stop.$HOST" ]]; then
  echo "ERROR: stop sentinel $ROOT/stop.$HOST exists."
  exit 1
fi

if [[ "$WORKERS" != "0" ]]; then
  if pgrep -af "scripts.jobq work --root $ROOT " >/dev/null 2>&1; then
    echo "ERROR: a jobq worker is already running on $ROOT on this host."
    exit 1
  fi
  IFS=',' read -r -a XOPT_ROOT_LIST <<< "$XOPT_ROOTS"
  for XR in "${XOPT_ROOT_LIST[@]}"; do
    if pgrep -af "scripts.jobq work --root $XR " >/dev/null 2>&1 \
      || [[ -e "$XR/gpu_policy.$HOST.json" && ! -e "$XR/stop.$HOST" ]]; then
      echo "ERROR: the paused xopt pool on $XR must stay paused (stop.$HOST present, no live worker)."
      exit 1
    fi
  done
  uv run python -c "from pipelines.xopt2_hard_campaign_pipeline import write_policy;import socket;print('policy', write_policy(socket.gethostname()))"
  open_window "$WIN_WORKERS"
  tmux send-keys -t "=$SESSION:=$WIN_WORKERS" \
    "export SL_JOBQ_LOCK_PREFIX=$LOCK; bash scripts/jobq_pool_loop.sh $ROOT '' $WORKERS $LOCK $DONE" C-m
fi

if [[ "$DRIVER" == "--driver" ]]; then
  open_window "$WIN_DRIVER"
  tmux send-keys -t "=$SESSION:=$WIN_DRIVER" \
    "backoff=60; while true; do uv run python -m scripts.run_xopt2_hard_campaign watch --interval 300 --auto-requeue 3 --code-sha $PIN_SHA; ec=\$?; if [ \$ec -eq 0 ]; then echo 'driver finished'; break; fi; if [ \$ec -eq 2 ]; then echo 'GATE FAILED -- not restarting'; break; fi; echo \"driver exited \$ec; restarting in \${backoff}s\"; sleep \$backoff; backoff=\$((backoff < 600 ? backoff * 2 : 600)); done" C-m
fi

cat <<MSG

Started on $HOST (pin $PIN_SHA), tmux session '$SESSION':
  $( [[ "$WORKERS" != "0" ]] && echo "$WIN_WORKERS -> jobq pool loop ($WORKERS threads, lock prefix $LOCK), root=$ROOT, exits on $DONE" || echo "no workers on this host" )
  $( [[ "$DRIVER" == "--driver" ]] && echo "$WIN_DRIVER  -> restarting watch loop (5-min polls, --auto-requeue 3)" || echo "no driver on this host (home node: $HOME_NODE)" )

Attach:  tmux attach -t $SESSION      Detach: Ctrl-b d
Status:  cd $REPO && uv run python -m scripts.run_xopt2_hard_campaign status   (on $HOME_NODE)
MSG
