#!/usr/bin/env bash
# Launcher: rev1 (reversal rank 1) on THIS host, one tmux session `rev1`. On a worker host
# (rev1_plan.WORKER_HOSTS: soar, cluster-3, cluster-4, goncalo, cluster-5) window `workers`
# = one jobq pool on the reversal queue root (it claims only queues pinned to this host);
# with --driver (home node soar only) window `driver` = the restarting watch loop. Run from
# <main clone>/.code_pins/<sha>.
#
# Refuses: a checkout that is not the pin of <sha>; an unpushed SHA or one failing the fleet
# guard; SL_JOBQ_LOCK_PREFIX set to anything but the plan's prefix for this host; a missing
# paths YAML (scripts.run_rev1_campaign stage writes it); an existing `rev1` session, a live
# pool on the reversal queue root, or a stop sentinel. Writes only this host's gpu_policy
# file in the reversal queue root. Touches no other queue root.
#
# With --ext: the rank-1 extension rev1x (session rev1x, queue root reversal/queue_x,
# rev1_plan.EXT_* hosts); without it, rev1 exactly as before.
#
# Usage: bash scripts/launch_rev1_campaign.sh <sha> [--driver] [--ext]
set -euo pipefail

SHA="${1:-}"
DRIVER=""
EXT=""
for arg in "${@:2}"; do
  case "$arg" in
    --driver) DRIVER="--driver" ;;
    --ext) EXT="--ext" ;;
    *) echo "usage: bash scripts/launch_rev1_campaign.sh <sha> [--driver] [--ext]"; exit 2 ;;
  esac
done
if [[ -z "$SHA" ]]; then
  echo "usage: bash scripts/launch_rev1_campaign.sh <sha> [--driver] [--ext]"
  exit 2
fi
EXTPY=$([[ -n "$EXT" ]] && echo True || echo False)

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
  && git show "$FULL_SHA:scripts/launch_rev1_campaign.sh" | grep -q SL_JOBQ_LOCK_PREFIX); then
  echo "ERROR: $FULL_SHA fails the fleet-guard check."
  exit 1
fi

uv sync --frozen --quiet

PLAN="from subliminal_learning.campaign import rev1_plan as p"
read -r LOCK SESSION WORKERS PATHS HOME_NODE ROOT < <(uv run python -c "$PLAN;import socket;h=socket.gethostname();x=$EXTPY;print(p.LOCK_PREFIX.get(h, '-'), p.EXT_CAMPAIGN if x else p.TMUX_SESSION, p.workers_per_node(h, x), p.PATHS_CONFIG_PATH, p.HOME_NODE, p.QUEUE_ROOT_X if x else p.QUEUE_ROOT)")
if [[ "$WORKERS" == "0" && "$DRIVER" != "--driver" ]]; then
  echo "ERROR: $HOST runs no rev1 workers."
  exit 1
fi
if [[ "$DRIVER" == "--driver" && "$HOST" != "$HOME_NODE" ]]; then
  echo "ERROR: the rev1 driver runs only on its home node $HOME_NODE (this is $HOST)."
  exit 1
fi
if [[ "$WORKERS" != "0" ]]; then
  if [[ "$LOCK" == "-" ]]; then
    echo "ERROR: no lock prefix for $HOST."
    exit 1
  fi
  if [[ -n "${SL_JOBQ_LOCK_PREFIX:-}" && "$SL_JOBQ_LOCK_PREFIX" != "$LOCK" ]]; then
    echo "ERROR: SL_JOBQ_LOCK_PREFIX='$SL_JOBQ_LOCK_PREFIX' but rev1 on $HOST uses '$LOCK'."
    exit 1
  fi
  export SL_JOBQ_LOCK_PREFIX="$LOCK"
fi
if [[ ! -f "$PATHS" ]]; then
  echo "ERROR: $PATHS missing; run scripts.run_rev1_campaign stage on $HOME_NODE first."
  exit 1
fi
mkdir -p "$ROOT/logs"

if tmux has-session -t "=$SESSION" 2>/dev/null; then
  echo "ERROR: tmux session '$SESSION' already exists -- attach with: tmux attach -t $SESSION"
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
  uv run python -m scripts.run_rev1_campaign write-policy $EXT
  QUEUES_CMD="$PLAN;import socket;print(','.join(p.host_queues(socket.gethostname(), $EXTPY)))"
  tmux new-session -d -s "$SESSION" -n workers -c "$REPO"
  tmux send-keys -t "$SESSION:workers" \
    "export SL_JOBQ_LOCK_PREFIX=$LOCK; Q=\$(uv run python -c \"$QUEUES_CMD\"); while true; do uv run python -m scripts.jobq work --root $ROOT --workers $WORKERS --queues \"\$Q\" --lock-prefix $LOCK; echo 'pool exited; restarting in 60s (Ctrl-C to stop)'; sleep 60; done" C-m
fi

if [[ "$DRIVER" == "--driver" ]]; then
  if [[ "$WORKERS" != "0" ]]; then
    tmux new-window -t "$SESSION" -n driver -c "$REPO"
  else
    tmux new-session -d -s "$SESSION" -n driver -c "$REPO"
  fi
  tmux send-keys -t "$SESSION:driver" \
    "backoff=60; while true; do uv run python -m scripts.run_rev1_campaign watch --interval 120 --auto-requeue 2 $EXT; ec=\$?; if [ \$ec -eq 0 ]; then echo 'driver finished'; break; fi; echo \"driver exited \$ec; restarting in \${backoff}s\"; sleep \$backoff; backoff=\$((backoff < 600 ? backoff * 2 : 600)); done" C-m
fi

cat <<MSG

Started tmux session '$SESSION' on $HOST (pin $PIN_SHA):
  $( [[ "$WORKERS" != "0" ]] && echo "workers -> jobq pool ($WORKERS threads, lock prefix $LOCK), root=$ROOT" || echo "no workers on this host" )
  $( [[ "$DRIVER" == "--driver" ]] && echo "driver  -> restarting watch loop (2-min polls, 2 attempts)" || echo "no driver on this host (home node: $HOME_NODE)" )

Attach:  tmux attach -t $SESSION      Detach: Ctrl-b d
MSG
