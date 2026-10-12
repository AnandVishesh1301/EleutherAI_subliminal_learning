#!/usr/bin/env bash
# Launcher: one channel of the S2 core campaign (s2core) on THIS host, tmux session
# `s2core-<channel>`. On a worker host of the channel (s2core_plan.CHANNEL_HOSTS: soft =
# soar / cluster-3 / cluster-4, hard = cluster-5 / goncalo) window `workers` = one jobq
# pool on the channel's queue root (it claims only queues pinned to this host); with
# --driver (home node soar only) window `driver` = the restarting watch loop. soar is a
# soft worker host and the home node of both drivers, so `soft --driver` there starts
# both windows and `hard --driver` only the driver. Run from <main clone>/.code_pins/<sha>.
#
# Refuses: a checkout that is not the pin of <sha>; an unpushed SHA or one failing the
# fleet guard; a host that is neither a worker host of the channel nor (with --driver)
# the home node; SL_JOBQ_LOCK_PREFIX set to anything but the plan's prefix for this host;
# a missing paths YAML (scripts.stage_s2core_assets soft writes it); an existing session,
# a live pool on the channel's queue root, or a stop sentinel. Writes only this host's
# gpu_policy file in the channel's queue root (create or equal). Touches no other root.
#
# Usage: bash scripts/launch_s2core_campaign.sh <sha> <soft|hard> [--driver]
set -euo pipefail

SHA="${1:-}"
CHANNEL="${2:-}"
DRIVER="${3:-}"
if [[ -z "$SHA" || ( "$CHANNEL" != "soft" && "$CHANNEL" != "hard" ) || ( -n "$DRIVER" && "$DRIVER" != "--driver" ) ]]; then
  echo "usage: bash scripts/launch_s2core_campaign.sh <sha> <soft|hard> [--driver]"
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
  && git show "$FULL_SHA:scripts/launch_s2core_campaign.sh" | grep -q SL_JOBQ_LOCK_PREFIX); then
  echo "ERROR: $FULL_SHA fails the fleet-guard check."
  exit 1
fi

uv sync --frozen --quiet

PLAN="from subliminal_learning.campaign import s2core_plan as p"
read -r LOCK SESSION WORKERS PATHS HOME_NODE ROOT < <(uv run python -c "$PLAN;import socket;h=socket.gethostname();c='$CHANNEL';print(p.LOCK_PREFIX.get(h, '-'), p.TMUX_SESSION[c], p.workers_per_node(h) if h in p.CHANNEL_HOSTS[c] else 0, p.paths_config_path(c), p.HOME_NODE, p.queue_root(c))")
if [[ "$WORKERS" == "0" && "$DRIVER" != "--driver" ]]; then
  echo "ERROR: $HOST runs no s2core-$CHANNEL workers."
  exit 1
fi
if [[ "$DRIVER" == "--driver" && "$HOST" != "$HOME_NODE" ]]; then
  echo "ERROR: the s2core drivers run only on their home node $HOME_NODE (this is $HOST)."
  exit 1
fi
if [[ "$WORKERS" != "0" ]]; then
  if [[ "$LOCK" == "-" ]]; then
    echo "ERROR: no lock prefix for $HOST."
    exit 1
  fi
  if [[ -n "${SL_JOBQ_LOCK_PREFIX:-}" && "$SL_JOBQ_LOCK_PREFIX" != "$LOCK" ]]; then
    echo "ERROR: SL_JOBQ_LOCK_PREFIX='$SL_JOBQ_LOCK_PREFIX' but s2core on $HOST uses '$LOCK'."
    exit 1
  fi
fi
if [[ ! -f "$PATHS" ]]; then
  echo "ERROR: $PATHS missing; run scripts.stage_s2core_assets soft on $HOME_NODE first."
  exit 1
fi
for d in "$ROOT" "$ROOT/logs"; do
  [[ -d "$d" ]] || mkdir -p "$d"
done

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
  uv run python -m scripts.run_s2core_campaign write-policy --host "$HOST"
  QUEUES_CMD="from subliminal_learning.campaign import s2core_jobs as j;import socket;print(','.join(j.pool_queues(socket.gethostname())))"
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
    "backoff=60; while true; do uv run python -m scripts.run_s2core_campaign watch --channel $CHANNEL --interval 300 --auto-requeue 3 --code-sha $PIN_SHA; ec=\$?; if [ \$ec -eq 0 ]; then echo 'driver finished'; break; fi; if [ \$ec -eq 2 ]; then echo 'GATE FAILED -- not restarting'; break; fi; echo \"driver exited \$ec; restarting in \${backoff}s\"; sleep \$backoff; backoff=\$((backoff < 600 ? backoff * 2 : 600)); done" C-m
fi

cat <<MSG

Started tmux session '$SESSION' on $HOST (pin $PIN_SHA):
  $( [[ "$WORKERS" != "0" ]] && echo "workers -> jobq pool ($WORKERS threads, lock prefix $LOCK), root=$ROOT" || echo "no workers on this host" )
  $( [[ "$DRIVER" == "--driver" ]] && echo "driver  -> restarting watch loop (5-min polls, --auto-requeue 3)" || echo "no driver on this host (home node: $HOME_NODE)" )

Attach:  tmux attach -t $SESSION      Detach: Ctrl-b d
Status:  cd $REPO && uv run python -m scripts.run_s2core_campaign status --channel $CHANNEL   (on $HOME_NODE)
MSG
