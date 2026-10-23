#!/usr/bin/env bash
# Launcher: `placebo2` (trait-free placebo corpus v2, docgen only) on THIS host, one
# tmux session `placebo2`:
#   workers (soar, goncalo, cluster-3, cluster-4) -> scripts/jobq_pool_loop.sh on the
#            campaign queue root (every unpinned run queue; exits on CAMPAIGN_DONE)
#   driver  (home node, soar)                      -> the restarting watch loop
# Run from <main clone>/.code_pins/<sha>.
#
# Refuses: a checkout that is not the pin of <sha>; an unpushed SHA or one failing the
# fleet guard; a host with no role in that mode; SL_JOBQ_LOCK_PREFIX set to anything
# but the plan's prefix for this host; an existing `placebo2` session, a live pool on
# the root, or a stop sentinel. Writes only this host's gpu_policy file.
#
# Usage: bash scripts/launch_placebo2_campaign.sh <sha> workers|driver
set -euo pipefail

SHA="${1:-}"
MODE="${2:-}"
if [[ -z "$SHA" || ( "$MODE" != "workers" && "$MODE" != "driver" ) ]]; then
  echo "usage: bash scripts/launch_placebo2_campaign.sh <sha> workers|driver"
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
  && git show "$FULL_SHA:scripts/launch_placebo2_campaign.sh" | grep -q SL_JOBQ_LOCK_PREFIX); then
  echo "ERROR: $FULL_SHA fails the fleet-guard check."
  exit 1
fi

uv sync --frozen --quiet

PLAN="from subliminal_learning.campaign import placebo2_plan as p"
read -r LOCK SESSION WORKERS HOME_NODE ROOT DONE IS_HOST < <(uv run python -c "$PLAN;import socket;h=socket.gethostname();print(p.LOCK_PREFIX.get(h, '-'), p.TMUX_SESSION, p.WORKERS_PER_NODE, p.HOME_NODE, p.QUEUE_ROOT, p.DONE_MARKER, int(h in p.HOSTS))")
if [[ "$IS_HOST" != "1" || "$LOCK" == "-" ]]; then
  echo "ERROR: $HOST is not a placebo2 host."
  exit 1
fi
if [[ "$MODE" == "driver" && "$HOST" != "$HOME_NODE" ]]; then
  echo "ERROR: the placebo2 driver runs only on its home node $HOME_NODE (this is $HOST)."
  exit 1
fi
if [[ -n "${SL_JOBQ_LOCK_PREFIX:-}" && "$SL_JOBQ_LOCK_PREFIX" != "$LOCK" ]]; then
  echo "ERROR: SL_JOBQ_LOCK_PREFIX='$SL_JOBQ_LOCK_PREFIX' but placebo2 on $HOST uses '$LOCK'."
  exit 1
fi
export SL_JOBQ_LOCK_PREFIX="$LOCK"
if [[ ! -x "$REPO/../../tools/sdf_docgen/.venv/bin/python" ]]; then
  echo "ERROR: no docgen venv at $REPO/../../tools/sdf_docgen/.venv."
  exit 1
fi

if [[ "$MODE" == "workers" ]]; then
  WIN=workers
  if pgrep -af "scripts.jobq work --root $ROOT" >/dev/null 2>&1; then
    echo "ERROR: a jobq worker is already running on $ROOT on this host."
    exit 1
  fi
  if [[ -e "$ROOT/stop.$HOST" ]]; then
    echo "ERROR: stop sentinel $ROOT/stop.$HOST exists."
    exit 1
  fi
  mkdir -p "$ROOT/logs"
  uv run python -m scripts.run_placebo2_campaign policy
  CMD="export SL_JOBQ_LOCK_PREFIX=$LOCK; bash scripts/jobq_pool_loop.sh $ROOT '' $WORKERS $LOCK $DONE"
else
  WIN=driver
  CMD="backoff=60; while true; do uv run python -m scripts.run_placebo2_campaign watch --code-sha $PIN_SHA --interval 120; ec=\$?; if [ \$ec -eq 0 ]; then echo 'driver finished'; break; fi; echo \"driver exited \$ec; restarting in \${backoff}s\"; sleep \$backoff; backoff=\$((backoff < 600 ? backoff * 2 : 600)); done"
fi

if tmux has-session -t "=$SESSION" 2>/dev/null; then
  if tmux list-windows -t "=$SESSION" -F '#W' | grep -qx "$WIN"; then
    echo "ERROR: tmux window '$SESSION:$WIN' already exists."
    exit 1
  fi
  tmux new-window -t "=$SESSION" -n "$WIN" -c "$REPO"
else
  tmux new-session -d -s "$SESSION" -n "$WIN" -c "$REPO"
fi
tmux send-keys -t "$SESSION:$WIN" "$CMD" C-m

cat <<MSG

Started tmux window '$SESSION:$WIN' on $HOST (pin $PIN_SHA), root=$ROOT
Attach:  tmux attach -t $SESSION      Detach: Ctrl-b d
Status:  cd $REPO && uv run python -m scripts.run_placebo2_campaign status   (on $HOME_NODE)
MSG
