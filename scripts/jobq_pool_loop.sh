#!/usr/bin/env bash
# Restart loop for one node's jobq pool (the campaign tmux "workers" window, run from the
# pin). Idles while <root>/stop.<host> exists, so a fleet pause or yield never needs a
# Ctrl-C, and exits for good once the campaign's done marker exists, so a finished
# campaign leaves no loop respawning pools.
#   bash scripts/jobq_pool_loop.sh ROOT QUEUES WORKERS LOCK_PREFIX [DONE_MARKER]
# QUEUES is a comma list ("" = every queue in ROOT pinned to this host or unpinned).
set -u
ROOT=$1
QUEUES=$2
WORKERS=$3
LOCK=$4
DONE=${5:-}
HOST=$(hostname)
while true; do
  if [ -n "$DONE" ] && [ -e "$DONE" ]; then
    echo "done marker $DONE present; loop exiting"
    exit 0
  fi
  if [ -e "$ROOT/stop.$HOST" ]; then
    sleep 60
    continue
  fi
  if [ -n "$QUEUES" ]; then
    uv run python -m scripts.jobq work --root "$ROOT" --workers "$WORKERS" \
      --queues "$QUEUES" --lock-prefix "$LOCK"
  else
    uv run python -m scripts.jobq work --root "$ROOT" --workers "$WORKERS" \
      --lock-prefix "$LOCK"
  fi
  echo "pool exited; restarting in 60s (Ctrl-C to stop)"
  sleep 60
done
