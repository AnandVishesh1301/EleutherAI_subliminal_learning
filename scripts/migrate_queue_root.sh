#!/usr/bin/env bash
# Move this account's jobq state out of a shared/foreign queue root into the
# `queue_root` configured in the paths YAML.
#
# The repo-relative `run_logs/` symlink is created by a bootstrap script shared
# across accounts and has pointed into another user's SSD subdir, putting this
# account's queues, per-job logs, and GPU-policy file in their tree. jobq makes
# that tree world-writable so cross-node workers can share it, so the mix is
# silent until two pools collide on a lock.
#
# This cannot run while a worker pool is live: the pool caches its root path, so
# moving the tree underneath it splits queue state and orphans the log file
# handles of every running job. Run it at a deliberate pause (campaign complete,
# or after `jobq stop` has drained the pool).
#
#   scripts/migrate_queue_root.sh --dry-run
#   scripts/migrate_queue_root.sh
#
# Queues are moved, not copied; shared root-level files (GPU policy, master
# logs) are copied so the queues left behind keep working.
set -euo pipefail

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
HOST=$(hostname)
MY_QUEUES=(slb-a1-hpscreen slb-a2-controlfloor slb-a1-evals slb-a3-doseladder
           slb-a3-evals slb-b-headline slb-b-ext slb-b-evals slb-c slb-c-evals
           catswide-eval-backlog)

DST=$(uv run python -c "import yaml;from pathlib import Path;from subliminal_learning.config import resolve_queue_root;print(resolve_queue_root(yaml.safe_load(Path('configs/paths/paths_local.yaml').read_text())))")
SRC=$(readlink -f run_logs)/queue

run() { if [[ $DRY_RUN == 1 ]]; then echo "  would: $*"; else "$@"; fi; }

echo "source (live) : $SRC"
echo "target (config): $DST"
[[ "$SRC" == "$DST" ]] && { echo "already migrated; nothing to do."; exit 0; }

# 1. Refuse while a pool is live — moving the tree under it splits queue state.
PIDFILE="$SRC/worker.$HOST.pid"
if [[ -f "$PIDFILE" ]]; then
  PID=$(cut -d: -f2 "$PIDFILE")
  if kill -0 "$PID" 2>/dev/null; then
    echo "REFUSING: worker pool pid $PID is alive on $HOST."
    echo "  Stop it first:  uv run jobq stop"
    echo "  Wait for the pane to exit (running jobs finish), then re-run this."
    exit 1
  fi
  echo "stale pid file (pid $PID dead) — ignoring"
fi

# 2. Move each queue, verifying the file count survives the move.
run mkdir -p "$DST/locks" "$DST/logs"
for q in "${MY_QUEUES[@]}"; do
  [[ -d "$SRC/$q" ]] || continue
  if [[ -e "$DST/$q" ]]; then echo "  skip $q (already at target)"; continue; fi
  before=$(find "$SRC/$q" -type f | wc -l)
  echo "  move $q ($before files)"
  if [[ $DRY_RUN == 0 ]]; then
    mv "$SRC/$q" "$DST/$q"
    after=$(find "$DST/$q" -type f | wc -l)
    [[ "$before" == "$after" ]] || { echo "FILE COUNT MISMATCH $q: $before -> $after"; exit 1; }
  fi
done

# 3. Copy (never move) what the queues left behind still need.
run cp -n "$SRC/gpu_policy.$HOST.json" "$DST/gpu_policy.$HOST.json"
for f in "$SRC"/logs/*worker."$HOST".log; do
  [[ -e "$f" ]] && run cp -n "$f" "$DST/logs/"
done

# 4. Repoint the repo symlink so an ad-hoc `jobq --root run_logs/queue` agrees
#    with the paths YAML instead of resolving back into the foreign subdir.
MY_HOME_LOGS=$(dirname "$DST")
if [[ -L run_logs ]]; then
  run rm run_logs
  run ln -s "$MY_HOME_LOGS" run_logs
fi

echo
echo "done. Restart the campaign with:"
echo "  scripts/launch_student_hp_campaign.sh"
