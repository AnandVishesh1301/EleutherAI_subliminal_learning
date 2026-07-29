#!/usr/bin/env bash
# Launcher: 20260727_slbreadth_160m student-HP + cross-init campaign.
#
# Single-entry-point for the USER :
#   1. writes gpu_policy.$HOST.json verbatim from campaign.plan.GPU_POLICY (the SSOT —
#      never restate its numbers here; a stale copy is a direct route to an OOM policy),
#   2. starts ONE tmux session `slbreadth` with three windows:
#        workers -> the jobq worker pool (restart loop, so the pool survives the
#                   gaps between stages while the driver hasn't submitted the
#                   next queue yet),
#        driver  -> the campaign driver in --watch mode (submits A1/A2 now,
#                   then A3 / Stage B automatically as gates pass),
#        notify  -> isolated Discord sidecar (tools/campaign_notify) when .env is set.
#
# Stage A0 prerequisites (driver gate refuses to submit without them):
#   uv run python -m scripts.check_step0_init_equality
#   uv run python -m scripts.calibrate_gkd_throughput --gpu 2
# If calibration recommends a different cap, edit $ROOT/gpu_policy.$HOST.json — it is
# hot-reloaded on every acquire and this script will not overwrite an existing one. Keep
# cap_per_gpu * free_mem_mib under the card's 48 GB (see the measurement note in
# campaign/plan.py before raising the cap).
#
# Usage: bash scripts/launch_student_hp_campaign.sh [--skip-a0] [--include-stage-c]
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
HOST=$(hostname)
SESSION=slbreadth
ROOT="$REPO/run_logs/queue"
DRIVER_FLAGS="${*:-}"

mkdir -p "$ROOT" "$ROOT/locks" "$ROOT/logs"

# Policy + worker count come from campaign.plan (SSOT). An existing policy is never
# clobbered: it is hot-reloaded on every GPU acquire, so hand-edits (e.g. dropping a GPU)
# must survive a re-launch. Delete the file to regenerate it.
POLICY="$ROOT/gpu_policy.$HOST.json"
if [ -f "$POLICY" ]; then
  echo "keeping existing $POLICY: $(cat "$POLICY")"
else
  uv run python -c "import json;from subliminal_learning.campaign import plan;print(json.dumps(plan.GPU_POLICY))" > "$POLICY"
  echo "wrote $POLICY: $(cat "$POLICY")"
fi
WORKERS=$(uv run python -c "from subliminal_learning.campaign import plan;print(plan.WORKER_THREADS)")

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists -- not re-launching."
  echo "Attach with: tmux attach -t $SESSION"
  exit 0
fi

# Window 1: worker pool in a restart loop (one thread per GPU slot-unit).
# The pool exits when every visible queue is complete; the loop brings it back
# so newly submitted stage queues get picked up within 60 s.
tmux new-session -d -s "$SESSION" -n workers -c "$REPO"
tmux send-keys -t "$SESSION:workers" \
  "while true; do uv run python -m scripts.jobq work --root $ROOT --workers $WORKERS --lock-prefix $ROOT/locks/jobq; echo 'pool exited; restarting in 60s (Ctrl-C to stop)'; sleep 60; done" C-m

# Window 2: campaign driver (submits stages as their gates pass; dies loudly on
# any gate failure -- reattach, fix, and rerun `advance`).
tmux new-window -t "$SESSION" -n driver -c "$REPO"
tmux send-keys -t "$SESSION:driver" \
  "uv run python -m scripts.run_student_hp_campaign watch --queue-root $ROOT --interval 300 $DRIVER_FLAGS" C-m

# Window 3: Discord notify sidecar (isolated tools/campaign_notify uv project).
# Additive only — never imports training code; fails soft if webhook/.env missing.
NOTIFY_TOOL="$REPO/tools/campaign_notify"
if [ -f "$NOTIFY_TOOL/.env" ] || [ -n "${DISCORD_WEBHOOK_URL:-}" ]; then
  tmux new-window -t "$SESSION" -n notify -c "$REPO"
  tmux send-keys -t "$SESSION:notify" \
    "uv sync --project $NOTIFY_TOOL && uv run --project $NOTIFY_TOOL python -m campaign_notify watch" C-m
  NOTIFY_LINE="  notify  -> Discord campaign_notify watch (ETA/GPU/probes)"
else
  NOTIFY_LINE="  notify  -> SKIPPED (no tools/campaign_notify/.env or DISCORD_WEBHOOK_URL)"
fi

cat <<EOF

Started tmux session '$SESSION':
  workers -> jobq pool ($WORKERS threads, restart loop), root=$ROOT
  driver  -> campaign watch loop (5-min polls)
$NOTIFY_LINE

Attach:  tmux attach -t $SESSION      Detach: Ctrl-b d
Status:  uv run python -m scripts.run_student_hp_campaign status --queue-root $ROOT
         uv run python -m scripts.jobq status --root $ROOT
         uv run --project tools/campaign_notify python -m campaign_notify status
Stop:    uv run python -m scripts.jobq stop --root $ROOT   (graceful pool stop)
EOF
