#!/usr/bin/env bash
# The jobq queue folder is now taken from jobq_paths.toml in the working directory, so these scripts need one written for their root before use.
# Launcher: 20260801_optparam_160m optimizer x parameterization campaign.
#
# Single entry point for the USER:
#   1. creates the campaign's own jobq root under vishesh/ (the repo-relative run_logs/
#      symlink points into a teammate's SSD subdir — never launch against it),
#   2. writes gpu_policy.$HOST.json verbatim from campaign.optparam_plan.GPU_POLICY (the
#      SSOT — never restate its numbers here; a stale copy is a direct route to an OOM
#      policy),
#   3. starts ONE tmux session `optparam` with:
#        workers -> the jobq worker pool in a restart loop, so the pool survives the gaps
#                   between phases while the driver has not yet submitted the next queue,
#        driver  -> the campaign driver in watch mode (submits Phase 0 now, then Phase 1 /
#                   Phase 2 automatically as the gates pass). Runs with
#                   --auto-reconcile-sha so a commit landing mid-campaign is recorded on
#                   the run-set card instead of stranding every queued cell on the
#                   staleness guard,
#        notify  -> isolated Discord sidecar (tools/campaign_notify) when .env is set.
#
# Before launching:
#   - Free the node: this campaign is ~256 GPU-hours and owns all 8 cards. Stop any of
#     your own long-running sweeps first (foreign processes are ignored per node policy).
#   - Re-measure the heavy arm and set the policy from it, not from the LoRA-derived
#     default:  uv run python -m scripts.calibrate_gkd_throughput --gpu 2
#     An existing gpu_policy.$HOST.json is never overwritten (it is hot-reloaded on every
#     acquire, so hand-edits must survive a re-launch). Delete it to regenerate.
#   - Disk: the Phase-2 submit needs ~380 GB free on /mnt/ssd-2 (one full-FT 160M leaf is
#     ~620 MB vs ~5 MB for a LoRA adapter). See docs/COLD_ARCHIVE_SSD1.md.
#
# Usage: bash scripts/launch_optparam_campaign.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
HOST=$(hostname)
SESSION=optparam
ROOT=$(uv run python -c "from subliminal_learning.jobq_folder import configured_queue_folder;print(configured_queue_folder())")

mkdir -p "$ROOT" "$ROOT/locks" "$ROOT/logs"

POLICY="$ROOT/gpu_policy.$HOST.json"
if [ -f "$POLICY" ]; then
  echo "keeping existing $POLICY: $(cat "$POLICY")"
else
  uv run python -c "import json;from subliminal_learning.campaign import optparam_plan as p;print(json.dumps(p.GPU_POLICY))" > "$POLICY"
  echo "wrote $POLICY: $(cat "$POLICY")"
fi
WORKERS=$(uv run python -c "from subliminal_learning.campaign import optparam_plan as p;print(p.WORKER_THREADS)")

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists -- not re-launching."
  echo "Attach with: tmux attach -t $SESSION"
  exit 0
fi

# Window 1: worker pool in a restart loop. The pool exits when every visible queue is
# complete; the loop brings it back so a newly submitted phase queue is picked up within
# 60 s instead of waiting for a manual restart.
tmux new-session -d -s "$SESSION" -n workers -c "$REPO"
tmux send-keys -t "$SESSION:workers" \
  "while true; do uv run jobq work --workers $WORKERS; echo 'pool exited; restarting in 60s (Ctrl-C to stop)'; sleep 60; done" C-m

# Window 2: campaign driver. Dies loudly on any gate failure (a too-narrow Phase-0 ladder,
# an unusable loss-match reference, a non-flat control floor) -- reattach, fix, rerun
# `advance`. An arm with no rung inside the loss-match band is not a gate failure: it is
# logged as a warning and carried on its best LR (see docs/PREREG_OPTPARAM.md).
tmux new-window -t "$SESSION" -n driver -c "$REPO"
tmux send-keys -t "$SESSION:driver" \
  "uv run python -m scripts.run_optparam_campaign watch --interval 300 --auto-reconcile-sha" C-m

# Window 3: Discord notify sidecar (isolated tools/campaign_notify uv project).
# Profile `optparam` + `.env.optparam` target #ft_optimizer_parametrization_pipeline
# (bot username ft-optimizer-param). Does not touch the main training package.
NOTIFY_TOOL="$REPO/tools/campaign_notify"
NOTIFY_ENV="$NOTIFY_TOOL/.env.optparam"
if [ -f "$NOTIFY_ENV" ] || [ -f "$NOTIFY_TOOL/.env" ] || [ -n "${DISCORD_WEBHOOK_URL:-}" ]; then
  tmux new-window -t "$SESSION" -n notify -c "$REPO"
  NOTIFY_CMD="uv sync --project $NOTIFY_TOOL && uv run --project $NOTIFY_TOOL python -m campaign_notify watch --profile optparam --queue-root $ROOT"
  if [ -f "$NOTIFY_ENV" ]; then
    NOTIFY_CMD="$NOTIFY_CMD --env-file $NOTIFY_ENV"
  fi
  tmux send-keys -t "$SESSION:notify" "$NOTIFY_CMD" C-m
  NOTIFY_LINE="  notify  -> Discord campaign_notify watch --profile optparam (ETA/GPU/probes)"
else
  NOTIFY_LINE="  notify  -> SKIPPED (no tools/campaign_notify/.env.optparam or DISCORD_WEBHOOK_URL)"
fi

cat <<EOF

Started tmux session '$SESSION':
  workers -> jobq pool ($WORKERS threads, restart loop), root=$ROOT
  driver  -> campaign watch loop (5-min polls)
$NOTIFY_LINE

Attach:  tmux attach -t $SESSION      Detach: Ctrl-b d
Status:  uv run python -m scripts.run_optparam_campaign status
         uv run jobq status
         uv run --project tools/campaign_notify python -m campaign_notify status --profile optparam --queue-root $ROOT
Results: uv run python -m scripts.build_optparam_results
Stop:    uv run jobq stop   (graceful pool stop)
EOF
