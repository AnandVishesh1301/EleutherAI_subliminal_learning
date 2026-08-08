#!/usr/bin/env bash
# Launcher: 20260804_breadth5_160m five-animal student-HP + cross-init campaign.
#
# Sibling of scripts/launch_student_hp_campaign.sh, driving campaign.breadth_plan
# instead of campaign.plan. Two differences that matter:
#   1. --campaign breadth5 selects breadth_plan.SPEC (run-set, b5-* queues,
#      generated/breadth5 config dir, storytelling eval bundle, reference arm),
#   2. the worker pool is scoped with --queues to this campaign's b5-* queues, so
#      a breadth pool sharing the canonical jobq root can never drain slb-* or
#      optparam-* work belonging to another campaign.
#
# Stage A0 prerequisites (driver gate refuses to submit without them):
#   uv run python -m scripts.check_step0_init_equality --run-set-id 20260804_breadth5_160m
#   uv run python -m scripts.calibrate_gkd_throughput --gpu 2 --run-set-id 20260804_breadth5_160m
#
# Usage: bash scripts/launch_breadth5_campaign.sh [--skip-a0] [--auto-requeue 2]
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
HOST=$(hostname)
SESSION=breadth5
ROOT=$(uv run python -c "import yaml;from pathlib import Path;from subliminal_learning.config import resolve_queue_root;print(resolve_queue_root(yaml.safe_load(Path('configs/paths/paths_local.yaml').read_text())))")
DRIVER_FLAGS="${*:-}"

mkdir -p "$ROOT" "$ROOT/locks" "$ROOT/logs"

# Every window tees to a file as well as tmux scrollback: scrollback is bounded and
# dies with the pane, and an agent auditing a pause/crash needs the driver's gate
# messages and the pool's WAIT/STEAL lines after the fact.
WLOG="$ROOT/logs/breadth5_workers.$HOST.log"
DLOG="$ROOT/logs/breadth5_driver.$HOST.log"
NLOG="$ROOT/logs/breadth5_notify.$HOST.log"

# Policy + worker count come from campaign.breadth_plan (SSOT). An existing policy is
# never clobbered: it is hot-reloaded on every GPU acquire, so hand-edits must survive a
# re-launch. Delete the file to regenerate it.
POLICY="$ROOT/gpu_policy.$HOST.json"
if [ -f "$POLICY" ]; then
  echo "keeping existing $POLICY: $(cat "$POLICY")"
else
  uv run python -c "import json;from subliminal_learning.campaign import breadth_plan;print(json.dumps(breadth_plan.GPU_POLICY))" > "$POLICY"
  echo "wrote $POLICY: $(cat "$POLICY")"
fi
WORKERS=$(uv run python -c "from subliminal_learning.campaign import breadth_plan;print(breadth_plan.WORKER_THREADS)")
QUEUES=$(uv run python -c "from subliminal_learning.campaign import breadth_plan;print(','.join(breadth_plan.QUEUE_PRIORITIES))")

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists -- not re-launching."
  echo "Attach with: tmux attach -t $SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" -n workers -c "$REPO"
tmux send-keys -t "$SESSION:workers" \
  "while true; do uv run python -m scripts.jobq work --root $ROOT --workers $WORKERS --queues $QUEUES --lock-prefix $ROOT/locks/jobq; echo 'pool exited (stop sentinel or crash); restarting in 60s (Ctrl-C to stop)'; sleep 60; done 2>&1 | tee -a $WLOG" C-m

# The driver deliberately has no restart loop: a CampaignGateError must stay dead so
# the halt is investigated, not looped over. The rc line records how it ended.
tmux new-window -t "$SESSION" -n driver -c "$REPO"
tmux send-keys -t "$SESSION:driver" \
  "{ uv run python -m scripts.run_student_hp_campaign watch --campaign breadth5 --queue-root $ROOT --interval 300 --auto-reconcile-sha $DRIVER_FLAGS; echo \"driver exited rc=\$?\"; } 2>&1 | tee -a $DLOG" C-m

# Window 3: Discord notify sidecar (isolated tools/campaign_notify uv project).
# Profile `breadth5` + `.env.breadth5` → #hp_gkd_cross_full_campaign_pipeline
# (bot username hp_gkd_full_campaign); bi-hourly digests via DISCORD_*_SECONDS=7200.
NOTIFY_TOOL="$REPO/tools/campaign_notify"
NOTIFY_ENV="$NOTIFY_TOOL/.env.breadth5"
if [ -f "$NOTIFY_ENV" ] || [ -f "$NOTIFY_TOOL/.env" ] || [ -n "${DISCORD_WEBHOOK_URL:-}" ]; then
  tmux new-window -t "$SESSION" -n notify -c "$REPO"
  NOTIFY_CMD="uv sync --project $NOTIFY_TOOL && uv run --project $NOTIFY_TOOL python -m campaign_notify watch --profile breadth5 --queue-root $ROOT"
  if [ -f "$NOTIFY_ENV" ]; then
    NOTIFY_CMD="$NOTIFY_CMD --env-file $NOTIFY_ENV"
  fi
  tmux send-keys -t "$SESSION:notify" "$NOTIFY_CMD 2>&1 | tee -a $NLOG" C-m
  NOTIFY_LINE="  notify  -> Discord campaign_notify watch --profile breadth5 (bi-hourly ETA/GPU)"
else
  NOTIFY_LINE="  notify  -> SKIPPED (no tools/campaign_notify/.env.breadth5 or DISCORD_WEBHOOK_URL)"
fi

cat <<EOF

Started tmux session '$SESSION':
  workers -> jobq pool ($WORKERS threads, restart loop), root=$ROOT
             scoped to queues: $QUEUES
  driver  -> breadth5 campaign watch loop (5-min polls)
$NOTIFY_LINE

Attach:  tmux attach -t $SESSION      Detach: Ctrl-b d
Logs:    workers $WLOG
         driver  $DLOG
         notify  $NLOG
         per-cell training logs: $ROOT/<queue>/logs/
Status:  uv run python -m scripts.campaign_control status --campaign breadth5
         uv run python -m scripts.run_student_hp_campaign status --campaign breadth5 --queue-root $ROOT
         uv run python -m scripts.jobq status --root $ROOT
Pause:   uv run python -m scripts.campaign_control pause  --campaign breadth5 --reason "..."
Resume:  uv run python -m scripts.campaign_control resume --campaign breadth5 --reason "..."
         (in-flight cells finish before the pool exits; resume needs this session's
          workers window alive -- its restart loop re-enters jobq work by itself)
EOF
