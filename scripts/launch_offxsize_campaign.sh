#!/usr/bin/env bash
# Launcher: 20260909_offxsize-dog_mixed off-policy cross-size campaign.
#
# Single entry point for the USER:
#   1. writes gpu_policy.$HOST.json from campaign.offxsize_plan.GPU_POLICY,
#   2. starts ONE tmux session `offxsize` with workers + driver watch.
#
# Staging (adopt Angela's 160m teachers/pools onto cold-1, gates) is a separate
# pre-flight that must run first:
#   uv run python -m scripts.stage_offxsize_assets
#
# The driver PAUSES at number generation. When `watch` logs "waiting on: numbers",
# run the operator CLI, then leave `watch` running -- it resumes on its own:
#   uv run python -m scripts.offxsize_numbers plan
#
# Usage: bash scripts/launch_offxsize_campaign.sh
# Optional: OFFXSIZE_KEEP_POLICY=1 to keep an existing gpu_policy file.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
HOST=$(hostname)
SESSION=offxsize
ROOT=$(uv run python -c "import yaml;from pathlib import Path;from subliminal_learning.config import resolve_queue_root;print(resolve_queue_root(yaml.safe_load(Path('configs/paths/paths_local.yaml').read_text())))")

mkdir -p "$ROOT" "$ROOT/locks" "$ROOT/logs"

POLICY="$ROOT/gpu_policy.$HOST.json"
if [[ "${OFFXSIZE_KEEP_POLICY:-0}" == "1" && -f "$POLICY" ]]; then
  echo "OFFXSIZE_KEEP_POLICY=1: keeping existing $POLICY: $(cat "$POLICY")"
else
  if [[ -f "$POLICY" ]]; then
    cp -a "$POLICY" "$POLICY.bak_pre_launch_$(date -u +%Y%m%dT%H%M%SZ)"
  fi
  uv run python -c "import json;from subliminal_learning.campaign import offxsize_plan as p;print(json.dumps(p.gpu_policy_json(), indent=2))" > "$POLICY"
  echo "wrote $POLICY: $(cat "$POLICY")"
fi
WORKERS=$(uv run python -c "from subliminal_learning.campaign import offxsize_plan as p;print(2 * 8 * int(p.GPU_POLICY['cap_per_gpu']))")
QUEUES=$(uv run python -c "from subliminal_learning.campaign import offxsize_plan as p;print(','.join(p.QUEUE_PRIORITIES))")

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists -- not re-launching."
  echo "Attach with: tmux attach -t $SESSION"
  exit 0
fi

if pgrep -af "scripts.jobq work --root $ROOT" >/dev/null 2>&1; then
  echo "ERROR: a jobq worker is already running on $ROOT."
  echo "Stop it (or use a dedicated queue_root) before launching offxsize."
  pgrep -af "scripts.jobq work --root $ROOT" || true
  exit 1
fi

tmux new-session -d -s "$SESSION" -n workers -c "$REPO"
tmux send-keys -t "$SESSION:workers" \
  "while true; do uv run python -m scripts.jobq work --root $ROOT --workers $WORKERS --queues $QUEUES --lock-prefix $ROOT/locks/jobq; echo 'pool exited; restarting in 60s (Ctrl-C to stop)'; sleep 60; done" C-m

tmux new-window -t "$SESSION" -n driver -c "$REPO"
# No --auto-reconcile-sha: an intentional pin change is an explicit operator action
# (docs/PROJECT_DESIGN.md section 4), never a silent auto-repin on a poll.
tmux send-keys -t "$SESSION:driver" \
  "uv run python -m scripts.run_offxsize_campaign watch --queue-root $ROOT --interval 300 --auto-requeue 2" C-m

cat <<EOF

Started tmux session '$SESSION':
  workers -> jobq pool ($WORKERS threads, queues=$QUEUES), root=$ROOT
  driver  -> campaign watch loop (5-min polls, --auto-requeue 2)

The driver stops at the number-generation pause and keeps polling. When it logs
"waiting on: numbers", generate the pools, then do nothing else:
  uv run python -m scripts.offxsize_numbers plan
  uv run python -m scripts.offxsize_numbers status

Attach:  tmux attach -t $SESSION      Detach: Ctrl-b d
Status:  uv run python -m scripts.run_offxsize_campaign status --queue-root $ROOT
         uv run python -m scripts.jobq status --root $ROOT
Stop:    uv run python -m scripts.jobq stop --root $ROOT
EOF
