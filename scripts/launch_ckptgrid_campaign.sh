#!/usr/bin/env bash
# Launcher: 20260828_ckptgrid-bear_160m pretraining-checkpoint grid campaign.
#
# Single entry point for the USER:
#   1. creates the campaign's own jobq root under vishesh/ (never the teammate
#      run_logs/ symlink),
#   2. writes gpu_policy.$HOST.json from campaign.ckptgrid_plan.GPU_POLICY,
#   3. starts ONE tmux session `ckptgrid` with workers + driver watch.
#
# Staging (Hub prefetch + cold-1 symlinks) is a separate pre-flight:
#   uv run python -m scripts.stage_ckptgrid_assets
#
# Usage: bash scripts/launch_ckptgrid_campaign.sh
# Optional: CKPTGRID_KEEP_POLICY=1 to keep an existing gpu_policy file.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
HOST=$(hostname)
SESSION=ckptgrid
ROOT=$(uv run python -c "import yaml;from pathlib import Path;from subliminal_learning.config import resolve_queue_root;print(resolve_queue_root(yaml.safe_load(Path('configs/paths/paths_local.yaml').read_text())))")

mkdir -p "$ROOT" "$ROOT/locks" "$ROOT/logs"

POLICY="$ROOT/gpu_policy.$HOST.json"
if [[ "${CKPTGRID_KEEP_POLICY:-0}" == "1" && -f "$POLICY" ]]; then
  echo "CKPTGRID_KEEP_POLICY=1: keeping existing $POLICY: $(cat "$POLICY")"
else
  if [[ -f "$POLICY" ]]; then
    cp -a "$POLICY" "$POLICY.bak_pre_launch_$(date -u +%Y%m%dT%H%M%SZ)"
  fi
  uv run python -c "import json;from subliminal_learning.campaign import ckptgrid_plan as p;print(json.dumps(p.gpu_policy_json(), indent=2))" > "$POLICY"
  echo "wrote $POLICY: $(cat "$POLICY")"
fi
WORKERS=$(uv run python -c "from subliminal_learning.campaign import ckptgrid_plan as p;print(2 * 8 * int(p.GPU_POLICY['cap_per_gpu']))")
QUEUES=$(uv run python -c "from subliminal_learning.campaign import ckptgrid_plan as p;print(','.join(p.QUEUE_PRIORITIES))")

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists -- not re-launching."
  echo "Attach with: tmux attach -t $SESSION"
  exit 0
fi

if pgrep -af "scripts.jobq work --root $ROOT" >/dev/null 2>&1; then
  echo "ERROR: a jobq worker is already running on $ROOT."
  echo "Stop it (or use a dedicated queue_root) before launching ckptgrid."
  pgrep -af "scripts.jobq work --root $ROOT" || true
  exit 1
fi

tmux new-session -d -s "$SESSION" -n workers -c "$REPO"
tmux send-keys -t "$SESSION:workers" \
  "while true; do uv run python -m scripts.jobq work --root $ROOT --workers $WORKERS --queues $QUEUES --lock-prefix $ROOT/locks/jobq; echo 'pool exited; restarting in 60s (Ctrl-C to stop)'; sleep 60; done" C-m

tmux new-window -t "$SESSION" -n driver -c "$REPO"
# No --auto-reconcile-sha: the code-pin contract (docs/PROJECT_DESIGN.md §4)
# requires an explicit `scripts.reconcile_run_set` / --auto-reconcile-sha
# decision when the pin must change, not a silent auto-repin on every poll.
tmux send-keys -t "$SESSION:driver" \
  "uv run python -m scripts.run_ckptgrid_campaign watch --queue-root $ROOT --interval 300 --auto-requeue 2" C-m

cat <<EOF

Started tmux session '$SESSION':
  workers -> jobq pool ($WORKERS threads, queues=$QUEUES), root=$ROOT
  driver  -> campaign watch loop (5-min polls, --auto-requeue 2)

Attach:  tmux attach -t $SESSION      Detach: Ctrl-b d
Status:  uv run python -m scripts.run_ckptgrid_campaign status --queue-root $ROOT
         uv run python -m scripts.jobq status --root $ROOT
Stop:    uv run python -m scripts.jobq stop --root $ROOT
EOF
