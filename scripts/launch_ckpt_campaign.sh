#!/usr/bin/env bash
# Launcher: 20260805_ckptladder_160m pretraining-checkpoint ladder campaign.
#
# Single entry point for the USER:
#   1. creates the campaign's own jobq root under vishesh/ (never the teammate
#      run_logs/ symlink),
#   2. writes gpu_policy.$HOST.json from campaign.ckpt_plan.GPU_POLICY (SSOT),
#   3. starts ONE tmux session `ckptladder` with workers + driver watch.
#
# Disk free-space is gated by the driver against the filesystem behind the
# configured output_root / hf_cache_root (SSD-3 hot root — see
# docs/SSD3_HOT_MIRROR.md), not a hardcoded SSD-2 mount.
#
# Usage: bash scripts/launch_ckpt_campaign.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
HOST=$(hostname)
SESSION=ckptladder
ROOT=$(uv run python -c "import yaml;from pathlib import Path;from subliminal_learning.config import resolve_queue_root;print(resolve_queue_root(yaml.safe_load(Path('configs/paths/paths_local.yaml').read_text())))")

mkdir -p "$ROOT" "$ROOT/locks" "$ROOT/logs"

POLICY="$ROOT/gpu_policy.$HOST.json"
if [ -f "$POLICY" ]; then
  echo "keeping existing $POLICY: $(cat "$POLICY")"
else
  uv run python -c "import json;from subliminal_learning.campaign import ckpt_plan as p;print(json.dumps(p.gpu_policy_json()))" > "$POLICY"
  echo "wrote $POLICY: $(cat "$POLICY")"
fi
WORKERS=$(uv run python -c "from subliminal_learning.campaign import ckpt_plan as p;print(p.WORKER_THREADS)")

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists -- not re-launching."
  echo "Attach with: tmux attach -t $SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" -n workers -c "$REPO"
tmux send-keys -t "$SESSION:workers" \
  "while true; do uv run python -m scripts.jobq work --root $ROOT --workers $WORKERS --lock-prefix $ROOT/locks/jobq; echo 'pool exited; restarting in 60s (Ctrl-C to stop)'; sleep 60; done" C-m

tmux new-window -t "$SESSION" -n driver -c "$REPO"
# No --auto-reconcile-sha: the code-pin contract (docs/PROJECT_DESIGN.md §4)
# requires an explicit `scripts.reconcile_run_set` / --auto-reconcile-sha
# decision when git HEAD drifts mid-campaign, not a silent auto-repin on every
# 5-minute poll.
tmux send-keys -t "$SESSION:driver" \
  "uv run python -m scripts.run_ckpt_campaign watch --queue-root $ROOT --interval 300" C-m

cat <<EOF

Started tmux session '$SESSION':
  workers -> jobq pool ($WORKERS threads, restart loop), root=$ROOT
  driver  -> campaign watch loop (5-min polls)

Attach:  tmux attach -t $SESSION      Detach: Ctrl-b d
Status:  uv run python -m scripts.run_ckpt_campaign status --queue-root $ROOT
         uv run python -m scripts.jobq status --root $ROOT
Stop:    uv run python -m scripts.jobq stop --root $ROOT
EOF
