#!/usr/bin/env bash
# Launcher: 20260908_ckptgrid2-bear_160m checkpoint-grid stage-2 campaign.
#
# Single entry point for the USER:
#   1. creates the campaign's own jobq root under vishesh/ (never the teammate
#      run_logs/ symlink),
#   2. writes gpu_policy.$HOST.json from campaign.ckptgrid2_plan.GPU_POLICY,
#   3. records pool_scope in campaign_state.json (campaign iff this launch
#      owns the pool — I13),
#   4. starts ONE tmux session `ckptgrid2` with workers + a restarting driver.
#
# Staging (Hub prefetch + cold-1 + assembly) is a separate pre-flight:
#   uv run python -m scripts.stage_ckptgrid2_assets
#
# Usage: bash scripts/launch_ckptgrid2_campaign.sh
# Optional: CKPTGRID2_KEEP_POLICY=1 to keep an existing gpu_policy file.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
HOST=$(hostname)
SESSION=ckptgrid2
ROOT=$(uv run python -c "import yaml;from pathlib import Path;from subliminal_learning.config import resolve_queue_root;print(resolve_queue_root(yaml.safe_load(Path('configs/paths/paths_local.yaml').read_text())))")

mkdir -p "$ROOT" "$ROOT/locks" "$ROOT/logs"

POLICY="$ROOT/gpu_policy.$HOST.json"
if [[ "${CKPTGRID2_KEEP_POLICY:-0}" == "1" && -f "$POLICY" ]]; then
  echo "CKPTGRID2_KEEP_POLICY=1: keeping existing $POLICY: $(cat "$POLICY")"
else
  if [[ -f "$POLICY" ]]; then
    cp -a "$POLICY" "$POLICY.bak_pre_launch_$(date -u +%Y%m%dT%H%M%SZ)"
  fi
  uv run python -c "import json;from subliminal_learning.campaign import ckptgrid2_plan as p;print(json.dumps(p.gpu_policy_json(), indent=2))" > "$POLICY"
  echo "wrote $POLICY: $(cat "$POLICY")"
fi
WORKERS=$(uv run python -c "from subliminal_learning.campaign import ckptgrid2_plan as p;print(2 * 8 * int(p.GPU_POLICY['cap_per_gpu']))")
QUEUES=$(uv run python -c "from subliminal_learning.campaign import ckptgrid2_plan as p;print(','.join(p.QUEUE_PRIORITIES))")

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists -- not re-launching."
  echo "Attach with: tmux attach -t $SESSION"
  exit 0
fi

POOL_SCOPE=campaign
if pgrep -af "scripts.jobq work --root $ROOT" >/dev/null 2>&1; then
  echo "ERROR: a jobq worker is already running on $ROOT."
  echo "Stop it (or use a dedicated queue_root) before launching ckptgrid2."
  pgrep -af "scripts.jobq work --root $ROOT" || true
  exit 1
fi

uv run python -c "
import yaml
from pathlib import Path
from pipelines.ckptgrid2_campaign_pipeline import driver_config
from subliminal_learning.campaign import driver_io
paths = yaml.safe_load(Path('configs/paths/paths_local.yaml').read_text())
cfg = driver_config(
    output_root=Path(paths['output_root']),
    queue_root=Path(r'''$ROOT'''),
    repo_root=Path('.').resolve(),
)
state = driver_io.load_state(cfg)
state['pool_scope'] = '''$POOL_SCOPE'''
driver_io.save_state(cfg, state)
print('pool_scope', state['pool_scope'])
"

tmux new-session -d -s "$SESSION" -n workers -c "$REPO"
tmux send-keys -t "$SESSION:workers" \
  "while true; do uv run python -m scripts.jobq work --root $ROOT --workers $WORKERS --queues $QUEUES --lock-prefix $ROOT/locks/jobq; echo 'pool exited; restarting in 60s (Ctrl-C to stop)'; sleep 60; done" C-m

tmux new-window -t "$SESSION" -n driver -c "$REPO"
# Driver watch is a single-point-of-failure in the sibling launcher; wrap it
# in its own restart loop. Exit 0 (done) and exit 2 (gate) do not restart.
# No --auto-reconcile-sha: the code-pin contract requires an explicit
# reconcile when the pin must change.
tmux send-keys -t "$SESSION:driver" \
  "backoff=60; while true; do uv run python -m scripts.run_ckptgrid2_campaign watch --queue-root $ROOT --interval 300 --auto-requeue 2 --deadline-hours 48; ec=\$?; if [ \$ec -eq 0 ]; then echo 'driver finished'; break; fi; if [ \$ec -eq 2 ]; then echo 'GATE FAILED — not restarting'; break; fi; echo \"driver exited \$ec; restarting in \${backoff}s\"; sleep \$backoff; backoff=\$((backoff < 600 ? backoff * 2 : 600)); done" C-m

cat <<EOF

Started tmux session '$SESSION':
  workers -> jobq pool ($WORKERS threads, queues=$QUEUES), root=$ROOT
  driver  -> campaign watch loop (5-min polls, --auto-requeue 2,
             --deadline-hours 48, restart-on-crash)
  pool_scope=$POOL_SCOPE

Attach:  tmux attach -t $SESSION      Detach: Ctrl-b d
Status:  uv run python -m scripts.run_ckptgrid2_campaign status --queue-root $ROOT
         uv run python -m scripts.jobq status --root $ROOT
Stop:    uv run python -m scripts.jobq stop --root $ROOT
EOF
