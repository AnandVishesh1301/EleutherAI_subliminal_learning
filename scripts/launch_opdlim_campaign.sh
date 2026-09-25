#!/usr/bin/env bash
# Launcher: OPD-limits campaign, one tier per home node.
#   tier 1 -> 20260914_opdlim1-bear_mixed on soar-subliminal-2 (a40-cluster-3)
#   tier 2 -> 20260914_opdlim2-bear_mixed on soar-subliminal-0 (a40-cluster-1)
#   tier 3 -> 20260922_opdlim3-bear_mixed on goncalo-temp-1 (a40-cluster-2), λ=0 ladder
#   tier 4 -> 20260925_opdlim4-bear_mixed on soar-subliminal-0 (a40-cluster-1), P3 vmask
#
# Run from the campaign code pin (<main checkout>/.code_pins/<sha>) on the tier's own
# node, after staging:
#   uv run python -m scripts.stage_opdlim_assets --tier N
#
#   1. refuses any host other than the tier's home node,
#   2. refuses a lock prefix other than the tier's: node-local /tmp on cluster-3 and
#      goncalo, the legacy shared $ROOT/locks/jobq on soar (only if SHARED_LOCK_OWNER is soar),
#   3. writes gpu_policy.$HOST.json from campaign.opdlim_plan.GPU_POLICY (backup first),
#   4. starts ONE tmux session `opdlimN` with a worker pool + a restarting driver.
#
# Usage: bash scripts/launch_opdlim_campaign.sh <1|2|3|4>
# Optional: OPDLIM_KEEP_POLICY=1 to keep an existing gpu_policy file.
set -euo pipefail

TIER="${1:-}"
if [[ "$TIER" != "1" && "$TIER" != "2" && "$TIER" != "3" && "$TIER" != "4" ]]; then
  echo "usage: bash scripts/launch_opdlim_campaign.sh <1|2|3|4>"
  exit 2
fi

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
HOST=$(hostname)
PIN_SHA=$(git -C "$REPO" rev-parse HEAD)
if [[ "$(basename "$(dirname "$REPO")")" != ".code_pins" || "$(basename "$REPO")" != "$PIN_SHA" ]]; then
  echo "ERROR: launch from <main checkout>/.code_pins/<sha> (this is $REPO at $PIN_SHA)."
  exit 1
fi
SESSION="opdlim$TIER"
read -r NODE LOCK < <(uv run python -c "from subliminal_learning.campaign import opdlim_plan as p;t=p.tier($TIER);print(t.node, t.lock_prefix or '')")

if [[ "$HOST" != "$NODE" ]]; then
  echo "ERROR: tier $TIER must launch on $NODE; this host is $HOST."
  exit 1
fi

ROOT=$(uv run python -c "import yaml;from pathlib import Path;from subliminal_learning.config import resolve_queue_root;print(resolve_queue_root(yaml.safe_load(Path('configs/paths/paths_local.yaml').read_text())))")
mkdir -p "$ROOT" "$ROOT/logs"

LOCK_PREFIX="${SL_JOBQ_LOCK_PREFIX:-$ROOT/locks/jobq}"
if [[ "$LOCK_PREFIX" != "${LOCK:-$ROOT/locks/jobq}" ]]; then
  echo "ERROR: lock prefix '$LOCK_PREFIX' but tier $TIER requires '${LOCK:-$ROOT/locks/jobq}'."
  exit 1
fi
if [[ -z "$LOCK" && "$(cat "$ROOT/locks/SHARED_LOCK_OWNER" 2>/dev/null)" != "$HOST" ]]; then
  echo "ERROR: $ROOT/locks/SHARED_LOCK_OWNER does not name $HOST; refusing the shared lock prefix."
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists -- not re-launching."
  echo "Attach with: tmux attach -t $SESSION"
  exit 0
fi
# Trailing space: `$ROOT` must not match a sibling root such as `${ROOT}_ws160m3`.
if pgrep -af "scripts.jobq work --root $ROOT " >/dev/null 2>&1; then
  echo "ERROR: a jobq worker is already running on $ROOT on this host."
  pgrep -af "scripts.jobq work --root $ROOT " || true
  exit 1
fi

POLICY="$ROOT/gpu_policy.$HOST.json"
if [[ "${OPDLIM_KEEP_POLICY:-0}" == "1" && -f "$POLICY" ]]; then
  echo "OPDLIM_KEEP_POLICY=1: keeping existing $POLICY: $(cat "$POLICY")"
else
  if [[ -f "$POLICY" ]]; then
    cp -a "$POLICY" "$POLICY.bak_pre_launch_$(date -u +%Y%m%dT%H%M%SZ)"
  fi
  uv run python -c "import json;from subliminal_learning.campaign import opdlim_plan as p;print(json.dumps(p.gpu_policy_json(), indent=2))" > "$POLICY"
  echo "wrote $POLICY: $(cat "$POLICY")"
fi
WORKERS=$(uv run python -c "from subliminal_learning.campaign import opdlim_plan as p;print(p.total_slots())")
# The queue list is built inside the pane: typed inline, tier 2's list overruns the
# terminal's ~4 KB input line and the loop never starts.
QUEUES_CMD="from subliminal_learning.campaign import opdlim_plan as p;print(','.join(sorted(p.queue_priorities(p.tier($TIER)))))"

tmux new-session -d -s "$SESSION" -n workers -c "$REPO"
tmux send-keys -t "$SESSION:workers" \
  "Q=\$(uv run python -c \"$QUEUES_CMD\"); while true; do uv run python -m scripts.jobq work --root $ROOT --workers $WORKERS --queues \$Q --lock-prefix $LOCK_PREFIX; echo 'pool exited; restarting in 60s (Ctrl-C to stop)'; sleep 60; done" C-m

tmux new-window -t "$SESSION" -n driver -c "$REPO"
# No --auto-reconcile-sha: a pin change is an explicit operator action
# (docs/PROJECT_DESIGN.md section 4). Exit 0 (done) and exit 2 (a gate) do not restart.
tmux send-keys -t "$SESSION:driver" \
  "backoff=60; while true; do uv run python -m scripts.run_opdlim_campaign watch --tier $TIER --queue-root $ROOT --interval 300 --auto-requeue 2 --code-sha $PIN_SHA; ec=\$?; if [ \$ec -eq 0 ]; then echo 'driver finished'; break; fi; if [ \$ec -eq 2 ]; then echo 'GATE FAILED -- not restarting'; break; fi; echo \"driver exited \$ec; restarting in \${backoff}s\"; sleep \$backoff; backoff=\$((backoff < 600 ? backoff * 2 : 600)); done" C-m

cat <<MSG

Started tmux session '$SESSION' on $HOST:
  workers -> jobq pool ($WORKERS threads, lock prefix $LOCK_PREFIX), root=$ROOT
  driver  -> restarting campaign watch loop (5-min polls, --auto-requeue 2; no deadline)

Attach:  tmux attach -t $SESSION      Detach: Ctrl-b d
Status:  uv run python -m scripts.run_opdlim_campaign status --tier $TIER --queue-root $ROOT
Stop:    uv run python -m scripts.jobq stop --root $ROOT
MSG
