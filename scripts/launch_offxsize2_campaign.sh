#!/usr/bin/env bash
# Launcher: 20260913_offxsize2-doglion_mixed off-policy cross-size campaign v2.
#
# Single entry point for the USER:
#   1. refuses to run anywhere but the campaign node,
#   2. writes ONLY gpu_policy.<campaign node>.json -- soar's policy file for the live
#      cross-family campaign shares this queue root and is never touched,
#   3. starts ONE tmux session `offxsize2` with a worker pool + a restarting driver.
#
# Staging (adoption + gate reports) is a separate pre-flight that must run first:
#   uv run python -m scripts.stage_offxsize2_assets
#
# Usage: bash scripts/launch_offxsize2_campaign.sh
# Optional: OFFXSIZE2_KEEP_POLICY=1 to keep an existing gpu_policy file.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
SESSION=offxsize2
HOST=$(hostname)
PLAN="from subliminal_learning.campaign import offxsize2_plan as p"

CAMPAIGN_NODE=$(uv run python -c "$PLAN;print(p.CAMPAIGN_NODE)")
FORBIDDEN_NODE=$(uv run python -c "$PLAN;print(p.FORBIDDEN_NODE)")

# jobq pins a queue to the submitting host and there is exactly one
# gpu_policy.<host>.json per host per queue root. Launching from the wrong node would
# file this campaign's ~600 cells onto the node already running the cross-family
# campaign, and rewrite that node's live policy on the way.
if [[ "$HOST" == "$FORBIDDEN_NODE" ]]; then
  echo "ERROR: $HOST runs the live cross-family campaign; $SESSION belongs on $CAMPAIGN_NODE." >&2
  exit 1
fi
if [[ "$HOST" != "$CAMPAIGN_NODE" ]]; then
  echo "ERROR: $SESSION is pinned to $CAMPAIGN_NODE, but this is $HOST." >&2
  exit 1
fi

ROOT=$(uv run python -c "import yaml;from pathlib import Path;from subliminal_learning.config import resolve_queue_root;print(resolve_queue_root(yaml.safe_load(Path('configs/paths/paths_local.yaml').read_text())))")
mkdir -p "$ROOT" "$ROOT/logs"

# Named for the campaign node, never for \$HOST: the two are equal by the guard above,
# and spelling it out keeps a copied line from ever writing soar's file.
POLICY="$ROOT/gpu_policy.$CAMPAIGN_NODE.json"
if [[ "${OFFXSIZE2_KEEP_POLICY:-0}" == "1" && -f "$POLICY" ]]; then
  echo "OFFXSIZE2_KEEP_POLICY=1: keeping existing $POLICY: $(cat "$POLICY")"
else
  if [[ -f "$POLICY" ]]; then
    cp -a "$POLICY" "$POLICY.bak_pre_launch_$(date -u +%Y%m%dT%H%M%SZ)"
  fi
  uv run python -c "import json;$PLAN;print(json.dumps(p.gpu_policy_json(), indent=2))" > "$POLICY"
  echo "wrote $POLICY: $(cat "$POLICY")"
fi

# Every queue by name. A glob would hand this pool the cross-family campaign's queues
# off the shared root the first time one of them is created.
QUEUES=$(uv run python -c "$PLAN;print(','.join(sorted(p.QUEUE_PRIORITIES)))")
# Admission is slot-weighted, so the most concurrent cells a card can hold is
# cap_per_gpu / the lightest footprint; more worker threads than that only ever block.
WORKERS=$(uv run python -c "$PLAN;
tables=(p.STUDENT_FOOTPRINT,p.TEACHER_FOOTPRINT,p.NUMGEN_FOOTPRINT,p.CAPABILITY_FOOTPRINT)
light=min(f.slots for t in tables for f in t.values())
print(len(p.GPU_POLICY['gpus']) * (int(p.GPU_POLICY['cap_per_gpu']) // light))")
# Node-local GPU slot locks. The queue root is shared with soar, so locking under it
# would make this pool contend with the cross-family campaign's cards.
LOCKS=$(uv run python -c "$PLAN;print(p.JOBQ_LOCK_PREFIX)")

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists -- not re-launching."
  echo "Attach with: tmux attach -t $SESSION"
  exit 0
fi

if pgrep -af "scripts.jobq work --root $ROOT" >/dev/null 2>&1; then
  echo "ERROR: a jobq worker is already running on $ROOT." >&2
  pgrep -af "scripts.jobq work --root $ROOT" || true
  exit 1
fi

tmux new-session -d -s "$SESSION" -n workers -c "$REPO"
tmux send-keys -t "$SESSION:workers" \
  "while true; do uv run python -m scripts.jobq work --root $ROOT --workers $WORKERS --queues $QUEUES --lock-prefix $LOCKS; echo 'pool exited; restarting in 60s (Ctrl-C to stop)'; sleep 60; done" C-m

tmux new-window -t "$SESSION" -n driver -c "$REPO"
# Exit 0 (done) and exit 2 (a gate, including the concordance stop gate) do not restart.
# No --auto-reconcile-sha: an intentional pin change is an explicit operator action
# (docs/PROJECT_DESIGN.md section 4), never a silent auto-repin on a poll.
tmux send-keys -t "$SESSION:driver" \
  "backoff=60; while true; do uv run python -m scripts.run_offxsize2_campaign watch --queue-root $ROOT --interval 300 --auto-requeue 2 --deadline-hours 48; ec=\$?; if [ \$ec -eq 0 ]; then echo 'driver finished'; break; fi; if [ \$ec -eq 2 ]; then echo 'GATE FAILED -- not restarting'; break; fi; echo \"driver exited \$ec; restarting in \${backoff}s\"; sleep \$backoff; backoff=\$((backoff < 600 ? backoff * 2 : 600)); done" C-m

cat <<EOF

Started tmux session '$SESSION' on $HOST:
  workers -> jobq pool ($WORKERS threads, lock-prefix=$LOCKS), root=$ROOT
  driver  -> campaign watch loop (5-min polls, --auto-requeue 2,
             --deadline-hours 48, restart-on-crash)
  policy  -> $POLICY

Attach:  tmux attach -t $SESSION      Detach: Ctrl-b d
Status:  uv run python -m scripts.run_offxsize2_campaign status --queue-root $ROOT
         uv run python -m scripts.jobq status --root $ROOT --lock-prefix $LOCKS
Stop:    uv run python -m scripts.jobq stop --root $ROOT
EOF
